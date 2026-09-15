#!/usr/bin/env python
"""Layer-wise linear probe (评测C): do the models' internal layers encode
vessel class at all, even though prompting fails?

Protocol (Alain & Bengio 2016 / SUPERB-style — NO large-scale training):
  1. frozen model, ONE forward per clip with output_hidden_states=True
     → every decoder layer's hidden state, mean-pooled over attended tokens;
  2. per layer: StandardScaler + multinomial logistic regression (C by 3-fold
     CV) trained on a train-subset → test metrics. Minutes per sweep on CPU.
The expensive part is only feature EXTRACTION (inference), sharded + resumable.

The prompt is the SAME zero-shot template the prompting evals use, so the
probed activations are exactly those the model had when answering (wrongly) —
"the same activations, a better readout". Sweep matrix (11, grouped by model
to pay each model load once; Omni first — the same-backbone 看 vs 听 layer
curves are the paper's key figure):

  qwen3_omni_30b : audio mel stft demon     (device_map=auto)
  qwen2_audio    : audio                     (cuda:0)
  qwen3_vl_8b    : mel stft demon            (cuda:0)
  qwen3_vl_32b   : mel stft demon            (device_map=auto)

Extraction order per sweep: ALL test clips first (curves become meaningful
early), then train clips in a seeded shuffle, capped at --max-train (default
3000 — a 4-class linear head saturates far below that; --full-train extends
the shard sequence). Features float16 ≈ 2-8 GB per sweep.

Outputs: outputs/features/layerprobe/{model}_{spec}_{manifest_hash}/shard_*.npz
         outputs/results/layer_probe/{model}_{spec}.json (per-layer curve)
         outputs/results/layer_probe/curves.csv (one row per layer per sweep)
Done marker (full default matrix only): outputs/results/layer_probe.done
"""
from __future__ import annotations

import argparse, gc, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from splash.audio.io import read_segment
from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.models.factory import build_model
from splash.prompting.templates import build_prompt
from splash.tracking.results import _write_text

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")
FEATURES = Path("outputs/features/layerprobe")
SEED = 42
SHARD = 250

MODEL_SPECS = {
    "qwen3_omni_30b": {"path": "/models/Qwen3-Omni-30B-A3B-Instruct",
                       "kwargs": {"device_map": "auto"},
                       "specs": ["audio", "mel", "stft", "demon"]},
    "qwen2_audio": {"path": "/models/Qwen2-Audio-7B-Instruct",
                    "kwargs": {"device": "cuda:0"},
                    "specs": ["audio"]},
    "qwen3_vl_8b": {"path": "/models/Qwen3-VL-8B-Instruct",
                    "kwargs": {"device": "cuda:0", "device_map": None},
                    "specs": ["mel", "stft", "demon"]},
    "qwen3_vl_32b": {"path": "/models/Qwen3-VL-32B-Instruct",
                     "kwargs": {"device_map": "auto"},
                     "specs": ["mel", "stft", "demon"]},
}
ORDER = ["qwen3_omni_30b", "qwen2_audio", "qwen3_vl_8b", "qwen3_vl_32b"]


def ordered_rows(manifest: pd.DataFrame, max_train: int, full_train: bool):
    """Row indices: all test (manifest order) then train (seeded shuffle)."""
    test_idx = manifest.index[manifest["split"] == "test"].to_numpy()
    train_df = manifest[manifest["split"] == "train"]
    perm = np.random.default_rng(SEED).permutation(len(train_df))
    train_idx = train_df.index.to_numpy()[perm]
    if not full_train:
        train_idx = train_idx[:max_train]
    return list(test_idx) + list(train_idx)


def load_media(row, spec, clip_samples):
    if spec == "audio":
        return read_segment(row["path"], row["start"], row["end"],
                            target_sr=16000, pad_to_samples=clip_samples)
    return Image.open(row[f"{spec}_path"]).convert("RGB")


def extract_sweep(wrapper, manifest, spec, out_dir: Path, order, clip_samples):
    """Extract per-layer pooled features for the ordered rows → sharded npz."""
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    names = class_names(DATASET, "en")
    media_type = "audio" if spec == "audio" else "image"
    n_shards = (len(order) + SHARD - 1) // SHARD
    for s in range(n_shards):
        rows = manifest.iloc[order[s * SHARD:(s + 1) * SHARD]]
        path = out_dir / f"shard_{s:04d}.npz"
        if path.exists():
            try:
                if len(np.load(path, allow_pickle=False)["y"]) == len(rows):
                    continue
            except Exception:
                print(f"  shard {s}: corrupt — recomputing")
        feats = []
        for r in rows.itertuples(index=False):
            rd = r._asdict()
            media = load_media(rd, spec, clip_samples)
            messages = build_prompt(media=media, label_names=names, regime="zero_shot",
                                    support=None, media_type=media_type, lang="en",
                                    enrich_text=None)
            pooled = wrapper.hidden_states([messages])[0]      # [L+1, D] fp32
            feats.append(pooled.to(torch.float16).cpu().numpy())
        X = np.stack(feats)                                     # [n, L+1, D]
        payload = dict(X=X, y=rows["label_id"].to_numpy(np.int16),
                       split=(rows["split"] == "test").to_numpy(np.int8),
                       rid=rows["recording_id"].to_numpy().astype(str),
                       cidx=rows["clip_idx"].to_numpy(np.int32))
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        os.replace(tmp, path)
        print(f"  shard {s + 1}/{n_shards}: {len(rows)} clips → {X.shape}")
    return out_dir


def probe_sweep(out_dir: Path, model_name: str, spec: str):
    """Per-layer logistic-regression probe → curve json + curves.csv rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, accuracy_score

    shards = sorted(out_dir.glob("shard_*.npz"))
    if not shards:
        raise RuntimeError(f"no shards in {out_dir}")
    data = [np.load(p, allow_pickle=False) for p in shards]
    X = np.concatenate([d["X"] for d in data])                 # [N, L+1, D] fp16
    y = np.concatenate([d["y"] for d in data]).astype(int)
    is_test = np.concatenate([d["split"] for d in data]).astype(bool)
    n_layers = X.shape[1]
    tr, te = ~is_test, is_test
    print(f"  probe: N={len(y)} (train {tr.sum()} / test {te.sum()}), "
          f"layers={n_layers}, dim={X.shape[2]}")

    curves = []
    for layer in range(n_layers):
        Xl = X[:, layer, :].astype(np.float32)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
                            cv=3, n_jobs=4)
        grid.fit(Xl[tr], y[tr])
        pred = grid.predict(Xl[te])
        curves.append({
            "layer": layer,
            "acc": float(accuracy_score(y[te], pred)),
            "f1_macro": float(f1_score(y[te], pred, average="macro")),
            "C": float(grid.best_params_["logisticregression__C"]),
            "cv_acc": float(grid.best_score_),
        })
        if layer % 8 == 0 or layer == n_layers - 1:
            c = curves[-1]
            print(f"    layer {layer:2d}: acc={c['acc']:.3f} f1={c['f1_macro']:.3f} "
                  f"(C={c['C']}, cv={c['cv_acc']:.3f})")

    best = max(curves, key=lambda c: c["acc"])
    payload = {"model": model_name, "spec": spec, "dataset": DATASET,
               "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "n_layers_total": n_layers, "best": best, "curves": curves,
               "random_acc": 0.25}
    out_json = RESULTS / "layer_probe" / f"{model_name}_{spec}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_text(out_json, json.dumps(payload, indent=2))
    rows = pd.DataFrame([{**{"model": model_name, "spec": spec}, **c} for c in curves])
    csv = RESULTS / "layer_probe" / "curves.csv"
    if csv.exists():
        prev = pd.read_csv(csv)
        prev = prev[~((prev["model"] == model_name) & (prev["spec"] == spec))]
        rows = pd.concat([prev, rows], ignore_index=True)
    rows.to_csv(csv, index=False)
    print(f"  ✅ best layer {best['layer']}: acc={best['acc']:.3f} "
          f"f1={best['f1_macro']:.3f} → {out_json.name}")
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(ORDER))
    ap.add_argument("--specs", default=None, help="filter specs (e.g. 'audio,mel')")
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--full-train", action="store_true")
    ap.add_argument("--probe-only", action="store_true", help="skip extraction, probe existing shards")
    ap.add_argument("--device", default=None,
                    help="override device for single-GPU models (e.g. cuda:1) — lets this run "
                         "on the idle GPU while another stage occupies cuda:0")
    args = ap.parse_args()
    models = [m for m in ORDER if m in args.models.split(",")]
    if args.device:
        for m in models:
            if "device" in MODEL_SPECS[m]["kwargs"]:        # single-GPU models only
                MODEL_SPECS[m]["kwargs"]["device"] = args.device

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    full = pd.read_csv(aug if aug.exists() else manifest)
    clip_samples = int(CLIP_LEN * 16000)
    order = ordered_rows(full, args.max_train, args.full_train)
    print(f"[layerprobe] models={models} rows={len(order)} "
          f"(test {int((full.split=='test').sum())} + train {len(order)-int((full.split=='test').sum())})")

    import torch
    for model_name in models:
        cfg = MODEL_SPECS[model_name]
        specs = [s for s in cfg["specs"] if (args.specs is None or s in args.specs.split(","))]
        wrapper = build_model({"name": model_name, "path": cfg["path"], **cfg["kwargs"]})
        print(f"\n=== {model_name} (specs={specs}) ===")
        for spec in specs:
            out_dir = FEATURES / f"{model_name}_{spec}_{manifest.stem.split('_')[-1]}"
            if not args.probe_only:
                extract_sweep(wrapper, full, spec, out_dir, order, clip_samples)
            probe_sweep(out_dir, model_name, spec)
        del wrapper; gc.collect(); torch.cuda.empty_cache()

    full_matrix = (args.models == ",".join(ORDER) and args.specs is None
                   and not args.probe_only and not args.full_train)
    if args.full_train and args.models == ",".join(ORDER) and args.specs is None:
        (RESULTS / "layer_probe_full.done").write_text("done\n")
    elif full_matrix:
        (RESULTS / "layer_probe.done").write_text("done\n")
    print("\n✅ layer probe complete")


if __name__ == "__main__":
    main()
