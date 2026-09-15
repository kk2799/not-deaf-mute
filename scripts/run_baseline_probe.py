#!/usr/bin/env python
"""Frozen SSL encoder + linear-probe baseline (评测E anchor; SUPERB-style).

WavLM-Large / BEATs_iter3+ cannot be prompted — the supervised anchor is:
frozen encoder → mean-pooled embedding per clip → logistic-regression head
trained on the train split (C by 5-fold CV) → test-split metrics, reported in
the same results.csv/format as the prompting experiments (clip + recording
level, per-clip predictions saved for figures).

Feature extraction is sharded + resumable (outputs/features/…shard_*.npz).
The probe itself takes minutes on CPU. Writes one results.csv row per model
and finally outputs/results/baseline_probe.done.

Usage:
  python scripts/run_baseline_probe.py                     # both encoders
  python scripts/run_baseline_probe.py --models wavlm_large
"""
from __future__ import annotations

import argparse, os
from pathlib import Path

import numpy as np
import pandas as pd

from splash.audio.io import read_segment
from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.metrics.aggregation import aggregate_predictions
from splash.metrics.classification import classification_metrics, format_metrics
from splash.models.factory import build_model
from splash.prompting.templates import option_layout
from splash.tracking.results import save_result

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")
FEATURES = Path("outputs/features")

MODEL_PATHS = {
    "wavlm_large": "/models/WavLM-Large",
    "beats_iter3_plus": "/models/BEATs/BEATs_iter3_plus_AS2M.pt",
}


def extract_features(model, manifest: pd.DataFrame, out_dir: Path,
                     shard_size: int, target_sr: int, clip_samples: int) -> pd.DataFrame:
    """Extract embeddings for ALL manifest rows, sharded + resumable.

    Shards are written atomically (tmp + replace) so a crash can't leave a
    half-written file; corrupt/short shards are recomputed on the next run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(manifest)
    n_shards = (n + shard_size - 1) // shard_size
    Xs = [np.empty((0, model.embed_dim), dtype=np.float32) for _ in range(n_shards)]
    for s in range(n_shards):
        shard = manifest.iloc[s * shard_size:(s + 1) * shard_size]
        path = out_dir / f"shard_{s:04d}.npz"
        if path.exists():
            try:
                z = np.load(path)
                if len(z["X"]) == len(shard):
                    Xs[s] = z["X"]
                    continue
            except Exception:
                print(f"  shard {s}: corrupt — recomputing")
        wavs = [read_segment(r.path, r.start, r.end,
                             target_sr=target_sr, pad_to_samples=clip_samples)
                for r in shard.itertuples(index=False)]
        X = model.encode(wavs)                       # [n, D]
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, X=X)
        os.replace(tmp, path)
        Xs[s] = X
        print(f"  shard {s + 1}/{n_shards}: {len(shard)} clips → {X.shape}")
    return manifest.assign(X=list(np.vstack(Xs)))


def probe(df: pd.DataFrame, dataset_name: str):
    """Train logistic regression (C by CV) on train-split features → metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    names = class_names(dataset_name, "en")
    label_ids = sorted(names.keys())

    def XY(split):
        sub = df[df["split"] == split]
        return np.stack(sub["X"].tolist()), sub["label_id"].to_numpy(int)

    Xtr, ytr = XY("train")
    Xte, yte = XY("test")
    print(f"  probe: train {Xtr.shape} test {Xte.shape}")

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
                        cv=5, n_jobs=8)
    grid.fit(Xtr, ytr)
    print(f"  best C={grid.best_params_['logisticregression__C']} cv_acc={grid.best_score_:.3f}")

    proba = grid.predict_proba(Xte)                  # [n_test, n_classes]
    cols = grid.best_estimator_.classes_             # label ids, sorted
    pred = cols[proba.argmax(axis=1)]

    test = df[df["split"] == "test"].reset_index(drop=True)
    letters, label_order, _ = option_layout(names)
    letter_of = {lid: letters[label_order.index(lid)] for lid in label_ids}
    lid_of_letter = {v: k for k, v in letter_of.items()}
    records = []
    for i, r in test.iterrows():
        p = {letter_of[int(cols[j])]: float(np.log(max(proba[i, j], 1e-12)))
             for j in range(len(cols))}
        records.append({
            "recording_id": r["recording_id"], "clip_idx": int(r["clip_idx"]),
            "label_id": int(r["label_id"]),
            "pred_label_id": int(pred[i]), "pred_letter": letter_of[int(pred[i])],
            "parse_letter": None, "gen_text": "[linear probe]",
            "choice_logprobs": p, "letter_to_label": lid_of_letter,
        })
    metrics_clip = classification_metrics(yte.tolist(), pred.tolist(), label_ids, names)
    y_true_r, y_pred_r = aggregate_predictions(records, method="avg_prob")
    metrics_rec = classification_metrics(y_true_r, y_pred_r, label_ids, names)
    print(f"  clip: {format_metrics(metrics_clip)}")
    print(f"  rec : {format_metrics(metrics_rec)}")
    return metrics_clip, metrics_rec, records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="wavlm_large,beats_iter3_plus")
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None, help="cap manifest rows (smoke test)")
    args = ap.parse_args()

    manifest = find_manifest({"name": args.dataset, "clip_len": CLIP_LEN, "overlap": 0.5})
    full = pd.read_csv(manifest)
    if args.limit:
        full = full.head(args.limit)
    target_sr, clip_samples = 16000, int(CLIP_LEN * 16000)
    print(f"[probe] manifest={manifest.name} rows={len(full)} (test={int((full.split=='test').sum())})")

    for model_name in args.models.split(","):
        model = build_model({"name": model_name, "path": MODEL_PATHS[model_name], "device": args.device})
        print(f"\n=== {model_name} (frozen + linear probe) ===")
        # dir name embeds the manifest content hash — a re-segmented dataset
        # must not silently reuse shards computed for different clips
        feats = extract_features(model, full, FEATURES / f"{model_name}_{manifest.stem}",
                                 args.shard, target_sr, clip_samples)
        del model
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

        metrics_clip, metrics_rec, records = probe(feats, args.dataset)
        cfg = {"eval": {"name": "linear_probe"}, "model": {"name": model_name},
               "data": {"name": args.dataset, "clip_len": CLIP_LEN},
               "regime": {"name": "supervised", "shot": 0}, "lang": "en", "enrich": "none"}
        save_result(cfg, metrics_clip, metrics_rec, records)

    # stage marker only for the FULL default run — subset/smoke runs must not
    # make the watcher skip the remaining encoders (review finding)
    if args.models == "wavlm_large,beats_iter3_plus" and not args.limit:
        (RESULTS / "baseline_probe.done").write_text("done\n")
    print("\n✅ baseline probe complete")


if __name__ == "__main__":
    main()
