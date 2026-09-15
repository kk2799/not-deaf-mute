#!/usr/bin/env python
"""Activation steering (评测C causal upgrade): can the probe direction CAUSALLY
control the answer?  (protocol after Marks & Tegmark 2023; Belrose tuned-lens
lineage)

For Qwen3-VL-8B on mel spectrograms:
  1. train the layer-L* probe on cached layerprobe features (best layer by the
     sweep's curve json);
  2. register a forward hook on decoder layer L* that adds  α·û_c·||h̄||  to the
     hidden state (û_c = de-standardised probe direction for class c);
  3. for 200 test clips × α ∈ {0,1,2,4,8} × {true-class direction, random
     control direction}: score the 4-way answer.

If steering toward the TRUE class flips wrong answers to correct at rate ≫ the
random control, the representation is causally usable — the interface, not the
representation, is the bottleneck.  Outputs outputs/results/steering.json+.done.
Run AFTER the layer-probe stage (needs its feature shards + curve json).
"""
from __future__ import annotations

import gc, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.models.factory import build_model
from splash.prompting.templates import option_layout, build_prompt
from splash.models.third_party.beats import BEATs  # noqa: F401 (import guard not needed)

FEATURES = Path("outputs/features/layerprobe")
CURVES = Path("outputs/results/layer_probe")
RESULTS = Path("outputs/results")
DATASET, SPEC = "deepship", "mel"
MODEL, PATH = "qwen3_vl_8b", "/models/Qwen3-VL-8B-Instruct"
N_CLIPS, ALPHAS = 200, [0.0, 0.02, 0.05, 0.1, 0.2]   # α·||h̄||: 2–20% perturbation
SEED = 99


def load_features():
    d = sorted(FEATURES.glob(f"*_{MODEL}_{SPEC}")) or sorted(
        FEATURES.glob(f"{MODEL}_{SPEC}_*"))
    if not d:
        raise RuntimeError("no layerprobe shards — run the layer-probe stage first")
    shards = sorted(d[0].glob("shard_*.npz"))
    X = np.concatenate([np.load(p, allow_pickle=False)["X"] for p in shards])
    y = np.concatenate([np.load(p, allow_pickle=False)["y"] for p in shards]).astype(int)
    te = np.concatenate([np.load(p, allow_pickle=False)["split"] for p in shards]).astype(bool)
    return X, y, te


def probe_directions(X, y, te):
    """Best-layer multinomial LR → {class: unit direction in RAW activation space}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    cj = CURVES / f"{MODEL}_{SPEC}.json"
    L = json.loads(cj.read_text())["best"]["layer"] if cj.exists() else X.shape[1] // 2
    Xl = X[:, L, :].astype(np.float32)
    sc = StandardScaler().fit(Xl[~te])
    Z = sc.transform(Xl)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(Z[~te], y[~te])
    acc = accuracy_score(y[te], lr.predict(Z[te]))
    dirs = {}
    for c in range(lr.coef_.shape[0]):
        w = lr.coef_[c] / sc.scale_            # de-standardise
        dirs[c] = (w / np.linalg.norm(w)).astype(np.float32)
    print(f"[steering] layer {L}: probe acc {acc:.3f}")
    return L, dirs, acc


def main():
    names = class_names(DATASET, "en")
    letters, _ids, _ = option_layout(names)
    X, y, te = load_features()
    L, dirs, probe_acc = probe_directions(X, y, te)

    manifest = find_manifest({"name": DATASET, "clip_len": 30.0, "overlap": 0.5})
    aug = pd.read_csv(manifest.with_name(f"{manifest.stem}_spec.csv"))
    # STRATIFIED subset — the manifest is sorted by class, so head(N) would be
    # single-class and any collapse-to-one-letter would fake ~100% accuracy
    # (bug caught 2026-08-23: random-direction α=2 "hit" 0.975 on all-Cargo 200)
    te = aug[aug["split"] == "test"]
    test = pd.concat([
        g.sample(n=min(N_CLIPS // 4, len(g)), random_state=SEED)
        for _, g in te.groupby("label_id")]).reset_index(drop=True)

    import torch
    model = build_model({"name": MODEL, "path": PATH, "device": "cuda:0", "device_map": None})
    layer_module = model.model.model.language_model.layers[max(L - 1, 0)]
    rng = np.random.default_rng(SEED)
    rand_dirs = {c: (lambda v: v / np.linalg.norm(v))(rng.standard_normal(X.shape[2]).astype(np.float32))
                 for c in dirs}

    def make_hook(direction_for, alpha):
        def hook(_m, _i, output):
            if alpha == 0.0:
                return None
            # decoder layers may return a bare tensor or a tuple — handle both
            # (slicing a bare tensor with output[1:] yields scalar garbage)
            h = output[0] if isinstance(output, tuple) else output   # [1, T, D]
            pooled = h.float().mean(dim=1)                          # [1, D]
            delta = torch.from_numpy(direction_for).to(h.device, h.dtype)
            mod = h + alpha * delta * pooled.norm()
            return (mod,) + tuple(output[1:]) if isinstance(output, tuple) else mod
        return hook

    results = []
    for mode, dset in [("probe", dirs), ("random", rand_dirs)]:
        for alpha in ALPHAS:
            preds, golds = [], []
            handle = layer_module.register_forward_hook(
                make_hook(dset[0], alpha))                  # direction set per-clip below
            for row in test.itertuples(index=False):
                handle.remove()
                c_true = int(row.label_id)
                direction = dset[c_true] if mode == "probe" else rand_dirs[c_true]
                handle = layer_module.register_forward_hook(make_hook(direction, alpha))
                img = Image.open(getattr(row, f"{SPEC}_path")).convert("RGB")
                msgs = build_prompt(media=img, label_names=names, regime="zero_shot",
                                    support=None, media_type="image", lang="en", enrich_text=None)
                scores = model.choice_logprobs([msgs], choices=letters)[0]
                preds.append(int(_ids[letters.index(max(scores, key=scores.get))]))
                golds.append(c_true)
            handle.remove()
            acc = float(np.mean([p == g for p, g in zip(preds, golds)]))
            results.append({"mode": mode, "alpha": alpha, "acc": acc,
                            "n": len(preds)})
            print(f"[{mode} α={alpha}] acc={acc:.3f}")

    payload = {"model": MODEL, "spec": SPEC, "layer": L, "probe_acc": probe_acc,
               "alphas": ALPHAS, "results": results}
    (RESULTS / "steering.json").write_text(json.dumps(payload, indent=1))
    (RESULTS / "steering.done").write_text("done\n")
    print("\n✅ steering complete")


if __name__ == "__main__":
    main()
