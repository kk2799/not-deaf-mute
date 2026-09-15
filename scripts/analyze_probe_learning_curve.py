#!/usr/bin/env python
"""Probe learning-curve analysis (评测C companion): how many labelled clips does
a linear head need to salvage a frozen LALM/VLM? — the practical headline figure.

Covers ALL linear-probe experiments:
  * outputs/features/layerprobe/<model>_<spec>_<hash>/   (评测C LLM sweeps, 3k train)
  * outputs/features/anchorlayer/                         (WavLM 25-layer full-train,
                                                           BEATs 13-layer)
For each sweep: train the probe on stratified subsets of increasing size
(100/500/1000/2000/5000/all — sizes beyond availability are skipped) and report
per-layer test accuracy → learning_curves.json. CPU-only; C fixed to the value
the full-size sweep's curve json selected (fast, no inner CV — sizes compared).
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

FEATURES = Path("outputs/features/layerprobe")
ANCHOR = Path("outputs/features/anchorlayer")
RESULTS = Path("outputs/results/layer_probe")
SIZES = [100, 500, 1000, 2000, 5000, None]     # None = all available train clips
SEED = 123


def load_sweep(d: Path):
    shards = sorted(d.glob("shard_*.npz"))
    X = np.concatenate([np.load(p, allow_pickle=False)["X"] for p in shards])   # [N, L+1, D] fp16
    y = np.concatenate([np.load(p, allow_pickle=False)["y"] for p in shards]).astype(int)
    te = np.concatenate([np.load(p, allow_pickle=False)["split"] for p in shards]).astype(bool)
    return X, y, te


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweeps", default=None, help="comma list of sweep dir names (default: all)")
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    pool = [FEATURES, ANCHOR]
    dirs = []
    if args.sweeps:
        for s in args.sweeps.split(","):
            for base in pool:
                d = base / s
                if d.is_dir():
                    dirs.append(d)
    else:
        for base in pool:
            dirs += sorted(base.glob("*"))
    out = {}
    for d in dirs:
        if not d.is_dir() or not list(d.glob("shard_*.npz")):
            continue
        X, y, te = load_sweep(d)
        tr_mask = ~te
        c_fix = 1.0
        stem = d.name.rsplit("_", 1)[0] if d.parent == FEATURES else None
        cj = RESULTS / f"{stem}.json" if stem else None
        if cj and cj.exists():
            best = json.loads(cj.read_text())["best"]
            c_fix = best.get("C", 1.0)
        rng = np.random.default_rng(SEED)
        tr_idx = np.where(tr_mask)[0]
        curves, seen_n = {}, set()
        for n in SIZES:
            if n is None or n >= len(tr_idx):
                sub, tag = tr_idx, "all"
            else:
                # rng.choice returns positions WITHIN the train subset — map
                # through tr_idx to actual X row indices (review-caught bug:
                # using them directly trained on the first-n TEST rows, since
                # ordered_rows puts all test rows first)
                sub = tr_idx[np.concatenate([
                    rng.choice(np.where(y[tr_idx] == c)[0], size=min(n // 4, (y[tr_idx] == c).sum()),
                               replace=False)
                    for c in np.unique(y[tr_idx])])]
                tag = str(n)
            if len(sub) in seen_n:
                continue                      # skip sizes collapsing to same n
            assert not te[sub].any(), "train subsample leaked test rows"
            seen_n.add(len(sub))
            curves[tag] = {}
            for layer in range(X.shape[1]):
                Xl = X[:, layer, :].astype(np.float32)
                pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=c_fix))
                pipe.fit(Xl[sub], y[sub])
                pred = pipe.predict(Xl[te])
                curves[tag][layer] = float(accuracy_score(y[te], pred))
            best_l = max(curves[tag], key=curves[tag].get)
            print(f"[{d.name}] n={len(sub):5d}: best layer {best_l} "
                  f"acc={curves[tag][best_l]:.3f} (C={c_fix})", flush=True)
        out[d.name] = {"C": c_fix, "n_train_total": int(tr_mask.sum()),
                       "n_test": int(te.sum()), "curves": curves}
    (RESULTS / "learning_curves.json").write_text(json.dumps(out, indent=1))
    print(f"\n✅ → {RESULTS / 'learning_curves.json'}")


if __name__ == "__main__":
    main()
