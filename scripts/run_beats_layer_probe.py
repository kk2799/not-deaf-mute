#!/usr/bin/env python
"""Layer-wise probing for the BEATs anchor (protocol symmetry with 评测C).

The vendored backbone already collects per-layer outputs in
``TransformerEncoder`` (``layer_results``); extract_features returns
``(final_x, layer_results)`` where each entry is ``(x, z)`` per layer. We pool
each layer's x over time → 13 feature rows per clip (12 layers; layer_results
holds 12 entries) → same logistic-regression probe protocol as the other
sweeps (C by 5-fold CV on the 3,000-clip train subset, test = full 2,685).
Output: outputs/results/anchor_layer_probe_beats.json. GPU ~20 min.
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from splash.audio.io import read_segment
from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.models.factory import build_model
from splash.metrics.classification import classification_metrics
from splash.tracking.results import _write_text
from scripts.run_layer_probe import ordered_rows

RESULTS = Path("outputs/results")
FEATURES = Path("outputs/features/anchorlayer")
SHARD = 250
SEED = 42
MAX_TRAIN = 3000


@torch.inference_mode()
def extract(enc, rows, out_dir, clip_samples):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shard_all.npz"          # single-pass dataset (~5.7k rows)
    if not path.exists():
        # forward hooks on each of the 12 encoder layers — robust against the
        # vendored extract_features not returning intermediates (the direct
        # encoder path hung). Layers run in T×B×C layout, hence [:, 0].
        acts = []

        def hook(_m, _i, output):
            acts.append(output[0][:, 0].float().mean(dim=0).cpu())

        handles = [m.register_forward_hook(hook) for m in enc.model.encoder.layers]
        feats = []
        for r in rows.itertuples(index=False):
            wav = read_segment(r.path, r.start, r.end, target_sr=16000, pad_to_samples=clip_samples)
            x = torch.from_numpy(wav[None]).to(enc.device)
            acts.clear()
            enc.model.extract_features(x)     # standard proven path; hooks collect
            assert len(acts) == len(enc.model.encoder.layers), "hook miss"
            feats.append(torch.stack(acts).half().numpy())
        for h in handles:
            h.remove()
        X = np.stack(feats)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, X=X, y=rows["label_id"].to_numpy(np.int16),
                 split=(rows["split"] == "test").to_numpy(np.int8))
        os.replace(tmp, path)
        print(f"  extracted {X.shape}", flush=True)
    z = np.load(path)
    return z["X"], z["y"].astype(int), z["split"].astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    manifest = find_manifest({"name": "deepship", "clip_len": 30.0, "overlap": 0.5})
    full = pd.read_csv(manifest)
    order = ordered_rows(full, MAX_TRAIN, False)
    rows = full.iloc[order]
    names = class_names("deepship", "en")
    enc = build_model({"name": "beats_iter3_plus", "path": "/models/BEATs/BEATs_iter3_plus_AS2M.pt",
                       "device": args.device})
    X, y, te = extract(enc, rows, FEATURES / "beats_all_layers", int(30.0 * 16000))
    del enc
    print(f"[beats-layer] X={X.shape} train={(~te).sum()} test={te.sum()}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    curves = []
    for layer in range(X.shape[1]):
        Xl = X[:, layer, :].astype(np.float32)
        grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                           {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}, cv=5, n_jobs=8)
        grid.fit(Xl[~te], y[~te])
        pred = grid.predict(Xl[te])
        m = classification_metrics(y[te].tolist(), pred.tolist(), sorted(names), names)
        curves.append({"layer": layer, "acc": m["accuracy"], "f1_macro": m["f1_macro"]})
        print(f"  layer {layer:2d}: acc={m['accuracy']:.3f} f1={m['f1_macro']:.3f}")
    last, best = curves[-1], max(curves, key=lambda c: c["acc"])
    payload = {"model": "beats_iter3_plus", "protocol": "frozen + linear probe, 3k train (same as 评测C)",
               "n_layers": X.shape[1], "last_layer": last, "best_layer": best, "curves": curves,
               "main_table_anchor_last_layer_acc": 0.728}
    _write_text(RESULTS / "anchor_layer_probe_beats.json", json.dumps(payload, indent=1))
    print(f"\nlast={last['acc']:.3f} best=layer{best['layer']} {best['acc']:.3f} → anchor_layer_probe_beats.json")


if __name__ == "__main__":
    main()
