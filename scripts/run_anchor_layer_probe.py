#!/usr/bin/env python
"""Layer-wise anchor probing: give the WavLM anchor the SAME per-layer treatment
as 评测C gives the LLMs (defuses "last-layer-only / handicapped baseline").

WavLM-Large has 24 transformer layers; one forward with output_hidden_states
gives all 25 hidden states (incl. embeddings) — mean-pool each, probe each with
the identical logistic-regression protocol (C by 5-fold CV on the FULL train
split, test = full 2,685). Report the best layer vs the last layer used in the
main anchor table. CPU-light, GPU ~40 min. Outputs
outputs/results/anchor_layer_probe.json (+ curves.csv append).
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
from splash.metrics.classification import classification_metrics, format_metrics
from splash.tracking.results import _write_text

RESULTS = Path("outputs/results")
FEATURES = Path("outputs/features/anchorlayer")
SHARD = 500


@torch.inference_mode()
def extract(enc, manifest, out_dir, clip_samples):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(manifest)
    for s in range((n + SHARD - 1) // SHARD):
        rows = manifest.iloc[s * SHARD:(s + 1) * SHARD]
        path = out_dir / f"shard_{s:04d}.npz"
        if path.exists():
            try:
                if len(np.load(path)["y"]) == len(rows):
                    continue
            except Exception:
                pass
        feats = []
        for r in rows.itertuples(index=False):
            wav = read_segment(r.path, r.start, r.end, target_sr=16000, pad_to_samples=clip_samples)
            x = torch.from_numpy(wav[None]).to(enc.device)
            x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True) + 1e-7)
            out = enc.model(x, output_hidden_states=True)
            hs = torch.stack([h[0].float().mean(dim=0) for h in out.hidden_states])  # [L+1, D]
            feats.append(hs.half().cpu().numpy())
        X = np.stack(feats)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, X=X, y=rows["label_id"].to_numpy(np.int16),
                 split=(rows["split"] == "test").to_numpy(np.int8))
        os.replace(tmp, path)
        print(f"  shard {s + 1}: {len(rows)} → {X.shape}")


def probe(X, y, te, names):
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
        curves.append({"layer": layer, "acc": m["accuracy"], "f1_macro": m["f1_macro"],
                       "C": float(grid.best_params_["logisticregression__C"])})
        print(f"  layer {layer:2d}: acc={m['accuracy']:.3f} f1={m['f1_macro']:.3f}")
    return curves


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    manifest = find_manifest({"name": "deepship", "clip_len": 30.0, "overlap": 0.5})
    full = pd.read_csv(manifest)
    names = class_names("deepship", "en")
    enc = build_model({"name": "wavlm_large", "path": "/models/WavLM-Large", "device": args.device})
    out_dir = FEATURES / "wavlm_large_all_layers"
    extract(enc, full, out_dir, int(30.0 * 16000))
    del enc

    shards = sorted(out_dir.glob("shard_*.npz"))
    X = np.concatenate([np.load(p)["X"] for p in shards])
    y = np.concatenate([np.load(p)["y"] for p in shards]).astype(int)
    te = np.concatenate([np.load(p)["split"] for p in shards]).astype(bool)
    print(f"[anchor-layer] X={X.shape} train={(~te).sum()} test={te.sum()}")
    curves = probe(X, y, te, names)
    last, best = curves[-1], max(curves, key=lambda c: c["acc"])
    payload = {"model": "wavlm_large", "protocol": "frozen + linear probe, full train, per-layer best",
               "n_layers": X.shape[1], "last_layer": last, "best_layer": best, "curves": curves,
               "main_table_anchor_last_layer_acc": 0.634}
    _write_text(RESULTS / "anchor_layer_probe.json", json.dumps(payload, indent=1))
    print(f"\nlast={last['acc']:.3f} best=layer{best['layer']} {best['acc']:.3f} → anchor_layer_probe.json")


if __name__ == "__main__":
    main()
