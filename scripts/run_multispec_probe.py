#!/usr/bin/env python
"""Multi-spectrogram joint probe: concatenate features from different spectrograms.

Hypothesis: mel, STFT, and DEMON encode complementary information about the
same audio clip. A linear probe on concatenated features should outperform
any single-spectrogram probe.

  mel   L3  alone → 0.675
  stft  L4  alone → 0.659
  demon L4  alone → 0.483
  mel+stft      → ?   (expected 0.70+)
  mel+stft+demon → ?   (expected 0.71+)

Features are already on disk (layerprobe shards). This is CPU-only, ~10 min.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from splash.data.labels import class_names
from splash.metrics.classification import classification_metrics
from splash.tracking.results import _write_text

FEATURES = Path("outputs/features/layerprobe")
RESULTS = Path("outputs/results")
SHARD = 250

# best layers from single-spec full-budget probes
SPEC_LAYERS = {"mel": 3, "stft": 4, "demon": 4}
MODEL = "qwen3_vl_8b"


def load_features(spec: str, layer: int):
    """Load one spec's features at the given layer, aligned by (rid, cidx)."""
    d = sorted(FEATURES.glob(f"{MODEL}_{spec}_*"))
    if not d:
        raise FileNotFoundError(f"no shards for {MODEL}_{spec}")
    shards = sorted(d[0].glob("shard_*.npz"))
    Xs, ys, tes, rids, cidxs = [], [], [], [], []
    for p in shards:
        z = np.load(p, allow_pickle=False)
        Xs.append(z["X"][:, layer, :].astype(np.float32))   # this layer only
        ys.append(z["y"])
        tes.append(z["split"])
        rids.append(z["rid"])
        cidxs.append(z["cidx"])
    return (np.concatenate(Xs), np.concatenate(ys).astype(int),
            np.concatenate(tes).astype(bool),
            np.concatenate(rids), np.concatenate(cidxs))


def probe(Xtr, ytr, Xte, yte, label_ids, names, tag):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
                        cv=5, n_jobs=8)
    grid.fit(Xtr, ytr)
    pred = grid.predict(Xte)
    m = classification_metrics(yte.tolist(), pred.tolist(), label_ids, names)
    print(f"  [{tag}] clip acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
          f"(C={grid.best_params_['logisticregression__C']}, dim={Xtr.shape[1]})")
    return {"tag": tag, "acc": m["accuracy"], "f1": m["f1_macro"], "dim": Xtr.shape[1]}


def main():
    names = class_names("deepship", "en")
    label_ids = sorted(names)

    # load each spec at its best layer
    data = {}
    for spec, layer in SPEC_LAYERS.items():
        X, y, te, rids, cidxs = load_features(spec, layer)
        key = list(zip(rids, cidxs))
        data[spec] = {"X": X, "y": y, "te": te, "key": key}
        print(f"loaded {spec} L{layer}: {X.shape}, test={te.sum()}")

    # build a common clip order (intersection of all specs)
    common = set(data["mel"]["key"])
    for spec in data:
        common &= set(data[spec]["key"])
    common = sorted(common)
    print(f"common clips across all specs: {len(common)}")

    # index lookup
    idx = {spec: {k: i for i, k in enumerate(d["key"])} for spec, d in data.items()}

    def build(specs):
        """Concatenate features for the given spec list, return train/test split."""
        rows_tr, rows_te = [], []
        y_tr, y_te = [], []
        for k in common:
            i = idx["mel"][k]
            y = data["mel"]["y"][i]
            is_te = data["mel"]["te"][i]
            feats = [data[s]["X"][idx[s][k]] for s in specs]
            row = np.concatenate(feats)
            (rows_te if is_te else rows_tr).append(row)
            (y_te if is_te else y_tr).append(y)
        return np.array(rows_tr), np.array(y_tr), np.array(rows_te), np.array(y_te)

    results = []

    # single-spec baselines (re-verify)
    for spec in ["mel", "stft", "demon"]:
        Xtr, ytr, Xte, yte = build([spec])
        results.append(probe(Xtr, ytr, Xte, yte, label_ids, names, spec))

    # joint probes
    for combo in [("mel", "stft"), ("mel", "demon"), ("stft", "demon"),
                  ("mel", "stft", "demon")]:
        Xtr, ytr, Xte, yte = build(list(combo))
        results.append(probe(Xtr, ytr, Xte, yte, label_ids, names, "+".join(combo)))

    _write_text(RESULTS / "multispec_probe.json", json.dumps(results, indent=1))
    print(f"\n✅ → {RESULTS / 'multispec_probe.json'}")


if __name__ == "__main__":
    main()
