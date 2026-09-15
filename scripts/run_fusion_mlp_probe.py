#!/usr/bin/env python
"""Nonlinear probe heads on fused features — can we close the last gap to BEATs?

full_fusion_probe.py showed logistic regression on 7-source concat = 0.741
(BEATs 0.749). Two follow-ups tested here (CPU only):

  1. MLP head (512 hidden, ReLU, early stopping) vs LR on the top combos —
     does a mildly nonlinear head extract more from the same features?
  2. Per-source PCA-512 equalization before concat — vl32b_mel (5120d)
     dominates the raw concat; equalized fusion may be better calibrated.

Configs (train 8,350 / test 2,685 DeepShip clips, same split as all probes):
  triple : omni_aud + vl32b_mel + q2a_aud          (LR 0.733)
  five   : triple + vl8b_mel + vl8b_stft           (LR 0.737)
  seven  : five + omni_mel + vl32b_stft            (LR 0.741)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from splash.data.labels import class_names
from splash.metrics.classification import classification_metrics

FEATURES = Path("outputs/features/layerprobe")
OUT = Path("outputs/results/optimizations")

SOURCES = [
    ("vl8b_mel",   "outputs/features/layerprobe/qwen3_vl_8b_mel_*"),
    ("omni_aud",   "outputs/features/layerprobe/qwen3_omni_30b_audio_*"),
    ("vl32b_mel",  "outputs/features/layerprobe/qwen3_vl_32b_mel_*"),
    ("omni_mel",   "outputs/features/layerprobe/qwen3_omni_30b_mel_*"),
    ("q2a_aud",    "outputs/features/layerprobe/qwen2_audio_audio_*"),
    ("vl8b_stft",  "outputs/features/layerprobe/qwen3_vl_8b_stft_*"),
    ("vl32b_stft", "outputs/features/layerprobe/qwen3_vl_32b_stft_*"),
]
BEST_LAYERS = {
    "vl8b_mel": 3, "omni_aud": 31, "vl32b_mel": 3, "omni_mel": 2,
    "q2a_aud": 16, "vl8b_stft": 4, "vl32b_stft": 3,
}

COMBOS = {
    "triple": ["omni_aud", "vl32b_mel", "q2a_aud"],
    "five":   ["vl8b_mel", "omni_aud", "vl32b_mel", "q2a_aud", "vl8b_stft"],
    "seven":  list(BEST_LAYERS.keys()),
}


def load_source(name, path_glob):
    d = sorted(Path(".").glob(path_glob))
    shards = sorted(d[0].glob("shard_*.npz"))
    layer = BEST_LAYERS[name]
    Xs, ys, tes, rids, cidxs = [], [], [], [], []
    for p in shards:
        z = np.load(p, allow_pickle=False)
        Xs.append(z["X"][:, layer, :].astype(np.float32))
        ys.append(z["y"]); tes.append(z["split"])
        rids.append(z["rid"]); cidxs.append(z["cidx"])
    return {
        "X": np.concatenate(Xs), "y": np.concatenate(ys).astype(int),
        "te": np.concatenate(tes).astype(bool),
        "rid": np.concatenate(rids), "cid": np.concatenate(cidxs),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = class_names("deepship", "en")
    label_ids = sorted(names)

    data = {}
    for name, path_glob in SOURCES:
        data[name] = load_source(name, path_glob)
        print(f"loaded {name}: {data[name]['X'].shape}", flush=True)

    all_keys = None
    for d in data.values():
        keys = set(zip(d["rid"], d["cid"]))
        all_keys = keys if all_keys is None else (all_keys & keys)
    common = sorted(all_keys)
    print(f"common clips: {len(common)}", flush=True)
    idx = {n: {k: i for i, k in enumerate(zip(d["rid"], d["cid"]))}
           for n, d in data.items()}
    ref = data["vl8b_mel"]

    def build(names_list, pca_dim=None):
        """Concat (optionally per-source PCA-512) features; return split arrays."""
        per_source = {n: [] for n in names_list}
        labels, is_te = [], []
        for k in common:
            i = idx["vl8b_mel"][k]
            labels.append(ref["y"][i]); is_te.append(ref["te"][i])
            for n in names_list:
                per_source[n].append(data[n]["X"][idx[n][k]])
        labels = np.array(labels); is_te = np.array(is_te)
        feats = {}
        for n in names_list:
            M = np.array(per_source[n], dtype=np.float32)
            if pca_dim is not None and M.shape[1] > pca_dim:
                tr = M[~is_te]
                p = PCA(n_components=pca_dim, random_state=0).fit(tr)
                M = p.transform(M).astype(np.float32)
            feats[n] = M
        rows = np.concatenate([feats[n] for n in names_list], axis=1)
        return rows[~is_te], labels[~is_te], rows[is_te], labels[is_te]

    def lr_head(Xtr, ytr):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0]},
                            cv=3, n_jobs=8)
        return grid.fit(Xtr, ytr)

    def mlp_head(Xtr, ytr):
        pipe = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3, batch_size=256,
                          max_iter=200, early_stopping=True, n_iter_no_change=10,
                          random_state=0))
        return pipe.fit(Xtr, ytr)

    results = []
    for combo_name, srcs in COMBOS.items():
        for tag, pca_dim in [("raw", None), ("pca512", 512)]:
            Xtr, ytr, Xte, yte = build(srcs, pca_dim=pca_dim)
            for head_name, fit in [("lr", lr_head), ("mlp512", mlp_head)]:
                model = fit(Xtr, ytr)
                pred = model.predict(Xte)
                m = classification_metrics(yte.tolist(), pred.tolist(),
                                           label_ids, names)
                rec = {"combo": combo_name, "variant": tag, "head": head_name,
                       "acc": m["accuracy"], "f1": m["f1_macro"], "dim": Xtr.shape[1]}
                results.append(rec)
                print(f"[{combo_name:6s} {tag:6s} {head_name:6s}] "
                      f"acc={rec['acc']:.4f} f1={rec['f1']:.4f} dim={rec['dim']}",
                      flush=True)

    (OUT / "fusion_mlp_probe.json").write_text(json.dumps(results, indent=1))
    print(f"\n✅ → {OUT / 'fusion_mlp_probe.json'}")
    print("\n=== TOP 5 ===")
    for r in sorted(results, key=lambda x: -x["acc"])[:5]:
        print(f"  {r['acc']:.4f} | {r['combo']} {r['variant']} {r['head']}")


if __name__ == "__main__":
    main()
