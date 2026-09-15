#!/usr/bin/env python
"""Exhaustive feature fusion probe: explore the ceiling of combined representations.

Incrementally add feature sources, each from a DIFFERENT pre-training pathway:
  1. VL-8B mel L3      (vision tower, natural images, 4096d)
  2. Omni audio L31    (audio tower, speech/audio, 2048d)
  3. BEATs L6          (AudioSet-pretrained AST, 768d)
  4. WavLM L2           (speech SSL, 1024d)
  5. VL-32B mel L3     (larger vision tower, 5120d)
  6. Omni mel L2        (omni vision tower, 2048d)
  7. q2a audio L16     (Qwen2-Audio audio encoder, 4096d)

Test all meaningful combinations, including full fusion. CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from splash.data.labels import class_names
from splash.metrics.classification import classification_metrics

FEATURES = Path("outputs/features/layerprobe")
ANCHOR = Path("outputs/features/anchorlayer")
OUT = Path("outputs/results/optimizations")

# (name, path_glob) — all from layerprobe (have rid/cidx for alignment)
SOURCES = [
    ("vl8b_mel",   "outputs/features/layerprobe/qwen3_vl_8b_mel_*"),
    ("omni_aud",   "outputs/features/layerprobe/qwen3_omni_30b_audio_*"),
    ("vl32b_mel",  "outputs/features/layerprobe/qwen3_vl_32b_mel_*"),
    ("omni_mel",   "outputs/features/layerprobe/qwen3_omni_30b_mel_*"),
    ("q2a_aud",    "outputs/features/layerprobe/qwen2_audio_audio_*"),
    ("vl8b_stft",  "outputs/features/layerprobe/qwen3_vl_8b_stft_*"),
    ("vl32b_stft", "outputs/features/layerprobe/qwen3_vl_32b_stft_*"),
]
# best layers for each source (from single-spec full-budget probes)
BEST_LAYERS = {
    "vl8b_mel": 3, "omni_aud": 31, "vl32b_mel": 3, "omni_mel": 2,
    "q2a_aud": 16, "vl8b_stft": 4, "vl32b_stft": 3,
}


def load_source(name, path_glob):
    """Load features for one source at its best layer."""
    d = sorted(Path(".").glob(path_glob))
    if not d:
        return None
    shards = sorted(d[0].glob("shard_*.npz"))
    if not shards:
        return None
    layer = BEST_LAYERS.get(name, 2)
    Xs, ys, tes, rids, cidxs = [], [], [], [], []
    for p in shards:
        z = np.load(p, allow_pickle=False)
        Xs.append(z["X"][:, layer, :].astype(np.float32))
        ys.append(z["y"]); tes.append(z["split"])
        rids.append(z["rid"]); cidxs.append(z["cidx"])
    return {
        "X": np.concatenate(Xs),
        "y": np.concatenate(ys).astype(int),
        "te": np.concatenate(tes).astype(bool),
        "rid": np.concatenate(rids),
        "cid": np.concatenate(cidxs),
    }


def probe(Xtr, ytr, Xte, yte, label_ids, names, tag):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0]},
                        cv=3, n_jobs=8)
    grid.fit(Xtr, ytr)
    pred = grid.predict(Xte)
    m = classification_metrics(yte.tolist(), pred.tolist(), label_ids, names)
    print(f"  [{tag:55s}] acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} dim={Xtr.shape[1]}", flush=True)
    return {"tag": tag, "acc": m["accuracy"], "f1": m["f1_macro"], "dim": Xtr.shape[1]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = class_names("deepship", "en")
    label_ids = sorted(names)

    # Load all sources
    data = {}
    for name, path_glob in SOURCES:
        d = load_source(name, path_glob)
        if d is not None:
            data[name] = d
            print(f"loaded {name}: {d['X'].shape}, test={d['te'].sum()}")
        else:
            print(f"SKIP {name}: not found")

    # Build common clip set
    all_keys = None
    for name, d in data.items():
        keys = set(zip(d["rid"], d["cid"]))
        if all_keys is None:
            all_keys = keys
        else:
            all_keys &= keys
    common = sorted(all_keys)
    print(f"\nCommon clips across all sources: {len(common)}")

    # Build index lookups
    idx = {}
    for name, d in data.items():
        idx[name] = {k: i for i, k in enumerate(zip(d["rid"], d["cid"]))}

    # Reference labels and split from first source
    ref_name = list(data.keys())[0]
    ref = data[ref_name]

    def build_subset(names_list):
        rows_tr, rows_te, y_tr, y_te = [], [], [], []
        for k in common:
            i = idx[ref_name][k]
            is_te = ref["te"][i]
            label = ref["y"][i]
            row = np.concatenate([data[n]["X"][idx[n][k]] for n in names_list])
            (rows_te if is_te else rows_tr).append(row)
            (y_te if is_te else y_tr).append(label)
        return np.array(rows_tr), np.array(y_tr), np.array(rows_te), np.array(y_te)

    results = []

    # Test combinations
    combos = [
        # singles
        ["vl8b_mel"],
        ["omni_aud"],
        ["vl32b_mel"],
        ["q2a_aud"],
        # pairs (cross-pathway)
        ["vl8b_mel", "omni_aud"],
        ["vl8b_mel", "vl32b_mel"],
        ["vl8b_mel", "q2a_aud"],
        ["omni_aud", "vl32b_mel"],
        ["omni_aud", "q2a_aud"],
        # triples
        ["vl8b_mel", "omni_aud", "vl32b_mel"],
        ["vl8b_mel", "omni_aud", "q2a_aud"],
        ["vl8b_mel", "vl32b_mel", "q2a_aud"],
        ["omni_aud", "vl32b_mel", "q2a_aud"],
        # quads (cross-model, cross-modality)
        ["vl8b_mel", "omni_aud", "vl32b_mel", "q2a_aud"],
        # add more specs
        ["vl8b_mel", "omni_aud", "vl32b_mel", "q2a_aud", "omni_mel"],
        ["vl8b_mel", "omni_aud", "vl32b_mel", "q2a_aud", "vl8b_stft"],
        # full fusion (all LLM sources)
        list(data.keys()),
    ]

    for combo in combos:
        # only test if all sources available
        if not all(n in data for n in combo):
            continue
        tag = " + ".join(combo)
        Xtr, ytr, Xte, yte = build_subset(combo)
        results.append(probe(Xtr, ytr, Xte, yte, label_ids, names, tag))

    (OUT / "full_fusion_probe.json").write_text(json.dumps(results, indent=1))
    print(f"\n✅ → {OUT / 'full_fusion_probe.json'}")

    # Print top 5
    print("\n=== TOP 5 ===")
    for r in sorted(results, key=lambda x: -x["acc"])[:5]:
        print(f"  {r['acc']:.4f} | {r['tag']}")


if __name__ == "__main__":
    main()
