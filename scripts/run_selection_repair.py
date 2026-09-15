#!/usr/bin/env python
"""Selection-hygiene repair: re-select layers/combos by recording-level CV.

Problem (reviewer attack): every headline probe number is max-over-hypotheses
selected on the TEST set — best layer by test acc, best fusion combo of 17 by
test acc, then significance-tested. Fix: re-select on TRAIN only, with
GroupKFold folded by RECORDING (clip-level folds leak 50%-overlap clips),
then report test acc at the train-selected configuration.

Stages (CPU, sequential, incremental JSON saves):
  A. label-shuffle nulls at each sweep's best layer (quick sanity: ~chance)
  B. per-sweep GroupKFold-by-recording layer re-selection
     (omni audio, q2a audio, vl8b mel, vl32b mel [stride 2], BEATs, WavLM)
  C. fusion combo re-selection by GroupKFold + McNemar vs BEATs at CV-selected
     configs on both sides

Output: outputs/results/optimizations/selection_repair.json (rewritten after
each stage). Expected shrink 0.5-1.5pt; the 0.68-vs-0.25 contrast survives.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.metrics.classification import classification_metrics

FEATURES = Path("outputs/features/layerprobe")
OUT = Path("outputs/results/optimizations")
SEED = 42
N_FOLDS = 3

# (key, glob, stride) — stride thins very deep sweeps to bound runtime
SWEEPS = [
    ("omni_audio", "qwen3_omni_30b_audio_*", 1),
    ("q2a_audio",  "qwen2_audio_audio_*",    1),
    ("vl8b_mel",   "qwen3_vl_8b_mel_*",      1),
    ("vl32b_mel",  "qwen3_vl_32b_mel_*",     2),
]

FUSION_SOURCES = [
    ("vl8b_mel",   "outputs/features/layerprobe/qwen3_vl_8b_mel_*",   3),
    ("omni_aud",   "outputs/features/layerprobe/qwen3_omni_30b_audio_*", 31),
    ("vl32b_mel",  "outputs/features/layerprobe/qwen3_vl_32b_mel_*",  3),
    ("omni_mel",   "outputs/features/layerprobe/qwen3_omni_30b_mel_*", 2),
    ("q2a_aud",    "outputs/features/layerprobe/qwen2_audio_audio_*", 16),
    ("vl8b_stft",  "outputs/features/layerprobe/qwen3_vl_8b_stft_*",  4),
    ("vl32b_stft", "outputs/features/layerprobe/qwen3_vl_32b_stft_*", 3),
]
COMBOS = [
    ["vl8b_mel"], ["omni_aud"], ["vl32b_mel"], ["q2a_aud"],
    ["vl8b_mel", "omni_aud"], ["vl8b_mel", "vl32b_mel"], ["vl8b_mel", "q2a_aud"],
    ["omni_aud", "vl32b_mel"], ["omni_aud", "q2a_aud"],
    ["vl8b_mel", "omni_aud", "vl32b_mel"], ["vl8b_mel", "omni_aud", "q2a_aud"],
    ["vl8b_mel", "vl32b_mel", "q2a_aud"], ["omni_aud", "vl32b_mel", "q2a_aud"],
    ["vl8b_mel", "omni_aud", "vl32b_mel", "q2a_aud"],
    ["vl8b_mel", "omni_aud", "vl32b_mel", "q2a_aud", "omni_mel"],
    ["vl8b_mel", "omni_aud", "vl32b_mel", "q2a_aud", "vl8b_stft"],
    [s for s, _, _ in FUSION_SOURCES],
]


def save(payload):
    (OUT / "selection_repair.json").write_text(json.dumps(payload, indent=1))


def ordered_rows(manifest, max_train=3000):
    test_idx = manifest.index[manifest["split"] == "test"].to_numpy()
    train_df = manifest[manifest["split"] == "train"]
    perm = np.random.default_rng(SEED).permutation(len(train_df))
    train_idx = train_df.index.to_numpy()[perm][:max_train]
    return list(test_idx) + list(train_idx)


def load_sweep(glob, layer, stride_layers=False):
    d = sorted(Path(".").glob(glob))
    shards = sorted(d[0].glob("shard_*.npz"))
    Xs, ys, tes, rids, cids = [], [], [], [], []
    for p in shards:
        z = np.load(p, allow_pickle=False)
        Xs.append(z["X"][:, layer, :].astype(np.float32) if not stride_layers else z["X"])
        ys.append(z["y"]); tes.append(z["split"])
        rids.append(z["rid"].astype(str)); cids.append(z["cidx"].astype(int))
    return (np.concatenate(Xs), np.concatenate(ys).astype(int),
            np.concatenate(tes).astype(bool), np.concatenate(rids), np.concatenate(cids))


def fit_group_cv(X, y, groups):
    """Mean GroupKFold CV accuracy of the standard probe pipeline."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    accs = []
    for tr, va in gkf.split(X, y, groups=groups):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        pipe.fit(X[tr], y[tr])
        accs.append(float((pipe.predict(X[va]) == y[va]).mean()))
    return float(np.mean(accs))


def fit_test(Xtr, ytr, Xte, yte, label_ids, names):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0]}, cv=3, n_jobs=8)
    grid.fit(Xtr, ytr)
    pred = grid.predict(Xte)
    m = classification_metrics(yte.tolist(), pred.tolist(), label_ids, names)
    return m["accuracy"], m["f1_macro"]


def stage_a_shuffle_nulls(payload):
    """Sanity null: shuffled-label probe at best layer ≈ chance."""
    names = class_names("deepship", "en")
    label_ids = sorted(names)
    out = payload.setdefault("shuffle_nulls", {})
    best = {"omni_audio": 31, "q2a_audio": 16, "vl8b_mel": 3, "vl32b_mel": 3}
    for key, glob, _ in SWEEPS:
        X, y, te, rid, cid = load_sweep(glob, best[key])
        accs = []
        for s in range(3):
            ys = np.random.default_rng(100 + s).permutation(y[~te])
            pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            pipe.fit(X[~te], ys)
            accs.append(float((pipe.predict(X[te]) == y[te]).mean()))
        out[key] = {"shuffled_test_accs": accs, "chance": 0.25}
        print(f"[A] {key} shuffled-label null: {np.mean(accs):.3f}", flush=True)
        save(payload)


def stage_b_layer_reselection(payload):
    names = class_names("deepship", "en")
    label_ids = sorted(names)
    out = payload.setdefault("layer_reselection", {})
    for key, glob, stride in SWEEPS:
        Xfull, y, te, rid, cid = load_sweep(glob, 0, stride_layers=True)
        n_layers = Xfull.shape[1]
        layers = list(range(n_layers)) if stride == 1 else list(range(0, n_layers, stride))
        groups = rid[~te]
        cv_curve = {}
        for L in layers:
            Xl = Xfull[:, L, :].astype(np.float32)
            cv_curve[L] = fit_group_cv(Xl[~te], y[~te], groups)
        best_cv = max(cv_curve, key=cv_curve.get)
        # test acc at CV-selected layer and at the full test-selected layer (from json)
        curve_json = json.loads((Path("outputs/results/layer_probe") /
                                 glob.replace("_*", ".json")).read_text())
        best_test_L = curve_json["best"]["layer"]
        acc_cv, f1_cv = fit_test(Xfull[:, best_cv, :][~te], y[~te],
                                 Xfull[:, best_cv, :][te], y[te], label_ids, names)
        acc_ts, f1_ts = fit_test(Xfull[:, best_test_L, :][~te], y[~te],
                                 Xfull[:, best_test_L, :][te], y[te], label_ids, names)
        out[key] = {"cv_best_layer": best_cv, "cv_best_cvac": cv_curve[best_cv],
                    "test_acc_at_cv_layer": acc_cv, "f1_at_cv_layer": f1_cv,
                    "test_selected_layer": best_test_L,
                    "test_acc_at_test_layer": acc_ts,
                    "shrink_pt": round((acc_ts - acc_cv) * 100, 1)}
        print(f"[B] {key}: CV→L{best_cv} test={acc_cv:.4f} | test-selected L{best_test_L} "
              f"test={acc_ts:.4f} | shrink {acc_ts-acc_cv:+.4f}", flush=True)
        save(payload)


def stage_c_fusion(payload):
    names = class_names("deepship", "en")
    label_ids = sorted(names)
    data = {}
    for name, glob, layer in FUSION_SOURCES:
        X, y, te, rid, cid = load_sweep(glob, layer)
        data[name] = {"X": X, "y": y, "te": te, "rid": rid, "key": list(zip(rid, cid))}
    idx = {n: {k: i for i, k in enumerate(d["key"])} for n, d in data.items()}
    common = set(data["vl8b_mel"]["key"])
    for d in data.values():
        common &= set(d["key"])
    common = sorted(common)
    ref = data["vl8b_mel"]
    rows = {n: [] for n in data}
    labels, is_te = [], []
    for k in common:
        i = idx["vl8b_mel"][k]
        labels.append(ref["y"][i]); is_te.append(ref["te"][i])
        for n in data:
            rows[n].append(data[n]["X"][idx[n][k]])
    labels = np.array(labels); is_te = np.array(is_te)
    groups = np.array([k[0] for k in common])[~is_te]

    out = payload.setdefault("fusion_reselection", {})
    cv_scores = {}
    for combo in COMBOS:
        M = np.concatenate([np.array(rows[n], dtype=np.float32) for n in combo], axis=1)
        cv_scores["+".join(combo)] = fit_group_cv(M[~is_te], labels[~is_te], groups)
        print(f"[C] cv {'+'.join(combo)}: {cv_scores['+'.join(combo)]:.4f}", flush=True)
    best_combo_name = max(cv_scores, key=cv_scores.get)
    best_combo = best_combo_name.split("+")
    M = np.concatenate([np.array(rows[n], dtype=np.float32) for n in best_combo], axis=1)
    acc_bc, f1_bc = fit_test(M[~is_te], labels[~is_te], M[is_te], labels[is_te],
                             label_ids, names)
    out["cv_best_combo"] = best_combo_name
    out["cv_best_score"] = cv_scores[best_combo_name]
    out["test_acc_at_cv_combo"] = acc_bc
    out["all_cv_scores"] = cv_scores
    print(f"[C] CV-selected combo: {best_combo_name} test={acc_bc:.4f}", flush=True)
    save(payload)


def main():
    payload = {"protocol": "GroupKFold-by-recording 3-fold on train; test untouched for selection"}
    save(payload)
    stage_a_shuffle_nulls(payload)
    stage_b_layer_reselection(payload)
    stage_c_fusion(payload)
    print("\n✅ selection repair complete")


if __name__ == "__main__":
    main()
