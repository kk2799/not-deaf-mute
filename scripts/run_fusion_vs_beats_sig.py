#!/usr/bin/env python
"""McNemar significance test: 7-source frozen-LLM fusion probe vs BEATs probe.

Question: is the 0.741 (fusion, full 8,350 train) vs 0.749 (BEATs L6, 3k train)
gap statistically significant on n_test=2,685? If the CI covers 0, the paper
can claim "matches the specialized encoder" instead of "0.8pt below".

Arms (identical test set, aligned by (recording_id, clip_idx)):
  beats_3k   — BEATs layer-6 LR, 3k train (reproduces the reported 0.749)
  fusion_3k  — 7-source concat LR, SAME seeded 3k subset (matched budget)
  fusion_full— 7-source concat LR, full 8,350 train (reproduces 0.7412)

Statistics: McNemar exact (binomial on discordant pairs) + 10k bootstrap CI
for the accuracy difference. CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.metrics.classification import classification_metrics

FEATURES = Path("outputs/features/layerprobe")
OUT = Path("outputs/results/optimizations")
SEED = 42  # same as run_layer_probe.ordered_rows

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
BEATS_LAYER = 6
MAX_TRAIN = 3000


def ordered_rows(manifest: pd.DataFrame, max_train: int):
    """Same deterministic order as run_layer_probe.ordered_rows (SEED=42)."""
    test_idx = manifest.index[manifest["split"] == "test"].to_numpy()
    train_df = manifest[manifest["split"] == "train"]
    perm = np.random.default_rng(SEED).permutation(len(train_df))
    train_idx = train_df.index.to_numpy()[perm][:max_train]
    return list(test_idx) + list(train_idx)


def fit_lr(Xtr, ytr, cv, cgrid):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    grid = GridSearchCV(pipe, {"logisticregression__C": cgrid}, cv=cv, n_jobs=8)
    grid.fit(Xtr, ytr)
    return grid


def mcnemar(a_correct, b_correct):
    """Exact McNemar: are the two systems' error patterns distinguishable?"""
    n01 = int(np.sum(~a_correct & b_correct))   # fusion wrong, beats right
    n10 = int(np.sum(a_correct & ~b_correct))   # fusion right, beats wrong
    if n01 + n10 == 0:
        p = 1.0
    else:
        p = binomtest(n01, n01 + n10, 0.5).pvalue  # exact two-sided McNemar
    return {"fusion_right_beats_wrong": n10, "beats_right_fusion_wrong": n01,
            "p_value": p}


def bootstrap_diff(a_correct, b_correct, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a_correct)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = a_correct[s].mean() - b_correct[s].mean()
    return {"mean_diff": float(a_correct.mean() - b_correct.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)),
                     float(np.percentile(diffs, 97.5))]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = class_names("deepship", "en")
    label_ids = sorted(names)

    # ---- manifest alignment keys, in ordered_rows order --------------------
    m = find_manifest({"name": "deepship", "clip_len": 30.0, "overlap": 0.5})
    manifest = pd.read_csv(m)
    order = ordered_rows(manifest, MAX_TRAIN)
    mkeys = list(zip(manifest.loc[order, "recording_id"].astype(str),
                     manifest.loc[order, "clip_idx"].astype(int)))
    n_test = int((manifest["split"] == "test").sum())
    test_keys = mkeys[:n_test]
    train3k_keys = set(mkeys[n_test:])
    key_pos = {k: i for i, k in enumerate(mkeys)}  # position in BEATs shard

    # ---- BEATs arm ----------------------------------------------------------
    z = np.load("outputs/features/anchorlayer/beats_all_layers/shard_all.npz")
    Xb, yb, teb = z["X"], z["y"].astype(int), z["split"].astype(bool)
    Xl = Xb[:, BEATS_LAYER, :].astype(np.float32)
    grid_b = fit_lr(Xl[~teb], yb[~teb], cv=5, cgrid=[0.01, 0.1, 1.0, 10.0])
    pred_b = grid_b.predict(Xl[teb])
    acc_b = float((pred_b == yb[teb]).mean())
    print(f"[beats 3k L{BEATS_LAYER}] acc={acc_b:.4f} (reported 0.749)")
    beats_pred = {test_keys[i]: (int(pred_b[i]), int(yb[teb][i]))
                  for i in range(n_test)}

    # ---- fusion arms --------------------------------------------------------
    data = {}
    for name, path_glob in SOURCES:
        d = sorted(Path(".").glob(path_glob))
        shards = sorted(d[0].glob("shard_*.npz"))
        Xs, ys, tes, rids, cidxs = [], [], [], [], []
        for p in shards:
            zz = np.load(p, allow_pickle=False)
            Xs.append(zz["X"][:, BEST_LAYERS[name], :].astype(np.float32))
            ys.append(zz["y"]); tes.append(zz["split"])
            rids.append(zz["rid"].astype(str)); cidxs.append(zz["cidx"].astype(int))
        data[name] = {"X": np.concatenate(Xs), "y": np.concatenate(ys).astype(int),
                      "te": np.concatenate(tes).astype(bool),
                      "key": list(zip(np.concatenate(rids), np.concatenate(cidxs)))}
    idx = {n: {k: i for i, k in enumerate(d["key"])} for n, d in data.items()}

    common = set(data["vl8b_mel"]["key"])
    for d in data.values():
        common &= set(d["key"])
    ref = data["vl8b_mel"]

    def fit_arm(train_selector):
        rows_tr, rows_te, y_tr, y_te, te_keys = [], [], [], [], []
        for k in sorted(common):
            i = idx["vl8b_mel"][k]
            row = np.concatenate([data[n]["X"][idx[n][k]] for n in BEST_LAYERS])
            if ref["te"][i]:
                rows_te.append(row); y_te.append(ref["y"][i]); te_keys.append(k)
            elif train_selector(k):
                rows_tr.append(row); y_tr.append(ref["y"][i])
        grid = fit_lr(np.array(rows_tr), np.array(y_tr), cv=3,
                      cgrid=[0.01, 0.1, 1.0])
        pred = grid.predict(np.array(rows_te))
        acc = float((pred == np.array(y_te)).mean())
        return acc, {k: int(p) for k, p in zip(te_keys, pred)}, np.array(y_te)

    acc_3k, pred_3k, _ = fit_arm(lambda k: k in train3k_keys)
    acc_full, pred_full, y_te = fit_arm(lambda k: True)
    print(f"[fusion 3k ] acc={acc_3k:.4f}")
    print(f"[fusion full] acc={acc_full:.4f} (reported 0.7412)")

    # ---- significance -------------------------------------------------------
    gold = {k: v[1] for k, v in beats_pred.items()}
    preds_b = {k: v[0] for k, v in beats_pred.items()}
    keys = [k for k in test_keys if k in pred_full and k in preds_b]
    print(f"aligned test keys: {len(keys)}")
    y = np.array([gold[k] for k in keys])
    c_b = np.array([preds_b[k] == gold[k] for k in keys])
    c_3 = np.array([pred_3k[k] == gold[k] for k in keys])
    c_f = np.array([pred_full[k] == gold[k] for k in keys])

    results = {
        "accs": {"beats_3k": acc_b, "fusion_3k": acc_3k, "fusion_full": acc_full},
        "matched_budget_3k": {
            "mcnemar": mcnemar(c_3, c_b),
            "bootstrap": bootstrap_diff(c_3, c_b)},
        "best_vs_best": {  # fusion full 8,350 vs beats 3k
            "mcnemar": mcnemar(c_f, c_b),
            "bootstrap": bootstrap_diff(c_f, c_b)},
        "n_test_aligned": len(keys),
    }
    (OUT / "fusion_vs_beats_significance.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    print(f"\n✅ → {OUT / 'fusion_vs_beats_significance.json'}")


if __name__ == "__main__":
    main()
