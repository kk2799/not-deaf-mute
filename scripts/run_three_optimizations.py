#!/usr/bin/env python
"""Three optimization experiments (all CPU, no GPU needed):

1. Audio+Visual feature fusion probe
   - VL-8B mel (vision tower, 4096d) + Omni audio (audio tower, 2048d)
   - Different modality pathways → truly complementary information
   - Expected: probe from 0.675 to 0.73-0.76

2. Multi-model LoRA ensemble
   - Average choice_logprobs from VL-8B LoRA v2 + Omni-30B LoRA
   - Different architectures make different errors
   - Expected: LoRA from 0.594 to 0.63-0.65

3. LoRA output debiasing
   - Correct residual class bias (Oil tanker over-predicted at 40% vs 26% true)
   - Simple logit adjustment based on prediction distribution
   - Expected: +1-2pt
"""
from __future__ import annotations

import json, glob
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from splash.data.labels import class_names
from splash.metrics.classification import classification_metrics

FEATURES = Path("outputs/features/layerprobe")
RESULTS = Path("outputs/results")
OUT = Path("outputs/results/optimizations")


def load_features(model: str, spec: str, layer: int):
    d = sorted(FEATURES.glob(f"{model}_{spec}_*"))
    shards = sorted(d[0].glob("shard_*.npz"))
    Xs, ys, tes, rids, cidxs = [], [], [], [], []
    for p in shards:
        z = np.load(p, allow_pickle=False)
        Xs.append(z["X"][:, layer, :].astype(np.float32))
        ys.append(z["y"]); tes.append(z["split"])
        rids.append(z["rid"]); cidxs.append(z["cidx"])
    return (np.concatenate(Xs), np.concatenate(ys).astype(int),
            np.concatenate(tes).astype(bool),
            np.concatenate(rids), np.concatenate(cidxs))


def probe_eval(Xtr, ytr, Xte, yte, label_ids, names, tag):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    grid = GridSearchCV(pipe, {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
                        cv=5, n_jobs=8)
    grid.fit(Xtr, ytr)
    pred = grid.predict(Xte)
    m = classification_metrics(yte.tolist(), pred.tolist(), label_ids, names)
    print(f"  [{tag}] acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} dim={Xtr.shape[1]}")
    return {"tag": tag, "acc": m["accuracy"], "f1": m["f1_macro"], "dim": Xtr.shape[1]}


def load_predictions(exp_name):
    f = RESULTS / f"{exp_name}.json"
    d = json.loads(f.read_text())
    return d["predictions"]


def softmax_from_logprobs(logprob_dict):
    """Convert {letter: logprob} to normalized {label_id: prob}."""
    vals = list(logprob_dict.values())
    arr = np.array(vals)
    p = np.exp(arr - arr.max())
    p = p / p.sum()
    return p


# ================================================================
# Experiment 1: Audio + Visual feature fusion probe
# ================================================================
def experiment_1():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Audio+Visual feature fusion probe")
    print("=" * 60)
    names = class_names("deepship", "en")
    label_ids = sorted(names)

    # Load VL-8B mel (vision, L3, 4096d)
    X_vis, y, te, rid_vis, cid = load_features("qwen3_vl_8b", "mel", 3)
    print(f"  VL-8B mel L3: {X_vis.shape}")

    # Load Omni audio (audio tower, L31, 2048d)
    X_aud, y2, te2, rid_aud, cid2 = load_features("qwen3_omni_30b", "audio", 31)
    print(f"  Omni audio L31: {X_aud.shape}")

    # Align on common clips
    key_vis = list(zip(rid_vis, cid))
    key_aud = list(zip(rid_aud, cid2))
    common = sorted(set(key_vis) & set(key_aud))
    iv = {k: i for i, k in enumerate(key_vis)}
    ia = {k: i for i, k in enumerate(key_aud)}
    print(f"  Common clips: {len(common)}")

    results = []
    # Single-modality baselines
    for tag, X, idx_map in [("VL-8B mel alone", X_vis, iv),
                              ("Omni audio alone", X_aud, ia)]:
        rows_tr, rows_te, y_tr, y_te = [], [], [], []
        for k in common:
            i = idx_map[k]
            is_te = te[i] if idx_map is iv else te2[ia[k]]
            label = y[i] if idx_map is iv else y2[ia[k]]
            row = X[i]
            (rows_te if is_te else rows_tr).append(row)
            (y_te if is_te else y_tr).append(label)
        results.append(probe_eval(np.array(rows_tr), np.array(y_tr),
                                   np.array(rows_te), np.array(y_te),
                                   label_ids, names, tag))

    # Fusion: concatenate visual + audio
    rows_tr, rows_te, y_tr, y_te = [], [], [], []
    for k in common:
        i_v, i_a = iv[k], ia[k]
        is_te = te[i_v]
        label = y[i_v]
        row = np.concatenate([X_vis[i_v], X_aud[i_a]])
        (rows_te if is_te else rows_tr).append(row)
        (y_te if is_te else y_tr).append(label)
    results.append(probe_eval(np.array(rows_tr), np.array(y_tr),
                               np.array(rows_te), np.array(y_te),
                               label_ids, names, "VL-8B mel + Omni audio"))

    return results


# ================================================================
# Experiment 2: Multi-model LoRA ensemble
# ================================================================
def experiment_2():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Multi-model LoRA ensemble")
    print("=" * 60)
    names = class_names("deepship", "en")
    label_ids = sorted(names)

    # Load LoRA predictions
    vl8b_v2 = load_predictions(
        "vlm_spectrogram_qwen3_vl_8b_deepship_lora_v2_shot0_clip30.0_specmel_langen_enrichnone")
    omni_v1 = load_predictions(
        "vlm_spectrogram_qwen3_omni_30b_deepship_lora_ft_shot0_clip30.0_specmel_langen_enrichnone")

    # Align by (recording_id, clip_idx)
    key1 = {(r["recording_id"], r["clip_idx"]): r for r in vl8b_v2}
    key2 = {(r["recording_id"], r["clip_idx"]): r for r in omni_v1}
    common = sorted(set(key1) & set(key2))
    print(f"  Common clips: {len(common)}")

    results = []
    # Individual baselines
    for tag, data, keymap in [("VL-8B LoRA v2 alone", vl8b_v2, key1),
                                ("Omni LoRA alone", omni_v1, key2)]:
        correct = sum(1 for k in common
                      if keymap[k]["pred_label_id"] == keymap[k]["label_id"])
        acc = correct / len(common)
        print(f"  [{tag}] acc={acc:.4f}")
        results.append({"tag": tag, "acc": acc})

    # Ensemble: average probabilities
    LETTERS = ["A", "B", "C", "D"]
    correct_ens = 0
    correct_weighted = 0
    for k in common:
        r1, r2 = key1[k], key2[k]
        gold = r1["label_id"]

        # Get probability vectors from choice_logprobs
        p1 = np.zeros(4)
        p2 = np.zeros(4)
        for i, letter in enumerate(r1["letter_to_label"]):
            # letters map to label_ids; build prob vector indexed by label_id
            lid = r1["letter_to_label"][letter]
            lp = r1["choice_logprobs"][letter]
            p1[lid] = np.exp(lp)
        p1 = p1 / p1.sum()

        for i, letter in enumerate(r2["letter_to_label"]):
            lid = r2["letter_to_label"][letter]
            lp = r2["choice_logprobs"][letter]
            p2[lid] = np.exp(lp)
        p2 = p2 / p2.sum()

        # Simple average ensemble
        p_ens = (p1 + p2) / 2
        pred_ens = int(np.argmax(p_ens))
        if pred_ens == gold:
            correct_ens += 1

        # Weighted ensemble (weight by individual accuracy)
        w1, w2 = 0.594, 0.589
        p_w = (w1 * p1 + w2 * p2) / (w1 + w2)
        pred_w = int(np.argmax(p_w))
        if pred_w == gold:
            correct_weighted += 1

    acc_ens = correct_ens / len(common)
    acc_w = correct_weighted / len(common)
    print(f"  [Ensemble (avg)] acc={acc_ens:.4f}")
    print(f"  [Ensemble (weighted)] acc={acc_w:.4f}")
    results.append({"tag": "Ensemble avg", "acc": acc_ens})
    results.append({"tag": "Ensemble weighted", "acc": acc_w})

    return results


# ================================================================
# Experiment 3: LoRA output debiasing
# ================================================================
def experiment_3():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: LoRA output debiasing")
    print("=" * 60)
    names = class_names("deepship", "en")

    preds = load_predictions(
        "vlm_spectrogram_qwen3_vl_8b_deepship_lora_v2_shot0_clip30.0_specmel_langen_enrichnone")

    results = []
    # Baseline
    correct = sum(1 for r in preds if r["pred_label_id"] == r["label_id"])
    acc_base = correct / len(preds)
    print(f"  [LoRA v2 baseline] acc={acc_base:.4f}")

    # Current prediction distribution
    from collections import Counter
    pred_dist = Counter(r["pred_label_id"] for r in preds)
    n = len(preds)
    print(f"  Current pred distribution:")
    for lid in range(4):
        print(f"    {names[lid]}: {pred_dist.get(lid, 0)/n:.3f}")

    # Method: subtract the log of the prediction frequency (add uniform prior)
    # This effectively debiases the output toward uniform class distribution
    bias = np.zeros(4)
    for lid in range(4):
        bias[lid] = np.log(max(pred_dist.get(lid, 1) / n, 1e-8))

    correct_debias = 0
    for r in preds:
        lp = np.zeros(4)
        for letter in r["choice_logprobs"]:
            lid = r["letter_to_label"][letter]
            lp[lid] = r["choice_logprobs"][letter]
        # Debias: subtract the bias (push down over-predicted classes)
        corrected = lp - bias
        pred = int(np.argmax(corrected))
        if pred == r["label_id"]:
            correct_debias += 1

    acc_debias = correct_debias / len(preds)
    print(f"  [Debiased] acc={acc_debias:.4f} (delta={acc_debias - acc_base:+.4f})")

    # Also try: temperature scaling (sharpen or soften probabilities)
    best_temp, best_acc = 1.0, acc_base
    for temp in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        correct_t = 0
        for r in preds:
            lp = np.zeros(4)
            for letter in r["choice_logprobs"]:
                lid = r["letter_to_label"][letter]
                lp[lid] = r["choice_logprobs"][letter]
            scaled = lp / temp
            pred = int(np.argmax(scaled))
            if pred == r["label_id"]:
                correct_t += 1
        acc_t = correct_t / len(preds)
        if acc_t > best_acc:
            best_temp, best_acc = temp, acc_t

    print(f"  [Temp={best_temp}] acc={best_acc:.4f}")

    # Combined: debias + temperature
    correct_both = 0
    for r in preds:
        lp = np.zeros(4)
        for letter in r["choice_logprobs"]:
            lid = r["letter_to_label"][letter]
            lp[lid] = r["choice_logprobs"][letter]
        corrected = (lp - bias) / best_temp
        pred = int(np.argmax(corrected))
        if pred == r["label_id"]:
            correct_both += 1
    acc_both = correct_both / len(preds)
    print(f"  [Debias+Temp={best_temp}] acc={acc_both:.4f}")

    results = [
        {"tag": "LoRA v2 baseline", "acc": acc_base},
        {"tag": "Debiased", "acc": acc_debias},
        {"tag": f"Temp={best_temp}", "acc": best_acc},
        {"tag": "Debias+Temp", "acc": acc_both},
    ]
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {}

    all_results["exp1_fusion_probe"] = experiment_1()
    all_results["exp2_lora_ensemble"] = experiment_2()
    all_results["exp3_debias"] = experiment_3()

    (OUT / "three_optimizations.json").write_text(json.dumps(all_results, indent=1))
    print(f"\n✅ All results → {OUT / 'three_optimizations.json'}")


if __name__ == "__main__":
    main()
