#!/usr/bin/env python
"""Round-2 analyses (2026-09-15, CPU-only).

T1  TOST equivalence for A1 (7-source fusion) vs BEATs single layer:
    replicates the recording-clustered bootstrap of a1_a2_bootstrap.py
    exactly (10,000 resamples of 153 recordings, seed 0) and reports the
    90% CI plus the two one-sided p-values at a pre-specified +-2 pt
    margin. TOST rejects H0 (|Delta| >= margin) iff both one-sided
    p-values < 0.05, equivalently iff the 90% CI lies within +-2 pt.

T2  ShipsEar macro-F1 for every zero-shot run (review ask: accuracy alone
    is class-prior sensitive). Reads the per-clip predictions from the
    result JSONs, recomputes accuracy + macro-F1, cross-checks accuracy
    against the stored value.

Outputs: outputs/pilots/round2_analyses.json (print-only summary too).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

PREDS = Path("outputs/pilots/fusion_upgrade_preds")
RESULTS = Path("outputs/results")
N_BOOT = 10_000
SEED = 0
MARGIN_PT = 2.0


def clustered_diffs(path_a, path_b):
    a = json.loads(path_a.read_text())["preds"]
    b = json.loads(path_b.read_text())["preds"]
    keys = sorted(set(a) & set(b))
    assert len(keys) == 2685, len(keys)
    lab = np.array([a[k][0] for k in keys], dtype=np.int64)
    ca = np.array([a[k][1] for k in keys], dtype=np.int64) == lab
    cb = np.array([b[k][1] for k in keys], dtype=np.int64) == lab
    rids = np.array([k.rsplit("|", 1)[0] for k in keys])
    uniq, ridx = np.unique(rids, return_inverse=True)
    R = len(uniq)
    counts = np.bincount(ridx, minlength=R).astype(np.int64)
    rca = np.bincount(ridx, weights=ca.astype(np.float64), minlength=R)
    rcb = np.bincount(ridx, weights=cb.astype(np.float64), minlength=R)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, R, size=(N_BOOT, R))
    den = draws @ counts
    return (draws @ rca) / den - (draws @ rcb) / den


def main():
    out = {}

    # ---- T1: TOST ----
    d = clustered_diffs(PREDS / "A1.json", PREDS / "s1_beats.json") * 100  # pt
    ci90 = [float(np.percentile(d, 5)), float(np.percentile(d, 95))]
    p_ge = float((d >= MARGIN_PT).mean())   # H0: Delta >= +2
    p_le = float((d <= -MARGIN_PT).mean())  # H0: Delta <= -2
    tost_p = max(p_ge, p_le)
    out["tost_A1_vs_beats_single"] = {
        "margin_pt": MARGIN_PT, "alpha": 0.05, "n_boot": N_BOOT, "seed": SEED,
        "unit": "recording-clustered bootstrap (153 recordings)",
        "ci90_pt": ci90, "p_one_sided_ge": p_ge, "p_one_sided_le": p_le,
        "tost_p": tost_p,
        "verdict": "equivalent within margin" if tost_p < 0.05 else "not established",
    }
    print("T1  TOST  A1 vs BEATs single layer, margin +-%g pt" % MARGIN_PT)
    print("    90%% CI [%.2f, %.2f] pt   TOST p = %.4f -> %s"
          % (ci90[0], ci90[1], tost_p, out["tost_A1_vs_beats_single"]["verdict"]))

    # ---- T2: ShipsEar zero-shot macro-F1 ----
    rows = []
    for p in sorted(glob.glob(str(RESULTS / "*shipsear*zero_shot*.json"))):
        doc = json.loads(Path(p).read_text())
        preds = doc.get("predictions")
        if not preds:
            continue
        y = np.array([q["label_id"] for q in preds], dtype=np.int64)
        yp = np.array([q["pred_label_id"] for q in preds], dtype=np.int64)
        acc = float((yp == y).mean())
        f1 = doc["metrics_clip"]["f1_macro"]
        assert abs(acc - doc["metrics_clip"]["accuracy"]) < 1e-9, p
        rows.append({"file": Path(p).name, "n": len(y), "acc": round(acc, 4),
                     "f1_macro": round(f1, 4)})
    out["shipsear_zero_shot_macro_f1"] = rows
    accs = [r["acc"] for r in rows]; f1s = [r["f1_macro"] for r in rows]
    print("\nT2  ShipsEar zero-shot (%d runs)" % len(rows))
    print("    acc range      %.4f - %.4f" % (min(accs), max(accs)))
    print("    macro-F1 range %.4f - %.4f" % (min(f1s), max(f1s)))
    print("    chance acc 0.0833, chance macro-F1 (uniform prior) 0.0833")

    Path("outputs/pilots").mkdir(exist_ok=True)
    Path("outputs/pilots/round2_analyses.json").write_text(json.dumps(out, indent=1))
    print("\nwrote outputs/pilots/round2_analyses.json")


if __name__ == "__main__":
    main()
