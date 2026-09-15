#!/usr/bin/env python
"""Honest-selection gap fills for the SPLASH paper (CPU, container SPLASH).

GAP 1 -> outputs/pilots/demon_honest.json
  Honest single-layer numbers for the three DEMON sweeps that the fusion
  campaign never covered: qwen3_omni_30b_demon, qwen3_vl_8b_demon,
  qwen3_vl_32b_demon. Protocol is identical to fusion_upgrade stage 1:
  per-layer GroupKFold(3)-by-recording CV on the FULL 8,350-clip train
  (canonical ordered_rows SEED=42), per-layer C from {0.01,0.1,1,10};
  the test set is touched exactly ONCE per sweep at the CV-argmax layer
  (tie -> smaller layer / smaller C), refit on all train rows.
  Records layer, C, cv acc, test acc, test macro-F1.

GAP 2 -> outputs/pilots/honest_learning_curve.json
  Honest learning curve for omni audio (qwen3_omni_30b_audio) and BEATs
  (12-layer features, now full-budget under outputs/features/anchorlayer/
  beats_all_layers) at budgets [100,500,1000,2000,3000,8350]. At EACH
  budget the layer AND C are selected by the same GroupKFold(3)-by-recording
  train CV on the first-N rows of the canonical shuffled train order; test
  is evaluated exactly ONCE for that CV-argmax config.
  FLAGGED DIFFERENCE (also recorded in meta): outputs/results/layer_probe/
  learning_curves.json reported per-budget per-layer TEST accuracies (layers
  effectively picked on test) with one fixed C per model and train capped at
  3,000 rows; the present run never selects on test.

Discipline, CPU hygiene, loading and CV helpers are imported from
scripts/run_fusion_upgrade.py: same manifest, canonical row order,
GroupKFold(3) splits, C grid, tie-breaking, StandardScaler+LogisticRegression
pipeline, threadpoolctl + loky inner_max_num_threads pinning. Never writes
outputs/results and never calls splash.tracking save_result.

Resume-safe: incremental atomic JSON writes; complete units skipped.
Log: stdout is tee'd to outputs/pilots/demon_honest.log by the caller.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

import scripts.run_fusion_upgrade as rfu

T0 = time.time()

# ----------------------------------------------------------------- config ----
C_GRID = rfu.C_GRID                    # {0.01, 0.1, 1, 10}
N_FOLDS = rfu.N_FOLDS                  # 3
DEMON_SOURCES = {                      # sweep -> feature glob (layerprobe)
    "qwen3_omni_30b_demon": "outputs/features/layerprobe/qwen3_omni_30b_demon_*",
    "qwen3_vl_8b_demon":    "outputs/features/layerprobe/qwen3_vl_8b_demon_*",
    "qwen3_vl_32b_demon":   "outputs/features/layerprobe/qwen3_vl_32b_demon_*",
}
LC_SOURCES = ["omni_aud", "beats"]     # omni_aud = qwen3_omni_30b_audio
LC_BUDGETS = [100, 500, 1000, 2000, 3000, 8350]

PILOTS = Path("outputs/pilots")
DEMON_JSON = PILOTS / "demon_honest.json"
LC_JSON = PILOTS / "honest_learning_curve.json"
FUSION_JSON = PILOTS / "fusion_upgrade.json"   # cross-check refs only


def log(msg):
    print(f"[{time.time() - T0:9.1f}s] {msg}", flush=True)


def save_json(path, payload):
    PILOTS.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def load_payload(path, top_keys):
    payload = {k: {} for k in top_keys}
    if path.exists():
        try:
            saved = json.loads(path.read_text())
            for k in top_keys:
                saved.setdefault(k, {})
            payload = saved
            log(f"resuming from existing {path}")
        except Exception as e:  # corrupt -> fresh
            log(f"could not parse {path} ({e}) -> starting fresh")
    return payload


def load_lc_stack(name, canon):
    return (rfu.load_beats_stack(canon) if name == rfu.BEATS_NAME
            else rfu.load_llm_stack(name, canon))


def select_and_eval(Xc, canon, n_train, curve=None):
    """Shared protocol: per-layer C-tuned GroupKFold(3) landscape on the
    first n_train canonical train rows -> CV-argmax layer/C (tie -> smaller
    layer, smaller C) -> refit on all n_train rows -> ONE test eval.
    Returns (rec_fields, curve_dict)."""
    n_test = canon["n_test"]
    Xtr3d = Xc[n_test:n_test + n_train]
    ytr, groups = canon["y"][n_test:n_test + n_train], canon["rid"][n_test:n_test + n_train]
    layers = list(range(int(Xc.shape[1])))
    if curve is None:
        res, _splits, secs = rfu.run_landscape(Xtr3d, ytr, groups, layers)
        curve = res
        landscape_seconds = round(secs, 1)
    else:
        landscape_seconds = None
    scores = {L: max(curve[L].values()) for L in curve}
    best_L = max(scores, key=lambda L: (scores[L], -L))
    best_C = max(C_GRID, key=lambda C: (curve[best_L][f"{C:g}"], -C))
    with warnings.catch_warnings(), threadpool_limits(8):
        warnings.simplefilter("ignore")
        pipe = rfu._fit_pipe(best_C, rfu.MAX_ITER_ARM).fit(
            Xtr3d[:, best_L, :].astype(np.float32), ytr)
        y_pred = pipe.predict(Xc[:n_test, best_L, :].astype(np.float32))
    yte = canon["y"][:n_test]
    fields = {
        "best_layer": int(best_L), "best_C": float(best_C),
        "cv_acc": float(scores[best_L]),
        "test_acc": float((y_pred == yte).mean()),
        "test_f1_macro": float(f1_score(yte, y_pred, average="macro")),
        "landscape_seconds": landscape_seconds,
    }
    log(f"    CV-argmax L{best_L} (C={best_C:g}, cv={scores[best_L]:.4f})"
        f" -> test acc={fields['test_acc']:.4f} f1_macro={fields['test_f1_macro']:.4f}")
    return fields, curve


def curve_rows(curve):
    return [{"layer": L,
             **{f"{C:g}": curve[L][f"{C:g}"] for C in C_GRID},
             "best_C": max(C_GRID, key=lambda C: (curve[L][f"{C:g}"], -C)),
             "best_acc": max(curve[L][f"{C:g}"] for C in C_GRID)}
            for L in sorted(curve)]


def curve_from_rows(rows):
    ckeys = {f"{C:g}" for C in C_GRID}
    return {int(d["layer"]): {k: float(v) for k, v in d.items() if k in ckeys}
            for d in rows}


# ================================================================ GAP 1 ======
def gap1(payload, canon):
    log("=" * 70)
    log("GAP 1: honest single-layer DEMON sweeps (fusion_upgrade stage-1 protocol)")
    n_test = canon["n_test"]
    for sweep, pat in DEMON_SOURCES.items():
        rec = payload["demon"].setdefault(sweep, {})
        if rec.get("complete") and "test_acc" in rec:
            log(f"[gap1] {sweep}: complete -> skip (resume)")
            continue
        t_src = time.time()
        rfu.LLM_SOURCES[sweep] = pat
        Xc = rfu.load_llm_stack(sweep, canon)          # [n_rows, L, D] fp16
        n_layers_total = int(Xc.shape[1])
        cached = (curve_from_rows(rec["curve"])
                  if "curve" in rec and len(rec["curve"]) == n_layers_total else None)
        if cached is not None:
            log(f"[gap1] {sweep}: landscape cached ({n_layers_total} layers) -> reuse")
        else:
            log(f"[gap1] {sweep}: X{Xc.shape} {Xc.dtype} | {n_layers_total} layers "
                f"x {len(C_GRID)}C x {N_FOLDS} folds | train={canon['n_train']}")
        fields, curve = select_and_eval(Xc, canon, canon["n_train"], curve=cached)
        rec.update(feature_glob=pat, n_layers_total=n_layers_total, **fields,
                   curve=curve_rows(curve), complete=True,
                   source_seconds=round(time.time() - t_src, 1))
        save_json(DEMON_JSON, payload)
        del Xc


# ================================================================ GAP 2 ======
def gap2(payload, canon):
    log("=" * 70)
    log("GAP 2: honest learning curves (layer AND C CV-chosen per budget)")
    n_test = canon["n_test"]
    for name in LC_SOURCES:
        rec = payload.setdefault(name, {})
        if all(rec.get("budgets", {}).get(str(b), {}).get("complete")
               for b in LC_BUDGETS):
            log(f"[gap2] {name}: all budgets complete -> skip (resume)")
            continue
        t_src = time.time()
        Xc = load_lc_stack(name, canon)                # [n_rows, L, D] fp16
        rec["n_layers_total"] = int(Xc.shape[1])
        budgets = rec.setdefault("budgets", {})
        log(f"[gap2] {name}: X{Xc.shape} {Xc.dtype} | budgets {LC_BUDGETS} "
            f"| test={n_test}")
        for b in LC_BUDGETS:
            bud = budgets.setdefault(str(b), {})
            if bud.get("complete") and "test_acc" in bud:
                log(f"[gap2] {name} budget {b}: complete -> skip (resume)")
                continue
            log(f"[gap2] {name} budget {b}:")
            fields, curve = select_and_eval(Xc, canon, b)
            bud.update(n_train=int(b), **fields, curve=curve_rows(curve),
                       complete=True)
            save_json(LC_JSON, payload)
            if b == canon["n_train"] and FUSION_JSON.exists():  # cross-check ref
                s1 = json.loads(FUSION_JSON.read_text())["stage1"].get(name, {})
                if s1.get("complete"):
                    bud["fusion_stage1_crosscheck"] = {
                        k: s1.get(k) for k in
                        ("best_layer", "best_C", "best_cv_acc", "test_acc",
                         "test_f1_macro")}
                    log(f"[gap2] {name} budget {b} vs fusion stage1: "
                        f"L{fields['best_layer']} vs L{s1['best_layer']}, "
                        f"test {fields['test_acc']:.4f} vs {s1['test_acc']:.4f}")
        rec["source_seconds"] = round(time.time() - t_src, 1)
        save_json(LC_JSON, payload)
        del Xc


# ================================================================= main ======
def main():
    canon = rfu.canonical_setup()
    d_payload = load_payload(DEMON_JSON, ["meta", "demon"])
    d_payload["meta"].update({
        "campaign": "demon_honest",
        "seed": rfu.SEED,
        "manifest": canon["manifest"],
        "n_test": canon["n_test"], "n_train": canon["n_train"],
        "c_grid": C_GRID, "n_folds": N_FOLDS,
        "protocol": (
            "per-layer GroupKFold(3)-by-recording CV on the FULL train "
            "(canonical ordered_rows SEED=42: test rows first, then "
            "seeded-shuffled train), C from {0.01,0.1,1,10} per layer; test "
            "evaluated exactly ONCE per sweep at the CV-argmax layer (tie -> "
            "smaller layer / smaller C), refit on all train rows; macro-F1 "
            "via sklearn f1_score(average='macro')"),
        "borrowed_from": ("scripts/run_fusion_upgrade.py: canonical_setup, "
                          "load_llm_stack, run_landscape, _fit_pipe"),
        "sources": dict(DEMON_SOURCES),
        "started": time.strftime("%F %T"),
    })
    save_json(DEMON_JSON, d_payload)
    log(f"campaign start | train={canon['n_train']} test={canon['n_test']} "
        f"manifest={canon['manifest']}")
    log(f"artifacts: {DEMON_JSON} | {LC_JSON}")

    gap1(d_payload, canon)
    d_payload["finished_gap1"] = time.strftime("%F %T")
    save_json(DEMON_JSON, d_payload)
    del d_payload

    l_payload = load_payload(LC_JSON, ["meta", *LC_SOURCES])
    l_payload["meta"].update({
        "campaign": "honest_learning_curve",
        "seed": rfu.SEED,
        "manifest": canon["manifest"],
        "n_test": canon["n_test"], "n_train_total": canon["n_train"],
        "c_grid": C_GRID, "n_folds": N_FOLDS,
        "budgets": LC_BUDGETS,
        "protocol": (
            "at EACH budget: per-layer GroupKFold(3)-by-recording CV on the "
            "first-N rows of the canonical shuffled train order (ordered_rows "
            "SEED=42), C from {0.01,0.1,1,10} per layer; layer AND C chosen at "
            "the CV argmax (tie -> smaller layer / smaller C); model refit on "
            "the budget's train rows; test evaluated exactly ONCE per budget"),
        "flagged_difference_vs": {
            "path": "outputs/results/layer_probe/learning_curves.json",
            "how_it_differed": (
                "that file reports per-budget per-layer TEST accuracies, so "
                "the reported layer was effectively selected on test; it also "
                "used one fixed C per model (omni_aud C=0.1, beats C=1.0) and "
                "capped train at 3,000 rows ('all' budget). The present file "
                "selects layer AND C on train-only CV per budget and touches "
                "test exactly once per budget, up to the full 8,350 train."),
            "old_budgets": [100, 500, 1000, 2000, "all=3000"],
        },
        "sources": {"omni_aud": rfu.LLM_SOURCES["omni_aud"],
                    "beats": str(rfu.BEATS_DIR)},
        "borrowed_from": ("scripts/run_fusion_upgrade.py: canonical_setup, "
                          "load_llm_stack / load_beats_stack, run_landscape, "
                          "_fit_pipe"),
        "started": time.strftime("%F %T"),
    })
    save_json(LC_JSON, l_payload)
    gap2(l_payload, canon)
    l_payload["finished"] = time.strftime("%F %T")
    l_payload["wall_total_seconds"] = round(time.time() - T0, 1)
    save_json(LC_JSON, l_payload)
    log(f"DONE in {l_payload['wall_total_seconds']}s -> {DEMON_JSON}, {LC_JSON}")


if __name__ == "__main__":
    main()
