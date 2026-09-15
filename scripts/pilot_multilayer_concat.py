#!/usr/bin/env python
"""PILOT: multi-layer concatenation probing on cached features (CPU only).

Question: how much does concatenating MULTIPLE layers (vs the single best
layer) buy, per source (omni_aud, q2a_aud) and for a 2-source audio fusion,
at the seeded 3k-train budget?

Protocol (selection discipline — no test peeking):
  * rows follow run_layer_probe.ordered_rows (test first, then train in
    SEED=42 shuffle); the 3k subset is the FIRST 3000 shuffled train rows,
    re-derived from the manifest and cross-checked against shard positions;
  * per-layer landscape: GroupKFold(3)-by-recording CV, fixed C=1.0 (same as
    run_selection_repair.fit_group_cv) — used ONLY to pick layers
    (best / top-3 / neighbours);
  * candidate configs (a)-(g) are SCORED by GroupKFold(3)-by-recording CV on
    the 3k train (C grid selected by that same CV; manual Parallel
    implementation of the GridSearchCV protocol, identical folds per C);
    the per-source winner is the CV-argmax config; each config then gets
    EXACTLY ONE test eval;
  * fusion: single-best-per-source concat vs top-2-per-source concat, both
    scored by the same GroupKFold CV, one test eval each;
  * the historically-reported layers (omni L31, q2a L16) are shown as a
    report-only arm (their lineage descends from full-train TEST selection).

Configs per source:
  a single CV-best layer | b CV-top-3 concat | c adjacent {b-1,b,b+1} concat
  d {L0, mid, best, last} concat | e ALL layers concat (C grid +1e-3)
  f single best + per-row L2 norm | g single best + elasticnet (saga, 0.5)

Threading: unpinned BLAS thrashes on the 128-visible-CPU container (28s vs
1.2s per fit) — all fits run under threadpoolctl limits, parallelism across
(fold, C) tasks via joblib processes.

Output: outputs/pilots/multilayer_concat_pilot.json   (NOT outputs/results)
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from threadpoolctl import threadpool_limits

FEATURES = Path("outputs/features/layerprobe")
OUT = Path("outputs/pilots/multilayer_concat_pilot.json")
SEED = 42           # same as run_layer_probe.ordered_rows
MAX_TRAIN = 3000    # pilot budget
N_FOLDS = 3
C_GRID = [0.01, 0.1, 1.0, 10.0]
C_GRID_ALL = [1e-3, 0.01, 0.1, 1.0]     # (e): extended at the small end (d/n huge)
C_GRID_ENET = [0.01, 0.1, 1.0]          # (g): saga is ~3 min/fit; C=10 useless at d~n

SOURCES = {
    "omni_aud": "outputs/features/layerprobe/qwen3_omni_30b_audio_*",
    "q2a_aud":  "outputs/features/layerprobe/qwen2_audio_audio_*",
}
HIST_LAYERS = {"omni_aud": 31, "q2a_aud": 16}   # full-train TEST-selected lineage


def find_manifest(name="deepship", clip_len=30.0, overlap=0.5) -> Path:
    """Local copy of splash.eval.harness.find_manifest (avoids its heavy import
    chain: librosa etc. are absent from the freshly-restored image)."""
    manifest_dir = Path("outputs/manifests")
    pattern = f"{name}_clip{float(clip_len):g}_ov{float(overlap):g}_*.csv"
    matches = sorted(m for m in manifest_dir.glob(pattern) if not m.name.endswith("_spec.csv"))
    if not matches:
        raise FileNotFoundError(f"No manifest matching {manifest_dir}/{pattern}")
    return matches[-1]


def ordered_rows(manifest: pd.DataFrame, max_train: int):
    """Same deterministic order as run_layer_probe.ordered_rows (SEED=42)."""
    test_idx = manifest.index[manifest["split"] == "test"].to_numpy()
    train_df = manifest[manifest["split"] == "train"]
    perm = np.random.default_rng(SEED).permutation(len(train_df))
    train_idx = train_df.index.to_numpy()[perm][:max_train]
    return list(test_idx) + list(train_idx)


def load_sweep(glob):
    d = sorted(Path(".").glob(glob))
    assert d, f"no dir for {glob}"
    Xs, ys, tes, rids, cids = [], [], [], [], []
    for p in sorted(d[0].glob("shard_*.npz")):
        z = np.load(p, allow_pickle=False)
        Xs.append(z["X"])                                   # keep fp16 master
        ys.append(z["y"]); tes.append(z["split"])
        rids.append(z["rid"].astype(str)); cids.append(z["cidx"].astype(int))
    return (np.concatenate(Xs), np.concatenate(ys).astype(int),
            np.concatenate(tes).astype(bool),
            np.concatenate(rids), np.concatenate(cids))


def _pipe(C, mode, max_iter):
    if mode == "l2":
        est = LogisticRegression(max_iter=max_iter, C=C)
        return make_pipeline(StandardScaler(), est)
    if mode == "l2norm":                                    # (f) unit-norm rows
        est = LogisticRegression(max_iter=max_iter, C=C)
        return make_pipeline(Normalizer(), StandardScaler(), est)
    if mode == "enet":                                      # (g) elastic-net
        est = LogisticRegression(penalty="elasticnet", solver="saga", C=C,
                                 l1_ratio=0.5, max_iter=max_iter, random_state=0)
        return make_pipeline(StandardScaler(), est)
    raise ValueError(mode)


def _fold_acc(X, y, tr, va, C, mode, max_iter, threads):
    """One (C, fold) fit. X/y are the SHARED full train matrix (joblib memmaps
    them once); fold slicing happens inside the worker — per-task big slices
    would each get their own /dev/shm memmap and blow the 16GB limit."""
    with warnings.catch_warnings(), threadpool_limits(threads):
        warnings.simplefilter("ignore")
        pipe = _pipe(C, mode, max_iter).fit(X[tr], y[tr])
        return float((pipe.predict(X[va]) == y[va]).mean())


def _layer_cv(X3k, y3k, splits, L, max_iter=2000, C=1.0):
    """3 fold fits for one layer (landscape); X3k is [n3k, n_layers, D] fp16."""
    accs = []
    for tr, va in splits:
        with warnings.catch_warnings(), threadpool_limits(1):
            warnings.simplefilter("ignore")
            pipe = _pipe(C, "l2", max_iter).fit(X3k[tr, L, :], y3k[tr])
            accs.append(float((pipe.predict(X3k[va, L, :]) == y3k[va]).mean()))
    return float(np.mean(accs))


def cv_eval(Xtr, ytr, splits, cgrid, mode, max_iter, n_jobs, threads):
    """Manual GridSearchCV: identical GroupKFold folds per C, parallel tasks.

    Returns (fitted_full_pipe, cv_table, best_C, seconds).
    """
    t0 = time.time()
    tasks = [(Xtr, ytr, tr, va, C, mode, max_iter, threads)
             for C in cgrid for tr, va in splits]
    with parallel_backend("loky", inner_max_num_threads=threads):
        accs = Parallel(n_jobs=n_jobs)(delayed(_fold_acc)(*t) for t in tasks)
    table = {C: float(np.mean(accs[i * len(splits):(i + 1) * len(splits)]))
             for i, C in enumerate(cgrid)}
    best_C = max(cgrid, key=lambda C: (table[C], -C))      # tie -> smaller C
    with warnings.catch_warnings(), threadpool_limits(max(threads, 4)):
        warnings.simplefilter("ignore")
        pipe = _pipe(best_C, mode, max_iter).fit(Xtr, ytr)  # single test-ready refit
    return pipe, table, best_C, time.time() - t0


def eval_on_test(pipe, Xte, yte):
    with threadpool_limits(4):
        pred = pipe.predict(Xte)
    return float((pred == yte).mean()), float(f1_score(yte, pred, average="macro"))


def concat_layers(X16, tr_idx, te_idx, layers):
    """fp32 concat of chosen layers for train/test rows (memory-frugal)."""
    Xtr = np.concatenate([X16[tr_idx, l, :].astype(np.float32) for l in layers], axis=1)
    Xte = np.concatenate([X16[te_idx, l, :].astype(np.float32) for l in layers], axis=1)
    return Xtr, Xte


def main():
    t_start = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fresh = {
        "meta": {
            "seed": SEED, "n_train_budget": MAX_TRAIN, "n_folds": N_FOLDS,
            "c_grid": C_GRID, "c_grid_all_layers": C_GRID_ALL,
            "c_grid_enet": C_GRID_ENET,
            "protocol": ("layers picked by GroupKFold(3)-by-recording CV at fixed "
                         "C=1.0 (landscape); configs scored by GroupKFold(3)-by-"
                         "recording CV over C grid on 3k train only; CV-argmax "
                         "config = selected; exactly ONE test eval per config"),
            "discipline": ("ALL selection (layers, configs, C, fusion arm) on "
                           "train CV only; test numbers for non-selected configs "
                           "are report-only and never used to choose"),
        },
        "sources": {}, "fusion": {}, "timings": {},
    }
    payload = fresh
    if OUT.exists():      # resume: skip sources already fully done
        try:
            payload = json.loads(OUT.read_text())
            payload.setdefault("fusion", {}); payload.setdefault("timings", {})
            payload["timings"]["resumed_from_partial"] = True
            print(f"[pilot] resuming from partial {OUT}", flush=True)
        except Exception:
            payload = fresh

    # ---- manifest / 3k subset keys (same as run_fusion_vs_beats_sig) --------
    m = find_manifest()
    manifest = pd.read_csv(m)
    order = ordered_rows(manifest, MAX_TRAIN)
    n_test = int((manifest["split"] == "test").sum())
    test_keys = list(zip(manifest.loc[order[:n_test], "recording_id"].astype(str),
                         manifest.loc[order[:n_test], "clip_idx"].astype(int)))
    train3k_keys = list(zip(manifest.loc[order[n_test:], "recording_id"].astype(str),
                            manifest.loc[order[n_test:], "clip_idx"].astype(int)))
    key_pos3k = {k: i for i, k in enumerate(train3k_keys)}
    payload["meta"].update(n_test=n_test, n_train3k=len(train3k_keys), manifest=m.name)
    print(f"[pilot] manifest={m.name} n_test={n_test} n_train3k={len(train3k_keys)}",
          flush=True)

    sweeps = {}
    for name, glob in SOURCES.items():
        X, y, te, rid, cid = load_sweep(glob)
        keys = list(zip(rid, cid))
        te_first = bool(te[:n_test].all() and not te[n_test:].any())
        pos3k_keys = [k for k, is_te in zip(keys, te) if not is_te][:MAX_TRAIN]
        pos_matches = set(pos3k_keys) == set(train3k_keys)
        print(f"[pilot] {name}: X={X.shape} {X.dtype} | test-first={te_first} "
              f"| positional-3k==manifest-3k: {pos_matches}", flush=True)
        sweeps[name] = {"X": X, "y": y, "te": te, "key": keys}
        payload["sources"].setdefault(name, {})["alignment"] = {
            "shape": list(X.shape), "test_rows_first": te_first,
            "positional_3k_matches_manifest": pos_matches}

    # ================= per-source stage =======================================
    for name in SOURCES:
        if "selected_config" in payload["sources"].get(name, {}):
            print(f"[pilot] {name}: already complete — skipping (resume)", flush=True)
            continue
        t_src = time.time()
        sw = sweeps[name]
        X, y, te = sw["X"], sw["y"], sw["te"]
        n_layers = X.shape[1]
        te_idx = np.where(te)[0]
        tr_idx = np.array([i for i, k in enumerate(sw["key"])
                           if k in key_pos3k])          # manifest 3k subset
        X3k = X[tr_idx]                                  # [3000, L, D] fp16 view
        y3k, yte = y[tr_idx], y[te_idx]
        groups = np.array([sw["key"][i][0] for i in tr_idx])
        gkf = GroupKFold(n_splits=N_FOLDS)
        splits = list(gkf.split(np.zeros(len(tr_idx)), y3k, groups=groups))
        rec = payload["sources"][name]
        rec.update(n_layers=n_layers, dim=int(X.shape[2]),
                   n_train_rows=int(len(tr_idx)), n_test_rows=int(len(te_idx)))

        # ---- 5) per-layer landscape (fixed C=1.0) ---------------------------
        if "landscape_cv_c1" in rec and len(rec["landscape_cv_c1"]) == n_layers:
            land = rec["landscape_cv_c1"]
            print(f"[pilot] {name}: landscape cached — reusing", flush=True)
        else:
            t0 = time.time()
            with parallel_backend("loky", inner_max_num_threads=1):
                land_accs = Parallel(n_jobs=16)(
                    delayed(_layer_cv)(X3k, y3k, splits, L) for L in range(n_layers))
            land = [{"layer": L, "cv_acc": a} for L, a in enumerate(land_accs)]
            rec["landscape_cv_c1"] = land
            rec["landscape_seconds"] = round(time.time() - t0, 1)
        cv_of = {d["layer"]: d["cv_acc"] for d in land}
        best_L = max(cv_of, key=lambda l: (cv_of[l], -l))
        order_by_cv = sorted(cv_of, key=lambda l: (-cv_of[l], l))
        top3 = order_by_cv[:3]
        mid_L = (n_layers - 1) // 2
        rec.update(cv_best_layer=best_L, cv_best_cvacc=cv_of[best_L],
                   cv_top3_layers=top3,
                   cv_gap_top2_pt=round(100 * (cv_of[order_by_cv[0]] - cv_of[order_by_cv[1]]), 2))
        print(f"[pilot] {name}: landscape best_cv=L{best_L} ({cv_of[best_L]:.4f}) "
              f"top3={top3}", flush=True)

        # ---- candidate configs ----------------------------------------------
        configs = {
            "a_single_best":   {"layers": [best_L], "mode": "l2", "cgrid": C_GRID,
                                "max_iter": 3000, "n_jobs": 12, "threads": 1},
            "b_top3_concat":   {"layers": top3,     "mode": "l2", "cgrid": C_GRID,
                                "max_iter": 3000, "n_jobs": 12, "threads": 1},
            "c_adjacent3":     {"layers": sorted({min(max(best_L + d, 0), n_layers - 1)
                                                 for d in (-1, 0, 1)}),
                                "mode": "l2", "cgrid": C_GRID,
                                "max_iter": 3000, "n_jobs": 12, "threads": 1},
            "d_anchor_concat": {"layers": sorted({0, mid_L, best_L, n_layers - 1}),
                                "mode": "l2", "cgrid": C_GRID,
                                "max_iter": 3000, "n_jobs": 12, "threads": 1},
            "e_all_layers":    {"layers": list(range(n_layers)), "mode": "l2",
                                "cgrid": C_GRID_ALL, "max_iter": 3000,
                                "n_jobs": 6, "threads": 4},
            "f_l2norm":        {"layers": [best_L], "mode": "l2norm", "cgrid": C_GRID,
                                "max_iter": 3000, "n_jobs": 12, "threads": 1},
            "g_elasticnet":    {"layers": [best_L], "mode": "enet", "cgrid": C_GRID_ENET,
                                "max_iter": 4000, "n_jobs": 9, "threads": 4},
        }
        rec.setdefault("configs", {})
        for cname, cfg in configs.items():
            if cname in rec["configs"]:
                print(f"[pilot]   {name} {cname:15s} cached — skipping (resume)",
                      flush=True)
                continue
            Xtr, Xev = concat_layers(X, tr_idx, te_idx, cfg["layers"])
            pipe, table, C, secs = cv_eval(Xtr, y3k, splits, cfg["cgrid"],
                                           cfg["mode"], cfg["max_iter"],
                                           cfg["n_jobs"], cfg["threads"])
            acc, f1m = eval_on_test(pipe, Xev, yte)
            lr = pipe.named_steps.get("logisticregression")
            rec["configs"][cname] = {
                "layers": cfg["layers"], "n_features": int(Xtr.shape[1]),
                "mode": cfg["mode"], "C": C, "cv_acc": table[C], "cv_table": table,
                "n_iter_final": int(np.max(lr.n_iter_)) if lr is not None else None,
                "test_acc": acc, "test_f1_macro": f1m,
                "fit_seconds": round(secs, 1), "selected": False}
            nlay = str(len(cfg["layers"])) + "L" if len(cfg["layers"]) > 5 else cfg["layers"]
            print(f"[pilot]   {name} {cname:15s} layers={nlay} cv={table[C]:.4f} "
                  f"C={C} test={acc:.4f} f1={f1m:.4f} [{secs:.0f}s]", flush=True)
            del Xtr, Xev, pipe
        for v in rec["configs"].values():
            v["selected"] = False
        sel = max(rec["configs"], key=lambda c: rec["configs"][c]["cv_acc"])
        rec["configs"][sel]["selected"] = True
        rec["selected_config"] = sel
        base = rec["configs"]["a_single_best"]["test_acc"]
        rec["test_gain_vs_single_best_pt"] = {
            c: round(100 * (v["test_acc"] - base), 2)
            for c, v in rec["configs"].items()}
        rec["source_seconds"] = round(time.time() - t_src, 1)
        print(f"[pilot] {name}: CV-selected config = {sel} "
              f"(test {rec['configs'][sel]['test_acc']:.4f} vs single-best "
              f"{base:.4f})", flush=True)
        OUT.write_text(json.dumps(payload, indent=1))
        del X3k

    # ================= fusion stage ===========================================
    t_f = time.time()
    common = set(sweeps["omni_aud"]["key"]) & set(sweeps["q2a_aud"]["key"])
    idx = {n: {k: i for i, k in enumerate(sw["key"])} for n, sw in sweeps.items()}
    te_keys = [k for k in test_keys if k in common]
    tr_keys = [k for k in train3k_keys if k in common]
    y_f = {part: np.array([sweeps["omni_aud"]["y"][idx["omni_aud"][k]] for k in ks])
           for part, ks in (("te", te_keys), ("tr", tr_keys))}
    groups_f = np.array([k[0] for k in tr_keys])
    gkf = GroupKFold(n_splits=N_FOLDS)
    splits_f = list(gkf.split(np.zeros(len(tr_keys)), y_f["tr"], groups=groups_f))
    payload["fusion"].update(n_train_rows=len(tr_keys), n_test_rows=len(te_keys))

    def blocks(source, layers, part_keys):
        sw = sweeps[source]
        return np.concatenate(
            [sw["X"][[idx[source][k] for k in part_keys], L, :].astype(np.float32)
             .reshape(len(part_keys), -1) for L in layers], axis=1)

    lay_omni = payload["sources"]["omni_aud"]["cv_best_layer"]
    lay_q2a = payload["sources"]["q2a_aud"]["cv_best_layer"]
    top2_omni = payload["sources"]["omni_aud"]["cv_top3_layers"][:2]
    top2_q2a = payload["sources"]["q2a_aud"]["cv_top3_layers"][:2]
    arms = {
        "single_best_concat": {"omni_aud": [lay_omni], "q2a_aud": [lay_q2a]},
        "top2_per_source_concat": {"omni_aud": top2_omni, "q2a_aud": top2_q2a},
        "historical_L31_L16": {"omni_aud": [HIST_LAYERS["omni_aud"]],
                               "q2a_aud": [HIST_LAYERS["q2a_aud"]]},
    }
    payload["fusion"]["arms"] = {}
    for aname, spec in arms.items():
        Xtr = np.concatenate([blocks(s, ls, tr_keys) for s, ls in spec.items()], axis=1)
        Xte = np.concatenate([blocks(s, ls, te_keys) for s, ls in spec.items()], axis=1)
        pipe, table, C, secs = cv_eval(Xtr, y_f["tr"], splits_f, C_GRID, "l2",
                                       3000, 12, 1)
        acc, f1m = eval_on_test(pipe, Xte, y_f["te"])
        payload["fusion"]["arms"][aname] = {
            "layers": spec, "C": C, "cv_acc": table[C], "cv_table": table,
            "test_acc": acc, "test_f1_macro": f1m,
            "fit_seconds": round(secs, 1),
            "report_only": aname == "historical_L31_L16",
            "note": ("layer lineage descends from full-train TEST-selected layers"
                     if aname == "historical_L31_L16" else
                     "layers chosen by 3k GroupKFold CV landscape")}
        print(f"[pilot] fusion {aname:22s} cv={table[C]:.4f} C={C} "
              f"test={acc:.4f} f1={f1m:.4f} [{secs:.0f}s]", flush=True)
        del Xtr, Xte, pipe
    sel_arm = max((a for a in payload["fusion"]["arms"]
                   if not payload["fusion"]["arms"][a]["report_only"]),
                  key=lambda a: payload["fusion"]["arms"][a]["cv_acc"])
    payload["fusion"]["selected_arm_by_cv"] = sel_arm
    payload["fusion"]["gain_top2_vs_single_pt"] = round(100 * (
        payload["fusion"]["arms"]["top2_per_source_concat"]["test_acc"]
        - payload["fusion"]["arms"]["single_best_concat"]["test_acc"]), 2)
    payload["timings"]["fusion_seconds"] = round(time.time() - t_f, 1)
    payload["timings"]["total_seconds"] = round(time.time() - t_start, 1)

    OUT.write_text(json.dumps(payload, indent=1))
    print(f"\n[pilot] total {payload['timings']['total_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
