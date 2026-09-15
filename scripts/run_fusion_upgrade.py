#!/usr/bin/env python
"""FUSION UPGRADE campaign: maximize the HONEST 7-source LLM fusion
(DeepShip 4-class, frozen representations, full 8,350-clip train budget)
and test via McNemar + paired bootstrap whether it can beat the
same-treatment BEATs arms.

DISCIPLINE (non-negotiable): every layer / subset / C / arm choice is made on
TRAIN-ONLY GroupKFold-by-recording CV. The test set is touched exactly ONCE per
arm's final config. The only test-lineage numbers are the explicitly flagged
R0 (historical test-selected layers, report-only) and the recorded
"expectation" sanity references from earlier pilots; they are never fed back
into any choice.

Stages
  1  per-source per-layer C-tuned landscapes (GroupKFold(3)-by-recording on
     train; C from {0.01,0.1,1,10} per layer; full curve recorded). One test
     eval per source at the CV-argmax layer (honest single-layer numbers).
  2  fusion arms, selection by repeated GroupKFold(3) x 3 on train:
       R0  historical single-layer concat (report-only, test-lineage layers)
       A1  per-source CV-argmax single layers, concat
       A2  per-source greedy-SPREAD top-3 layers (>=3-layer distance from all
           already-picked), concat (~80k dims -> C grid pruned to {0.1,1})
       A3  per-source anchors {L0, CV-best, last}, concat
       A4  SUPERB-style learnable per-layer softmax weights per source,
           weighted-summed to D dims per source, concat (~26.6k dims), torch:
           layer weights + torch linear head trained JOINTLY (head choice),
           Adam lr 1e-2 wd 1e-4, batch 256, early-stop (patience 8) on a
           held-out group half of the sub-train, retrain at best epoch.
  3  BEATs same-treatment arms:
       B1  single CV-best layer (= stage-1 beats eval; expectation 0.7393)
       B2  greedy-spread top-3 concat
       B2t plain top-3 by groupCV concat (the {9,6,5} set behind the
           expectation 0.7665; added because strict spread cannot pick {9,6,5})
       B3  weighted-sum over the 12 layers (torch, same protocol as A4)
  4  significance: McNemar exact + 10,000 paired bootstrap (seed 0) for all
     pairs of saved arm predictions; primary pair = best honest LLM arm
     (by train CV) vs best honest BEATs arm (by train CV).

CPU hygiene (128 visible CPUs, 16GB /dev/shm): every fit runs under
threadpool_limits; parallelism across (C, fold) tasks via joblib loky with
inner_max_num_threads; ONE feature matrix per source/arm is passed to the
Parallel call so joblib memmaps it once and all workers share it; fold
slicing happens inside workers.

Resume-safe: incremental JSON after every unit; completed units are skipped
on restart. Done marker written ONLY after stage 4 completes.

Outputs (never outputs/results, never splash.tracking.results.save_result):
  outputs/pilots/fusion_upgrade.json
  outputs/pilots/fusion_upgrade_preds/<arm>.json
  outputs/pilots/fusion_upgrade.done
Smoke mode (FUSION_SMOKE=1): 2 sources x layer-subset x 500 train rows,
separate file names (fusion_upgrade_smoke*), R0 skipped.
"""
from __future__ import annotations

import itertools
import json
import os
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
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

T0 = time.time()

# ----------------------------------------------------------------- config ----
SEED = 42
N_FOLDS = 3
N_REPEATS = 3                      # arm CV = GroupKFold(3) x 3 repeats
REPEAT_SEEDS = [None, 1001, 1002]  # repeat 0 = canonical unshuffled GroupKFold
C_GRID = [0.01, 0.1, 1.0, 10.0]
C_GRID_BIG = [0.1, 1.0]            # arms with > BIG_DIM_THRESHOLD features
BIG_DIM_THRESHOLD = 40_000
MAX_ITER_LANDSCAPE = 2000
MAX_ITER_ARM = 3000

SMOKE = os.environ.get("FUSION_SMOKE", "0") == "1"
_TRAIN_BUDGET = int(os.environ.get("FUSION_TRAIN_BUDGET", "500" if SMOKE else "0"))
TRAIN_BUDGET = _TRAIN_BUDGET if _TRAIN_BUDGET > 0 else None   # None = all 8,350

LLM_SOURCES = {                     # the historical 7-source core set
    "vl8b_mel":  "outputs/features/layerprobe/qwen3_vl_8b_mel_*",
    "omni_aud":  "outputs/features/layerprobe/qwen3_omni_30b_audio_*",
    "vl32b_mel": "outputs/features/layerprobe/qwen3_vl_32b_mel_*",
    "omni_mel":  "outputs/features/layerprobe/qwen3_omni_30b_mel_*",
    "q2a_aud":   "outputs/features/layerprobe/qwen2_audio_audio_*",
    "vl8b_stft": "outputs/features/layerprobe/qwen3_vl_8b_stft_*",
    "vl32b_stft": "outputs/features/layerprobe/qwen3_vl_32b_stft_*",
}
ARM_ORDER = list(LLM_SOURCES)       # fixed concat order across arms
HIST_LAYERS = {                     # R0 reference (test-lineage, report-only)
    "vl8b_mel": 3, "omni_aud": 31, "vl32b_mel": 3, "omni_mel": 2,
    "q2a_aud": 16, "vl8b_stft": 4, "vl32b_stft": 3,
}
SMOKE_SOURCES = ["q2a_aud", "omni_aud"]
BEATS_DIR = Path("outputs/features/anchorlayer/beats_all_layers")
BEATS_NAME = "beats"
MANIFEST_CANDIDATES = sorted(
    m for m in Path("outputs/manifests").glob("deepship_clip30_ov0.5_*.csv")
    if not m.name.endswith("_spec.csv"))
MANIFEST = MANIFEST_CANDIDATES[-1]           # same convention as the pilots

PILOTS = Path("outputs/pilots")
TAG = "fusion_upgrade_smoke" if SMOKE else "fusion_upgrade"
OUT_JSON = PILOTS / f"{TAG}.json"
PREDS_DIR = PILOTS / f"{TAG}_preds"
DONE_MARKER = PILOTS / f"{TAG}.done"

# A4 / B3 torch hyperparameters (fixed a priori, no tuning)
A4_LR = 1e-2
A4_WD = 1e-4
A4_BATCH = 256
A4_MAX_EPOCHS = 80
A4_PATIENCE = 8
A4_MAX_EPOCHS_SMOKE = 40

EXPECTATIONS = {                    # test-lineage references from prior pilots
    "B1": {"test_acc": 0.7393, "note": "pilot_beats_full stage2b: groupCV-argmax L9, full budget"},
    "B2t": {"test_acc": 0.7665, "note": "pilot_beats_full stage3 top3_by_groupCV {9,6,5}, full budget"},
    "R0": {"test_acc": 0.741, "note": "historical 7-source single-layer-concat fusion (test-selected lineage)"},
}


def log(msg):
    print(f"[{time.time() - T0:9.1f}s] {msg}", flush=True)


def save(payload):
    PILOTS.mkdir(parents=True, exist_ok=True)
    tmp = OUT_JSON.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, OUT_JSON)


def load_payload():
    payload = {"meta": {}, "stage1": {}, "stage2": {}, "stage3": {}, "stage4": {}}
    if OUT_JSON.exists():
        try:
            saved = json.loads(OUT_JSON.read_text())
            for k in payload:
                saved.setdefault(k, {})
            payload = saved
            log(f"resuming from existing {OUT_JSON}")
        except Exception as e:  # corrupt -> fresh
            log(f"could not parse existing {OUT_JSON} ({e}) -> starting fresh")
    return payload


# ------------------------------------------------------------- row order -----
def canonical_setup():
    """Canonical ordered_rows(SEED=42): all test rows (manifest order) first,
    then seeded-shuffled train rows truncated to the budget."""
    manifest = pd.read_csv(MANIFEST)
    test_idx = manifest.index[manifest["split"] == "test"].to_numpy()
    train_df = manifest[manifest["split"] == "train"]
    perm = np.random.default_rng(SEED).permutation(len(train_df))
    train_idx = train_df.index.to_numpy()[perm]
    if TRAIN_BUDGET is not None:
        train_idx = train_idx[:TRAIN_BUDGET]
    order = np.concatenate([test_idx, train_idx])
    rows = manifest.iloc[order]
    keys = list(zip(rows["recording_id"].astype(str), rows["clip_idx"].astype(int)))
    te = (rows["split"] == "test").to_numpy(bool)
    n_test = int(te.sum())
    assert te[:n_test].all() and not te[n_test:].any()
    return {
        "manifest": MANIFEST.name,
        "keys": keys,
        "key_pos": {k: i for i, k in enumerate(keys)},
        "y": rows["label_id"].to_numpy(np.int64),
        "te": te,
        "rid": rows["recording_id"].to_numpy().astype(str),
        "n_test": n_test,
        "n_train": len(keys) - n_test,
    }


# --------------------------------------------------------- stack loading -----
def _shard_stack(npz_paths):
    Xs, ys, tes, rids, cids = [], [], [], [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=False)
        Xs.append(z["X"])                                  # fp16 master
        ys.append(z["y"])
        tes.append(z["split"])
        if "rid" in z:
            rids.append(z["rid"].astype(str))
            cids.append(z["cidx"].astype(int))
    X = np.concatenate(Xs)
    y = np.concatenate(ys).astype(int)
    te = np.concatenate(tes).astype(bool)
    rid = np.concatenate(rids) if rids else None
    return X, y, te, rid, (np.concatenate(cids) if cids else None)


def load_llm_stack(name, canon):
    """[n_rows, L, D] fp16 in CANONICAL order, verified against the manifest."""
    d = sorted(Path(".").glob(LLM_SOURCES[name]))
    assert d, f"no feature dir for {name} ({LLM_SOURCES[name]})"
    shards = sorted(d[0].glob("shard_*.npz"))
    assert shards, f"no shards under {d[0]}"
    X, y, te, rid, cid = _shard_stack(shards)
    keys = list(zip(rid, cid))
    pos = {k: i for i, k in enumerate(keys)}
    missing = [k for k in canon["keys"] if k not in pos]
    assert not missing, f"{name}: {len(missing)} canonical rows missing from shards"
    take = np.array([pos[k] for k in canon["keys"]], dtype=np.int64)
    Xc = np.ascontiguousarray(X[take])                     # [n_rows, L, D] fp16
    assert np.array_equal(y[take], canon["y"]), f"{name}: y mismatch vs manifest"
    assert np.array_equal(te[take], canon["te"]), f"{name}: split mismatch vs manifest"
    del X
    return Xc


def load_beats_stack(canon):
    """shard_all (test + first 3k train, canonical) + shard_ext_* (rest, in
    canonical order) -> [n_rows, 12, 768] fp16; verified like pilot stage0."""
    p_all = BEATS_DIR / "shard_all.npz"
    ext = sorted(BEATS_DIR.glob("shard_ext_*.npz"))
    assert p_all.exists(), f"missing {p_all}"
    z0 = np.load(p_all, allow_pickle=False)
    X0, y0, te0 = z0["X"], z0["y"].astype(int), z0["split"].astype(bool)
    n0 = len(y0)
    n_need = len(canon["y"])
    assert n_need <= n0 + sum(len(np.load(p, allow_pickle=False)["y"]) for p in ext), \
        "BEATs shards do not cover the requested rows"
    if n_need <= n0:
        Xc = np.ascontiguousarray(X0[:n_need])
        yc, tec = y0[:n_need], te0[:n_need]
    else:
        Xe, ye, tee, ride, cide = _shard_stack(ext)
        need_ext = n_need - n0
        Xc = np.ascontiguousarray(np.concatenate([X0, Xe[:need_ext]]))
        yc = np.concatenate([y0, ye[:need_ext]])
        tec = np.concatenate([te0, tee[:need_ext]])
        ext_keys = list(zip(ride[:need_ext], cide[:need_ext]))
        assert ext_keys == canon["keys"][n0:n_need], "BEATs ext row order mismatch"
    assert np.array_equal(yc, canon["y"]), "BEATs y mismatch vs manifest"
    assert np.array_equal(tec, canon["te"]), "BEATs split mismatch vs manifest"
    return Xc


def load_stack(name, canon):
    return load_beats_stack(canon) if name == BEATS_NAME else load_llm_stack(name, canon)


# ------------------------------------------------------------- CV helpers ----
def _fit_pipe(C, max_iter):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=C))


def _landscape_layer(X3d, y, splits, L, threads):
    """One layer: 3 folds x full C grid (12 fits), sequential in one worker."""
    out = {}
    with warnings.catch_warnings(), threadpool_limits(threads):
        warnings.simplefilter("ignore")
        for C in C_GRID:
            accs = []
            for tr, va in splits:
                pipe = _fit_pipe(C, MAX_ITER_LANDSCAPE)
                pipe.fit(X3d[tr, L, :].astype(np.float32), y[tr])
                accs.append(float((pipe.predict(X3d[va, L, :].astype(np.float32)) == y[va]).mean()))
            out[f"{C:g}"] = float(np.mean(accs))
    return out


def run_landscape(Xtr3d, ytr, groups, layers, n_jobs=32):
    """Per-layer C-tuned GroupKFold(3) landscape; ONE Parallel call so joblib
    memmaps the shared X3d once; results logged as they arrive."""
    splits = list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(ytr)), ytr, groups))
    res, t0 = {}, time.time()
    with parallel_backend("loky", inner_max_num_threads=1):
        gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_landscape_layer)(Xtr3d, ytr, splits, L, 1) for L in layers)
        for L, out in zip(layers, gen):
            res[L] = out
            bc = max(C_GRID, key=lambda C: (out[f"{C:g}"], -C))
            log(f"    layer {L:2d}: " +
                " ".join(f"{C:g}={out[f'{C:g}']:.4f}" for C in C_GRID) +
                f" | best C={bc:g} acc={out[f'{bc:g}']:.4f}")
    return res, splits, time.time() - t0


def repeated_splits(y, groups):
    """3 folds x 3 repeats; repeat 0 = canonical unshuffled GroupKFold(3)."""
    out = []
    for r in range(N_REPEATS):
        gkf = GroupKFold(n_splits=N_FOLDS, shuffle=(r > 0),
                         random_state=REPEAT_SEEDS[r])
        out.extend(gkf.split(np.zeros(len(y)), y, groups))
    return out


def _fold_acc(X, y, tr, va, C, max_iter, threads):
    """One (C, fold) arm fit; X is the SHARED matrix (joblib memmaps it once),
    fold slicing happens inside the worker."""
    with warnings.catch_warnings(), threadpool_limits(threads):
        warnings.simplefilter("ignore")
        pipe = _fit_pipe(C, max_iter)
        pipe.fit(X[tr], y[tr])
        return float((pipe.predict(X[va]) == y[va]).mean())


def arm_cv(Xtr, ytr, groups, cgrid, max_iter=MAX_ITER_ARM, n_jobs=16, threads=1):
    """Manual GridSearchCV over repeated group folds: identical splits per C,
    parallel across (C, fold, repeat) tasks. Returns test-ready refit."""
    t0 = time.time()
    splits = repeated_splits(ytr, groups)
    tasks = [(C, tr, va) for C in cgrid for tr, va in splits]
    with parallel_backend("loky", inner_max_num_threads=threads):
        accs = Parallel(n_jobs=n_jobs)(
            delayed(_fold_acc)(Xtr, ytr, tr, va, C, max_iter, threads)
            for (C, tr, va) in tasks)
    table = {f"{C:g}": float(np.mean(accs[i * len(splits):(i + 1) * len(splits)]))
             for i, C in enumerate(cgrid)}
    best_C = max(cgrid, key=lambda C: (table[f"{C:g}"], -C))       # tie -> smaller C
    with warnings.catch_warnings(), threadpool_limits(8):
        warnings.simplefilter("ignore")
        pipe = _fit_pipe(best_C, max_iter).fit(Xtr, ytr)
    return pipe, table, best_C, time.time() - t0, len(splits)


# ------------------------------------------------------------ predictions ----
def save_arm_preds(arm, y_pred, yte, test_keys, report_only=False, extra=None):
    acc = float((np.asarray(y_pred) == yte).mean())
    f1 = float(f1_score(yte, y_pred, average="macro"))
    doc = {"arm": arm, "report_only": bool(report_only), "acc": acc,
           "f1_macro": f1,
           "preds": {f"{rid}|{cid}": [int(t), int(p)]
                     for (rid, cid), t, p in zip(test_keys, yte, y_pred)}}
    if extra:
        doc.update(extra)
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PREDS_DIR / f".{arm}.tmp"
    tmp.write_text(json.dumps(doc))
    os.replace(tmp, PREDS_DIR / f"{arm}.json")
    return acc, f1


def preds_file(arm):
    return PREDS_DIR / f"{arm}.json"


def arm_done(payload, stage_key, arm):
    rec = payload[stage_key].get(arm, {})
    return ("test_acc" in rec) and preds_file(arm).exists()


# ------------------------------------------------------- layer-set helpers ---
def stage1_layers(n_layers, is_beats=False):
    if not SMOKE:
        return list(range(n_layers))
    stride = 3 if is_beats else 8
    ls = list(range(0, n_layers, stride))
    if n_layers - 1 not in ls:
        ls.append(n_layers - 1)
    return sorted(set(ls))


def curve_scores(rec):
    return {int(d["layer"]): float(d["best_acc"]) for d in rec.get("curve", [])}


def spread_layers(scores, min_dist=3, k=3):
    """Greedy spread: CV-best first, then next-best at >= min_dist from ALL
    picked layers (adjacent layers are redundant per the pilot)."""
    order = sorted(scores, key=lambda L: (-scores[L], L))
    picked = []
    for L in order:
        if all(abs(L - p) >= min_dist for p in picked):
            picked.append(L)
            if len(picked) == k:
                break
    return sorted(picked)


def topk_layers(scores, k=3):
    return sorted(sorted(scores, key=lambda L: (-scores[L], L))[:k])


def anchor_layers(scores, layers_scored):
    best = max(scores, key=lambda L: (scores[L], -L))
    return sorted({min(layers_scored), best, max(layers_scored)})


# ============================================================== STAGE 1 ======
def stage1(payload, canon):
    log("=" * 70)
    log("STAGE 1: per-source per-layer C-tuned GroupKFold(3) landscapes")
    names = ([BEATS_NAME] + SMOKE_SOURCES) if SMOKE else ([BEATS_NAME] + ARM_ORDER)
    for name in names:
        rec = payload["stage1"].setdefault(name, {})
        if rec.get("complete"):
            log(f"[stage1] {name}: complete -> skip (resume)")
            continue
        t_src = time.time()
        Xc = load_stack(name, canon)                    # [n_rows, L, D] fp16
        n_test = canon["n_test"]
        Xtr3d = Xc[n_test:]
        ytr, groups = canon["y"][n_test:], canon["rid"][n_test:]
        n_layers_total = int(Xc.shape[1])
        layers = stage1_layers(n_layers_total, is_beats=(name == BEATS_NAME))
        if "curve" in rec and len(rec["curve"]) == len(layers):
            ckeys = {f"{C:g}" for C in C_GRID}
            curve = {int(d["layer"]): {k: float(v) for k, v in d.items() if k in ckeys}
                     for d in rec["curve"]}
            log(f"[stage1] {name}: landscape cached ({len(layers)} layers) -> reuse")
        else:
            log(f"[stage1] {name}: X{Xc.shape} {Xc.dtype} | {len(layers)} layers "
                f"x {len(C_GRID)}C x {N_FOLDS} folds | train={len(ytr)}")
            res, _splits, secs = run_landscape(Xtr3d, ytr, groups, layers)
            curve = res
            rec["curve"] = [{"layer": L,
                             **{f"{C:g}": res[L][f"{C:g}"] for C in C_GRID},
                             "best_C": max(C_GRID, key=lambda C: (res[L][f"{C:g}"], -C)),
                             "best_acc": max(res[L][f"{C:g}"] for C in C_GRID)}
                            for L in layers]
            rec["landscape_seconds"] = round(secs, 1)
            save(payload)
        scores = {L: max(curve[L].values()) for L in curve}
        best_L = max(scores, key=lambda L: (scores[L], -L))
        best_C = max(C_GRID, key=lambda C: (curve[best_L][f"{C:g}"], -C))
        rec.update(n_layers_total=n_layers_total,
                   layers_scored=layers,
                   best_layer=int(best_L), best_C=float(best_C),
                   best_cv_acc=float(scores[best_L]))
        # --- single honest test eval at the CV-argmax layer (exactly once) ---
        with warnings.catch_warnings(), threadpool_limits(8):
            warnings.simplefilter("ignore")
            pipe = _fit_pipe(best_C, MAX_ITER_ARM).fit(
                Xtr3d[:, best_L, :].astype(np.float32), ytr)
        y_pred = pipe.predict(Xc[:n_test, best_L, :].astype(np.float32))
        acc, f1 = save_arm_preds(
            f"s1_{name}", y_pred, canon["y"][:n_test], canon["keys"][:n_test],
            extra={"kind": "per-source single-layer honest eval (stage 1)",
                   "layer": int(best_L), "C": float(best_C),
                   "cv_acc": float(scores[best_L])})
        rec.update(test_acc=acc, test_f1_macro=f1, complete=True,
                   source_seconds=round(time.time() - t_src, 1))
        exp = EXPECTATIONS.get("B1") if name == BEATS_NAME else None
        log(f"[stage1] {name}: CV-argmax L{best_L} (C={best_C:g}, cv={scores[best_L]:.4f})"
            f" -> test acc={acc:.4f} f1={f1:.4f}"
            + (f"  [expectation {exp['test_acc']}]" if exp else ""))
        save(payload)
        del Xc, Xtr3d


# ============================================================== STAGE 2 ======
def fusion_sources():
    return SMOKE_SOURCES if SMOKE else ARM_ORDER


def build_concat_matrices(needed, canon):
    """One pass over sources; gathers the union of layers needed by all
    active concat arms into per-arm train/test fp32 matrices."""
    n_test = canon["n_test"]
    blocks = {arm: {"tr": [], "te": []} for arm in needed}
    for src in fusion_sources():
        if not any(src in spec for spec in needed.values()):
            continue
        Xc = load_stack(src, canon)
        for arm, spec in needed.items():
            for L in spec.get(src, []):
                blocks[arm]["tr"].append(Xc[n_test:, L, :].astype(np.float32))
                blocks[arm]["te"].append(Xc[:n_test, L, :].astype(np.float32))
        del Xc
    mats = {}
    for arm, b in blocks.items():
        mats[arm] = (np.concatenate(b["tr"], axis=1), np.concatenate(b["te"], axis=1))
    return mats


def run_concat_arm(arm, layer_spec, payload, stage_key, mats, canon, report_only=False):
    Xtr, Xte = mats[arm]
    ytr, yte = canon["y"][canon["n_test"]:], canon["y"][:canon["n_test"]]
    groups = canon["rid"][canon["n_test"]:]
    d = Xtr.shape[1]
    cgrid = C_GRID_BIG if d > BIG_DIM_THRESHOLD else C_GRID
    n_jobs, threads = (12, 2) if d > BIG_DIM_THRESHOLD else (24, 1)
    pipe, table, best_C, secs, n_splits = arm_cv(Xtr, ytr, groups, cgrid,
                                                 n_jobs=n_jobs, threads=threads)
    with warnings.catch_warnings(), threadpool_limits(8):
        warnings.simplefilter("ignore")
        y_pred = pipe.predict(Xte)
    acc, f1 = save_arm_preds(arm, y_pred, yte, canon["keys"][:canon["n_test"]],
                             report_only=report_only)
    rec = {"layer_spec": {s: list(ls) for s, ls in layer_spec.items()},
           "n_features": int(d), "C": float(best_C), "cv_table": table,
           "cv_acc": float(table[f"{best_C:g}"]), "cv_splits": int(n_splits),
           "test_acc": acc, "test_f1_macro": f1,
           "report_only": bool(report_only),
           "fit_seconds": round(secs, 1)}
    exp = EXPECTATIONS.get(arm)
    if exp:
        rec["expectation_test_lineage"] = exp
    payload[stage_key][arm] = rec
    log(f"[{stage_key}] {arm}: d={d} cv={rec['cv_acc']:.4f} C={best_C:g} "
        f"-> test acc={acc:.4f} f1={f1:.4f} [{secs:.0f}s]"
        + (" (report-only, test-lineage layers)" if report_only else ""))
    save(payload)


def stage2_concat_arms(payload, canon):
    s1 = payload["stage1"]
    srcs = fusion_sources()
    best = {n: s1[n]["best_layer"] for n in srcs}
    scores = {n: curve_scores(s1[n]) for n in srcs}
    layers_scored = {n: s1[n]["layers_scored"] for n in srcs}
    needed = {}
    specs = {
        "A1": {n: [best[n]] for n in srcs},
        "A2": {n: spread_layers(scores[n]) for n in srcs},
        "A3": {n: anchor_layers(scores[n], layers_scored[n]) for n in srcs},
    }
    for arm, spec in specs.items():
        if arm_done(payload, "stage2", arm):
            log(f"[stage2] {arm}: complete -> skip (resume)")
        else:
            needed[arm] = spec
    if SMOKE:
        payload["stage2"]["R0"] = {"skipped": "smoke mode (needs all 7 sources)"}
    elif not arm_done(payload, "stage2", "R0"):
        needed["R0"] = {n: [HIST_LAYERS[n]] for n in srcs}
    else:
        log("[stage2] R0: complete -> skip (resume)")
    if not needed:
        return
    log(f"[stage2] building concat matrices for arms: {sorted(needed)}")
    t0 = time.time()
    mats = build_concat_matrices(needed, canon)
    log(f"[stage2] matrices built in {time.time() - t0:.0f}s: "
        + ", ".join(f"{a}={mats[a][0].shape[1]}d" for a in mats))
    for arm in ["A1", "A2", "A3", "R0"]:
        if arm in needed:
            run_concat_arm(arm, needed[arm], payload, "stage2", mats, canon,
                           report_only=(arm == "R0"))


# --------------------------------------------------- A4/B3: torch weighted ---
def _pick_torch_device():
    import torch
    pref = os.environ.get("FUSION_DEVICE", "cuda:1")
    if not torch.cuda.is_available() or pref == "cpu":
        return torch.device("cpu")
    idx = 0
    if pref.startswith("cuda"):
        idx = int(pref.split(":")[1]) if ":" in pref else 0
        if idx >= torch.cuda.device_count():
            idx = 0
    return torch.device(f"cuda:{idx}")


def weighted_arm(arm, sources, payload, stage_key, canon):
    """SUPERB-style learnable per-layer softmax weights per source ->
    weighted sum to D dims per source -> concat -> torch linear head, trained
    jointly (Adam lr 1e-2 wd 1e-4, batch 256). Honest nested CV: outer
    GroupKFold(3)x3 eval fold; sub-train split by GroupKFold(2) into fit/stop
    halves; early stop on the stop half; retrain on the full sub-train at the
    best epoch; predict the eval fold."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = _pick_torch_device()

    def load_stacks(dev):
        st, mt = [], []
        for name in sources:
            Xc = load_stack(name, canon)
            mt.append((name, int(Xc.shape[1]), int(Xc.shape[2])))
            st.append(torch.from_numpy(Xc).to(dev))          # fp16 resident
            del Xc
        return st, mt

    try:
        stacks, meta = load_stacks(device)
    except RuntimeError as e:                                # GPU OOM -> CPU
        log(f"[{stage_key}] {arm}: GPU residency failed ({e}) -> falling back to CPU")
        device = torch.device("cpu")
        stacks, meta = load_stacks(device)
    if device.type == "cpu":
        torch.set_num_threads(min(32, os.cpu_count() or 8))
    max_epochs = A4_MAX_EPOCHS_SMOKE if SMOKE else A4_MAX_EPOCHS
    n_repeats = 1 if device.type == "cpu" else N_REPEATS     # CPU time bound
    n_test = canon["n_test"]
    y_all = canon["y"]
    y_dev = torch.as_tensor(y_all, dtype=torch.long, device=device)
    tr_rows = np.arange(n_test, len(y_all))
    te_rows = np.arange(n_test)
    groups = canon["rid"][n_test:]
    log(f"[{stage_key}] {arm}: device={device} sources={len(stacks)} "
        + " ".join(f"{n}(L{l}xD{d})" for n, l, d in meta))

    class WSProbe(nn.Module):
        def __init__(self, layer_counts, dims, n_classes):
            super().__init__()
            self.w = nn.ParameterList([nn.Parameter(torch.zeros(L))
                                       for L in layer_counts])
            self.head = nn.Linear(sum(dims), n_classes)

        def forward(self, feats):
            reps = [torch.einsum("bld,l->bd", f, torch.softmax(w, dim=0))
                    for f, w in zip(feats, self.w)]
            return self.head(torch.cat(reps, dim=1))

    means, stds = [None] * len(stacks), [None] * len(stacks)  # set per phase

    def set_stats(rows_np):
        nonlocal means, stds
        stats = []
        for s in stacks:
            L, D = s.shape[1], s.shape[2]
            sm = torch.zeros(L, D, dtype=torch.float32, device=s.device)
            sm2 = torch.zeros_like(sm)
            n = 0
            for i in range(0, len(rows_np), 512):
                r = torch.as_tensor(np.ascontiguousarray(rows_np[i:i + 512]),
                                    dtype=torch.long, device=s.device)
                x = s[r].to(torch.float32)
                sm += x.sum(0)
                sm2 += (x * x).sum(0)
                n += r.numel()
            m = sm / n
            stats.append((m, ((sm2 / n) - m * m).clamp_min(1e-10).sqrt()))
        means = [m for m, _ in stats]
        stds = [sd for _, sd in stats]

    def gather(rows_np):
        r = torch.as_tensor(np.ascontiguousarray(rows_np), dtype=torch.long,
                            device=device)
        feats = [((s[r].to(torch.float32) - m) / sd)
                 for s, m, sd in zip(stacks, means, stds)]
        return r, feats

    def acc_on(model, rows_np, batch=1024):
        model.eval()
        c = 0
        with torch.no_grad():
            for i in range(0, len(rows_np), batch):
                r, feats = gather(rows_np[i:i + batch])
                c += int((model(feats).argmax(1) == y_dev[r]).sum().item())
        return c / len(rows_np)

    def predict_on(model, rows_np, batch=1024):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(rows_np), batch):
                r, feats = gather(rows_np[i:i + batch])
                out.append(model(feats).argmax(1).cpu().numpy())
        return np.concatenate(out)

    def new_model(seed):
        torch.manual_seed(seed)
        return WSProbe([s.shape[1] for s in stacks], [s.shape[2] for s in stacks],
                       int(y_all.max()) + 1).to(device)

    def run_epoch(model, opt, rows_np, seed):
        model.train()
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(len(rows_np), generator=g).numpy()
        for i in range(0, len(perm), A4_BATCH):
            r, feats = gather(rows_np[perm[i:i + A4_BATCH]])
            loss = F.cross_entropy(model(feats), y_dev[r])
            opt.zero_grad()
            loss.backward()
            opt.step()

    def make_opt(model):
        return torch.optim.Adam(model.parameters(), lr=A4_LR, weight_decay=A4_WD)

    def fit_early_stop(fit_rows, stop_rows, seed):
        """Train on fit_rows, early-stop on stop_rows -> (best_epoch, best_acc)."""
        model = new_model(seed)
        opt = make_opt(model)
        best = {"acc": -1.0, "epoch": 1}
        bad = 0
        for ep in range(1, max_epochs + 1):
            run_epoch(model, opt, fit_rows, seed=seed + 100 + ep)
            a = acc_on(model, stop_rows)
            if a > best["acc"]:
                best, bad = {"acc": a, "epoch": ep}, 0
            else:
                bad += 1
                if bad >= A4_PATIENCE:
                    break
        return best

    def retrain(rows_np, n_epochs, seed):
        model = new_model(seed + 7)
        opt = make_opt(model)
        for ep in range(n_epochs):
            run_epoch(model, opt, rows_np, seed=seed + 200 + ep)
        return model

    # ---- CV over repeated group folds (honest nested early stopping) ----
    t0 = time.time()
    cv_accs = []
    for r in range(n_repeats):
        gkf = GroupKFold(n_splits=N_FOLDS, shuffle=(r > 0),
                         random_state=REPEAT_SEEDS[r])
        for k, (tr, va) in enumerate(gkf.split(np.zeros(len(tr_rows)),
                                               y_all[n_test:], groups)):
            sub = tr_rows[tr]
            va_rows = tr_rows[va]
            g2 = GroupKFold(n_splits=2, shuffle=True, random_state=7000 + 100 * r + k)
            fit_i, stop_i = next(iter(g2.split(np.zeros(len(sub)),
                                               y_all[n_test:][tr], groups[tr])))
            fit_rows, stop_rows = sub[fit_i], sub[stop_i]
            set_stats(fit_rows)
            best = fit_early_stop(fit_rows, stop_rows, seed=3000 + 100 * r + k)
            set_stats(sub)
            model = retrain(sub, best["epoch"], seed=5000 + 100 * r + k)
            cv_accs.append(acc_on(model, va_rows))
            log(f"[{stage_key}] {arm}: repeat{r} fold{k} stop_ep={best['epoch']} "
                f"stop_acc={best['acc']:.4f} eval_acc={cv_accs[-1]:.4f}")
    cv_acc = float(np.mean(cv_accs))

    # ---- final: early-stop epoch on a group half of FULL train, retrain all ----
    g2 = GroupKFold(n_splits=2, shuffle=True, random_state=9999)
    fit_i, stop_i = next(iter(g2.split(np.zeros(len(tr_rows)),
                                       y_all[n_test:], groups)))
    set_stats(tr_rows[fit_i])
    best = fit_early_stop(tr_rows[fit_i], tr_rows[stop_i], seed=7000)
    set_stats(tr_rows)
    model = retrain(tr_rows, best["epoch"], seed=8000)
    y_pred = predict_on(model, te_rows)
    wsoft = {name: [round(float(v), 4) for v in
                    torch.softmax(w, dim=0).detach().cpu().numpy()]
             for name, w in zip([m[0] for m in meta], model.w)}
    acc, f1 = save_arm_preds(
        arm, y_pred, y_all[:n_test], canon["keys"][:n_test],
        extra={"kind": "weighted-sum torch arm",
               "cv_acc": cv_acc, "final_best_epoch": best["epoch"],
               "final_stop_half_acc": best["acc"]})
    payload[stage_key][arm] = {
        "sources": sources, "protocol": (
            f"per-layer softmax weights per source -> weighted sum -> concat "
            f"-> torch linear head trained jointly; Adam lr={A4_LR} wd={A4_WD} "
            f"batch={A4_BATCH} max_epochs={max_epochs} patience={A4_PATIENCE}; "
            f"per-layer standardization from fit rows; honest nested early stop "
            f"(GroupKFold(2) inside each sub-train, retrain at best epoch)"),
        "device": str(device), "cv_repeats": n_repeats,
        "cv_acc": cv_acc, "cv_accs": [round(a, 4) for a in cv_accs],
        "test_acc": acc, "test_f1_macro": f1,
        "final_best_epoch": best["epoch"],
        "layer_softmax_weights": wsoft,
        "seconds": round(time.time() - t0, 1)}
    log(f"[{stage_key}] {arm}: cv={cv_acc:.4f} (x{n_repeats} repeats) "
        f"-> test acc={acc:.4f} f1={f1:.4f} [{time.time() - t0:.0f}s]")
    save(payload)
    del stacks, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def stage2(payload, canon):
    log("=" * 70)
    log("STAGE 2: fusion arms (selection on train repeated GroupKFold(3)x3)")
    for n in fusion_sources():
        assert payload["stage1"].get(n, {}).get("complete"), f"stage1 {n} missing"
    stage2_concat_arms(payload, canon)
    if arm_done(payload, "stage2", "A4"):
        log("[stage2] A4: complete -> skip (resume)")
    else:
        weighted_arm("A4", fusion_sources(), payload, "stage2", canon)


# ============================================================== STAGE 3 ======
def stage3(payload, canon):
    log("=" * 70)
    log("STAGE 3: BEATs same-treatment arms")
    s1 = payload["stage1"][BEATS_NAME]
    scores = curve_scores(s1)
    layers_scored = s1["layers_scored"]
    best_L = s1["best_layer"]
    n_test = canon["n_test"]

    def beats_run(arm, layer_list, report_only=False):
        if arm_done(payload, "stage3", arm):
            log(f"[stage3] {arm}: complete -> skip (resume)")
            return
        Xc = load_beats_stack(canon)
        ls = sorted(set(layer_list))
        Xtr = np.concatenate([Xc[n_test:, L, :].astype(np.float32) for L in ls], axis=1)
        Xte = np.concatenate([Xc[:n_test, L, :].astype(np.float32) for L in ls], axis=1)
        del Xc
        run_concat_arm(arm, {BEATS_NAME: ls}, payload, "stage3",
                       {arm: (Xtr, Xte)}, canon)

    beats_run("B1", [best_L])
    beats_run("B2", spread_layers(scores))
    beats_run("B2t", topk_layers(scores))
    if arm_done(payload, "stage3", "B3"):
        log("[stage3] B3: complete -> skip (resume)")
    else:
        weighted_arm("B3", [BEATS_NAME], payload, "stage3", canon)
    # flag the expectations (test-lineage references, never used to choose)
    for arm in ["B1", "B2t"]:
        if arm in payload["stage3"] and "test_acc" in payload["stage3"][arm]:
            payload["stage3"][arm]["expectation_test_lineage"] = EXPECTATIONS[arm]
    save(payload)


# ============================================================== STAGE 4 ======
def mcnemar(a_correct, b_correct):
    from scipy.stats import binomtest
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return {"a_right_b_wrong": n10, "b_right_a_wrong": n01, "p_value": float(p)}


def bootstrap_diff(a_correct, b_correct, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a_correct)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = a_correct[idx].mean(1) - b_correct[idx].mean(1)
    return {"mean_diff": float(a_correct.mean() - b_correct.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def stage4(payload, canon):
    log("=" * 70)
    log("STAGE 4: McNemar exact + 10k paired bootstrap (seed 0)")
    arm_names = [a for a in ["A1", "A2", "A3", "A4", "R0", "B1", "B2", "B2t", "B3"]
                 if preds_file(a).exists()]
    docs = {}
    for a in arm_names:
        d = json.loads(preds_file(a).read_text())
        docs[a] = {k: v for k, v in d["preds"].items()}
    keys = sorted(set.intersection(*[set(d) for d in docs.values()]))
    log(f"[stage4] aligned test keys: {len(keys)} across arms {arm_names}")
    y = np.array([docs[arm_names[0]][k][0] for k in keys])
    correct = {a: np.array([docs[a][k][1] == docs[a][k][0] for k in keys])
               for a in arm_names}
    for a in arm_names:
        assert all(docs[a][k][0] == y[i] for i, k in enumerate(keys)), "y mismatch"

    pairs = {}
    for a, b in itertools.combinations(arm_names, 2):
        acc_a, acc_b = float(correct[a].mean()), float(correct[b].mean())
        pairs[f"{a}_vs_{b}"] = {
            "acc_a": acc_a, "acc_b": acc_b, "dacc": acc_a - acc_b,
            "mcnemar": mcnemar(correct[a], correct[b]),
            "bootstrap": bootstrap_diff(correct[a], correct[b]),
            "involves_report_only_arm": bool(
                payload["stage2"].get(a, {}).get("report_only") or
                payload["stage2"].get(b, {}).get("report_only")),
        }

    def cv_of(stage, arm):
        return payload[stage].get(arm, {}).get("cv_acc", -1)

    llm_arms = [a for a in ["A1", "A2", "A3", "A4"] if a in docs]
    beats_arms = [a for a in ["B1", "B2", "B2t", "B3"] if a in docs]
    best_llm = max(llm_arms, key=lambda a: (cv_of("stage2", a), a)) if llm_arms else None
    best_beats = max(beats_arms, key=lambda a: (cv_of("stage3", a), a)) if beats_arms else None
    payload["stage4"] = {
        "n_aligned": len(keys), "pairs": pairs,
        "best_llm_arm_by_cv": {"arm": best_llm, "cv_acc": cv_of("stage2", best_llm)} if best_llm else None,
        "best_beats_arm_by_cv": {"arm": best_beats, "cv_acc": cv_of("stage3", best_beats)} if best_beats else None,
        "primary_pair": (f"{best_llm}_vs_{best_beats}"
                         if best_llm and best_beats and
                         f"{best_llm}_vs_{best_beats}" in pairs else
                         (f"{best_beats}_vs_{best_llm}" if best_llm and best_beats else None)),
        "llm_vs_B1_pairs": [k for k in pairs if k.endswith("_vs_B1") and
                            k.split("_vs_")[0] in llm_arms],
    }
    save(payload)
    if best_llm and best_beats:
        pr = pairs.get(f"{best_llm}_vs_{best_beats}") or pairs.get(f"{best_beats}_vs_{best_llm}")
        log(f"[stage4] PRIMARY {best_llm} vs {best_beats}: "
            f"dacc={pr['dacc']:+.4f} mcnemar_p={pr['mcnemar']['p_value']:.4f} "
            f"ci95=[{pr['bootstrap']['ci95'][0]:+.4f},{pr['bootstrap']['ci95'][1]:+.4f}]")
    for k in payload["stage4"]["llm_vs_B1_pairs"]:
        pr = pairs[k]
        log(f"[stage4] {k}: dacc={pr['dacc']:+.4f} p={pr['mcnemar']['p_value']:.4f}")
    DONE_MARKER.write_text(time.strftime("%F %T") + "\n")
    log(f"[stage4] done marker -> {DONE_MARKER}")


# ================================================================= main ======
def main():
    payload = load_payload()
    canon = canonical_setup()
    payload["meta"].update({
        "campaign": "fusion_upgrade",
        "smoke": SMOKE,
        "seed": SEED,
        "manifest": canon["manifest"],
        "n_test": canon["n_test"], "n_train": canon["n_train"],
        "train_budget": TRAIN_BUDGET,
        "sources": SMOKE_SOURCES if SMOKE else ARM_ORDER,
        "c_grid": C_GRID, "c_grid_big": C_GRID_BIG,
        "big_dim_threshold": BIG_DIM_THRESHOLD,
        "n_folds": N_FOLDS, "n_repeats_arm_cv": N_REPEATS,
        "repeat_seeds": REPEAT_SEEDS,
        "protocol": (
            "stage1: per-layer GroupKFold(3)-by-recording CV on train, C from "
            "{0.01,0.1,1,10} per layer; one test eval per source at the "
            "CV-argmax layer. stages 2-3: all layer/subset/C/arm selection by "
            "repeated GroupKFold(3)x3 on train (repeat 0 canonical, 1-2 "
            "shuffled seeds 1001/1002); final model refit on ALL train with "
            "the CV-chosen C; exactly ONE test pass per arm, per-clip "
            "predictions saved. A4/B3: torch weighted-sum arms (see their "
            "protocol field). stage4: McNemar exact + 10k paired bootstrap "
            "(seed 0) on saved predictions."),
        "discipline": (
            "ALL selection on train-only CV. Test touched exactly once per "
            "arm's final config. Test-lineage references (R0 layers, B1/B2t "
            "expectation numbers from prior pilots) are report-only and never "
            "used to choose anything."),
        "expectations_test_lineage": EXPECTATIONS,
        "started": time.strftime("%F %T"),
    })
    save(payload)
    log(f"campaign start | smoke={SMOKE} train={canon['n_train']} "
        f"test={canon['n_test']} manifest={canon['manifest']}")
    log(f"artifacts: {OUT_JSON} | preds {PREDS_DIR}/ | done {DONE_MARKER}")

    stage1(payload, canon)
    stage2(payload, canon)
    stage3(payload, canon)
    stage4(payload, canon)

    payload["finished"] = time.strftime("%F %T")
    payload["wall_total_seconds"] = round(time.time() - T0, 1)
    save(payload)
    log(f"DONE in {payload['wall_total_seconds']}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
