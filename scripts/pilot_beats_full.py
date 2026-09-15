#!/usr/bin/env python
"""PILOT (selection-hygiene extension): BEATs anchor at the FULL 8,350-clip
train budget.

The 0.749 anchor was measured on the 3k subset only (run_beats_layer_probe.py,
MAX_TRAIN=3000) and its learning curve 2k=0.739 -> 3k=0.749 was still rising.
This pilot (a) extracts BEATs 12-layer features for the remaining 5,350 train
clips (GPU, ~20 min), (b) probes every layer at full budget with the STANDARD
protocol (StandardScaler + LogisticRegression(max_iter=2000), C grid
{0.01,0.1,1,10} by 3-fold CV on train — run_layer_probe.py protocol), (c)
selects the layer by TRAIN CV (plain 3-fold AND GroupKFold-by-recording) and
reports test acc/macro-F1 at the CV-selected layer (honest) alongside the
test-max layer (diagnostic), (d) probes the CV-top-3 layer concat, and (e)
runs WavLM-Large full-budget as a pipeline sanity anchor (expect best layer 2
acc ~ 0.730, matching outputs/results/anchor_layer_probe.json).

Artifacts (PILOT ONLY — never touches outputs/results/ or save_result):
  outputs/pilots/beats_full_budget.json     incremental results
  outputs/pilots/beats_full_budget.log      captured stdout
  outputs/features/anchorlayer/beats_all_layers/shard_ext_{s:04d}.npz
Row order is the canonical ordered_rows(SEED=42) sequence, verified against
the existing shard_all.npz y/split columns before anything runs.
"""
from __future__ import annotations

import json, os, time
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = Path("outputs/features/anchorlayer")
PILOTS = Path("outputs/pilots")
SEED = 42
SHARD = 500
MANIFEST = Path("outputs/manifests/deepship_clip30_ov0.5_5300b861c63f.csv")
OUT_JSON = PILOTS / "beats_full_budget.json"
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_FOLDS = 3

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def save(payload):
    PILOTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=1))


def ordered_rows_full(manifest: pd.DataFrame):
    """run_layer_probe.ordered_rows(manifest, max_train, full_train=True):
    all test rows (manifest order) first, then ALL train rows seeded-shuffled."""
    test_idx = manifest.index[manifest["split"] == "test"].to_numpy()
    train_df = manifest[manifest["split"] == "train"]
    perm = np.random.default_rng(SEED).permutation(len(train_df))
    train_idx = train_df.index.to_numpy()[perm]
    return list(test_idx) + list(train_idx)


# ---------------------------------------------------------------- probes ----
def probe_layer(Xtr, ytr, Xte, yte):
    """Standard protocol: GridSearchCV(3-fold) over C on train, eval once."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    grid = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, cv=N_FOLDS, n_jobs=8)
    grid.fit(Xtr, ytr)
    pred = grid.predict(Xte)
    return {"acc": float(accuracy_score(yte, pred)),
            "f1_macro": float(f1_score(yte, pred, average="macro")),
            "C": float(grid.best_params_["logisticregression__C"]),
            "cv_acc": float(grid.best_score_),
            "fit_s": round(time.time() - t0, 1)}


def group_cv_layer(Xtr, ytr, groups):
    """GroupKFold-by-recording mean CV acc per C (selection-repair pattern)."""
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=N_FOLDS)
    splits = list(gkf.split(Xtr, ytr, groups=groups))

    def one(C, tr, va):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))
        pipe.fit(Xtr[tr], ytr[tr])
        return float((pipe.predict(Xtr[va]) == ytr[va]).mean())

    t0 = time.time()
    out = Parallel(n_jobs=12)(
        delayed(one)(C, tr, va) for C in C_GRID for tr, va in splits)
    per_C = {C: float(np.mean(out[i * N_FOLDS:(i + 1) * N_FOLDS]))
             for i, C in enumerate(C_GRID)}
    best_C = max(per_C, key=per_C.get)
    return {"best_C": float(best_C), "best_cv_acc": per_C[best_C], "per_C": per_C,
            "fit_s": round(time.time() - t0, 1)}


# ------------------------------------------------------------ extraction ----
def extract_beats_extension(rows_ext: pd.DataFrame, out_dir: Path, device: str):
    """BEATs 12-layer features for the extension rows, sharded + resumable.
    Mirrors run_beats_layer_probe.extract (forward hooks, [:,0] time-pool)."""
    import torch
    from splash.audio.io import read_segment
    from splash.models.factory import build_model

    n_shards = (len(rows_ext) + SHARD - 1) // SHARD
    enc = build_model({"name": "beats_iter3_plus",
                       "path": "/models/BEATs/BEATs_iter3_plus_AS2M.pt",
                       "device": device})
    clip_samples = int(30.0 * 16000)
    for s in range(n_shards):
        path = out_dir / f"shard_ext_{s:04d}.npz"
        if path.exists():
            try:
                if len(np.load(path, allow_pickle=False)["y"]) == min(SHARD, len(rows_ext) - s * SHARD):
                    continue
            except Exception:
                print(f"  ext shard {s}: corrupt — recomputing")
        rows = rows_ext.iloc[s * SHARD:(s + 1) * SHARD]
        acts = []

        def hook(_m, _i, output):
            acts.append(output[0][:, 0].float().mean(dim=0).cpu())

        handles = [m.register_forward_hook(hook) for m in enc.model.encoder.layers]
        feats = []
        with torch.inference_mode():
            for r in rows.itertuples(index=False):
                wav = read_segment(r.path, r.start, r.end, target_sr=16000,
                                   pad_to_samples=clip_samples)
                x = torch.from_numpy(wav[None]).to(enc.device)
                acts.clear()
                enc.model.extract_features(x)
                assert len(acts) == len(enc.model.encoder.layers), "hook miss"
                feats.append(torch.stack(acts).half().numpy())
        for h in handles:
            h.remove()
        X = np.stack(feats)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, X=X, y=rows["label_id"].to_numpy(np.int16),
                 split=(rows["split"] == "test").to_numpy(np.int8),
                 rid=rows["recording_id"].to_numpy().astype(str),
                 cidx=rows["clip_idx"].to_numpy(np.int32))
        os.replace(tmp, path)
        log(f"  ext shard {s + 1}/{n_shards}: {len(rows)} clips -> {X.shape}")
    del enc
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    shards = sorted(out_dir.glob("shard_ext_*.npz"))
    X = np.concatenate([np.load(p, allow_pickle=False)["X"] for p in shards])
    y = np.concatenate([np.load(p, allow_pickle=False)["y"] for p in shards]).astype(int)
    te = np.concatenate([np.load(p, allow_pickle=False)["split"] for p in shards]).astype(bool)
    rid = np.concatenate([np.load(p, allow_pickle=False)["rid"].astype(str) for p in shards])
    return X, y, te, rid


def main():
    device = os.environ.get("PILOT_DEVICE", "cuda:0")
    payload = {"pilot": "beats_full_budget", "started": time.strftime("%F %T"),
               "device": device, "protocol":
               "StandardScaler+LR(max_iter=2000), C grid {0.01,0.1,1,10} by 3-fold CV on train; "
               "layer selection by TRAIN CV (plain + GroupKFold-by-recording)", "stages": {}}
    save(payload)

    # ---- stage 0: canonical row order + verify against existing 3k shard ----
    full = pd.read_csv(MANIFEST)
    order = ordered_rows_full(full)
    rows_full = full.iloc[order]
    z0 = np.load(FEATURES / "beats_all_layers" / "shard_all.npz", allow_pickle=False)
    X0, y0, te0 = z0["X"], z0["y"].astype(int), z0["split"].astype(bool)
    n0 = len(y0)
    head = rows_full.iloc[:n0]
    ok_y = np.array_equal(y0, head["label_id"].to_numpy(int))
    ok_te = np.array_equal(te0, (head["split"] == "test").to_numpy())
    log(f"stage0: manifest rows={len(rows_full)} existing shard n={n0} "
        f"(train {int((~te0).sum())}) y_match={ok_y} split_match={ok_te}")
    if not (ok_y and ok_te):
        raise RuntimeError("row-order verification FAILED — aborting")
    rid_all = rows_full["recording_id"].to_numpy().astype(str)
    payload["stages"]["stage0_verify"] = {"n_manifest": len(rows_full), "n_existing": n0,
                                          "n_test": int(te0.sum()),
                                          "train_full": int((rows_full.split == 'train').sum()),
                                          "y_match": bool(ok_y), "split_match": bool(ok_te)}
    save(payload)

    # ---- stage 1: extract extension rows through BEATs (GPU) ----
    t_ext = time.time()
    rows_ext = rows_full.iloc[n0:]
    log(f"stage1: extracting {len(rows_ext)} extension rows on {device}")
    Xe, ye, tee, ride = extract_beats_extension(rows_ext, FEATURES / "beats_all_layers", device)
    assert np.array_equal(ye, rows_ext["label_id"].to_numpy(int))
    assert not tee.any()
    payload["stages"]["stage1_extract"] = {"n_rows": int(len(ye)), "shape": list(Xe.shape),
                                           "wall_min": round((time.time() - t_ext) / 60, 1)}
    save(payload)

    # ---- assemble full-budget arrays ----
    X = np.concatenate([X0, Xe])                      # [N, 12, 768] fp16
    y = np.concatenate([y0, ye])
    te = np.concatenate([te0, tee])
    rid = np.concatenate([rid_all[:n0], ride])
    tr = ~te
    log(f"assembled X={X.shape} train={int(tr.sum())} test={int(te.sum())}")
    payload["n_train_full"] = int(tr.sum()); payload["n_test"] = int(te.sum())
    payload["n_layers_beats"] = int(X.shape[1])
    save(payload)

    # ---- stage 2: per-layer standard probe at FULL budget ----
    t2 = time.time()
    curves = []
    for L in range(X.shape[1]):
        Xl = X[:, L, :].astype(np.float32)
        r = probe_layer(Xl[tr], y[tr], Xl[te], y[te])
        r["layer"] = L
        curves.append(r)
        log(f"stage2: layer {L:2d} cv={r['cv_acc']:.4f} C={r['C']} "
            f"test acc={r['acc']:.4f} f1={r['f1_macro']:.4f} ({r['fit_s']}s)")
    by_cv = max(curves, key=lambda c: c["cv_acc"])
    by_test = max(curves, key=lambda c: c["acc"])
    payload["stages"]["stage2_full_budget_layers"] = {
        "curves": curves, "wall_min": round((time.time() - t2) / 60, 1)}
    payload["beats_full"] = {
        "layer_by_trainCV": by_cv["layer"], "cv_acc": by_cv["cv_acc"],
        "test_acc_at_trainCV_layer": by_cv["acc"], "test_f1_at_trainCV_layer": by_cv["f1_macro"],
        "layer_by_test": by_test["layer"], "test_acc_at_test_layer": by_test["acc"],
        "test_f1_at_test_layer": by_test["f1_macro"]}
    save(payload)

    # ---- stage 2b: GroupKFold-by-recording layer selection ----
    t2b = time.time()
    groups = rid[tr]
    gcurves = []
    for L in range(X.shape[1]):
        Xl = X[:, L, :].astype(np.float32)
        g = group_cv_layer(Xl[tr], y[tr], groups)
        g["layer"] = L
        gcurves.append(g)
        log(f"stage2b: layer {L:2d} groupCV={g['best_cv_acc']:.4f} C={g['best_C']}")
    g_by_cv = max(gcurves, key=lambda c: c["best_cv_acc"])
    payload["stages"]["stage2b_groupkfold"] = {
        "curves": gcurves, "wall_min": round((time.time() - t2b) / 60, 1),
        "layer_by_groupCV": g_by_cv["layer"], "groupCV_acc": g_by_cv["best_cv_acc"]}
    save(payload)

    # ---- stage 3: CV-top-3 layer concat at full budget ----
    t3 = time.time()
    top3_plain = [c["layer"] for c in sorted(curves, key=lambda c: -c["cv_acc"])[:3]]
    top3_group = [c["layer"] for c in sorted(gcurves, key=lambda c: -c["best_cv_acc"])[:3]]
    out3 = {}
    for tag, layers in [("top3_by_plainCV", top3_plain), ("top3_by_groupCV", top3_group)]:
        M = np.concatenate([X[:, L, :].astype(np.float32) for L in layers], axis=1)
        r = probe_layer(M[tr], y[tr], M[te], y[te])
        r["layers"] = layers
        out3[tag] = r
        log(f"stage3: {tag} layers={layers} cv={r['cv_acc']:.4f} test acc={r['acc']:.4f} "
            f"f1={r['f1_macro']:.4f}")
    payload["stages"]["stage3_concat_top3"] = {**out3, "wall_min": round((time.time() - t3) / 60, 1)}
    save(payload)

    # ---- stage 4: learning-curve delta at L6 / best layer (internal 3k) ----
    t4 = time.time()
    tr_idx = np.where(tr)[0]
    sub3k = tr_idx[:3000]                # same seeded subset as the original 3k sweep
    delta = {}
    for tag, L in [("L6", 6), ("full_best_trainCV", by_cv["layer"]),
                   ("full_best_groupCV", g_by_cv["layer"])]:
        Xl = X[:, L, :].astype(np.float32)
        r3 = probe_layer(Xl[sub3k], y[sub3k], Xl[te], y[te])
        delta[tag] = {"layer": L, "acc_3k": r3["acc"], "f1_3k": r3["f1_macro"],
                      "acc_full": next(c["acc"] for c in curves if c["layer"] == L),
                      "delta_pt": round(100 * (next(c["acc"] for c in curves if c["layer"] == L)
                                               - r3["acc"]), 2)}
        log(f"stage4: {tag} L={L}: 3k={r3['acc']:.4f} -> full={delta[tag]['acc_full']:.4f} "
            f"(+{delta[tag]['delta_pt']}pt)")
    payload["stages"]["stage4_3k_delta"] = {**delta, "wall_min": round((time.time() - t4) / 60, 1),
                                            "ref_3k_official": {"L6_acc": 0.7493,
                                                                "protocol": "cv=5, max_iter=3000"}}
    save(payload)

    # ---- stage 5: WavLM full-budget sanity (expect best ~L2, acc ~0.730) ----
    t5 = time.time()
    wdir = FEATURES / "wavlm_large_all_layers"
    wshards = sorted(wdir.glob("shard_*.npz"))
    Xw = np.concatenate([np.load(p, allow_pickle=False)["X"] for p in wshards])
    yw = np.concatenate([np.load(p, allow_pickle=False)["y"] for p in wshards]).astype(int)
    tew = np.concatenate([np.load(p, allow_pickle=False)["split"] for p in wshards]).astype(bool)
    log(f"stage5: WavLM X={Xw.shape} train={int((~tew).sum())} test={int(tew.sum())}")
    wcurves = []
    for L in range(Xw.shape[1]):
        Xl = Xw[:, L, :].astype(np.float32)
        r = probe_layer(Xl[~tew], yw[~tew], Xl[tew], yw[tew])
        r["layer"] = L
        wcurves.append(r)
        log(f"stage5: layer {L:2d} cv={r['cv_acc']:.4f} C={r['C']} test acc={r['acc']:.4f}")
    wb = max(wcurves, key=lambda c: c["acc"])
    wbcv = max(wcurves, key=lambda c: c["cv_acc"])
    payload["stages"]["stage5_wavlm_sanity"] = {
        "curves": wcurves, "wall_min": round((time.time() - t5) / 60, 1),
        "best_by_test": {"layer": wb["layer"], "acc": wb["acc"]},
        "best_by_trainCV": {"layer": wbcv["layer"], "acc": wbcv["acc"]},
        "reference": {"source": "outputs/results/anchor_layer_probe.json",
                      "protocol": "cv=5, max_iter=3000",
                      "best_layer": 2, "best_acc": 0.7300}}
    save(payload)

    payload["finished"] = time.strftime("%F %T")
    payload["wall_total_min"] = round((time.time() - T0) / 60, 1)
    save(payload)
    log(f"DONE in {payload['wall_total_min']} min -> {OUT_JSON}")


if __name__ == "__main__":
    main()
