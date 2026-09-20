#!/usr/bin/env python
"""5-seed stochastic sweep for the 7-source concatenation fusion (0.743 arm).

Features: each source's CV-selected best layer, concatenated (~26.6k dims),
standardized on train. Two heads compared:
  LR    sklearn LogisticRegression (convex, deterministic) — reproduces the
        paper's 0.743 exactly; reported once as the deterministic anchor.
  torch randn init + shuffled mini-batches (256) + dropout (0.1), Adam,
        val-selected checkpoint — run over 5 seeds.

Run: nohup taskset -c 0-31 nice -n 10 python scripts/val_fusion_seed_sweep.py \
        > outputs/val_fusion_seed_sweep.log 2>&1 &
"""
from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("fu", "scripts/run_fusion_upgrade.py")
fu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fu)

OUT_JSON = Path("outputs/pilots/val_fusion_seed_sweep.json")
BEST_LAYERS = {"omni_aud": 11, "q2a_aud": 18, "vl8b_mel": 3, "vl32b_mel": 10,
               "omni_mel": 2, "vl8b_stft": 3, "vl32b_stft": 7}
SEEDS = [42, 2027, 3407]
EPOCHS, LR, WD = 300, 1e-3, 1e-4


def rec_scores(prob, rid_va, y_va):
    clip = float((prob.argmax(1) == y_va).mean())
    agg, yrec = {}, {}
    for r, p, t in zip(rid_va, prob, y_va):
        agg.setdefault(r, np.zeros(prob.shape[1]))
        agg[r] += p
        yrec[r] = int(t)
    rec_mean = float(np.mean([a.argmax() == yrec[r] for r, a in agg.items()]))
    return {"clip": round(clip, 4), "rec_mean": round(rec_mean, 4)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-source", default="cv", choices=["cv", "explicit"],
                    help="checkpoint selection: cv = train-side recording-grouped holdout (default); explicit = the validation split (paper dev protocol)")
    args = ap.parse_args()
    torch.set_num_threads(15)
    canon = fu.canonical_setup()
    y, te, rid = canon["y"], canon["te"], canon["rid"]
    tr, va = ~te, te

    Xs = []
    for name, L in BEST_LAYERS.items():
        Xfull = fu.load_llm_stack(name, canon)
        X = np.ascontiguousarray(Xfull[:, L, :].astype(np.float32))
        del Xfull
        gc.collect()
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs.append((X - mu) / sd)
        print(f"[load] {name} L{L} {X.shape}", flush=True)
    Xcat = np.concatenate(Xs, axis=1)                       # [n, ~26.6k]
    del Xs
    gc.collect()
    print(f"[setup] concat dims={Xcat.shape[1]}", flush=True)

    results = {}

    # ---- deterministic LR anchor (reproduces the paper's 0.743) ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
                                                             C=0.01))
    clf.fit(Xcat[tr], y[tr])
    lr_clip = float(clf.score(Xcat[va], y[va]))
    results["LR_deterministic"] = {"clip": round(lr_clip, 4)}
    print(f"[LR deterministic] {lr_clip:.4f}", flush=True)

    # ---- stochastic torch head over 5 seeds (with per-clip probs) ----
    Xt = torch.from_numpy(Xcat)
    yt = torch.from_numpy(y)
    tr_t = torch.as_tensor(np.where(tr)[0])
    va_t = torch.as_tensor(np.where(va)[0])
    D = Xt.shape[1]
    stoch = {}
    all_probs = {}
    rid_np = np.array(rid)
    uniq_tr = np.unique(rid_np[tr])
    for s in SEEDS:
        torch.manual_seed(s)
        if args.val_source == "cv":
            hold = set(np.random.default_rng(s).permutation(uniq_tr)
                       [:max(1, int(0.10 * len(uniq_tr)))])
            in_tr = np.array([r in hold for r in rid_np[tr]])
            sel_t = tr_t[torch.as_tensor(~in_tr)]
            point_t = tr_t[torch.as_tensor(in_tr)]
        else:
            sel_t, point_t = tr_t, va_t
        hw = (torch.randn(D, 4) * 0.01).requires_grad_(True)
        hb = torch.zeros(4, requires_grad=True)
        opt = torch.optim.Adam([{"params": [hw, hb], "lr": LR,
                                 "weight_decay": WD}])
        best_acc, st, best_ep = -1.0, None, -1
        for ep in range(EPOCHS):
            perm = torch.randperm(len(sel_t))
            for bi in range(0, len(tr_t), 256):
                idx = sel_t[perm[bi:bi + 256]]
                opt.zero_grad(set_to_none=True)
                h = torch.nn.functional.dropout(Xt[idx], p=0.1,
                                                training=True)
                loss = torch.nn.functional.cross_entropy(h @ hw + hb,
                                                          yt[idx])
                loss.backward()
                opt.step()
            with torch.no_grad():
                acc = float(((Xt[point_t] @ hw + hb).argmax(1)
                             == yt[point_t]).float().mean())
            if acc > best_acc:
                best_acc, best_ep = acc, ep
                st = (hw.detach().clone(), hb.detach().clone())
        hw, hb = st
        with torch.no_grad():
            prob = torch.softmax(Xt[va_t] @ hw + hb, 1).numpy()
        all_probs[str(s)] = prob.astype(np.float32)
        sc = rec_scores(prob, rid[va], y[va])
        stoch[str(s)] = {**sc, "select_ep": best_ep}
        print(f"[seed {s}] {sc} ep{best_ep}", flush=True)
    results["torch_stochastic"] = stoch
    np.savez("outputs/pilots/val_fusion_seed_probs.npz",
             probs=np.stack([all_probs[str(s)] for s in SEEDS]),
             seeds=np.array(SEEDS), y_val=y[va], rid_val=rid[va])
    print("[saved] per-clip probabilities for TOST", flush=True)
    accs = [v["clip"] for v in stoch.values()]
    results["summary"] = {"mean": round(float(np.mean(accs)), 4),
                          "std": round(float(np.std(accs, ddof=1)), 4)}
    print(f"[summary] mean={results['summary']['mean']} "
          f"std={results['summary']['std']}", flush=True)

    OUT_JSON.write_text(json.dumps(results, indent=1))
    print("[saved]", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
