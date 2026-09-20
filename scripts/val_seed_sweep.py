#!/usr/bin/env python
"""Seed sweep for the paper's new layer-attention numbers.

Three configs x 5 seeds (42, 123, 2026, 2027, 3407):
  A  v1 features (layerprobe omni_aud, 3k-train rows fixed by the
     extraction-time seed 42) + global 49-layer attention, linear head.
     Seed varies the head initialization only (row order is baked into
     the shards; noted in the output).
  B  v2 features (layerprobe2 omni _full, 11,035 rows) + ch3 channel
     + full 8,350 train + same attention. Seed varies head init only
     (full-batch training: row order has no effect).
  C  v2 features + ch3 + 3k train subset re-drawn per seed
     (rng(seed).permutation) --- isolates subset-selection variance.

All runs select the checkpoint on the validation split (dev protocol).

Run: nohup taskset -c 0-31 nice -n 10 python scripts/val_seed_sweep.py \
        > outputs/val_seed_sweep.log 2>&1 &
"""
from __future__ import annotations

import collections
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("fu", "scripts/run_fusion_upgrade.py")
fu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fu)

OUT_JSON = Path("outputs/pilots/val_seed_sweep.json")
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


def load_v1():
    """omni_aud v1 stack in canonical row order -> [5685, 49, D] fp32 std."""
    canon = fu.canonical_setup()
    Xfull = fu.load_llm_stack("omni_aud", canon)
    X = Xfull.astype(np.float32)
    del Xfull
    gc.collect()
    y, te, rid = canon["y"], canon["te"], canon["rid"]
    tr = ~te
    mu = X[tr].mean(axis=0, dtype=np.float32)
    sd = X[tr].std(axis=0, dtype=np.float32) + 1e-6
    return torch.from_numpy((X - mu) / sd), torch.from_numpy(y), \
        torch.from_numpy(te), rid


def load_v2_full(ch=3):
    import glob
    d = sorted(glob.glob("outputs/features/layerprobe2/"
                         "qwen3_omni_30b_audio_*_full/shard_*.npz"))
    Xs, ys, tes, rids = [], [], [], []
    for p in sorted(d):
        z = np.load(p)
        Xs.append(z["X"]); ys.append(z["y"]); tes.append(z["split"])
        rids.append(z["rid"].astype(str))
    X = np.concatenate(Xs)                       # [n, 49, 4, D]
    y = np.concatenate(ys).astype(np.int64)
    te = np.concatenate(tes).astype(bool)
    rid = np.concatenate(rids)
    Xc = np.ascontiguousarray(X[:, :, ch, :].astype(np.float32))
    del X
    gc.collect()
    tr = ~te
    mu = Xc[tr].mean(axis=0, dtype=np.float32)
    sd = Xc[tr].std(axis=0, dtype=np.float32) + 1e-6
    return torch.from_numpy((Xc - mu) / sd), torch.from_numpy(y), \
        torch.from_numpy(te), rid


def run_attn(Xt, yt, te_t, rid, seed, train_rows=None, stochastic=False,
            val_mode="cv"):
    """val_mode: 'cv' selects the checkpoint on a recording-grouped 10%
    holdout carved from TRAIN (default, no test contact); 'explicit'
    selects on the validation split itself (reproduces the paper's dev
    protocol). Final metrics are always reported on the validation
    split rows."""
    """Global layer attention + linear head; val-selected checkpoint.

    stochastic=True introduces the three classic randomness sources:
    randn init (0.01), per-epoch shuffled mini-batches (256), and
    feature dropout (p=0.1) on the aggregated vector — all driven by
    torch.manual_seed(seed).
    """
    torch.manual_seed(seed)
    L, D = Xt.shape[1], Xt.shape[2]
    tr_rows = train_rows if train_rows is not None \
        else torch.as_tensor(np.where(~te_t.numpy())[0])
    va_rows = torch.as_tensor(np.where(te_t.numpy())[0])
    if val_mode == "cv":
        rid_np = rid if isinstance(rid, np.ndarray) else np.array(rid)
        tr_rec = np.array([rid_np[i] for i in tr_rows.numpy()])
        uniq = np.unique(tr_rec)
        hold = set(np.random.default_rng(seed).permutation(uniq)
                   [:max(1, int(0.10 * len(uniq)))])
        is_hold = np.array([r in hold for r in tr_rec])
        point_rows = tr_rows[torch.as_tensor(is_hold)]
        tr_rows = tr_rows[torch.as_tensor(~is_hold)]
    else:
        point_rows = va_rows
    if stochastic:
        w = (torch.randn(L) * 0.01).requires_grad_(True)
        hw = (torch.randn(D, 4) * 0.01).requires_grad_(True)
        hb = torch.zeros(4, requires_grad=True)
    else:
        w = torch.zeros(L, requires_grad=True)
        hw = torch.zeros(D, 4, requires_grad=True)
        hb = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [w], "lr": 1e-2, "weight_decay": WD},
        {"params": [hw, hb], "lr": LR, "weight_decay": WD}])
    BATCH = 256
    best_acc, st, best_ep = -1.0, None, -1
    for ep in range(EPOCHS):
        if stochastic:
            perm = torch.randperm(len(tr_rows))
            for bi in range(0, len(tr_rows), BATCH):
                idx = tr_rows[perm[bi:bi + BATCH]]
                opt.zero_grad(set_to_none=True)
                a = torch.softmax(w, 0)
                h = torch.einsum("l,nld->nd", a.to(Xt.dtype), Xt[idx])
                h = torch.nn.functional.dropout(h, p=0.1, training=True)
                loss = torch.nn.functional.cross_entropy(h @ hw + hb, yt[idx])
                loss.backward()
                opt.step()
        else:
            opt.zero_grad(set_to_none=True)
            a = torch.softmax(w, 0)
            h = torch.einsum("l,nld->nd", a.to(Xt.dtype), Xt[tr_rows])
            loss = torch.nn.functional.cross_entropy(h @ hw + hb, yt[tr_rows])
            loss.backward()
            opt.step()
        with torch.no_grad():
            av = torch.softmax(w, 0).to(Xt.dtype)
            hv = torch.einsum("l,nld->nd", av, Xt[point_rows])
            acc = float(((hv @ hw + hb).argmax(1) == yt[point_rows]).float().mean())
        if acc > best_acc:
            best_acc, best_ep = acc, ep
            st = (w.detach().clone(), hw.detach().clone(), hb.detach().clone())
    w, hw, hb = st
    with torch.no_grad():
        av = torch.softmax(w, 0).to(Xt.dtype)
        hv = torch.einsum("l,nld->nd", av, Xt[va_rows])
        prob = torch.softmax(hv @ hw + hb, 1).numpy()
    s = rec_scores(prob, rid[te_t.numpy()], yt[va_rows].numpy())
    return {"val": s, "select_ep": best_ep}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="A", choices=["A", "B", "C"])
    ap.add_argument("--val-source", default="cv", choices=["cv", "explicit"],
                    help="checkpoint selection: cv = train-side recording-grouped"
                         " holdout (default); explicit = the validation split"
                         " (paper dev protocol)")
    args = ap.parse_args()
    torch.set_num_threads(15)
    out = OUT_JSON.with_name(OUT_JSON.stem + f"_{args.config}_stoch.json")
    results = {}
    if out.exists():
        try:
            results = json.loads(out.read_text())
        except Exception:
            results = {}

    if args.config == "A":
        Xt, yt, te_t, rid = load_v1()
        results["A_stoch"] = {str(s): run_attn(Xt, yt, te_t, rid, s,
                                               stochastic=True)
                              for s in SEEDS}
        del Xt
        gc.collect()
    elif args.config == "B":
        Xt, yt, te_t, rid = load_v2_full(ch=3)
        results["B_stoch"] = {str(s): run_attn(Xt, yt, te_t, rid, s,
                                               stochastic=True)
                              for s in SEEDS}
        del Xt
        gc.collect()
    else:  # C: per-seed 3k subset
        Xt, yt, te_t, rid = load_v2_full(ch=3)
        tr_all = np.where(~te_t.numpy())[0]
        res_c = {}
        for s in SEEDS:
            rng = np.random.default_rng(s)
            sub = rng.permutation(len(tr_all))[:3000]
            rows = torch.as_tensor(tr_all[sub])
            res_c[str(s)] = run_attn(Xt, yt, te_t, rid, s, train_rows=rows,
                         val_mode=args.val_source)
        results["C_v2_3k_subset"] = res_c
        del Xt
        gc.collect()

    out.write_text(json.dumps(results, indent=1))
    for k, v in results.items():
        accs = [vv["val"]["clip"] for vv in v.values()]
        print(f"[summary] {k}: accs={accs} mean={np.mean(accs):.4f} "
              f"std={np.std(accs, ddof=1):.4f}", flush=True)
    print("[saved]", out, flush=True)


if __name__ == "__main__":
    main()
