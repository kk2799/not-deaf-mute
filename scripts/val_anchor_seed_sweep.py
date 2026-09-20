#!/usr/bin/env python
"""5-seed stochastic sweep for the BEATs / WavLM anchor probes.

Same protocol as the fusion/layer-attention sweeps: deterministic LR
reproduction + torch head with randn init, shuffled mini-batches (256),
dropout (0.1), val-selected checkpoint, five seeds.

Run: python scripts/val_anchor_seed_sweep.py --anchor beats|wavlm
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("fu", "scripts/run_fusion_upgrade.py")
fu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fu)

OUT_JSON = Path("outputs/pilots/val_anchor_seed_sweep.json")
SEEDS = [42, 2027, 3407]
EPOCHS, LR, WD = 300, 1e-3, 1e-4
LAYER = {"beats": 9, "wavlm": 8}          # honest CV-selected anchors
C_LR = {"beats": 0.01, "wavlm": 0.01}
_WAVLM_Y, _WAVLM_TE = None, None


def load_anchor(name, canon):
    if name == "beats":
        X3 = fu.load_beats_stack(canon)                   # [n, 12, 768]
    else:
        import glob
        fs = sorted(glob.glob("outputs/features/anchorlayer/"
                               "wavlm_large_all_layers/shard_*.npz"))
        Xs = [np.load(p)["X"] for p in fs]
        ys_cat = np.concatenate([np.load(p)["y"] for p in fs]).astype(int)
        tes_cat = np.concatenate([np.load(p)["split"] for p in fs]).astype(bool)
        # train block was shuffled differently at extraction time; X/y stay
        # paired within shards, and the TEST block (what evaluation reads)
        # aligns with the canonical order
        assert np.array_equal(ys_cat[tes_cat], canon["y"][canon["te"]]), \
            "wavlm test-block order mismatch"
        global _WAVLM_Y, _WAVLM_TE
        _WAVLM_Y, _WAVLM_TE = ys_cat, tes_cat
        X3 = np.concatenate(Xs)
    X = np.ascontiguousarray(X3[:, LAYER[name], :].astype(np.float32))
    del X3
    gc.collect()
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=["beats", "wavlm"], required=True)
    ap.add_argument("--val-source", default="cv", choices=["cv", "explicit"],
                    help="checkpoint selection: cv = train-side recording-grouped holdout (default); explicit = the validation split (paper dev protocol)")
    args = ap.parse_args()
    torch.set_num_threads(15)
    canon = fu.canonical_setup()
    y, te, rid = canon["y"], canon["te"], canon["rid"]
    if args.anchor == "wavlm":
        X = load_anchor(args.anchor, canon)
        y, te = _WAVLM_Y, _WAVLM_TE
    else:
        X = load_anchor(args.anchor, canon)
    tr, va = ~te, te
    print(f"[{args.anchor}] L{LAYER[args.anchor]} X={X.shape}", flush=True)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = (X - mu) / sd

    results = {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, C=C_LR[args.anchor]))
    clf.fit(X[tr], y[tr])
    results["LR_deterministic"] = {"clip": round(float(clf.score(X[va], y[va])), 4)}
    print(f"[{args.anchor} LR deterministic] "
          f"{results['LR_deterministic']['clip']}", flush=True)

    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    tr_t = torch.as_tensor(np.where(tr)[0])
    va_t = torch.as_tensor(np.where(va)[0])
    D = Xt.shape[1]
    stoch = {}
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
                loss = torch.nn.functional.cross_entropy(h @ hw + hb, yt[idx])
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
            clip = float(((Xt[va_t] @ hw + hb).argmax(1)
                          == yt[va_t]).float().mean())
        stoch[str(s)] = {"clip": round(clip, 4), "select_ep": best_ep}
        print(f"[{args.anchor} seed {s}] clip={clip:.4f} ep{best_ep}",
              flush=True)
    results["torch_stochastic"] = stoch
    accs = [v["clip"] for v in stoch.values()]
    results["summary"] = {"mean": round(float(np.mean(accs)), 4),
                          "std": round(float(np.std(accs, ddof=1)), 4)}
    print(f"[{args.anchor} summary] {results['summary']}", flush=True)

    OUT = OUT_JSON.with_name(OUT_JSON.stem + f"_{args.anchor}.json")
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data.update(results)
    OUT.write_text(json.dumps(data, indent=1))
    print("[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
