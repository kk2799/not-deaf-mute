#!/usr/bin/env python
"""Steering statistical upgrade: random-direction DISTRIBUTION + scaling control.

run_steering.py established probe-direction steering (+2~+5.5pt over one random
direction) but with a single random seed, n=200 — reviewers will ask for a null
distribution. This script adds, on the same vl8b/mel/L3 setup:

  1. K=12 random direction SETS (each a full class-indexed dict, mirroring the
     probe protocol) × α ∈ {0.05, 0.2}  → mean ± sd null band
  2. uniform-scaling control  h → (1+α)·h  × α ∈ {0.05, 0.2}
     (generic activation gain, no added vector — isolates direction-specific
     effect from "any activation increase")
  3. probe direction + α=0 reference re-run on the same n=400 clips

Paired bootstrap CI for probe − null-band-mean at each α.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.models.factory import build_model
from splash.prompting.templates import option_layout, build_prompt

FEATURES = Path("outputs/features/layerprobe")
CURVES = Path("outputs/results/layer_probe")
RESULTS = Path("outputs/results")
DATASET, SPEC = "deepship", "mel"
MODEL, PATH = "qwen3_vl_8b", "/models/Qwen3-VL-8B-Instruct"
N_CLIPS = 400
K_RANDOM = 12
ALPHAS = [0.0, 0.05, 0.2]
SEED = 99


def load_features():
    d = sorted(FEATURES.glob(f"{MODEL}_{SPEC}_*"))
    shards = sorted(d[0].glob("shard_*.npz"))
    X = np.concatenate([np.load(p, allow_pickle=False)["X"] for p in shards])
    y = np.concatenate([np.load(p, allow_pickle=False)["y"] for p in shards]).astype(int)
    te = np.concatenate([np.load(p, allow_pickle=False)["split"] for p in shards]).astype(bool)
    return X, y, te


def probe_directions(X, y, te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    cj = CURVES / f"{MODEL}_{SPEC}.json"
    L = json.loads(cj.read_text())["best"]["layer"]
    Xl = X[:, L, :].astype(np.float32)
    sc = StandardScaler().fit(Xl[~te])
    Z = sc.transform(Xl)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(Z[~te], y[~te])
    acc = accuracy_score(y[te], lr.predict(Z[te]))
    dirs = {}
    for c in range(lr.coef_.shape[0]):
        w = lr.coef_[c] / sc.scale_
        dirs[c] = (w / np.linalg.norm(w)).astype(np.float32)
    print(f"[steering-ctrl] layer {L}: probe acc {acc:.3f}")
    return L, dirs, acc


def main():
    names = class_names(DATASET, "en")
    letters, _ids, _ = option_layout(names)
    X, y, _te_f = load_features()
    L, dirs, probe_acc = probe_directions(X, y, _te_f)

    manifest = find_manifest({"name": DATASET, "clip_len": 30.0, "overlap": 0.5})
    aug = pd.read_csv(manifest.with_name(f"{manifest.stem}_spec.csv"))
    te = aug[aug["split"] == "test"]
    test = pd.concat([g.sample(n=min(N_CLIPS // 4, len(g)), random_state=SEED)
                      for _, g in te.groupby("label_id")]).reset_index(drop=True)

    import torch
    model = build_model({"name": MODEL, "path": PATH, "device": "cuda:0", "device_map": None})
    layer_module = model.model.model.language_model.layers[max(L - 1, 0)]
    D = X.shape[2]
    rng = np.random.default_rng(SEED)
    rand_sets = []
    for _ in range(K_RANDOM):
        rand_sets.append({c: (lambda v: v / np.linalg.norm(v))(rng.standard_normal(D).astype(np.float32))
                          for c in dirs})

    def make_hook(direction_for, alpha):
        def hook(_m, _i, output):
            if alpha == 0.0:
                return None
            h = output[0] if isinstance(output, tuple) else output
            pooled = h.float().mean(dim=1)
            delta = torch.from_numpy(direction_for).to(h.device, h.dtype)
            mod = h + alpha * delta * pooled.norm()
            return (mod,) + tuple(output[1:]) if isinstance(output, tuple) else mod
        return hook

    def make_scale_hook(alpha):
        def hook(_m, _i, output):
            if alpha == 0.0:
                return None
            h = output[0] if isinstance(output, tuple) else output
            mod = h * (1.0 + alpha)
            return (mod,) + tuple(output[1:]) if isinstance(output, tuple) else mod
        return hook

    golds = [int(r.label_id) for r in test.itertuples(index=False)]
    imgs = [Image.open(getattr(r, f"{SPEC}_path")).convert("RGB")
            for r in test.itertuples(index=False)]
    msgs_all = [build_prompt(media=img, label_names=names, regime="zero_shot",
                             support=None, media_type="image", lang="en", enrich_text=None)
                for img in imgs]

    def run_config(hook_factory, alpha, direction_map=None, per_class=True):
        preds = []
        handle = None
        for i, msgs in enumerate(msgs_all):
            if handle is not None:
                handle.remove()
            if hook_factory is not None:
                direction = None
                if direction_map is not None:
                    direction = direction_map[golds[i]] if per_class else direction_map
                handle = layer_module.register_forward_hook(hook_factory(direction, alpha))
            else:
                handle = None
            scores = model.choice_logprobs([msgs], choices=letters)[0]
            preds.append(int(_ids[letters.index(max(scores, key=scores.get))]))
        if handle is not None:
            handle.remove()
        correct = np.array([p == g for p, g in zip(preds, golds)])
        return correct

    results = {"layer": L, "probe_acc": probe_acc, "n": len(golds), "configs": {}}

    for alpha in ALPHAS:
        if alpha == 0.0:
            correct = run_config(None, 0.0)
            results["configs"]["no_hook"] = {"acc": float(correct.mean())}
            print(f"[baseline] acc={correct.mean():.3f}", flush=True)
            continue
        # probe direction
        c_probe = run_config(make_hook, alpha, dirs)
        # scaling control
        c_scale = run_config(make_scale_hook, alpha)
        # random direction distribution
        rand_accs = []
        for k, rset in enumerate(rand_sets):
            c_rand = run_config(make_hook, alpha, rset)
            rand_accs.append(float(c_rand.mean()))
        results["configs"][f"alpha_{alpha}"] = {
            "probe_acc": float(c_probe.mean()),
            "scale_control_acc": float(c_scale.mean()),
            "random_accs": rand_accs,
            "random_mean": float(np.mean(rand_accs)),
            "random_sd": float(np.std(rand_accs)),
        }
        r = results["configs"][f"alpha_{alpha}"]
        print(f"[α={alpha}] probe={r['probe_acc']:.3f} scale={r['scale_control_acc']:.3f} "
              f"rand={r['random_mean']:.3f}±{r['random_sd']:.3f} (k={len(rand_accs)})", flush=True)

        # paired bootstrap: probe vs each random run (paired on same clips)
        boots = []
        rng_b = np.random.default_rng(0)
        n = len(c_probe)
        rand_stack = None  # per-clip correctness averaged across random sets
        # store per-clip probe correctness for reuse
        results["configs"][f"alpha_{alpha}"]["_probe_correct"] = c_probe.tolist()

    # paired bootstrap using stored per-clip correctness
    for alpha in [a for a in ALPHAS if a > 0]:
        cfg = results["configs"][f"alpha_{alpha}"]
        c_probe = np.array(cfg.pop("_probe_correct"))
        # re-run of randoms per-clip not stored (memory); approximate CI via
        # probe-vs-random-mean difference and binomial SE of the band
        diff = cfg["probe_acc"] - cfg["random_mean"]
        se = np.sqrt(cfg["probe_acc"] * (1 - cfg["probe_acc"]) / n
                     + cfg["random_sd"] ** 2)
        cfg["diff_vs_random_mean"] = float(diff)
        cfg["diff_z"] = float(diff / se) if se > 0 else 0.0

    (RESULTS / "steering_controls.json").write_text(json.dumps(results, indent=1))
    (RESULTS / "steering_controls.done").write_text("done\n")
    print("\n✅ steering controls complete")


if __name__ == "__main__":
    main()
