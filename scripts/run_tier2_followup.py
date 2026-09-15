#!/usr/bin/env python
"""Tier-2 followup: Qwen3-VL-32B few-shot (mel/stft/demon × k5) on DeepShip.

Completes the VL-32B matrix (zero-shot rows already in results.csv). Must run
AFTER the Omni campaign frees both GPUs — 32B bf16 ≈ 64 GB → device_map=auto
across both L20s. Per-clip checkpointing + results.csv skip make it crash-safe;
launched by scripts/ops/run_tier2_watcher.sh when tier2_omni.done appears.

Writes outputs/results/tier2_vl32b_few.done when all 3 conditions are saved.
"""
from __future__ import annotations

import argparse, gc
from pathlib import Path

import torch

from splash.eval.harness import find_manifest
from splash.eval.prompting_runner import run_prompting_eval
from splash.models.factory import build_model
from splash.tracking.results import experiment_name, save_result

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")


def cfg_for(spec, regime, shot):
    return {"eval": {"name": "vlm_spectrogram", "max_new_tokens": 8, "spectrogram": spec},
            "model": {"name": "qwen3_vl_32b"},
            "data": {"name": DATASET, "clip_len": CLIP_LEN},
            "regime": {"name": regime, "shot": shot}, "lang": "en", "enrich": "none"}


def already_done(cfg, expected_n):
    csv = RESULTS / "results.csv"
    if not csv.exists():
        return False
    import pandas as pd
    df = pd.read_csv(csv)
    row = df[df["experiment"] == experiment_name(cfg)]
    if row.empty:
        return False
    if expected_n and not pd.isna(row.iloc[0].get("n_clips", float("nan"))):
        if int(row.iloc[0]["n_clips"]) != int(expected_n):
            return False
    return True


def run_one(model, aug, spec, regime, shot, expected_n):
    cfg = cfg_for(spec, regime, shot)
    exp = experiment_name(cfg)
    if already_done(cfg, expected_n):
        print(f"[skip] {exp}"); return
    print(f"\n=== {exp} ===")
    res = run_prompting_eval(
        model, aug, dataset_name=DATASET, regime=regime, shot=shot, target_sr=16000,
        clip_len=CLIP_LEN, lang="en", media_type="image", spec_mode=spec,
        max_recordings=None, max_new_tokens=8, experiment=exp)
    save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
    print(f"[saved] {exp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--specs", default=None, help="filter specs (e.g. 'mel,demon')")
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    import pandas as pd
    expected_n = int((pd.read_csv(manifest)["split"] == "test").sum())
    specs = args.specs.split(",") if args.specs else ["mel", "stft", "demon"]
    print(f"[tier2-followup] specs={specs} test_clips={expected_n} aug={'OK' if aug.exists() else 'MISSING'}")

    print("\n加载 Qwen3-VL-32B (device_map=auto, ~64GB)...")
    vl = build_model({"name": "qwen3_vl_32b", "path": "/models/Qwen3-VL-32B-Instruct", "device_map": "auto"})
    print(f"loaded; device={vl.device}; caps={vl.capabilities}")

    for spec in specs:
        run_one(vl, aug, spec, "few_shot", args.shots, expected_n)

    del vl; gc.collect(); torch.cuda.empty_cache()
    # stage marker only for the full default spec set (no --specs filter)
    if not args.specs:
        (RESULTS / "tier2_vl32b_few.done").write_text("done\n")
    print("\n✅ VL-32B few-shot followup complete")


if __name__ == "__main__":
    main()
