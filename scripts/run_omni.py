#!/usr/bin/env python
"""Omni full experiment matrix (Tier-2A): same-backbone 看 vs 听 + fusion.

Qwen3-Omni-30B (full-modal MoE, 262k ctx) — the ONLY model that processes both
audio AND images. Three experiment groups:

  A (single modality, 8 conditions):
    audio / mel / stft / demon × zero-shot / few-shot(k5)
    → same-backbone "听 vs 看" comparison + spectrogram-type ablation

  B (audio+X fusion zero-shot, 3 conditions):
    audio+mel / audio+stft / audio+demon
    → does multi-modal fusion beat single modality? (Omni-unique experiment)

  D (fusion few-shot k5, 3 conditions):
    audio+mel / audio+stft / audio+demon × few-shot

  Total: 14 conditions (en) + optional zh subset (--lang zh)

Usage:
  python scripts/run_omni.py                           # all 14 conditions, en
  python scripts/run_omni.py --phase A                 # single-modality only
  python scripts/run_omni.py --phase B                 # fusion zero-shot only
  python scripts/run_omni.py --lang zh --phase A --skip-few-shot  # zh ablation subset
"""
from __future__ import annotations

import argparse, gc
from pathlib import Path
import torch

from splash.eval.harness import find_manifest
from splash.eval.prompting_runner import run_prompting_eval, run_fusion_eval
from splash.models.factory import build_model
from splash.tracking.results import experiment_name, save_result

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")
SPECS = ["audio", "mel", "stft", "demon"]
FUSION_SPECS = ["audio+mel", "audio+stft", "audio+demon", "audio+mel+stft+demon"]


def cfg_for(eval_name, regime, shot, spec, lang):
    e = {"name": eval_name, "max_new_tokens": 8}
    if spec and spec != "audio":
        e["spectrogram"] = spec
    return {"eval": e, "model": {"name": "qwen3_omni_30b"},
            "data": {"name": DATASET, "clip_len": CLIP_LEN},
            "regime": {"name": regime, "shot": shot}, "lang": lang, "enrich": "none"}


def cfg_for_fusion(fusion_spec, regime, shot, lang):
    return {"eval": {"name": "fusion", "fusion": fusion_spec, "max_new_tokens": 8},
            "model": {"name": "qwen3_omni_30b"},
            "data": {"name": DATASET, "clip_len": CLIP_LEN},
            "regime": {"name": regime, "shot": shot}, "lang": lang, "enrich": "none"}


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


def run_one(model, manifest, spec, regime, shot, expected_n, lang):
    """Single-modality condition (audio or spectrogram image)."""
    media_type = "audio" if spec == "audio" else "image"
    eval_name = "lalm_prompting" if spec == "audio" else "vlm_spectrogram"
    cfg = cfg_for(eval_name, regime, shot, spec, lang)
    exp = experiment_name(cfg)
    if already_done(cfg, expected_n):
        print(f"[skip] {exp}"); return
    print(f"\n=== {exp} ===")
    res = run_prompting_eval(
        model, manifest, dataset_name=DATASET, regime=regime, shot=shot, target_sr=16000,
        clip_len=CLIP_LEN, lang=lang, media_type=media_type, spec_mode=(spec if spec != "audio" else None),
        max_recordings=None, max_new_tokens=8, experiment=exp)
    save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
    print(f"[saved] {exp}")


def run_fusion_one(model, aug, fusion_spec, regime, shot, expected_n, lang):
    """Fusion condition (audio+image in one prompt)."""
    cfg = cfg_for_fusion(fusion_spec, regime, shot, lang)
    exp = experiment_name(cfg)
    if already_done(cfg, expected_n):
        print(f"[skip] {exp}"); return
    print(f"\n=== {exp} ===")
    res = run_fusion_eval(
        model, aug, fusion_spec=fusion_spec, dataset_name=DATASET, regime=regime, shot=shot,
        target_sr=16000, clip_len=CLIP_LEN, lang=lang, max_recordings=None,
        max_new_tokens=8, experiment=exp)
    save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
    print(f"[saved] {exp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="all", help="A=single, B=fusion-zero, D=fusion-few, all=A+B+D")
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--skip-few-shot", action="store_true")
    ap.add_argument("--specs", default=None, help="filter single-modality specs (e.g. 'audio,mel')")
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    import pandas as pd
    expected_n = int((pd.read_csv(manifest)["split"] == "test").sum())
    specs = args.specs.split(",") if args.specs else SPECS
    print(f"lang={args.lang} phase={args.phase} specs={specs} test_clips={expected_n} aug={'OK' if aug.exists() else 'MISSING'}")

    print("\n加载 Qwen3-Omni-30B (device_map=auto, ~60GB)...")
    omni = build_model({"name": "qwen3_omni_30b", "path": "/models/Qwen3-Omni-30B-A3B-Instruct", "device_map": "auto"})
    print(f"loaded; device={omni.device}; caps={omni.capabilities}")

    # ---- Phase A: single modality ----
    if args.phase in ("A", "all"):
        for spec in specs:
            run_one(omni, manifest if spec == "audio" else aug, spec, "zero_shot", 0, expected_n, args.lang)
            if not args.skip_few_shot and spec in ("audio", "mel"):
                run_one(omni, manifest if spec == "audio" else aug, spec, "few_shot", args.shots, expected_n, args.lang)
        if not args.skip_few_shot and ("stft" in specs or "demon" in specs):
            for spec in specs:
                if spec in ("stft", "demon"):
                    run_one(omni, aug, spec, "few_shot", args.shots, expected_n, args.lang)

    # ---- Phase B: fusion zero-shot ----
    if args.phase in ("B", "all"):
        for fspec in FUSION_SPECS:
            run_fusion_one(omni, aug, fspec, "zero_shot", 0, expected_n, args.lang)

    # ---- Phase D: fusion few-shot ----
    if args.phase in ("D", "all") and not args.skip_few_shot:
        for fspec in FUSION_SPECS:
            run_fusion_one(omni, aug, fspec, "few_shot", args.shots, expected_n, args.lang)

    del omni; gc.collect(); torch.cuda.empty_cache()
    (RESULTS / "tier2_omni.done").write_text("done\n")
    print(f"\n✅ Omni phase '{args.phase}' lang={args.lang} complete")


if __name__ == "__main__":
    main()
