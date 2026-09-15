#!/usr/bin/env python
"""Tier-1 campaign: "看谱图 vs 听音频" on the FULL DeepShip test split.

Two independent routes (run in parallel on 2 GPUs by the watcher):
  --route audio  (cuda:0): Qwen2-Audio zero-shot + pairwise-logprob-margin few-shot
  --route vision (cuda:1): Qwen3-VL-8B on mel/stft/demon × zero/few (in-context)

Full test split by default (max_recordings=None → all 150 recordings / 2639 clips).
Per-clip checkpointing: each runner resumes from outputs/results/.ckpt/<exp>.jsonl
across crashes; conditions already in results.csv are skipped.
"""
from __future__ import annotations

import argparse, gc
from pathlib import Path

import torch

from splash.eval.harness import find_manifest
from splash.eval.prompting_runner import run_prompting_eval, run_similarity_few_shot
from splash.models.factory import build_model
from splash.tracking.results import experiment_name, save_result

DATASET = "deepship"
CLIP_LEN = 30.0
OVERLAP = 0.5
RESULTS = Path("outputs/results")


def cfg_for(model_name, eval_name, regime, shot, spec=None, max_recordings=None):
    e = {"name": eval_name, "max_new_tokens": 8}
    if spec:
        e["spectrogram"] = spec
    cfg = {"eval": e, "model": {"name": model_name}, "data": {"name": DATASET, "clip_len": CLIP_LEN},
           "regime": {"name": regime, "shot": shot}, "lang": "en", "enrich": "none"}
    if max_recordings is not None:
        cfg["max_recordings"] = max_recordings
    return cfg


def already_done(cfg, expected_n=None):
    """True only if experiment is in results.csv AND (when expected_n given) its
    n_clips matches the current test-set size — so a result saved on a stale
    (smaller) test set is treated as NOT done and re-run (reusing the checkpoint,
    so only new clips are computed)."""
    csv = RESULTS / "results.csv"
    if not csv.exists():
        return False
    import pandas as pd
    df = pd.read_csv(csv)
    row = df[df["experiment"] == experiment_name(cfg)]
    if row.empty:
        return False
    if expected_n is not None and "n_clips" in row and not pd.isna(row.iloc[0]["n_clips"]):
        if int(row.iloc[0]["n_clips"]) != int(expected_n):
            print(f"  [stale] {experiment_name(cfg)}: saved n_clips={int(row.iloc[0]['n_clips'])} ≠ current {expected_n} → 重跑(只补新增)")
            return False
    return True


def run_one(model, manifest, model_name, eval_name, media_type, spec, regime, shot,
            max_recordings, limit=None, expected_n=None):
    cfg = cfg_for(model_name, eval_name, regime, shot, spec, max_recordings)
    exp = experiment_name(cfg)
    if already_done(cfg, expected_n):
        print(f"[skip] {exp} (already in results.csv)"); return
    print(f"\n=== {exp} ===")
    res = run_prompting_eval(
        model, manifest, dataset_name=DATASET, regime=regime, shot=shot, target_sr=16000,
        clip_len=CLIP_LEN, lang="en", media_type=media_type, spec_mode=spec,
        max_recordings=max_recordings, limit=limit, max_new_tokens=8, experiment=exp)
    save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
    print(f"[saved] {exp}")


def route_audio(args, manifest, mr, lim, expected_n):
    audio_model = build_model({"name": "qwen2_audio", "path": "/models/Qwen2-Audio-7B-Instruct", "device": args.device})
    # zero-shot
    run_one(audio_model, manifest, "qwen2_audio", "lalm_prompting", "audio", None, "zero_shot", 0, mr, lim, expected_n)
    # pairwise-logprob-margin few-shot (in-context infeasible on Qwen2-Audio 8192 ctx)
    sim_shot = 1
    sim_cfg = cfg_for("qwen2_audio", "lalm_prompting", "few_shot", sim_shot, None, mr)
    exp = experiment_name(sim_cfg)
    if not already_done(sim_cfg, expected_n):
        print(f"\n=== {exp} (pairwise logprob-margin) ===")
        res = run_similarity_few_shot(audio_model, manifest, dataset_name=DATASET, shot=sim_shot,
                                      target_sr=16000, clip_len=CLIP_LEN, lang="en",
                                      max_recordings=mr, limit=lim, experiment=exp)
        save_result(sim_cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
        print(f"[saved] {exp}")
    else:
        print(f"[skip] {exp}")
    del audio_model; gc.collect(); torch.cuda.empty_cache()
    (RESULTS / "tier1_audio.done").write_text("done\n")


def _vision_all_done(expected_n):
    """True only if all 6 vision conditions (mel/stft/demon × zero/few) are in
    results.csv with the current test-set size — so a partial route (e.g. only
    demon on a second GPU) doesn't write tier1_vision.done prematurely."""
    csv = RESULTS / "results.csv"
    if not csv.exists():
        return False
    import pandas as _pd
    df = _pd.read_csv(csv)
    need = []
    for spec in ["mel", "stft", "demon"]:
        for regime, shot in [("zero_shot", 0), ("few_shot", 5)]:
            need.append(experiment_name(cfg_for("qwen3_vl_8b", "vlm_spectrogram", regime, shot, spec, None)))
    have = set(df["experiment"])
    for exp in need:
        if exp not in have:
            return False
        row = df[df["experiment"] == exp].iloc[0]
        if expected_n is not None and not _pd.isna(row.get("n_clips", float("nan"))) and int(row["n_clips"]) != int(expected_n):
            return False
    return True


def route_vision(args, aug, mr, lim, expected_n):
    vl = build_model({"name": "qwen3_vl_8b", "path": "/models/Qwen3-VL-8B-Instruct",
                      "device": args.device, "device_map": None})  # pin to one GPU
    specs = args.specs.split(",") if args.specs else ["mel", "stft", "demon"]
    for spec in specs:
        run_one(vl, aug, "qwen3_vl_8b", "vlm_spectrogram", "image", spec, "zero_shot", 0, mr, lim, expected_n)
        run_one(vl, aug, "qwen3_vl_8b", "vlm_spectrogram", "image", spec, "few_shot", args.shots, mr, lim, expected_n)
    del vl; gc.collect(); torch.cuda.empty_cache()
    # only mark vision done if ALL 6 conditions are present (supports parallel GPU split)
    if _vision_all_done(expected_n):
        (RESULTS / "tier1_vision.done").write_text("done\n")
    else:
        print(f"[route vision specs={specs}] finished its specs but not all 6 vision conditions done — not writing tier1_vision.done")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", choices=["audio", "vision", "all"], default="all")
    ap.add_argument("--device", default="cuda:0", help="GPU to pin the model to (e.g. cuda:0 / cuda:1)")
    ap.add_argument("--max-recordings", type=int, default=None, help="None=full test split (150 rec / 2639 clips)")
    ap.add_argument("--shots", type=int, default=5, help="vision in-context few-shot k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--specs", default=None, help="comma list for vision route (e.g. 'demon' or 'stft,demon')")
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": OVERLAP})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    import pandas as _pd
    expected_n = int((_pd.read_csv(manifest)["split"] == "test").sum())  # current test-set size (stale check)
    print(f"[route={args.route} device={args.device}] manifest={manifest.name} aug={'OK' if aug.exists() else 'MISSING'} mr={args.max_recordings} test_clips={expected_n}")
    mr, lim = args.max_recordings, args.limit

    if args.route in ("audio", "all"):
        route_audio(args, manifest, mr, lim, expected_n)
    if args.route in ("vision", "all"):
        if not aug.exists():
            print("augmented manifest missing — skip vision");
        else:
            route_vision(args, aug, mr, lim, expected_n)

    if args.route == "all":
        (RESULTS / "tier1.done").write_text("done\n")
    print(f"\n✅ route '{args.route}' complete")


if __name__ == "__main__":
    main()
