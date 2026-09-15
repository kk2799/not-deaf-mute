#!/usr/bin/env python
"""ShipsEar cross-dataset validation: re-run the key DeepShip conditions.

Prompts at the native 12-class granularity; the 5-class and 9-class
granularities are derived post-hoc from the saved per-clip JSONs (mapping in
splash/data/shipsear_maps.py). Conditions (104 test clips each — cheap; the
real cost is the 4 model loads):

  qwen2_audio  : audio zero-shot + pairwise logprob-margin few-shot k1
  qwen3_vl_8b  : mel/stft/demon zero-shot + demon few-shot k5
  qwen3_vl_32b : mel/stft/demon zero-shot (matrix completion)
  qwen3_omni   : audio/mel/demon zero-shot + audio+demon fusion zero-shot

Crash-safe (per-clip checkpoints + results.csv skip). Launched by
scripts/ops/run_tier2_watcher.sh after the VL-32B followup finishes.
Writes outputs/results/shipsear.done.
"""
from __future__ import annotations

import argparse, gc
from pathlib import Path

import torch

from splash.eval.harness import find_manifest
from splash.eval.prompting_runner import run_prompting_eval, run_fusion_eval, run_similarity_few_shot
from splash.models.factory import build_model
from splash.tracking.results import experiment_name, save_result

DATASET = "shipsear"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")


def cfg_for(model_name, eval_name, regime, shot, spec=None):
    e = {"name": eval_name, "max_new_tokens": 8}
    if spec:
        e["spectrogram"] = spec
    return {"eval": e, "model": {"name": model_name},
            "data": {"name": DATASET, "clip_len": CLIP_LEN},
            "regime": {"name": regime, "shot": shot}, "lang": "en", "enrich": "none"}


def cfg_for_fusion(fusion_spec, regime, shot):
    return {"eval": {"name": "fusion", "fusion": fusion_spec, "max_new_tokens": 8},
            "model": {"name": "qwen3_omni_30b"},
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


def run_one(model, manifest, model_name, eval_name, media_type, spec, regime, shot, expected_n):
    cfg = cfg_for(model_name, eval_name, regime, shot, spec)
    exp = experiment_name(cfg)
    if already_done(cfg, expected_n):
        print(f"[skip] {exp}"); return
    print(f"\n=== {exp} ===")
    res = run_prompting_eval(
        model, manifest, dataset_name=DATASET, regime=regime, shot=shot, target_sr=16000,
        clip_len=CLIP_LEN, lang="en", media_type=media_type, spec_mode=spec,
        max_recordings=None, max_new_tokens=8, experiment=exp)
    save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
    print(f"[saved] {exp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", default="q2a,vl8b,vl32b,omni",
                    help="which model groups to run (comma list)")
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--device", default="cuda:0",
                    help="GPU for the single-GPU groups (q2a, vl8b)")
    args = ap.parse_args()
    groups = set(args.groups.split(","))

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    import pandas as pd
    expected_n = int((pd.read_csv(manifest)["split"] == "test").sum())
    print(f"[shipsear] groups={sorted(groups)} test_clips={expected_n} aug={'OK' if aug.exists() else 'MISSING'}")

    # ---- Qwen2-Audio (7B): audio zero-shot + pairwise k1 ----
    if "q2a" in groups:
        q2a = build_model({"name": "qwen2_audio", "path": "/models/Qwen2-Audio-7B-Instruct", "device": args.device})
        run_one(q2a, manifest, "qwen2_audio", "lalm_prompting", "audio", None, "zero_shot", 0, expected_n)
        sim_cfg = cfg_for("qwen2_audio", "lalm_prompting", "few_shot", 1, None)
        exp = experiment_name(sim_cfg)
        if not already_done(sim_cfg, expected_n):
            print(f"\n=== {exp} (pairwise logprob-margin) ===")
            res = run_similarity_few_shot(q2a, manifest, dataset_name=DATASET, shot=1,
                                          target_sr=16000, clip_len=CLIP_LEN, lang="en",
                                          experiment=exp)
            save_result(sim_cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
            print(f"[saved] {exp}")
        else:
            print(f"[skip] {exp}")
        del q2a; gc.collect(); torch.cuda.empty_cache()

    # ---- Qwen3-VL-8B: mel/stft/demon zero + demon few k5 ----
    if "vl8b" in groups:
        vl8 = build_model({"name": "qwen3_vl_8b", "path": "/models/Qwen3-VL-8B-Instruct",
                           "device": args.device, "device_map": None})
        for spec in ["mel", "stft", "demon"]:
            run_one(vl8, aug, "qwen3_vl_8b", "vlm_spectrogram", "image", spec, "zero_shot", 0, expected_n)
        run_one(vl8, aug, "qwen3_vl_8b", "vlm_spectrogram", "image", "demon", "few_shot", args.shots, expected_n)
        del vl8; gc.collect(); torch.cuda.empty_cache()

    # ---- Qwen3-VL-32B: mel/stft/demon zero (matrix completion) ----
    if "vl32b" in groups:
        vl32 = build_model({"name": "qwen3_vl_32b", "path": "/models/Qwen3-VL-32B-Instruct", "device_map": "auto"})
        for spec in ["mel", "stft", "demon"]:
            run_one(vl32, aug, "qwen3_vl_32b", "vlm_spectrogram", "image", spec, "zero_shot", 0, expected_n)
        del vl32; gc.collect(); torch.cuda.empty_cache()

    # ---- Qwen3-Omni: audio/mel/demon zero + audio+demon fusion zero ----
    if "omni" in groups:
        omni = build_model({"name": "qwen3_omni_30b", "path": "/models/Qwen3-Omni-30B-A3B-Instruct",
                            "device_map": "auto"})
        run_one(omni, manifest, "qwen3_omni_30b", "lalm_prompting", "audio", None, "zero_shot", 0, expected_n)
        for spec in ["mel", "stft", "demon"]:
            run_one(omni, aug, "qwen3_omni_30b", "vlm_spectrogram", "image", spec, "zero_shot", 0, expected_n)
        fcfg = cfg_for_fusion("audio+demon", "zero_shot", 0)
        fexp = experiment_name(fcfg)
        if not already_done(fcfg, expected_n):
            print(f"\n=== {fexp} ===")
            res = run_fusion_eval(omni, aug, fusion_spec="audio+demon", dataset_name=DATASET,
                                  regime="zero_shot", shot=0, target_sr=16000, clip_len=CLIP_LEN,
                                  lang="en", max_recordings=None, max_new_tokens=8, experiment=fexp)
            save_result(fcfg, res["metrics_clip"], res["metrics_recording"], res["records"])
            print(f"[saved] {fexp}")
        else:
            print(f"[skip] {fexp}")
        del omni; gc.collect(); torch.cuda.empty_cache()

    # stage marker only when ALL four model groups were requested — a --groups
    # subset must not make the watcher skip the rest (review finding)
    if groups == {"q2a", "vl8b", "vl32b", "omni"}:
        (RESULTS / "shipsear.done").write_text("done\n")
    print("\n✅ ShipsEar campaign complete")


if __name__ == "__main__":
    main()
