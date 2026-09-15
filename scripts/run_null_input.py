#!/usr/bin/env python
"""Null-input / Audio-Gain protocol (MMStar-style modality-use control).

Question: do the models actually USE the audio/spectrogram, or answer from the
prompt prior? For each (model, modality) arm we replace the real input with:
  audio arms : 30s digital silence | 30s white noise | NO media (prompt only)
  image arms : uniform grey spectrogram | random-noise spectrogram
and compare against the SAME clips' predictions from the original full runs.

Reported per arm (200-clip stratified subset):
  acc, prediction-class distribution, %clips keeping the ORIGINAL prediction,
  KL(real || null) of the answer distribution, dominant class.
Expected if collapse is prior-driven: acc ≈ chance, same dominant class,
~100% prediction retention, KL ≈ 0  →  "the model does not listen".
Outputs outputs/results/null_input/<arm>.json + SUMMARY.json + done marker.
"""
from __future__ import annotations

import argparse, gc, json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from splash.audio.io import read_segment
from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.models.factory import build_model
from splash.prompting.templates import LETTERS, _INSTR, _media_item, option_layout

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")
CKPT = RESULTS / ".ckpt"
PER_CLASS = 50
SEED = 7

# (arm, model, kwargs, spec, variant)  variant ∈ {silence,noise,nomedia,grey,pixnoise}
ARMS = [
    ("q2a.audio.silence",  "qwen2_audio",    {"device": "cuda:0"}, None,  "silence"),
    ("q2a.audio.noise",    "qwen2_audio",    {"device": "cuda:0"}, None,  "noise"),
    # (q2a nomedia dropped: its wrapper passes audio=[] to the processor, which
    #  chokes on an empty list — omni's wrapper omits the kwarg, so it covers
    #  the prompt-only arm)
    ("vl8b.mel.grey",      "qwen3_vl_8b",   {"device": "cuda:0", "device_map": None}, "mel", "grey"),
    ("vl8b.mel.pixnoise",  "qwen3_vl_8b",   {"device": "cuda:0", "device_map": None}, "mel", "pixnoise"),
    ("omni.audio.silence", "qwen3_omni_30b", {"device_map": "auto"}, None, "silence"),
    ("omni.audio.noise",   "qwen3_omni_30b", {"device_map": "auto"}, None, "noise"),
    ("omni.audio.nomedia", "qwen3_omni_30b", {"device_map": "auto"}, None, "nomedia"),
    ("omni.mel.grey",      "qwen3_omni_30b", {"device_map": "auto"}, "mel", "grey"),
    ("omni.mel.pixnoise",  "qwen3_omni_30b", {"device_map": "auto"}, "mel", "pixnoise"),
]
MODEL_PATHS = {"qwen2_audio": "/models/Qwen2-Audio-7B-Instruct",
               "qwen3_vl_8b": "/models/Qwen3-VL-8B-Instruct",
               "qwen3_omni_30b": "/models/Qwen3-Omni-30B-A3B-Instruct"}
ORIG_JSON = {"qwen2_audio+None": "lalm_prompting_qwen2_audio_deepship_zero_shot_shot0_clip30.0_langen_enrichnone",
             "qwen3_vl_8b+mel": "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone",
             "qwen3_omni_30b+None": "lalm_prompting_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_langen_enrichnone",
             "qwen3_omni_30b+mel": "vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone"}


def null_media(variant, clip_samples, size=(650, 390)):
    if variant == "silence":
        return np.zeros(clip_samples, dtype=np.float32)
    if variant == "noise":
        return (np.random.default_rng(0).standard_normal(clip_samples) * 0.1).astype(np.float32)
    if variant == "grey":
        return Image.new("RGB", size, (128, 128, 128))
    if variant == "pixnoise":
        rng = np.random.default_rng(1)
        return Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8))
    raise ValueError(variant)


def build_prompt(media, names, media_type, with_media=True):
    letters, _ids, options = option_layout(names)
    instr = _INSTR["en"].format(letters="/".join(letters), options=options, enrich="")
    content = [{"type": "text", "text": instr}]
    if with_media:
        content.append(_media_item(media_type, media))
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(a for a, *_ in ARMS))
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    names = class_names(DATASET, "en")
    letters, _ids, _ = option_layout(names)
    full = pd.read_csv(manifest)
    rng = np.random.default_rng(SEED)
    parts = [sub.sample(n=min(PER_CLASS, len(sub)), random_state=int(rng.integers(1 << 31)))
             for _, sub in full[full["split"] == "test"].groupby("label_id")]
    sub = pd.concat(parts).reset_index(drop=True)
    print(f"[null-input] subset {len(sub)} clips ({PER_CLASS}/类)")
    clip_samples = int(CLIP_LEN * 16000)

    import torch
    summary = []
    cur, cur_name = None, None
    for arm, model_name, kwargs, spec, variant in ARMS:
        if arm not in args.arms.split(","):
            continue
        if cur_name != model_name:
            if cur is not None:
                del cur; gc.collect(); torch.cuda.empty_cache()
            cur = build_model({"name": model_name, "path": MODEL_PATHS[model_name], **kwargs})
            cur_name = model_name
        media_type = "audio" if spec is None else "image"
        ckpt_file = CKPT / f"null_input_{arm}.jsonl"
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if ckpt_file.exists():
            done = {(json.loads(l)["recording_id"], json.loads(l)["clip_idx"])
                    for l in ckpt_file.read_text().splitlines() if l.strip()}
        for row in sub.itertuples(index=False):
            if (row.recording_id, int(row.clip_idx)) in done:
                continue
            media = None if variant == "nomedia" else null_media(variant, clip_samples)
            msgs = build_prompt(media, names, media_type, with_media=(variant != "nomedia"))
            scores = cur.choice_logprobs([msgs], choices=letters)[0]
            pred_letter = max(scores, key=scores.get)
            lid = int(LETTERS.index(pred_letter))  # canonical order = sorted ids
            rec = {"recording_id": row.recording_id, "clip_idx": int(row.clip_idx),
                   "label_id": int(row.label_id), "pred_letter": pred_letter,
                   "pred_label_id": lid, "choice_logprobs": scores}
            with open(ckpt_file, "a") as f:
                f.write(json.dumps(rec) + "\n")
        recs = [json.loads(l) for l in ckpt_file.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(recs)
        orig = pd.DataFrame(json.loads(
            (RESULTS / f"{ORIG_JSON[model_name + '+' + str(spec)]}.json").read_text())["predictions"]
        ).set_index(["recording_id", "clip_idx"]).loc[
            [(r["recording_id"], r["clip_idx"]) for r in recs]]
        dist = df.pred_label_id.value_counts(normalize=True)
        def _softmax_mean():
            P = []
            for r in recs:
                lp = list(r["choice_logprobs"].values()); m = max(lp)
                p = np.exp(np.array(lp) - m); P.append(p / p.sum())
            return np.mean(P, axis=0)
        p_null = _softmax_mean()
        lp0 = [list(o.values()) for o in orig.loc[[(r['recording_id'], r['clip_idx']) for r in recs]]["choice_logprobs"]]
        p_real = None
        if lp0 and isinstance(lp0[0], list):
            P = []
            for l in lp0:
                m = max(l); p = np.exp(np.array(l) - m); P.append(p / p.sum())
            p_real = np.mean(P, axis=0)
        kl = float(np.sum(p_real * np.log(np.clip(p_real, 1e-12, 1) / np.clip(p_null, 1e-12, 1)))) if p_real is not None else float("nan")
        keep = float((df.pred_label_id.to_numpy() == orig.pred_label_id.to_numpy()).mean())
        acc = float((df.pred_label_id == df.label_id).mean())
        dom = int(dist.idxmax())
        names_l = class_names(DATASET, "en")
        row_s = {"arm": arm, "n": len(df), "acc": acc, "dominant": names_l[dom],
                 "dominant_frac": float(dist.iloc[0]), "keep_orig_pred": keep, "KL(real||null)": kl}
        summary.append(row_s)
        print(f"[{arm}] acc={acc:.3f} dom={names_l[dom]}({dist.iloc[0]:.2f}) "
              f"keep={keep:.3f} KL={kl:.4f}")

    (RESULTS / "null_input").mkdir(exist_ok=True)
    (RESULTS / "null_input" / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    if args.arms == ",".join(a for a, *_ in ARMS):
        (RESULTS / "null_input.done").write_text("done\n")
    print("\n✅ null-input protocol complete")


if __name__ == "__main__":
    main()
