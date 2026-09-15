#!/usr/bin/env python
"""Class-name log-likelihood scoring — the missing non-MCQ generative baseline.

Reviewer attack this closes: "the standard way to classify with an LLM without
fine-tuning is log-likelihood scoring of class names; your MCQ-letter collapse
may be an artifact of the multiple-choice interface."

Protocol (audio pathway, full 2,685-clip DeepShip test split):
  prompt  = free-form instruction, NO option list, NO letters
  score_i = choice_logprobs(prompt + audio, choices=4 class names)
  pred    = argmax_i score_i
Both raw-sum and length-normalized (per-token mean) argmax are reported —
multi-token name sums are length-biased.

Models: qwen2_audio (LALM), qwen3_omni_30b (omni, audio pathway).
Expected outcomes: ~0.25 seals generative-interface failure; ~0.5+ reframes it
as MCQ-specific (still a diagnosis consistent with probes). Either helps.

Resume-safe: per-clip jsonl checkpoint; done pairs skipped on relaunch.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from splash.audio.io import read_segment
from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.models.factory import build_model

DATASET, CLIP_LEN = "deepship", 30.0
RESULTS = Path("outputs/results/ll_scoring")
CLIP_SAMPLES = int(CLIP_LEN * 16000)

INSTR = (
    "You are an underwater acoustics expert. Listen to the recording and "
    "identify the type of ship making the sound. Answer with ONLY the ship "
    "type name."
)

MODELS = {
    "qwen2_audio":   {"name": "qwen2_audio", "path": "/models/Qwen2-Audio-7B-Instruct"},
    "qwen3_omni_30b": {"name": "qwen3_omni_30b", "path": "/models/Qwen3-Omni-30B-A3B-Instruct",
                        "device_map": "auto"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2_audio,qwen3_omni_30b")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    names = class_names(DATASET, "en")               # {id: name}
    label_by_name = {v: k for k, v in names.items()}
    choices = [names[i] for i in sorted(names)]
    n_toks = None  # filled per model from tokenizer

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = pd.read_csv(manifest.with_name(f"{manifest.stem}_spec.csv"))
    test = aug[aug["split"] == "test"].reset_index(drop=True)

    for key in args.models.split(","):
        cfg = MODELS[key]
        ckpt = RESULTS / f"{key}.jsonl"
        done = set()
        if ckpt.exists():
            for line in ckpt.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done.add((r["recording_id"], r["clip_idx"]))
        todo = test[~test.apply(lambda r: (r["recording_id"], r["clip_idx"]) in done,
                                axis=1)].reset_index(drop=True)
        print(f"[{key}] {len(done)} done, {len(todo)} to go", flush=True)
        if not todo.empty:
            model = build_model({**{k: v for k, v in cfg.items() if k != "device_map"},
                                 "device": args.device,
                                 **({"device_map": cfg["device_map"]} if "device_map" in cfg else {})})
            tok = model.processor.tokenizer
            n_toks = {c: len(tok.encode(" " + c, add_special_tokens=False)) for c in choices}
            with ckpt.open("a") as f:
                for i, row in todo.iterrows():
                    audio = read_segment(row["path"], row["start"], row["end"],
                                         target_sr=16000, pad_to_samples=CLIP_SAMPLES)
                    msgs = [{"role": "user", "content": [
                        {"type": "text", "text": INSTR},
                        {"type": "audio", "audio": audio}]}]
                    scores = model.choice_logprobs([msgs], choices)[0]
                    rec = {"recording_id": row["recording_id"], "clip_idx": row["clip_idx"],
                           "label_id": int(row["label_id"]),
                           "scores": scores,
                           "pred_raw": max(scores, key=scores.get),
                           "pred_norm": max(scores, key=lambda c: scores[c] / n_toks[c])}
                    f.write(json.dumps(rec) + "\n")
                    if (i + 1) % 100 == 0:
                        f.flush()
                        print(f"[{key}] {i+1}/{len(todo)}", flush=True)
            del model
            import gc, torch
            gc.collect(); torch.cuda.empty_cache()

        # ---- metrics ----
        recs = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
        y = [r["label_id"] for r in recs]
        acc_raw = float(np.mean([label_by_name[r["pred_raw"]] == g for r, g in zip(recs, y)]))
        acc_norm = float(np.mean([label_by_name[r["pred_norm"]] == g for r, g in zip(recs, y)]))
        from collections import Counter
        dist_raw = Counter(label_by_name[r["pred_raw"]] for r in recs)
        out = {"model": key, "n": len(recs), "acc_raw_sum": acc_raw,
               "acc_len_normalized": acc_norm,
               "pred_distribution": {str(k): v for k, v in sorted(dist_raw.items())},
               "chance": 0.25}
        (RESULTS / f"{key}_metrics.json").write_text(json.dumps(out, indent=1))
        print(f"[{key}] LL-scoring raw={acc_raw:.4f} len-norm={acc_norm:.4f}", flush=True)

    (RESULTS.parent / "ll_scoring.done").write_text("done\n")
    print("\n✅ LL-scoring complete")


if __name__ == "__main__":
    main()
