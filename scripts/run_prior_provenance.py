#!/usr/bin/env python
"""Prior-provenance probes: WHERE does the collapsed answer come from?

Two text-level probes per model (q2a uses a 30s digital-silence carrier because
its processor requires an audio input; vl8b/omni run text-only):

  A. free-text (10 paraphrases): "Which vessel type is most frequently recorded
     in underwater audio?" etc. → generated answers; count which class (if any)
     each model names as most common.
  B. text-only MCQ letter distribution: the standard 4-way question with NO
     audio/spectrogram, ~20 paraphrase/ordering variants → the pure question-
     form prior over letters.

If a model's text-level prior matches its audio-collapse class, the collapse is
a language-level prior; if not, it arises from audio-token interaction.
Outputs outputs/results/prior_provenance.json + .done.
"""
from __future__ import annotations

import gc, json
from pathlib import Path

import numpy as np

from splash.data.labels import class_names
from splash.models.factory import build_model
from splash.prompting.templates import LETTERS, _INSTR, _media_item, option_layout

RESULTS = Path("outputs/results")
DATASET = "deepship"

QUESTIONS = [
    "Which vessel type is most frequently recorded in underwater audio?",
    "In underwater acoustic recordings, what is the most common type of ship?",
    "If I record random ocean traffic audio, which ship type am I most likely to capture?",
    "Rank these vessel types by how often they appear in underwater recordings.",
    "Which of these ships is the most typical source of underwater noise?",
    "A hydrophone records a random vessel. What type is it most likely to be?",
    "Which vessel class dominates underwater acoustic datasets?",
    "What kind of ship produces the background noise in most ocean recordings?",
    "You hear an unidentifiable distant vessel. What is your best guess for its type?",
    "Which ship type would a random underwater recording most likely contain?",
]

MODEL_PATHS = {"qwen2_audio": ("/models/Qwen2-Audio-7B-Instruct", {"device": "cuda:0"}),
               "qwen3_vl_8b": ("/models/Qwen3-VL-8B-Instruct", {"device": "cuda:0", "device_map": None}),
               "qwen3_omni_30b": ("/models/Qwen3-Omni-30B-A3B-Instruct", {"device_map": "auto"})}


def mcq_variants(names):
    """Standard MCQ prompt, no media, under 4 option orders × 5 wording tweaks."""
    ids = sorted(names)
    orders = [ids, ids[::-1], [ids[2], ids[0], ids[3], ids[1]], [ids[1], ids[3], ids[0], ids[2]]]
    tweaks = ["", "Answer with only the letter.", "Give your single best guess.",
              "Even if unsure, you must choose.", "What is your prior belief?"]
    variants = []
    for oi, order in enumerate(orders):
        letters = LETTERS[:len(order)]
        options = "\n".join(f"{l}. {names[i]}" for l, i in zip(letters, order))
        for ti, tw in enumerate(tweaks):
            instr = (_INSTR["en"].format(letters="/".join(letters), options=options, enrich="")
                     + (" " + tw if tw else ""))
            variants.append({"id": f"o{oi}t{ti}", "order": order, "instr": instr})
    return variants


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="override device for single-GPU models")
    args = ap.parse_args()
    names = class_names(DATASET, "en")
    letters_canon, _ids, _ = option_layout(names)
    silence = np.zeros(480000, dtype=np.float32)
    out = {}

    import torch
    paths = dict(MODEL_PATHS)
    if args.device:
        paths = {k: (p, ({**kw, "device": args.device} if "device" in kw else kw))
                 for k, (p, kw) in paths.items()}
    for model_name, (path, kwargs) in paths.items():
        model = build_model({"name": model_name, "path": path, **kwargs})
        is_audio = model_name == "qwen2_audio"

        # A. free-text
        gens = []
        for q in QUESTIONS:
            content = [{"type": "text", "text": q + " Answer in one short sentence."}]
            if is_audio:
                content.append(_media_item("audio", silence))
            msgs = [{"role": "user", "content": content}]
            gens.append(model.generate([msgs], max_new_tokens=48)[0])
        out[model_name] = {"free_text": {q: g for q, g in zip(QUESTIONS, gens)}}

        # B. text-only MCQ letter distribution
        scores_by_variant = {}
        for v in mcq_variants(names):
            vl = LETTERS[:len(v["order"])]
            content = [{"type": "text", "text": v["instr"]}]
            if is_audio:
                content.append(_media_item("audio", silence))
            msgs = [{"role": "user", "content": content}]
            scores = model.choice_logprobs([msgs], choices=vl)[0]
            best = max(scores, key=scores.get)
            scores_by_variant[v["id"]] = {
                "pred_letter": best,
                "pred_label": names[v["order"][vl.index(best)]],
                "order": v["order"]}
        # aggregate: fraction of variants naming each class
        from collections import Counter
        cls_counts = Counter(s["pred_label"] for s in scores_by_variant.values())
        let_counts = Counter(s["pred_letter"] for s in scores_by_variant.values())
        out[model_name]["mcq_no_media"] = {
            "class_dist": dict(cls_counts), "letter_dist": dict(let_counts),
            "variants": scores_by_variant}
        print(f"[{model_name}] text-MCQ class dist: {dict(cls_counts)} | letters: {dict(let_counts)}")
        print(f"[{model_name}] free-text sample: {gens[0][:120]}")
        del model; gc.collect(); torch.cuda.empty_cache()

    (RESULTS / "prior_provenance.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    (RESULTS / "prior_provenance.done").write_text("done\n")
    print("\n✅ prior provenance complete")


if __name__ == "__main__":
    main()
