#!/usr/bin/env python
"""Option-permutation protocol: is the zero-shot class COLLAPSE positional or
semantic? (upgrade of the reversal ablation; protocol after "Hearing the Order",
arXiv 2510.00628, and Zheng et al., ICLR 2024)

Per-sample analysis showed every zero-shot run collapses to ~one class
(Qwen2/VL → Oil tanker, Omni → Tug). We re-run key zero-shot conditions on a
stratified 100-clip subset under 8 option permutations (identity, reversal, 6
seeded random). Reported per condition:

  * per-permutation accuracy + PERMUTATION-AGGREGATED accuracy (bias-corrected)
  * per-position selection frequency (selection-bias profile)
  * CLASS-FLIP RATE: fraction of clips whose predicted CLASS (not letter)
    changes across permutations — a listening model flips; a collapsed model
    tracks either one position or one class regardless of order.

Conditions: vl8b.mel / vl8b.demon / omni.audio / omni.mel.
Per-clip checkpointed; writes outputs/results/option_order/{key}_{pi}.json,
SUMMARY.json, and (full default set only) option_order.done.
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
from splash.prompting.templates import LETTERS, _INSTR, _media_item
from splash.metrics.classification import classification_metrics

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")
CKPT = RESULTS / ".ckpt"
PER_CLASS = 25
SEED = 42
N_PERMS = 8

CONDITIONS = [
    ("vl8b.mel",    "qwen3_vl_8b",    {"device": "cuda:0", "device_map": None}, "mel"),
    ("vl8b.demon",  "qwen3_vl_8b",    {"device": "cuda:0", "device_map": None}, "demon"),
    ("omni.audio",  "qwen3_omni_30b", {"device_map": "auto"}, None),
    ("omni.mel",    "qwen3_omni_30b", {"device_map": "auto"}, "mel"),
]
MODEL_PATHS = {"qwen3_vl_8b": "/models/Qwen3-VL-8B-Instruct",
               "qwen3_omni_30b": "/models/Qwen3-Omni-30B-A3B-Instruct"}
ORIG_JSON = {
    "vl8b.mel":   "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone",
    "vl8b.demon": "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specdemon_langen_enrichnone",
    "omni.audio": "lalm_prompting_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_langen_enrichnone",
    "omni.mel":   "vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone",
}


def permutations(ids: list[int]) -> list[list[int]]:
    """identity, reversal, then seeded random permutations (duplicates dropped).

    All elements cast to plain int — np.int64 would break json.dumps.
    """
    perms = [[int(i) for i in ids], [int(i) for i in ids][::-1]]
    rng = np.random.default_rng(SEED)
    while len(perms) < N_PERMS:
        p = [int(x) for x in rng.permutation(ids)]
        if p not in perms:
            perms.append(p)
    return perms


def stratified_subset(manifest: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    parts = []
    for lid, sub in manifest[manifest["split"] == "test"].groupby("label_id"):
        parts.append(sub.sample(n=min(PER_CLASS, len(sub)), random_state=int(rng.integers(1 << 31))))
    return pd.concat(parts).reset_index(drop=True)


def build_perm_prompt(media, names, perm_order, media_type):
    letters = LETTERS[:len(perm_order)]
    options = "\n".join(f"{l}. {names[i]}" for l, i in zip(letters, perm_order))
    instr = _INSTR["en"].format(letters="/".join(letters), options=options, enrich="")
    return [{"role": "user", "content": [{"type": "text", "text": instr},
                                         _media_item(media_type, media)]}]


def run_perm(model, rows, spec, names, perm_order, exp_name):
    letters = LETTERS[:len(perm_order)]
    letter_idx = {l: i for i, l in enumerate(letters)}
    ckpt_file = CKPT / f"{exp_name}.jsonl"
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if ckpt_file.exists():
        for line in ckpt_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["recording_id"], r["clip_idx"]))
    clip_samples = int(CLIP_LEN * 16000)
    for row in rows.itertuples(index=False):
        if (row.recording_id, int(row.clip_idx)) in done:
            continue
        media = (read_segment(row.path, row.start, row.end, target_sr=16000,
                              pad_to_samples=clip_samples) if spec is None
                 else Image.open(getattr(row, f"{spec}_path")).convert("RGB"))
        msgs = build_perm_prompt(media, names, perm_order, "audio" if spec is None else "image")
        scores = model.choice_logprobs([msgs], choices=letters)[0]
        pred_letter = max(scores, key=scores.get)
        rec = {"recording_id": row.recording_id, "clip_idx": int(row.clip_idx),
               "label_id": int(row.label_id), "pred_letter": pred_letter,
               "pred_label_id": int(perm_order[letter_idx[pred_letter]]),
               "choice_logprobs": scores,
               "letter_to_label": {l: int(perm_order[i]) for i, l in enumerate(letters)}}
        with open(ckpt_file, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return [json.loads(l) for l in ckpt_file.read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default=",".join(k for k, *_ in CONDITIONS))
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    names = class_names(DATASET, "en")
    ids = sorted(names)
    perms = permutations(ids)
    print(f"[optorder] permutations: {perms}")

    src = pd.read_csv(aug if aug.exists() else manifest)
    sub = stratified_subset(pd.read_csv(manifest))
    rows = sub.merge(src, on=["recording_id", "clip_idx"], suffixes=("", "_y"))
    print(f"[optorder] subset: {len(rows)} clips ({PER_CLASS}/类, seed={SEED})")

    import torch
    summary = []
    cur, cur_name = None, None
    for key, model_name, kwargs, spec in CONDITIONS:
        if key not in args.conditions.split(","):
            continue
        if cur_name != model_name:
            if cur is not None:
                del cur; gc.collect(); torch.cuda.empty_cache()
            cur = build_model({"name": model_name, "path": MODEL_PATHS[model_name], **kwargs})
            cur_name = model_name
        per_perm = {}
        for pi, perm in enumerate(perms):
            exp = f"option_order_{key}_p{pi}"
            recs = run_perm(cur, rows, spec, names, perm, exp)
            df = pd.DataFrame(recs)
            acc = float((df.pred_label_id == df.label_id).mean())
            per_perm[pi] = df.set_index(["recording_id", "clip_idx"]).sort_index()
            m = classification_metrics(df.label_id.tolist(), df.pred_label_id.tolist(), ids, names)
            (RESULTS / "option_order").mkdir(exist_ok=True)
            (RESULTS / "option_order" / f"{key}_p{pi}.json").write_text(json.dumps(
                {"experiment": exp, "permutation": [int(x) for x in perm], "n": len(df),
                 "acc": acc, "metrics_clip": {k: v for k, v in m.items()},
                 "predictions": recs}, indent=1))
            print(f"[{key} p{pi} {perm}] acc={acc:.3f}")
        # aggregate: class-flip rate + position profile + perm-mean acc
        mats = pd.concat({pi: d.pred_label_id for pi, d in per_perm.items()}, axis=1)
        letters_mat = pd.concat({pi: d.pred_letter for pi, d in per_perm.items()}, axis=1)
        n_classes_per_clip = mats.nunique(axis=1)
        flip = float((n_classes_per_clip > 1).mean())
        pos_track = float(letters_mat.nunique(axis=1).eq(1).mean())   # same letter under every perm
        accs = [float((d.pred_label_id == d.label_id).mean()) for d in per_perm.values()]
        orig = pd.DataFrame(json.loads((RESULTS / f"{ORIG_JSON[key]}.json").read_text())["predictions"]
                            ).set_index(["recording_id", "clip_idx"]).sort_index().loc[list(mats.index)]
        s = {"key": key, "perm_accs": [round(a, 3) for a in accs],
             "perm_mean_acc": float(np.mean(accs)), "acc_range": [min(accs), max(accs)],
             "class_flip_rate": flip, "position_anchored_frac": pos_track,
             "orig_acc_same_clips": float((orig.pred_label_id == orig.label_id).mean()),
             "dominant_class": names[int(mats.stack().mode().iloc[0])],
             "dominant_frac": float((mats == mats.stack().mode().iloc[0]).mean().mean())}
        summary.append(s)
        print(f"  → [{key}] flip={flip:.3f} pos-anchored={pos_track:.3f} "
              f"perm-mean={s['perm_mean_acc']:.3f} dom={s['dominant_class']}({s['dominant_frac']:.2f})")

    (RESULTS / "option_order").mkdir(exist_ok=True)
    (RESULTS / "option_order" / "SUMMARY.json").write_text(json.dumps(summary, indent=1))
    if args.conditions == ",".join(k for k, *_ in CONDITIONS):
        (RESULTS / "option_order.done").write_text("done\n")
    print("\n✅ option-permutation protocol complete")


if __name__ == "__main__":
    main()
