"""评测 A/B runner: zero-shot / few-shot prompting over a test split.

Media-agnostic: serves the audio route (评测 B, ``media_type="audio"``) and the
spectrogram-image route (评测 A, ``media_type="image"`` + ``spec_mode``). For the
image route the manifest must be the augmented one (carrying ``<mode>_path``
columns from ``generate_spectrograms.py``).

Iterates test instances, builds a per-instance prompt (with a shared few-shot
support set when ``regime="few_shot"``), scores option letters via
``choice_logprobs`` (and captures free-form ``generate`` text for error
analysis), then reports clip-level + recording-level metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from ..audio.io import read_segment
from ..data.labels import class_names
from ..data.splits import few_shot_indices
from ..metrics.aggregation import aggregate_predictions
from ..metrics.classification import classification_metrics, format_metrics


# --- per-clip checkpointing (resume long runs across crashes) ---
CKPT_DIR = Path("outputs/results/.ckpt")


def _ckpt_path(experiment: str | None) -> Path | None:
    return CKPT_DIR / f"{experiment}.jsonl" if experiment else None


def _load_ckpt(path: Path | None):
    """Return (records_so_far, done_keys). Each checkpoint line = one clip record."""
    records, done = [], set()
    if path and path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            records.append(r)
            done.add((r["recording_id"], r["clip_idx"]))
    return records, done


def _append_ckpt(path: Path | None, record: dict):
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
from ..prompting.answer_parse import parse_answer
from ..prompting.templates import LETTERS, build_pairwise, build_prompt, letter_to_label, option_layout


def run_prompting_eval(
    model,
    manifest,
    *,
    dataset_name: str = "deepship",
    regime: str = "zero_shot",
    shot: int = 5,
    support_seed: int = 0,
    target_sr: int = 16000,
    clip_len: float = 30.0,
    lang: str = "en",
    media_type: str = "audio",
    spec_mode: str | None = None,
    support_clip_len: float | None = None,  # few-shot support audio loaded at this length (fit context)
    limit: int | None = None,
    max_recordings: int | None = None,
    max_new_tokens: int = 8,
    eval_split: str = "test",
    experiment: str | None = None,
) -> dict:
    """Run a prompting evaluation → ``{metrics_clip, metrics_recording, records}``."""
    names = class_names(dataset_name, lang)
    letters, label_order, _options = option_layout(names)
    letter_to_label_map = {l: letter_to_label(l, label_order) for l in letters}

    full = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
    train_df = full[full["split"] == "train"].reset_index(drop=True)
    test_df = full[full["split"] == eval_split].reset_index(drop=True)

    # Stratified recording cap for speed (keeps class balance).
    if max_recordings is not None:
        per_class = max(1, max_recordings // test_df["label_id"].nunique())
        recs = test_df.drop_duplicates("recording_id")
        keep = []
        for _lid, sub in recs.groupby("label_id"):
            keep.extend(sub["recording_id"].head(per_class).tolist())
        test_df = test_df[test_df["recording_id"].isin(keep)].reset_index(drop=True)
    if limit:
        test_df = test_df.head(limit)

    clip_samples = int(round(clip_len * target_sr))

    def load_media(row) -> object:
        if media_type == "audio":
            return read_segment(row["path"], row["start"], row["end"],
                                target_sr=target_sr, pad_to_samples=clip_samples)
        col = f"{spec_mode}_path"
        if col not in row:
            raise KeyError(f"image route needs '{col}' in manifest; pass the augmented _spec manifest.")
        return Image.open(row[col]).convert("RGB")

    # Shared few-shot support set (sampled once, reused for every query).
    # Audio support is loaded SHORT (support_clip_len) so the few-shot prompt fits
    # small-context models (e.g. Qwen2-Audio 8192: 4×5s + 1×30s ≈ 4.8k tokens).
    support = None
    if regime == "few_shot":
        sup_idx = few_shot_indices(train_df, k=shot, seed=support_seed)
        sup_samples = int(round((support_clip_len or clip_len) * target_sr))
        sup_len = support_clip_len or clip_len

        def load_support(row):
            if media_type == "audio":
                return read_segment(row["path"], row["start"], row["start"] + sup_len,
                                    target_sr=target_sr, pad_to_samples=sup_samples)
            return load_media(row)  # images: full spectrogram (VLM context is large)

        support = [
            {"label_id": int(train_df.iloc[j]["label_id"]), media_type: load_support(train_df.iloc[j])}
            for j in sup_idx
        ]

    # per-clip checkpoint: resume across crashes (skip clips already scored)
    ckpt = _ckpt_path(experiment)
    ckpt_records, done = _load_ckpt(ckpt)
    new_records = []
    n_skip = 0
    iterator = tqdm(test_df.itertuples(index=False), total=len(test_df),
                    desc=f"{model.name}/{regime}/{media_type}", unit="clip")
    for row in iterator:
        key = (row.recording_id, int(row.clip_idx))
        if key in done:
            n_skip += 1
            continue
        media = load_media(row._asdict())
        prompt = build_prompt(
            media=media, label_names=names, regime=regime, support=support,
            media_type=media_type, lang=lang, enrich_text=None,
        )
        scores = model.choice_logprobs([prompt], choices=letters)[0]
        gen_text = model.generate([prompt], max_new_tokens=max_new_tokens)[0]

        pred_letter = max(scores, key=scores.get)
        pred_label = letter_to_label_map[pred_letter]
        parse_letter, parse_label = parse_answer(gen_text, letters, label_order, names)

        rec = {
            "recording_id": row.recording_id,
            "clip_idx": int(row.clip_idx),
            "label_id": int(row.label_id),
            "pred_letter": pred_letter,
            "pred_label_id": int(pred_label),
            "parse_letter": parse_letter,
            "gen_text": gen_text,
            "choice_logprobs": scores,
            "letter_to_label": letter_to_label_map,
        }
        new_records.append(rec)
        _append_ckpt(ckpt, rec)
    if n_skip:
        print(f"  (resumed {n_skip} clips from checkpoint)")

    # order all records (ckpt + new) by the test split for stable output
    by_key = {(r["recording_id"], r["clip_idx"]): r for r in (ckpt_records + new_records)}
    records = [by_key[(r.recording_id, int(r.clip_idx))]
               for r in test_df.itertuples(index=False)
               if (r.recording_id, int(r.clip_idx)) in by_key]

    label_ids = sorted(names.keys())
    y_true = [r["label_id"] for r in records]
    y_pred = [r["pred_label_id"] for r in records]
    metrics_clip = classification_metrics(y_true, y_pred, label_ids, names)
    metrics_rec_avg = _recording_metrics(records, label_ids, names, "avg_prob")
    metrics_rec_maj = _recording_metrics(records, label_ids, names, "majority")

    print(f"  clip-level   : {format_metrics(metrics_clip)}")
    print(f"  rec(avg_prob): {format_metrics(metrics_rec_avg)}")
    print(f"  rec(majority): {format_metrics(metrics_rec_maj)}")

    return {
        "metrics_clip": metrics_clip,
        "metrics_recording": metrics_rec_avg,
        "metrics_recording_majority": metrics_rec_maj,
        "records": records,
    }


def _recording_metrics(records, label_ids, names, method):
    y_true, y_pred = aggregate_predictions(records, method=method)
    m = classification_metrics(y_true, y_pred, label_ids, names)
    m["method"] = method
    return m


def run_similarity_few_shot(
    model,
    manifest,
    *,
    dataset_name: str = "deepship",
    shot: int = 1,
    support_seed: int = 0,
    target_sr: int = 16000,
    clip_len: float = 30.0,
    lang: str = "en",
    limit: int | None = None,
    max_recordings: int | None = None,
    eval_split: str = "test",
    experiment: str | None = None,
) -> dict:
    """Pairwise logprob-margin few-shot for context-limited LALMs (评测 B).

    In-context few-shot is infeasible on small-context audio models (e.g.
    Qwen2-Audio 8192: each 30 s audio ≈ 2900 tokens → ≥5 audios overflow). Instead
    we compare the query to ONE reference at a time (2 audios ≈ 5800 tokens, fits),
    score the pair by the yes/no **logprob margin** (log-odds of "same vessel type"),
    and classify by the class whose best reference wins:

        score(query, ref) = logprob("yes") − logprob("no")
        class_score(c)   = max over refs of class c of score(query, ref)
        pred             = argmax_c class_score(c)

    The margin is a continuous similarity → no yes/yes ties. This is "LALM as a
    similarity metric for k-NN few-shot" — a small methodological contribution.
    Returns the same ``{metrics_clip, metrics_recording, records}`` shape.
    """
    names = class_names(dataset_name, lang)
    label_ids = sorted(names.keys())
    _, label_order, _ = option_layout(names)
    class_letter = {lid: LETTERS[label_order.index(lid)] for lid in label_ids}

    full = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
    train_df = full[full["split"] == "train"].reset_index(drop=True)
    test_df = full[full["split"] == eval_split].reset_index(drop=True)
    if max_recordings is not None:
        per_class = max(1, max_recordings // test_df["label_id"].nunique())
        recs = test_df.drop_duplicates("recording_id")
        keep = []
        for _lid, sub in recs.groupby("label_id"):
            keep.extend(sub["recording_id"].head(per_class).tolist())
        test_df = test_df[test_df["recording_id"].isin(keep)].reset_index(drop=True)
    if limit:
        test_df = test_df.head(limit)

    clip_samples = int(round(clip_len * target_sr))
    sup_idx = few_shot_indices(train_df, k=shot, seed=support_seed)
    support = []
    for j in sup_idx:
        r = train_df.iloc[j]
        support.append({"label_id": int(r["label_id"]),
                        "audio": read_segment(r["path"], r["start"], r["end"],
                                              target_sr=target_sr, pad_to_samples=clip_samples)})

    ckpt = _ckpt_path(experiment)
    ckpt_records, done = _load_ckpt(ckpt)
    new_records = []
    n_skip = 0
    iterator = tqdm(test_df.itertuples(index=False), total=len(test_df),
                    desc=f"{model.name}/similarity/k{shot}", unit="clip")
    for row in iterator:
        key = (row.recording_id, int(row.clip_idx))
        if key in done:
            n_skip += 1
            continue
        query = read_segment(row.path, row.start, row.end,
                             target_sr=target_sr, pad_to_samples=clip_samples)
        class_scores = {lid: -1e9 for lid in label_ids}
        for sup in support:
            msgs = build_pairwise(sup["audio"], query, names[sup["label_id"]], "audio", lang)
            sc = model.choice_logprobs([msgs], choices=["yes", "no"])[0]
            pair = sc["yes"] - sc["no"]
            lid = sup["label_id"]
            if pair > class_scores[lid]:
                class_scores[lid] = pair
        pred_lid = max(class_scores, key=class_scores.get)
        rec = {
            "recording_id": row.recording_id,
            "clip_idx": int(row.clip_idx),
            "label_id": int(row.label_id),
            "pred_label_id": int(pred_lid),
            "pred_letter": class_letter[pred_lid],
            "parse_letter": None,
            "gen_text": "",
            "choice_logprobs": {class_letter[lid]: class_scores[lid] for lid in label_ids},
            "letter_to_label": {class_letter[lid]: lid for lid in label_ids},
        }
        new_records.append(rec)
        _append_ckpt(ckpt, rec)
    if n_skip:
        print(f"  (resumed {n_skip} clips from checkpoint)")

    by_key = {(r["recording_id"], r["clip_idx"]): r for r in (ckpt_records + new_records)}
    records = [by_key[(r.recording_id, int(r.clip_idx))]
               for r in test_df.itertuples(index=False)
               if (r.recording_id, int(r.clip_idx)) in by_key]

    y_true = [r["label_id"] for r in records]
    y_pred = [r["pred_label_id"] for r in records]
    metrics_clip = classification_metrics(y_true, y_pred, label_ids, names)
    metrics_rec_avg = _recording_metrics(records, label_ids, names, "avg_prob")
    metrics_rec_maj = _recording_metrics(records, label_ids, names, "majority")
    print(f"  clip-level   : {format_metrics(metrics_clip)}  [pairwise logprob-margin k={shot}]")
    print(f"  rec(avg_prob): {format_metrics(metrics_rec_avg)}")
    return {"metrics_clip": metrics_clip, "metrics_recording": metrics_rec_avg,
            "metrics_recording_majority": metrics_rec_maj, "records": records}


def run_fusion_eval(
    model,
    manifest,
    *,
    fusion_spec: str = "audio+mel",
    dataset_name: str = "deepship",
    regime: str = "zero_shot",
    shot: int = 5,
    support_seed: int = 0,
    target_sr: int = 16000,
    clip_len: float = 30.0,
    lang: str = "en",
    limit: int | None = None,
    max_recordings: int | None = None,
    max_new_tokens: int = 8,
    eval_split: str = "test",
    experiment: str | None = None,
) -> dict:
    """Fusion prompting: multiple media items (audio+image) per clip (评测 Omni B/D).

    ``fusion_spec`` e.g. "audio+mel" → loads both audio (from path) and mel
    spectrogram (from mel_path) per clip, puts both in one prompt.
    """
    from ..prompting.templates import build_fusion_prompt

    parts = fusion_spec.split("+")
    names = class_names(dataset_name, lang)
    letters, label_order, _ = option_layout(names)
    letter_to_label_map = {l: letter_to_label(l, label_order) for l in letters}
    clip_samples = int(round(clip_len * target_sr))

    full = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
    train_df = full[full["split"] == "train"].reset_index(drop=True)
    test_df = full[full["split"] == eval_split].reset_index(drop=True)
    if max_recordings is not None:
        per_class = max(1, max_recordings // test_df["label_id"].nunique())
        recs = test_df.drop_duplicates("recording_id")
        keep = []
        for _lid, sub in recs.groupby("label_id"):
            keep.extend(sub["recording_id"].head(per_class).tolist())
        test_df = test_df[test_df["recording_id"].isin(keep)].reset_index(drop=True)
    if limit:
        test_df = test_df.head(limit)

    def load_fusion(row) -> list[tuple]:
        ml = []
        for part in parts:
            if part == "audio":
                wav = read_segment(row["path"], row["start"], row["end"],
                                   target_sr=target_sr, pad_to_samples=clip_samples)
                ml.append((wav, "audio"))
            else:
                col = f"{part}_path"
                ml.append((Image.open(row[col]).convert("RGB"), "image"))
        return ml

    # few-shot support (each support clip → both media)
    support = None
    if regime == "few_shot":
        sup_idx = few_shot_indices(train_df, k=shot, seed=support_seed)
        support = [{"label_id": int(train_df.iloc[j]["label_id"]),
                    "media_list": load_fusion(train_df.iloc[j])} for j in sup_idx]

    # checkpoint
    ckpt = _ckpt_path(experiment)
    ckpt_records, done = _load_ckpt(ckpt)
    new_records = []
    n_skip = 0
    iterator = tqdm(test_df.itertuples(index=False), total=len(test_df),
                    desc=f"{model.name}/fusion/{fusion_spec}/{regime}", unit="clip")
    for row in iterator:
        key = (row.recording_id, int(row.clip_idx))
        if key in done:
            n_skip += 1
            continue
        media_list = load_fusion(row._asdict())
        prompt = build_fusion_prompt(media_list, names, regime, support, lang, parts)
        scores = model.choice_logprobs([prompt], choices=letters)[0]
        gen_text = model.generate([prompt], max_new_tokens=max_new_tokens)[0]
        pred_letter = max(scores, key=scores.get)
        pred_label = letter_to_label_map[pred_letter]
        parse_letter, parse_label = parse_answer(gen_text, letters, label_order, names)
        rec = {
            "recording_id": row.recording_id, "clip_idx": int(row.clip_idx),
            "label_id": int(row.label_id), "pred_letter": pred_letter,
            "pred_label_id": int(pred_label), "parse_letter": parse_letter,
            "gen_text": gen_text, "choice_logprobs": scores,
            "letter_to_label": letter_to_label_map,
        }
        new_records.append(rec)
        _append_ckpt(ckpt, rec)
    if n_skip:
        print(f"  (resumed {n_skip} clips from checkpoint)")

    by_key = {(r["recording_id"], r["clip_idx"]): r for r in (ckpt_records + new_records)}
    records = [by_key[(r.recording_id, int(r.clip_idx))]
               for r in test_df.itertuples(index=False)
               if (r.recording_id, int(r.clip_idx)) in by_key]
    label_ids = sorted(names.keys())
    y_true = [r["label_id"] for r in records]
    y_pred = [r["pred_label_id"] for r in records]
    metrics_clip = classification_metrics(y_true, y_pred, label_ids, names)
    metrics_rec_avg = _recording_metrics(records, label_ids, names, "avg_prob")
    metrics_rec_maj = _recording_metrics(records, label_ids, names, "majority")
    print(f"  clip-level   : {format_metrics(metrics_clip)}  [fusion={fusion_spec} {regime}]")
    print(f"  rec(avg_prob): {format_metrics(metrics_rec_avg)}")
    return {"metrics_clip": metrics_clip, "metrics_recording": metrics_rec_avg,
            "metrics_recording_majority": metrics_rec_maj, "records": records}
