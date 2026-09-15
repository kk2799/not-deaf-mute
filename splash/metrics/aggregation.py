"""Aggregate clip-level predictions to recording-level.

Since long recordings are segmented into overlapping clips, predictions are
clip-level. We aggregate to recording-level two ways and report both:
  * ``avg_prob``   — average the per-class choice probabilities across a
                     recording's clips, argmax → recording label. Smoother,
                     uses confidence.
  * ``majority``   — one vote per clip, majority label per recording.

A recording's gold label is the same across all its clips (set at segmentation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _softmax_over_logprobs(logprob_dicts: list[dict[str, float]]) -> np.ndarray:
    """Stack per-choice logprobs across clips → softmax-normalised [clips, choices]."""
    keys = list(logprob_dicts[0].keys())
    arr = np.array([[d[k] for k in keys] for d in logprob_dicts], dtype=float)
    arr = arr - arr.max(axis=1, keepdims=True)
    e = np.exp(arr)
    return e / e.sum(axis=1, keepdims=True), keys


def aggregate_predictions(
    records: list[dict],
    method: str = "avg_prob",
) -> tuple[list[int], list[int]]:
    """Aggregate per-clip ``records`` to per-recording (y_true, y_pred).

    Each record: {recording_id, label_id (gold), choice_logprobs: {choice: lp},
                  pred_label_id (clip)}. Returns aligned per-recording lists.
    """
    df = pd.DataFrame(records)
    y_true, y_pred = [], []
    for rid, sub in df.groupby("recording_id"):
        gold = int(sub["label_id"].iloc[0])
        y_true.append(gold)
        if method == "avg_prob" and "choice_logprobs" in sub.columns:
            lp = list(sub["choice_logprobs"])
            probs, keys = _softmax_over_logprobs(lp)
            # keys are letters; map back to label via the per-record pred mapping
            # (letter order is consistent across clips in one experiment).
            # Fallback: use clip pred_label_id majority if mapping ambiguous.
            avg = probs.mean(axis=0)
            best_letter = keys[int(np.argmax(avg))]
            pred = _letter_to_label(best_letter, sub.iloc[0])
            y_pred.append(pred if pred is not None else int(sub["pred_label_id"].mode()[0]))
        else:  # majority vote on clip predictions
            y_pred.append(int(sub["pred_label_id"].mode()[0]))
    return y_true, y_pred


def _letter_to_label(letter: str, record: dict):
    """Recover label id from a letter using per-record mapping fields if present."""
    mapping = record.get("letter_to_label")
    if mapping:
        return mapping.get(letter)
    return None
