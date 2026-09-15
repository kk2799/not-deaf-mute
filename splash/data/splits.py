"""Sampling helpers: few-shot support sets and train/val splits.

All samplers are deterministic given a seed → fully reproducible. They operate
on a manifest DataFrame (one row per clip) so they're decoupled from audio IO.

Regime semantics for the project:
* **zero-shot** — no support set; the test split is evaluated directly.
* **few-shot (k)** — draw ``k`` clips per class from the *train* split as the
  in-context support set; evaluate on the *test* split (support clips removed
  from any future "full train" use, which is fine since they're train-side).
* **full_train** — train/val/test are the manifest splits; optionally carve a
  validation set out of train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def few_shot_indices(
    df: pd.DataFrame,
    k: int,
    seed: int = 0,
    label_col: str = "label_id",
    prefer_col: str | None = "recording_id",
) -> np.ndarray:
    """Balanced ``k`` clips per class from ``df`` (positional row indices).

    Returns positional indices into ``df`` (use ``df.iloc[out]``), so results are
    correct regardless of the frame's index state. Samples ``k`` per unique
    label. When ``prefer_col`` is given, we first try to pick support clips from
    *distinct recordings* (so a few-shot support set spans recordings, not
    repeated windows of one), falling back to plain sampling when there aren't
    enough distinct recordings.
    """
    rng = np.random.default_rng(seed)
    work = df.reset_index(drop=True)  # positional indices stable vs `df` order
    out: list[int] = []
    for _label, sub in work.groupby(label_col):
        idx = sub.index.to_numpy()
        if len(idx) <= k:
            picked = idx
        elif prefer_col and prefer_col in work.columns:
            recs = work.iloc[idx][prefer_col].to_numpy()
            order = rng.permutation(len(idx))
            seen_recs, chosen = set(), []
            for j in order:
                r = recs[j]
                if r not in seen_recs:
                    chosen.append(int(idx[j]))
                    seen_recs.add(r)
                if len(chosen) == k:
                    break
            if len(chosen) < k:  # ran out of distinct recordings; top up
                chosen_set = set(chosen)
                for j in order:
                    if int(idx[j]) not in chosen_set:
                        chosen.append(int(idx[j]))
                    if len(chosen) == k:
                        break
            picked = np.array(chosen, dtype=int)
        else:
            picked = rng.choice(idx, size=k, replace=False)
        out.extend(int(x) for x in picked)
    return np.array(sorted(out), dtype=int)


def train_val_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    seed: int = 0,
    label_col: str = "label_id",
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set out of ``df`` (assumed already train-only).

    Stratified by label when ``stratify`` is True so class balance is preserved.
    Returns ``(train_df, val_df)`` with reset indices.
    """
    train_parts, val_parts = [], []
    if stratify:
        groups = df.groupby(label_col)
    else:
        groups = [(None, df)]
    for _, sub in groups:
        sub = sub.sample(frac=1.0, random_state=seed)
        n_val = max(1, int(round(len(sub) * val_frac)))
        val_parts.append(sub.iloc[:n_val])
        train_parts.append(sub.iloc[n_val:])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    return train_df, val_df
