"""Fixed-length, overlapping segmentation of long recordings into clips.

DeepShip recordings are 60–440 s; UATR instances are short clips. Default
policy (per project decision): ``clip_len=30 s``, ``overlap=0.5`` (50 %), i.e.
a 15 s hop. Both are first-class config values so duration ablations can sweep
them via Hydra (``clip_len=10,30,60``).

Clips are computed *within* each recording. Since train/test are split at the
recording level on disk, no clip ever crosses the train/test boundary → no data
leakage. A recording shorter than ``clip_len`` yields a single clip that is
zero-padded at load time (see ``splash.audio.io.read_segment``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Clip:
    """A single fixed-length window into a recording."""

    recording_id: str
    path: str
    clip_idx: int
    start: float          # seconds
    end: float            # seconds (== start + clip_len)
    label_id: int
    split: str

    def as_row(self) -> dict:
        return {
            "recording_id": self.recording_id,
            "clip_idx": self.clip_idx,
            "path": self.path,
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "label_id": self.label_id,
            "split": self.split,
        }


def compute_clip_starts(duration: float, clip_len: float, overlap: float) -> list[float]:
    """Start times (s) of clips covering ``[0, duration)`` with given overlap.

    * ``overlap`` is a fraction in ``[0, 1)``; hop = ``clip_len * (1 - overlap)``.
    * Recordings shorter than ``clip_len`` → a single start at 0.
    * The tail is covered by an end-anchored final clip when the regular grid
      would leave a gap (avoids dropping the last partial window).
    """
    if duration <= clip_len:
        return [0.0]
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    hop = clip_len * (1.0 - overlap)
    eps = 1e-6
    starts = list(np.arange(0.0, duration - clip_len + eps, hop))
    # Cover an uncovered tail with an end-anchored clip (no tail data dropped),
    # but skip it if it would just duplicate the last grid start.
    if starts[-1] + clip_len < duration - eps:
        anchor = duration - clip_len
        if anchor > starts[-1] + eps:
            starts.append(anchor)
    return starts


def segment_recording(
    recording_id: str,
    path: str,
    duration: float,
    label_id: int,
    split: str,
    clip_len: float = 30.0,
    overlap: float = 0.5,
) -> list[Clip]:
    """Slice one recording into Clips."""
    starts = compute_clip_starts(duration, clip_len, overlap)
    return [
        Clip(
            recording_id=recording_id,
            path=path,
            clip_idx=i,
            start=s,
            end=s + clip_len,
            label_id=label_id,
            split=split,
        )
        for i, s in enumerate(starts)
    ]
