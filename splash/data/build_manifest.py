"""Scan a dataset on disk and build a clip manifest.

Directory convention (DeepShip-style):
    ``<data_root>/<split>/<label_id>/<recording>.wav``
e.g. ``deepship/train/0/0_101.wav``. The integer folder name *is* the label id
(see ``splash.data.labels``). ShipsEar's layout differs and is handled later.

A manifest is one row per *clip* (after fixed-length segmentation), cached to
``outputs/manifests/<dataset>_clip<len>_ov<ov>_<hash>.csv``. The cache key
fingerprints the dataset name, segmentation params, and the full file list with
durations — so it auto-invalidates if the audio or params change.

Segmentation happens *within* each recording; train/test are already split at
the recording level on disk, so no clip crosses splits (no leakage).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ..audio.io import audio_info
from .labels import class_names
from .segmentation import Clip, segment_recording

DEFAULT_SPLITS = ("train", "test")


def _scan_split(
    split_root: Path,
    split: str,
    clip_len: float,
    overlap: float,
) -> tuple[list[Clip], list[tuple[str, float]]]:
    """Scan one split dir → clips + (relpath, duration) fingerprint tuples."""
    clips: list[Clip] = []
    fingerprint: list[tuple[str, float]] = []
    if not split_root.is_dir():
        return clips, fingerprint
    for class_dir in sorted(split_root.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue
        label_id = int(class_dir.name)
        for wav in sorted(class_dir.glob("*.wav")):
            duration = audio_info(wav).duration
            fingerprint.append((str(wav.relative_to(split_root.parent)), round(duration, 3)))
            clips.extend(
                segment_recording(
                    recording_id=wav.stem,
                    path=str(wav),
                    duration=duration,
                    label_id=label_id,
                    split=split,
                    clip_len=clip_len,
                    overlap=overlap,
                )
            )
    return clips, fingerprint


def _fingerprint(dataset_name: str, clip_len: float, overlap: float, files: list[tuple[str, float]]) -> str:
    payload = f"{dataset_name}|clip={clip_len}|ov={overlap}|" + ",".join(f"{p}:{d}" for p, d in files)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def manifest_filename(dataset_name: str, clip_len: float, overlap: float, fp: str) -> str:
    return f"{dataset_name}_clip{clip_len:g}_ov{overlap:g}_{fp}.csv"


def build_manifest(
    data_root: str | Path,
    dataset_name: str,
    clip_len: float = 30.0,
    overlap: float = 0.5,
    splits=DEFAULT_SPLITS,
    out_dir: str | Path = "outputs/manifests",
    lang: str = "en",
    force: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Build (or load cached) clip manifest for a dataset.

    Returns ``(manifest_df, manifest_path)``. Columns: recording_id, clip_idx,
    path, start, end, label_id, label_name, split, dataset.
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect files for the fingerprint (need durations anyway).
    all_clips: list[Clip] = []
    all_files: list[tuple[str, float]] = []
    for split in splits:
        clips, files = _scan_split(data_root / split, split, clip_len, overlap)
        all_clips.extend(clips)
        all_files.extend(files)

    fp = _fingerprint(dataset_name, clip_len, overlap, sorted(all_files))
    out_path = out_dir / manifest_filename(dataset_name, clip_len, overlap, fp)

    if out_path.exists() and not force:
        return pd.read_csv(out_path), out_path

    names = class_names(dataset_name, lang)
    rows = []
    for c in all_clips:
        row = c.as_row()
        row["label_name"] = names.get(c.label_id, str(c.label_id))
        row["dataset"] = dataset_name
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["split", "label_id", "recording_id", "clip_idx"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df, out_path


def manifest_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-split × per-class clip counts, for sanity printing."""
    return (
        df.groupby(["split", "label_id", "label_name"])
        .size()
        .reset_index(name="clips")
        .pivot_table(index=["label_id", "label_name"], columns="split", values="clips", fill_value=0)
    )
