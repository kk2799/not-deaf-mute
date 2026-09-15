"""Clip-indexed UATR dataset.

A ``UATRDataset`` wraps a manifest (one row per clip) produced by
``scripts/prepare_data.py`` and lazily reads each clip's audio window from disk
(via ``splash.audio.io.read_segment``). It is regime-agnostic: pass a manifest
filtered to the split / few-shot support set you want.

Returns plain Python/numpy dicts (not tensors) so the same dataset serves
prompting (评测 A/B) and future probing / fine-tuning (评测 C/D) runners, which
collate differently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..audio.io import read_segment
from .labels import class_names

MANIFEST_COLUMNS = [
    "recording_id", "clip_idx", "path", "start", "end", "label_id", "split",
]


class UATRDataset(Dataset):
    """Reads clip audio on demand from a manifest DataFrame or CSV file.

    Parameters
    ----------
    manifest : DataFrame or path
        Manifest with the columns in ``MANIFEST_COLUMNS``.
    target_sr : int
        Sample rate the consuming model expects (audio is resampled to this).
    clip_len : float
        Clip length in seconds; audio is padded/truncated to ``clip_len*target_sr``.
    dataset_name : str
        Dataset key into ``splash.data.labels`` (for label names).
    lang : str
        ``en`` or ``zh`` — which label name to attach.
    split : str, optional
        If given, filter the manifest to this split (``train``/``val``/``test``).
    indices : array-like, optional
        Explicit row indices (e.g. a few-shot support set from ``splits.py``).
    """

    def __init__(
        self,
        manifest,
        target_sr: int = 16000,
        clip_len: float = 30.0,
        dataset_name: str = "deepship",
        lang: str = "en",
        split: str | None = None,
        indices=None,
    ):
        if isinstance(manifest, (str, Path)):
            df = pd.read_csv(manifest)
        else:
            df = manifest.reset_index(drop=True)

        # Validate columns (tolerate extra columns like label_name/dataset).
        missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"manifest missing columns: {missing}")

        if split is not None:
            df = df[df["split"] == split].reset_index(drop=True)
        if indices is not None:
            df = df.iloc[list(indices)].reset_index(drop=True)

        self.df = df
        self.target_sr = target_sr
        self.clip_samples = int(round(clip_len * target_sr))
        self.clip_len = clip_len
        self._dataset_name = dataset_name
        self._lang = lang
        self._names = class_names(dataset_name, lang)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        wav = read_segment(
            row["path"],
            start_s=row["start"],
            end_s=row["end"],
            target_sr=self.target_sr,
            pad_to_samples=self.clip_samples,
        )
        lid = int(row["label_id"])
        return {
            "audio": wav,                       # float32 [clip_samples]
            "sample_rate": self.target_sr,
            "label_id": lid,
            "label_name": self._names.get(lid, str(lid)),
            "recording_id": row["recording_id"],
            "clip_idx": int(row["clip_idx"]),
            "split": row["split"],
        }

    @property
    def label_ids(self) -> np.ndarray:
        return self.df["label_id"].to_numpy()

    def subset(self, indices) -> "UATRDataset":
        """Return a new dataset over the given row indices (shares config)."""
        return UATRDataset(
            self.df.iloc[list(indices)],
            target_sr=self.target_sr,
            clip_len=self.clip_len,
            dataset_name=self._dataset_name,
            lang=self._lang,
        )


def load_manifest(path) -> pd.DataFrame:
    """Load a manifest CSV, validating required columns."""
    df = pd.read_csv(path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest {path} missing columns: {missing}")
    return df
