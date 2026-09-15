"""Audio I/O: loading, slice reading, and resampling.

All loaders return mono float32 waveforms resampled to a model's target sample
rate. Two entry points:

* ``load_audio``        — read a whole file (use for short clips / smoke tests).
* ``read_segment``      — read a time window directly from disk (efficient for
                          clip-indexed datasets over long recordings: only the
                          requested window is read, then resampled).

DeepShip recordings are 22500 Hz mono; ShipsEar is 52734 Hz mono. Models pick
their own target rate (Whisper/Qwen2-Audio/BEATs/WavLM all expect 16000 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AudioPath = str | Path


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    frames: int
    channels: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def audio_info(path: AudioPath) -> AudioInfo:
    """Cheap metadata probe (does not decode audio)."""
    info = sf.info(str(path))
    return AudioInfo(info.samplerate, info.frames, info.channels)


def _to_mono(wav: np.ndarray) -> np.ndarray:
    """Collapse to mono by averaging channels if needed."""
    if wav.ndim == 2:
        return wav.mean(axis=1)
    return wav


def load_audio(path: AudioPath, target_sr: int = 16000, mono: bool = True) -> np.ndarray:
    """Load an entire file, resample to ``target_sr``, return mono float32."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if mono:
        wav = _to_mono(wav)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return wav.astype(np.float32, copy=False)


def read_segment(
    path: AudioPath,
    start_s: float,
    end_s: float,
    target_sr: int = 16000,
    pad_to_samples: int | None = None,
) -> np.ndarray:
    """Read the time window ``[start_s, end_s)`` from disk and resample.

    Reads only the requested native-sr frames (efficient for long recordings),
    resamples to ``target_sr``, and optionally zero-pads the result to a fixed
    length (so every clip is exactly ``clip_len * target_sr`` samples for models
    like Whisper that expect a fixed 30 s window).

    If the window extends past the end of the file, the result is right-padded
    with zeros (tail clips of long recordings / short recordings).
    """
    info = audio_info(path)
    file_sr = info.sample_rate
    start_frame = max(0, int(round(start_s * file_sr)))
    # requested native frames; clamp to what's available
    req_frames = int(round((end_s - start_s) * file_sr))
    avail_frames = max(0, info.frames - start_frame)
    wav, _ = sf.read(
        str(path),
        start=start_frame,
        frames=min(req_frames, avail_frames),
        dtype="float32",
        always_2d=False,
    )
    wav = _to_mono(wav)
    if file_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=target_sr, res_type="soxr_hq")
    wav = wav.astype(np.float32, copy=False)

    # Pad if the window ran past EOF or a fixed length is requested.
    target_len = pad_to_samples if pad_to_samples is not None else int(round((end_s - start_s) * target_sr))
    if len(wav) < target_len:
        wav = np.pad(wav, (0, target_len - len(wav)))
    elif len(wav) > target_len:
        wav = wav[:target_len]
    return wav
