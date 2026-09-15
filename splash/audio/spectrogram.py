"""Spectrogram extraction & rendering for 评测 A (VLM reads spectrograms) / 评测 F.

**Reuses the project's validated UATR feature pipeline** (`fea-extract/fea_extract.py`)
rather than re-implementing DSP, so the rendered images are bit-consistent with
the features the classifiers in other experiments consume.

Three representations, computed exactly as in `fea_extract.py`:
* ``stft``  — 50 ms / 25 ms framing, Hamming window, pre-emphasis 0.97, FFT
              magnitude cropped to [low_freq=10, high_freq=8000] Hz.  (get_stft_feature)
* ``mel``   — Kaldi-style log Mel-fbank, **300 filters**, hanning, 50/25 ms,
              pre-emphasis 0.97, power, high=8000/low=10.               (get_fbank_feature)
* ``demon`` — multi-band amplitude demodulation: bandpass-sweep 6000–8000 Hz
              (250 Hz step → 9 carrier bands), square → FFT → 0.9–40 Hz
              modulation band. Exposes shaft/blade-rate structure.       (get_demon_feature)

Each 2D feature array is rendered to a clean RGB PNG (no axes/chart chrome) via
a perceptually uniform colormap; magnitude-type features (STFT, DEMON) are
log-scaled first. Normalization is per-clip robust percentile → consistent display.

Note: `fea-extract/` ships with a hyphen in its dir name (not importable as a
package), so we add it to ``sys.path`` and import the module directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# --- make fea-extract importable (hyphenated dir name -> sys.path, not package)
_FEA_DIR = Path(__file__).resolve().parents[2] / "fea-extract"
_FEA = None


def _fea():
    """Lazy-import fea_extract (adds fea-extract/ to sys.path once)."""
    global _FEA
    if _FEA is None:
        p = str(_FEA_DIR)
        if p not in sys.path:
            sys.path.insert(0, p)
        import fea_extract as fe  # noqa
        _FEA = fe
    return _FEA


# Parameters mirror fea_extract.py globals (overridable per call via `cfg`).
DEFAULTS = {
    "frame_length_ms": 50.0,
    "frame_shift_ms": 25.0,
    "high_freq": 8000.0,
    "low_freq": 10.0,
    "n_mels": 300,                # fbank_filters in fea_extract
    # DEMON (1D 平方解调包络谱)
    "demon_low_frequency": 2000.0,
    "demon_high_frequency": 8000.0,
    "demon_freq_resolution": 250.0,
    "f_dem_max": 50.0,             # 保留调制频率 0..50 Hz
    "demon_dc_cut": 0.4,           # 低于此频率 (Hz) 直接置 0（去直流/近直流）
    "demon_cont_window_hz": 2.0,   # 减连续谱时的平滑窗 (Hz)
    # rendering
    "cmap": "viridis",
    "p_low": 1.0,
    "p_high": 99.0,
    "apply_mvn": True,            # per-clip mean/variance normalization on waveform (matches fea_extract)
}


def _prep_wav(wav: np.ndarray, cfg: dict) -> np.ndarray:
    """Mono float32 + optional per-clip MVN (fea_extract line 654)."""
    wav = np.asarray(wav, dtype=np.float32).flatten()
    if cfg.get("apply_mvn", True):
        s = wav.std() + 1e-40
        wav = (wav - wav.mean()) / s
    return wav


def compute_stft(wav: np.ndarray, fs: int, cfg: dict | None = None) -> np.ndarray:
    """STFT magnitude [frames, freq_bins] (fea_extract.get_stft_feature)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    fe = _fea()
    fe.frame_length = cfg["frame_length_ms"]
    fe.frame_shift = cfg["frame_shift_ms"]
    fe.high_freq = cfg["high_freq"]
    fe.low_freq = cfg["low_freq"]
    fe.fbank_filters = cfg["n_mels"]
    return fe.get_stft_feature(_prep_wav(wav, cfg), fs, cfg["high_freq"], cfg["low_freq"])


def compute_mel(wav: np.ndarray, fs: int, cfg: dict | None = None) -> np.ndarray:
    """Log Mel-fbank [frames, n_mels=300] (fea_extract.get_fbank_feature)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    fe = _fea()
    fe.frame_length = cfg["frame_length_ms"]
    fe.frame_shift = cfg["frame_shift_ms"]
    fe.high_freq = cfg["high_freq"]
    fe.low_freq = cfg["low_freq"]
    fe.fbank_filters = cfg["n_mels"]
    return fe.get_fbank_feature(_prep_wav(wav, cfg), fs, cfg["high_freq"], cfg["low_freq"])


def compute_demon(wav: np.ndarray, fs: int, cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """1D DEMON envelope spectrum via **squared demodulation**.

    Pipeline (per project spec):
      bandpass [demon_low, demon_high] → square (demodulate) → |rfft| → crop
      0..f_dem_max Hz → zero below ``demon_dc_cut`` Hz → subtract the continuous
      (median-smoothed) spectrum to highlight line peaks → min-max normalize.
    Returns ``(spectrum[0,1], freqs)``. No dB.
    """
    from scipy.signal import butter, sosfiltfilt
    from scipy.ndimage import median_filter

    cfg = {**DEFAULTS, **(cfg or {})}
    x = _prep_wav(wav, cfg)

    # 1) bandpass the cavitation band
    nyq = fs / 2.0
    bp_low = float(cfg["demon_low_frequency"])
    bp_high = float(cfg["demon_high_frequency"])
    if 0 < bp_low < bp_high < nyq:
        sos = butter(4, [bp_low / nyq, bp_high / nyq], btype="band", output="sos")
        x = sosfiltfilt(sos, x)

    # 2) squared demodulation + magnitude spectrum
    spec = np.abs(np.fft.rfft(x ** 2)).astype(np.float64)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)

    # 3) crop to [0, f_dem_max]
    f_max = float(cfg["f_dem_max"])
    keep = freqs <= f_max
    spec, freqs = spec[keep], freqs[keep]

    # 4) zero out near-DC (< demon_dc_cut Hz) — kills the residual DC spike
    spec[freqs < float(cfg["demon_dc_cut"])] = 0.0

    # 5) subtract the continuous (median-smoothed) spectrum → highlight peaks
    if len(spec) > 1:
        df = float(freqs[1] - freqs[0])
        win = max(3, int(round(float(cfg["demon_cont_window_hz"]) / df)) | 1)  # odd
        cont = median_filter(spec, size=win)
        spec = np.clip(spec - cont, 0.0, None)

    # 6) min-max normalize to [0, 1]  (no dB)
    smax = spec.max()
    spec = spec / (smax + 1e-12) if smax > 1e-12 else spec
    return spec.astype(np.float32), freqs


_COMPUTERS = {"stft": compute_stft, "mel": compute_mel}


def compute_spectrogram(wav: np.ndarray, fs: int, mode: str, cfg: dict | None = None) -> np.ndarray:
    """2D feature array for stft/mel (DEMON is 1D — use ``compute_demon``)."""
    if mode == "demon":
        raise ValueError("DEMON is 1D; call compute_demon() / render_spectrogram(mode='demon') instead.")
    if mode not in _COMPUTERS:
        raise ValueError(f"unknown mode '{mode}' (use 'stft'|'mel'|'demon')")
    return _COMPUTERS[mode](wav, fs, cfg)


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def _fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], mode="RGB")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return img


def render_image(arr: np.ndarray, mode: str, fs: int, cfg: dict | None = None) -> Image.Image:
    """Render a 2D feature array [frames, freq_bins] as a labeled figure.

    White background, colormap imshow, with readable axes:
      * STFT: y = Frequency (Hz, linear 10–8000), x = Time (s)
      * Mel:   y = Frequency (Hz, ticks via mel scale), x = Time (s)
    STFT magnitude is log-scaled for display; mel is already log-fbank.
    Robust percentile normalization → consistent contrast.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    cfg = {**DEFAULTS, **(cfg or {})}
    a = np.asarray(arr, dtype=np.float32)
    if mode == "stft":                       # magnitude -> log for display
        a = np.log10(a + 1e-12)
    lo, hi = np.percentile(a, cfg["p_low"]), np.percentile(a, cfg["p_high"])
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    n_frames, _ = a.shape
    t_max = (n_frames - 1) * cfg["frame_shift_ms"] / 1000.0
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=130)
    if mode == "stft":
        f_lo, f_hi = float(cfg["low_freq"]), float(cfg["high_freq"])
        ax.imshow(a.T, aspect="auto", origin="lower",
                  extent=[0, t_max, f_lo, f_hi], cmap=cfg["cmap"])
        ax.set_ylabel("Frequency (Hz)")
    else:                                    # mel: axis linear in mel, ticks labelled in Hz
        m_lo, m_hi = float(_hz_to_mel(cfg["low_freq"])), float(_hz_to_mel(cfg["high_freq"]))
        ax.imshow(a.T, aspect="auto", origin="lower",
                  extent=[0, t_max, m_lo, m_hi], cmap=cfg["cmap"])
        mt = np.linspace(m_lo, m_hi, 6)
        ax.set_yticks(mt, [f"{float(_mel_to_hz(m)):.0f}" for m in mt])
        ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)
    return _fig_to_image(fig)


def render_spectrum_1d(spectrum: np.ndarray, freqs: np.ndarray, cfg: dict | None = None) -> Image.Image:
    """Render the (already normalized [0,1]) DEMON spectrum as a labeled line plot.

    White background, x = Modulation frequency (Hz, ticks every 10 Hz),
    y = Amplitude. No dB / percentile (all done in ``compute_demon``).
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    s = np.asarray(spectrum, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.0, 3.0), dpi=130)
    ax.fill_between(freqs, 0.0, s, facecolor="#3a7bd5", edgecolor="#1f4fa8", linewidth=0.9, alpha=0.85)
    ax.plot(freqs, s, color="#1f4fa8", linewidth=0.9)
    ax.set_xlim(float(freqs[0]), float(freqs[-1]))
    ax.set_ylim(0.0, 1.0)
    f_max = float(freqs[-1])
    step = 10 if f_max <= 60 else (20 if f_max <= 120 else 50)
    ax.set_xticks(np.arange(0, f_max + 1, step))
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_xlabel("Modulation frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    fig.tight_layout(pad=0.6)
    return _fig_to_image(fig)


def render_spectrogram(wav: np.ndarray, fs: int, mode: str, cfg: dict | None = None) -> Image.Image:
    cfg = {**DEFAULTS, **(cfg or {})}
    if mode == "demon":                     # 1D envelope spectrum
        spec, freqs = compute_demon(wav, fs, cfg)
        return render_spectrum_1d(spec, freqs, cfg)
    arr = compute_spectrogram(wav, fs, mode, cfg)   # 2D path (stft / mel)
    return render_image(arr, mode, fs, cfg)
