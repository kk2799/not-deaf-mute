#!/usr/bin/env python
"""Pre-render spectrogram images for every clip in a manifest.

Reuses the project's validated DSP (`fea-extract/fea_extract.py`) via
``splash.audio.spectrogram`` so images match the features other experiments use.
Default modes: **stft, mel, demon** (300 mel filters, 6000–8000 Hz DEMON band).

Output (deterministic, cached — skips existing PNGs):
    outputs/spectrograms/<mode>/<dataset>_<clip><len>_ov<ov>_<hash>/<split>/<label>/<rec>_c<idx>.png
Plus an augmented manifest ``<base>_spec.csv`` with a ``<mode>_path`` column each.

Examples
--------
    python scripts/generate_spectrograms.py --dataset deepship
    python scripts/generate_spectrograms.py --dataset deepship --modes mel,demon
    python scripts/generate_spectrograms.py --dataset deepship --limit 8        # sample
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from splash.audio import spectrogram as spec
from splash.audio.io import read_segment

MANIFEST_DIR = Path("outputs/manifests")
SPEC_DIR = Path("outputs/spectrograms")


def find_manifest(dataset: str, clip_len: float, overlap: float) -> Path:
    pat = f"{dataset}_clip{float(clip_len):g}_ov{float(overlap):g}_*.csv"
    # exclude augmented _spec manifests from a previous spectrogram run
    matches = sorted(m for m in MANIFEST_DIR.glob(pat) if not m.name.endswith("_spec.csv"))
    if not matches:
        raise FileNotFoundError(f"No manifest for {MANIFEST_DIR}/{pat}; run prepare_data.py first.")
    return matches[-1]


def fingerprint(dataset: str, cfg: dict) -> str:
    keys = ["frame_length_ms", "frame_shift_ms", "high_freq", "low_freq", "n_mels",
            "demon_low_frequency", "demon_high_frequency", "demon_freq_resolution",
            "cmap", "p_low", "p_high", "apply_mvn"]
    payload = dataset + "|" + "|".join(f"{k}={cfg.get(k)}" for k in keys)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clip-len", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--modes", default="stft,mel,demon")
    ap.add_argument("--sample-rate", type=int, default=22500, help="feature sample rate (fea_extract uses 22500)")
    ap.add_argument("--n-mels", type=int, default=300)
    ap.add_argument("--high-freq", type=float, default=8000.0)
    ap.add_argument("--low-freq", type=float, default=10.0)
    ap.add_argument("--demon-low", type=float, default=6000.0)
    ap.add_argument("--demon-high", type=float, default=8000.0)
    ap.add_argument("--demon-res", type=float, default=250.0)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--no-mvn", action="store_true", help="skip per-clip waveform MVN")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    manifest = find_manifest(args.dataset, args.clip_len, args.overlap)
    df = pd.read_csv(manifest)
    if args.limit:
        df = df.head(args.limit)
    print(f"manifest: {manifest.name}  clips={len(df)}  modes={modes}  fs={args.sample_rate}  n_mels={args.n_mels}")

    base_cfg = dict(spec.DEFAULTS)
    base_cfg.update(n_mels=args.n_mels, high_freq=args.high_freq, low_freq=args.low_freq,
                    demon_low_frequency=args.demon_low, demon_high_frequency=args.demon_high,
                    demon_freq_resolution=args.demon_res, cmap=args.cmap,
                    apply_mvn=not args.no_mvn)

    plans = []
    for mode in modes:
        fp = fingerprint(args.dataset, {**base_cfg, "_mode": mode})
        outdir = SPEC_DIR / mode / f"{args.dataset}_clip{args.clip_len:g}_ov{args.overlap:g}_{fp}"
        outdir.mkdir(parents=True, exist_ok=True)
        plans.append((mode, outdir))
        print(f"  [{mode}] -> {outdir}")

    clip_samples = int(round(args.clip_len * args.sample_rate))
    paths = {mode: [None] * len(df) for mode in modes}

    done = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="spectrograms", unit="clip"):
        outpaths = {mode: outdir / row["split"] / str(row["label_id"]) / f"{row['recording_id']}_c{row['clip_idx']}.png"
                    for mode, outdir in plans}
        # resume: skip clips whose all-mode PNGs already exist (no recompute)
        if not args.force and all(p.exists() for p in outpaths.values()):
            for mode in modes:
                paths[mode][i] = str(outpaths[mode])
            done += 1
            continue
        wav = read_segment(row["path"], row["start"], row["end"],
                           target_sr=args.sample_rate, pad_to_samples=clip_samples)
        for mode, outdir in plans:
            outpath = outpaths[mode]
            outpath.parent.mkdir(parents=True, exist_ok=True)
            if args.force or not outpath.exists():
                img = spec.render_spectrogram(wav, args.sample_rate, mode, base_cfg)
                img.save(outpath)
            paths[mode][i] = str(outpath)
    if done:
        print(f"(resumed: {done} clips already had all-mode PNGs)")

    out = df.copy()
    for mode in modes:
        out[f"{mode}_path"] = paths[mode]
    aug = MANIFEST_DIR / f"{manifest.stem}_spec.csv"
    out.to_csv(aug, index=False)
    print(f"\n✅ augmented manifest: {aug}")
    for mode in modes:
        print(f"   {mode}: {sum(p is not None for p in paths[mode])} images")

    # done flag (for monitoring / resume watcher)
    (SPEC_DIR / f"{args.dataset}_clip{args.clip_len:g}_ov{args.overlap:g}.done").write_text(
        f"modes={','.join(modes)} clips={len(df)}\n")


if __name__ == "__main__":
    main()
