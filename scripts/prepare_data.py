#!/usr/bin/env python
"""Build a clip manifest for a UATR dataset.

Scans ``<root>/<split>/<label_id>/*.wav``, segments each recording into
fixed-length overlapping clips, and writes a cached manifest CSV. Re-running
with the same params loads the cache; changing ``--clip-len`` / ``--overlap``
or the underlying audio produces a new manifest (content-hashed filename).

Examples
--------
    python scripts/prepare_data.py --dataset deepship \
        --root /datasets/audio/deepship --clip-len 30 --overlap 0.5
    # duration ablation later:
    python scripts/prepare_data.py --dataset deepship --clip-len 10 --overlap 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from splash.data.build_manifest import build_manifest, manifest_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Dataset key in splash.data.labels (e.g. deepship)")
    ap.add_argument("--root", required=True, help="Dataset root containing <split>/<label_id>/*.wav")
    ap.add_argument("--clip-len", type=float, default=30.0, help="Clip length in seconds (default 30)")
    ap.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction in [0,1) (default 0.5)")
    ap.add_argument("--out-dir", default="outputs/manifests", help="Manifest output directory")
    ap.add_argument("--lang", default="en", choices=["en", "zh"])
    ap.add_argument("--force", action="store_true", help="Rebuild even if a cached manifest exists")
    args = ap.parse_args()

    df, path = build_manifest(
        data_root=args.root,
        dataset_name=args.dataset,
        clip_len=args.clip_len,
        overlap=args.overlap,
        out_dir=args.out_dir,
        lang=args.lang,
        force=args.force,
    )

    n_rec = df["recording_id"].nunique()
    print(f"\n✅ manifest written: {path}")
    print(f"   dataset={args.dataset}  clip_len={args.clip_len}s  overlap={args.overlap}")
    print(f"   recordings={n_rec}  clips={len(df)}  (train={sum(df.split=='train')}, test={sum(df.split=='test')})")
    print("\n   per-class clip counts:")
    print(manifest_summary(df).to_string())

    # Leakage sanity: each recording must belong to exactly one split.
    bad = df.groupby("recording_id")["split"].nunique()
    leaked = bad[bad > 1]
    if len(leaked):
        print(f"\n⚠️  {len(leaked)} recordings appear in multiple splits (leakage!)")
    else:
        print(f"\n🟢 no train/test leakage: all {n_rec} recordings are split-pure")


if __name__ == "__main__":
    main()
