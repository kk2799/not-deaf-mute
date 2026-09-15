#!/usr/bin/env python
"""Build the ShipsEar clip manifest.

ShipsEar is flat: ``<root>/<split>/<Type>_<id>__*.wav`` — the vessel TYPE is the
filename prefix (before the first ``_``). This parses it, normalizes to the
canonical 12-class name (labels.py), segments each recording (30 s / 50 %), and
writes a manifest in the same format/naming as DeepShip so the rest of the
pipeline (generate_spectrograms, experiments) works unchanged.

Filename prefixes are normalized: Musselboat→"Mussel boat", Oceanliner→"Ocean
liner", Naturalambientnoise→"Natural ambient noise", Pilotship→"Pilot ship".
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from splash.audio.io import audio_info
from splash.data.build_manifest import manifest_filename  # reuse naming
from splash.data.labels import class_names
from splash.data.segmentation import segment_recording

DATASET = "shipsear"
# filename prefix -> canonical label name
_PREFIX_NORM = {
    "Musselboat": "Mussel boat",
    "Oceanliner": "Ocean liner",
    "Naturalambientnoise": "Natural ambient noise",
    "Pilotship": "Pilot ship",
}


def parse_label(wav_name: str, name_to_id: dict) -> int:
    prefix = wav_name.split("_", 1)[0]
    canon = _PREFIX_NORM.get(prefix, prefix)
    lid = name_to_id.get(canon)
    if lid is None:
        raise ValueError(f"unrecognized type prefix '{prefix}' (canon '{canon}') in {wav_name}")
    return lid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/datasets/audio/shipsear")
    ap.add_argument("--clip-len", type=float, default=30.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    names = class_names(DATASET, args.lang)            # {id: name}
    name_to_id = {n: i for i, n in names.items()}
    rows, fingerprint = [], []
    for split in ("train", "test"):
        split_dir = Path(args.root) / split
        if not split_dir.is_dir():
            continue
        for wav in sorted(split_dir.glob("*.wav")):
            lid = parse_label(wav.name, name_to_id)
            dur = audio_info(wav).duration
            fingerprint.append((str(wav.name), round(dur, 3)))
            for c in segment_recording(wav.stem, str(wav), dur, lid, split,
                                       clip_len=args.clip_len, overlap=args.overlap):
                r = c.as_row()
                r["label_name"] = names[lid]
                r["dataset"] = DATASET
                rows.append(r)

    fp = hashlib.sha1((DATASET + "|" + ",".join(f"{p}:{d}" for p, d in sorted(fingerprint))).encode()).hexdigest()[:12]
    out_dir = Path("outputs/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / manifest_filename(DATASET, args.clip_len, args.overlap, fp)
    df = (pd.DataFrame(rows)
          .sort_values(["split", "label_id", "recording_id", "clip_idx"])
          .reset_index(drop=True))
    df.to_csv(out, index=False)

    n_rec = df["recording_id"].nunique()
    print(f"✅ manifest: {out}")
    print(f"   recordings={n_rec}  clips={len(df)} (train={sum(df.split=='train')}, test={sum(df.split=='test')})")
    print("\n   per-class clip counts:")
    print(pd.crosstab(df["label_name"], df["split"]).to_string())


if __name__ == "__main__":
    main()
