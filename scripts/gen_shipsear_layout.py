#!/usr/bin/env python
"""Generate the ShipsEar letter-to-class layout table (A-L by sorted label-id).

This is the layout promised in the paper's artifact footnote: options are
lettered in sorted label-id order, so letter = chr(65 + rank(label_id)).
"""
import csv
from collections import Counter

SRC = "/workspace/SPLASH/outputs/manifests/shipsear_clip30_ov0.5_15f41dbed293.csv"
DST = "/workspace/SPLASH/outputs/pilots/shipsear_letter_layout.csv"

rows = list(csv.DictReader(open(SRC)))
names, ntr, nte = {}, Counter(), Counter()
for r in rows:
    lid = int(r["label_id"])
    names[lid] = r["label_name"]
    (ntr if r["split"] == "train" else nte)[lid] += 1

with open(DST, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["letter", "label_id", "class_name", "n_train", "n_test"])
    for i, lid in enumerate(sorted(names)):
        w.writerow([chr(65 + i), lid, names[lid], ntr[lid], nte[lid]])

print(open(DST).read())
