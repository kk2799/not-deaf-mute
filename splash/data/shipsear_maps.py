"""ShipsEar granularity mappings: 12-class (native) → 9-class → 5-class reporting.

Per project spec, ShipsEar results are reported at THREE granularities:
  - **12-class**: all native vessel types (incl. Pilot ship, Trawler, Tugboat).
  - **9-class** : only the 9 "mapping-participating" types, at native granularity
                 (Pilot ship / Trawler / Tugboat samples EXCLUDED).
  - **5-class** : those 9 types merged into 5 super-classes A–E (the 3 excluded
                 types still excluded). Predictions of an excluded type count as wrong.

5-class mapping (user-specified):
  A = Dredger + Fishboat + Mussel boat
  B = Motorboat + Sailboat
  C = Passengers
  D = Ocean liner + RORO
  E = Natural ambient noise
"""

from __future__ import annotations

# native type -> 5-class super-class letter
SHIPSEAR_5CLASS: dict[str, str] = {
    "Dredger": "A", "Fishboat": "A", "Mussel boat": "A",
    "Motorboat": "B", "Sailboat": "B",
    "Passengers": "C",
    "Ocean liner": "D", "RORO": "D",
    "Natural ambient noise": "E",
}
# 5-class letter -> human description
SHIPSEAR_5CLASS_NAMES: dict[str, str] = {
    "A": "A (Dredger/Fishboat/Mussel boat)",
    "B": "B (Motorboat/Sailboat)",
    "C": "C (Passengers)",
    "D": "D (Ocean liner/RORO)",
    "E": "E (Natural ambient noise)",
}
# the 9 participating native types
SHIPSEAR_9: list[str] = list(SHIPSEAR_5CLASS.keys())
# excluded from 5/9-class (only counted in 12-class)
SHIPSEAR_EXCLUDED: list[str] = ["Pilot ship", "Trawler", "Tugboat"]
