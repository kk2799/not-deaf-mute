"""评测 C: layer-wise linear probe (interface skeleton).

Extracts per-layer hidden states from an LALM over the train/test clips and
trains a logistic-regression probe per layer, producing the F1-vs-layer curve
that reveals where UATR-relevant structure lives inside the model. Implemented
in a later phase; the data/model/metrics layers are already reusable here.
"""

from __future__ import annotations


def run_linear_probe(*args, **kwargs):
    raise NotImplementedError("评测 C (layer-wise linear probe) — implemented in a later phase.")
