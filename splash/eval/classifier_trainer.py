"""评测 E/F: trained classifier baselines (interface skeleton).

E: audio-encoder + classifier (BEATs / WavLM + linear head) — Hummel/Huang baselines.
F: spectrogram-image classifiers (EfficientNet / CRNN / MHT-Transformer) — Cao SOTA.
Implemented in a later phase; reuses the segmentation/manifest pipeline.
"""

from __future__ import annotations


def run_classifier_baseline(*args, **kwargs):
    raise NotImplementedError("评测 E/F (classifier baselines) — implemented in a later phase.")
