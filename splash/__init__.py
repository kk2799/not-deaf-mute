"""SPLASH: systematic evaluation framework for Large Audio-Language Models (LALMs)
on Underwater Acoustic Target Recognition (UATR).

Package layout:
    data       — datasets, segmentation, manifests, splits, ShipsEar metadata
    audio      — audio I/O + resampling, STFT/DEMON spectrograms
    models     — unified UATRModel interface + per-model wrappers (LALM/VLM/baselines)
    prompting  — prompt templates, text-enrichment variants, answer parsing
    eval       — evaluation harness + per-paradigm runners (A–F)
    metrics    — classification metrics + clip→recording aggregation
    tracking   — local JSON/CSV result store + optional TensorBoard
    viz        — layer-wise curves, confusion matrices, t-SNE
"""

__version__ = "0.1.0"
