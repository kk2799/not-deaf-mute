#!/usr/bin/env python
"""SPLASH experiment entry point (Hydra-driven).

Composes a config from the ``conf/`` groups and runs one experiment via
``splash.eval.harness.run``. Results are written under ``outputs/results/``.

Examples
--------
    # zero-shot 评测 B on DeepShip (defaults)
    python scripts/run_eval.py

    # few-shot, k=5
    python scripts/run_eval.py regime=few_shot regime.shot=5

    # quick validation on 40 test clips
    python scripts/run_eval.py limit=40

    # sweep over shot counts (writes one result row each)
    python scripts/run_eval.py -m regime=few_shot regime.shot=1,5,10,20

    # duration ablation later (re-run prepare_data.py for each clip_len first)
    python scripts/run_eval.py data.clip_len=10
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from splash.eval.harness import run


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
