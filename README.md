# Not Deaf, but Mute

**Why Large Audio-Language Models Fail at Underwater Acoustic Target Recognition**

Artifact release for the ICASSP 2027 paper. Four Qwen-lineage multimodal
LLMs (Qwen2-Audio, Qwen3-VL-8B/32B, Qwen3-Omni) take a 44-condition
listening/looking/fusion exam on DeepShip and ShipsEar; a diagnostic
triad (silence / shuffle / prior) localizes the failure to the
generative readout, while linear probes on the same forward passes
decode macro-F1 0.65--0.69.

## What is here

| Path | Content |
|---|---|
| `splash/` | The framework: manifest building, unified model wrappers (LALM/VLM/Omni/BEATs/WavLM), prompting harness, metrics, results store |
| `conf/` | Hydra configs (data × model × eval × regime × spectrogram) |
| `scripts/` | All experiment entry points: prompting eval, per-layer probes, baselines, LoRA, steering, the three diagnostic tests, fusion, analyses, paper figures |
| `manifests/` | Clip manifests (content-hash-named CSVs; `label_id`, `split`, spectrogram paths) |
| `predictions/generative/` | Per-clip predictions of the 44 generative conditions (JSON: full config, confusion matrix, per-class F1, per-clip `choice_logprobs`) + option-order and null-input diagnostics; `results.csv` = one row per experiment |
| `predictions/probes_fusion/` | Test-set predictions of the seven single-source probes and the fusion arms (A1--A4, BEATs variants) |
| `predictions/analyses/` | Aggregated numbers behind the paper: fusion upgrade, honest learning curves, Omni audio-path band, recording-level aggregation, WavLM honest selection, ShipsEar audit, bootstrap CIs |
| `predictions/ll_scoring/` | Class-name log-likelihood control (letter-free): per-clip scores, raw and length-normalized metrics |
| `shipsear_letter_layout.csv` | The ShipsEar A--L letter-to-class layout (options are lettered in sorted label-id order) |

Models were read from `/models`, datasets from `/datasets/audio` — neither
is redistributed here (DeepShip/ShipsEar are available from their owners).

## Environment

Everything runs in the provided container layout: torch 2.4.1+cu118,
transformers 5.x, peft, accelerate (see `requirements.txt`, a version
anchor rather than an install list). `PYTHONPATH=<repo root>` makes
`import splash` work without installation.

## Reproducing

```bash
# data prep (content-hash cached; re-runs are free)
python scripts/prepare_data.py --dataset deepship --root /datasets/audio/deepship
python scripts/build_shipsear_manifest.py --root /datasets/audio/shipsear
python scripts/generate_spectrograms.py --dataset deepship

# the exam (Hydra; one generative condition)
python scripts/run_eval.py                                        # zero-shot default
python scripts/run_eval.py regime=few_shot regime.shot=5

# readouts
python scripts/run_layer_probe.py                                 # frozen per-layer probes
python scripts/run_baseline_probe.py                              # BEATs/WavLM anchors
python scripts/run_fusion_upgrade.py                              # 7-source fusion vs BEATs
python scripts/run_lora_vl8b_v3b.py --specs mel --device cuda:1    # LoRA

# the three diagnostic tests
python scripts/run_null_input.py                                  # silence test
python scripts/run_option_order_ablation.py                       # shuffle test
python scripts/run_prior_provenance.py                            # prior test
python scripts/run_steering.py                                    # probe-direction steering
```

Every experiment name in `results.csv` is self-describing
(`<eval>_<model>_<dataset>_<regime>_shot<k>_clip<len>_lang…_enrich…`);
the JSON with the same name holds its per-clip predictions.

## Statistical procedures

**TOST equivalence test (7-source fusion vs.\ BEATs single layer).**
Resampling unit: test *recording* (153 recordings; paired clip
outcomes retained within each recording). Procedure: draw B = 10,000
bootstrap resamples of the recordings with replacement (NumPy
`default_rng(0)`, matching `outputs/pilots/round1_analyses/
a1_a2_bootstrap.py`); per resample compute the accuracy difference
(fusion minus BEATs) over the resampled clip multiset. Equivalence
bounds: [−0.02, +0.02] accuracy (±2 percentage points). One-sided
p-values are the bootstrap tail fractions P(Δ ≥ +0.02) and P(Δ ≤
−0.02); TOST p = max of the two. Result: 90% percentile CI
[−1.56, +1.54] pt, TOST p = 0.019 — the CI construction is
percentile; the clip-level exact McNemar test (p = 1.0, 256:257
discordant pairs) is reported alongside and tests exact equality, a
different null. Script: `scripts/tost_and_shipsear_f1.py`; raw
inputs: `predictions/probes_fusion/{A1,s1_beats}.json`; summary:
`predictions/analyses/round2_analyses.json`.

## Citation

```bibtex
@inproceedings{liu2027notdeafmute,
  title     = {Not Deaf, but Mute: Why Large Audio-Language Models Fail at
               Underwater Acoustic Target Recognition},
  author    = {Liu, Zhengkun and Liang, Yunpeng and Wang, Zixuan and
               Guo, Yutong and Ren, Jiawei and Xu, Ji},
  booktitle = {Proc. IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP)},
  year      = {2027}
}
```
