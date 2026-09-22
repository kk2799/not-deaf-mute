# Not Deaf, but Mute

**Why Large Audio-Language Models Fail at Underwater Acoustic Target Recognition**

Four Qwen-lineage multimodal LLMs (Qwen2-Audio, Qwen3-VL-8B/32B, Qwen3-Omni) take a 44-condition
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
