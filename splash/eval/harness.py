"""Evaluation harness: compose a Hydra config → run one experiment → save result.

Locates the matching manifest (by dataset/clip_len/overlap, no re-scan), builds
the model, dispatches to the paradigm runner, and writes results. New paradigms
plug in as additional ``elif`` branches here.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from ..models.factory import build_model
from ..tracking.results import save_result
from .prompting_runner import run_prompting_eval


def find_manifest(data_cfg) -> Path:
    """Locate a previously-built manifest without rescanning audio.

    Manifest filenames embed the dataset/clip_len/overlap plus a content hash;
    globbing on the known prefix finds it. Raises if ``prepare_data.py`` hasn't
    been run for these parameters.
    """
    manifest_dir = Path(data_cfg.get("manifest_dir", "outputs/manifests"))
    pattern = f"{data_cfg.get('name')}_clip{float(data_cfg.get('clip_len')):g}_ov{float(data_cfg.get('overlap')):g}_*.csv"
    matches = sorted(m for m in manifest_dir.glob(pattern) if not m.name.endswith("_spec.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No manifest matching {manifest_dir}/{pattern}. "
            f"Run: python scripts/prepare_data.py --dataset {data_cfg.name} "
            f"--root <root> --clip-len {data_cfg.clip_len} --overlap {data_cfg.overlap}"
        )
    return matches[-1]


def run(cfg) -> dict:
    """Execute one experiment from a composed Hydra config."""
    manifest = find_manifest(cfg.data)
    print(f"[harness] manifest: {manifest}")

    model = build_model(cfg.model)
    print(f"[harness] model: {model.name}  capabilities={model.capabilities}")

    eval_name = cfg.eval.name
    regime = cfg.regime

    if eval_name in ("lalm_prompting", "vlm_spectrogram"):
        media_type = "image" if eval_name == "vlm_spectrogram" else "audio"
        spec_mode = None
        use_manifest = manifest
        if media_type == "image":
            if not model.supports("image"):
                raise ValueError(f"eval={eval_name} needs an image-capable model; "
                                 f"{model.name} capabilities={model.capabilities}")
            spec_mode = cfg.spectrogram.name
            aug = manifest.with_name(f"{manifest.stem}_spec.csv")
            if not aug.exists():
                raise FileNotFoundError(
                    f"评测A needs the augmented manifest {aug.name} (with {spec_mode}_path). "
                    f"Run: python scripts/generate_spectrograms.py --dataset {cfg.data.name} "
                    f"--modes {spec_mode}")
            use_manifest = aug
        result = run_prompting_eval(
            model,
            use_manifest,
            dataset_name=cfg.data.name,
            regime=regime.name,
            shot=int(regime.get("shot", 5) or 5),
            support_seed=int(regime.get("support_seed", 0) or 0),
            target_sr=int(cfg.model.get("target_sr", 16000)),
            clip_len=float(cfg.data.clip_len),
            lang=cfg.get("lang", "en"),
            media_type=media_type,
            spec_mode=spec_mode,
            limit=int(cfg.get("limit")) if cfg.get("limit") else None,
            max_recordings=int(cfg["max_recordings"]) if cfg.get("max_recordings") else None,
            max_new_tokens=int(cfg.eval.get("max_new_tokens", 8)),
            eval_split=cfg.get("eval_split", "test"),
        )
    elif eval_name == "linear_probe":
        raise NotImplementedError("评测 C (linear probe) — Phase-later.")
    elif eval_name == "lora_finetune":
        raise NotImplementedError("评测 D (LoRA fine-tune) — Phase-later.")
    elif eval_name == "baseline_classifier":
        raise NotImplementedError("评测 E/F (classifier baselines) — Phase-later.")
    else:
        raise KeyError(f"unknown eval '{eval_name}'")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # inject spectrogram name into eval sub-dict (Hydra keeps it as separate group)
    # so experiment_name + feature column correctly capture the spec mode
    spec_name = cfg_dict.get("spectrogram", {}).get("name")
    if spec_name and "eval" in cfg_dict:
        cfg_dict["eval"]["spectrogram"] = spec_name
    json_path = save_result(
        cfg_dict,
        result["metrics_clip"],
        result["metrics_recording"],
        result["records"],
    )
    print(f"[harness] ✅ results saved: {json_path}")
    return result
