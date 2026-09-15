"""Model factory: build a UATRModel from a (Hydra/OmegaConf) model config.

Adding a new model = (1) write a wrapper class in this package, (2) add one
line to ``_load_registry`` here, (3) add a ``conf/model/<name>.yaml``. Nothing
else in the framework changes.

Registry is built lazily (wrapper modules are imported on first use) so the
heavy transformers stack only loads when a model is actually built.
"""

from __future__ import annotations

from .base import UATRModel

_REGISTRY: dict[str, type[UATRModel]] = {}
_LOADED = False


def _load_registry() -> None:
    """Import wrapper classes and map names → classes (runs once)."""
    global _LOADED
    if _LOADED:
        return
    from .lalm import Qwen2AudioModel
    from .vlm import Qwen3VLModel
    from .omni import Qwen3OmniModel
    from .baselines import BEATsEncoder, WavLMEncoder

    _REGISTRY.update(
        {
            "qwen2_audio": Qwen2AudioModel,
            "qwen3_vl": Qwen3VLModel,
            "qwen3_vl_8b": Qwen3VLModel,
            "qwen3_vl_32b": Qwen3VLModel,
            "qwen3_omni_30b": Qwen3OmniModel,
            "beats_iter3_plus": BEATsEncoder,
            "wavlm_large": WavLMEncoder,
        }
    )
    _LOADED = True


def available_models() -> list[str]:
    _load_registry()
    return sorted(_REGISTRY)


def build_model(cfg) -> UATRModel:
    """Build from a config with at least ``name`` and ``path`` keys.

    All other keys are forwarded to the wrapper constructor except ``name``.
    """
    _load_registry()
    name = cfg["name"]
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. available: {available_models()}")
    cls = _REGISTRY[name]
    kwargs = {k: v for k, v in dict(cfg).items() if k not in {"name", "path"}}
    return cls(model_path=cfg["path"], **kwargs)
