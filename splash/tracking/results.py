"""Local experiment result store: one summary row per experiment in a CSV +
a detailed JSON (full metrics, confusion matrix, per-clip predictions).

No external services. The CSV is the paper's "main results table" seed; the JSON
holds everything needed to redraw figures / recompute downstream.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("outputs/results")


def experiment_name(cfg) -> str:
    """Deterministic name from the composed config."""
    parts = [
        cfg.get("eval", {}).get("name", "eval") if isinstance(cfg.get("eval"), dict) else cfg.get("eval", "eval"),
        cfg.get("model", {}).get("name", "model") if isinstance(cfg.get("model"), dict) else cfg.get("model", "model"),
        cfg.get("data", {}).get("name", "data") if isinstance(cfg.get("data"), dict) else cfg.get("data", "data"),
        cfg.get("regime", {}).get("name", "regime") if isinstance(cfg.get("regime"), dict) else cfg.get("regime", "regime"),
    ]
    # append distinctive scalars (shot, clip_len, lang, enrich) when present
    extra = []
    regime = cfg.get("regime", {})
    if isinstance(regime, dict) and regime.get("shot") is not None:
        extra.append(f"shot{regime['shot']}")
    data = cfg.get("data", {})
    if isinstance(data, dict):
        if data.get("clip_len") is not None:
            extra.append(f"clip{data['clip_len']}")
    if cfg.get("max_recordings") is not None:
        extra.append(f"rec{cfg['max_recordings']}")
    ev = cfg.get("eval", {})
    if isinstance(ev, dict) and ev.get("spectrogram"):
        extra.append(f"spec{ev['spectrogram']}")
    if isinstance(ev, dict) and ev.get("fusion"):
        extra.append(f"fusion{ev['fusion']}")
    for k in ("lang", "enrich"):
        v = cfg.get(k)
        if v:
            extra.append(f"{k}{v}")
    return "_".join(parts + extra)


def save_result(
    cfg: dict,
    metrics_clip: dict,
    metrics_rec: dict | None,
    predictions: list[dict],
    out_dir: str | Path = RESULTS_DIR,
) -> Path:
    """Persist detailed JSON + append a summary CSV row. Returns the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = experiment_name(cfg)
    json_path = out_dir / f"{name}.json"

    payload = {
        "experiment": name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": _jsonable(cfg),
        "metrics_clip": _jsonable(metrics_clip),
        "metrics_recording": _jsonable(metrics_rec),
        "n_predictions": len(predictions),
        "predictions": [_jsonable(p) for p in predictions],
    }
    _write_text(json_path, json.dumps(payload, indent=2, ensure_ascii=False))

    # summary row
    rec = metrics_rec or metrics_clip
    row = {
        "timestamp": payload["timestamp"],
        "experiment": name,
        "eval": _cfg_get(cfg, "eval", "name"),
        "feature": (cfg.get("eval", {}).get("spectrogram") or "audio"),
        "model": _cfg_get(cfg, "model", "name"),
        "dataset": _cfg_get(cfg, "data", "name"),
        "regime": _cfg_get(cfg, "regime", "name"),
        "shot": _cfg_get(cfg, "regime", "shot"),
        "clip_len": _cfg_get(cfg, "data", "clip_len"),
        "n_clips": metrics_clip.get("n"),
        "n_recordings": (metrics_rec or {}).get("n"),
        "clip_acc": metrics_clip.get("accuracy"),
        "clip_f1_macro": metrics_clip.get("f1_macro"),
        "rec_acc": (metrics_rec or {}).get("accuracy"),
        "rec_f1_macro": (metrics_rec or {}).get("f1_macro"),
    }
    csv_path = out_dir / "results.csv"
    # UPSERT (replace any stale row for the same experiment, e.g. after data change)
    # + backfill 'feature' for old rows + sort (model → feature → regime)
    # with retry: Docker-Desktop/WSL2 + Windows mounts can transiently deny permission.
    _feat_order = {"audio": 0, "mel": 1, "stft": 2, "demon": 3}
    _reg_order = {"zero_shot": 0, "few_shot": 1}

    def _backfill_feature(df):
        """Derive 'feature' from experiment name (always, most robust)."""
        if "feature" not in df.columns:
            df["feature"] = "audio"
        for i in df.index:
            exp = str(df.loc[i, "experiment"])
            for f in ("mel", "stft", "demon"):
                if f"spec{f}" in exp:
                    df.at[i, "feature"] = f
                    break
            else:
                df.at[i, "feature"] = "audio"
        return df

    def _sort_results(df):
        df = _backfill_feature(df)
        df["_f"] = df["feature"].map(_feat_order).fillna(99)
        df["_r"] = df["regime"].map(_reg_order).fillna(99)
        df = df.sort_values(["model", "_f", "_r"]).drop(columns=["_f", "_r"])
        return df.reset_index(drop=True)

    for attempt in range(6):
        try:
            if csv_path.exists():
                prev = pd.read_csv(csv_path)
                prev = prev[prev["experiment"] != name]  # drop stale row(s)
                out_df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
                out_df = _sort_results(out_df)
                out_df.to_csv(csv_path, index=False)
            else:
                pd.DataFrame([row]).to_csv(csv_path, index=False)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)  # 1,2,4,8,16s backoff
    return json_path


def _write_text(path: Path, text: str, retries: int = 6) -> None:
    """write_text with PermissionError retry (WSL2/Windows mount robustness)."""
    for attempt in range(retries):
        try:
            path.write_text(text)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _cfg_get(cfg: dict, group: str, key: str):
    g = cfg.get(group, {})
    return g.get(key) if isinstance(g, dict) else None


def _jsonable(obj):
    """Best-effort conversion of numpy/omegaconf objects to JSON-native types."""
    import numpy as np
    try:
        from omegaconf import OmegaConf
        if hasattr(obj, "_metadata") or isinstance(obj, dict):
            try:
                obj = OmegaConf.to_container(obj, resolve=True)
            except Exception:
                pass
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
