#!/usr/bin/env python
"""Layer-probe v2: token-aware pooling of hidden states.

The v1 features mean-pool over ALL tokens (fixed prompt text included), which
discards audio-token temporal structure and dilutes the signal with constant
text tokens. v2 keeps, per layer:
    ch0 audio-token mean   ch1 audio-token max   ch2 audio-token std
    ch3 all-token mean (v1 compatibility / fallback)
Audio tokens are identified via mm_token_type_ids != 0 under the attention
mask (fallback: all attended tokens, logged loudly).

Outputs: outputs/features/layerprobe2/{model}_{spec}_{hash}/shard_*.npz with
X [n, L+1, 4, D] fp16 + y/split/rid/cidx, row order identical to v1.

Smoke first: python scripts/run_layer_probe_v2.py --models qwen2_audio --limit 6
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd

import run_layer_probe as v1
from splash.audio.io import read_segment
from splash.data.labels import class_names
from splash.eval.harness import find_manifest
from splash.models.factory import build_model
from splash.prompting.templates import build_prompt, build_fusion_prompt

DATASET = "deepship"
CLIP_LEN = 30.0
FEATURES = Path("outputs/features/layerprobe2")
SHARD = 250


_RENDER_CTX = None  # thread-mode context (kept for compatibility)
_WORKER = {"proc": None, "meta": None}


def _worker_init(model_name, model_path):
    """Spawn-clean: each worker builds its own CPU processor (no CUDA)."""
    from transformers import AutoProcessor
    _WORKER["proc"] = AutoProcessor.from_pretrained(model_path)


def _render_row(payload):
    """Runs in spawned workers: media load + processor render (CPU only)."""
    r, cols, spec, clip_samples, names = payload
    rd = dict(zip(cols, r))
    proc = _WORKER["proc"]
    parts = spec.split("+")
    if len(parts) == 1:
        media = v1.load_media(rd, spec, clip_samples)
        messages = build_prompt(media=media, label_names=names,
                                regime="zero_shot", support=None,
                                media_type="audio" if spec == "audio"
                                else "image", lang="en", enrich_text=None)
        text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=False)
        kw = dict(text=text, return_tensors="pt")
        if spec == "audio":
            kw["audio"] = [media]
        else:
            kw["images"] = [media]
    else:
        media_list = []
        if "audio" in parts:
            media_list.append((v1.load_media(rd, "audio", clip_samples),
                               "audio"))
        for p in parts:
            if p != "audio":
                media_list.append((v1.load_media(rd, p, clip_samples),
                                   "image"))
        messages = build_fusion_prompt(media_list, names, "zero_shot", None,
                                       "en", parts)
        text = proc.apply_chat_template(messages, add_generation_prompt=True,
                                        tokenize=False)
        kw = dict(text=text, return_tensors="pt")
        kw["audio"] = [m for m, t in media_list if t == "audio"] or None
        kw["images"] = [m for m, t in media_list if t == "image"] or None
    inputs = proc(**kw)
    return {k: v.numpy() for k, v in inputs.items()}


def extract_sweep(wrapper, manifest, spec, out_dir: Path, order, clip_samples,
                  limit=None, model_name=None, model_path=None):
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    names = class_names(DATASET, "en")
    media_type = "audio" if spec == "audio" else "image"
    rows_all = order[:limit] if limit else order
    n_shards = (len(rows_all) + SHARD - 1) // SHARD
    audio_tok_reported = False
    from collections import deque
    import multiprocessing as mp
    pool = None  # created once, reused across shards
    LOOKAHEAD = 24
    global _RENDER_CTX

    for s in range(n_shards):
        rows = manifest.iloc[rows_all[s * SHARD:(s + 1) * SHARD]]
        path = out_dir / f"shard_{s:04d}.npz"
        if path.exists():
            try:
                if len(np.load(path, allow_pickle=False)["y"]) == len(rows):
                    continue
            except Exception:
                print(f"  shard {s}: corrupt — recomputing", flush=True)
        row_list = [tuple(r) for r in rows.itertuples(index=False)]
        cols = tuple(rows.columns)
        payloads = [(r, cols, spec, clip_samples, names) for r in row_list]
        if not pool:
            pool = mp.get_context("spawn").Pool(
                8, initializer=_worker_init,
                initargs=(model_name, model_path))
        feats = []
        pending = deque()
        tok_idx = getattr(wrapper.model.config, "audio_token_index", None)
        if tok_idx is None:
            tok_idx = getattr(wrapper.model.config, "audio_token_id", None)
        img_idx = None
        for attr in (wrapper.model.config,
                     getattr(wrapper.model.config, "vision_config", None),
                     getattr(wrapper.model.config, "audio_config", None)):
            if attr is not None:
                img_idx = getattr(attr, "image_token_id", None)
                if img_idx is not None:
                    break

        def to_device(raw):
            out = {}
            for k, v in raw.items():
                t = torch.from_numpy(v)
                if t.dtype.is_floating_point:
                    out[k] = t.to(wrapper.device, wrapper.torch_dtype)
                else:
                    out[k] = t.to(wrapper.device)
            return out

        # BATCH MUST stay 1: omni Thinker forward at batch>1 systematically
        # corrupts features (single-layer LR curve collapses 0.69->0.59,
        # verified 2026-09-18). Speed comes from the spawn render pool only.
        BATCH = 1
        MIXED = len(spec.split("+")) > 1  # render mixed inputs in-process:
        # spawn-worker output corrupts audio+image batches (grid_thw error,
        # verified 2026-09-20); native wrapper._render is correct and cheap.
        i = 0
        while i < len(payloads):
            if MIXED:
                r, cols, _spec, _cs, _names = payloads[i]
                rd = dict(zip(cols, r))
                parts = spec.split("+")
                media_list = []
                if "audio" in parts:
                    media_list.append((v1.load_media(rd, "audio", clip_samples), "audio"))
                for p in parts:
                    if p != "audio":
                        media_list.append((v1.load_media(rd, p, clip_samples), "image"))
                messages = build_fusion_prompt(media_list, names, "zero_shot",
                                               None, "en", parts)
                inputs = wrapper._render(messages)
                i += 1
            else:
                while len(pending) < LOOKAHEAD and i + len(pending) < len(payloads):
                    pending.append(pool.apply_async(
                        _render_row, (payloads[i + len(pending)],)))
                raws = []
                while pending and len(raws) < BATCH:
                    raws.append(pending.popleft().get())
                inputs = {}
                for k in raws[0]:
                    if k == "input_features" or "feature" in k or k in (
                            "input_ids", "attention_mask", "feature_attention_mask"):
                        inputs[k] = torch.cat([to_device({k: r[k]})[k]
                                               for r in raws], dim=0)
                    else:
                        per = [r[k] for r in raws]
                        if all(p.shape == per[0].shape for p in per):
                            inputs[k] = torch.stack(
                                [torch.from_numpy(p) for p in per]).to(
                                wrapper.device)
                        else:  # ragged aux keys: only keep batch-of-one
                            inputs = to_device(raws[0])
                            raws = raws[:1]
                            break
            with torch.inference_mode():
                out = wrapper.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states                 # tuple(L+1) [B,T,D]
            am = inputs.get("attention_mask")
            am = am.bool() if am is not None else None
            ids = inputs["input_ids"]
            B = hs[0].shape[0]
            audio_m = (ids == tok_idx) if tok_idx is not None else torch.zeros_like(ids, dtype=torch.bool)
            if am is not None:
                audio_m &= am
            image_m = (ids == img_idx) if img_idx is not None else None
            if image_m is not None and am is not None:
                image_m &= am
            if not audio_tok_reported:
                print(f"  [diag] batch={B} audio_tokens={int(audio_m[0].sum())} "
                      f"image_tokens={int(image_m[0].sum()) if image_m is not None else 0} "
                      f"attended={int(am[0].sum()) if am is not None else -1}",
                      flush=True)
                audio_tok_reported = True
            pooled_all = []
            for h in hs:  # [B, T, D]
                hf = h.float()
                m = audio_m.unsqueeze(-1).to(hf.dtype)
                cnt = audio_m.sum(1, keepdim=True).clamp(min=1)
                amean = (hf * m).sum(1) / cnt
                amax = hf.masked_fill(
                    ~audio_m.unsqueeze(-1), float("-inf")).max(1).values
                var = ((hf - amean.unsqueeze(1)) ** 2 * m).sum(1) / \
                    (audio_m.sum(1, keepdim=True) - 1).clamp(min=1)
                chans = [amean, amax, var.sqrt()]
                if image_m is not None and int(image_m.sum()) > 0:
                    mi = image_m.unsqueeze(-1).to(hf.dtype)
                    cnti = image_m.sum(1, keepdim=True).clamp(min=1)
                    imean = (hf * mi).sum(1) / cnti
                    imax = hf.masked_fill(
                        ~image_m.unsqueeze(-1), float("-inf")).max(1).values
                    ivar = ((hf - imean.unsqueeze(1)) ** 2 * mi).sum(1) / \
                        (image_m.sum(1, keepdim=True) - 1).clamp(min=1)
                    chans += [imean, imax, ivar.sqrt()]
                ma = am.unsqueeze(-1).to(hf.dtype) if am is not None else 1.0
                cnta = am.sum(1, keepdim=True).clamp(min=1) if am is not None \
                    else torch.tensor(float(hf.shape[1]), device=hf.device)
                chans.append((hf * ma).sum(1) / cnta)
                pooled_all.append(torch.stack(chans, dim=1))   # [B, C, D]
            P = torch.stack(pooled_all, dim=1).to(torch.float16).cpu()  # [B,L+1,C,D]
            for b in range(B):
                feats.append(P[b])
            del out, hs, pooled_all, P
            if MIXED:
                pass  # i already advanced in the mixed branch
            else:
                i += len(raws)
        X = torch.stack(feats).numpy()                 # [n, L+1, 4, D]
        payload = dict(X=X, y=rows["label_id"].to_numpy(np.int16),
                       split=(rows["split"] == "test").to_numpy(np.int8),
                       rid=rows["recording_id"].to_numpy().astype(str),
                       cidx=rows["clip_idx"].to_numpy(np.int32))
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        os.replace(tmp, path)
        print(f"  shard {s + 1}/{n_shards}: {len(rows)} clips → {X.shape}",
              flush=True)
        del feats, X
        gc.collect()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="qwen2_audio",
                    help="comma list from v1 registry")
    ap.add_argument("--spec", default=None,
                    help="override spec; supports multimodal combos like "
                         "audio+mel or audio+mel+stft+demon")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--full-train", action="store_true")
    ap.add_argument("--device", default=None,
                    help="override single-GPU placement (e.g. cuda:1)")
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN,
                              "overlap": 0.5})
    clip_samples = int(CLIP_LEN * 16000)
    manifest_df = pd.read_csv(manifest)
    order = v1.ordered_rows(manifest_df, max_train=3000,
                            full_train=args.full_train)
    tag = "full" if args.full_train else "3k"
    print(f"[v2] manifest={manifest.name} rows={len(order)} ({tag}) "
          f"limit={args.limit}", flush=True)

    for model_name in args.models.split(","):
        cfg = v1.MODEL_SPECS[model_name]
        kw = dict(cfg["kwargs"])
        if args.device and "device" in kw:
            kw["device"] = args.device
        wrapper = build_model({"name": model_name, "path": cfg["path"], **kw})
        specs = [args.spec] if args.spec else ["audio"]
        # image inputs live in the spec-augmented manifest (<stem>_spec.csv)
        if any(p != "audio" for p in specs[0].split("+")):
            manifest_df = pd.read_csv(
                manifest.with_name(f"{manifest.stem}_spec.csv"))
            order = v1.ordered_rows(manifest_df, max_train=3000,
                                    full_train=args.full_train)
            print(f"[v2] using spec manifest for {specs[0]} "
                  f"(rows={len(order)})", flush=True)
        for spec in specs:
            if spec != "audio" and model_name == "qwen2_audio":
                print(f"[skip] {model_name}/{spec} (q2a has no image route)",
                      flush=True)
                continue
            safe = spec.replace("+", "-")
            out_dir = FEATURES / f"{model_name}_{safe}_{manifest.stem}_{tag}"
            extract_sweep(wrapper, manifest_df, spec, out_dir, order,
                          clip_samples, args.limit, model_name, cfg["path"])
        del wrapper
        gc.collect()
    print("[v2] extraction done", flush=True)


if __name__ == "__main__":
    main()
