#!/usr/bin/env python
"""LoRA v3: VL-8B adapters on attention AND MLP (language model + vision tower).

Extends v2 (which adapted attention only, mel 0.594):
  1. LM MLP adapters — gate/up/down_proj on every decoder layer.
  2. Vision MLP adapters — Qwen3-VL vision tower uses FUSED linear_fc1/fc2
     (NOT gate/up/down like the LM); verified via meta-device instantiation.
  3. rank 32 kept (more modules, same per-module rank).
  4. regime name = "lora_v3" → separate results.csv rows.

Baseline comparison (v2 to beat):
  mel:   v1 0.578 / v2 0.594 / probe 0.675 → v3 target ≥ 0.61

Usage: python scripts/run_lora_vl8b_v3.py --specs mel
"""
from __future__ import annotations

import argparse, gc, os, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from splash.eval.harness import find_manifest
from splash.eval.prompting_runner import run_prompting_eval
from splash.data.labels import class_names
from splash.models.factory import build_model
from splash.prompting.templates import option_layout, build_prompt
from splash.tracking.results import experiment_name, save_result

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")
LORA_DIR = Path("outputs/lora_v3")
SEED = 42

# v3: LM attention + LM MLP + vision attention (fused qkv/proj) + vision MLP (fused fc1/fc2)
LORA_TARGET = (
    r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|"
    r"model\.visual\.blocks\.\d+\.attn\.(qkv|proj)"
    r"|"
    r"model\.visual\.blocks\.\d+\.mlp\.(linear_fc1|linear_fc2)"
)


def build_target(tokenizer, letter: str) -> list[int]:
    return tokenizer.encode(" " + letter, add_special_tokens=False)


def render_sample(wrapper, image, names, letter):
    messages = build_prompt(media=image, label_names=names, regime="zero_shot",
                            support=None, media_type="image", lang="en", enrich_text=None)
    text = wrapper.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = wrapper.processor(text=text, images=[image], return_tensors="pt")
    target = build_target(wrapper.processor.tokenizer, letter)
    ids = inputs["input_ids"][0].tolist() + target
    labels = [-100] * (len(ids) - len(target)) + target
    feats = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    mm = feats.get("mm_token_type_ids")
    if mm is not None:
        feats["mm_token_type_ids"] = torch.cat(
            [mm, torch.zeros((mm.shape[0], len(target)), dtype=mm.dtype)], dim=1)
    return ids, labels, feats


def collate(samples, pad_id):
    maxlen = max(len(s[0]) for s in samples)
    input_ids, labels, am = [], [], []
    for ids, lab, _ in samples:
        p = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * p)
        labels.append(lab + [-100] * p)
        am.append([1] * len(ids) + [0] * p)
    batch = {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
             "attention_mask": torch.tensor(am)}
    for k in samples[0][2]:
        vals = [s[2][k] for s in samples]
        if k == "mm_token_type_ids":
            vals = [torch.cat([v, torch.zeros((v.shape[0], maxlen - v.shape[1]),
                              dtype=v.dtype)], dim=1) for v in vals]
        if all(v.shape == vals[0].shape for v in vals):
            batch[k] = torch.cat(vals, dim=0)
        else:
            raise RuntimeError(f"ragged '{k}' — use --batch-size 1")
    return batch


def save_state(peft_model, optimizer, scheduler, epoch, pos, gs, state_file):
    from peft import get_peft_model_state_dict
    bundle = {"adapter": get_peft_model_state_dict(peft_model),
              "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
              "epoch": epoch, "pos": pos, "global_step": gs}
    tmp = state_file.with_suffix(".tmp.pt")
    torch.save(bundle, tmp)
    os.replace(tmp, state_file)


def train_one_spec(wrapper, aug_df, spec, args, names, letters, label_order):
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    train_df = aug_df[aug_df["split"] == "train"].reset_index(drop=True)
    n = args.limit or len(train_df)
    total_steps = (n + args.batch_size - 1) // args.batch_size * args.epochs // args.grad_accum

    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      bias="none", target_modules=LORA_TARGET, task_type="CAUSAL_LM")
    peft_model = get_peft_model(wrapper.model, lcfg)
    peft_model.print_trainable_parameters()
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.train()

    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup)

    state_dir = LORA_DIR / f"qwen3_vl_8b_spec{spec}"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "state.pt"
    start_ep, start_pos, gs = 0, 0, 0
    if state_file.exists():
        b = torch.load(state_file, map_location="cpu")
        start_ep, start_pos, gs = b["epoch"], b["pos"], b["global_step"]
        set_peft_model_state_dict(peft_model, b["adapter"])
        optimizer.load_state_dict(b["optimizer"])
        scheduler.load_state_dict(b["scheduler"])
        print(f"[resume] {spec} ep={start_ep} pos={start_pos} gs={gs}")

    pad_id = wrapper.processor.tokenizer.pad_token_id or wrapper.processor.tokenizer.eos_token_id
    clip_samples = int(CLIP_LEN * 16000)

    for ep in range(start_ep, args.epochs):
        perm = np.random.default_rng(SEED + ep).permutation(n)
        micro = [perm[i:i + args.batch_size] for i in range(0, n, args.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        for mi, idxs in enumerate(micro):
            if start_pos > 0 and mi * args.batch_size < start_pos:
                continue
            samples = []
            for i in idxs:
                r = train_df.iloc[int(i)]
                img = Image.open(r[f"{spec}_path"]).convert("RGB")
                lid = int(r["label_id"])
                samples.append(render_sample(wrapper, img, names, letters[label_order.index(lid)]))
            batch = collate(samples, pad_id)
            batch = {k: v.to(wrapper.device) for k, v in batch.items()}
            loss = peft_model(**batch).loss / args.grad_accum
            loss.backward()
            if (mi + 1) % args.grad_accum == 0 or mi == len(micro) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                gs += 1
                if gs % 25 == 0:
                    print(f"[{spec} ep{ep} {mi*args.batch_size}/{n}] gs={gs}/{total_steps} "
                          f"loss={loss.item()*args.grad_accum:.4f}", flush=True)
                if gs % args.save_every == 0:
                    save_state(peft_model, optimizer, scheduler, ep,
                               (mi + 1) * args.batch_size, gs, state_file)
        save_state(peft_model, optimizer, scheduler, ep + 1, 0, gs, state_file)
        print(f"[{spec}] epoch {ep+1}/{args.epochs} done", flush=True)

    peft_model.save_pretrained(state_dir)
    (state_dir / "FINAL").write_text("trained\n")
    return peft_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", default="mel")  # default: only mel for quick validation
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)  # v2: 32 (v1 was 8)
    ap.add_argument("--alpha", type=int, default=64)  # v2: 64 (=2×rank, v1 was 16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    aug_df = pd.read_csv(aug)
    names = class_names(DATASET, "en")
    letters, label_order, _ = option_layout(names)
    print(f"[lora-v2] specs={args.specs} rank={args.rank} alpha={args.alpha} "
          f"train={int((aug_df.split=='train').sum())}")

    wrapper = build_model({"name": "qwen3_vl_8b", "path": "/models/Qwen3-VL-8B-Instruct",
                           "device": args.device, "device_map": None})

    for spec in args.specs.split(","):
        state_dir = LORA_DIR / f"qwen3_vl_8b_spec{spec}"
        # v2 uses regime "lora_v3" — separate from v1's "lora_ft"
        cfg = {"eval": {"name": "vlm_spectrogram", "max_new_tokens": 8, "spectrogram": spec},
               "model": {"name": "qwen3_vl_8b"},
               "data": {"name": DATASET, "clip_len": CLIP_LEN},
               "regime": {"name": "lora_v3", "shot": 0}, "lang": "en", "enrich": "none"}
        exp = experiment_name(cfg)

        # skip if already in results.csv
        csv = RESULTS / "results.csv"
        row_done = False
        if csv.exists():
            df = pd.read_csv(csv)
            row = df[df["experiment"] == exp]
            if not row.empty:
                row_done = True
        if row_done:
            print(f"[skip] {exp}"); continue

        if not (state_dir / "FINAL").exists():
            wrapper.model = train_one_spec(wrapper, aug_df, spec, args, names, letters, label_order)
        else:
            from peft import PeftModel
            wrapper.model = PeftModel.from_pretrained(wrapper.model, str(state_dir))
        wrapper.model.eval()

        print(f"\n=== eval {exp} (attn+MLP LoRA r{args.rank}) ===")
        res = run_prompting_eval(
            wrapper, aug, dataset_name=DATASET, regime="zero_shot", shot=0, target_sr=16000,
            clip_len=CLIP_LEN, lang="en", media_type="image", spec_mode=spec,
            max_recordings=None, max_new_tokens=8, experiment=exp)
        save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
        print(f"[saved] {exp}")

        # rebuild wrapper for next spec
        del wrapper; gc.collect(); torch.cuda.empty_cache()
        wrapper = build_model({"name": "qwen3_vl_8b", "path": "/models/Qwen3-VL-8B-Instruct",
                               "device": args.device, "device_map": None})

    if args.specs == "mel,stft,demon" and not args.limit:
        (RESULTS / "lora_v3_vl8b.done").write_text("done\n")
    print("\n✅ LoRA v3 (attn+MLP) complete")


if __name__ == "__main__":
    main()
