#!/usr/bin/env python
"""LoRA fine-tuning (评测D, conference subset): Qwen3-VL-8B on DeepShip spectrograms.

Trains per spectrogram type (mel/stft/demon): the SAME zero-shot prompt the
eval uses (image + option letters), teacher-forced on the correct letter token
→ cross-entropy. This directly optimises the choice_logprobs pathway, so the
post-training eval is directly comparable to the prompting rows.

Hyper-params (defaults): LoRA r=8 α=16 dropout=0.05 on the language model's
attention projections only (regex excludes the vision tower), lr=1e-4 with
linear warmup then constant, 2 epochs over the 8,350 train clips, effective
batch 16 (micro 2 × accum 8), bf16 + gradient checkpointing on ONE GPU.

Crash-safe: deterministic per-epoch data order + atomic checkpoints (adapter +
optimizer + scheduler + position) every --save-every optimizer steps; on
restart it resumes mid-epoch. After training, runs the standard eval on the
test split (adapter attached) and writes a results.csv row (regime=lora_ft).

Writes outputs/results/lora_vl8b.done when all --specs are finished.
"""
from __future__ import annotations

import argparse, gc, os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from splash.eval.harness import find_manifest
from splash.eval.prompting_runner import run_prompting_eval
from splash.data.labels import class_names
from splash.models.factory import build_model
from splash.prompting.templates import option_layout
from splash.prompting.templates import build_prompt
from splash.tracking.results import experiment_name, save_result

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")
LORA_DIR = Path("outputs/lora")
SEED = 42


def build_target(tokenizer, letter: str) -> list[int]:
    """Token ids for the answer letter — same encoding as choice_logprobs (" A")."""
    return tokenizer.encode(" " + letter, add_special_tokens=False)


def render_sample(wrapper, image: Image.Image, names, letter: str):
    """One training sample → (ids with target appended, labels, extra feats)."""
    messages = build_prompt(media=image, label_names=names, regime="zero_shot",
                            support=None, media_type="image", lang="en", enrich_text=None)
    text = wrapper.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # return_tensors="pt" is REQUIRED: without it Qwen3-VL's processor returns
    # input_ids/attention_mask/mm_token_type_ids as Python lists (review finding).
    inputs = wrapper.processor(text=text, images=[image], return_tensors="pt")
    target = build_target(wrapper.processor.tokenizer, letter)
    ids = inputs["input_ids"][0].tolist() + target                 # append answer token(s)
    labels = [-100] * (len(ids) - len(target)) + target
    feats = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    # keep mm_token_type_ids aligned with the appended text token(s) (0 = text)
    mm = feats.get("mm_token_type_ids")
    if mm is not None:
        pad = torch.zeros((mm.shape[0], len(target)), dtype=mm.dtype)
        feats["mm_token_type_ids"] = torch.cat([mm, pad], dim=1)
    return ids, labels, feats


def collate(samples, pad_id: int):
    """Pad a list of rendered samples into one batch dict."""
    maxlen = max(len(s[0]) for s in samples)
    input_ids, labels, attention_mask = [], [], []
    for ids, lab, _ in samples:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    batch = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }
    for k in samples[0][2]:                       # mm_token_type_ids, pixel_values, …
        vals = [s[2][k] for s in samples]
        if k == "mm_token_type_ids":              # per-token → pad with 0 (text)
            vals = [torch.cat([v, torch.zeros((v.shape[0], maxlen - v.shape[1]),
                                              dtype=v.dtype)], dim=1) for v in vals]
            batch[k] = torch.cat(vals, dim=0)
        elif all(v.shape == vals[0].shape for v in vals):
            batch[k] = torch.cat(vals, dim=0)     # same shape → stack along batch
        else:
            raise RuntimeError(f"collate: ragged feature '{k}' — rerun with --batch-size 1")
    return batch


def save_state(peft_model, optimizer, scheduler, epoch, pos, global_step, state_file: Path):
    """Single-file atomic checkpoint (tmp + os.replace).

    One torch.save bundle avoids the multi-file consistency window a directory
    of adapter/optimizer/scheduler files has — os.replace on one file is the
    only atomic swap the 9p/WSL2 mount guarantees (review finding).
    """
    from peft import get_peft_model_state_dict

    bundle = {
        "adapter": get_peft_model_state_dict(peft_model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch, "pos": pos, "global_step": global_step,
    }
    tmp = state_file.with_suffix(".tmp.pt")
    torch.save(bundle, tmp)
    os.replace(tmp, state_file)


def train_one_spec(wrapper, aug, spec, args, names, letters, label_order):
    from peft import LoraConfig, get_peft_model

    train_df = aug[aug["split"] == "train"].reset_index(drop=True)
    n = args.limit or len(train_df)
    steps_per_epoch = (n + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs // args.grad_accum

    lcfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout, bias="none",
        target_modules=r"model\.language_model\..*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        task_type="CAUSAL_LM")
    peft_model = get_peft_model(wrapper.model, lcfg)
    peft_model.print_trainable_parameters()
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.train()

    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup)

    state_dir = LORA_DIR / f"qwen3_vl_8b_spec{spec}"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "state.pt"
    start_epoch, start_pos, global_step = 0, 0, 0
    if state_file.exists():
        from peft import set_peft_model_state_dict
        bundle = torch.load(state_file, map_location="cpu")
        start_epoch = bundle["epoch"]; start_pos = bundle["pos"]
        global_step = bundle["global_step"]
        result = set_peft_model_state_dict(peft_model, bundle["adapter"])
        unexpected = [k for k in result.unexpected_keys if "lora" in k]
        if unexpected:
            raise RuntimeError(f"adapter resume mismatch: {unexpected[:5]}")
        optimizer.load_state_dict(bundle["optimizer"])
        scheduler.load_state_dict(bundle["scheduler"])
        print(f"[resume] spec={spec} epoch={start_epoch} pos={start_pos} step={global_step}")

    pad_id = wrapper.processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = wrapper.processor.tokenizer.eos_token_id

    skip = start_pos
    for epoch in range(start_epoch, args.epochs):
        perm = np.random.default_rng(SEED + epoch).permutation(n)
        micro_batches = [perm[i:i + args.batch_size] for i in range(0, n, args.batch_size)]
        optimizer.zero_grad(set_to_none=True)
        for mb_i, idxs in enumerate(micro_batches):
            if skip >= (mb_i + 1) * args.batch_size:
                continue                                   # fast-forward after resume
            samples = []
            for i in idxs:
                row = train_df.iloc[int(i)]
                img = Image.open(row[f"{spec}_path"]).convert("RGB")
                lid = int(row["label_id"])
                samples.append(render_sample(wrapper, img, names,
                                             letters[label_order.index(lid)]))
            batch = collate(samples, pad_id)
            batch = {k: v.to(wrapper.device) for k, v in batch.items()}
            loss = peft_model(**batch).loss / args.grad_accum
            loss.backward()
            if (mb_i + 1) % args.grad_accum == 0 or mb_i == len(micro_batches) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 50 == 0:
                    print(f"[spec={spec} ep{epoch} {mb_i*args.batch_size}/{n}] "
                          f"step={global_step}/{total_steps} loss={loss.item()*args.grad_accum:.4f}")
                if global_step % args.save_every == 0:
                    save_state(peft_model, optimizer, scheduler, epoch,
                               (mb_i + 1) * args.batch_size, global_step, state_file)
        save_state(peft_model, optimizer, scheduler, epoch + 1, 0, global_step, state_file)
        print(f"[spec={spec}] epoch {epoch + 1}/{args.epochs} done")

    # keep a portable peft-format adapter for reuse/inference outside this script
    peft_model.save_pretrained(state_dir)
    (state_dir / "FINAL").write_text("trained\n")
    return peft_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", default="mel,stft,demon")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="cap train clips (smoke test)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    import pandas as pd
    aug_df = pd.read_csv(aug)
    expected_n = int((aug_df["split"] == "test").sum())
    names = class_names(DATASET, "en")
    letters, label_order, _ = option_layout(names)
    print(f"[lora] specs={args.specs} train={int((aug_df.split=='train').sum())} "
          f"test={expected_n} letters={letters}")

    wrapper = build_model({"name": "qwen3_vl_8b", "path": "/models/Qwen3-VL-8B-Instruct",
                           "device": args.device, "device_map": None})

    for spec in args.specs.split(","):
        state_dir = LORA_DIR / f"qwen3_vl_8b_spec{spec}"
        cfg = {"eval": {"name": "vlm_spectrogram", "max_new_tokens": 8, "spectrogram": spec},
               "model": {"name": "qwen3_vl_8b"},
               "data": {"name": DATASET, "clip_len": CLIP_LEN},
               "regime": {"name": "lora_ft", "shot": 0}, "lang": "en", "enrich": "none"}
        exp = experiment_name(cfg)
        done_file = state_dir / "FINAL"
        csv = RESULTS / "results.csv"
        row_done = False
        if csv.exists():
            df = pd.read_csv(csv)
            row = df[df["experiment"] == exp]
            row_done = (not row.empty and not pd.isna(row.iloc[0].get("n_clips", float("nan")))
                        and int(row.iloc[0]["n_clips"]) == expected_n)
        if row_done:
            print(f"[skip] {exp} (already in results.csv)"); continue
        if not done_file.exists():
            # NOTE: get_peft_model() returns a NEW wrapper object — rebind
            # wrapper.model to it, or the eval below runs on the bare base model.
            wrapper.model = train_one_spec(wrapper, aug_df, spec, args, names, letters, label_order)
        else:
            from peft import PeftModel
            wrapper.model = PeftModel.from_pretrained(wrapper.model, str(state_dir))
        wrapper.model.eval()
        print(f"\n=== eval {exp} (adapter attached) ===")
        res = run_prompting_eval(
            wrapper, aug, dataset_name=DATASET, regime="zero_shot", shot=0, target_sr=16000,
            clip_len=CLIP_LEN, lang="en", media_type="image", spec_mode=spec,
            max_recordings=None, max_new_tokens=8, experiment=exp)
        save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
        print(f"[saved] {exp}")
        # rebuild the wrapper fresh for the next spec (peft injects LoRA layers
        # in-place — reusing the model would nest adapters)
        del wrapper
        gc.collect(); torch.cuda.empty_cache()
        wrapper = build_model({"name": "qwen3_vl_8b", "path": "/models/Qwen3-VL-8B-Instruct",
                               "device": args.device, "device_map": None})

    # stage marker only for the FULL default matrix — a --specs/--limit subset run
    # must not make the watcher skip the remaining conditions (review finding)
    if args.specs == "mel,stft,demon" and not args.limit:
        (RESULTS / "lora_vl8b.done").write_text("done\n")
    print("\n✅ VL-8B LoRA complete")


if __name__ == "__main__":
    main()
