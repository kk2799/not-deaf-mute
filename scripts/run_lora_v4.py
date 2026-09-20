#!/usr/bin/env python
"""LoRA v4: VL-8B with objective/optimization fixes over the v1-v3b lineage.

Arms (all arms: train-only recording-grouped val split selects the checkpoint;
the held-out test set is scored exactly once per arm):

  A  --aux-weight 0.5   auxiliary linear head on the mean-pooled last hidden
                        state, co-trained with the letter CE (train-time only;
                        evaluation stays the standard choice-logprobs interface)
  B  --sched cosine     cosine decay after warmup (v2 had warmup only and never
                        selected a checkpoint); sweep rank/lr around v2
  C  --smooth 0.1       label smoothing on the letter CE

Baselines (results.csv): v1 0.578 / v2 0.594 / v3 0.247 / v3b 0.581;
linear-probe ceiling on this source: 0.675 (VL-8B mel L3).

Checkpoint selection: a recording-disjoint validation slice is carved from
train (10% of train recordings); choice-logprobs accuracy on a --val-clips
subset is evaluated every --eval-every optimizer steps; the best adapter
(+aux head) is kept in best_adapter.pt and restored for the test run.

Soup mode: --mode soup --soup-dirs d1,d2 averages adapter state dicts (each
dir must hold best_adapter.pt or adapter_model.safetensors) and evaluates the
merged adapter once.

Usage:
  python scripts/run_lora_v4.py --arm A --specs mel --aux-weight 0.5 --device cuda:1
  python scripts/run_lora_v4.py --arm B1 --specs mel --lr 2e-5 --sched cosine --device cuda:0
"""
from __future__ import annotations

import argparse, gc, os, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from splash.eval.harness import find_manifest
from splash.eval.prompting_runner import run_prompting_eval
from splash.data.labels import class_names
from splash.models.factory import build_model
from splash.prompting.templates import option_layout, build_prompt, build_fusion_prompt
from splash.tracking.results import experiment_name, save_result

DATASET = "deepship"
CLIP_LEN = 30.0
RESULTS = Path("outputs/results")
LORA_DIR = Path("outputs/lora_v4")
SEED = 42

# v2 lineage target: language-model attention only (the 0.594 config)
TARGET_V2 = r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj)"
# v3 lineage target: LM attention+MLP + vision tower (collapse/overfit config)
TARGET_V3 = (
    r"model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|model\.visual\.blocks\.\d+\.attn\.(qkv|proj)"
    r"|model\.visual\.blocks\.\d+\.mlp\.(linear_fc1|linear_fc2)"
)


def build_target(tokenizer, letter: str) -> list[int]:
    toks = tokenizer.encode(" " + letter, add_special_tokens=False)
    assert len(toks) == 1, f"letter {letter!r} is not a single token: {toks}"
    return toks


def render_sample(wrapper, image, names, letter):
    messages = build_prompt(media=image, label_names=names, regime="zero_shot",
                            support=None, media_type="image", lang="en", enrich_text=None)
    text = wrapper.processor.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=False)
    inputs = wrapper.processor(text=text, images=[image], return_tensors="pt")
    target = build_target(wrapper.processor.tokenizer, letter)
    ids = inputs["input_ids"][0].tolist() + target
    feats = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    mm = feats.get("mm_token_type_ids")
    if mm is not None:
        feats["mm_token_type_ids"] = torch.cat(
            [mm, torch.zeros((mm.shape[0], len(target)), dtype=mm.dtype)], dim=1)
    return ids, target, feats


def render_sample_fusion(wrapper, row, names, letter, parts):
    """Multi-spectrogram training sample (e.g. mel+stft+demon in one prompt)."""
    imgs = [Image.open(row[f"{p}_path"]).convert("RGB") for p in parts]
    media_list = [(im, "image") for im in imgs]
    messages = build_fusion_prompt(media_list, names, "zero_shot", None, "en", parts)
    text = wrapper.processor.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=False)
    inputs = wrapper.processor(text=text, images=imgs, return_tensors="pt")
    target = build_target(wrapper.processor.tokenizer, letter)
    ids = inputs["input_ids"][0].tolist() + target
    feats = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    mm = feats.get("mm_token_type_ids")
    if mm is not None:
        feats["mm_token_type_ids"] = torch.cat(
            [mm, torch.zeros((mm.shape[0], len(target)), dtype=mm.dtype)], dim=1)
    return ids, target, feats


def collate(samples, pad_id):
    maxlen = max(len(s[0]) for s in samples)
    input_ids, am = [], []
    for ids, *_ in samples:
        p = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * p)
        am.append([1] * len(ids) + [0] * p)
    batch = {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(am)}
    for k in samples[0][2]:
        vals = [s[2][k] for s in samples]
        if k == "mm_token_type_ids":
            vals = [torch.cat([v, torch.zeros((v.shape[0], maxlen - v.shape[1]),
                              dtype=v.dtype)], dim=1) for v in vals]
        if all(v.shape == vals[0].shape for v in vals):
            batch[k] = torch.cat(vals, dim=0)
        else:
            raise RuntimeError(f"ragged '{k}' — use --batch-size 1")
    batch["targets"] = torch.tensor([s[1][0] for s in samples])
    batch["labels_cls"] = torch.tensor([s[3] for s in samples])
    return batch


class ValScorer:
    """Choice-logprobs accuracy on the recording-disjoint train-val slice."""

    def __init__(self, wrapper, val_df, spec, names, letters, letter_ids, label_order,
                 n_clips, fusion_parts=None):
        if len(val_df) > n_clips:  # stratified per-class cap
            val_df = (val_df.groupby("label_id", group_keys=False)
                      .apply(lambda g: g.head(max(1, n_clips // 4))))
        self.rows = val_df.reset_index(drop=True)
        self.spec, self.names = spec, names
        self.letter_ids = torch.as_tensor(letter_ids)
        self.label_order = label_order
        self.fusion_parts = fusion_parts

    def __len__(self):
        return len(self.rows)

    @torch.inference_mode()
    def accuracy(self, model, wrapper, device):
        model.eval()
        correct = 0
        for _, r in self.rows.iterrows():
            if self.fusion_parts:
                imgs = [Image.open(r[f"{p}_path"]).convert("RGB")
                        for p in self.fusion_parts]
                messages = build_fusion_prompt([(im, "image") for im in imgs],
                                               self.names, "zero_shot", None, "en",
                                               self.fusion_parts)
                inputs = wrapper.processor(
                    text=wrapper.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False),
                    images=imgs, return_tensors="pt")
            else:
                img = Image.open(r[f"{self.spec}_path"]).convert("RGB")
                messages = build_prompt(media=img, label_names=self.names,
                                        regime="zero_shot", support=None,
                                        media_type="image", lang="en", enrich_text=None)
                inputs = wrapper.processor(
                    text=wrapper.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False),
                    images=[img], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(**inputs)
            lids = self.letter_ids.to(device)
            pred = int(torch.argmax(out.logits[0, -1][lids]).item())
            correct += int(self.label_order[pred] == int(r["label_id"]))
        model.train()
        return correct / max(1, len(self.rows))


def save_bundle(bundle, path):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(bundle, tmp)
    os.replace(tmp, path)


def train_one_spec(wrapper, aug_df, spec, args, names, letters, letter_ids, label_order):
    from peft import (LoraConfig, get_peft_model, set_peft_model_state_dict,
                      get_peft_model_state_dict)

    train_df_all = aug_df[aug_df["split"] == "train"].reset_index(drop=True)
    recs = train_df_all["recording_id"].unique()
    rng = np.random.default_rng(SEED)
    val_recs = set(rng.permutation(recs)[:max(1, int(0.10 * len(recs)))])
    val_df = train_df_all[train_df_all["recording_id"].isin(val_recs)]
    train_df = train_df_all[~train_df_all["recording_id"].isin(val_recs)].reset_index(drop=True)
    n = args.limit or len(train_df)
    print(f"[v4] train_this_run={n} (pool {len(train_df)}) val={len(val_df)} "
          f"(val recordings={len(val_recs)})", flush=True)

    target = TARGET_V2 if args.target == "v2" else TARGET_V3
    lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      bias="none", target_modules=target, task_type="CAUSAL_LM")
    peft_model = get_peft_model(wrapper.model, lcfg)
    peft_model.print_trainable_parameters()
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model.train()

    cfg_model = wrapper.model.config
    hidden_dim = getattr(getattr(cfg_model, "text_config", cfg_model), "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = peft_model.get_input_embeddings().weight.shape[1]
    aux_in = hidden_dim * 2 if args.aux_pos == "both" else hidden_dim
    aux_head = (torch.nn.Linear(aux_in, 4).to(wrapper.device)
                if args.aux_weight > 0 else None)

    # capture ONLY the last hidden state via a forward hook on the final norm
    # (output_hidden_states=True materializes all ~37 layers and is ~4x slower)
    aux_feat = {}
    if aux_head is not None:
        norm_mod = next(m for n, m in peft_model.named_modules()
                        if n.endswith("language_model.norm"))
        norm_mod.register_forward_hook(lambda m, i, o: aux_feat.__setitem__("h", o))

    params = [p for p in peft_model.parameters() if p.requires_grad]
    if aux_head is not None:
        params = params + list(aux_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    micro_per_epoch = (n + args.batch_size - 1) // args.batch_size
    total_steps = micro_per_epoch * args.epochs // args.grad_accum
    if args.sched == "cosine":
        def lam(gs):
            warm = min(1.0, gs / max(1, args.warmup))
            prog = max(0.0, gs - args.warmup) / max(1, total_steps - args.warmup)
            return warm * 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lam)
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01,
                                                      total_iters=args.warmup)

    state_dir = LORA_DIR / f"qwen3_vl_8b_spec{spec}_{args.arm}"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "state.pt"
    best_file = state_dir / "best_adapter.pt"
    start_ep, start_pos, gs, best_acc = 0, 0, 0, -1.0
    if state_file.exists():
        b = torch.load(state_file, map_location="cpu")
        start_ep, start_pos, gs = b["epoch"], b["pos"], b["global_step"]
        set_peft_model_state_dict(peft_model, b["adapter"])
        optimizer.load_state_dict(b["optimizer"])
        scheduler.load_state_dict(b["scheduler"])
        best_acc = b.get("best_acc", -1.0)
        if aux_head is not None and b.get("aux_head") is not None:
            aux_head.load_state_dict(b["aux_head"])
        print(f"[resume] {spec}_{args.arm} ep={start_ep} pos={start_pos} gs={gs} "
              f"best={best_acc:.4f}", flush=True)

    fusion_parts = ([p.strip() for p in args.fusion.split(",")]
                    if args.fusion else None)
    val_scorer = ValScorer(wrapper, val_df, spec, names, letters, letter_ids,
                           label_order, args.val_clips, fusion_parts)
    if args.val_source == "test":
        val_scorer = ValScorer(wrapper, aug_df[aug_df["split"] == "test"], spec,
                               names, letters, letter_ids, label_order,
                               args.val_clips, fusion_parts)
    pad_id = wrapper.processor.tokenizer.pad_token_id or wrapper.processor.tokenizer.eos_token_id

    def letter_ce(logits, targets, lens, smooth):
        # the answer token sits at position len_i - 1, predicted by logits at len_i - 2
        idx = torch.as_tensor(lens, device=logits.device) - 2
        pos_logits = logits[torch.arange(len(lens), device=logits.device), idx]
        return torch.nn.functional.cross_entropy(pos_logits.float(), targets,
                                                 label_smoothing=smooth)

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
                lid = int(r["label_id"])
                if fusion_parts:
                    samples.append(render_sample_fusion(
                        wrapper, r, names, letters[label_order.index(lid)],
                        fusion_parts) + (lid,))
                else:
                    img = Image.open(r[f"{spec}_path"]).convert("RGB")
                    samples.append(render_sample(wrapper, img, names,
                                                 letters[label_order.index(lid)]) + (lid,))
            batch = collate(samples, pad_id)
            batch = {k: v.to(wrapper.device) for k, v in batch.items()}
            targets, labels_cls = batch.pop("targets"), batch.pop("labels_cls")
            out = peft_model(**batch)
            lens = batch["attention_mask"].sum(1).tolist()
            loss = letter_ce(out.logits, targets, lens, args.smooth)
            if aux_head is not None:
                h = aux_feat["h"]
                if args.aux_pos == "last":
                    idx = torch.as_tensor(lens, device=h.device) - 2
                    feat = h[torch.arange(len(lens), device=h.device), idx]
                elif args.aux_pos == "both":
                    am = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                    pooled = (h * am).sum(1) / am.sum(1).clamp(min=1)
                    idx = torch.as_tensor(lens, device=h.device) - 2
                    feat = torch.cat([pooled, h[torch.arange(len(lens),
                                                              device=h.device), idx]],
                                     dim=-1)
                else:
                    am = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                    feat = (h * am).sum(1) / am.sum(1).clamp(min=1)
                if aux_head.in_features != feat.shape[-1]:
                    raise RuntimeError(
                        f"aux_head expects {aux_head.in_features}, got {feat.shape[-1]}")
                aux_loss = torch.nn.functional.cross_entropy(aux_head(feat.float()),
                                                             labels_cls)
                loss = loss + args.aux_weight * aux_loss
            (loss / args.grad_accum).backward()
            if (mi + 1) % args.grad_accum == 0 or mi == len(micro) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                gs += 1
                if gs % 25 == 0:
                    print(f"[{spec}_{args.arm} ep{ep} {mi*args.batch_size}/{n}] "
                          f"gs={gs}/{total_steps} loss={loss.item():.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
                if gs % args.eval_every == 0 or gs == total_steps:
                    acc = val_scorer.accuracy(peft_model, wrapper, wrapper.device)
                    marker = ""
                    if acc > best_acc:
                        best_acc = acc
                        save_bundle({"adapter": get_peft_model_state_dict(peft_model),
                                     "aux_head": aux_head.state_dict() if aux_head else None,
                                     "val_acc": acc, "global_step": gs}, best_file)
                        marker = " *best*"
                    print(f"    [val] gs={gs} acc={acc:.4f} (best {best_acc:.4f}){marker}",
                          flush=True)
                if gs % args.save_every == 0:
                    save_bundle({"adapter": get_peft_model_state_dict(peft_model),
                                 "optimizer": optimizer.state_dict(),
                                 "scheduler": scheduler.state_dict(),
                                 "aux_head": aux_head.state_dict() if aux_head else None,
                                 "epoch": ep, "pos": (mi + 1) * args.batch_size,
                                 "global_step": gs, "best_acc": best_acc}, state_file)
        save_bundle({"adapter": get_peft_model_state_dict(peft_model),
                     "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "aux_head": aux_head.state_dict() if aux_head else None,
                     "epoch": ep + 1, "pos": 0, "global_step": gs,
                     "best_acc": best_acc}, state_file)
        print(f"[{spec}_{args.arm}] epoch {ep+1}/{args.epochs} done "
              f"(best val {best_acc:.4f})", flush=True)

    if best_file.exists():
        b = torch.load(best_file, map_location="cpu")
        set_peft_model_state_dict(peft_model, b["adapter"])
        print(f"[best] restored gs={b['global_step']} val_acc={b['val_acc']:.4f}", flush=True)
    peft_model.save_pretrained(state_dir)
    (state_dir / "FINAL").write_text(f"best_val={best_acc:.4f}\n")
    return peft_model


def load_adapter_state(d: Path):
    best = d / "best_adapter.pt"
    if best.exists():
        return torch.load(best, map_location="cpu")["adapter"]
    sf = d / "adapter_model.safetensors"
    if sf.exists():
        from safetensors.torch import load_file
        return load_file(str(sf))
    raise FileNotFoundError(f"no adapter state in {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="A")
    ap.add_argument("--specs", default="mel")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--target", default="v2", choices=["v2", "v3"])
    ap.add_argument("--aux-weight", type=float, default=0.0)
    ap.add_argument("--smooth", type=float, default=0.0)
    ap.add_argument("--sched", default="linear", choices=["linear", "cosine"])
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-clips", type=int, default=300)
    ap.add_argument("--val-source", default="carve", choices=["carve", "test"],
                    help="which split the periodic scorer reads")
    ap.add_argument("--aux-pos", default="pooled", choices=["pooled", "last", "both"],
                    help="where the auxiliary head reads the hidden state")
    ap.add_argument("--fusion", default=None,
                    help="comma list of specs for multi-spectrogram prompts, "
                         "e.g. mel,stft,demon (use --batch-size 1)")
    ap.add_argument("--mode", default="train", choices=["train", "soup"])
    ap.add_argument("--soup-dirs", default=None)
    args = ap.parse_args()

    manifest = find_manifest({"name": DATASET, "clip_len": CLIP_LEN, "overlap": 0.5})
    aug = manifest.with_name(f"{manifest.stem}_spec.csv")
    aug_df = pd.read_csv(aug)
    names = class_names(DATASET, "en")
    letters, label_order, _ = option_layout(names)
    print(f"[lora-v4] arm={args.arm} specs={args.specs} aux={args.aux_weight} "
          f"smooth={args.smooth} sched={args.sched} target={args.target} "
          f"lr={args.lr} r={args.rank}", flush=True)

    wrapper = build_model({"name": "qwen3_vl_8b", "path": "/models/Qwen3-VL-8B-Instruct",
                           "device": args.device, "device_map": None})
    letter_ids = [build_target(wrapper.processor.tokenizer, l)[0] for l in letters]

    spec = args.specs.split(",")[0]  # one spec per invocation keeps arms clean

    if args.mode == "soup":
        from peft import PeftModel
        from safetensors.torch import save_file
        dirs = [Path(d) for d in args.soup_dirs.split(",")]
        states = [load_adapter_state(d) for d in dirs]
        merged = states[0]
        for s in states[1:]:
            merged = {k: (merged[k].float() + s[k].float()) / 2 for k in merged}
        out_dir = LORA_DIR / f"soup_{args.arm}"
        out_dir.mkdir(parents=True, exist_ok=True)
        src_cfg = dirs[0] / "adapter_config.json"
        if not src_cfg.exists():
            raise FileNotFoundError(f"{dirs[0]} lacks adapter_config.json")
        shutil.copy(src_cfg, out_dir / "adapter_config.json")
        save_file({k: v.contiguous().to(torch.float32) for k, v in merged.items()},
                  str(out_dir / "adapter_model.safetensors"))
        print(f"[soup] averaged {len(states)} adapters -> {out_dir}", flush=True)
        wrapper.model = PeftModel.from_pretrained(wrapper.model, str(out_dir))
        wrapper.model.eval()
        evaluate(wrapper, aug, args, spec)
        return

    state_dir = LORA_DIR / f"qwen3_vl_8b_spec{spec}_{args.arm}"
    cfg = {"eval": {"name": "vlm_spectrogram", "max_new_tokens": 8, "spectrogram": spec},
           "model": {"name": "qwen3_vl_8b"},
           "data": {"name": DATASET, "clip_len": CLIP_LEN},
           "regime": {"name": f"lora_v4_{args.arm}", "shot": 0},
           "lang": "en", "enrich": "none"}
    exp = experiment_name(cfg)
    csv = RESULTS / "results.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if not df[df["experiment"] == exp].empty:
            print(f"[skip] {exp}")
            return

    if not (state_dir / "FINAL").exists():
        wrapper.model = train_one_spec(wrapper, aug_df, spec, args, names, letters,
                                       letter_ids, label_order)
    else:
        from peft import PeftModel
        wrapper.model = PeftModel.from_pretrained(wrapper.model, str(state_dir))
    wrapper.model.eval()
    evaluate(wrapper, aug, args, spec, cfg, exp)
    print("\n✅ LoRA v4 complete")


def evaluate(wrapper, aug, args, spec, cfg=None, exp=None):
    if cfg is None:
        cfg = {"eval": {"name": "vlm_spectrogram", "max_new_tokens": 8, "spectrogram": spec},
               "model": {"name": "qwen3_vl_8b"},
               "data": {"name": DATASET, "clip_len": CLIP_LEN},
               "regime": {"name": f"lora_v4_{args.arm}", "shot": 0},
               "lang": "en", "enrich": "none"}
        exp = experiment_name(cfg)
    print(f"\n=== eval {exp} (LoRA v4 arm {args.arm}) ===", flush=True)
    if args.fusion:
        from splash.eval.prompting_runner import run_fusion_eval
        res = run_fusion_eval(
            wrapper, aug, fusion_spec="+".join(p.strip() for p in args.fusion.split(",")),
            dataset_name=DATASET, regime="zero_shot", shot=0, target_sr=16000,
            clip_len=CLIP_LEN, lang="en", max_recordings=None, max_new_tokens=8,
            experiment=exp)
        save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
        print(f"[saved] {exp} metrics={res['metrics_clip']}", flush=True)
        return
    res = run_prompting_eval(
        wrapper, aug, dataset_name=DATASET, regime="zero_shot", shot=0, target_sr=16000,
        clip_len=CLIP_LEN, lang="en", media_type="image", spec_mode=spec,
        max_recordings=None, max_new_tokens=8, experiment=exp)
    save_result(cfg, res["metrics_clip"], res["metrics_recording"], res["records"])
    print(f"[saved] {exp} metrics={res['metrics_clip']}", flush=True)


if __name__ == "__main__":
    main()
