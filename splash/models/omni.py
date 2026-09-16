"""Qwen3-Omni wrapper — the same-backbone "看 vs 听" model (评测 A + B).

Qwen3-Omni-30B-A3B is a full-modal MoE (text+image+audio+video). This wrapper
supports BOTH ``audio`` (评测 B, like Qwen2-Audio) and ``image`` (评测 A, like
Qwen3-VL) inputs via the SAME model — enabling the cleanest possible
"seeing spectrograms vs listening to audio" comparison on an identical backbone.

Processor: ``Qwen3OmniMoeProcessor`` takes singular ``audio=`` / ``images=`` (transformers-5 unified API).
Model: ``Qwen3OmniMoeForConditionalGeneration`` (30B MoE / 3B active, ~60 GB → device_map="auto" across both GPUs).
"""

from __future__ import annotations

import torch

from .base import UATRModel


def _collect_media(messages: list[dict]) -> tuple[list, list]:
    """Return (audios, images) flattened from a message list, in order."""
    audios, images = [], []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            continue
        for item in content:
            t = item.get("type")
            if t == "audio":
                audios.append(item["audio"])
            elif t == "image":
                images.append(item["image"])
    return audios, images


class Qwen3OmniModel(UATRModel):
    """Wrapper for Qwen3-Omni-30B-A3B-Instruct (full-modal MoE).

    Supports both audio and image inputs → same-backbone 看 vs 听 comparison.
    """

    name = "qwen3_omni"
    capabilities = {"generate", "logprobs", "audio", "image", "hidden_states", "trainable"}
    target_sr = 16000  # Whisper-based audio encoder

    def __init__(self, model_path: str, device: str = "cuda", torch_dtype=None,
                 device_map: str | None = "auto", **load_kwargs):
        # Thinker ONLY (text generation) — the full Qwen3OmniMoeForConditionalGeneration
        # also runs the Talker (speech synthesis), which (a) we don't need and
        # (b) has a multi-GPU device bug in generate().
        from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        self.torch_dtype = torch_dtype

        self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
        load_kw = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        if device_map:
            load_kw["device_map"] = device_map  # 30B MoE across both GPUs
        self.model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
            model_path, **load_kw).eval()
        self.device = next(self.model.parameters()).device

    def _render(self, messages):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images = _collect_media(messages)
        kw = dict(text=text, return_tensors="pt")
        if audios:
            kw["audio"] = audios
        if images:
            kw["images"] = images
        inputs = self.processor(**kw)
        # cast floating-point tensors to model dtype (processor outputs fp32 audio
        # features; model weights are bf16 → conv2d rejects the mismatch)
        inputs = inputs.to(self.device, self.torch_dtype)
        return inputs

    @torch.inference_mode()
    def generate(self, messages_batch, max_new_tokens: int = 16, **gen_kwargs) -> list[str]:
        out_texts = []
        for messages in messages_batch:
            inputs = self._render(messages)
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
            new = gen_ids[:, inputs["input_ids"].shape[1]:]
            out_texts.append(self.processor.batch_decode(new, skip_special_tokens=True)[0].strip())
        return out_texts

    @torch.inference_mode()
    def choice_logprobs(self, messages_batch, choices, **_gen_kwargs) -> list[dict[str, float]]:
        choice_token_ids = {}
        for c in choices:
            ids = self.processor.tokenizer.encode(" " + c, add_special_tokens=False)
            choice_token_ids[c] = ids

        results = []
        max_ctoks = max(len(v) for v in choice_token_ids.values())
        for messages in messages_batch:
            inputs = self._render(messages)
            prompt_len = inputs["input_ids"].shape[1]
            scores = {}
            if max_ctoks == 1:
                # single-token choices: one forward, read the last position
                logits = self.model(**inputs).logits[0]  # [T, V]
                lp = torch.log_softmax(logits[prompt_len - 1].float(), dim=-1)
                for c, tok_ids in choice_token_ids.items():
                    scores[c] = float(lp[tok_ids[0]].item())
            else:
                # multi-token choices: one forward per choice with the choice
                # tokens appended (teacher forcing); logprobs of those positions
                for c, tok_ids in choice_token_ids.items():
                    ids = inputs["input_ids"]
                    ext = torch.tensor([tok_ids], dtype=ids.dtype, device=ids.device)
                    fwd = dict(inputs)
                    fwd["input_ids"] = torch.cat([ids, ext], dim=1)
                    am = inputs.get("attention_mask")
                    if am is not None:
                        fwd["attention_mask"] = torch.cat(
                            [am, torch.ones((1, len(tok_ids)), dtype=am.dtype,
                                            device=am.device)], dim=1)
                    logits = self.model(**fwd).logits[0]
                    rows = logits[prompt_len - 1: prompt_len - 1 + len(tok_ids)].float()
                    lp = torch.log_softmax(rows, dim=-1)
                    scores[c] = float(sum(lp[i, tid].item()
                                          for i, tid in enumerate(tok_ids)))
            results.append(scores)
        return results

    @property
    def lora_target_modules(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
