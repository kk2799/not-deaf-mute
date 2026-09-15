"""Vision-Language Model wrappers (评测 A: VLM reads spectrogram images).

Implements Qwen3-VL (8B/32B). Same prompting contract as the LALM wrapper
(generate + choice_logprobs over option letters) but the media is a spectrogram
*image* (PIL) fed via the processor. Verified end-to-end: a Qwen3-VL-8B smoke
test read a mel spectrogram and correctly answered the ship type.

Note on the transformers-5 API: Qwen3-VL's processor takes the **plural**
``images=`` kwarg (unlike Qwen2-Audio's singular ``audio=``); both were checked
empirically.
"""

from __future__ import annotations

import torch

from .base import UATRModel


def _collect_images(messages: list[dict]) -> list:
    """Flatten all PIL images from a message list, in order."""
    imgs = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            continue
        for item in content:
            if item.get("type") == "image":
                imgs.append(item["image"])
    return imgs


class Qwen3VLModel(UATRModel):
    """Wrapper for Qwen3-VL (e.g. Qwen3-VL-8B/32B-Instruct)."""

    name = "qwen3_vl"
    capabilities = {"generate", "logprobs", "hidden_states", "trainable", "image"}
    target_sr = 22050  # unused for image input; kept for interface symmetry

    def __init__(self, model_path: str, device: str = "cuda", torch_dtype=None,
                 device_map: str | None = "auto", **load_kwargs):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        self.torch_dtype = torch_dtype

        self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
        load_kw = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True)
        if device_map:                      # "auto" / {"": "cuda:1"} → accelerate places it
            load_kw["device_map"] = device_map
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw).eval()
            self.device = next(self.model.parameters()).device
        else:                               # pin to a single device (parallel routes)
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kw).to(device).eval()
            self.device = device

    # ---- helpers ----------------------------------------------------------
    def _render(self, messages):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        images = _collect_images(messages)
        kw = dict(text=text, return_tensors="pt")
        if images:
            kw["images"] = images
        inputs = self.processor(**kw)
        inputs = inputs.to(self.device)
        return inputs

    # ---- prompting --------------------------------------------------------
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
        """Score each option letter by its first-token logprob given the prompt+image."""
        choice_token_ids = {}
        for c in choices:
            ids = self.processor.tokenizer.encode(" " + c, add_special_tokens=False)
            choice_token_ids[c] = ids

        results = []
        max_ctoks = max(len(v) for v in choice_token_ids.values())
        for messages in messages_batch:
            inputs = self._render(messages)
            prompt_len = inputs["input_ids"].shape[1]
            logits = self.model(**inputs).logits[0]  # [T, V]
            # log_softmax ONLY at answer-token positions (prevents OOM on long prompts).
            need = slice(prompt_len - 1, prompt_len - 1 + max_ctoks)
            lp = torch.log_softmax(logits[need].float(), dim=-1)  # [max_ctoks, V]
            scores = {}
            for c, tok_ids in choice_token_ids.items():
                scores[c] = float(sum(lp[i, tid].item() for i, tid in enumerate(tok_ids)))
            results.append(scores)
        return results

    @property
    def lora_target_modules(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
