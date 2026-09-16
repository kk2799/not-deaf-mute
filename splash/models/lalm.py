"""Large Audio-Language Model wrappers (评测 B/C/D).

Currently implements Qwen2-Audio (the first runnable model). Qwen3-Omni will
share this interface once downloaded; only the processor/model class and the
native message format differ.

Two scoring paths (the runner may use either):
* ``generate``         — free-form text (parse the answer from it).
* ``choice_logprobs``  — sum of token logprobs for each candidate answer string
                         → robust multiple-choice scoring that avoids parsing
                         failures. Cheap when choices are single tokens (e.g.
                         option letters A/B/C/D).
"""

from __future__ import annotations

import torch

from .base import UATRModel


def _collect_audios(messages: list[dict]) -> list:
    """Flatten all audio arrays from a message list, in order."""
    audios = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            continue
        for item in content:
            if item.get("type") == "audio":
                audios.append(item["audio"])
    return audios


class Qwen2AudioModel(UATRModel):
    """Wrapper for ``Qwen2AudioForConditionalGeneration`` (e.g. Qwen2-Audio-7B-Instruct)."""

    name = "qwen2_audio"
    capabilities = {"generate", "logprobs", "hidden_states", "trainable", "audio"}
    target_sr = 16000  # Whisper-based audio encoder

    def __init__(self, model_path: str, device: str = "cuda", torch_dtype=None,
                 target_sr: int = 16000, **load_kwargs):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.device = device
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        elif isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)
        self.torch_dtype = torch_dtype
        self.target_sr = target_sr
        self.processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.torch_dtype, **load_kwargs
        ).to(device).eval()

    # ---- helpers ----------------------------------------------------------
    def _render(self, messages):
        """Apply chat template (no tokenize) + gather audios + build input batch."""
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios = _collect_audios(messages)
        # transformers 5.x unified processor uses the singular `audio=` kwarg
        # (the old `audios=` is silently ignored — model would answer "blind").
        inputs = self.processor(
            text=text, audio=audios, return_tensors="pt", sampling_rate=self.target_sr
        ).to(self.device)
        return inputs

    # ---- prompting --------------------------------------------------------
    @torch.inference_mode()
    def generate(self, messages_batch, max_new_tokens: int = 32, **gen_kwargs) -> list[str]:
        """Generate one reply per sample (processed sequentially for correctness
        with variable-length audio; batching can be added later)."""
        out_texts = []
        for messages in messages_batch:
            inputs = self._render(messages)
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
            new = gen_ids[:, inputs["input_ids"].shape[1]:]
            out_texts.append(self.processor.batch_decode(new, skip_special_tokens=True)[0].strip())
        return out_texts

    @torch.inference_mode()
    def choice_logprobs(self, messages_batch, choices, **_gen_kwargs) -> list[dict[str, float]]:
        """Score each candidate ``choice`` string as the continuation of the prompt.

        Sums the logprobs of the choice's tokens given the prompt+audio. Works
        generally; cheapest/most robust when choices are short (e.g. "A".."D").
        """
        # Tokenise each choice once (with a leading space to match continuation).
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

    # ---- later paradigms (interfaces only) --------------------------------

    @property
    def lora_target_modules(self) -> list[str]:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
