"""Unified model interface (UATRModel) for all six evaluation paradigms.

A model wrapper exposes *capabilities* (what it can do) and implements the
matching methods. The eval harness queries capabilities and dispatches:

    capability      method                  paradigms
    -----------     --------------------    ---------------------
    generate        generate()              评测 A/B prompting
    logprobs        choice_logprobs()       评测 A/B robust multiple-choice
    hidden_states   hidden_states()         评测 C layer-wise linear probe
    encode          encode()                评测 E audio-encoder baselines
    trainable       lora_target_modules     评测 D LoRA fine-tune
    audio           accepts audio input     听觉路线 (LALM)
    image           accepts image input     视觉路线 (VLM / spectrogram)

Not every model implements everything — e.g. a pure VLM has no ``audio`` /
``hidden_states``-for-audio. Wrappers declare ``capabilities`` and raise
``NotImplementedError`` for the rest, so adding a new model is local.

Messages use the HuggingFace structured-content format (model-agnostic):
    {"role": "user", "content": [
        {"type": "text",  "text": "Identify the ship type..."},
        {"type": "audio", "audio": np.ndarray(1d float32 @ model sr)},
        {"type": "image", "image": PIL.Image.Image},
    ]}
Each wrapper translates this to its processor's native format.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ModelOutput:
    """Result of one prompting inference (评测 A/B)."""

    text: str                       # free-form generated answer
    choice_logprobs: dict[str, float] | None = None  # {choice_text: total_logprob}, if scored
    choice: str | None = None       # argmax choice (or parsed from text)

    @property
    def predicted_choice(self) -> str | None:
        """Best choice by logprob if available, else None (caller parses text)."""
        if self.choice_logprobs:
            return max(self.choice_logprobs, key=self.choice_logprobs.get)
        return self.choice


class UATRModel(ABC):
    """Base class for all SPLASH model wrappers."""

    name: str = "base"
    capabilities: set[str] = set()  # subset of the table above
    target_sr: int = 16000          # audio sample rate this model expects

    # ---- prompting (评测 A/B) ---------------------------------------------
    def generate(self, messages_batch, **gen_kwargs) -> list[str]:
        """Free-form generation for a batch of message-lists → decoded texts."""
        raise NotImplementedError

    def choice_logprobs(self, messages_batch, choices, **gen_kwargs) -> list[dict[str, float]]:
        """For each sample, total logprob of each ``choice`` string as the
        continuation. Used for robust multiple-choice scoring."""
        raise NotImplementedError

    # ---- representation / training (评测 C/D/E) ----------------------------
    def hidden_states(self, messages_batch, layer_ids=None):
        """Per-layer mean-pooled hidden states for linear probing (评测 C).

        For each message list: one forward pass with ``output_hidden_states=True``
        → every decoder layer's hidden state, mean-pooled over attended tokens
        → fp32 CPU tensor [L+1, D] (row 0 = embedding output, row k = layer k
        output). ``layer_ids`` optionally selects a subset of rows. Wrappers only
        need ``_render`` + ``model`` (all three implement both).
        """
        import torch

        results = []
        for messages in messages_batch:
            inputs = self._render(messages)
            with torch.inference_mode():
                out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states                     # tuple(L+1) of [1, T, D]
            rows = range(len(hs)) if layer_ids is None else layer_ids
            mask = inputs.get("attention_mask")
            m = mask[0].bool() if mask is not None else None
            pooled = torch.stack([
                (h[0][m] if m is not None else h[0]).float().mean(dim=0)
                for h in (hs[i] for i in rows)
            ])                                          # [n_sel, D]
            results.append(pooled)
        return results

    def encode(self, audio_batch):
        """Embeddings for classifier baselines (评测 E)."""
        raise NotImplementedError

    @property
    def lora_target_modules(self) -> list[str] | None:
        """LoRA injection points for fine-tuning (评测 D)."""
        return None

    # ---- capability queries ----------------------------------------------
    def supports(self, *caps: str) -> bool:
        return all(c in self.capabilities for c in caps)
