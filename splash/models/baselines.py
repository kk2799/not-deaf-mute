"""SSL audio-encoder baselines (评测E anchor): WavLM + BEATs → frozen embeddings.

These are SUPERVISED-paradigm anchors, not prompting models: they have no
language interface and cannot be zero-shot prompted (capabilities = encode
only). Protocol (SUPERB-style): frozen encoder → mean-pooled embedding →
linear head trained on the dataset's train split (see
scripts/run_baseline_probe.py). Both encoders take 16 kHz waveforms.

Loading quirks handled here:
  * WavLM-Large ships only pytorch_model.bin; transformers 5.x refuses
    torch.load on .bin with torch<2.6 (CVE-2025-32434 gate) → fall back to a
    manual state-dict load (official weights, just bypassing the gate).
  * BEATs (Microsoft unilm, vendored under third_party/beats) checkpoints from
    fine-tuned releases carry a 527-class AudioSet predictor that hijacks
    extract_features() — we drop it so the call returns frame embeddings.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import UATRModel


class _FrozenAudioEncoder(UATRModel):
    """Shared encode() for frozen audio encoders: batch of waveforms → [B, D]."""

    name = "frozen_audio"
    capabilities = {"encode", "audio"}
    target_sr = 16000
    embed_dim: int = 0

    def _forward_hidden(self, wav: torch.Tensor) -> torch.Tensor:
        """[B, T] waveform → [B, frames, D] hidden states (subclasses)."""
        raise NotImplementedError

    @torch.inference_mode()
    def encode(self, audio_batch: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """Waveforms (float32 @16k) → mean-pooled embeddings [B, D].

        Clips are equal-length by construction (manifest pads to clip_len), but
        mixed lengths are handled by grouping same-length clips into batches.
        """
        self.model.eval()
        order = sorted(range(len(audio_batch)), key=lambda i: len(audio_batch[i]))
        outs: list[np.ndarray] = [None] * len(audio_batch)  # type: ignore[list-item]
        i = 0
        while i < len(order):
            n = len(audio_batch[order[i]])
            idxs = [k for k in order[i:i + batch_size] if len(audio_batch[k]) == n]
            wav = torch.from_numpy(np.stack([audio_batch[k] for k in idxs])).to(self.device)
            h = self._forward_hidden(wav)              # [b, frames, D]
            pooled = h.mean(dim=1).float().cpu().numpy()
            for j, k in enumerate(idxs):
                outs[k] = pooled[j]
            i += len(idxs)
        return np.stack(outs)


class WavLMEncoder(_FrozenAudioEncoder):
    """WavLM-Large (transformers) — wav2vec 2.0-style SSL encoder, 1024-d."""

    name = "wavlm_large"
    embed_dim = 1024

    def __init__(self, model_path: str, device: str = "cuda:0"):
        from transformers import AutoConfig, WavLMModel

        try:
            self.model = WavLMModel.from_pretrained(model_path)
        except ValueError as e:  # transformers 5.x + torch<2.6: .bin refused
            if "torch.load" not in str(e) and "v2.6" not in str(e):
                raise
            cfg = AutoConfig.from_pretrained(model_path)
            self.model = WavLMModel(cfg)
            sd = torch.load(f"{model_path}/pytorch_model.bin",
                            map_location="cpu", weights_only=True)
            self.model.load_state_dict(sd)
        self.model = self.model.to(device).eval()
        self.device = device

    def _forward_hidden(self, wav: torch.Tensor) -> torch.Tensor:
        # wav2vec2/WavLM normalisation: zero-mean unit-var per clip
        x = (wav - wav.mean(dim=-1, keepdim=True)) / torch.sqrt(wav.var(dim=-1, keepdim=True) + 1e-7)
        return self.model(x).last_hidden_state


class BEATsEncoder(_FrozenAudioEncoder):
    """BEATs_iter3+ AS2M (vendored unilm code) — audio spectrogram SSL, 768-d."""

    name = "beats_iter3_plus"
    embed_dim = 768

    def __init__(self, model_path: str, device: str = "cuda:0"):
        from .third_party.beats import BEATs, BEATsConfig

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = BEATs(BEATsConfig(ckpt["cfg"]))
        self.model.load_state_dict(ckpt["model"])
        self.model.predictor = None   # drop AudioSet head → extract_features returns embeddings
        self.model = self.model.to(device).eval()
        self.device = device

    def _forward_hidden(self, wav: torch.Tensor) -> torch.Tensor:
        # BEATs preprocess() expects waveform scaled to int16 range (done internally)
        h, _ = self.model.extract_features(wav)
        return h
