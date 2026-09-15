#!/usr/bin/env python
"""Answer-confidence analysis: are the collapsed predictions CONFIDENT?

For each zero-shot run, softmax-normalise the saved per-clip choice_logprobs
and report the top-1 probability distribution. High confidence (top1 ≈ 1)
on the collapsed class = strong prior; near-uniform (top1 ≈ 0.25) = degenerate
argmax over noise. Complements docs/样本级分析_20260819.md.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

R = Path("outputs/results")
EXPS = [
    ("q2a.audio",    "lalm_prompting_qwen2_audio_deepship_zero_shot_shot0_clip30.0_langen_enrichnone"),
    ("vl8b.mel",     "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone"),
    ("vl8b.stft",    "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specstft_langen_enrichnone"),
    ("vl8b.demon",   "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specdemon_langen_enrichnone"),
    ("vl32b.mel",    "vlm_spectrogram_qwen3_vl_32b_deepship_zero_shot_shot0_clip30_specmel_langen_enrichnone"),
    ("vl32b.demon",  "vlm_spectrogram_qwen3_vl_32b_deepship_zero_shot_shot0_clip30_specdemon_langen_enrichnone"),
    ("omni.audio",   "lalm_prompting_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_langen_enrichnone"),
    ("omni.mel",     "vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone"),
    ("omni.quad",    "fusion_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_fusionaudio+mel+stft+demon_langen_enrichnone"),
]


def main():
    print(f"{'experiment':<13}{'E[top1]':>8}{'median':>8}{'P>0.9':>7}{'E[top1|collapse]':>17}{'margin':>8}")
    for tag, name in EXPS:
        recs = json.loads((R / f"{name}.json").read_text())["predictions"]
        dom = Counter(r["pred_label_id"] for r in recs).most_common(1)[0][0]
        top1, top1_dom, margins = [], [], []
        for r in recs:
            lp = list(r["choice_logprobs"].values())
            m = max(lp)
            p = np.exp(np.array(lp) - m)
            p = p / p.sum()
            srt = np.sort(p)
            top1.append(srt[-1])
            margins.append(srt[-1] - srt[-2])
            if r["pred_label_id"] == dom:
                top1_dom.append(srt[-1])
        t = np.array(top1)
        print(f"{tag:<13}{t.mean():>8.3f}{np.median(t):>8.3f}{(t > 0.9).mean():>7.2f}"
              f"{np.mean(top1_dom):>17.3f}{np.mean(margins):>8.3f}")


if __name__ == "__main__":
    main()
