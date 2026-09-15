#!/usr/bin/env python
"""Per-sample agreement analysis (error-analysis supplement).

Questions answered (DeepShip test, 2,685 clips):
  1. How similar are DIFFERENT models' per-clip predictions? (pairwise agreement
     + prediction-class distributions — does everyone collapse to one class?)
  2. Within one model, how much does few-shot CHANGE the per-clip predictions
     vs zero-shot? (agreement, and accuracy on changed vs unchanged clips)
  3. Per-class recall — which classes (if any) does each model get right?
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from splash.data.labels import class_names

R = Path("outputs/results")
EXPS = {
    "q2a.audio": ("lalm_prompting_qwen2_audio_deepship_zero_shot_shot0_clip30.0_langen_enrichnone",
                  "lalm_prompting_qwen2_audio_deepship_few_shot_shot1_clip30.0_langen_enrichnone"),
    "vl8b.mel": ("vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone",
                 "vlm_spectrogram_qwen3_vl_8b_deepship_few_shot_shot5_clip30.0_specmel_langen_enrichnone"),
    "vl8b.stft": ("vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specstft_langen_enrichnone",
                  "vlm_spectrogram_qwen3_vl_8b_deepship_few_shot_shot5_clip30.0_specstft_langen_enrichnone"),
    "vl8b.demon": ("vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specdemon_langen_enrichnone",
                   "vlm_spectrogram_qwen3_vl_8b_deepship_few_shot_shot5_clip30.0_specdemon_langen_enrichnone"),
    "vl32b.mel": ("vlm_spectrogram_qwen3_vl_32b_deepship_zero_shot_shot0_clip30_specmel_langen_enrichnone", None),
    "vl32b.stft": ("vlm_spectrogram_qwen3_vl_32b_deepship_zero_shot_shot0_clip30_specstft_langen_enrichnone", None),
    "vl32b.demon": ("vlm_spectrogram_qwen3_vl_32b_deepship_zero_shot_shot0_clip30_specdemon_langen_enrichnone", None),
    "omni.audio": ("lalm_prompting_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_langen_enrichnone",
                   "lalm_prompting_qwen3_omni_30b_deepship_few_shot_shot5_clip30.0_langen_enrichnone"),
    "omni.mel": ("vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone",
                 "vlm_spectrogram_qwen3_omni_30b_deepship_few_shot_shot5_clip30.0_specmel_langen_enrichnone"),
    "omni.stft": ("vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specstft_langen_enrichnone",
                  "vlm_spectrogram_qwen3_omni_30b_deepship_few_shot_shot5_clip30.0_specstft_langen_enrichnone"),
    "omni.demon": ("vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specdemon_langen_enrichnone",
                   "vlm_spectrogram_qwen3_omni_30b_deepship_few_shot_shot5_clip30.0_specdemon_langen_enrichnone"),
    "omni.a+mel": ("fusion_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_fusionaudio+mel_langen_enrichnone",
                   "fusion_qwen3_omni_30b_deepship_few_shot_shot5_clip30.0_fusionaudio+mel_langen_enrichnone"),
    "omni.a+stft": ("fusion_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_fusionaudio+stft_langen_enrichnone", None),
    "omni.a+demon": ("fusion_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_fusionaudio+demon_langen_enrichnone", None),
    "omni.quad": ("fusion_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_fusionaudio+mel+stft+demon_langen_enrichnone", None),
}


def load(name):
    d = json.loads((R / f"{name}.json").read_text())
    df = pd.DataFrame(d["predictions"])
    return df.set_index(["recording_id", "clip_idx"]).sort_index()


def main():
    names = class_names("deepship", "en")
    lids = sorted(names)
    short = {i: names[i].split()[0][:6] for i in lids}
    preds = {}
    for k, (z, f) in EXPS.items():
        preds[f"{k}.z"] = load(z)
        if f:
            preds[f"{k}.f"] = load(f)
    ref = next(iter(preds.values()))
    truth = ref["label_id"]
    print(f"n_clips={len(ref)}  真实类别分布: " +
          " ".join(f"{short[i]}={v/len(truth):.2f}" for i, v in truth.value_counts().sort_index().items()))

    # ---- 1) prediction-class distribution + per-class recall (zero-shot) ----
    print("\n=== 预测类别分布 / 各实验（z=zero, f=few）===")
    print(f"{'experiment':<14}{'acc':>6}  " + " ".join(f"p:{short[i]:>6}" for i in lids) + "   " + " ".join(f"r:{short[i]:>6}" for i in lids))
    for k, df in preds.items():
        p = df["pred_label_id"]
        dist = [p.value_counts(normalize=True).get(i, 0.0) for i in lids]
        rec = [(p[df["label_id"] == i] == i).mean() if (df["label_id"] == i).any() else float("nan") for i in lids]
        print(f"{k:<14}{(p == df['label_id']).mean():>6.3f}  " +
              " ".join(f"{d:>8.2f}" for d in dist) + "   " + " ".join(f"{r:>8.2f}" for r in rec))

    # ---- 2) pairwise agreement across zero-shot experiments ----
    zs = [k for k in preds if k.endswith(".z")]
    print(f"\n=== 跨实验逐样本预测一致率（仅 zero-shot，下三角；随机独立≈Σp_a·p_b）===")
    hdr = f"{'':<14}" + "".join(f"{k.replace('.z',''):>11}" for k in zs)
    print(hdr)
    for a in zs:
        row = f"{a.replace('.z',''):<14}"
        for b in zs:
            if b == a:
                row += f"{'—':>11}"
            elif zs.index(b) < zs.index(a):
                j = preds[a]["pred_label_id"].to_frame("a").join(
                    preds[b]["pred_label_id"].to_frame("b"), how="inner")
                row += f"{(j.a == j.b).mean():>11.3f}"
            else:
                row += f"{'':>11}"
        print(row)

    # ---- 3) zero vs few within experiment ----
    print("\n=== 同实验 zero vs few 逐样本对比 ===")
    print(f"{'experiment':<14}{'agree':>7}{'n_changed':>10}{'acc|agree':>10}{'acc|changed':>12}{'Δacc':>7}")
    for k, (z, f) in EXPS.items():
        if not f:
            continue
        a, b = preds[f"{k}.z"], preds[f"{k}.f"]
        j = a[["pred_label_id", "label_id"]].join(b["pred_label_id"], rsuffix="_few", how="inner")
        same = j.pred_label_id == j.pred_label_id_few
        acc_same = (j.pred_label_id[same] == j.label_id[same]).mean()
        acc_chg = (j.pred_label_id_few[~same] == j.label_id[~same]).mean() if (~same).any() else float("nan")
        acc_z, acc_f = (j.pred_label_id == j.label_id).mean(), (j.pred_label_id_few == j.label_id).mean()
        print(f"{k:<14}{same.mean():>7.3f}{(~same).sum():>10}{acc_same:>10.3f}{acc_chg:>12.3f}{acc_f-acc_z:>+7.3f}")


if __name__ == "__main__":
    main()
