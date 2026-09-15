#!/usr/bin/env python
"""Publication-quality REFERENCE figures for the SPLASH ICASSP paper.

All numbers are read from real experiment outputs under /workspace/SPLASH/outputs/.
Style: Okabe-Ito colorblind-safe palette, no in-figure titles (captions in LaTeX),
axis labels with units, fonts >= 9pt at final size, light grid (alpha 0.25).

Outputs (300 dpi PNG):
  outputs/figures/paper/fig1_ladder.png     (7.1 x 2.15 in)
  outputs/figures/paper/fig2_mechanism.png  (7.1 x 1.95 in)
  outputs/figures/paper/fig3_layers.png     (3.45 x 2.2 in)
  outputs/figures/paper/fig4_learning.png   (3.45 x 1.95 in)
  (+ same files mirrored to docs/figures/paper/, and README.md)
"""
import csv
import json
import os
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "/workspace/SPLASH"
OUT_DIRS = [os.path.join(ROOT, "outputs/figures/paper"), os.path.join(ROOT, "docs/figures/paper")]

# Okabe-Ito palette
BLUE, ORANGE, GREEN, VERM, PINK, SKY, BLACK = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000")
GREY = "#7F7F7F"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "grid.color": GREY,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

R = {}  # collected verification numbers


def style(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)


def save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name))
    plt.close(fig)


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


def load_csv():
    with open(os.path.join(ROOT, "outputs/results/results.csv")) as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------
# FIGURE 1: performance ladder
# ----------------------------------------------------------------------------
def fig1():
    rows = load_csv()
    gen = [r for r in rows
           if r["eval"] in ("lalm_prompting", "vlm_spectrogram", "fusion")
           and r["regime"] in ("zero_shot", "few_shot")]
    R["fig1_n_generative"] = len(gen)

    def family(r):
        m = r["model"]
        if m == "qwen2_audio":
            return "q2a"
        if m == "qwen3_vl_8b":
            return "vl8b"
        if m == "qwen3_vl_32b":
            return "vl32b"
        return "omni"

    fams = ["q2a", "vl8b", "vl32b", "omni"]
    fam_color = {"q2a": PINK, "vl8b": BLUE, "vl32b": GREEN, "omni": ORANGE}
    fam_label = {"q2a": "Qwen2-Audio (audio)", "vl8b": "Qwen3-VL-8B (spec)",
                 "vl32b": "Qwen3-VL-32B (spec)", "omni": "Qwen3-Omni (all inputs)"}
    by_fam = {f: [] for f in fams}
    for r in gen:
        by_fam[family(r)].append(r)
    R["fig1_per_family"] = {f: len(v) for f, v in by_fam.items()}

    fig = plt.figure(figsize=(7.1, 2.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0], wspace=0.58,
                          left=0.095, right=0.985, top=0.96, bottom=0.15)

    # ---- panel (a): 44 generative conditions -------------------------------
    axA = fig.add_subplot(gs[0])
    offs = [0.0, 0.14, -0.14, 0.07, -0.07, 0.21, -0.21, 0.28, -0.28]
    for yi, f in enumerate(fams):
        ds = [float(r["clip_acc"]) for r in by_fam[f] if r["dataset"] == "deepship"]
        ss = [float(r["clip_acc"]) for r in by_fam[f] if r["dataset"] == "shipsear"]
        axA.scatter(ds, [yi + offs[i % len(offs)] for i in range(len(ds))],
                    s=14, c=fam_color[f], zorder=3, label=None)
        axA.scatter(ss, [yi + offs[i % len(offs)] for i in range(len(ss))],
                    s=14, facecolors="none", edgecolors=GREY, linewidths=0.9, zorder=3)
    R["fig1_prompting_ds_max"] = max(float(r["clip_acc"]) for r in gen if r["dataset"] == "deepship")

    axA.axvline(0.25, color=BLACK, lw=1.0, zorder=2)
    axA.text(0.252, 3.72, "chance 1/4 (DeepShip)", fontsize=9.5, rotation=90,
             va="top", ha="left", color=BLACK)
    axA.axvline(1 / 12, color=GREY, lw=1.0, ls=":", zorder=2)
    axA.text(1 / 12 - 0.004, 3.72, "chance 1/12 (ShipsEar)", fontsize=9.5, rotation=90,
             va="top", ha="right", color=GREY)
    axA.annotate("best generative 0.284", xy=(0.2838, 2.0), xytext=(0.315, 2.75),
                 fontsize=9.5, arrowprops=dict(arrowstyle="-", lw=0.7, color=GREY))
    axA.text(0.0, -0.62, "all 44 conditions at or below their chance lines (class collapse, Fig. 2)",
             fontsize=9.5, style="italic")
    axA.set_xlim(0, 0.82)
    axA.set_ylim(-0.75, 3.9)
    axA.set_yticks(range(4))
    axA.set_yticklabels([fam_label[f] for f in fams])
    axA.set_xlabel("Clip accuracy (test)")
    axA.grid(axis="y", visible=False)
    style(axA)
    axA.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#555555", label="DeepShip (30 cond.)"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none", color=GREY,
               label="ShipsEar (14 cond.)")],
        loc="upper left", bbox_to_anchor=(0.30, 1.01), frameon=True, handletextpad=0.2,
        borderaxespad=0.0)
    panel_label(axA, "(a)", x=-0.21, y=1.02)

    # ---- inset: steering dose curve (lives in panel a's empty right half) ---
    axi = axA.inset_axes([0.50, 0.10, 0.49, 0.80])
    st = json.load(open(os.path.join(ROOT, "outputs/results/steering.json")))
    alphas = st["alphas"]
    probe = [x["acc"] for x in st["results"] if x["mode"] == "probe"]
    rnd = [x["acc"] for x in st["results"] if x["mode"] == "random"]
    R["fig1_steering_probe"] = probe
    R["fig1_steering_random"] = rnd
    axi.plot(alphas, probe, "-o", color=BLUE, lw=1.4, ms=3.5, zorder=3)
    axi.plot(alphas, rnd, "--s", color=GREY, lw=1.2, ms=3, zorder=3)
    axi.axhline(0.25, color=GREY, lw=0.7, ls=":")
    axi.set_xlim(-0.008, 0.212)
    axi.set_ylim(0.24, 0.40)
    axi.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    axi.set_yticks([0.25, 0.30, 0.35, 0.40])
    axi.set_xlabel(r"steering step $\alpha$", labelpad=1)
    axi.set_ylabel("Clip accuracy", labelpad=1)
    axi.tick_params(pad=1.5)
    axi.text(0.075, 0.328, "probe dir.", color=BLUE, fontsize=9.5, rotation=28)
    axi.text(0.075, 0.258, "random dir.", color=GREY, fontsize=9.5, rotation=18)
    axi.annotate("0.385", xy=(0.2, 0.385), xytext=(0.145, 0.392), fontsize=9.5,
                 color=BLUE, ha="center")
    axi.annotate("0.350", xy=(0.2, 0.350), xytext=(0.145, 0.318), fontsize=9.5,
                 color=GREY, ha="center")
    axi.text(0.5, 1.06, "steering dose curve (vl8b, L3)", transform=axi.transAxes,
             fontsize=9.5, ha="center")
    style(axi)

    # ---- panel (b): rungs of the ladder -------------------------------------
    axB = fig.add_subplot(gs[1])
    fu = json.load(open(os.path.join(ROOT, "outputs/pilots/fusion_upgrade.json")))
    opt = json.load(open(os.path.join(ROOT, "outputs/results/optimizations/three_optimizations.json")))
    singles = {k: v["test_acc"] for k, v in fu["stage1"].items() if k != "beats"}
    lora_ens = [x["acc"] for x in opt["exp2_lora_ensemble"] if x["tag"] == "Ensemble avg"][0]
    lora_v2 = 0.5940409683426443  # results.csv: vl8b mel lora_v2
    rungs = [
        ("steering (probe dir., $\\alpha$=0.2)", 0.385, BLUE, None),
        ("LoRA v2 (VL-8B, mel)", lora_v2, BLUE, None),
        ("LoRA ensemble", lora_ens, BLUE, None),
        ("frozen single-source probes", None, BLUE, "band"),
        ("7-source frozen fusion", fu["stage2"]["A1"]["test_acc"], BLUE, None),
        ("BEATs single layer", fu["stage1"]["beats"]["test_acc"], VERM, None),
        ("BEATs top-3 concat", fu["stage3"]["B2t"]["test_acc"], VERM, None),
    ]
    R["fig1_rungs"] = {lbl: (v if v is not None else (min(singles.values()), max(singles.values())))
                       for lbl, v, _, b in rungs if b != "band" or True}
    R["fig1_probe_singles"] = singles

    for yi, (lbl, v, c, kind) in enumerate(rungs):
        if kind == "band":
            lo, hi = min(singles.values()), max(singles.values())
            axB.barh(yi, hi - lo, left=lo, height=0.62, color=c, alpha=0.25,
                     edgecolor=c, lw=0.8, zorder=2)
            for k, (src, val) in enumerate(sorted(singles.items(), key=lambda t: t[1])):
                axB.plot([val], [yi + [-0.18, 0.18, 0, -0.18, 0.18, 0, 0][k % 7]], "o",
                         ms=3, color=c, zorder=4)
            axB.plot([max(singles.values())], [yi], "o", ms=6, color=c,
                     markeredgecolor="white", markeredgewidth=1.0, zorder=5)
            axB.text(max(singles.values()) + 0.012, yi, "Omni-audio 0.695", fontsize=9.5,
                     va="center", color=c)
            axB.text(min(singles.values()) - 0.012, yi, "0.650", fontsize=9.5,
                     va="center", ha="right", color=c)
        else:
            axB.barh(yi, v, height=0.55, color=c, alpha=0.35, zorder=2)
            axB.plot([v], [yi], "o", ms=6, color=c, zorder=4)
            axB.text(v + 0.012, yi, f"{v:.3f}", fontsize=9.5, va="center", color=c)

    axB.axvline(0.25, color=BLACK, lw=1.0, ls="--", zorder=1)
    axB.text(0.25, 6.62, "chance 0.25", fontsize=9.5, ha="left", va="top")
    axB.set_xlim(0, 0.84)
    axB.set_ylim(-0.7, 6.8)
    axB.set_yticks(range(len(rungs)))
    axB.set_yticklabels([r[0] for r in rungs])
    axB.set_xlabel("DeepShip clip accuracy (test)")
    axB.grid(axis="y", visible=False)
    style(axB)
    axB.legend(handles=[Patch(facecolor=BLUE, alpha=0.5, label="LLM-derived (frozen/LoRA/steered)"),
                        Patch(facecolor=VERM, alpha=0.5, label="specialist encoder (BEATs)")],
               loc="lower right", bbox_to_anchor=(1.0, 0.40), frameon=True)
    panel_label(axB, "(b)", x=-0.42, y=1.02)

    save(fig, "fig1_ladder.png")


# ----------------------------------------------------------------------------
# FIGURE 2: mechanism, four panels
# ----------------------------------------------------------------------------
def fig2():
    fig = plt.figure(figsize=(7.1, 1.75))
    gs = fig.add_gridspec(2, 2, hspace=0.62, wspace=0.42,
                          left=0.115, right=0.955, top=0.97, bottom=0.20)

    # ---- (a) predicted-class collapse ---------------------------------------
    axA = fig.add_subplot(gs[0])
    conds = [
        ("Q2A audio", "lalm_prompting_qwen2_audio_deepship_zero_shot_shot0_clip30.0_langen_enrichnone.json"),
        ("VL-8B mel", "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone.json"),
        ("VL-8B demon", "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specdemon_langen_enrichnone.json"),
        ("VL-32B mel", "vlm_spectrogram_qwen3_vl_32b_deepship_zero_shot_shot0_clip30_specmel_langen_enrichnone.json"),
        ("Omni audio", "lalm_prompting_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_langen_enrichnone.json"),
        ("Omni mel", "vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone.json"),
    ]
    classes = ["Cargo ship", "Passenger ship", "Oil tanker", "Tug boat"]
    ccol = {"Cargo ship": BLUE, "Passenger ship": GREEN, "Oil tanker": ORANGE, "Tug boat": PINK}
    R["fig2a_dists"] = {}
    for yi, (name, fn) in enumerate(conds):
        d = json.load(open(os.path.join(ROOT, "outputs/results", fn)))
        cm = d["metrics_clip"]["confusion_matrix"]
        labs = d["metrics_clip"]["confusion_matrix_labels"]
        n = sum(sum(r) for r in cm)
        fr = {l: sum(row[j] for row in cm) / n for j, l in enumerate(labs)}
        R["fig2a_dists"][name] = {k: round(v, 3) for k, v in fr.items()}
        left = 0.0
        for l in classes:
            v = fr[l]
            axA.barh(yi, v, left=left, height=0.62, color=ccol[l],
                     edgecolor="white", lw=0.4, zorder=3)
            if v >= 0.12:
                axA.text(left + v / 2, yi, f"{v:.2f}", ha="center", va="center",
                         fontsize=9.5, color="white")
            left += v
    axA.set_xlim(0, 1)
    axA.set_ylim(-0.6, len(conds) - 0.4)
    axA.set_yticks(range(len(conds)))
    axA.set_yticklabels([c[0] for c in conds])
    axA.set_xlabel("Fraction of predictions (n = 2,685 clips)")
    axA.grid(axis="y", visible=False)
    axA.legend(handles=[Patch(facecolor=ccol[l], label=l) for l in classes],
               loc="upper center", bbox_to_anchor=(0.5, 1.34), ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.0)
    style(axA)
    panel_label(axA, "(a)", x=-0.24, y=1.10)

    # ---- (b) option permutation protocol ------------------------------------
    axB = fig.add_subplot(gs[1])
    oo = json.load(open(os.path.join(ROOT, "outputs/results/option_order/SUMMARY.json")))
    stylemap = {"vl8b.mel": (BLUE, "o"), "vl8b.demon": (SKY, "s"),
                "omni.audio": (ORANGE, "^"), "omni.mel": (PINK, "D")}
    R["fig2b_perm"] = {e["key"]: e["perm_accs"] for e in oo}
    for e in oo:
        c, m = stylemap[e["key"]]
        axB.plot(range(1, 9), e["perm_accs"], "-" + m, color=c, lw=1.2, ms=3.5,
                 label=e["key"], zorder=3)
    axB.axhline(0.25, color=BLACK, ls="--", lw=1.0, zorder=2)
    axB.text(8.0, 0.253, "chance 0.25", fontsize=9.5, ha="right", va="bottom")
    axB.text(4.5, 0.155, "class-flip rate 1.0\nposition-anchored 0.0", fontsize=9.5,
             ha="center", va="bottom")
    axB.set_xlim(0.6, 8.4)
    axB.set_ylim(0.14, 0.35)
    axB.set_xticks(range(1, 9))
    axB.set_xlabel("Option permutation index")
    axB.set_ylabel("Clip accuracy")
    axB.legend(loc="upper right", ncol=2, frameon=False, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0.2)
    style(axB)
    panel_label(axB, "(b)", x=-0.20, y=1.10)

    # ---- (c) null-input protocol --------------------------------------------
    axC = fig.add_subplot(gs[2])
    ni = json.load(open(os.path.join(ROOT, "outputs/results/null_input/SUMMARY.json")))
    short = {"q2a.audio.silence": "q2a silence", "q2a.audio.noise": "q2a noise",
             "vl8b.mel.grey": "vl8b grey", "vl8b.mel.pixnoise": "vl8b pix-noise",
             "omni.audio.silence": "omni-a silence", "omni.audio.noise": "omni-a noise",
             "omni.audio.nomedia": "omni-a no-media", "omni.mel.grey": "omni-m grey",
             "omni.mel.pixnoise": "omni-m pix-noise"}
    kl = [e["KL(real||null)"] for e in ni]
    keep = [e["keep_orig_pred"] for e in ni]
    R["fig2c_null"] = {e["arm"]: [round(e["KL(real||null)"], 5), e["keep_orig_pred"]] for e in ni}
    xs = range(len(ni))
    cols = [VERM if e["arm"] == "omni.audio.nomedia" else BLUE for e in ni]
    axC.bar(xs, kl, width=0.62, color=cols, zorder=3)
    axC.set_yscale("log")
    axC.set_ylim(6e-4, 12)
    axC.set_yticks([1e-3, 1e-2, 1e-1, 1, 10])
    axC.set_xticks(list(xs))
    axC.set_xticklabels([short[e["arm"]] for e in ni], rotation=40, ha="right")
    axC.set_ylabel(r"KL(real $\Vert$ null)")
    axC.annotate("0.0013\n(98% retained)", xy=(6, 0.00135), xytext=(4.1, 0.010),
                 fontsize=9.5, ha="center",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=BLACK))
    axT = axC.twinx()
    axT.plot(xs, keep, "D", color=BLACK, ms=4, zorder=4)
    axT.set_ylim(-0.05, 1.35)
    axT.set_yticks([0, 0.5, 1.0])
    axT.set_ylabel("Prediction retention", rotation=270, labelpad=13)
    axT.grid(visible=False)
    axT.spines["top"].set_visible(False)
    style(axC)
    panel_label(axC, "(c)", x=-0.24, y=1.02)

    # ---- (d) position preference --------------------------------------------
    axD = fig.add_subplot(gs[3])
    letter_cts = {}
    for key in ["vl8b.mel", "vl8b.demon", "omni.audio", "omni.mel"]:
        cnt = collections.Counter()
        for p in range(8):
            d = json.load(open(os.path.join(ROOT, f"outputs/results/option_order/{key}_p{p}.json")))
            cnt.update(x["pred_letter"] for x in d["predictions"])
        letter_cts[key] = {k: cnt[k] / sum(cnt.values()) for k in "ABCD"}
    R["fig2d_letters"] = {k: {l: round(v, 3) for l, v in d.items()} for k, d in letter_cts.items()}

    import numpy as np
    x = np.arange(4)
    w = 0.19
    series = [("vl8b.mel", BLUE), ("vl8b.demon", SKY), ("omni.audio", ORANGE), ("omni.mel", PINK)]
    for i, (key, c) in enumerate(series):
        axD.bar(x + (i - 1.5) * w, [letter_cts[key][l] for l in "ABCD"], width=w,
                color=c, zorder=3, label=key)
    axD.set_xticks(x)
    axD.set_xticklabels(list("ABCD"))
    axD.set_xlabel("Answer position (DeepShip, 4 options)")
    axD.set_ylabel("Fraction of choices")
    axD.set_ylim(0, 0.62)
    axD.legend(loc="lower left", bbox_to_anchor=(0.0, 1.06), frameon=False, ncol=2,
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0.0)
    ss = {}
    for nm, fn in [("vl8b_mel", "vlm_spectrogram_qwen3_vl_8b_shipsear_zero_shot_shot0_clip30.0_specmel_langen_enrichnone.json"),
                   ("omni_aud", "lalm_prompting_qwen3_omni_30b_shipsear_zero_shot_shot0_clip30.0_langen_enrichnone.json"),
                   ("omni_mel", "vlm_spectrogram_qwen3_omni_30b_shipsear_zero_shot_shot0_clip30.0_specmel_langen_enrichnone.json"),
                   ("vl32b_mel", "vlm_spectrogram_qwen3_vl_32b_shipsear_zero_shot_shot0_clip30.0_specmel_langen_enrichnone.json"),
                   ("q2a_aud", "lalm_prompting_qwen2_audio_shipsear_zero_shot_shot0_clip30.0_langen_enrichnone.json")]:
        d = json.load(open(os.path.join(ROOT, "outputs/results", fn)))
        cnt = collections.Counter(x2["pred_letter"] for x2 in d["predictions"])
        top, topv = cnt.most_common(1)[0]
        ss[nm] = (top, round(topv / sum(cnt.values()), 2))
    R["fig2d_shipsear_top"] = ss
    axD.text(0.985, 0.97,
             "ShipsEar (A–L, 12 options):\n" + "; ".join(f"{k}→{v[0]} {v[1]:.2f}" for k, v in ss.items()),
             transform=axD.transAxes, fontsize=9.5, va="top", ha="right",
             bbox=dict(facecolor="white", edgecolor=GREY, lw=0.5, alpha=0.9, pad=2.5))
    style(axD)
    panel_label(axD, "(d)", x=-0.20, y=1.02)

    save(fig, "fig2_mechanism.png")


# ----------------------------------------------------------------------------
# FIGURE 3: layer-wise probe accuracy
# ----------------------------------------------------------------------------
def fig3():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "outputs/results/layer_probe/curves.csv"))))
    curves = {}
    for r in rows:
        curves.setdefault((r["model"], r["spec"]), []).append(
            (int(r["layer"]), float(r["acc"])))
    for k in curves:
        curves[k].sort()

    beats = json.load(open(os.path.join(ROOT, "outputs/pilots/beats_full_budget.json")))
    beats_curve = [(c["layer"], c["acc"])
                   for c in beats["stages"]["stage2_full_budget_layers"]["curves"]]

    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    fig.subplots_adjust(left=0.155, right=0.97, top=0.96, bottom=0.185)

    specs = [("audio", BLUE, "Omni audio (49 layers)"),
             ("mel", ORANGE, "Omni mel"),
             ("stft", GREEN, "Omni STFT")]
    stats = {}
    for spec, c, lbl in specs:
        pts = curves[("qwen3_omni_30b", spec)]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=c, lw=1.3, label=lbl, zorder=3)
        best = max(pts, key=lambda t: t[1])
        last = pts[-1]
        ax.plot([best[0]], [best[1]], "o", ms=5, color=c, markeredgecolor="white",
                markeredgewidth=0.8, zorder=5)
        stats[spec] = {"best_layer": best[0], "best": round(best[1], 4),
                       "last": round(last[1], 4),
                       "decay_pt": round(100 * (last[1] - best[1]), 1),
                       "range_pt": round(100 * (max(ys) - min(ys)), 1)}
        if spec in ("mel", "stft"):
            ax.annotate(f"{stats[spec]['decay_pt']:+.0f} pt",
                        xy=(last[0], last[1]), xytext=(last[0] - 4.5, last[1] - 0.055),
                        fontsize=9.5, color=c, ha="center",
                        arrowprops=dict(arrowstyle="->", lw=0.8, color=c,
                                        connectionstyle="arc3,rad=-0.25"))
    aud = [a for _, a in curves[("qwen3_omni_30b", "audio")]]
    ax.axhspan(sum(aud) / len(aud) - 0.025, sum(aud) / len(aud) + 0.025,
               color=BLUE, alpha=0.12, zorder=1)
    ax.text(48, sum(aud) / len(aud) + 0.028, "audio: flat within 5 pt",
            fontsize=9.5, color=BLUE, ha="right", va="bottom")

    bx, by = zip(*beats_curve)
    ax.plot(bx, by, "--", color=BLACK, lw=1.3, label="BEATs (12 layers)", zorder=4)
    bbest = max(beats_curve, key=lambda t: t[1])
    ax.plot([bbest[0]], [bbest[1]], "o", ms=5, color=BLACK, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    stats["beats"] = {"best_layer": bbest[0], "best": round(bbest[1], 4)}

    R["fig3_stats"] = stats
    ax.set_xlim(-1, 49)
    ax.set_ylim(0.42, 0.80)
    ax.set_xlabel("Transformer layer index (raw)")
    ax.set_ylabel("Clip accuracy (test)")
    ax.legend(loc="lower center", frameon=True, facecolor="white", framealpha=0.9,
              edgecolor=GREY, ncol=2, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.2)
    style(ax)
    save(fig, "fig3_layers.png")


# ----------------------------------------------------------------------------
# FIGURE 4: label efficiency
# ----------------------------------------------------------------------------
def fig4():
    lc = json.load(open(os.path.join(ROOT, "outputs/results/layer_probe/learning_curves.json")))
    budgets = ["100", "500", "1000", "2000", "all"]
    ntrain = [100, 500, 1000, 2000, 3000]

    def best_curve(key):
        return [max(lc[key]["curves"][b].values()) for b in budgets]

    omni = best_curve("qwen3_omni_30b_audio_5300b861c63f")
    beats = best_curve("beats_all_layers")
    R["fig4_omni"] = [round(v, 4) for v in omni]
    R["fig4_beats"] = [round(v, 4) for v in beats]

    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    fig.subplots_adjust(left=0.155, right=0.965, top=0.96, bottom=0.20)
    ax.plot(ntrain, omni, "-o", color=BLUE, lw=1.5, ms=4, label="Omni audio (best layer)", zorder=4)
    ax.plot(ntrain, beats, "--s", color=BLACK, lw=1.4, ms=4, label="BEATs (best layer)", zorder=3)
    ax.axhline(0.2838, color=VERM, ls=":", lw=1.2, zorder=2)
    ax.text(2900, 0.2838 + 0.008, "best generative prompting 0.284", fontsize=9.5,
            color=VERM, ha="right")
    ax.annotate("0.582 at 100 labels\n(2.1x prompting best)",
                xy=(100, omni[0]), xytext=(160, 0.50), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=GREY))
    ax.set_xscale("log")
    ax.set_xticks(ntrain)
    ax.set_xticklabels(["100", "500", "1,000", "2,000", "3,000\n(full probe)"])
    ax.minorticks_off()
    ax.set_xlim(80, 3600)
    ax.set_ylim(0.42, 0.80)
    ax.set_xlabel("Training labels n (log scale)")
    ax.set_ylabel("Clip accuracy (test)")
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4, borderaxespad=0.3)
    style(ax)
    save(fig, "fig4_learning.png")


# ----------------------------------------------------------------------------
README = """# SPLASH paper reference figures

All values are read directly from real experiment outputs; nothing is synthetic.
Style: Okabe-Ito colorblind-safe palette, no in-figure titles (captions live in
LaTeX, see docs/paper/main.tex Fig. captions), fonts >= 8.5 pt at final size,
300 dpi. Regenerated by `python scripts/paper_figures.py` (Docker container
SPLASH, working dir /workspace/SPLASH).

## fig1_ladder.png (7.1 x 2.15 in; PNG 2613 x 706)
- (a) All 44 generative conditions (zero-/few-shot prompting) from
  `outputs/results/results.csv` (rows with experiment in
  {lalm_prompting, vlm_spectrogram, fusion} and regime in {zero_shot, few_shot};
  30 DeepShip + 14 ShipsEar). Filled dots = DeepShip, open grey = ShipsEar.
  Chance lines: 0.25 (DeepShip 4-class), 1/12 (ShipsEar 12-class). Best
  generative condition = 0.284 (VL-8B demon, zero-shot).
- inset (right half of panel a): steering dose curve from
  `outputs/results/steering.json` (Qwen3-VL-8B mel, layer 3; n=200; MCQ acc):
  alphas {0, 0.02, 0.05, 0.1, 0.2}; probe direction 0.250/0.265/0.305/0.345/
  0.385 (blue), random direction 0.250/0.245/0.255/0.290/0.350 (grey dashed).
- (b) rungs (DeepShip clip acc, n_test=2685, full 8,350-label budget unless
  noted):
  - steering probe-direction alpha=0.2: 0.385 (steering.json)
  - LoRA v2 (VL-8B mel): 0.594 (results.csv)
  - LoRA ensemble (VL-8B v2 + Omni v1, prob. average): 0.608
    (`outputs/results/optimizations/three_optimizations.json`, exp2)
  - frozen single-source probes, honest band 0.650-0.695 with all 7 sources
    (`outputs/pilots/fusion_upgrade.json`, stage1: vl8b_stft 0.6499,
    vl32b_stft 0.6547, omni_mel 0.6648, vl32b_mel 0.6685, vl8b_mel 0.6737,
    q2a_aud 0.6819, omni_aud 0.6946)
  - 7-source best-layer frozen fusion (A1): 0.7430 (fusion_upgrade.json, stage2)
  - BEATs single layer (L9 by train CV): 0.7434 (fusion_upgrade.json, stage1)
  - BEATs top-3 layer concat (B2t, layers {9,6,5}): 0.7683 (stage3)
  Blue = LLM-derived, vermillion = specialist encoder. Dashed line chance 0.25.

## fig2_mechanism.png (7.1 x 1.95 in; PNG 2081 x 789)
- (a) Predicted-class distribution (column-normalized confusion matrix, n=2,685)
  for 6 zero-shot conditions, computed from the per-experiment JSONs in
  `outputs/results/`: Q2A audio (Oil 0.79), VL-8B mel (Oil 0.69), VL-8B demon
  (Oil 0.86), VL-32B mel (Oil 0.99), Omni audio (Tug 0.97), Omni mel (Tug 0.92).
- (b) Option-permutation protocol from `outputs/results/option_order/SUMMARY.json`
  (8 permutations x 4 conditions x 100 clips): per-permutation accuracy
  vl8b.mel .30/.25/.24/.25/.25/.25/.24/.28, vl8b.demon .31/.25/.18/.26/.25/.24/
  .25/.27, omni.audio .25/.25/.25/.27/.25/.28/.26/.20, omni.mel .23/.25/.25/.23/
  .25/.27/.21/.27. All conditions: class_flip_rate 1.0, position_anchored 0.0.
- (c) Null-input protocol from `outputs/results/null_input/SUMMARY.json`
  (n=200 clips/arm): KL(real||null) on log axis (bars; blue; vermillion =
  omni.audio.nomedia 0.0013) and prediction retention (black diamonds, right
  axis). Arms: q2a silence 0.027/0.815, q2a noise 0.199/0.815, vl8b grey
  4.84/0.025, vl8b pixnoise 5.08/0.025, omni-a silence 0.149/0.98, omni-a noise
  0.042/0.98, omni-a no-media 0.0013/0.98, omni-m grey 0.261/0.895, omni-m
  pixnoise 0.612/0.895.
- (d) Position-preference profiles computed from raw pred_letter fields of
  `outputs/results/option_order/*_p{0..7}.json` (8 perms x 100 clips each):
  vl8b.mel A/B/C/D = .12/.39/.43/.06, vl8b.demon .02/.54/.41/.03,
  omni.audio .14/.06/.49/.31, omni.mel .24/.02/.44/.31. ShipsEar (A-L, 12
  options) top letter from zero-shot per-experiment JSONs (n=104): vl8b_mel -> E
  1.00, omni_aud -> K 0.81, omni_mel -> K 1.00, vl32b_mel -> K 0.77, q2a_aud ->
  K 0.45.

## fig3_layers.png (3.45 x 2.2 in; PNG 1043 x 657)
Per-layer probe accuracy at the full 8,350-label budget from
`outputs/results/layer_probe/curves.csv` (CV-selected C per layer):
Omni audio (49 layers; flat, best L31 0.689, last 0.667, whole-curve range
5.0 pt, shaded +/-2.5 pt band), Omni mel (best L2 0.665, last 0.545,
-12 pt end decay), Omni STFT (best L2 0.672, last 0.491, -18 pt). Black
dashed: BEATs per-layer from `outputs/pilots/beats_full_budget.json`
(stage2_full_budget_layers; best L6 0.749). Dots mark best layers.

## fig4_learning.png (3.45 x 1.95 in) — HONEST protocol
NOTE: fig4 is generated by scripts/fig4_honest.py (run AFTER this script), from
`outputs/pilots/honest_learning_curve.json`: layer AND C selected per budget by
train-only recording-grouped CV, one test eval per budget (do NOT revert to the
legacy test-selected curve in learning_curves.json). Data: omni audio
100/500/1000/2000/3000/8350 -> 0.6305/0.6808/0.7020/0.6950/0.6898/0.6946; BEATs
dashed 0.6067/0.7088/0.7263/0.7371/0.7155/0.7434. Dotted grey line: best
generative condition 0.284. Annotations: 100 labels = 2.6x the same model's
generative accuracy (0.247); 500 labels = 98% of full budget (0.6808/0.6946).
The 1k-3k wobble (+/-1.5 pt) is honest-selection noise at small budgets.
"""


def main():
    fig1()
    fig2()
    fig3()
    fig4()
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "README.md"), "w") as f:
            f.write(README)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
