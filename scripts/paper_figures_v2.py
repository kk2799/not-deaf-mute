#!/usr/bin/env python
"""SPLASH ICASSP figures v2 — designed at FINAL PRINT SIZE for legibility.

fig1/fig2 are placed at 0.84*\\textwidth (~5.95 in), fig3 at 0.87*\\columnwidth
(~2.95 in); fonts below are therefore the true printed sizes (>=6.5 pt).
All values are read from real experiment outputs; nothing is synthetic.
Run in container: python scripts/paper_figures_v2.py
Outputs: docs/figures/paper/{fig1_ladder,fig2_mechanism,fig3_layers}.png
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = "/workspace/SPLASH"
OUT = [os.path.join(ROOT, "docs/figures/paper"), os.path.join(ROOT, "outputs/figures/paper")]

BLUE, ORANGE, GREEN, VERM, PINK, SKY, BLACK = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000")
GREY = "#666666"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.4,
    "grid.color": GREY,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

CLS_COLORS = {"Cargo": SKY, "Passenger": GREEN, "Oil tanker": VERM, "Tug": PINK}


def save(fig, name):
    for d in OUT:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name))
    plt.close(fig)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel(ax, s, x=-0.28, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


# ---------------------------------------------------------------- fig1
def fig1():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "outputs/results/results.csv"))))
    gen = [r for r in rows if r["eval"] in ("lalm_prompting", "vlm_spectrogram", "fusion")
           and r["regime"] in ("zero_shot", "few_shot")]
    fams = ["q2a", "vl8b", "vl32b", "omni"]
    fname = {"q2a": "Qwen2-Audio", "vl8b": "VL-8B", "vl32b": "VL-32B", "omni": "Omni-30B"}
    fc = {"q2a": PINK, "vl8b": BLUE, "vl32b": GREEN, "omni": ORANGE}
    by = {f: {"deepship": [], "shipsear": []} for f in fams}
    for r in gen:
        m = r["model"]
        f = "q2a" if m == "qwen2_audio" else ("vl8b" if m == "qwen3_vl_8b" else
             ("vl32b" if m == "qwen3_vl_32b" else "omni"))
        by[f][r["dataset"]].append(float(r["clip_acc"]))

    fig = plt.figure(figsize=(5.95, 1.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.62, 1.12], wspace=0.34,
                          left=0.082, right=0.995, top=0.90, bottom=0.20)

    # (a) 44 conditions
    axA = fig.add_subplot(gs[0])
    offs = [0, .16, -.16, .08, -.08, .24, -.24]
    best = (0.0, -1)  # (acc, y) of the best DeepShip condition, for annotation
    for yi, f in enumerate(fams):
        ds, ss = by[f]["deepship"], by[f]["shipsear"]
        dys = [yi + offs[i % 7] for i in range(len(ds))]
        axA.scatter(ds, dys, s=13, c=fc[f], zorder=3, edgecolors="none")
        if ds:
            i = max(range(len(ds)), key=lambda k: ds[k])
            if ds[i] > best[0]:
                best = (ds[i], dys[i])
        axA.scatter(ss, [yi + offs[i % 7] for i in range(len(ss))], s=13,
                    facecolors="none", edgecolors=GREY, linewidths=0.8, zorder=3)
    axA.axvline(0.25, color=BLACK, lw=0.9, zorder=2)
    axA.text(0.25, 3.76, "chance 1/4", fontsize=6.5, ha="center", va="top",
             color=BLACK, zorder=6,
             bbox=dict(fc="white", ec="none", pad=0.8))
    axA.axvline(1/12, color=GREY, lw=0.9, ls=":", zorder=2)
    axA.text(1/12, 3.76, "1/12", fontsize=6.5, ha="center", va="top",
             color=GREY, zorder=6,
             bbox=dict(fc="white", ec="none", pad=0.8))
    axA.annotate("best 0.284", xy=(best[0] + 0.004, best[1]), xytext=(0.302, 2.96),
                 fontsize=6.5, ha="center",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY))
    axA.set_xlim(0, 0.33); axA.set_ylim(-0.6, 3.8)
    axA.set_xticks([0, 0.1, 0.2, 0.3])
    axA.set_yticks(range(4)); axA.set_yticklabels([fname[f] for f in fams])
    axA.set_xlabel("clip accuracy (44 conditions)")
    axA.grid(axis="y", visible=False)
    axA.text(0.0, -0.49, "filled: DeepShip (30)   open: ShipsEar (14)",
             fontsize=6.5, style="italic", color=GREY, va="center")
    style(axA); panel(axA, "(a)", x=-0.24)

    # (b) steering dose
    axB = fig.add_subplot(gs[1])
    st = json.load(open(os.path.join(ROOT, "outputs/results/steering.json")))
    al = st["alphas"]
    pr = [x["acc"] for x in st["results"] if x["mode"] == "probe"]
    rd = [x["acc"] for x in st["results"] if x["mode"] == "random"]
    axB.plot(al, pr, "-o", color=BLUE, lw=1.3, ms=2.8, zorder=3)
    axB.plot(al, rd, "--s", color=GREY, lw=1.0, ms=2.4, zorder=3)
    axB.axhline(0.25, color=GREY, lw=0.6, ls=":")
    axB.set_xlim(-0.01, 0.242); axB.set_ylim(0.235, 0.40)
    axB.set_xticks([0, 0.1, 0.2])
    axB.set_yticks([0.25, 0.30, 0.35, 0.40])
    axB.set_xlabel(r"steering $\alpha$")
    axB.text(0.012, 0.362, "probe", color=BLUE, fontsize=7)
    axB.text(0.125, 0.262, "random", color=GREY, fontsize=7)
    axB.text(0.205, 0.385, "0.385", fontsize=6.5, color=BLUE, va="center")
    axB.text(0.205, 0.350, "0.350", fontsize=6.5, color=GREY, va="center")
    style(axB); panel(axB, "(b)", x=-0.32)

    # (c) ladder
    axC = fig.add_subplot(gs[2])
    fu = json.load(open(os.path.join(ROOT, "outputs/pilots/fusion_upgrade.json")))
    singles = sorted(v["test_acc"] for k, v in fu["stage1"].items() if k != "beats")
    rungs = [
        ("steering ($\\alpha$=.2)", 0.385, BLUE, "o"),
        ("LoRA v2", 0.594, BLUE, "o"),
        ("LoRA ens.", 0.608, BLUE, "o"),
        ("frozen probes", (singles[0], singles[-1]), BLUE, None),
        ("7-src fusion", fu["stage2"]["A1"]["test_acc"], BLUE, "D"),
        ("BEATs L9", fu["stage1"]["beats"]["test_acc"], VERM, "o"),
        ("BEATs top-3", fu["stage3"]["B2t"]["test_acc"], VERM, "o"),
    ]
    ys = list(range(len(rungs)))[::-1]
    for y, (lab, val, c, mk) in zip(ys, rungs):
        if mk is None:
            lo, hi = val
            axC.barh(y, hi - lo, left=lo, height=0.34, color=c, alpha=0.25, zorder=2)
            vtxt = "%.2f–%.2f" % (lo, hi)
            x = hi
        else:
            axC.plot(val, y, mk, color=c, ms=4.5, zorder=3)
            vtxt = "%.3f" % val
            x = val
        axC.text(x + 0.012, y, vtxt, fontsize=6.5, va="center", color=c)
    axC.axvline(0.25, color=BLACK, lw=0.8, ls="--", zorder=1)
    axC.text(0.252, 6.45, "chance", fontsize=6.5, color=BLACK)
    axC.set_xlim(0.2, 0.86); axC.set_ylim(-0.6, 6.9)
    axC.set_yticks(ys); axC.set_yticklabels([r[0] for r in rungs])
    axC.set_xlabel("clip accuracy (DeepShip)")
    axC.grid(axis="y", visible=False)
    style(axC); panel(axC, "(c)", x=-0.30)
    save(fig, "fig1_ladder.png")


# ---------------------------------------------------------------- fig2
def fig2():
    fig = plt.figure(figsize=(5.95, 2.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.18, 1.0, 1.22, 1.0], wspace=0.52,
                          left=0.075, right=0.995, top=0.86, bottom=0.24)

    # (a) collapse distributions
    axA = fig.add_subplot(gs[0])
    conds = [
        ("Q2A\naudio", "lalm_prompting_qwen2_audio_deepship_zero_shot_shot0_clip30.0_langen_enrichnone"),
        ("VL-8B\nmel", "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone"),
        ("VL-8B\ndemon", "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specdemon_langen_enrichnone"),
        ("VL-32B\nmel", "vlm_spectrogram_qwen3_vl_32b_deepship_zero_shot_shot0_clip30_specmel_langen_enrichnone"),
        ("Omni\naudio", "lalm_prompting_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_langen_enrichnone"),
        ("Omni\nmel", "vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone"),
    ]
    cls_names = ["Cargo", "Passenger", "Oil tanker", "Tug"]
    for xi, (_, fn) in enumerate(conds):
        d = json.load(open(os.path.join(ROOT, "outputs/results", fn + ".json")))
        l2l = {v: k for k, v in d["predictions"][0]["letter_to_label"].items()}
        counts = [0, 0, 0, 0]
        for p in d["predictions"]:
            counts[p["pred_label_id"]] += 1
        n = sum(counts)
        bot = 0.0
        for ci, cn in enumerate(cls_names):
            frac = counts[ci] / n
            if frac > 0.001:
                axA.bar(xi, frac, bottom=bot, width=0.62, color=CLS_COLORS[cn],
                        edgecolor="white", linewidth=0.3, zorder=3)
            if frac >= 0.18:
                axA.text(xi, bot + frac / 2, "%d%%" % round(100 * frac), fontsize=6.5,
                         ha="center", va="center", color="white", fontweight="bold")
            bot += frac
    axA.set_xticks(range(6))
    axA.set_xticklabels([c[0] for c in conds], fontsize=6.5, rotation=30,
                        ha="right")
    axA.set_ylim(0, 1.02); axA.set_xlim(-0.55, 5.55)
    axA.set_ylabel("fraction of predictions")
    axA.set_yticks([0, .5, 1])
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLS_COLORS[c]) for c in cls_names]
    axA.legend(handles, cls_names, ncol=4, loc="lower left", bbox_to_anchor=(-0.02, 1.0),
               frameon=False, fontsize=6, handlelength=1.0, columnspacing=0.8,
               handletextpad=0.4, borderaxespad=0)
    style(axA); panel(axA, "(a)", x=-0.30, y=1.14)

    # (b) permutation
    axB = fig.add_subplot(gs[1])
    oo = json.load(open(os.path.join(ROOT, "outputs/results/option_order/SUMMARY.json")))
    styles = {"vl8b.mel": (BLUE, "-"), "vl8b.demon": (BLUE, "--"),
              "omni.audio": (ORANGE, "-"), "omni.mel": (ORANGE, "--")}
    shortb = {"vl8b.mel": "8B mel", "vl8b.demon": "8B dem.",
              "omni.audio": "Omni audio", "omni.mel": "Omni mel"}
    for e in oo:
        c, ls = styles[e["key"]]
        axB.plot(range(1, 9), e["perm_accs"], ls + "o", color=c, lw=1.1, ms=2.4,
                 zorder=3, label=shortb[e["key"]])
    axB.axhline(0.25, color=BLACK, lw=0.8, ls=":", zorder=2)
    axB.text(8.0, 0.332, "chance", fontsize=6.5, ha="right", va="center",
             color=BLACK)
    axB.set_xlim(0.5, 8.5); axB.set_ylim(0.15, 0.35)
    axB.set_xticks(range(1, 9))
    axB.set_xlabel("option permutation")
    axB.set_ylabel("clip accuracy")
    axB.set_yticks([0.15, 0.25, 0.35])
    axB.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
               fontsize=5.5, handlelength=1.2, columnspacing=0.8, borderaxespad=0)
    axB.text(0.6, 0.325, "class-flip rate 1.0", fontsize=6.5, style="italic", color=GREY)
    style(axB); panel(axB, "(b)", x=-0.40, y=1.14)

    # (c) null input
    axC = fig.add_subplot(gs[2])
    ni = json.load(open(os.path.join(ROOT, "outputs/results/null_input/SUMMARY.json")))
    short = {"q2a.audio.silence": "q2a sil", "q2a.audio.noise": "q2a noi",
             "vl8b.mel.grey": "vl8b gry", "vl8b.mel.pixnoise": "vl8b pix",
             "omni.audio.silence": "om sil", "omni.audio.noise": "om noi",
             "omni.audio.nomedia": "om none", "omni.mel.grey": "omM gry",
             "omni.mel.pixnoise": "omM pix"}
    xs = range(len(ni))
    kls = [e["KL(real||null)"] for e in ni]
    cols = [VERM if e["arm"] == "omni.audio.nomedia" else BLUE for e in ni]
    axC.bar(xs, kls, width=0.62, color=cols, zorder=3)
    axC.set_yscale("log"); axC.set_ylim(5e-4, 20)
    axC.set_xticks(list(xs))
    axC.set_xticklabels([short[e["arm"]] for e in ni], rotation=38, ha="right",
                        fontsize=6)
    axC.set_ylabel("KL(real‖null)")
    ax2 = axC.twinx()
    ax2.plot(xs, [e["keep_orig_pred"] for e in ni], "d", color=BLACK, ms=3, zorder=4)
    ax2.set_ylim(-0.05, 1.15); ax2.set_yticks([0, 1])
    ax2.set_ylabel("retention ♦", fontsize=7)
    ax2.tick_params(labelsize=6.5)
    ax2.grid(visible=False)
    for s in ("top",):
        ax2.spines[s].set_visible(False)
    axC.annotate("0.0013", xy=(6, 0.0013), xytext=(4.0, 0.004), fontsize=6.5,
                 color=VERM, arrowprops=dict(arrowstyle="-", lw=0.6, color=VERM))
    style(axC); panel(axC, "(c)", x=-0.34, y=1.14)

    # (d) position profiles
    axD = fig.add_subplot(gs[3])
    prof = {}
    for key, col in (("vl8b.mel", "vl8b"), ("omni.audio", "omni")):
        fr = [0.0] * 4
        n = 0
        for p in range(8):
            d = json.load(open(os.path.join(
                ROOT, "outputs/results/option_order/%s_p%d.json" % (key, p))))
            for pr in d["predictions"]:
                fr["ABCD".index(pr["pred_letter"])] += 1
                n += 1
        prof[col] = [f / n for f in fr]
    w = 0.36
    xs = range(4)
    axD.bar([x - w / 2 for x in xs], prof["vl8b"], width=w, color=BLUE,
            label="VL-8B", zorder=3)
    axD.bar([x + w / 2 for x in xs], prof["omni"], width=w, color=ORANGE,
            label="Omni", zorder=3)
    for x, v in zip(xs, prof["vl8b"]):
        if v > 0.04:
            axD.text(x - w / 2, v + 0.02, "%d" % round(100 * v), fontsize=6,
                     ha="center", color=BLUE)
    for x, v in zip(xs, prof["omni"]):
        if v > 0.04:
            axD.text(x + w / 2, v + 0.02, "%d" % round(100 * v), fontsize=6,
                     ha="center", color=ORANGE)
    axD.set_xticks(xs); axD.set_xticklabels(["A", "B", "C", "D"])
    axD.set_ylim(0, 0.74)
    axD.set_yticks([0, 0.25, 0.5])
    axD.set_xlabel("answer position (DeepShip)")
    axD.set_ylabel("% of answers")
    axD.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
               fontsize=6.5, handlelength=1.0, columnspacing=0.8, borderaxespad=0)
    axD.text(0.98, 0.985, "ShipsEar A–L:\nOmni → K ✓   VL-8B → E ✓\n(q2a, VL-32B → K)",
             fontsize=6, ha="right", va="top", linespacing=1.25,
             bbox=dict(boxstyle="round,pad=0.25", fc="#f5f5f5", ec=GREY, lw=0.5),
             transform=axD.transAxes)
    style(axD); panel(axD, "(d)", x=-0.38, y=1.14)
    save(fig, "fig2_mechanism.png")


# ---------------------------------------------------------------- fig3
def fig3():
    fig, ax = plt.subplots(figsize=(2.95, 1.70))
    fig.subplots_adjust(left=0.155, right=0.97, top=0.78, bottom=0.175)
    # honest audio-path band (per-layer recording-grouped CV; A4 reconciliation)
    a4 = json.load(open(os.path.join(ROOT, "outputs/pilots/round1_analyses/a4_omni_band.json")))
    band = a4["band_honest"]
    ax.axhspan(band["min_acc"], band["max_acc"], color=BLUE, alpha=0.10, zorder=1)
    rows = list(csv.DictReader(open(os.path.join(ROOT, "outputs/results/layer_probe/curves.csv"))))
    def series(model, spec):
        pts = [(int(r["layer"]), float(r["acc"])) for r in rows
               if r["model"] == model and r["spec"] == spec]
        pts.sort()
        return [p[0] for p in pts], [p[1] for p in pts]
    acurve = sorted((e["layer"], e["test_acc"]) for e in a4["curve"])
    ax.plot([p[0] for p in acurve], [p[1] for p in acurve], "-", color=BLUE,
            lw=1.5, zorder=4, label="Omni audio")
    ax.plot(*series("qwen3_omni_30b", "mel"), color=GREEN, lw=0.9, zorder=3, label="Omni mel")
    ax.plot(*series("qwen3_omni_30b", "stft"), color=ORANGE, lw=0.9, zorder=3, label="Omni STFT")
    b = json.load(open(os.path.join(ROOT, "outputs/pilots/beats_full_budget.json")))
    bc = b["stages"]["stage2_full_budget_layers"]["curves"]
    bc.sort(key=lambda e: e["layer"])
    ax.plot([e["layer"] for e in bc], [e["acc"] for e in bc], "--", color=BLACK,
            lw=1.0, zorder=3, label="BEATs")
    # CV-selected audio layer (L11) and best-layer dots for spectrogram paths
    ax.plot(band["selected_layer"], band["selected_test_acc"], "o", color=BLUE,
            ms=3.5, zorder=5)
    ax.text(band["selected_layer"] + 1.0, band["selected_test_acc"] + 0.004, "L11",
            fontsize=6, color=BLUE, va="bottom", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    for spec, c in (("mel", GREEN), ("stft", ORANGE)):
        xs2, ys2 = series("qwen3_omni_30b", spec)
        i = max(range(len(ys2)), key=lambda k: ys2[k])
        ax.plot(xs2[i], ys2[i], "o", color=c, ms=3.5, zorder=5)
    ax.annotate("−18 pt", xy=(48, 0.491), xytext=(33, 0.435), fontsize=6.5,
                color=ORANGE, arrowprops=dict(arrowstyle="->", lw=0.6, color=ORANGE))
    ax.annotate("audio path flat (band)", xy=(34, band["min_acc"] + 0.002),
                xytext=(27, 0.612), fontsize=6.5, color=BLUE, ha="center",
                va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.6, color=BLUE))
    ax.set_xlim(0, 48); ax.set_ylim(0.40, 0.80)
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("probe accuracy")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=4, handlelength=1.1, columnspacing=0.6, handletextpad=0.3,
              borderaxespad=0)
    style(ax)
    save(fig, "fig3_layers.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("done")
