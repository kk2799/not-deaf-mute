#!/usr/bin/env python
"""SPLASH ICASSP figures v3 — message-first redesign.

One panel, one message, stated IN the figure (bold takeaway lines).
fig1: two panels — (a) the exam (44 conditions vs chance, dose inset moved
      into (b)) — (b) the ladder as gain-over-chance bars.
fig2: four panels — (c) redesigned from 9 rotated-label log bars + twin axis
      to a sorted horizontal dot plot (horizontal labels, no twin axis).
fig3: unchanged content from v2 (already single-message).
All values from real experiment outputs. Print size = design size
(fig1/2 at 0.83\\textwidth ~ 5.94in, fig3 at 0.85\\columnwidth ~ 2.95in).
Run in container: python scripts/paper_figures_v3.py
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
    # render once, write bytes to both dirs (double savefig can desync the
    # tight-bbox expansion and clip edge text, e.g. the rightmost xlabel)
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = buf.getvalue()
    for d in OUT:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, name), "wb") as f:
            f.write(data)
    plt.close(fig)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel(ax, s, x=-0.28, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


def panel_below(ax, s, y=-0.28):
    ax.text(0.5, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="center")


def headline(ax, s, x=0.02, y=0.97):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7,
            fontweight="bold", va="top", ha="left", color=BLACK,
            path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


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

    fig = plt.figure(figsize=(5.95, 1.52))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.30,
                          left=0.085, right=0.99, top=0.90, bottom=0.24)

    # (a) the exam: 44 conditions
    axA = fig.add_subplot(gs[0])
    offs = [0, .13, -.13, .065, -.065, .195, -.195]
    best = (0.0, -1)
    for yi, f in enumerate(fams):
        ds, ss = by[f]["deepship"], by[f]["shipsear"]
        dys = [yi + offs[i % 7] for i in range(len(ds))]
        axA.scatter(ds, dys, s=15, c=fc[f], zorder=3, edgecolors="none")
        if ds:
            i = max(range(len(ds)), key=lambda k: ds[k])
            if ds[i] > best[0]:
                best = (ds[i], dys[i])
        axA.scatter(ss, [yi + offs[i % 7] for i in range(len(ss))], s=15,
                    facecolors="none", edgecolors=GREY, linewidths=0.9, zorder=3)
    axA.axvline(0.25, color=BLACK, lw=0.9, zorder=2)
    axA.text(0.25, 3.52, "chance 1/4", fontsize=6.5, ha="center", va="top",
             color=BLACK, zorder=6, bbox=dict(fc="white", ec="none", pad=0.8))
    axA.axvline(1/12, color=GREY, lw=0.9, ls=":", zorder=2)
    axA.text(1/12, 3.52, "chance 1/12", fontsize=6.5, ha="center", va="top",
             color=GREY, zorder=6, bbox=dict(fc="white", ec="none", pad=0.8))
    axA.set_xlim(0, 0.33); axA.set_ylim(-0.5, 3.55)
    axA.set_xticks([0, 0.1, 0.2, 0.3])
    axA.set_yticks(range(4)); axA.set_yticklabels([fname[f] for f in fams])
    axA.set_xlabel("clip accuracy (44 conditions)")
    axA.grid(axis="y", visible=False)
    style(axA); panel_below(axA, "(a)", y=-0.30)

    # (b) the ladder: gain over chance, bars
    axB = fig.add_subplot(gs[1])
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
    CH = 0.25
    for y, (lab, val, c, mk) in zip(ys, rungs):
        if mk is None:
            lo, hi = val
            axB.barh(y, lo - CH, left=CH, height=0.42, color=c, alpha=0.30, zorder=2)
            axB.barh(y, hi - lo, left=lo, height=0.42, color=c, alpha=0.95, zorder=3)
            vtxt = "%.2f–%.2f" % (lo, hi)
            x = hi
        else:
            axB.barh(y, val - CH, left=CH, height=0.42, color=c, alpha=0.85, zorder=3)
            vtxt = "%.3f" % val
            x = val
        axB.text(x + 0.010, y, vtxt, fontsize=6.5, va="center", color=c)
    axB.axvline(CH, color=BLACK, lw=1.0, zorder=4)
    axB.text(CH + 0.004, 6.42, "chance", fontsize=6.5, color=BLACK, ha="left")
    axB.set_xlim(CH, 0.88); axB.set_ylim(-0.7, 6.9)
    axB.set_xticks([0.25, 0.4, 0.6, 0.8])
    axB.set_yticks(ys); axB.set_yticklabels([r[0] for r in rungs])
    axB.set_xlabel("clip accuracy (bars start at 0.25)")
    axB.grid(axis="y", visible=False)
    lhand = [plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.85),
             plt.Rectangle((0, 0), 1, 1, color=VERM, alpha=0.85)]
    axB.legend(lhand, ["LLM-derived", "BEATs"], frameon=False,
               loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=6,
               handlelength=1.0, borderaxespad=0)
    style(axB); panel_below(axB, "(b)", y=-0.30)
    save(fig, "fig1_ladder.png")


# ---------------------------------------------------------------- fig2
def fig2():
    fig = plt.figure(figsize=(7.0, 1.74))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.14, 1.0, 1.02, 1.0], wspace=0.75,
                          left=0.060, right=0.995, top=0.86, bottom=0.30)

    # (a) collapse distributions
    axA = fig.add_subplot(gs[0])
    conds = [
        ("Q2A audio", "lalm_prompting_qwen2_audio_deepship_zero_shot_shot0_clip30.0_langen_enrichnone"),
        ("VL-8B mel", "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone"),
        ("VL-8B demon", "vlm_spectrogram_qwen3_vl_8b_deepship_zero_shot_shot0_clip30.0_specdemon_langen_enrichnone"),
        ("VL-32B mel", "vlm_spectrogram_qwen3_vl_32b_deepship_zero_shot_shot0_clip30_specmel_langen_enrichnone"),
        ("Omni audio", "lalm_prompting_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_langen_enrichnone"),
        ("Omni mel", "vlm_spectrogram_qwen3_omni_30b_deepship_zero_shot_shot0_clip30.0_specmel_langen_enrichnone"),
    ]
    cls_names = ["Cargo", "Passenger", "Oil tanker", "Tug"]
    for xi, (_, fn) in enumerate(conds):
        d = json.load(open(os.path.join(ROOT, "outputs/results", fn + ".json")))
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
            bot += frac
    axA.set_xticks(range(6))
    axA.set_xticklabels([c[0] for c in conds], fontsize=6, rotation=32, ha="right")
    axA.set_ylim(0, 1.02); axA.set_xlim(-0.55, 5.55)
    axA.set_ylabel("fraction of predictions")
    axA.set_yticks([0, .5, 1])
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLS_COLORS[c]) for c in cls_names]
    axA.legend(handles, cls_names, ncol=2, loc="lower left", bbox_to_anchor=(-0.02, 1.04),
               frameon=False, fontsize=6, handlelength=1.0, columnspacing=0.7,
               handletextpad=0.4, borderaxespad=0)
    style(axA); panel_below(axA, "(a)", y=-0.40)

    # (b) permutation
    axB = fig.add_subplot(gs[1])
    oo = json.load(open(os.path.join(ROOT, "outputs/results/option_order/SUMMARY.json")))
    styles = {"vl8b.mel": (BLUE, "-"), "vl8b.demon": (BLUE, "--"),
              "omni.audio": (ORANGE, "-"), "omni.mel": (ORANGE, "--")}
    shortb = {"vl8b.mel": "8B mel", "vl8b.demon": "8B dem.",
              "omni.audio": "Omni aud.", "omni.mel": "Omni mel"}
    for e in oo:
        c, ls = styles[e["key"]]
        axB.plot(range(1, 9), e["perm_accs"], ls + "o", color=c, lw=1.1, ms=2.4,
                 zorder=3, label=shortb[e["key"]])
    axB.axhline(0.25, color=BLACK, lw=0.8, ls=":", zorder=2)
    axB.text(8.0, 0.332, "chance", fontsize=6.5, ha="right", va="center",
             color=BLACK)
    axB.set_xlim(0.5, 8.5); axB.set_ylim(0.15, 0.36)
    axB.set_xticks(range(1, 9))
    axB.set_xlabel("option permutation")
    axB.set_ylabel("clip accuracy")
    axB.set_yticks([0.15, 0.25, 0.35])
    axB.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
               fontsize=6, handlelength=1.2, columnspacing=0.8, borderaxespad=0)
    style(axB); panel_below(axB, "(b)", y=-0.40)

    # (c) null input — horizontal dot plot, sorted by KL
    axC = fig.add_subplot(gs[2])
    ni = json.load(open(os.path.join(ROOT, "outputs/results/null_input/SUMMARY.json")))
    famc = {"q2a": PINK, "vl8b": BLUE, "oma": ORANGE, "omm": GREEN}
    famlab = {"q2a": "q2a audio", "vl8b": "vl8b mel",
              "oma": "om audio", "omm": "om-mel"}
    def fam(arm):
        if arm.startswith("q2a"): return "q2a"
        if arm.startswith("vl8b"): return "vl8b"
        if arm.startswith("omni.audio"): return "oma"
        return "omm"
    def armlab(arm):
        return (arm.split(".", 2)[2].replace("pixnoise", "pixel")
                .replace("nomedia", "no media").replace("silence", "silence"))
    ni_sorted = sorted(ni, key=lambda e: e["KL(real||null)"], reverse=True)
    ys = list(range(len(ni_sorted)))[::-1]
    for y, e in zip(ys, ni_sorted):
        c = famc[fam(e["arm"])]
        kl = e["KL(real||null)"]
        if e["arm"] == "omni.audio.nomedia":
            axC.plot(kl, y, "o", color=VERM, ms=5.5, zorder=5,
                     markeredgecolor="black", markeredgewidth=0.7)
        else:
            axC.plot(kl, y, "o", color=c, ms=4.5, zorder=4)
    axC.set_xscale("log"); axC.set_xlim(5e-4, 20)
    axC.set_xticks([1e-3, 1e-1, 1, 10])
    axC.set_xticklabels(["0.001", "0.1", "1", "10"], fontsize=6)
    axC.set_yticks(ys)
    ylab = {"q2a.audio.silence": "q2a silence", "q2a.audio.noise": "q2a noise",
            "vl8b.mel.grey": "vl8b grey", "vl8b.mel.pixnoise": "vl8b pixel",
            "omni.audio.silence": "om silence", "omni.audio.noise": "om noise",
            "omni.audio.nomedia": "om no media", "omni.mel.grey": "om-mel grey",
            "omni.mel.pixnoise": "om-mel pixel"}
    axC.set_yticklabels([ylab[e["arm"]] for e in ni_sorted], fontsize=6)
    axC.set_xlabel("KL(real‖null)")
    axC.grid(axis="y", visible=False)
    lhand = [plt.Line2D([], [], color=famc[k], marker="o", ls="", ms=4)
             for k in ("q2a", "vl8b", "oma", "omm")]
    lhand.append(plt.Line2D([], [], color=VERM, marker="o", ls="", ms=4,
                            markeredgecolor="black"))
    axC.legend(lhand, [famlab[k] for k in ("q2a", "vl8b", "oma", "omm")] +
               ["om no media"], frameon=False, loc="lower left",
               bbox_to_anchor=(0.0, 1.02), ncol=2,
               fontsize=6, handlelength=0.9, columnspacing=0.7, borderaxespad=0)
    style(axC); panel_below(axC, "(c)", y=-0.40)

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
    axD.set_xlabel("answer position")
    axD.set_ylabel("% of answers")
    axD.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2,
               fontsize=6.5, handlelength=1.0, columnspacing=0.8, borderaxespad=0)
    style(axD); panel_below(axD, "(d)", y=-0.40)
    save(fig, "fig2_mechanism.png")


# ---------------------------------------------------------------- fig3
def fig3():
    fig, ax = plt.subplots(figsize=(2.95, 1.46))
    fig.subplots_adjust(left=0.155, right=0.97, top=0.78, bottom=0.175)
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
    ax.plot(band["selected_layer"], band["selected_test_acc"], "o", color=BLUE,
            ms=3.5, zorder=5)
    ax.text(band["selected_layer"] + 1.0, band["selected_test_acc"] + 0.004, "L11",
            fontsize=6, color=BLUE, va="bottom", zorder=6,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
    for spec, c in (("mel", GREEN), ("stft", ORANGE)):
        xs2, ys2 = series("qwen3_omni_30b", spec)
        i = max(range(len(ys2)), key=lambda k: ys2[k])
        if spec == "stft":
            ax.plot(xs2[i], ys2[i], "o", color=c, ms=3.5, zorder=5)
    ax.annotate("−18 pt", xy=(48, 0.491), xytext=(33, 0.435), fontsize=6.5,
                color=ORANGE, arrowprops=dict(arrowstyle="->", lw=0.6, color=ORANGE))
    ax.annotate("audio path flat (0.663–0.712)", xy=(44, 0.7095),
                xytext=(37, 0.722), fontsize=6, color=BLUE, ha="center",
                va="bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
                arrowprops=dict(arrowstyle="-", lw=0.6, color=BLUE))
    ax.set_xlim(0, 48); ax.set_ylim(0.40, 0.80)
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("probe accuracy")
    ax.legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.01),
              ncol=4, handlelength=1.1, columnspacing=0.6, handletextpad=0.3,
              borderaxespad=0)
    style(ax)
    save(fig, "fig3_layers.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
    print("done")
