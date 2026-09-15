#!/usr/bin/env python
"""Regenerate fig4_learning.png from the HONEST learning curve
(outputs/pilots/honest_learning_curve.json: layer+C chosen per budget by
train-only recording-grouped CV). Replaces the legacy-curve rendering."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "outputs/pilots/honest_learning_curve.json"
OUTS = ["docs/figures/paper/fig4_learning.png", "outputs/figures/paper/fig4_learning.png"]

d = json.load(open(SRC))
# structure: {"omni_aud": {budget: {layer, C, cv, test_acc, test_f1}}, "beats": {...}}
def series(key):
    b = d[key]["budgets"]
    xs, ys = [], []
    for n in sorted(int(k) for k in b):
        v = b.get(str(n), b.get(n))
        xs.append(n); ys.append(v["test_acc"])
    return xs, ys

x1, y1 = series("omni_aud")
x2, y2 = series("beats")
print("omni:", list(zip(x1, ["%.4f" % v for v in y1])))
print("beats:", list(zip(x2, ["%.4f" % v for v in y2])))

BLUE, VERM = "#0072B2", "#D55E00"
fig, ax = plt.subplots(figsize=(3.45, 1.95), dpi=300)
ax.plot(x1, y1, "-o", color=BLUE, ms=3.5, lw=1.4, label="Omni-audio (frozen probe)")
ax.plot(x2, y2, "--s", color=VERM, ms=3.5, lw=1.4, label="BEATs (frozen probe)")
ax.axhline(0.284, color="grey", ls=":", lw=1.0)
ax.text(120, 0.30, "best generative condition (0.284)", fontsize=7.5, color="grey")
ax.annotate("0.631 = 2.6x same model's\ngenerative acc. (0.247)",
            xy=(100, y1[0]), xytext=(180, 0.545), fontsize=7.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
ax.annotate("500 labels -> 98% of full", xy=(500, y1[1]), xytext=(900, 0.63),
            fontsize=7.5, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
ax.set_xscale("log")
ax.set_xticks([100, 500, 1000, 2000, 3000, 8350])
ax.set_xticklabels(["100", "500", "1k", "2k", "3k", "8350"])
ax.set_xlabel("labeled training clips (layer + C by train-only CV)")
ax.set_ylabel("test clip accuracy")
ax.set_ylim(0.42, 0.80)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=7.5, loc="lower right", frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout(pad=0.4)
for out in OUTS:
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)
