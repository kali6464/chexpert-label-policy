"""Figure 2: effect of the Atelectasis + Edema policy on each pathology (vs U-Zeros, test set).

  python fig2_per_pathology.py        # run fig1_comparisons.py first; this reads figures/comparisons.json
"""
import os

from common import COMPETITION_LABELS
from fig_common import BACKBONES, GRID, INK, OUT, backbone_legend, interval_point, load_json, plt, save

CHANGED = ("Atelectasis", "Edema")


def main():
    path = os.path.join(OUT, "comparisons.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found. Run fig1_comparisons.py first.")
    rows = {r["backbone"]: r["diff"] for r in load_json(path)["policy"]
            if r["a"] == "Atelectasis + Edema" and r["b"] == "U-Zeros"}

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.spines["left"].set_visible(False)
    xs = []
    for i, label in enumerate(COMPETITION_LABELS):
        for j, b in enumerate(BACKBONES):
            d, lo, hi = rows[b][label]
            _, color, marker = BACKBONES[b]
            interval_point(ax, -i + (1 - j) * 0.24, d, lo, hi, color, marker)
            xs += [lo, hi]
    pad = 0.06 * (max(xs) - min(xs))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.set_yticks([-i for i in range(len(COMPETITION_LABELS))])
    ax.set_yticklabels([n + "\n(rule changed)" if n in CHANGED else n for n in COMPETITION_LABELS])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-len(COMPETITION_LABELS) + 0.4, 0.6)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Difference in AUC, Atelectasis + Edema minus U-Zeros (test set, 95% interval)")
    backbone_legend(ax, loc="lower right")
    save(fig, "fig2_per_pathology")


if __name__ == "__main__":
    main()
