"""Figure 1: the twelve pre-specified test-set comparisons (difference in mean AUC, 95% interval).

  python fig1_comparisons.py

Recomputes the paired bootstrap from the saved test predictions (takes a few minutes)
and stores every number in figures/comparisons.json, which fig2_per_pathology.py reuses.
"""
import json
import os

import numpy as np

from common import COMPETITION_LABELS
from evaluate import bootstrap_samples
from fig_common import BACKBONES, GRID, INK, MUTED, OUT, backbone_legend, interval_point, load_ensemble, plt, save

N_BOOT, SEED = 5000, 0
NAMES = COMPETITION_LABELS + ["mean_competition"]
B = list(BACKBONES)
POLICY_FAMILIES = [
    ("Atelectasis + Edema\nvs U-Zeros (primary)", "Atelectasis + Edema", "U-Zeros"),
    ("Atelectasis\nvs U-Zeros", "Atelectasis", "U-Zeros"),
    ("Atelectasis + Edema\nvs Atelectasis", "Atelectasis + Edema", "Atelectasis"),
]
BACKBONE_PAIRS = [("ConvNeXt-Base", "DenseNet-121"), ("DeiT-Base", "DenseNet-121"), ("DeiT-Base", "ConvNeXt-Base")]


def compare(side_a, side_b):
    """Paired bootstrap A - B. Each side is (backbone, policy). Returns {label: [diff, lo, hi]}."""
    ids_a, yt_a, yp_a = load_ensemble(*side_a)
    ids_b, yt_b, yp_b = load_ensemble(*side_b)
    if not (np.array_equal(ids_a, ids_b) and np.array_equal(yt_a, yt_b)):
        raise SystemExit("The two sides were not scored on the same studies.")
    s = bootstrap_samples(yt_a, [yp_a, yp_b], N_BOOT, SEED)
    diff = s[:, 0, :] - s[:, 1, :]
    out = {}
    for i, name in enumerate(NAMES):
        d = diff[:, i][~np.isnan(diff[:, i])]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = [float(d.mean()), float(lo), float(hi)]
    return out


def main():
    results = {"policy": [], "backbone": []}
    for label, pol_a, pol_b in POLICY_FAMILIES:
        for b in B:
            print(f"bootstrapping: {b}, {pol_a} vs {pol_b}", flush=True)
            results["policy"].append({"family": label, "backbone": b, "a": pol_a, "b": pol_b,
                                      "diff": compare((b, pol_a), (b, pol_b))})
    for a, b in BACKBONE_PAIRS:
        print(f"bootstrapping: {a} vs {b}", flush=True)
        results["backbone"].append({"a": a, "b": b,
                                    "diff": compare((a, "Atelectasis + Edema"), (b, "Atelectasis + Edema"))})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "comparisons.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.spines["left"].set_visible(False)
    y, ticks, labels, xs = 0.0, [], [], []
    for label, _, _ in POLICY_FAMILIES:
        ys = []
        for row in [r for r in results["policy"] if r["family"] == label]:
            d, lo, hi = row["diff"]["mean_competition"]
            _, color, marker = BACKBONES[row["backbone"]]
            interval_point(ax, y, d, lo, hi, color, marker)
            ys.append(y); xs += [lo, hi]; y -= 1
        ticks.append(float(np.mean(ys))); labels.append(label); y -= 0.8
    header_y = y + 0.55
    y -= 0.35
    for row in results["backbone"]:
        d, lo, hi = row["diff"]["mean_competition"]
        interval_point(ax, y, d, lo, hi, "#55544e", "D")
        ticks.append(y); labels.append(f"{row['a'].split('-')[0]} − {row['b'].split('-')[0]}")
        xs += [lo, hi]; y -= 1
    pad = 0.08 * (max(xs) - min(xs))
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.text(min(xs) - pad, header_y, "Backbone vs backbone (Atelectasis + Edema policy)", fontsize=8, color=MUTED,
            ha="left", va="center", zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.axvline(0, color=INK, lw=0.9, zorder=1)
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8.5); ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.4, 0.8)
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Difference in mean AUC over five pathologies (test set, 95% interval)")
    backbone_legend(ax, loc="lower right")
    save(fig, "fig1_comparisons")

    print("\nMean-AUC differences [95% interval]:")
    for r in results["policy"]:
        d, lo, hi = r["diff"]["mean_competition"]
        print(f"  {r['backbone']:14s} {r['a']} vs {r['b']}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for r in results["backbone"]:
        d, lo, hi = r["diff"]["mean_competition"]
        print(f"  {r['a']} minus {r['b']}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
