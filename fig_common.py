"""Shared helpers for the manuscript figures. Run every figure script from the
CheXpert_pipeline folder (the one that contains runs/, common.py and evaluate.py)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

RUNS = "runs"
OUT = "figures"
SEEDS = [42, 43, 44]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"

# display name -> (run-name prefix, colour, marker)
BACKBONES = {
    "DenseNet-121": ("densenet121", "#2a78d6", "o"),
    "DeiT-Base": ("deit_base_distilled_patch16_384", "#eb6834", "s"),
    "ConvNeXt-Base": ("convnext_base", "#1baf7a", "^"),
}
# display name -> (run-name tag, colour)
POLICIES = {
    "U-Zeros": ("zeros", "#2a78d6"),
    "Atelectasis": ("Atelectasis", "#eb6834"),
    "Atelectasis + Edema": ("Atelectasis+Edema", "#1baf7a"),
    "U-Ones": ("ones", "#eda100"),
    "Learned": ("learned", "#e87ba4"),
}

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def run_dir(backbone, policy, seed):
    return os.path.join(RUNS, f"{BACKBONES[backbone][0]}__{POLICIES[policy][0]}__s{seed}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_ensemble(backbone, policy, seeds=SEEDS, fname="test_predictions.npz"):
    """Average the per-study probabilities of several seeds.
    Returns (study_ids, y_true [N,14], y_prob [N,14]), sorted by study id."""
    ids0 = yt0 = None
    probs = []
    for s in seeds:
        path = os.path.join(run_dir(backbone, policy, s), fname)
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run score_test.py first, from the CheXpert_pipeline folder.")
        d = np.load(path)
        order = np.argsort(d["study_ids"])
        ids, yt, yp = d["study_ids"][order], d["y_true_study"][order], d["y_prob_study"][order]
        if ids0 is None:
            ids0, yt0 = ids, yt
        elif not (np.array_equal(ids, ids0) and np.array_equal(yt, yt0)):
            raise SystemExit(f"{path} was scored on different studies from the other seeds.")
        probs.append(yp)
    return ids0, yt0, np.mean(probs, axis=0)


def backbone_legend(ax, **kw):
    handles = [Line2D([], [], color=c, marker=m, ls="", ms=6, label=n) for n, (_, c, m) in BACKBONES.items()]
    return ax.legend(handles=handles, frameon=False, fontsize=8, **kw)


def interval_point(ax, y, d, lo, hi, color, marker):
    ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round", zorder=2)
    ax.plot(d, y, marker=marker, color=color, ms=6.5, mec="white", mew=1.0, zorder=3)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {OUT}/{name}.png and {OUT}/{name}.pdf")
