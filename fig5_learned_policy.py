"""Figure 5 (for Section 5): how the learned target alpha moved during training, seed 42.

  python fig5_learned_policy.py
"""
import json
import os

from common import COMPETITION_LABELS
from fig_common import BACKBONES, GRID, MUTED, OUT, Line2D, plt, run_dir


def main():
    fig, axes = plt.subplots(1, 5, figsize=(10.0, 2.5), sharey=True)
    found = False
    for b, (_, color, marker) in BACKBONES.items():
        path = os.path.join(run_dir(b, "Learned", 42), "log.jsonl")
        if not os.path.exists(path):
            print(f"skipping {b}: {path} not found")
            continue
        found = True
        rows = [json.loads(line) for line in open(path) if line.strip()]
        steps = [r["step"] / 1000 for r in rows]
        for ax, label in zip(axes, COMPETITION_LABELS):
            alpha = [r["policy_alpha_beta"][label][0] for r in rows]
            ax.plot([0] + steps, [0.5] + alpha, color=color, lw=1.6, marker=marker, ms=3.5, markevery=3,
                    mec="white", mew=0.6)
    if not found:
        raise SystemExit("No learned-policy runs found under runs/.")
    for ax, label in zip(axes, COMPETITION_LABELS):
        ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_title(label, fontsize=9, color="#1f1f1e", loc="left")
        ax.set_ylim(-0.03, 1.03); ax.set_yticks([0, 0.5, 1])
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.set_xlabel("Training step (thousands)")
    axes[0].set_ylabel("Learned target alpha\n(0 = negative, 1 = positive)")
    fig.tight_layout()
    handles = [Line2D([], [], color=c, marker=m, lw=1.6, ms=5, label=n) for n, (_, c, m) in BACKBONES.items()]
    handles.append(Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="Starting value"))
    leg = fig.legend(handles=handles, frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):  # the legend sits above the panels; pass it so it is not cropped
        fig.savefig(os.path.join(OUT, f"fig5_learned_policy.{ext}"), bbox_extra_artists=(leg,))
    plt.close(fig)
    print(f"saved {OUT}/fig5_learned_policy.png and .pdf")


if __name__ == "__main__":
    main()
