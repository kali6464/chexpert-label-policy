"""Figure 4: test-set ROC curves for the five pathologies, one curve per backbone
(three-seed ensembles under the Atelectasis + Edema policy).

  python fig4_roc.py
"""
from sklearn.metrics import roc_auc_score, roc_curve

from common import ALL_LABELS, COMPETITION_LABELS
from fig_common import BACKBONES, GRID, MUTED, Line2D, load_ensemble, plt, save

POLICY = "Atelectasis + Edema"


def main():
    data = {b: load_ensemble(b, POLICY) for b in BACKBONES}
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6))
    axes = axes.ravel()
    for k, label in enumerate(COMPETITION_LABELS):
        ax, j = axes[k], ALL_LABELS.index(label)
        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        lines = []
        for b, (_, color, _) in BACKBONES.items():
            _, yt, yp = data[b]
            keep = yt[:, j] != -1
            fpr, tpr, _ = roc_curve(yt[keep, j], yp[keep, j])
            auc = roc_auc_score(yt[keep, j], yp[keep, j])
            ax.plot(fpr, tpr, color=color, lw=1.6, zorder=3)
            lines.append(Line2D([], [], color=color, lw=1.6, label=f"{b.split('-')[0]}  {auc:.3f}"))
        n_pos = int((data[next(iter(BACKBONES))][1][:, j] == 1).sum())
        ax.set_title(f"{label} ({n_pos} positive studies)", fontsize=9, color="#1f1f1e", loc="left")
        ax.legend(handles=lines, title="AUC", frameon=False, fontsize=7.5, title_fontsize=7.5,
                  loc="lower right", alignment="left")
        ax.set_xlim(-0.01, 1.0); ax.set_ylim(0.0, 1.01); ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        if k >= 2:
            ax.set_xlabel("False positive rate")
        if k % 3 == 0:
            ax.set_ylabel("True positive rate")
    ax = axes[5]
    ax.axis("off")
    handles = [Line2D([], [], color=c, lw=1.6, label=n) for n, (_, c, _) in BACKBONES.items()]
    handles.append(Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="Chance"))
    ax.legend(handles=handles, title="Three-seed ensembles,\nAtelectasis + Edema policy", frameon=False,
              fontsize=8.5, title_fontsize=8.5, loc="center left", alignment="left")
    fig.tight_layout()
    save(fig, "fig4_roc")


if __name__ == "__main__":
    main()
