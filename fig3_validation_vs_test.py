"""Figure 3: validation against test mean AUC for every finished run.

  python fig3_validation_vs_test.py
"""
import os

import numpy as np

from fig_common import BACKBONES, GRID, MUTED, OUT, POLICIES, SEEDS, Line2D, load_json, plt, run_dir


def main():
    pts = []
    for b in BACKBONES:
        for p in POLICIES:
            for s in SEEDS:
                d = run_dir(b, p, s)
                v, t = os.path.join(d, "results.json"), os.path.join(d, "test_results.json")
                if os.path.exists(v) and os.path.exists(t):
                    pts.append((b, p, load_json(v)["official_val_study"]["mean_competition"],
                                load_json(t)["test_study"]["mean_competition"]))
    if not pts:
        raise SystemExit("No runs with both results.json and test_results.json were found under runs/.")
    val = np.array([p[2] for p in pts]); test = np.array([p[3] for p in pts])
    lo = min(val.min(), test.min()) - 0.002
    hi = max(val.max(), test.max()) + 0.002

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1, label="validation = test")
    for b, p, v, t in pts:
        ax.scatter(v, t, s=46, marker=BACKBONES[b][2], color=POLICIES[p][1], edgecolor="white", linewidth=1.0, zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlabel("Mean AUC, validation set (200 studies)"); ax.set_ylabel("Mean AUC, test set (500 studies)")
    used = [p for p in POLICIES if any(q[1] == p for q in pts)]
    h1 = [Line2D([], [], color=POLICIES[p][1], marker="o", ls="", ms=6, label=p) for p in used]
    h2 = [Line2D([], [], color="#55544e", marker=m, ls="", ms=6, label=n) for n, (_, _, m) in BACKBONES.items()]
    h2.append(Line2D([], [], color=MUTED, lw=0.9, ls=(0, (4, 3)), label="validation = test"))
    kw = dict(frameon=False, fontsize=7.5, title_fontsize=7.5, alignment="left")
    l1 = ax.legend(handles=h1, title="Policy (colour)", loc="upper left", bbox_to_anchor=(1.02, 1.0), **kw)
    l2 = ax.legend(handles=h2, title="Backbone (shape)", loc="upper left", bbox_to_anchor=(1.02, 0.55), **kw)
    ax.add_artist(l1)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):  # the legends sit outside the axes; pass them so they are not cropped
        fig.savefig(os.path.join(OUT, f"fig3_validation_vs_test.{ext}"), bbox_extra_artists=(l1, l2))
    plt.close(fig)
    print(f"saved {OUT}/fig3_validation_vs_test.png and .pdf")

    r = np.corrcoef(val, test)[0, 1]
    rho = np.corrcoef(np.argsort(np.argsort(val)), np.argsort(np.argsort(test)))[0, 1]
    print(f"{len(pts)} runs: Pearson r = {r:.2f}, Spearman rho = {rho:.2f}")


if __name__ == "__main__":
    main()
