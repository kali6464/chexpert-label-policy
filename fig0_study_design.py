"""Study design schematic (Figure 1). No data needed:  python fig0_study_design.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED = "#1f1f1e", "#6b6a63"
REPORT, EXPERT, STEP, OUTC = "#eeede8", "#dbe9fb", "#ffffff", "#e3f4ec"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(8.0, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 105); ax.axis("off")


def box(x, y, w, h, title, body, fill, edge=MUTED):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fill, ec=edge, lw=0.9))
    ax.text(x + 2, y + h - 3.6, title, fontsize=8.5, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(x + 2, y + h - 8.0, body, fontsize=7.0, color=INK, va="top", ha="left", linespacing=1.4)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0, shrinkA=0, shrinkB=0))


H, W1, W2, W3 = 19, 30, 32, 28
X1, X2, X3 = 0, 35, 72
ROWS = [79, 56, 33, 3]  # bottom y of each row

ax.text(X1, 100.5, "Data", fontsize=9, color=MUTED, va="bottom")
ax.text(X2, 100.5, "Used for", fontsize=9, color=MUTED, va="bottom")
ax.text(X3, 100.5, "Produces", fontsize=9, color=MUTED, va="bottom")

box(X1, ROWS[0], W1, H, "Training", "62,540 patients, 216,722 images\nReport-derived labels", REPORT)
box(X1, ROWS[1], W1, H, "Selection", "2,000 patients, 6,692 images\nReport-derived labels", REPORT)
box(X1, ROWS[2], W1, H, "Validation", "200 studies, 234 images\nRadiologist labels", EXPERT)
box(X1, ROWS[3], W1, H, "Test", "500 studies, 668 images\nRadiologist labels", EXPERT)

box(X2, ROWS[0], W2, H, "Fit the model", "3 backbones, one shared pipeline\n3 epochs (20,316 steps)", STEP)
box(X2, ROWS[1], W2, H, "Choose the checkpoint", "Scored every 1,000 steps\nBest checkpoint kept", STEP)
box(X2, ROWS[2], W2, H, "Develop", "Choose Atelectasis + Edema rule\nGuide the learned policy", STEP)
box(X2, ROWS[3], W2, H, "Evaluate once", "Per study, seed ensembles\nPaired bootstrap, 5,000 resamples", STEP)

top, bot = ROWS[0] + H, ROWS[2]
box(X3, bot, W3, top - bot, "33 trained runs",
    "3 backbones\nx 3 fixed policies\nx 3 seeds = 27 runs\n\nplus U-Ones,\n1 seed per backbone (3)\n\nplus learned policy,\n1 seed per backbone (3)", OUTC)
box(X3, ROWS[3], W3, H, "Reported results", "12 pre-specified comparisons\nTables 2 to 4", OUTC)

for r in ROWS:
    arrow(X1 + W1 + 0.8, r + H / 2, X2 - 0.8, r + H / 2)
for r in ROWS[:3]:
    arrow(X2 + W2 + 0.8, r + H / 2, X3 - 0.8, r + H / 2)
arrow(X2 + W2 + 0.8, ROWS[3] + H / 2, X3 - 0.8, ROWS[3] + H / 2)
arrow(X3 + W3 / 2, bot - 0.8, X3 + W3 / 2, ROWS[3] + H + 0.8)

yl = (ROWS[2] + ROWS[3] + H) / 2
ax.plot([0, 67], [yl, yl], color=INK, lw=0.9, ls=(0, (5, 3)))
ax.text(0, yl + 1.2, "Configurations and comparisons frozen before the test set is opened", fontsize=7.8,
        color=INK, va="bottom", ha="left", style="italic")

os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig0_study_design.{ext}")
print("saved figures/fig0_study_design.png and .pdf")
