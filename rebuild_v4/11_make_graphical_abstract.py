"""Draw the v4 graphical abstract from verified manuscript claims."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures_v4"
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(13.4, 5.35), dpi=100)
fig.patch.set_facecolor("#f7f9fb")
ax.set(xlim=(0, 13.4), ylim=(0, 5.35))
ax.axis("off")

ax.text(0.48, 4.88, "Battery lifetime transfer depends on data definition",
        fontsize=21, fontweight="bold", color="#18364d", va="center")
ax.text(0.48, 4.45, "Retrospective six-source audit · cycle-100 landmark · remaining-life prediction",
        fontsize=11.5, color="#526b7b", va="center")

cards = [
    (0.45, "1  Verify the cohort", "#e6f1f8", "#3475a7",
     ["6 public sources", "429 verified cells", "376 eligible at cycle 100", "228 observed EOL events"]),
    (4.65, "2  Compare models", "#eaf4ef", "#338365",
     ["5 source-held-out methods", "Different winners by source", "NASA: one scoring horizon", "Integrated score unavailable"]),
    (8.85, "3  Test target adaptation", "#f9f0e8", "#b36a36",
     ["20% target-cell calibration", "No general Brier improvement", "Only CALCE mean improves", "Just 2 calibration cells"]),
]
for x, title, face, edge, lines in cards:
    ax.add_patch(FancyBboxPatch((x, 1.25), 3.78, 2.92,
                               boxstyle="round,pad=0.03,rounding_size=0.16",
                               facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(x + 0.24, 3.75, title, fontsize=13.2, fontweight="bold", color=edge)
    for j, line in enumerate(lines):
        ax.text(x + 0.24, 3.23 - j * 0.49, line, fontsize=11.1, color="#233f51")

for start in (4.25, 8.45):
    ax.add_patch(FancyArrowPatch((start, 2.69), (start + 0.32, 2.69),
                                 arrowstyle="-|>", mutation_scale=18, linewidth=2,
                                 color="#718896"))

ax.add_patch(FancyBboxPatch((0.45, 0.28), 12.18, 0.65,
                           boxstyle="round,pad=0.02,rounding_size=0.08",
                           facecolor="#18364d", edgecolor="none"))
ax.text(0.68, 0.60,
        "Conclusion: model rankings and endpoint choices vary; no universal transfer or validated dispatch benefit",
        fontsize=10.8, color="white", va="center")

fig.savefig(OUT / "graphical_abstract_v4.pdf", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
fig.savefig(OUT / "graphical_abstract_v4.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print(OUT / "graphical_abstract_v4.pdf")
