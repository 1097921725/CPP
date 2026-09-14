"""Redraw Figure 1 without internal worksheet labels, preserving audited style."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "analysis/output/v4.0/cpp_rhgh_integrated_20260910"
PACKAGE = DATA / "bmc_submission_20260911/professional_dataset_naming_20260911"
OUT = PACKAGE / "figures"
UPLOAD = PACKAGE / "upload_after_author_confirmation"
OUT.mkdir(parents=True, exist_ok=True)
UPLOAD.mkdir(parents=True, exist_ok=True)

BLUE = "#24658B"; CORAL = "#CB624D"; TEAL = "#519D9A"; INK = "#233647"; GREY = "#8B97A3"; LIGHT = "#EAF0F4"
COLORS = [CORAL, BLUE, TEAL]
FORMS = ["Long-acting cartridge", "Short-acting aqueous", "Short-acting powder"]
rates = pd.read_csv(DATA / "tables/followup_rates_by_formulation.csv").set_index("formulation").loc[FORMS]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7.2,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.8, "ytick.labelsize": 7, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": .65, "pdf.fonttype": 42, "svg.fonttype": "none",
    "savefig.facecolor": "white", "axes.facecolor": "white", "figure.facecolor": "white",
    "svg.hashsalt": "cpp-professional-labels-20260911",
})

fig = plt.figure(figsize=(170 / 25.4, 125 / 25.4))
a = fig.add_axes([.07, .17, .56, .74]); a.axis("off")
box = a.get_position(); fig.text(box.x0 - .035, box.y1 + .055 * box.height, "a", fontsize=10, fontweight="bold", va="bottom")
a.text(0, 1.06, "Cohort construction", transform=a.transAxes, fontsize=8.2, fontweight="bold", va="bottom")
blocks = [
    ("Source baseline records", "8,907"),
    ("Clinical CPP; eligible baseline cohort", "3,606"),
    ("Linked follow-up assessment", "770"),
    ("120–240-day observation interval", "749"),
    ("Analyzable growth outcome", "735"),
]
ys = [.88, .68, .48, .28, .08]
for i, ((label, n), y) in enumerate(zip(blocks, ys)):
    a.add_patch(FancyBboxPatch((.03, y - .07), .58, .145, boxstyle="round,pad=.008,rounding_size=.016",
                               fc=BLUE if i in [1, 4] else LIGHT, ec="none"))
    color = "white" if i in [1, 4] else INK
    a.text(.06, y + .022, label, fontsize=6.7, color=color, va="center")
    a.text(.06, y - .029, n, fontsize=15, fontweight="bold", color="white" if i in [1, 4] else BLUE, va="center")
    if i < 4:
        a.annotate("", xy=(.31, ys[i + 1] + .082), xytext=(.31, y - .08),
                   arrowprops={"arrowstyle": "->", "color": GREY, "lw": .9})
for y, text in [(.77, "5,299 other diagnoses\n2 outside age range"), (.57, "2,836 without linked\nfollow-up assessment"),
                (.37, "21 outside\ntime window"), (.17, "14 unresolved\nvelocity flags")]:
    a.plot([.31, .65], [y, y], color=GREY, lw=.65)
    a.text(.67, y, text, va="center", fontsize=6.4)
a.text(.03, -.075, "Follow-up source: 1,685 records; 770 linked to the eligible baseline cohort.", fontsize=6.1, color=GREY)

b = fig.add_axes([.76, .21, .21, .65])
box = b.get_position(); fig.text(box.x0 - .035, box.y1 + .055 * box.height, "b", fontsize=10, fontweight="bold", va="bottom")
b.text(0, 1.06, "Formulation mix", transform=b.transAxes, fontsize=8.2, fontweight="bold", va="bottom")
for x, col in enumerate(["baseline_n", "observed_primary_n"]):
    vals = rates[col].to_numpy(); bottom = 0
    for color, n in zip(COLORS, vals):
        height = n / vals.sum() * 100
        b.bar(x, height, bottom=bottom, width=.6, color=color, edgecolor="white", lw=.7)
        b.text(x, bottom + height / 2, f"{height:.1f}%", ha="center", va="center", color="white", fontsize=6.6, fontweight="bold")
        bottom += height
    b.text(x, 104, f"n={vals.sum():,}", ha="center", fontsize=6.5)
b.set_ylim(0, 112); b.set_yticks([0, 50, 100]); b.set_ylabel("Patients (%)")
b.set_xticks([0, 1], ["Baseline\ncohort", "Analyzable\nfollow-up"])
b.set_axisbelow(True); b.grid(axis="y", color=LIGHT, lw=.55); b.tick_params(length=3, width=.65)
for i, (label, color) in enumerate(zip(["Long-acting", "Short-acting aqueous", "Short-acting powder"], COLORS)):
    fig.text(.10 + i * .29, .035, "●  " + label, color=color, fontsize=7)

stem = OUT / "Figure1_cohort_and_composition_professional_labels"
for ext in ["pdf", "svg", "png", "tiff"]:
    kwargs = {"dpi": 600 if ext == "tiff" else 300}
    if ext == "tiff": kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(stem.with_suffix("." + ext), **kwargs)
plt.close(fig)

(UPLOAD / "Figure_1.pdf").write_bytes(stem.with_suffix(".pdf").read_bytes())
qa = {
    "figure": "Figure 1", "width_mm": 170, "height_mm": 125,
    "internal_labels_removed": ["V1", "V2"],
    "pdf_sha256": hashlib.sha256((UPLOAD / "Figure_1.pdf").read_bytes()).hexdigest(),
}
(OUT / "Figure1_professional_labels_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(UPLOAD / "Figure_1.pdf")
