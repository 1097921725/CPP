"""Create Supplementary Figure S3 in the audited Nature-style visual language."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "analysis/output/v4.0/cpp_rhgh_integrated_20260910/gnrha_secondary_20260911"
PACKAGE = ROOT / "analysis/output/v4.0/cpp_rhgh_integrated_20260910/bmc_submission_20260911/gnrha_effectiveness_extension_20260911"
OUT = PACKAGE / "figures"
UPLOAD = PACKAGE / "upload_after_author_confirmation"
OUT.mkdir(parents=True, exist_ok=True)
UPLOAD.mkdir(parents=True, exist_ok=True)

BLUE = "#24658B"
CORAL = "#CB624D"
TEAL = "#519D9A"
INK = "#233647"
GREY = "#8B97A3"
LIGHT = "#EAF0F4"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 7.2, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
    "ytick.labelsize": 7.0, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.65, "pdf.fonttype": 42,
    "svg.fonttype": "none", "savefig.facecolor": "white", "axes.facecolor": "white",
    "figure.facecolor": "white", "svg.hashsalt": "cpp-gnrha-20260911",
})

effects = pd.read_csv(DATA / "gnrha_effectiveness_estimates.csv")
lookup = effects.set_index("analysis")
keys = [
    "Recorded concurrent GnRHa: long-acting versus pooled short-acting (interaction model)",
    "No explicit GnRHa record: long-acting versus pooled short-acting",
    "Formulation-by-recorded-GnRHa interaction",
]
rows = lookup.loc[keys]
labels = [
    "Recorded concurrent GnRHa\n(n=544; 58 hospitals)",
    "No explicit GnRHa record\n(n=191; 51 hospitals)",
    "Interaction: difference\nbetween formulation contrasts",
]
colors = [CORAL, BLUE, TEAL]

fig = plt.figure(figsize=(170 / 25.4, 88 / 25.4))
ax = fig.add_axes([0.34, 0.24, 0.46, 0.64])
ax.axvline(0, color=GREY, linestyle=(0, (3, 3)), linewidth=0.8)
for i, ((_, row), color) in enumerate(zip(rows.iterrows(), colors)):
    if i % 2 == 0:
        ax.axhspan(i - 0.40, i + 0.40, color=LIGHT, alpha=0.65, zorder=0)
    ax.plot([row.ci_low, row.ci_high], [i, i], color=color, linewidth=1.8, solid_capstyle="round")
    ax.scatter(row.estimate, i, s=38, color=color, edgecolors="white", linewidths=0.6, zorder=3)
    ax.text(1.04, i, f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]",
            transform=ax.get_yaxis_transform(), va="center", fontsize=6.8, color=INK)
ax.text(1.04, 1.04, "Estimate [95% CI]", transform=ax.transAxes, fontsize=6.6, color=GREY)
ax.set_yticks(range(3), labels)
ax.set_ylim(2.55, -0.55)
ax.set_xlim(-2.05, 1.25)
ax.set_xticks([-2, -1, 0, 1])
ax.set_xlabel("Adjusted difference in annualized height velocity (cm/year)", labelpad=7)
ax.tick_params(axis="y", length=0, pad=8)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color(GREY)
ax.set_axisbelow(True)
ax.grid(axis="x", color=LIGHT, linewidth=0.55)
fig.text(0.08, 0.09,
         "Long-acting minus pooled short-acting rhGH. Negative values indicate lower observed growth with long-acting use.",
         fontsize=6.4, color=GREY)

stem = OUT / "SupplementaryFigureS3_recorded_GnRHa_effectiveness"
fig.savefig(stem.with_suffix(".pdf"))
fig.savefig(stem.with_suffix(".svg"))
fig.savefig(stem.with_suffix(".png"), dpi=300)
fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)

shutil.copy2(stem.with_suffix(".pdf"), UPLOAD / "Additional_file_3.pdf")
png = Image.open(stem.with_suffix(".png"))
qa = {
    "width_mm": 170,
    "height_mm": 88,
    "png_pixels": list(png.size),
    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("SupplementaryFigureS3*")},
    "source_sha256": hashlib.sha256((DATA / "gnrha_effectiveness_estimates.csv").read_bytes()).hexdigest(),
    "scope": "effectiveness only; no safety analysis",
}
(OUT / "SupplementaryFigureS3_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
print(stem.with_suffix(".pdf"))
print(UPLOAD / "Additional_file_3.pdf")
