"""Reproduce the Figure 2I quantification from the shipped table — no images needed.

    python reproduce_from_tables.py

Reads data/sept9_mt_3d_manders_by_genotype.csv (per-cell 3D voxel-Manders M1 of SEPT9 on
microtubules, nucleus-excluded, one row per U2OS cell) and regenerates the Fig 2I comparison:
Manders M1 by genotype (per-cell points + median) with pairwise Mann-Whitney U vs wild type.
Writes sept9_mt_manders_by_genotype.png + _stats.csv to the output dir.

Note: the per-cell values were produced from 3D image volumes by sept9_mt_coloc_3d.py (needs the
images, see README 'Analyze your own data'); the paper's reported p-values were computed in Prism.
"""
import os
import _config as C
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ARM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ARM, "data", "sept9_mt_3d_manders_by_genotype.csv")
COL = "manders_M1_SEP_in_MT_exNuc"
ORDER = ["WT", "S85A", "S82A/S85A", "S85E", "S82E/S85E"]

d = pd.read_csv(TABLE)
present = [g for g in ORDER if g in set(d.genotype)]
colors = {"WT": "#444444", "S85A": "#2171b5", "S82A/S85A": "#08519c",
          "S85E": "#d94801", "S82E/S85E": "#a63603"}

fig, ax = plt.subplots(figsize=(7, 5.5))
rng = np.random.default_rng(0)
for i, g in enumerate(present):
    v = d.loc[d.genotype == g, COL].values
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=34, alpha=0.75,
               color=colors.get(g, "#666"), edgecolor="none", zorder=3)
    ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=2.5, zorder=4)
ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=15)
ax.set_ylabel("Manders M1  (fraction of SEPT9 on microtubules, 3D, nucleus-excluded)")
ax.set_title("SEPT9–microtubule colocalization by genotype (U2OS, Fig 2I)")
ax.set_ylim(0, None)

# pairwise Mann-Whitney U vs WT
rows = []
wt = d.loc[d.genotype == "WT", COL].values
for g in present:
    v = d.loc[d.genotype == g, COL].values
    if g == "WT":
        p = np.nan
    else:
        p = mannwhitneyu(wt, v, alternative="two-sided").pvalue
    rows.append(dict(genotype=g, n=len(v), median_M1=round(float(np.median(v)), 3),
                     mannwhitney_p_vs_WT=(round(float(p), 4) if p == p else "")))
stats = pd.DataFrame(rows)
ax.text(0.5, -0.22, "median bars; Mann-Whitney U vs WT (two-sided) in *_stats.csv. Paper p-values were computed in Prism.",
        transform=ax.transAxes, ha="center", fontsize=8, color="#555")
plt.tight_layout()
import matplotlib as mpl; mpl.rcParams["pdf.fonttype"] = 42
plt.savefig(C.out("sept9_mt_manders_by_genotype.png"), dpi=200, bbox_inches="tight")
stats.to_csv(C.out("sept9_mt_manders_by_genotype_stats.csv"), index=False)
print("saved sept9_mt_manders_by_genotype.png + _stats.csv to", C.OUTDIR)
print(stats.to_string(index=False))
