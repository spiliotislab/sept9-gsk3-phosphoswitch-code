"""Reproduce the neuron growth-cone Pearson colocalization (Fig 6 / S10) FROM THE SHIPPED
TABLES — no raw image stacks needed.

    python reproduce_from_tables.py

Reads the two per-cell metric tables in ../data and regenerates the Fig 6 / S10 comparison:
per-compartment (growth cone vs axon shaft) Pearson colocalization of SEPT9 with each partner —
acetylated-tubulin, F-actin, and pSer9-GSK3beta — for Control (vehicle) vs CHIR, with a two-sided
Mann-Whitney U per compartment/partner. Writes one figure (neuron_pearson_by_compartment.png) and a
stats table (neuron_pearson_by_compartment_stats.csv) to the output dir.

Inputs (per-cell rows, produced from images by the analysis engines — see README 'Analyze your
own data'):
  data/gc_shaft_metrics_actub.csv          SEPT9-tubulin & SEPT9-actin Pearson (acetyl-tubulin dataset)
      columns: condition, gc_pearson_s9_tub, gc_pearson_s9_act, shaft_pearson_s9_tub, shaft_pearson_s9_act
  data/gc_shaft_metrics_gskp.csv           SEPT9-pGSK3beta Pearson (GSKp dataset)
      (the pGSK acquisition's third channel is pSer9-GSK3beta, carried under the generic
       gc_pearson_s9_tub / shaft_pearson_s9_tub columns; the loader also accepts explicit
       *_pearson_s9_gsk* columns if a table provides them)

This is a 2D analysis — Pearson is computed on a single optical section / max-intensity projection,
per compartment. The paper's reported p-values (Mann-Whitney U + Cohen's d; Fig 6, S10) were
computed in GraphPad Prism; the values printed here are indicative and use SciPy's two-sided
Mann-Whitney U. Any partner whose columns are absent from the shipped tables is skipped gracefully.
"""
import os
import _config as C
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ARM  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ARM, "data")
ACTUB_TABLE = os.path.join(DATA, "gc_shaft_metrics_actub.csv")
GSKP_TABLE  = os.path.join(DATA, "gc_shaft_metrics_gskp.csv")

ORDER  = ["Control", "CHIR"]
COLORS = {"Control": "#4d4d4d", "CHIR": "#c0392b"}


def _load(path):
    return pd.read_csv(path) if os.path.isfile(path) else None


def _pick(df, candidates):
    """First column in `candidates` that exists in df, else None."""
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


# (partner label, dataframe, gc-column-candidates, shaft-column-candidates)
actub = _load(ACTUB_TABLE)
gskp  = _load(GSKP_TABLE)
PARTNERS = [
    ("SEPT9-tubulin", actub, ["gc_pearson_s9_tub"],                        ["shaft_pearson_s9_tub"]),
    ("SEPT9-actin",   actub, ["gc_pearson_s9_act"],                        ["shaft_pearson_s9_act"]),
    # pGSK acquisition: third channel = pSer9-GSK3beta, stored under s9_tub (or explicit s9_gsk)
    ("SEPT9-pGSK",    gskp,  ["gc_pearson_s9_gsk", "gc_pearson_s9_gskp", "gc_pearson_s9_tub"],
                             ["shaft_pearson_s9_gsk", "shaft_pearson_s9_gskp", "shaft_pearson_s9_tub"]),
]
COMPARTMENTS = ["growth cone", "shaft"]

# ---- assemble the columns we can actually plot, skipping missing ones ----
panels = []   # (partner, compartment, df, column)
for label, df, gc_cands, sh_cands in PARTNERS:
    if df is None:
        print(f"[skip] {label}: table not found")
        continue
    if "condition" not in df.columns:
        print(f"[skip] {label}: no 'condition' column")
        continue
    gc_col, sh_col = _pick(df, gc_cands), _pick(df, sh_cands)
    for comp, col in (("growth cone", gc_col), ("shaft", sh_col)):
        if col is None:
            print(f"[skip] {label} / {comp}: Pearson column not present")
            continue
        panels.append((label, comp, df, col))

if not panels:
    raise SystemExit("No Pearson columns found in the shipped tables; nothing to reproduce.")

# ---- plot: one subplot per (partner x compartment); Control vs CHIR points + median ----
labels = [p[0] for p in PARTNERS]
ncols  = len(labels)
fig, axes = plt.subplots(len(COMPARTMENTS), ncols, figsize=(3.4 * ncols, 7.2), squeeze=False)
rng = np.random.default_rng(0)
rows = []
by_key = {(lab, comp): (df, col) for lab, comp, df, col in panels}

for r, comp in enumerate(COMPARTMENTS):
    for c, lab in enumerate(labels):
        ax = axes[r][c]
        entry = by_key.get((lab, comp))
        if entry is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{lab}\n{comp}\n(no data)", ha="center", va="center",
                    fontsize=9, color="#999", transform=ax.transAxes)
            continue
        df, col = entry
        groups = {}
        for i, cond in enumerate(ORDER):
            v = pd.to_numeric(df.loc[df.condition == cond, col], errors="coerce").dropna().values
            groups[cond] = v
            if len(v):
                ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=26, alpha=0.7,
                           color=COLORS[cond], edgecolor="none", zorder=3)
                ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color="k", lw=2.4, zorder=4)
        ctrl, chir = groups.get("Control", np.array([])), groups.get("CHIR", np.array([]))
        p = (mannwhitneyu(ctrl, chir, alternative="two-sided").pvalue
             if len(ctrl) and len(chir) else np.nan)
        rows.append(dict(
            partner=lab, compartment=comp, column=col,
            n_control=len(ctrl), n_chir=len(chir),
            median_control=(round(float(np.median(ctrl)), 4) if len(ctrl) else ""),
            median_chir=(round(float(np.median(chir)), 4) if len(chir) else ""),
            mannwhitney_p=(round(float(p), 5) if p == p else "")))
        ax.set_xticks([0, 1]); ax.set_xticklabels(ORDER, fontsize=9)
        ax.set_xlim(-0.6, 1.6)
        if c == 0:
            ax.set_ylabel(f"{comp}\nPearson r", fontsize=10)
        if r == 0:
            ax.set_title(lab, fontsize=11)
        if p == p:
            ax.text(0.5, 0.97, f"MWU p={p:.3g}", ha="center", va="top",
                    transform=ax.transAxes, fontsize=8, color="#555")

fig.suptitle("Neuron growth-cone SEPT9 Pearson colocalization by compartment "
             "(Control vs CHIR; Fig 6 / S10)", fontsize=12)
fig.text(0.5, 0.005,
         "2D single-plane / MIP Pearson; median bars; two-sided Mann-Whitney U (SciPy). "
         "Paper p-values (+ Cohen's d) were computed in Prism.",
         ha="center", fontsize=8, color="#666")
import matplotlib as mpl; mpl.rcParams["pdf.fonttype"] = 42
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

fig_path   = C.out("neuron_pearson_by_compartment.png")
stats_path = C.out("neuron_pearson_by_compartment_stats.csv")
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
stats = pd.DataFrame(rows)
stats.to_csv(stats_path, index=False)

print("saved neuron_pearson_by_compartment.png + _stats.csv to", C.OUTDIR)
print(stats.to_string(index=False))
