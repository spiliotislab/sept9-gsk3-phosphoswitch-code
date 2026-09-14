# 04_neuron_GC_Pearson — neuron growth-cone Pearson colocalization (Fig 6 / S10)

Code behind the neuron growth-cone colocalization figures: **2D Pearson colocalization of SEPT9**
with **acetylated-tubulin**, **F-actin**, and **pSer9-GSK3β**, in DIV3 hippocampal neurons, resolved
by compartment (**growth cone vs axon shaft**) and by growth-cone subdomain, **vehicle (Control) vs
CHIR99021**. Statistics are two-group Mann-Whitney U with Cohen's d (**Fig 6, S10**). Includes the
triple SEPT9 × acetyl-tubulin × pSer9-GSK3β colocalization (**Fig 6D**).

This is a **2D analysis** — every Pearson coefficient is computed on a single optical section /
max-intensity projection, per compartment. It is not a 3D volume analysis.

Runs with **no path editing**. Every script writes its outputs relative to this folder, with
environment-variable overrides that point the pipeline at your own images.

```
04_neuron_GC_Pearson/
  code/      analysis + figure scripts, _config.py, reproduce_from_tables.py
  data/      derived per-cell / per-compartment Pearson tables behind Fig 6 / S10 (flat CSVs)
  outputs/   created on first run — all figures/tables land here (under per-analysis subfolders)
```

Two ways to use it: **(A) reproduce the Fig 6 / S10 Pearson comparison from the shipped per-cell
tables** (no images), or **(B) run the analysis engines on your own `.nd2` images**.

## Install (once)
```bash
pip install -r ../requirements.txt        # whole deposit
```
Python 3.11+ (developed on 3.14). Reproducing from the shipped CSVs needs `pandas`, `scipy`,
`matplotlib`. The image-reading engines (Section B) additionally need `nd2`, `numpy`, `scikit-image`,
and `openpyxl` (they write `.xlsx` summaries).

## A. Reproduce Fig 6 / S10 from the shipped tables (no images needed)
From `code/`:
```bash
py reproduce_from_tables.py
```
Reads `data/gc_shaft_metrics_actub.csv` (SEPT9–tubulin and SEPT9–actin per-cell
Pearson) and `data/gc_shaft_metrics_gskp.csv` (SEPT9–pGSK3β per-cell Pearson),
and regenerates the Fig 6 / S10 comparison: per-compartment (growth cone, shaft) Pearson for each
partner, Control vs CHIR, with a two-sided Mann-Whitney U per panel. Writes
`neuron_pearson_by_compartment.png` + `neuron_pearson_by_compartment_stats.csv` to `outputs/`.
Missing columns are skipped gracefully, so it runs on whatever the shipped tables contain.

The paper's reported p-values (Mann-Whitney U + Cohen's d) were computed in GraphPad Prism; the
values this script prints use SciPy and are indicative.

## B. Analyze your own data
The engine scripts read the original fixed-cell `.nd2` stacks and write the per-cell Pearson tables
that the figures reproduce from. Point them at your images and run them.

There are **two image datasets**:

- **`PGSK_IMAGES`** — the 3-channel dataset (`031826` DIV3): actin / SEPT9 / (acetyl-tubulin *or*
  pGSK3β). Used by 5 of the 6 scripts.
- **`PGSK_IMAGES2`** — the 4-channel single-plane crops (`01326`): tubulin / SEPT9 / pGSK3β / actin.
  Used only by `triple_coloc_gsk_s9_tub.py` (Fig 6D).

```bash
# Windows (PowerShell)
$env:PGSK_IMAGES="C:\path\to\031826_stacks"
$env:PGSK_IMAGES2="C:\path\to\01326_crops-single-plane"   # only for the triple-coloc script
py chir_vs_control_gc_analysis.py

# macOS/Linux
export PGSK_IMAGES=/path/to/031826_stacks
export PGSK_IMAGES2=/path/to/01326_crops-single-plane
python chir_vs_control_gc_analysis.py
```

If channel order differs from the defaults below, set the `PGSK_CH_*` variables too.

### Run order among the 6 scripts
Run the two **engines** first — they read the images and write the master per-cell tables:

1. `chir_vs_control_gc_analysis.py`  → `outputs/gc_analysis_results/gc_shaft_metrics.csv` (acetyl-tubulin set)
2. `gskp_vs_control_analysis.py`      → `outputs/gskp_analysis_results/gskp_gc_shaft_metrics.csv` (pGSK set)

Then:

3. `gc_peripheral_s9_actub.py` — reads `gc_shaft_metrics.csv` (from step 1) **and** the images; run after step 1.

**Independent** (read images directly, any order): `subdomain_s9_actin_gskp.py`,
`subdomain_s9_gskp.py` (both need `PGSK_IMAGES`), and `triple_coloc_gsk_s9_tub.py` (needs
`PGSK_IMAGES2`).

### Configuration (environment variables — all optional)
| Variable | Meaning | Default |
|---|---|---|
| `PGSK_OUT` | output directory | `../outputs` |
| `PGSK_IMAGES` | 3-channel fixed `.nd2` stacks (actin / SEPT9 / AcTub-or-pGSK) | *(unset)* |
| `PGSK_IMAGES2` | 4-channel single-plane crops (tub / SEPT9 / pGSK / actin) | *(unset)* |
| `PGSK_CH_ACT` / `PGSK_CH_S9` / `PGSK_CH_TUB` / `PGSK_CH_GSKP` | 0-based channel indices, 3-channel sets | `0` / `1` / `2` / `2` |
| `PGSK_CH4_TUB` / `PGSK_CH4_S9` / `PGSK_CH4_GSK` / `PGSK_CH4_ACT` | 0-based channel indices, 4-channel crops | `0` / `1` / `2` / `3` |

All defaults live in `code/_config.py`. Each analysis writes into its own subfolder of `outputs/`
(`gc_analysis_results/`, `gskp_analysis_results/`, `gc_peripheral_results/`, `triple_coloc/`), so
per-dataset figures with the same filename never collide.

## Scripts
- `chir_vs_control_gc_analysis.py` — **engine (Fig 6, S10)**: GC-vs-shaft segmentation + SEPT9 / acetyl-tubulin / actin Pearson coloc, Control vs CHIR (acetyl-tubulin dataset); writes `gc_shaft_metrics.csv`.
- `gskp_vs_control_analysis.py` — **engine (Fig 6, S10)**: GC-vs-shaft SEPT9 / pSer9-GSK3β / actin Pearson coloc, Control vs CHIR (pGSK dataset); writes `gskp_gc_shaft_metrics.csv`.
- `gc_peripheral_s9_actub.py` — **S10**: growth-cone subdomain (central / transition / peripheral) SEPT9–acetyl-tubulin Pearson; writes `gc_subdomain_metrics.csv` / `gc_subdomain_stats.csv`.
- `subdomain_s9_actin_gskp.py` — **S10**: SEPT9–actin Pearson by growth-cone subdomain (pGSK dataset); writes `subdomain_s9_actin_pearson.csv`.
- `subdomain_s9_gskp.py` — **S10**: SEPT9–pSer9-GSK3β Pearson/Spearman by growth-cone subdomain; writes `subdomain_s9_gskp_pearson_spearman.csv`.
- `triple_coloc_gsk_s9_tub.py` — **Fig 6D**: triple pSer9-GSK3β × SEPT9 × tubulin colocalization, Control vs CHIR (`PGSK_IMAGES2`).
- `reproduce_from_tables.py` — regenerates the Fig 6 / S10 Pearson-by-compartment comparison + Mann-Whitney from the shipped tables (no images).

## Method notes
- **2D analysis:** Pearson is computed on a single optical section / max-intensity projection — this is not a 3D volume analysis.
- **Compartments:** each neurite is segmented from the combined channels, then split into the growth cone vs the axon shaft; the growth cone is further partitioned into central / transition / peripheral subdomains by a distance transform (EDT) from the GC edge.
- **Colocalization** = per-compartment **Pearson r** of SEPT9 vs each partner (acetyl-tubulin / F-actin / pSer9-GSK3β) on masked pixels (Manders / Jaccard / ICQ are carried as companions in the tables).
- **Statistics** = two-group **Mann-Whitney U** (Control vs CHIR) per compartment, with **Cohen's d** effect sizes; the reported p-values in the paper were computed in **GraphPad Prism** (`reproduce_from_tables.py` prints its own indicative SciPy Mann-Whitney).
- **Channel identities** are verified per script from its own header comments; the acetyl-tubulin and pGSK 3-channel sets share actin=0 / SEPT9=1, and the 4-channel crops reorder to tub / SEPT9 / pGSK / actin.

## Data availability
The **per-cell / per-compartment Pearson tables** behind Fig 6 / S10 are included as a flat set of
CSVs in `data/`: `gc_shaft_metrics_actub.csv` (SEPT9–tubulin & SEPT9–actin, acetyl-tubulin dataset)
and `gc_shaft_metrics_gskp.csv` (SEPT9–pGSK3β, GSKp dataset) are the two headline tables Section A
reproduces from; the growth-cone subdomain / whole-compartment Pearson tables
(`subdomain_s9_actin_pearson.csv`, `subdomain_s9_gskp_pearson_spearman.csv`, `gc_subdomain_metrics.csv`,
`gc_subdomain_stats.csv`, `gskp_coloc_per_cell.csv`, `pioneer_mt_s9_gskp.csv`, `pearson_subdomain.csv`,
`pearson_whole_shaft.csv`) back the S10 subdomain panels. Section A reproduces Fig 6 / S10 from these
tables alone, no images required. (When you re-run the engines in Section B they write these same
tables into `outputs/<engine>_results/`.)

The **raw fixed-cell image stacks** (the `031826` 3-channel `.nd2` stacks and the `01326` 4-channel
single-plane crops) are not on GitHub because of their size; they live in the lab archive
(`NOOR_pGSK_growthcone_fixed_code_and_data.zip`, which includes the images) and the image data
repository. Point `PGSK_IMAGES` / `PGSK_IMAGES2` at them to run the engines (section B).
