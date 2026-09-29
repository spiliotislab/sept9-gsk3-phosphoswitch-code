# Paper code deposit — Septin (SEPT9) crosstalk with microtubules and actin via a GSK3-dependent phosphoswitch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23046030.svg)](https://doi.org/10.5281/zenodo.23046030)

Custom analysis code for the paper, organized by analysis arm. Each arm has its own `code/`, an arm-level `README.md`, and (where applicable) `data/`. Install once from the shared `requirements.txt`.

## Arms

| Arm | Analysis | Scripts |
|---|---|---|
| **01_MCF_3D_voxel_Manders_live** | MCF live-cell CHIR99021 time-lapse; **3D voxel-Manders colocalization** of SEPT9 with microtubules and actin stress fibers (per-cell slopegraphs, Wilcoxon). | 11 + data |
| **02_centriole_EB1_comet_tracking** | Centriole EB1-comet tracking (Fig 3E–G): comet nucleation rate, outward velocity, and lifetime ON/OFF SEPT9 filaments; WT vs S82A/S85A. (Dataset 073126.) | 5 + data |
| **03_U2OS_fixed_3D_voxel_Manders** | Fixed U2OS; **3D voxel-Manders** of SEPT9↔microtubules (Fig 2I, WT vs phosphomutants) + SEPT9–stress-fiber coincidence (Fig 1F–G). | 5 + data |
| **04_neuron_GC_Pearson** | Fixed DIV3 hippocampal neurons Control vs CHIR; **2D Pearson colocalization** of SEPT9 with acetyl-tubulin / F-actin / pSer9-GSK3β by growth-cone vs axon-shaft compartment (Fig 6, S10, 6D). | 6 + data |
| **overlap_image_display** | Binary coincidence-mask TIFF export for the **Overlap image display** figures (Fig 1F fixed U2OS, Fig 1H live MCF); Python builds SEPT9/network masks and exports ImageJ composites shown as a binary yellow overlap. Visualization only. | 4 |

> **Not in this deposit:** live SEPT9-GFP WT/S85A/S85E growth-cone dynamics (puncta tracking, actin-flow coupling, 4-state kinetics) were analysed but are **not part of this paper**. That code + data are kept in the lab archive only (`NOOR_live_SEPT9_S85_growthcone_code_and_data.zip`).

## Environment
Python 3.11+ (developed on 3.14). `pip install -r requirements.txt`.

## How to run (works out of the box)
Each arm is self-contained and **needs no path editing**. A small `code/_config.py` resolves inputs
and outputs relative to the arm folder, with environment-variable overrides so you can point a script
at your own data. Outputs always land in that arm's `outputs/` folder.

- **Reproduce a figure from included data** (arm 01 ships its per-cell table):
  ```bash
  cd 01_MCF_3D_voxel_Manders_live/code
  pip install -r ../../requirements.txt
  python reproduce_figures.py          # writes every figure/table to ../outputs
  ```
- **Analyze your own data:** set the arm's image-folder env var (e.g. `MCF_IMAGES`, `U2OS_IMAGES`,
  `PGSK_IMAGES`, `EB1_IMAGES`) plus channel/voxel overrides, then run the arm's pipeline script.
  Each arm's `README.md` gives the exact variables, run order, and expected outputs.

See each arm's `README.md` for its quickstart, environment-variable table, and script list.

## Data availability (deposit-wide policy)
Each arm bundles the **small processed tables needed to reproduce its figures/statistics without the
raw images**, and provides a one-command reproducer where such tables exist:
- **01_MCF** — `data/per_cell_sf_mt.csv` → `reproduce_figures.py`.
- **02_centriole_EB1** — `data/` comet tables (detections, tracks, per-comet/per-aster parameters) → `reproduce_from_tables.py`.
- **04_neuron_GC_Pearson** — `data/` per-cell/per-compartment Pearson tables → `reproduce_from_tables.py`.
- **03_U2OS_fixed_3D_voxel_Manders** — `data/` per-cell Fig 2I Manders table → `reproduce_from_tables.py` (the Fig 1F–G stress-fiber panels are still produced directly from the `.nd2` images).

Large raw/deconvolved image stacks and movies are **not** on GitHub (size); they live in the lab
archive zips and the image data repository, with drive locations in each archive's `RAW_DATA_MANIFEST.csv`.

## Reproducibility notes
- Preprocessing (Noise2Void2 denoise + 3D Richardson–Lucy deconvolution) was run on the microscope-export `.nd2` stacks; preprocessing scripts are included where retained.
- Input/output locations are set in `code/_config.py` per arm and overridable by environment variables — no need to edit script bodies.
- Only **final figure/statistic producers** (plus shared engines and master data tables) are included; exploratory, pilot, superseded, deck-builder, and 3D-rendering/infrastructure scripts were intentionally excluded (each arm README notes shared engines/tables).

## Citation / availability
Archive a tagged release to Zenodo for a DOI and cite it in the paper's **Code Availability** statement. Raw and deconvolved image data are deposited separately (**Data Availability**).

## Use of AI tools
Analysis code was developed and executed with the assistance of Claude (Claude Opus 4.8; Anthropic, 2026), an AI coding assistant, under author direction; all parameter choices, analytical decisions, and interpretations were reviewed and verified by the authors, who take full responsibility for the accuracy of the results.
