# 03_U2OS_fixed_3D_voxel_Manders — fixed U2OS 3D voxel-Manders colocalization

Code behind the fixed U2OS figures. Headline analysis: **3D voxel-Manders colocalization of SEPT9
with microtubules** in cell volumes (Figure 2I; WT vs S82/S85 phosphomutants). Also included: SEPT9
decoration of and coincidence with actin stress fibers (Figure 1F–G; Control vs CHIR99021).

Runs out of the box — **no path editing**. Every script finds its inputs and writes its outputs
relative to this folder, with environment-variable overrides.

```
03_U2OS_fixed_3D_voxel_Manders/
  code/      analysis + figure scripts, _config.py, reproduce_from_tables.py
  data/      sept9_mt_3d_manders_by_genotype.csv  (per-cell Fig 2I Manders M1, 5 genotypes)
  outputs/   created on first run — all figures/tables/TIFFs land here
```

Two ways to use it: **(A) reproduce the Fig 2I quantification from the shipped per-cell table** (no
images), or **(B) run the pipelines on your own `.nd2` images** (Fig 1F–G stress-fiber coincidence,
and recomputing the Fig 2I per-cell Manders from image volumes).

## A. Reproduce Fig 2I from the shipped table (no images needed)
```bash
pip install -r ../requirements.txt
python reproduce_from_tables.py
```
Reads `data/sept9_mt_3d_manders_by_genotype.csv` and regenerates the Fig 2I comparison — SEPT9-on-MT
Manders M1 per genotype (per-cell points + median) with pairwise Mann-Whitney U vs wild type —
writing `sept9_mt_manders_by_genotype.png` + `_stats.csv` to `outputs/`.

## Install (once)
```bash
pip install -r ../requirements.txt
```
Python 3.11+ (developed on 3.14). The 3D pipeline (`sept9_mt_coloc_3d.py`) optionally uses a
CuPy GPU path and falls back to CPU automatically if CuPy is not installed.

## B. Analyze your own data
1. **Point the scripts at your images.** Put your Nikon `.nd2` stacks in one folder and set
   `U2OS_IMAGES`:
   ```bash
   # Windows (PowerShell)
   $env:U2OS_IMAGES="C:\path\to\nd2"
   # macOS/Linux
   export U2OS_IMAGES=/path/to/nd2
   ```
   Set channel indices / voxel size too if your acquisition differs (see table below).
2. **Run the stress-fiber scripts** (from `code/`). Run the coincidence map **first** — it
   writes the ImageJ-editable raw+mask TIFFs into `outputs/imagej/` that the brightness script reads:
   ```bash
   py build_u2os_channel_id.py       # optional: confirm which channel is which
   py build_u2os_coincmap.py         # SEPT9 ∩ stress-fiber map + Manders; writes outputs/imagej/*.tif
   py build_u2os_sept9_sf_quant.py   # SEPT9-on-SF brightness + continuity  (reads outputs/imagej/)
   ```
   The two example filenames (`U2OS_Cntrl_3.nd2`, `CHIR_theOne006.nd2`) are set at the top of
   `build_u2os_channel_id.py` / `build_u2os_coincmap.py` — edit them to match your files.
3. **Run the 3D MT colocalization pipeline** on the *separate* 2-channel MT/SEPT9 acquisition:
   ```bash
   py sept9_mt_coloc_3d.py                 # batch every matching .nd2 in U2OS_IMAGES
   py sept9_mt_coloc_3d.py cell01.nd2      # or a single file
   ```
   Writes `outputs/sept9_mt_coloc_3d/per_file/<stem>.csv`, `qc/<stem>_qc.png`, and `master.csv`.
   Channel order here is `ch0=561 tubulin, ch1=488 SEPT9` (override with `--mt-ch` / `--sep-ch`).

All outputs land in `outputs/` (created on first run).

### Configuration (environment variables — all optional)
| Variable | Meaning | Default |
|---|---|---|
| `U2OS_IMAGES` | folder of `.nd2` stacks (input for every script) | *(unset)* |
| `U2OS_OUT` | output directory | `../outputs` |
| `U2OS_CH_ACTIN` / `U2OS_CH_SEPT9` / `U2OS_CH_MT` | 0-based channel indices for the 3-channel stress-fiber stacks (640=actin, 561=SEPT9, 488=MT) | `0` / `1` / `2` |
| `U2OS_VOXEL_XY` / `U2OS_VOXEL_Z` | voxel size (µm); `sept9_mt_coloc_3d.py` reads the true voxel from each `.nd2` | `0.0384` / `0.2` |

All defaults live in `code/_config.py`.

## Scripts
**Stress-fiber arm (3-channel 640/561/488 `.nd2`; run `build_u2os_coincmap.py` first):**
- `build_u2os_channel_id.py` — channel verification: renders 640/561/488 at basal z with zoom crops + a Frangi network-energy score to tell the fine MT network from SEPT9 (punctate/bundled) and actin stress fibers.
- `build_u2os_coincmap.py` — SEPT9 (top-hat filaments+puncta) ∩ actin stress fibers (Frangi ridge) coincidence map + Manders M1/M2 at basal z, Control vs CHIR; also writes ImageJ-editable raw + mask TIFFs to `outputs/imagej/`.
- `build_u2os_sept9_sf_quant.py` — SEPT9-on-stress-fiber brightness (top-hat enrichment vs whole cell) and continuity (fraction of SF skeleton decorated, mean run-length), Control vs CHIR; reads the coincmap TIFFs.

**3D MT colocalization — Fig 2I (separate 2-channel MT/SEPT9 `.nd2` volumes):**
- `sept9_mt_coloc_3d.py` — **Fig 2I**: 3D SEPT9↔microtubule colocalization + filament-organization pipeline: cell/nucleus masks, white-top-hat SEPT9, anisotropy-aware 3D Frangi tubeness (optional CuPy GPU path), Manders M1/M2, skeleton overlap, orientation alignment, network-organization metrics → per-cell CSV + QC MIP + `master.csv` (the per-genotype master tables are combined into `data/sept9_mt_3d_manders_by_genotype.csv`).
- `reproduce_from_tables.py` — regenerates the Fig 2I Manders-M1-by-genotype comparison + Mann-Whitney from the shipped table (no images).

## Method notes
- **Channel map:** stress-fiber scripts use 640=actin (`U2OS_CH_ACTIN=0`), 561=SEPT9 (`U2OS_CH_SEPT9=1`), 488=MT (`U2OS_CH_MT=2`). `sept9_mt_coloc_3d.py` is a *different acquisition* — a 2-channel MT/SEPT9 stack, `ch0=561 tubulin`, `ch1=488 SEPT9` — and keeps its own indices (override with `--mt-ch` / `--sep-ch`).
- **SEPT9 masks** are filamentous + punctate (white-top-hat + Frangi ridge, nucleus-spared) — diffuse cytoplasmic signal is explicitly excluded; **stress-fiber masks** are Frangi ridges filtered to elongated objects. Colocalization = Manders M1/M2 with a 1-voxel dilation tolerance.
- **z-slices** are 1-based in the titles/filenames (`z1-3` = python `[0:3]`), basal planes.
- **Sample size:** the stress-fiber scripts operate on one cell per condition (n = 1 each) — the panels are **descriptive**, not inferential.
- **Statistics:** the paper's Fig 2I p-values (Mann-Whitney U, n = 5–8) were computed in GraphPad Prism. `reproduce_from_tables.py` prints its own two-sided Mann-Whitney vs WT for convenience — the direction matches the paper (phosphonull higher, phosphomimetic lower), but treat the printed p-values as indicative and refer to Prism for the reported values.
- **QC:** stress-fiber mask overlays and ImageJ-editable TIFFs are written to `outputs/` so every mask can be checked.

## Data availability
- **Included here (`data/`):** `sept9_mt_3d_manders_by_genotype.csv` — the per-cell 3D voxel-Manders values behind **Figure 2I** (33 cells across WT / S85A / S82A-S85A / S85E / S82E-S85E). Fig 2I reproduces from it via `reproduce_from_tables.py`, no images required.
- **Not included (size):** the fixed U2OS `.nd2` image stacks (Fig 1F–G stress-fiber panels are produced directly from these, and the Fig 2I per-cell values are recomputed from the MT/SEPT9 volumes). They are in the lab archive (`NOOR_U2OS_fixed_CHIR_code_and_data.zip`, which includes the images) and the image data repository; point `U2OS_IMAGES` at them.
