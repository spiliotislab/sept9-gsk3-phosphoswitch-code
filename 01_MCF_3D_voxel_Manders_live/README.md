# 01_MCF_3D_voxel_Manders_live — 3D voxel-Manders colocalization (live imaging)

Code behind the MCF live-cell figures and statistics: **3D voxel-Manders colocalization of SEPT9
with microtubules and actin stress fibers** in live-cell CHIR99021 time-lapse (per-cell slopegraphs,
Wilcoxon).

Runs out of the box — **no path editing**. Every script finds its inputs and writes its outputs
relative to this folder, with environment-variable overrides for your own data.

```
01_MCF_3D_voxel_Manders_live/
  code/      analysis + figure scripts, _config.py, reproduce_figures.py
  data/      per_cell_sf_mt.csv   (the per-cell 3D voxel-Manders table behind the figures)
  outputs/   created on first run — all figures/tables land here
```

## Install (once)
```bash
pip install -r requirements.txt        # or ../requirements.txt for the whole deposit
```
Python 3.11+ (developed on 3.14).

## A. Reproduce the paper figures (no images needed)
From `code/`:
```bash
python reproduce_figures.py
```
Runs every figure/stat script against `data/per_cell_sf_mt.csv` and writes to `outputs/`:
`responders_first220_PUB.*`, `manders_panel_stats_PUB.*`, `snapshot_220min.png`,
`positive_responders.*`, `SEPT9_timecourse_13responders.xlsx`,
`SEPT9_220min_PRISM.xlsx`. You can also run any single script on its own, e.g.
`python build_manders_panel_stats.py`.

Expected headline (n = 8 cells reaching ~220 min): SEPT9 on MT 0.54→0.60 (Wilcoxon p = 0.023),
on stress fibers 0.14→0.07 (p = 0.008).

## B. Analyze your own data
1. **Preprocess** your live stacks (denoise + deconvolve). We used Noise2Void2 (CAREamics) then
   3D Richardson–Lucy; any equivalent is fine. Save one multi-channel z-stack per cell per timepoint
   as `<region>__<index>_denoised.tif` in a single folder.
2. **Point the pipeline at your data** and run it:
   ```bash
   # Windows (PowerShell)
   $env:MCF_IMAGES="C:\path\to\stacks"; python build_faithful_population.py
   # macOS/Linux
   MCF_IMAGES=/path/to/stacks python build_faithful_population.py
   ```
   This segments + tracks cells, builds SEPT9 / MT / stress-fiber masks, computes the per-cell
   coincidence, and writes `outputs/per_cell_sf_mt_computed.csv` (same schema as the shipped table).
3. **Make the figures from your table**:
   ```bash
   MCF_DATA=/path/to/outputs/per_cell_sf_mt_computed.csv python reproduce_figures.py
   ```

### Configuration (environment variables — all optional)
| Variable | Meaning | Default |
|---|---|---|
| `MCF_DATA` | per-cell table the figure scripts read | `../data/per_cell_sf_mt.csv` |
| `MCF_OUT` | output directory | `../outputs` |
| `MCF_IMAGES` | folder of denoised multi-channel z-stacks (pipeline input) | *(unset)* |
| `MCF_XY4_DIR` | folder of XY_4 basal/apical crops (single-cell exemplar scripts) | `MCF_IMAGES`, else `code/crops` |
| `MCF_CH_ACTIN` / `MCF_CH_MT` / `MCF_CH_S9` | 0-based channel indices | `0` / `1` / `2` |
| `MCF_VOXEL_XY` / `MCF_VOXEL_Z` | voxel size (µm) | `0.108` / `0.30` |

All defaults live in `code/_config.py`.

## Scripts
**Figure / statistics (run on the table, `reproduce_figures.py` runs them all):**
- `build_positive_responders.py` — responder selection (redistribution score > 0, ≥ 7 timepoints) + companion panel + `positive_responders.csv`.
- `build_responders_first220.py` — main 220-min endpoint figure (Manders M1, SEPT9 on MT vs actin stress fibers; n = 8 + XY_4 003 exemplar).
- `build_manders_panel_stats.py` — single Manders panel with on-plot Wilcoxon p.
- `build_snapshot_220.py` — 220-min snapshot (population mean + exemplar).
- `build_timecourse_xlsx.py` — per-timepoint time-course workbook (peak highlighted).
- `build_prism_xlsx.py` — Prism-ready workbook: before-after tables + Wilcoxon/paired-t (medians, W, Z, effect size r).

**Pipeline / single-cell (need image stacks; set `MCF_IMAGES` / `MCF_XY4_DIR`):**
- `build_faithful_population.py` — population pipeline: segmentation + IoU tracking + SEPT9/MT/stress-fiber masks (top-hat + Frangi + median·MAD) → per-cell coincidence; writes `per_cell_sf_mt_computed.csv`.
- `build_mcf_regional.py` — single-cell 3D regional voxel-Manders (basal/apical z-substacks); SEPT9-on-MT vs on-actin stress fibers.
- `build_xy4_3d_coloc.py` — 3D voxel-Manders colocalization + QC for the XY_4 exemplar (masks + Manders M1/M2).
- `build_xy4_coincidence_tiff.py` — Fig 1H overlap display: exports ImageJ composite coincidence TIFFs (also in `../overlap_image_display/`).

## Method notes
- Colocalization metric = object/voxel **Manders M1** (fraction of SEPT9 voxels on the network, 1-voxel dilation tolerance) on top-hat + Frangi + median·MAD masks; these ratios are photobleach-invariant.
- Statistics = two-tailed **Wilcoxon matched-pairs signed-rank**; central tendency = median; effect size r = |Z|/√N.
- The MT:SF **partition** ratio was **not** included in the paper; this deposit reports the Manders M1 colocalization only.
- **Reproducibility caveat:** the figures reproduce exactly from the included `data/per_cell_sf_mt.csv`. `build_faithful_population.py` implements the documented segmentation→tracking→mask→coincidence method on the deconvolved stacks; the original per-timepoint export that produced the shipped table was not retained, so numbers computed from your own data should be QC'd (mask overlays are written to `outputs/`).
- Raw/deconvolved image stacks are deposited separately (see Data availability); `RAW_DATA_MANIFEST` in the lab archive lists where they live.

## Data dictionary — `data/per_cell_sf_mt.csv`
One row per tracked cell per timepoint (178 rows). Columns:

| Column | Meaning |
|---|---|
| `region` | field-of-view id (e.g. `Jul28_XY_3_20min`); the `20min`/`30min` suffix is the dish/batch tag, **not** an interval comparison |
| `dish` | `DISH1` (20-min tag) or `DISH2` (30-min tag) |
| `uid` | cell identity within a region (`<region>#<cell>`), stable across timepoints |
| `t_min` | **nominal** time = frame index × the dish's label (DISH1: 0,20,…,160; DISH2: 0,30,…,210). A label, not real minutes. |
| `elapsed_min` | **real** minutes after CHIR, = `t_min / max(t_min for that dish) × REALEND` with `REALEND` = 220 (DISH1) / 192.5 (DISH2), anchored to the acquisition end timestamp. Both dishes = ~27.5 min per frame. **Use this for time.** |
| `MT_area`, `SF_area` | microtubule / actin-stress-fiber mask volume (voxels) |
| `S9_total`, `S9_on_MT`, `S9_on_SF` | total SEPT9 mask voxels, and those coincident with MT / SF (1-voxel tolerance) |
| `frac_MT`, `frac_SF` | Manders M1 = `S9_on_MT/S9_total` and `S9_on_SF/S9_total` (fraction of SEPT9 on each network) |

> **Why no row shows `t_min = 220`:** `t_min` is the nominal frame label; the "~220 min" endpoint in the figures is the **real elapsed** time of the 20-min dish's last frame (`t_min = 160` → `elapsed_min = 220`). The 30-min dish tops out at `elapsed_min = 192.5`. The figure scripts compute this internally; `elapsed_min` now exposes it directly.

## Data availability
- **Included here (`data/`):** `per_cell_sf_mt.csv`, the per-cell 3D voxel-Manders table behind every figure/statistic. The figures reproduce from it with `reproduce_figures.py` — **no raw images required**.
- **Not included (size):** the denoised/deconvolved live-cell z-stacks. They are in the lab archive (`NOOR_MCF_live_CHIR_SEPT9_code_and_data.zip`) and the image data repository; point `MCF_IMAGES` at them to recompute the table with `build_faithful_population.py`. Drive locations are listed in the lab archive's `RAW_DATA_MANIFEST.csv`.
