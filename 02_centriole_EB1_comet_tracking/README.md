# 02_centriole_EB1_comet_tracking — centriole EB1 comet tracking (nocodazole washout, 073126)

Code behind the paper's live EB1-dsRed comet analysis (**Figure 3E–G**): MDCK-EB1-dsRed cells
co-expressing GFP-SEPT9_i1 (**wild type vs S82A/S85A**), imaged during microtubule regrowth after
nocodazole washout. Three reported quantities, per centrosome region (WT n = 26, S82A/S85A n = 28):

- **Figure 3E** — EB1 comets emerging from the centrosome **per minute** (nucleation rate).
- **Figure 3F** — outward comet **velocity**.
- **Figure 3G** — comet **lifetime ON vs OFF** SEPT9 filaments.

Channels: **561 = EB1-dsRed**, **488 = GFP-SEPT9_i1**. Centrosomal regions are drawn by hand in Fiji.
Comet detection/tracking uses Trackpy. (Astral MT length/number in Fig 3C–D were measured by manual
Fiji tracing, not by this code.)

```
02_centriole_EB1_comet_tracking/
  code/      detection + the 3 reported analyses, _config.py, reproduce_from_tables.py
  data/      per-comet result tables:
               aster_allcomet_persistence_comets.csv  (Fig 3G lifetime + speed input for 3F)
               nucleation_full_073126.csv              (Fig 3E comets/min)
  rois/      manual centrosome ROI .zip sets (supply your own; override with EB1_ROIS)
  outputs/   created on first run
```

## Install (once)
```bash
pip install -r ../requirements.txt
```
Python 3.11+ (developed on 3.14). Key packages: `trackpy`, `tifffile`, `roifile`, `nd2`,
`scikit-image`, `scipy`, `matplotlib`. The top-hat preprocessing step also uses `torch` (GPU optional).

## A. Reproduce from the shipped table (no movies needed)
```bash
python reproduce_from_tables.py
```
Seeds `outputs/` from `data/` and runs `speed_outward_fig.py`, reproducing the **Figure 3F** outward
velocity from `aster_allcomet_persistence_comets.csv`. (Figures 3E and 3G are produced by scripts that
detect comets directly from the movies — see B.)

## B. Analyze your own data (movies + ROIs)
**1. Supply manual ROIs.** Centrosome regions are drawn by hand in Fiji (one ImageJ ROI `.zip` per
movie: a polygon around each centrosome-centered GFP-SEPT9 aster, plus a point at the centriole).
Point `EB1_ROIS` at that folder (default `../rois`).

**2. Point at your movies and run.** Set the two paths, then run the scripts from `code/`.

Windows (PowerShell):
```powershell
# optional preprocessing: raw .nd2 -> 2-channel stacks (EB1 raw + SEPT9 median-3 + white top-hat)
$env:EB1_RAW_ND2="C:\path\to\nd2"; $env:EB1_IMAGES="C:\path\to\eb1_s9tophat"
py eb1_sept9_tophat_batch.py

$env:EB1_IMAGES="C:\path\to\eb1_s9tophat"; $env:EB1_ROIS="C:\path\to\rois"
```
macOS / Linux (bash):
```bash
# optional preprocessing: raw .nd2 -> 2-channel stacks (EB1 raw + SEPT9 median-3 + white top-hat)
export EB1_RAW_ND2=/path/to/nd2 EB1_IMAGES=/path/to/eb1_s9tophat
python eb1_sept9_tophat_batch.py

export EB1_IMAGES=/path/to/eb1_s9tophat EB1_ROIS=/path/to/rois
```
then, with the environment set (either OS):
```bash
python cache_detections.py            # Trackpy comet detection per centrosome -> outputs/detections/
python nucleation_full_073126.py      # Fig 3E — comets/min at the centriole
python aster_allcomet_persistence.py  # Fig 3G — per-comet lifetime ON/OFF septin (writes the table 3F reads)
python speed_outward_fig.py           # Fig 3F — outward-comet velocity
```
(`EB1_IMAGES` = a folder of processed 2-channel timelapses, one multi-page TIFF per movie,
`ch0 = EB1`, `ch1 = SEPT9`.)

### Configuration (environment variables — all optional)
| Variable | Meaning | Default |
|---|---|---|
| `EB1_OUT` | output directory | `../outputs` |
| `EB1_IMAGES` | folder of processed 2-channel timelapses (`ch0=EB1`, `ch1=SEPT9`) | *(unset)* |
| `EB1_RAW_ND2` | folder of raw `.nd2` (top-hat preprocessing step only) | *(unset)* |
| `EB1_ROIS` | folder of manual centrosome ROI `.zip` sets | `../rois` |
| `EB1_CH_EB1` / `EB1_CH_SEPT9` | 0-based channel indices | `0` / `1` |
| `EB1_PX` | pixel size (µm) | `0.1076` |
| `EB1_DT` | frame interval (s) | `1.09` |

Defaults live in `code/_config.py`, imported as `CFG` (scripts use a local `C`, so the module is
aliased to avoid shadowing). Intermediate files chain through `CFG.out(...)` inside `outputs/`.

## Scripts (code/)
- `eb1_sept9_tophat_batch.py` — preprocessing: raw `.nd2` → 2-channel stacks (EB1 raw + SEPT9 median-3 + white top-hat; GPU optional).
- `cache_detections.py` — EB1 comet detection with Trackpy inside each hand-drawn centrosome ROI → `detections/*.csv`.
- `nucleation_full_073126.py` — **Fig 3E**: comet births near the centriole (≤ 10 px), rate per minute; WT vs S82A/S85A.
- `aster_allcomet_persistence.py` — **Fig 3G**: per-comet lifetime and ON/OFF-septin classification along each track → `aster_allcomet_persistence_comets.csv`.
- `speed_outward_fig.py` — **Fig 3F**: mean speed of outward (astral) comets, from the per-comet table above.

## Method notes
- **Detection:** per-pixel temporal-median subtraction, then `trackpy.locate` restricted to the hand-drawn centrosome polygon; SEPT9 is prepared as a median-3 + white top-hat channel (diffuse cytoplasm removed, filaments kept).
- **Nucleation (3E):** a nucleation event is the first particle of a linked track appearing within R = 10 px (1.08 µm) of the marked centriole; rate = comets / time-lapse duration.
- **Velocity (3F):** outward comets (radial gain ≥ 2 px from the centriole) speed = path length / elapsed time (nm s⁻¹); each track is one growth event (growth-phase speeds, not joined into compound tracks).
- **Lifetime ON/OFF septin (3G):** per-comet lifetime split by whether the track colocalizes with a GFP-SEPT9 filament (top-hat SEPT9 along the track).
- **Statistics** (reported in the paper): Welch's t-test for 3E/3F; Kruskal-Wallis + Dunn's for 3G. Inferential statistics were computed in GraphPad Prism.

## Data availability
- **Included here (`data/`):** the per-comet result tables behind Fig 3E–G — `aster_allcomet_persistence_comets.csv` (lifetime/ON-OFF septin for 3G; also the input for 3F) and `nucleation_full_073126.csv` (comets/min for 3E). Figure 3F reproduces from these via `reproduce_from_tables.py`.
- **Not included (size):** the raw/processed movie stacks and the manual centrosome ROI `.zip` sets — needed to recompute 3E and 3G from scratch. They are in the lab archive (`NOOR_073126_code_for_submission.zip`) and the image data repository; supply them via `EB1_IMAGES` / `EB1_ROIS`. Drive locations are in the lab archive's `RAW_DATA_MANIFEST.csv`.

> **Scope:** this deposit contains only the analyses reported in the paper (Fig 3E–G). Reviewer-response
> and robustness analyses (alternate linkers, density-adjusted ANCOVA of speed, crowding controls, aster
> morphology) are retained in the lab archive, not here.
