"""Central configuration for the MCF live-CHIR SEPT9 analysis package.

Everything is resolved relative to the repository, with environment-variable overrides,
so the scripts run out of the box on any machine and can also be pointed at your own data.

Paths (override with environment variables):
  MCF_DATA    input per-cell table          default: ../data/per_cell_sf_mt.csv
  MCF_OUT     output directory (created)     default: ../outputs
  MCF_IMAGES  folder of denoised/deconvolved multi-channel z-stacks (for the pipeline)

Acquisition (override if your data differs):
  MCF_CH_ACTIN / MCF_CH_MT / MCF_CH_S9   0-based channel indices  (default 0 / 1 / 2)
  MCF_VOXEL_XY / MCF_VOXEL_Z             micrometres              (default 0.108 / 0.30)

Usage in a script:
    import _config as C
    df = pandas.read_csv(C.DATA_CSV)
    plt.savefig(C.out("figure.png"))
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARM  = os.path.dirname(_HERE)                     # 01_MCF_3D_voxel_Manders_live/

def _env(key, default): return os.environ.get(key, default)

# --- data / output ---
DATA_CSV = _env("MCF_DATA", os.path.join(_ARM, "data", "per_cell_sf_mt.csv"))
OUTDIR   = _env("MCF_OUT",  os.path.join(_ARM, "outputs"))
os.makedirs(OUTDIR, exist_ok=True)

def out(name):
    """Absolute path inside the output directory (creates OUTDIR on import)."""
    return os.path.join(OUTDIR, name)

# --- image pipeline (analyze-your-own-data) ---
IMAGES   = _env("MCF_IMAGES", "")                  # empty => pipeline prints how to set it
CH_ACTIN = int(_env("MCF_CH_ACTIN", "0"))
CH_MT    = int(_env("MCF_CH_MT",    "1"))
CH_S9    = int(_env("MCF_CH_S9",    "2"))
VOXEL_XY = float(_env("MCF_VOXEL_XY", "0.108"))
VOXEL_Z  = float(_env("MCF_VOXEL_Z",  "0.30"))

def require_images():
    """Return IMAGES or raise a friendly error telling the user how to set it."""
    if not IMAGES or not os.path.isdir(IMAGES):
        raise SystemExit(
            "No image folder set. Point MCF_IMAGES at a folder of denoised/deconvolved "
            "multi-channel z-stacks (one TIFF per cell per timepoint), e.g.\n"
            "  Windows : set MCF_IMAGES=C:\\path\\to\\stacks\n"
            "  macOS/Linux : export MCF_IMAGES=/path/to/stacks\n"
            "Also set channel indices (MCF_CH_ACTIN/MT/S9) and voxel size (MCF_VOXEL_XY/Z) if your data differs.")
    return IMAGES
