"""Central configuration for the U2OS fixed-CHIR SEPT9 analysis package.

Everything is resolved relative to the repository, with environment-variable overrides,
so the scripts run out of the box on any machine and can be pointed at your own images.

This arm ships NO data table — it is "analyze your own data" only. Point U2OS_IMAGES at a
folder of Nikon .nd2 stacks, set the channel indices / voxel size if your acquisition differs,
then run each script; all outputs land in ../outputs (created on first import).

Paths (override with environment variables):
  U2OS_IMAGES  folder of .nd2 image stacks (pipeline input)   default: (unset)
  U2OS_OUT     output directory (created)                     default: ../outputs

Acquisition (override if your data differs):
  U2OS_CH_ACTIN / U2OS_CH_SEPT9 / U2OS_CH_MT   0-based channel indices  (default 0 / 1 / 2)
      -> matches the 3-channel stress-fiber stacks: 640=actin, 561=SEPT9, 488=MT
      -> NOTE: sept9_mt_coloc_3d.py reads a *different* 2-channel MT/SEPT9 acquisition
         (ch0=561 tubulin, ch1=488 SEPT9) and keeps its own indices / --mt-ch / --sep-ch flags.
  U2OS_VOXEL_XY / U2OS_VOXEL_Z                 micrometres              (default 0.0384 / 0.2)
      -> sept9_mt_coloc_3d.py reads the true voxel size from each .nd2; these are only used
         where a script needs a fallback / for reference.

Usage in a script:
    import _config as C
    folder = C.require_images()
    plt.savefig(C.out("figure.png"))
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARM  = os.path.dirname(_HERE)                     # 03_U2OS_fixed_3D_voxel_Manders/

def _env(key, default): return os.environ.get(key, default)

# --- output ---
OUTDIR = _env("U2OS_OUT", os.path.join(_ARM, "outputs"))
os.makedirs(OUTDIR, exist_ok=True)

def out(name):
    """Absolute path inside the output directory (creates OUTDIR on import)."""
    return os.path.join(OUTDIR, name)

# --- image input (analyze-your-own-data) ---
IMAGES     = _env("U2OS_IMAGES", "")               # empty => scripts print how to set it
CH_ACTIN   = int(_env("U2OS_CH_ACTIN", "0"))
CH_SEPT9   = int(_env("U2OS_CH_SEPT9", "1"))
CH_MT      = int(_env("U2OS_CH_MT",    "2"))
VOXEL_XY   = float(_env("U2OS_VOXEL_XY", "0.0384"))
VOXEL_Z    = float(_env("U2OS_VOXEL_Z",  "0.2"))

def require_images():
    """Return IMAGES or raise a friendly error telling the user how to set it."""
    if not IMAGES or not os.path.isdir(IMAGES):
        raise SystemExit(
            "No image folder set. Point U2OS_IMAGES at a folder of Nikon .nd2 stacks, e.g.\n"
            "  Windows (PowerShell): $env:U2OS_IMAGES=\"C:\\path\\to\\nd2\"\n"
            "  Windows (cmd)       : set U2OS_IMAGES=C:\\path\\to\\nd2\n"
            "  macOS/Linux         : export U2OS_IMAGES=/path/to/nd2\n"
            "Also set channel indices (U2OS_CH_ACTIN/SEPT9/MT) and voxel size "
            "(U2OS_VOXEL_XY/Z) if your acquisition differs.")
    return IMAGES
