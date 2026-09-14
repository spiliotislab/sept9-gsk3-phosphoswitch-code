"""Config for the overlap-image-display scripts (Fig 1F fixed U2OS, Fig 1H live MCF).
Inputs and outputs resolve relative to this folder, with environment-variable overrides.

  OVERLAP_IMAGES  folder of the source stacks:
                    - MCF (Fig 1H): the XY_4 basal/apical decon crops
                    - U2OS (Fig 1F): the .nd2 stacks (a ./nd2 subfolder is also searched)
  OVERLAP_OUT     output directory (default ../outputs)

Channel indices (override if your data differs):
  3-channel stacks   OVERLAP_CH_ACTIN / OVERLAP_CH_MT / OVERLAP_CH_S9   default 0 / 1 / 2
  (build_u2os_coincmap.py also reads CH_SEPT9 = OVERLAP_CH_S9 by name)
"""
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ARM  = os.path.dirname(_HERE)
def _env(k, d): return os.environ.get(k, d)

OUTDIR = _env("OVERLAP_OUT", os.path.join(_ARM, "outputs"))
os.makedirs(OUTDIR, exist_ok=True)
def out(name): return os.path.join(OUTDIR, name)

IMAGES   = _env("OVERLAP_IMAGES", "")
CH_ACTIN = int(_env("OVERLAP_CH_ACTIN", "0"))
CH_MT    = int(_env("OVERLAP_CH_MT",    "1"))
CH_S9    = int(_env("OVERLAP_CH_S9",    "2"))
CH_SEPT9 = CH_S9

def require_images():
    if not IMAGES or not os.path.isdir(IMAGES):
        raise SystemExit(
            "Set OVERLAP_IMAGES to the folder holding the source stacks, e.g.\n"
            "  Windows : set OVERLAP_IMAGES=C:\\path\\to\\stacks\n"
            "  macOS/Linux : export OVERLAP_IMAGES=/path/to/stacks")
    return IMAGES
