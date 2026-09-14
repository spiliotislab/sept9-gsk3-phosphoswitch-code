"""Central configuration for the pGSK / CHIR fixed growth-cone colocalization arm.

Everything is resolved relative to the repository, with environment-variable overrides,
so the scripts run out of the box on any machine and can also be pointed at your own data.

Paths (override with environment variables):
  PGSK_OUT      output directory (created)   default: ../outputs
  PGSK_IMAGES   folder of the 3-channel fixed z-stacks / .nd2 files
                (031826 DIV3 dataset: actin / SEPT9 / acetyl-tubulin-or-pGSK3b)
  PGSK_IMAGES2  folder of the 4-channel single-plane crops
                (01326 dataset: tubulin / SEPT9 / pGSK3b / actin) used by the
                triple-colocalization and Costes scripts only

Acquisition channel indices (override if your data differs — read straight from the
scripts' own inline values, not guessed):
  3-channel datasets (actin=0, SEPT9=1, third=2):
    PGSK_CH_ACT / PGSK_CH_S9 / PGSK_CH_TUB / PGSK_CH_GSKP   default 0 / 1 / 2 / 2
  4-channel single-plane crops (tub=0, SEPT9=1, pGSK=2, actin=3):
    PGSK_CH4_TUB / PGSK_CH4_S9 / PGSK_CH4_GSK / PGSK_CH4_ACT   default 0 / 1 / 2 / 3

Usage in a script:
    import _config as C
    BASE = pathlib.Path(C.require_images())
    OUT  = pathlib.Path(C.out("gc_analysis_results")); OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / "figure.png")
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARM  = os.path.dirname(_HERE)                     # 04_neuron_GC_Pearson/

def _env(key, default): return os.environ.get(key, default)

# --- output ---
OUTDIR = _env("PGSK_OUT", os.path.join(_ARM, "outputs"))
os.makedirs(OUTDIR, exist_ok=True)

def out(name):
    """Absolute path inside the output directory (creates OUTDIR and any sub-directory).

    Pass a plain basename ("gc_shaft_metrics.csv") for a file, a subdirectory
    name ("gc_analysis_results") that scripts then populate, or a nested path
    ("gc_analysis_results/gc_shaft_metrics.csv"): the parent directory is created
    for you. Scripts that read a table an earlier script wrote resolve it through
    the same call, so the whole pipeline chains inside outputs/.
    """
    p = os.path.join(OUTDIR, name)
    d = os.path.dirname(p)
    if d: os.makedirs(d, exist_ok=True)
    return p

# --- image inputs (analyze-your-own-data) ---
IMAGES  = _env("PGSK_IMAGES",  "")                 # empty => scripts print how to set it
IMAGES2 = _env("PGSK_IMAGES2", "")                 # 4-channel single-plane crops dataset

# 3-channel fixed datasets: actin / SEPT9 / (acetyl-tubulin or pGSK3b)
CH_ACT  = int(_env("PGSK_CH_ACT",  "0"))
CH_S9   = int(_env("PGSK_CH_S9",   "1"))
CH_TUB  = int(_env("PGSK_CH_TUB",  "2"))           # acetyl-tubulin dataset
CH_GSKP = int(_env("PGSK_CH_GSKP", "2"))           # pGSK3b dataset

# 4-channel single-plane crops: tubulin / SEPT9 / pGSK3b / actin
CH4_TUB = int(_env("PGSK_CH4_TUB", "0"))
CH4_S9  = int(_env("PGSK_CH4_S9",  "1"))
CH4_GSK = int(_env("PGSK_CH4_GSK", "2"))
CH4_ACT = int(_env("PGSK_CH4_ACT", "3"))

def _require(path, var, what):
    if not path or not os.path.isdir(path):
        raise SystemExit(
            f"No image folder set. Point {var} at {what}, e.g.\n"
            f"  Windows     : set {var}=C:\\path\\to\\stacks\n"
            f"  macOS/Linux : export {var}=/path/to/stacks\n"
            "Also set the channel indices (PGSK_CH_*) if your acquisition order differs.")
    return path

def require_images():
    """Return PGSK_IMAGES or raise a friendly error telling the user how to set it."""
    return _require(IMAGES, "PGSK_IMAGES",
                    "the folder of 3-channel fixed .nd2 stacks (actin / SEPT9 / acetyl-tubulin or pGSK3b)")

def require_images2():
    """Return PGSK_IMAGES2 (4-channel single-plane crops) or a friendly error."""
    return _require(IMAGES2, "PGSK_IMAGES2",
                    "the folder of 4-channel single-plane crops (tubulin / SEPT9 / pGSK3b / actin)")
