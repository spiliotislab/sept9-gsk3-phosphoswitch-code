"""Central configuration for the 02_centriole_EB1_comet_tracking analysis arm.

Live EB1-comet dynamics after nocodazole washout and their coupling to SEPT9 filaments:
comet detection/tracking, centrosomal nucleation, and the density-controlled speed analysis.
Rescue constructs WT vs S85A(AA). Dataset 073126.

Everything is resolved relative to the arm, with environment-variable overrides, so the
scripts run out of the box on any machine and can also be pointed at your own movies + ROIs.

Paths (override with environment variables):
  EB1_OUT      output directory (created)              default: ../outputs
  EB1_IMAGES   folder of processed 2-channel timelapses (ch0 = EB1, ch1 = SEPT9);
               this is what the detection engine + analysis scripts read
  EB1_RAW_ND2  folder of raw .nd2 timelapses (only the top-hat preprocessing step reads this)
  EB1_ROIS     folder of manually drawn centrosome/aster ROI .zip sets (Fiji)   default: ../rois

Acquisition (override if your data differs):
  EB1_CH_EB1 / EB1_CH_SEPT9   0-based channel indices                 (default 0 / 1)
  EB1_PX                      pixel size, micrometres                  (default 0.1076)
  EB1_DT                      frame interval, seconds                  (default 1.09)

Intermediate tables (detections, tracks, per-aster / per-comet parameter tables and the
figures) are written and re-read through out(), so the whole pipeline chains inside outputs/.

NOTE ON THE IMPORT ALIAS: nearly every script in this arm already uses a local variable
named ``C`` (a WT/AA colour dict, the channel-count ``C = s.shape[...]``, or a DataFrame),
so this module is imported as ``CFG`` rather than ``C`` to avoid shadowing it.

Usage in a script:
    import _config as CFG
    src  = CFG.require_images()          # processed movie folder
    rois = CFG.require_rois()            # manual ROI .zip folder
    df.to_csv(CFG.out("table.csv"))
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARM  = os.path.dirname(_HERE)                     # 02_centriole_EB1_comet_tracking/

def _env(key, default): return os.environ.get(key, default)

# --- output ---
OUTDIR = _env("EB1_OUT", os.path.join(_ARM, "outputs"))
os.makedirs(OUTDIR, exist_ok=True)

def out(name):
    """Absolute path inside the output directory (creates OUTDIR and any sub-directory)."""
    p = os.path.join(OUTDIR, name)
    d = os.path.dirname(p)
    if d: os.makedirs(d, exist_ok=True)
    return p

# --- movies / ROIs (analyze-your-own-data) ---
IMAGES  = _env("EB1_IMAGES",  "")                          # processed 2ch timelapses (ch0=EB1, ch1=SEPT9)
RAW_ND2 = _env("EB1_RAW_ND2", "")                          # raw .nd2 (top-hat preprocessing only)
ROIS    = _env("EB1_ROIS",    os.path.join(_ARM, "rois"))  # manual centrosome/aster ROI .zip sets

CH_EB1   = int(_env("EB1_CH_EB1",   "0"))                  # 561 = EB1
CH_SEPT9 = int(_env("EB1_CH_SEPT9", "1"))                  # 488 = SEPT9
PX = float(_env("EB1_PX", "0.1076"))                       # micrometres / pixel
DT = float(_env("EB1_DT", "1.09"))                         # seconds / frame

def require_images():
    """Return IMAGES or raise a friendly error telling the user how to set it."""
    if not IMAGES or not os.path.isdir(IMAGES):
        raise SystemExit(
            "No movie folder set. Point EB1_IMAGES at a folder of processed 2-channel timelapses "
            "(one multi-page TIFF per movie, ch0 = EB1 [561], ch1 = SEPT9 [488]), e.g.\n"
            "  Windows      : set EB1_IMAGES=C:\\path\\to\\eb1_s9tophat\n"
            "  macOS/Linux  : export EB1_IMAGES=/path/to/eb1_s9tophat\n"
            "These stacks are produced from raw .nd2 by eb1_sept9_tophat_batch.py "
            "(set EB1_RAW_ND2 + EB1_IMAGES and run it first).")
    return IMAGES

def require_rois():
    """Return ROIS or raise a friendly error. ROIs are drawn manually in Fiji."""
    if not ROIS or not os.path.isdir(ROIS):
        raise SystemExit(
            "No ROI folder found. Centrosome/aster ROIs are drawn manually in Fiji and saved as "
            "ImageJ ROI .zip sets (one .zip per movie: polygon loops for the aster region + point "
            "selections for the centriole). Point EB1_ROIS at that folder, e.g.\n"
            "  Windows      : set EB1_ROIS=C:\\path\\to\\rois_aster\n"
            "  macOS/Linux  : export EB1_ROIS=/path/to/rois_aster\n"
            f"(default: {ROIS})")
    return ROIS

def require_raw():
    """Return RAW_ND2 or raise a friendly error (top-hat preprocessing only)."""
    if not RAW_ND2 or not os.path.isdir(RAW_ND2):
        raise SystemExit(
            "No raw movie folder set. eb1_sept9_tophat_batch.py reads raw .nd2 timelapses and writes "
            "the processed 2-channel stacks the rest of the pipeline consumes. Point EB1_RAW_ND2 at "
            "the folder of .nd2 files (and EB1_IMAGES at the output folder), e.g.\n"
            "  Windows      : set EB1_RAW_ND2=C:\\path\\to\\nd2  &  set EB1_IMAGES=C:\\path\\to\\eb1_s9tophat\n"
            "  macOS/Linux  : export EB1_RAW_ND2=/path/to/nd2; export EB1_IMAGES=/path/to/eb1_s9tophat")
    return RAW_ND2

def images_out():
    """Destination for the top-hat preprocessing: EB1_IMAGES if set, else outputs/eb1_s9tophat."""
    d = IMAGES if IMAGES else out("eb1_s9tophat")
    os.makedirs(d, exist_ok=True)
    return d
