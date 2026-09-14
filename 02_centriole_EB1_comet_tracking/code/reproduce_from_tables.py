"""Reproduce the table-based EB1 figure from the SHIPPED tables — no raw movies needed.

    python reproduce_from_tables.py

Step 1 copies the precomputed per-comet tables in ../data into the working outputs dir
(CFG.OUTDIR). Step 2 runs the analysis that reads them: `speed_outward_fig.py` reproduces the
outward-comet velocity (Figure 3F) from `aster_allcomet_persistence_comets.csv`.

The other reported quantities require the raw movies + manual ROIs (they detect/track comets
directly): `nucleation_full_073126.py` (Figure 3E, comets/min) and `aster_allcomet_persistence.py`
(Figure 3G, comet lifetime ON/OFF septin). Point EB1_IMAGES / EB1_ROIS at the movies and ROIs
(see README 'Data availability') to run those.
"""
import os, shutil, runpy, sys, traceback
import _config as CFG

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

# 1) seed outputs/ from the shipped tables (never mutate ../data)
n = 0
if os.path.isdir(DATA):
    for root, _, files in os.walk(DATA):
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, DATA)
            dst = CFG.out(rel)                       # creates sub-dirs
            if not os.path.exists(dst):
                shutil.copy2(src, dst); n += 1
    print(f"seeded {n} table files from data/ -> {CFG.OUTDIR}")
else:
    print("WARNING: ../data not found; nothing to seed")
print("-"*60)

# 2) run the table-based analysis/figure scripts. Movie/ROI-dependent scripts self-skip
#    (they raise a friendly SystemExit from CFG.require_images()/require_rois()).
SCRIPTS = [
    "speed_outward_fig.py",   # Fig 3F — outward-comet velocity, from aster_allcomet_persistence_comets.csv
]
ok = skipped = failed = 0
for s in SCRIPTS:
    p = os.path.join(HERE, s)
    if not os.path.isfile(p): continue
    print(f"\n>>> {s}")
    try:
        runpy.run_path(p, run_name="__main__"); ok += 1
    except SystemExit as e:
        msg = str(e)
        if msg and ("EB1_IMAGES" in msg or "EB1_ROIS" in msg or "movie" in msg or "ROI" in msg):
            print(f"    [skipped — needs raw movies/ROIs]"); skipped += 1
        else:
            ok += 1
    except Exception:
        failed += 1; traceback.print_exc()
print("\n"+"="*60)
print(f"done: {ok} ran, {skipped} skipped (need raw movies/ROIs), {failed} failed.")
print(f"outputs in: {CFG.OUTDIR}")
sys.exit(1 if failed else 0)
