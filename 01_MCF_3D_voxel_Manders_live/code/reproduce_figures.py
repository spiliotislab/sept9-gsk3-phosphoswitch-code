"""One command to reproduce every MCF figure/statistic from the included per-cell table.

    python reproduce_figures.py

Runs each figure/stat script in turn against C.DATA_CSV and writes all outputs to C.OUTDIR
(../outputs by default). No image data or path editing needed. To use your OWN table instead,
set MCF_DATA to your CSV (see README / _config.py)."""
import os, runpy, sys, traceback
import _config as C

SCRIPTS = [
    "build_positive_responders.py",   # responder selection + companion panel + CSV
    "build_responders_first220.py",   # main 220-min endpoint figure (Manders M1)
    "build_manders_panel_stats.py",   # single Manders panel with on-plot Wilcoxon p
    "build_snapshot_220.py",          # 220-min snapshot (mean + exemplar)
    "build_timecourse_xlsx.py",       # per-timepoint time-course workbook
    "build_prism_xlsx.py",            # Prism-ready stats workbook
]
HERE = os.path.dirname(os.path.abspath(__file__))
print(f"Data table : {C.DATA_CSV}\nOutput dir : {C.OUTDIR}\n" + "-"*60)
ok = fail = 0
for s in SCRIPTS:
    print(f"\n>>> {s}")
    try:
        runpy.run_path(os.path.join(HERE, s), run_name="__main__")
        ok += 1
    except SystemExit:
        ok += 1
    except Exception:
        fail += 1
        traceback.print_exc()
print("\n" + "="*60)
print(f"done: {ok} ok, {fail} failed.  Figures & tables in: {C.OUTDIR}")
sys.exit(1 if fail else 0)
