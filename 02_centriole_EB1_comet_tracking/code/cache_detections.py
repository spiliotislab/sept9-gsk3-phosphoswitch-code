"""Detect EB1 comets ONCE per aster and cache the features, so that competing LINKERS can be
compared on identical detections.

Detection is byte-for-byte the same as aster_allcomet_persistence.py (temporal-median subtraction,
trackpy.locate diameter 7 / separation 5, per-aster 93rd-percentile minmass, restricted to the
marked polygon). Only the linking step differs downstream.

Out -> data/detections/<base>__<aster>.csv  (+ hub position and pixel/time calibration in a manifest)
"""
import os, glob, warnings, numpy as np, tifffile, pandas as pd, trackpy as tp, roifile, json, time
from matplotlib.path import Path as MPath
import _config as CFG
warnings.filterwarnings("ignore"); tp.quiet()

SRC  = CFG.require_images()
AST  = CFG.require_rois()
OUT  = CFG.out("detections")
os.makedirs(OUT, exist_ok=True)
PX, DT, DIAM, SEP, MASS_PCTL = CFG.PX, CFG.DT, 7, 5, 93
LIMIT = int(os.environ.get("LIMIT", "0"))   # >0 = only this many asters (timing test)

def geno(b): return "WT" if "WT" in b else "AA"
def base_of(z):
    b = os.path.basename(z)
    for s in ["_denoisedt.zip", "_denoised.zip", ".zip"]:
        if b.endswith(s): return b[:-len(s)]
    return b
def polys(z):
    rr = roifile.roiread(z); rr = rr if isinstance(rr, list) else [rr]; P, Q = [], []
    for r in rr:
        c = np.asarray(r.coordinates(), float)
        if int(r.roitype) == 10 or len(c) <= 2: Q.append(c.mean(0) if len(c) > 1 else c[0])
        else: P.append(c)
    return P, Q

manifest, n_done, t0 = [], 0, time.time()
for z in sorted(glob.glob(os.path.join(AST, "*.zip"))):
    base = base_of(z); g = geno(base); src = os.path.join(SRC, base + "_denoised.tif")
    if not os.path.exists(src): continue
    tf = tifffile.TiffFile(src); s = tf.series[0]
    C = s.shape[s.axes.index('C')]; T = s.shape[s.axes.index('T')]; H, W = s.shape[-2:]
    if (H, W) != (2304, 2304): tf.close(); continue
    _polys, _pts = polys(z)
    for ai, poly in enumerate(_polys):
        out_csv = os.path.join(OUT, f"{base}__{ai}.csv")
        pth = MPath(poly)
        x0 = max(0, int(poly[:, 0].min()) - 8); x1 = min(W, int(poly[:, 0].max()) + 8)
        y0 = max(0, int(poly[:, 1].min()) - 8); y1 = min(H, int(poly[:, 1].max()) + 8)
        Hc, Wc = y1 - y0, x1 - x0
        yy, xx = np.mgrid[y0:y1, x0:x1]
        loop = pth.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(Hc, Wc)
        if loop.sum() < 80: continue
        pin = [p for p in _pts if pth.contains_point((p[0], p[1]))]
        hx, hy = (pin[0] - [x0, y0] if pin else poly.mean(0) - [x0, y0])
        if os.path.exists(out_csv):
            manifest.append(dict(base=base, aster=ai, geno=g, hx=float(hx), hy=float(hy), T=int(T)))
            continue
        eb1 = np.stack([tf.pages[t * C + CFG.CH_EB1].asarray()[y0:y1, x0:x1] for t in range(T)]).astype(np.float32)
        eb1 = np.clip(eb1 - np.median(eb1, axis=0)[None], 0, None)
        mm = float(np.percentile(eb1[:, loop][eb1[:, loop] > 0], MASS_PCTL))
        fl = []
        for t in range(T):
            f = tp.locate(eb1[t], DIAM, minmass=mm, separation=SEP, engine="python")
            if len(f):
                ins = loop[np.clip(f.y.round().astype(int), 0, Hc - 1),
                           np.clip(f.x.round().astype(int), 0, Wc - 1)]
                f = f[ins]
                if len(f): fl.append(f.assign(frame=t))
        if len(fl) < 3: continue
        det = pd.concat(fl, ignore_index=True)[["x", "y", "frame", "mass"]]
        det.to_csv(out_csv, index=False)
        manifest.append(dict(base=base, aster=ai, geno=g, hx=float(hx), hy=float(hy), T=int(T)))
        n_done += 1
        print(f"  {g} {base[9:]:24s} aster{ai}  {len(det):6d} detections  "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
        if LIMIT and n_done >= LIMIT: break
    tf.close()
    if LIMIT and n_done >= LIMIT: break

with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(dict(px=PX, dt=DT, diam=DIAM, sep=SEP, mass_pctl=MASS_PCTL, asters=manifest), fh, indent=1)
print(f"\ncached {n_done} new asters, {len(manifest)} total in manifest, {time.time()-t0:.0f}s")
