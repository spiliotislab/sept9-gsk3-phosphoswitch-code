#!/usr/bin/env python3
"""
Subdomain S9-GSKp Pearson Analysis — GSKp Dataset
===================================================
Compute SEPTIN9-pGSK3β Pearson r by growth cone subdomain (C/T/P)
using EDT-based spatial segmentation. Same pipeline as S9-Actin subdomain.
"""

import numpy as np
import nd2
import gc as garbage_collect
import pandas as pd
from scipy import ndimage, stats
from skimage import filters, morphology, measure
from pathlib import Path
import sys, re, warnings
warnings.filterwarnings("ignore")

def log(msg):
    print(msg); sys.stdout.flush()

import _config as C
BASE = Path(C.require_images())
OUT = Path(C.out("gskp_analysis_results")); OUT.mkdir(parents=True, exist_ok=True)

PIXEL_UM = 0.0269310471755625
CH_ACT, CH_S9, CH_GSKP = C.CH_ACT, C.CH_S9, C.CH_GSKP

# ── File discovery ──
def discover_files(directory, cond_pattern, id_pattern):
    files = sorted([f for f in directory.glob("*.nd2")
                    if "Deconvolved" in f.name and not f.name.startswith("._")
                    and re.search(cond_pattern, f.name)])
    result = []; seen = set()
    for f in files:
        m = re.search(id_pattern, f.name)
        if m:
            raw_id = m.group(1)
            batch = "A" if "60m" in f.name else "B"
            cid = f"{batch}_{raw_id}"
            if cid not in seen:
                seen.add(cid); result.append((cid, f))
    return result

CHIR_FILES = discover_files(BASE, r'CHIR.*GSKp', r'688-(\d+)')
CTRL_FILES = discover_files(BASE, r'CON.*GSKp', r'688-(\d+)')
log(f"Files: {len(CTRL_FILES)} Control, {len(CHIR_FILES)} CHIR")

# ── Core functions (identical to S9-Actin subdomain pipeline) ──
def load_mip(filepath):
    with nd2.ND2File(str(filepath)) as f:
        raw = f.asarray()
    if raw.ndim == 4:
        mip = np.max(raw, axis=0).astype(np.float32)
    else:
        mip = raw.astype(np.float32)
    del raw; garbage_collect.collect()
    for c in range(mip.shape[0]):
        ch = mip[c]
        p1, p99 = np.percentile(ch, [1, 99.5])
        mip[c] = np.clip((ch - p1) / (p99 - p1 + 1e-10), 0, 1)
    return mip

def segment_neurite(mip):
    ds = 4
    act_ds = ndimage.zoom(mip[CH_ACT], 1/ds, order=1)
    combined = ndimage.gaussian_filter(act_ds, sigma=3)
    thresh = filters.threshold_otsu(combined) * 0.35
    mask_ds = combined > thresh
    mask_ds = morphology.remove_small_objects(mask_ds, min_size=500)
    mask_ds = morphology.binary_closing(mask_ds, morphology.disk(5))
    mask_ds = ndimage.binary_fill_holes(mask_ds)
    mask = ndimage.zoom(mask_ds.astype(np.float32), ds, order=0) > 0.5
    mask = mask[:mip.shape[1], :mip.shape[2]]
    if mask.shape != mip.shape[1:]:
        m = np.zeros(mip.shape[1:], dtype=bool)
        sy, sx = min(mask.shape[0], m.shape[0]), min(mask.shape[1], m.shape[1])
        m[:sy, :sx] = mask[:sy, :sx]
        mask = m
    return mask

def segment_gc_shaft(mip, mask):
    act_sm = ndimage.gaussian_filter(mip[CH_ACT], sigma=5)
    edt = ndimage.distance_transform_edt(mask) * PIXEL_UM
    labels = measure.label(mask)
    gc_mask = np.zeros_like(mask)
    shaft_mask = np.zeros_like(mask)
    for rid in range(1, labels.max()+1):
        region = labels == rid
        if region.sum() < 100: continue
        act_in = act_sm[region]; edt_in = edt[region]
        gc_pix = (act_in > np.percentile(act_in, 60)) & (edt_in > np.percentile(edt_in, 50))
        gc_r = np.zeros_like(mask); gc_r[region] = gc_pix
        sh_r = np.zeros_like(mask); sh_r[region] = ~gc_pix
        gc_mask |= gc_r; shaft_mask |= sh_r
    return gc_mask, shaft_mask, edt

def segment_gc_subdomains(mip, gc_mask):
    if gc_mask.sum() < 100:
        z = np.zeros_like(gc_mask)
        return z, z, z
    gc_edt = ndimage.distance_transform_edt(gc_mask) * PIXEL_UM
    gc_edt_vals = gc_edt[gc_mask]
    edt_p33 = np.percentile(gc_edt_vals, 33)
    edt_p66 = np.percentile(gc_edt_vals, 66)
    p_domain = gc_mask & (gc_edt <= edt_p33)
    c_domain = gc_mask & (gc_edt >= edt_p66)
    t_zone = gc_mask & (~c_domain) & (~p_domain)
    return c_domain, t_zone, p_domain

def pearson_r(a, b, mask):
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if len(x) < 20: return np.nan
    xm = x - x.mean(); ym = y - y.mean()
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    if denom < 1e-15: return np.nan
    return np.sum(xm * ym) / denom

def spearman_r(a, b, mask):
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if len(x) < 20: return np.nan
    rho, _ = stats.spearmanr(x, y)
    return rho

# ── Process all FOVs ──
log("\nProcessing all FOVs for subdomain S9-GSKp Pearson + Spearman...")
rows = []

for condition, file_list in [("Control", CTRL_FILES), ("CHIR", CHIR_FILES)]:
    log(f"\n{condition}: {len(file_list)} files")
    for idx, (cell_id, fpath) in enumerate(file_list):
        try:
            mip = load_mip(fpath)
            mask = segment_neurite(mip)
            gc_mask, shaft_mask, edt = segment_gc_shaft(mip, mask)
            c_dom, t_zone, p_dom = segment_gc_subdomains(mip, gc_mask)

            s9 = mip[CH_S9]
            gskp = mip[CH_GSKP]

            row = {
                'id': cell_id,
                'condition': condition,
            }

            # Compute Pearson and Spearman for each compartment
            for mask_name, m in [('whole', mask), ('gc', gc_mask), ('shaft', shaft_mask),
                                  ('c_domain', c_dom), ('t_zone', t_zone), ('p_domain', p_dom)]:
                row[f'{mask_name}_pearson_s9_gskp'] = pearson_r(s9, gskp, m)
                row[f'{mask_name}_spearman_s9_gskp'] = spearman_r(s9, gskp, m)
                row[f'{mask_name}_area_px'] = int(m.sum())

            rows.append(row)

            if (idx + 1) % 5 == 0 or idx == 0:
                log(f"  [{idx+1}/{len(file_list)}] {cell_id}: "
                    f"GC pear={row['gc_pearson_s9_gskp']:.3f} spear={row['gc_spearman_s9_gskp']:.3f}")

            del mip; garbage_collect.collect()

        except Exception as e:
            log(f"  [{idx+1}] {cell_id}: ERROR - {e}")
            continue

df = pd.DataFrame(rows)
csv_path = OUT / "subdomain_s9_gskp_pearson_spearman.csv"
df.to_csv(csv_path, index=False)
log(f"\nSaved: {csv_path}")
log(f"Total: {len(df)} FOVs ({df['condition'].value_counts().to_dict()})")

# ── Quick stats ──
log("\n" + "="*70)
log("RESULTS: S9-GSKp by Subdomain")
log("="*70)

ctrl = df[df['condition'] == 'Control']
chir = df[df['condition'] == 'CHIR']

for metric in ['pearson', 'spearman']:
    log(f"\n  --- {metric.upper()} ---")
    for comp, label in [
        ('whole', 'Whole Neurite'), ('gc', 'Growth Cone'),
        ('c_domain', 'C-Domain'), ('t_zone', 'T-Zone'),
        ('p_domain', 'P-Domain'), ('shaft', 'Shaft'),
    ]:
        col = f'{comp}_{metric}_s9_gskp'
        cv = ctrl[col].dropna(); hv = chir[col].dropna()
        if len(cv) > 2 and len(hv) > 2:
            u, p = stats.mannwhitneyu(cv, hv, alternative='two-sided')
            d = (hv.mean() - cv.mean()) / np.sqrt((cv.std()**2 + hv.std()**2) / 2)
            sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
            log(f"    {label:15s}: Ctrl={cv.mean():.3f}±{cv.sem():.3f}  "
                f"CHIR={hv.mean():.3f}±{hv.sem():.3f}  p={p:.4f} {sig}  d={d:.2f}")

log("\nDone!")
