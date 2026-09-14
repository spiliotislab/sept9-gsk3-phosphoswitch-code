#!/usr/bin/env python3
"""
CHIR (GSK3 inhibitor) vs Control — Growth Cone & Shaft Analysis
================================================================
DIV3 neurons, SoRa super-resolution, deconvolved, single-plane.
3 channels: GSKp-488, SEPTIN9-549, Phalloidin/Actin-647
GSKp dataset: CHIR vs Control

Analysis:
  1. Neurite segmentation → growth cone vs shaft identification
  2. GC morphology (area, perimeter, circularity, filopodia)
  3. Per-compartment channel intensity distributions
  4. Pairwise colocalization (Pearson, Manders, ICQ)
  5. Binary overlap & Venn fractions
  6. SEPTIN9 puncta detection per compartment
  7. Filament density (tubeness) per compartment
  8. Radial distribution within growth cone
  9. Statistical comparison: Control vs CHIR
"""

import numpy as np
import nd2
import gc as garbage_collect
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import ndimage, stats
from skimage import filters, morphology, measure, feature
from skimage.filters import sato
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from pathlib import Path
import sys
import warnings; warnings.filterwarnings("ignore")

def log(msg):
    print(msg)
    sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════════════
# PATHS & PARAMETERS
# ══════════════════════════════════════════════════════════════════════════
import _config as C
BASE = Path(C.require_images())
OUT = Path(C.out("gskp_analysis_results")); OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)

PIXEL_UM = 0.0269310471755625  # 27 nm SoRa

# Channel order: 0=640(Actin), 1=561(SEPTIN9), 2=488(GSKpulin)
CH_ACT, CH_S9, CH_GSKP = C.CH_ACT, C.CH_S9, C.CH_GSKP
CH_NAMES = ['Actin', 'SEPTIN9', 'GSKp']

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "figure.dpi": 200})

# Auto-discover all deconvolved files by condition
# Two batches exist with overlapping cell IDs:
#   Batch A: "60m" in name (e.g., CHIR-3uM-60m-GSKp-488-SEPT9-549-phall-688-XXXX)
#   Batch B: no "60m"  (e.g., CHIR-3uM-GSKp-488-SEPT9-phall-688-XXXX)
# We prefix IDs with A_/B_ to keep them unique.
import re

def discover_files_both_batches(directory, cond_pattern, id_pattern):
    """Find deconvolved .nd2 files for both batches, with unique IDs."""
    files = sorted([f for f in directory.glob("*.nd2")
                    if "Deconvolved" in f.name and not f.name.startswith("._")
                    and re.search(cond_pattern, f.name)])
    result = []
    seen = set()
    for f in files:
        m = re.search(id_pattern, f.name)
        if m:
            raw_id = m.group(1)
            batch = "A" if "60m" in f.name else "B"
            cid = f"{batch}_{raw_id}"
            if cid not in seen:
                seen.add(cid)
                result.append((cid, f))
    return result

CHIR_FILES = discover_files_both_batches(BASE, r'CHIR.*GSKp', r'688-(\d+)')
CTRL_FILES = discover_files_both_batches(BASE, r'CON.*GSKp', r'688-(\d+)')

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def normalize_2d(img):
    p2, p99 = np.percentile(img, [2, 99.5])
    return np.clip((img - p2) / (p99 - p2 + 1e-10), 0, 1)

def pearson_r(a, b, mask):
    va = a[mask].astype(np.float64)
    vb = b[mask].astype(np.float64)
    if len(va) < 50:
        return np.nan
    va -= va.mean(); vb -= vb.mean()
    denom = np.sqrt((va**2).sum() * (vb**2).sum())
    return (va * vb).sum() / (denom + 1e-10)

def manders_coeff(a, b, mask):
    va = a[mask].astype(np.float64)
    vb = b[mask].astype(np.float64)
    if len(va) < 50:
        return np.nan, np.nan
    ta = filters.threshold_otsu(va)
    tb = filters.threshold_otsu(vb)
    M1 = va[vb > tb].sum() / (va.sum() + 1e-10)
    M2 = vb[va > ta].sum() / (vb.sum() + 1e-10)
    return M1, M2

def icq(a, b, mask):
    va = a[mask].astype(np.float64)
    vb = b[mask].astype(np.float64)
    if len(va) < 50:
        return np.nan
    da = va - va.mean(); db = vb - vb.mean()
    return (np.sum(da * db > 0) / len(da)) - 0.5

def segment_neurites_and_gc(act, tub, s9, downsample=4):
    """
    Segment neurites and identify growth cones vs shaft.

    Strategy:
    1. Downsample for speed (27nm pixels → ~108nm)
    2. Segment all neurite signal using combined channels
    3. Label connected components
    4. For each neurite: find the tip (growth cone) using actin-rich, wide region
    5. Growth cone = actin-enriched bulbous tip
    6. Shaft = tubulin-enriched thin segment
    """
    ny, nx = act.shape

    # Downsample for segmentation
    act_ds = act[::downsample, ::downsample]
    gskp_ds = tub[::downsample, ::downsample]
    s9_ds = s9[::downsample, ::downsample]
    ny_ds, nx_ds = act_ds.shape

    # Combined signal for neurite detection
    # NOTE: use S9 instead of GSKp for segmentation — GSKp has high background
    act_sm = ndimage.gaussian_filter(act_ds, 3)
    s9_sm = ndimage.gaussian_filter(s9_ds, 3)
    gskp_sm = ndimage.gaussian_filter(gskp_ds, 3)
    combined = act_sm + 0.5 * s9_sm

    # Threshold
    th = filters.threshold_otsu(combined)
    neurite_mask = combined > th * 0.35
    neurite_mask = morphology.remove_small_objects(neurite_mask, 100)
    neurite_mask = ndimage.binary_fill_holes(neurite_mask)

    # Label individual neurites
    lbl, n_obj = ndimage.label(neurite_mask)

    # For each connected component, identify growth cone vs shaft
    gc_mask_ds = np.zeros_like(neurite_mask)
    shaft_mask_ds = np.zeros_like(neurite_mask)

    all_gcs = []

    for obj_id in range(1, n_obj + 1):
        obj_mask = lbl == obj_id
        obj_area = obj_mask.sum()
        if obj_area < 50:  # too small
            continue

        # Get object properties
        props = measure.regionprops(obj_mask.astype(int))
        if not props:
            continue

        # Actin/tubulin ratio map within this object
        act_obj = act_sm * obj_mask
        tub_obj = gskp_sm * obj_mask

        # Growth cone detection:
        # GC = region where actin is high AND structure is wide
        # Shaft = thin, tubulin-dominant

        # Use distance transform to find "thick" parts
        edt_obj = ndimage.distance_transform_edt(obj_mask)

        # Actin enrichment relative to tubulin
        ratio = np.where(tub_obj > 0, act_obj / (tub_obj + 1e-5), 0) * obj_mask

        # GC criteria: actin-rich + wide (high EDT) region
        # Use adaptive thresholds within this object
        act_vals = act_obj[obj_mask]
        if len(act_vals) < 20:
            continue

        act_high = act_obj > np.percentile(act_vals, 60)
        edt_vals = edt_obj[obj_mask]
        edt_high = edt_obj > np.percentile(edt_vals, 50)

        gc_region = act_high & edt_high & obj_mask
        gc_region = morphology.binary_dilation(gc_region, morphology.disk(3))
        gc_region = gc_region & obj_mask
        gc_region = morphology.remove_small_objects(gc_region, 30)

        shaft_region = obj_mask & ~gc_region

        gc_mask_ds |= gc_region
        shaft_mask_ds |= shaft_region

        # Store GC info
        gc_lbl, n_gc = ndimage.label(gc_region)
        for gc_id in range(1, n_gc + 1):
            gc_single = gc_lbl == gc_id
            if gc_single.sum() < 20:
                continue
            gc_props = measure.regionprops(gc_single.astype(int))
            if gc_props:
                all_gcs.append({
                    'mask_ds': gc_single,
                    'area_ds': gc_single.sum(),
                    'props': gc_props[0],
                })

    # Upscale masks back to original resolution
    gc_mask_full = np.kron(gc_mask_ds, np.ones((downsample, downsample), dtype=bool))[:ny, :nx]
    shaft_mask_full = np.kron(shaft_mask_ds, np.ones((downsample, downsample), dtype=bool))[:ny, :nx]
    neurite_mask_full = np.kron(neurite_mask, np.ones((downsample, downsample), dtype=bool))[:ny, :nx]

    return neurite_mask_full, gc_mask_full, shaft_mask_full, all_gcs

def detect_puncta(img_raw, mask, min_sigma=2, max_sigma=8, threshold=0.015):
    img_n = normalize_2d(img_raw * mask)
    blobs = feature.blob_dog(img_n, min_sigma=min_sigma, max_sigma=max_sigma,
                              threshold=threshold)
    puncta = []
    for b in blobs:
        y, x, sigma = b
        iy, ix = int(y), int(x)
        if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]:
            puncta.append({'y': y, 'x': x, 'sigma': sigma,
                          'intensity': img_n[iy, ix],
                          'radius_um': sigma * PIXEL_UM * np.sqrt(2)})
    return puncta

# ══════════════════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════
log("=" * 70)
log("CHIR vs CONTROL — GROWTH CONE & SHAFT ANALYSIS")
log("=" * 70)
log(f"  CHIR: {len(CHIR_FILES)} cells, Control: {len(CTRL_FILES)} cells")

all_metrics = []

def process_cell(cell_id, filepath, condition):
    metrics = {'id': cell_id, 'condition': condition}

    try:
        with nd2.ND2File(str(filepath)) as fh:
            data = fh.asarray().astype(np.float32)  # (C, Y, X)

        act_raw = data[CH_ACT]
        s9_raw = data[CH_S9]
        gskp_raw = data[CH_GSKP]
        del data

        act = normalize_2d(act_raw)
        s9 = normalize_2d(s9_raw)
        tub = normalize_2d(gskp_raw)
        ny, nx = act.shape

        # Segment neurites → GC vs shaft
        neurite_mask, gc_mask, shaft_mask, gcs = segment_neurites_and_gc(act, tub, s9)

        gc_area = gc_mask.sum() * PIXEL_UM**2
        shaft_area = shaft_mask.sum() * PIXEL_UM**2
        neurite_area = neurite_mask.sum() * PIXEL_UM**2

        metrics['neurite_area_um2'] = neurite_area
        metrics['gc_area_um2'] = gc_area
        metrics['shaft_area_um2'] = shaft_area
        metrics['gc_fraction'] = gc_area / (neurite_area + 1e-10)
        metrics['n_growth_cones'] = len(gcs)

        if neurite_area < 1:
            log(f"    SKIP {cell_id}: neurite area too small ({neurite_area:.1f} µm²)")
            return None

        # ── GC morphology ──
        gc_lbl_full, n_gc_full = ndimage.label(gc_mask)
        gc_areas = []
        gc_perimeters = []
        gc_circularities = []
        gc_solidities = []

        for gc_id in range(1, n_gc_full + 1):
            gc_single = gc_lbl_full == gc_id
            gc_area_single = gc_single.sum() * PIXEL_UM**2
            if gc_area_single < 0.5:  # skip tiny fragments
                continue
            props = measure.regionprops(gc_single.astype(int))
            if props:
                p = props[0]
                perim = p.perimeter * PIXEL_UM
                circ = 4 * np.pi * gc_area_single / (perim**2 + 1e-10)
                gc_areas.append(gc_area_single)
                gc_perimeters.append(perim)
                gc_circularities.append(circ)
                gc_solidities.append(p.solidity)

        if gc_areas:
            metrics['gc_mean_area'] = np.mean(gc_areas)
            metrics['gc_mean_perimeter'] = np.mean(gc_perimeters)
            metrics['gc_mean_circularity'] = np.mean(gc_circularities)
            metrics['gc_mean_solidity'] = np.mean(gc_solidities)
        else:
            metrics['gc_mean_area'] = 0
            metrics['gc_mean_perimeter'] = 0
            metrics['gc_mean_circularity'] = 0
            metrics['gc_mean_solidity'] = 0

        # ── Per-compartment intensities ──
        for comp_name, comp_mask in [('gc', gc_mask), ('shaft', shaft_mask), ('whole', neurite_mask)]:
            if comp_mask.sum() < 50:
                for ch_name in ['act', 's9', 'tub']:
                    metrics[f'{comp_name}_{ch_name}_mean'] = np.nan
                    metrics[f'{comp_name}_{ch_name}_cv'] = np.nan
                continue

            for ch_name, ch_img in [('act', act), ('s9', s9), ('tub', tub)]:
                vals = ch_img[comp_mask]
                metrics[f'{comp_name}_{ch_name}_mean'] = vals.mean()
                metrics[f'{comp_name}_{ch_name}_cv'] = vals.std() / (vals.mean() + 1e-10)

        # ── Colocalization per compartment ──
        pairs = [('s9', 'act', s9, act), ('s9', 'tub', s9, tub), ('act', 'tub', act, tub)]
        for comp_name, comp_mask in [('gc', gc_mask), ('shaft', shaft_mask), ('whole', neurite_mask)]:
            if comp_mask.sum() < 100:
                for a_name, b_name, _, _ in pairs:
                    metrics[f'{comp_name}_pearson_{a_name}_{b_name}'] = np.nan
                    metrics[f'{comp_name}_icq_{a_name}_{b_name}'] = np.nan
                continue
            for a_name, b_name, a_img, b_img in pairs:
                metrics[f'{comp_name}_pearson_{a_name}_{b_name}'] = pearson_r(a_img, b_img, comp_mask)
                metrics[f'{comp_name}_icq_{a_name}_{b_name}'] = icq(a_img, b_img, comp_mask)
                m1, m2 = manders_coeff(a_img, b_img, comp_mask)
                metrics[f'{comp_name}_manders_{a_name}_on_{b_name}'] = m1
                metrics[f'{comp_name}_manders_{b_name}_on_{a_name}'] = m2

        # ── Binary overlap (Venn) per compartment ──
        for comp_name, comp_mask in [('gc', gc_mask), ('shaft', shaft_mask)]:
            if comp_mask.sum() < 100:
                continue
            s9_th = filters.threshold_otsu(s9[comp_mask])
            act_th = filters.threshold_otsu(act[comp_mask])
            tub_th = filters.threshold_otsu(tub[comp_mask])
            s9_bin = (s9 > s9_th) & comp_mask
            act_bin = (act > act_th) & comp_mask
            tub_bin = (tub > tub_th) & comp_mask
            n_comp = comp_mask.sum()

            # Jaccard
            metrics[f'{comp_name}_jaccard_s9_act'] = (s9_bin & act_bin).sum() / ((s9_bin | act_bin).sum() + 1e-10)
            metrics[f'{comp_name}_jaccard_s9_tub'] = (s9_bin & tub_bin).sum() / ((s9_bin | tub_bin).sum() + 1e-10)
            metrics[f'{comp_name}_frac_all_three'] = (s9_bin & act_bin & tub_bin).sum() / n_comp
            metrics[f'{comp_name}_frac_s9_only'] = (s9_bin & ~act_bin & ~tub_bin).sum() / n_comp

        # ── S9 puncta per compartment ──
        # Downsample for puncta detection (full res too slow)
        ds = 2
        s9_ds = s9_raw[::ds, ::ds]
        gc_ds = gc_mask[::ds, ::ds]
        shaft_ds = shaft_mask[::ds, ::ds]

        for comp_name, comp_ds in [('gc', gc_ds), ('shaft', shaft_ds)]:
            if comp_ds.sum() < 50:
                metrics[f'{comp_name}_s9_puncta_count'] = 0
                metrics[f'{comp_name}_s9_puncta_density'] = 0
                continue
            puncta = detect_puncta(s9_ds, comp_ds, min_sigma=1, max_sigma=5, threshold=0.02)
            area_um2 = comp_ds.sum() * (PIXEL_UM * ds)**2
            metrics[f'{comp_name}_s9_puncta_count'] = len(puncta)
            metrics[f'{comp_name}_s9_puncta_density'] = len(puncta) / (area_um2 + 1e-10)
            if puncta:
                metrics[f'{comp_name}_s9_puncta_mean_int'] = np.mean([p['intensity'] for p in puncta])
                metrics[f'{comp_name}_s9_puncta_mean_rad'] = np.mean([p['radius_um'] for p in puncta]) * ds
            else:
                metrics[f'{comp_name}_s9_puncta_mean_int'] = 0
                metrics[f'{comp_name}_s9_puncta_mean_rad'] = 0

        # ── Filament density (tubeness) per compartment ──
        # Downsample for speed
        for comp_name, comp_ds in [('gc', gc_ds), ('shaft', shaft_ds)]:
            if comp_ds.sum() < 50:
                metrics[f'{comp_name}_act_tubeness'] = 0
                metrics[f'{comp_name}_gskp_tubeness'] = 0
                continue
            act_ds_img = act_raw[::ds, ::ds]
            gskp_ds_img = gskp_raw[::ds, ::ds]
            act_sm = ndimage.gaussian_filter(normalize_2d(act_ds_img) * comp_ds, 1.5)
            gskp_sm = ndimage.gaussian_filter(normalize_2d(gskp_ds_img) * comp_ds, 1.5)
            act_tube = sato(act_sm, sigmas=(1, 2, 3), black_ridges=False) * comp_ds
            gskp_tube = sato(gskp_sm, sigmas=(1, 2, 3), black_ridges=False) * comp_ds
            metrics[f'{comp_name}_act_tubeness'] = act_tube[comp_ds].mean() if comp_ds.sum() > 0 else 0
            metrics[f'{comp_name}_gskp_tubeness'] = gskp_tube[comp_ds].mean() if comp_ds.sum() > 0 else 0

        # ── Alignment (S9 vs Actin) per compartment ──
        for comp_name, comp_mask in [('gc', gc_mask[::ds, ::ds]), ('shaft', shaft_mask[::ds, ::ds])]:
            if comp_mask.sum() < 100:
                metrics[f'{comp_name}_alignment_deg'] = np.nan
                continue
            s9_sm = ndimage.gaussian_filter(normalize_2d(s9_raw[::ds, ::ds]) * comp_mask, 2)
            act_sm2 = ndimage.gaussian_filter(normalize_2d(act_raw[::ds, ::ds]) * comp_mask, 2)
            S_s9 = structure_tensor(s9_sm, sigma=3)
            S_act = structure_tensor(act_sm2, sigma=3)
            angle_s9 = 0.5 * np.arctan2(2 * S_s9[1], S_s9[0] - S_s9[2])
            angle_act = 0.5 * np.arctan2(2 * S_act[1], S_act[0] - S_act[2])
            diff = np.abs(angle_s9 - angle_act)
            diff = np.minimum(diff, np.pi - diff)
            metrics[f'{comp_name}_alignment_deg'] = np.degrees(diff[comp_mask].mean())

        # Store for visualization
        metrics['_act'] = act
        metrics['_s9'] = s9
        metrics['_tub'] = tub
        metrics['_gc_mask'] = gc_mask
        metrics['_shaft_mask'] = shaft_mask
        metrics['_neurite_mask'] = neurite_mask

    except Exception as e:
        log(f"    ERROR {cell_id}: {e}")
        import traceback; traceback.print_exc()
        return None

    return metrics

# Process all
for cond_name, file_list in [('Control', CTRL_FILES), ('CHIR', CHIR_FILES)]:
    log(f"\n  Processing {cond_name}...")
    for i, (cid, fp) in enumerate(file_list):
        log(f"    [{i+1}/{len(file_list)}] Cell {cid}...")
        m = process_cell(cid, fp, cond_name)
        if m is not None:
            all_metrics.append(m)
            log(f"      neurite={m['neurite_area_um2']:.0f}µm², GC={m['gc_area_um2']:.0f}µm², "
                f"shaft={m['shaft_area_um2']:.0f}µm², n_GC={m['n_growth_cones']}, "
                f"Pearson(S9-Act)gc={m.get('gc_pearson_s9_act', 'N/A')}")
        garbage_collect.collect()

ctrl = [m for m in all_metrics if m['condition'] == 'Control']
chir = [m for m in all_metrics if m['condition'] == 'CHIR']
log(f"\n  Processed: {len(ctrl)} Control, {len(chir)} CHIR")

# ══════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════
def get_vals(group, key):
    return np.array([m[key] for m in group if key in m
                     and m[key] is not None
                     and not (isinstance(m[key], float) and np.isnan(m[key]))], dtype=float)

def stat_test(cv, bv, label):
    if len(cv) < 3 or len(bv) < 3:
        return
    U, p = stats.mannwhitneyu(cv, bv, alternative='two-sided')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    d = (np.mean(bv) - np.mean(cv)) / (np.sqrt((np.std(cv)**2 + np.std(bv)**2) / 2) + 1e-10)
    log(f"  {label:<42} Ctrl={np.mean(cv):.4f}  CHIR={np.mean(bv):.4f}  p={p:.4f} {sig}  d={d:.2f}")
    return p

# ══════════════════════════════════════════════════════════════════════════
# STATISTICAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("STATISTICAL COMPARISON: CONTROL vs CHIR")
log(f"{'='*70}")

log("\n  --- Morphology ---")
for key, label in [
    ('gc_area_um2', 'GC area (µm²)'), ('gc_fraction', 'GC fraction'),
    ('gc_mean_circularity', 'GC circularity'), ('gc_mean_solidity', 'GC solidity'),
    ('n_growth_cones', 'N growth cones'),
]:
    stat_test(get_vals(ctrl, key), get_vals(chir, key), label)

log("\n  --- Growth Cone intensities ---")
for key, label in [
    ('gc_act_mean', 'GC Actin'), ('gc_s9_mean', 'GC SEPTIN9'), ('gc_gskp_mean', 'GC GSKp'),
]:
    stat_test(get_vals(ctrl, key), get_vals(chir, key), label)

log("\n  --- Shaft intensities ---")
for key, label in [
    ('shaft_act_mean', 'Shaft Actin'), ('shaft_s9_mean', 'Shaft SEPTIN9'), ('shaft_gskp_mean', 'Shaft GSKp'),
]:
    stat_test(get_vals(ctrl, key), get_vals(chir, key), label)

log("\n  --- Colocalization (GC) ---")
for key, label in [
    ('gc_pearson_s9_act', 'GC Pearson S9-Act'), ('gc_pearson_s9_tub', 'GC Pearson S9-GSKp'),
    ('gc_pearson_act_tub', 'GC Pearson Act-GSKp'),
    ('gc_icq_s9_act', 'GC ICQ S9-Act'), ('gc_icq_s9_tub', 'GC ICQ S9-GSKp'),
    ('gc_manders_s9_on_act', 'GC Manders S9→Act'), ('gc_manders_s9_on_tub', 'GC Manders S9→GSKp'),
]:
    stat_test(get_vals(ctrl, key), get_vals(chir, key), label)

log("\n  --- Colocalization (Shaft) ---")
for key, label in [
    ('shaft_pearson_s9_act', 'Shaft Pearson S9-Act'), ('shaft_pearson_s9_tub', 'Shaft Pearson S9-GSKp'),
    ('shaft_icq_s9_act', 'Shaft ICQ S9-Act'),
]:
    stat_test(get_vals(ctrl, key), get_vals(chir, key), label)

log("\n  --- S9 Puncta ---")
for key, label in [
    ('gc_s9_puncta_count', 'GC puncta count'), ('gc_s9_puncta_density', 'GC puncta density'),
    ('shaft_s9_puncta_count', 'Shaft puncta count'), ('shaft_s9_puncta_density', 'Shaft puncta density'),
]:
    stat_test(get_vals(ctrl, key), get_vals(chir, key), label)

log("\n  --- Overlap ---")
for key, label in [
    ('gc_jaccard_s9_act', 'GC Jaccard S9-Act'), ('gc_jaccard_s9_tub', 'GC Jaccard S9-Tub'),
    ('gc_frac_all_three', 'GC triple overlap'), ('shaft_jaccard_s9_act', 'Shaft Jaccard S9-Act'),
]:
    stat_test(get_vals(ctrl, key), get_vals(chir, key), label)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: REPRESENTATIVE MONTAGE WITH GC/SHAFT OVERLAY
# ══════════════════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("GENERATING FIGURES")
log(f"{'='*70}")
log("  Fig 01: Representative montage")

fig1 = plt.figure(figsize=(20, 10))
fig1.suptitle("Growth Cone & Shaft Segmentation — Control vs CHIR", fontsize=14, fontweight='bold')

n_show = min(3, len(ctrl), len(chir))
for row, (cond, group) in enumerate([('Control', ctrl), ('CHIR', chir)]):
    reps = group[:n_show]
    for col, m in enumerate(reps):
        act_n = m['_act']
        s9_n = m['_s9']
        tub_n = m['_tub']
        gc = m['_gc_mask']
        shaft = m['_shaft_mask']
        ny, nx = act_n.shape

        # Crop to neurite region for better visualization
        neurite = m['_neurite_mask']
        ys, xs = np.where(neurite)
        if len(ys) < 10:
            continue
        pad = 50
        y0 = max(0, ys.min() - pad); y1 = min(ny, ys.max() + pad)
        x0 = max(0, xs.min() - pad); x1 = min(nx, xs.max() + pad)

        # 3-channel merge
        ax = fig1.add_subplot(2, n_show * 2, row * n_show * 2 + col * 2 + 1)
        merge = np.stack([act_n[y0:y1, x0:x1] * 0.8,
                         s9_n[y0:y1, x0:x1] * 0.8,
                         tub_n[y0:y1, x0:x1] * 0.6], axis=-1)
        merge = np.clip(merge * 2, 0, 1)
        ax.imshow(merge)
        ax.set_title(f"{cond} #{m['id']}\nR=Act G=S9 B=GSKp", fontsize=8)
        if col == 0:
            ax.set_ylabel(cond, fontsize=12, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])

        # GC/Shaft overlay
        ax = fig1.add_subplot(2, n_show * 2, row * n_show * 2 + col * 2 + 2)
        overlay = np.stack([act_n[y0:y1, x0:x1] * 0.4] * 3, axis=-1)
        # GC in green
        gc_crop = gc[y0:y1, x0:x1]
        overlay[gc_crop, 0] = 0.1
        overlay[gc_crop, 1] = 0.9
        overlay[gc_crop, 2] = 0.1
        # Shaft in blue
        sh_crop = shaft[y0:y1, x0:x1]
        overlay[sh_crop, 0] = 0.1
        overlay[sh_crop, 1] = 0.3
        overlay[sh_crop, 2] = 0.9
        ax.imshow(np.clip(overlay, 0, 1))
        gc_a = gc_crop.sum() * PIXEL_UM**2
        sh_a = sh_crop.sum() * PIXEL_UM**2
        ax.set_title(f"GC(green)={gc_a:.0f}µm²\nShaft(blue)={sh_a:.0f}µm²", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

fig1.tight_layout()
fig1.savefig(str(OUT / "01_gc_shaft_montage.png"), dpi=200, bbox_inches='tight')
plt.close(fig1)
log("    Saved 01_gc_shaft_montage.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: COLOCALIZATION
# ══════════════════════════════════════════════════════════════════════════
log("  Fig 02: Colocalization")

fig2 = plt.figure(figsize=(20, 10))
gs2 = GridSpec(2, 4, figure=fig2, hspace=0.35, wspace=0.35)
fig2.suptitle("Colocalization: Control vs CHIR — Growth Cone & Shaft", fontsize=14, fontweight='bold')

cond_colors = {'Control': '#3498db', 'CHIR': '#e74c3c'}
w = 0.35

def sig_str(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def add_sig_bar(ax, x, cv, bv, offset=0):
    """Add significance bracket above a pair of boxplots at position x."""
    _, p = stats.mannwhitneyu(cv, bv, alternative='two-sided') if len(cv) >= 3 and len(bv) >= 3 else (0, 1)
    all_vals = np.concatenate([cv, bv])
    y_max = np.percentile(all_vals, 95) if len(all_vals) > 0 else 1
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim()[1] != ax.get_ylim()[0] else 1
    y = y_max + y_range * 0.08 + offset * y_range * 0.12
    ss = sig_str(p)
    color = 'red' if p < 0.05 else 'gray'
    ax.plot([x - w/2, x + w/2], [y, y], 'k-', lw=0.8)
    ax.text(x, y + y_range * 0.01, ss, ha='center', fontsize=7, color=color, fontweight='bold')

# A: Pearson in GC
ax = fig2.add_subplot(gs2[0, 0])
pairs_p = [('gc_pearson_s9_act', 'S9-Act'), ('gc_pearson_s9_tub', 'S9-GSKp'), ('gc_pearson_act_tub', 'Act-GSKp')]
for i, (key, label) in enumerate(pairs_p):
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (key, label) in enumerate(pairs_p):
    add_sig_bar(ax, i, get_vals(ctrl, key), get_vals(chir, key))
ax.set_xticks(range(len(pairs_p))); ax.set_xticklabels([p[1] for p in pairs_p], fontsize=8)
ax.set_ylabel("Pearson r"); ax.set_title("A. Pearson — Growth Cone"); ax.grid(alpha=0.3)

# B: Pearson in Shaft
ax = fig2.add_subplot(gs2[0, 1])
pairs_s = [('shaft_pearson_s9_act', 'S9-Act'), ('shaft_pearson_s9_tub', 'S9-GSKp'), ('shaft_pearson_act_tub', 'Act-GSKp')]
for i, (key, label) in enumerate(pairs_s):
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (key, label) in enumerate(pairs_s):
    add_sig_bar(ax, i, get_vals(ctrl, key), get_vals(chir, key))
ax.set_xticks(range(len(pairs_s))); ax.set_xticklabels([p[1] for p in pairs_s], fontsize=8)
ax.set_ylabel("Pearson r"); ax.set_title("B. Pearson — Shaft"); ax.grid(alpha=0.3)

# C: Manders in GC
ax = fig2.add_subplot(gs2[0, 2])
mk = [('gc_manders_s9_on_act', 'S9→Act'), ('gc_manders_act_on_s9', 'Act→S9'),
      ('gc_manders_s9_on_tub', 'S9→GSKp'), ('gc_manders_gskp_on_s9', 'GSKp→S9')]
for i, (key, label) in enumerate(mk):
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (key, label) in enumerate(mk):
    add_sig_bar(ax, i, get_vals(ctrl, key), get_vals(chir, key))
ax.set_xticks(range(len(mk))); ax.set_xticklabels([m[1] for m in mk], fontsize=7, rotation=15)
ax.set_ylabel("Manders"); ax.set_title("C. Manders — Growth Cone"); ax.grid(alpha=0.3)

# D: ICQ
ax = fig2.add_subplot(gs2[0, 3])
ik = [('gc_icq_s9_act', 'GC S9-Act'), ('gc_icq_s9_tub', 'GC S9-GSKp'),
      ('shaft_icq_s9_act', 'Shaft S9-Act'), ('shaft_icq_s9_tub', 'Shaft S9-GSKp')]
for i, (key, label) in enumerate(ik):
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (key, label) in enumerate(ik):
    add_sig_bar(ax, i, get_vals(ctrl, key), get_vals(chir, key))
ax.set_xticks(range(len(ik))); ax.set_xticklabels([i_[1] for i_ in ik], fontsize=7, rotation=15)
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_ylabel("ICQ"); ax.set_title("D. ICQ — GC & Shaft"); ax.grid(alpha=0.3)

# E: Jaccard overlap
ax = fig2.add_subplot(gs2[1, 0])
jk = [('gc_jaccard_s9_act', 'GC S9-Act'), ('gc_jaccard_s9_tub', 'GC S9-GSKp'),
      ('shaft_jaccard_s9_act', 'Shaft S9-Act'), ('shaft_jaccard_s9_tub', 'Shaft S9-GSKp')]
for i, (key, label) in enumerate(jk):
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (key, label) in enumerate(jk):
    add_sig_bar(ax, i, get_vals(ctrl, key), get_vals(chir, key))
ax.set_xticks(range(len(jk))); ax.set_xticklabels([j[1] for j in jk], fontsize=7, rotation=15)
ax.set_ylabel("Jaccard"); ax.set_title("E. Binary overlap (Jaccard)"); ax.grid(alpha=0.3)

# F: Triple overlap fraction
ax = fig2.add_subplot(gs2[1, 1])
for comp, label in [('gc_frac_all_three', 'GC'), ('shaft_frac_all_three', 'Shaft')]:
    idx = 0 if 'gc' in comp else 1
    cv = get_vals(ctrl, comp); bv = get_vals(chir, comp)
    ax.boxplot([cv], positions=[idx - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[idx + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for comp in [('gc_frac_all_three', 0), ('shaft_frac_all_three', 1)]:
    add_sig_bar(ax, comp[1], get_vals(ctrl, comp[0]), get_vals(chir, comp[0]))
ax.set_xticks([0, 1]); ax.set_xticklabels(['GC', 'Shaft'])
ax.set_ylabel("Fraction triple+"); ax.set_title("F. S9+Act+GSKp triple overlap"); ax.grid(alpha=0.3)

# G-H: Intensity per compartment
for col, ch, ch_label in [(2, 's9', 'SEPTIN9'), (3, 'tub', 'GSKp')]:
    ax = fig2.add_subplot(gs2[1, col])
    for i, comp in enumerate(['gc', 'shaft']):
        key = f'{comp}_{ch}_mean'
        cv = get_vals(ctrl, key); bv = get_vals(chir, key)
        ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
                   boxprops=dict(facecolor='#3498db', alpha=0.6))
        ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
                   boxprops=dict(facecolor='#e74c3c', alpha=0.6))
    for i, comp in enumerate(['gc', 'shaft']):
        key = f'{comp}_{ch}_mean'
        add_sig_bar(ax, i, get_vals(ctrl, key), get_vals(chir, key))
    ax.set_xticks([0, 1]); ax.set_xticklabels(['GC', 'Shaft'])
    ax.set_ylabel(f'{ch_label} intensity'); ax.set_title(f'{"G" if col==2 else "H"}. {ch_label} per compartment')
    ax.grid(alpha=0.3)

fig2.savefig(str(OUT / "02_colocalization.png"), dpi=200, bbox_inches='tight')
plt.close(fig2)
log("    Saved 02_colocalization.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: MORPHOLOGY & DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════
log("  Fig 03: Morphology & distributions")

fig3 = plt.figure(figsize=(20, 10))
gs3 = GridSpec(2, 4, figure=fig3, hspace=0.35, wspace=0.35)
fig3.suptitle("GC Morphology & Channel Distributions", fontsize=14, fontweight='bold')

# A: GC area
ax = fig3.add_subplot(gs3[0, 0])
cv = get_vals(ctrl, 'gc_area_um2'); bv = get_vals(chir, 'gc_area_um2')
bp1 = ax.boxplot([cv], positions=[0], widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor='#3498db', alpha=0.6))
bp2 = ax.boxplot([bv], positions=[1], widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor='#e74c3c', alpha=0.6))
add_sig_bar(ax, 0.5, get_vals(ctrl, 'gc_area_um2'), get_vals(chir, 'gc_area_um2'))
ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'CHIR'])
ax.set_ylabel('GC area (µm²)'); ax.set_title('A. Growth cone area'); ax.grid(alpha=0.3)

# B: GC fraction
ax = fig3.add_subplot(gs3[0, 1])
cv = get_vals(ctrl, 'gc_fraction'); bv = get_vals(chir, 'gc_fraction')
ax.boxplot([cv], positions=[0], widths=0.6, patch_artist=True,
           boxprops=dict(facecolor='#3498db', alpha=0.6))
ax.boxplot([bv], positions=[1], widths=0.6, patch_artist=True,
           boxprops=dict(facecolor='#e74c3c', alpha=0.6))
add_sig_bar(ax, 0.5, cv, bv)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'CHIR'])
ax.set_ylabel('GC / total neurite'); ax.set_title('B. GC fraction'); ax.grid(alpha=0.3)

# C: Circularity & solidity
ax = fig3.add_subplot(gs3[0, 2])
for i, (key, label) in enumerate([('gc_mean_circularity', 'Circ.'), ('gc_mean_solidity', 'Solid.')]):
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (key, label) in enumerate([('gc_mean_circularity', 'Circ.'), ('gc_mean_solidity', 'Solid.')]):
    add_sig_bar(ax, i, get_vals(ctrl, key), get_vals(chir, key))
ax.set_xticks([0, 1]); ax.set_xticklabels(['Circularity', 'Solidity'])
ax.set_ylabel('Value'); ax.set_title('C. GC shape'); ax.grid(alpha=0.3)

# D: N growth cones
ax = fig3.add_subplot(gs3[0, 3])
cv = get_vals(ctrl, 'n_growth_cones'); bv = get_vals(chir, 'n_growth_cones')
ax.boxplot([cv], positions=[0], widths=0.6, patch_artist=True,
           boxprops=dict(facecolor='#3498db', alpha=0.6))
ax.boxplot([bv], positions=[1], widths=0.6, patch_artist=True,
           boxprops=dict(facecolor='#e74c3c', alpha=0.6))
add_sig_bar(ax, 0.5, cv, bv)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'CHIR'])
ax.set_ylabel('Count'); ax.set_title('D. N growth cones / FOV'); ax.grid(alpha=0.3)

# E: All 3 channel intensities in GC
ax = fig3.add_subplot(gs3[1, 0])
for i, (ch, label) in enumerate([('act', 'Actin'), ('s9', 'SEPTIN9'), ('tub', 'GSKp')]):
    key = f'gc_{ch}_mean'
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (ch, label) in enumerate([('act', 'Actin'), ('s9', 'SEPTIN9'), ('tub', 'GSKp')]):
    add_sig_bar(ax, i, get_vals(ctrl, f'gc_{ch}_mean'), get_vals(chir, f'gc_{ch}_mean'))
ax.set_xticks(range(3)); ax.set_xticklabels(['Actin', 'S9', 'GSKp'], fontsize=8)
ax.set_ylabel('Intensity'); ax.set_title('E. GC channel intensities'); ax.grid(alpha=0.3)

# F: Shaft intensities
ax = fig3.add_subplot(gs3[1, 1])
for i, (ch, label) in enumerate([('act', 'Actin'), ('s9', 'SEPTIN9'), ('tub', 'GSKp')]):
    key = f'shaft_{ch}_mean'
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (ch, label) in enumerate([('act', 'Actin'), ('s9', 'SEPTIN9'), ('tub', 'GSKp')]):
    add_sig_bar(ax, i, get_vals(ctrl, f'shaft_{ch}_mean'), get_vals(chir, f'shaft_{ch}_mean'))
ax.set_xticks(range(3)); ax.set_xticklabels(['Actin', 'S9', 'GSKp'], fontsize=8)
ax.set_ylabel('Intensity'); ax.set_title('F. Shaft channel intensities'); ax.grid(alpha=0.3)

# G: S9 puncta density
ax = fig3.add_subplot(gs3[1, 2])
for i, comp in enumerate(['gc', 'shaft']):
    key = f'{comp}_s9_puncta_density'
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, comp in enumerate(['gc', 'shaft']):
    add_sig_bar(ax, i, get_vals(ctrl, f'{comp}_s9_puncta_density'), get_vals(chir, f'{comp}_s9_puncta_density'))
ax.set_xticks([0, 1]); ax.set_xticklabels(['GC', 'Shaft'])
ax.set_ylabel('S9 puncta / µm²'); ax.set_title('G. S9 puncta density'); ax.grid(alpha=0.3)

# H: Filament density
ax = fig3.add_subplot(gs3[1, 3])
for i, (comp, ch) in enumerate([('gc', 'act'), ('gc', 'gskp'), ('shaft', 'act'), ('shaft', 'gskp')]):
    key = f'{comp}_{ch}_tubeness'
    cv = get_vals(ctrl, key); bv = get_vals(chir, key)
    ax.boxplot([cv], positions=[i - w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    ax.boxplot([bv], positions=[i + w/2], widths=w*0.8, patch_artist=True,
               boxprops=dict(facecolor='#e74c3c', alpha=0.6))
for i, (comp, ch) in enumerate([('gc', 'act'), ('gc', 'gskp'), ('shaft', 'act'), ('shaft', 'gskp')]):
    add_sig_bar(ax, i, get_vals(ctrl, f'{comp}_{ch}_tubeness'), get_vals(chir, f'{comp}_{ch}_tubeness'))
ax.set_xticks(range(4)); ax.set_xticklabels(['GC\nAct', 'GC\nGSKp', 'Shaft\nAct', 'Shaft\nGSKp'], fontsize=7)
ax.set_ylabel('Tubeness'); ax.set_title('H. Filament density'); ax.grid(alpha=0.3)

fig3.savefig(str(OUT / "03_morphology_distributions.png"), dpi=200, bbox_inches='tight')
plt.close(fig3)
log("    Saved 03_morphology_distributions.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
log("  Fig 04: Summary dashboard")

fig4 = plt.figure(figsize=(20, 14))
fig4.suptitle(f"Control (n={len(ctrl)}) vs CHIR 3µM/60min (n={len(chir)}) — Summary",
              fontsize=14, fontweight='bold')

ax = fig4.add_subplot(111)
ax.axis('off')

lines = [
    f"{'METRIC':<45} {'CONTROL':<22} {'CHIR':<22} {'p-value':<10} {'Effect'}",
    "=" * 110,
    "--- MORPHOLOGY ---",
]

all_stat_keys = [
    ('gc_area_um2', 'GC area (µm²)'),
    ('gc_fraction', 'GC fraction of neurite'),
    ('gc_mean_circularity', 'GC circularity'),
    ('gc_mean_solidity', 'GC solidity'),
    ('n_growth_cones', 'N growth cones'),
    ('BREAK', '--- GROWTH CONE ---'),
    ('gc_act_mean', 'GC Actin intensity'),
    ('gc_s9_mean', 'GC SEPTIN9 intensity'),
    ('gc_gskp_mean', 'GC GSKp intensity'),
    ('gc_pearson_s9_act', 'GC Pearson S9-Actin'),
    ('gc_pearson_s9_tub', 'GC Pearson S9-GSKp'),
    ('gc_pearson_act_tub', 'GC Pearson Actin-GSKp'),
    ('gc_icq_s9_act', 'GC ICQ S9-Actin'),
    ('gc_manders_s9_on_act', 'GC Manders S9→Actin'),
    ('gc_manders_s9_on_tub', 'GC Manders S9→GSKp'),
    ('gc_jaccard_s9_act', 'GC Jaccard S9-Actin'),
    ('gc_jaccard_s9_tub', 'GC Jaccard S9-GSKp'),
    ('gc_s9_puncta_density', 'GC S9 puncta density'),
    ('gc_act_tubeness', 'GC Actin filament density'),
    ('gc_gskp_tubeness', 'GC MT filament density'),
    ('BREAK', '--- SHAFT ---'),
    ('shaft_act_mean', 'Shaft Actin intensity'),
    ('shaft_s9_mean', 'Shaft SEPTIN9 intensity'),
    ('shaft_gskp_mean', 'Shaft GSKp intensity'),
    ('shaft_pearson_s9_act', 'Shaft Pearson S9-Actin'),
    ('shaft_pearson_s9_tub', 'Shaft Pearson S9-GSKp'),
    ('shaft_s9_puncta_density', 'Shaft S9 puncta density'),
]

for key, label in all_stat_keys:
    if key == 'BREAK':
        lines.append(label)
        continue
    cv = get_vals(ctrl, key)
    bv = get_vals(chir, key)
    if len(cv) < 2 or len(bv) < 2:
        continue
    try:
        U, p = stats.mannwhitneyu(cv, bv, alternative='two-sided')
    except:
        p = 1.0
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    d = (np.mean(bv) - np.mean(cv)) / (np.sqrt((np.std(cv)**2 + np.std(bv)**2) / 2) + 1e-10)
    direction = "↑" if d > 0.3 else "↓" if d < -0.3 else "→"
    lines.append(f"{label:<45} {np.mean(cv):.4f}±{np.std(cv):.4f}{'':>2} "
                 f"{np.mean(bv):.4f}±{np.std(bv):.4f}{'':>2} "
                 f"{p:.4f} {sig:<4} {direction} d={d:.2f}")

ax.text(0.01, 0.98, '\n'.join(lines), transform=ax.transAxes, fontsize=8,
        va='top', ha='left', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig4.savefig(str(OUT / "04_summary_dashboard.png"), dpi=200, bbox_inches='tight')
plt.close(fig4)
log("    Saved 04_summary_dashboard.png")


# ══════════════════════════════════════════════════════════════════════════
# SAVE CSV
# ══════════════════════════════════════════════════════════════════════════
log("\n  Saving CSV...")
import csv
if not all_metrics:
    log("  ERROR: No cells processed! Check file discovery and segmentation.")
    sys.exit(1)
skip_keys = {k for k in all_metrics[0] if k.startswith('_')}
scalar_keys = sorted([k for k in all_metrics[0].keys() if k not in skip_keys
                      and not isinstance(all_metrics[0].get(k), np.ndarray)])
csv_path = OUT / "gskp_gc_shaft_metrics.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=scalar_keys)
    writer.writeheader()
    for m in all_metrics:
        row = {k: m.get(k, '') for k in scalar_keys}
        writer.writerow(row)
log(f"    Saved {csv_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
log(f"\n{'='*70}")
log("ANALYSIS COMPLETE")
log(f"{'='*70}")
log(f"  Control: {len(ctrl)}, CHIR: {len(chir)}")
log(f"  Figures 01-04 + CSV saved to: {OUT}")

log("\n  KEY FINDINGS:")
for key, label in all_stat_keys:
    if key == 'BREAK':
        continue
    cv = get_vals(ctrl, key)
    bv = get_vals(chir, key)
    if len(cv) >= 3 and len(bv) >= 3:
        try:
            U, p = stats.mannwhitneyu(cv, bv, alternative='two-sided')
            if p < 0.05:
                d = (np.mean(bv) - np.mean(cv)) / (np.sqrt((np.std(cv)**2 + np.std(bv)**2) / 2) + 1e-10)
                direction = "INCREASED" if d > 0 else "DECREASED"
                log(f"    {label}: {direction} (d={d:.2f}, p={p:.4f})")
        except:
            pass

log("\nDone!")
