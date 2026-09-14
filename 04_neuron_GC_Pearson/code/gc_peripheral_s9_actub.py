#!/usr/bin/env python3
"""
GC Peripheral Domain SEPTIN9–AcetylTub Colocalization
=====================================================
Sub-segments the growth cone into:
  - C-domain (central): high AcTub tubeness, interior (bundled MTs)
  - P-domain (peripheral): actin-rich, low AcTub, outer leading edge
  - T-zone (transition): intermediate

Measures S9–AcTub colocalization in each subdomain.
Finds top 3 CHIR examples of S9 tracking pioneer MTs into the P-domain.
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
from pathlib import Path
import sys, re, csv, warnings
warnings.filterwarnings("ignore")

def log(msg):
    print(msg); sys.stdout.flush()

import _config as C
BASE = Path(C.require_images())
CSV  = Path(C.out("gc_analysis_results")) / "gc_shaft_metrics.csv"
OUT  = Path(C.out("gc_peripheral_results"))
OUT.mkdir(exist_ok=True)

PIXEL_UM = 0.0269310471755625
CH_ACT, CH_S9, CH_TUB = C.CH_ACT, C.CH_S9, C.CH_TUB

# ── File finders ──
def find_actub_file(cell_id, condition='CHIR'):
    """Find AcetylTub nd2 file for cell."""
    cond_tag = 'CHIR' if condition == 'CHIR' else 'CON'
    for f in sorted(BASE.glob("*.nd2")):
        if "Deconvolved" not in f.name or f.name.startswith("._"):
            continue
        if "AcetyTub" not in f.name:
            continue
        if "Richardson" in f.name:
            continue
        if cond_tag not in f.name:
            continue
        # Match by cell_id
        if cell_id.startswith('n'):
            num = cell_id[1:]
            if f'.n{num}' in f.name and 'Denoised' in f.name:
                return f
        else:
            if f'phall-{cell_id}' in f.name and 'Denoised' in f.name:
                return f
    # Fallback: try without condition tag (old naming)
    for f in sorted(BASE.glob("*.nd2")):
        if "Deconvolved" not in f.name or f.name.startswith("._"):
            continue
        if "AcetyTub" not in f.name:
            continue
        if "Richardson" in f.name:
            continue
        if cell_id.startswith('n'):
            num = cell_id[1:]
            if f'.n{num}' in f.name and 'Denoised' in f.name:
                return f
        else:
            if f'phall-{cell_id}' in f.name and 'Denoised' in f.name:
                return f
    return None

def load_mip(filepath):
    """Load and return MIP, normalized 0-1 per channel."""
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
    """Segment neurite mask from MIP."""
    ds = 4
    act_ds = ndimage.zoom(mip[CH_ACT], 1/ds, order=1)
    tub_ds = ndimage.zoom(mip[CH_TUB], 1/ds, order=1)
    combined = ndimage.gaussian_filter(act_ds + 0.5*tub_ds, sigma=3)
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
    """Segment GC vs shaft within neurite mask."""
    act_sm = ndimage.gaussian_filter(mip[CH_ACT], sigma=5)
    edt = ndimage.distance_transform_edt(mask) * PIXEL_UM
    labels = measure.label(mask)
    gc_mask = np.zeros_like(mask)
    shaft_mask = np.zeros_like(mask)
    for rid in range(1, labels.max()+1):
        region = labels == rid
        if region.sum() < 100:
            continue
        act_in = act_sm[region]
        edt_in = edt[region]
        gc_pix = (act_in > np.percentile(act_in, 60)) & (edt_in > np.percentile(edt_in, 50))
        gc_r = np.zeros_like(mask); gc_r[region] = gc_pix
        sh_r = np.zeros_like(mask); sh_r[region] = ~gc_pix
        gc_mask |= gc_r; shaft_mask |= sh_r
    return gc_mask, shaft_mask, edt

def segment_gc_subdomains(mip, gc_mask, edt):
    """
    Subdivide GC into C-domain, T-zone, and P-domain.

    Strategy — SPATIAL definition first, then measure signals within:
    - Use EDT within GC to define spatial rings:
      C-domain: innermost 1/3 of GC (deep interior, where MT bundles are)
      T-zone:   middle 1/3
      P-domain: outermost 1/3 of GC (peripheral leading edge)
    - This avoids circular logic: domains are defined by position,
      not by the signal we want to measure.
    - Then within the spatially-defined P-domain, we can look for
      pioneer MTs (high tubeness peaks = MTs extending into periphery).
    """
    if gc_mask.sum() < 100:
        return np.zeros_like(gc_mask), np.zeros_like(gc_mask), np.zeros_like(gc_mask)

    # EDT within GC only (distance from GC boundary)
    gc_edt = ndimage.distance_transform_edt(gc_mask) * PIXEL_UM
    gc_edt_vals = gc_edt[gc_mask]
    edt_p33 = np.percentile(gc_edt_vals, 33)
    edt_p66 = np.percentile(gc_edt_vals, 66)

    is_gc = gc_mask.copy()

    # Spatial definition:
    # P-domain: outer ring (closest to edge) — edt < p33
    p_domain = is_gc & (gc_edt <= edt_p33)
    # C-domain: inner core — edt > p66
    c_domain = is_gc & (gc_edt >= edt_p66)
    # T-zone: middle ring
    t_zone = is_gc & (~c_domain) & (~p_domain)

    return c_domain, t_zone, p_domain

def pearson_r(a, b, mask):
    """Pearson correlation of two images within mask."""
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if len(x) < 20:
        return np.nan
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    if denom < 1e-15:
        return np.nan
    return np.sum(xm * ym) / denom

def manders_coeff(a, b, mask, thresh_pct=50):
    """Manders M1: fraction of A that overlaps with B (above threshold)."""
    av = a[mask].astype(np.float64)
    bv = b[mask].astype(np.float64)
    if len(av) < 20:
        return np.nan
    b_thresh = np.percentile(bv, thresh_pct)
    b_pos = bv > b_thresh
    if av.sum() < 1e-15:
        return np.nan
    return av[b_pos].sum() / av.sum()

def icq(a, b, mask):
    """Intensity Correlation Quotient."""
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    if len(x) < 20:
        return np.nan
    prod = (x - x.mean()) * (y - y.mean())
    return (np.sum(prod > 0) / len(prod)) - 0.5

# ── Process all AcetylTub cells ──
log("="*80)
log("GC PERIPHERAL DOMAIN S9-AcTub ANALYSIS")
log("="*80)

# Load CSV to get cell IDs and conditions
import pandas as pd
df = pd.read_csv(CSV)
log(f"Loaded {len(df)} cells from CSV")

results = []
for idx, row in df.iterrows():
    cell_id = str(row['id'])
    condition = row['condition']

    fp = find_actub_file(cell_id, condition)
    if fp is None:
        log(f"  [{idx+1}/{len(df)}] {cell_id} ({condition}): FILE NOT FOUND - skipping")
        continue

    log(f"  [{idx+1}/{len(df)}] {cell_id} ({condition}): {fp.name[:50]}...")

    try:
        mip = load_mip(fp)
        mask = segment_neurite(mip)
        gc_mask, shaft_mask, edt = segment_gc_shaft(mip, mask)

        if gc_mask.sum() < 200:
            log(f"    GC too small ({gc_mask.sum()} px), skipping")
            del mip; garbage_collect.collect()
            continue

        # Segment GC subdomains
        c_domain, t_zone, p_domain = segment_gc_subdomains(mip, gc_mask, edt)

        act = mip[CH_ACT]
        s9 = mip[CH_S9]
        tub = mip[CH_TUB]

        # Compute Sato tubeness ONCE per cell (expensive operation)
        tub_sm = ndimage.gaussian_filter(tub, sigma=1.0)
        ridges = filters.sato(tub_sm, sigmas=[1.0, 1.5, 2.0], black_ridges=False)

        # Compute metrics per subdomain
        rec = {
            'id': cell_id, 'condition': condition,
            'gc_area_px': int(gc_mask.sum()),
            'c_domain_area_px': int(c_domain.sum()),
            't_zone_area_px': int(t_zone.sum()),
            'p_domain_area_px': int(p_domain.sum()),
            'c_domain_frac': c_domain.sum() / gc_mask.sum() if gc_mask.sum() > 0 else np.nan,
            'p_domain_frac': p_domain.sum() / gc_mask.sum() if gc_mask.sum() > 0 else np.nan,
            't_zone_frac': t_zone.sum() / gc_mask.sum() if gc_mask.sum() > 0 else np.nan,
        }

        # Per-domain S9-AcTub colocalization
        for domain_name, domain_mask in [('gc', gc_mask), ('c_domain', c_domain),
                                          ('t_zone', t_zone), ('p_domain', p_domain),
                                          ('shaft', shaft_mask)]:
            if domain_mask.sum() < 50:
                rec[f'{domain_name}_pearson_s9_tub'] = np.nan
                rec[f'{domain_name}_manders_s9_on_tub'] = np.nan
                rec[f'{domain_name}_manders_tub_on_s9'] = np.nan
                rec[f'{domain_name}_icq_s9_tub'] = np.nan
                rec[f'{domain_name}_s9_mean'] = np.nan
                rec[f'{domain_name}_tub_mean'] = np.nan
                rec[f'{domain_name}_act_mean'] = np.nan
                rec[f'{domain_name}_tub_tubeness'] = np.nan
                continue

            rec[f'{domain_name}_pearson_s9_tub'] = pearson_r(s9, tub, domain_mask)
            rec[f'{domain_name}_manders_s9_on_tub'] = manders_coeff(s9, tub, domain_mask)
            rec[f'{domain_name}_manders_tub_on_s9'] = manders_coeff(tub, s9, domain_mask)
            rec[f'{domain_name}_icq_s9_tub'] = icq(s9, tub, domain_mask)
            rec[f'{domain_name}_s9_mean'] = float(np.mean(s9[domain_mask]))
            rec[f'{domain_name}_tub_mean'] = float(np.mean(tub[domain_mask]))
            rec[f'{domain_name}_act_mean'] = float(np.mean(act[domain_mask]))
            rec[f'{domain_name}_tub_tubeness'] = float(np.mean(ridges[domain_mask]))

        # Pioneer MT detection in P-domain:
        # Now P-domain is defined SPATIALLY (outer ring), so MTs can be present there.
        # Pioneer MTs = tubeness peaks in the P-domain (MTs extending into periphery)
        if p_domain.sum() > 50:
            # Pioneer MTs: tubeness above GC median — these are MTs reaching into periphery
            tub_gc_vals = ridges[gc_mask]
            tub_thresh = np.percentile(tub_gc_vals, 50)

            pioneer_mt = p_domain & (ridges > tub_thresh)
            rec['p_domain_pioneer_mt_frac'] = pioneer_mt.sum() / p_domain.sum() if p_domain.sum() > 0 else 0

            # S9 on pioneer MTs in P-domain
            if pioneer_mt.sum() > 20:
                rec['p_domain_pioneer_s9_mean'] = float(np.mean(s9[pioneer_mt]))
                rec['p_domain_pioneer_pearson'] = pearson_r(s9, tub, pioneer_mt)
                # S9 enrichment on pioneer MTs vs rest of P-domain
                non_pioneer_p = p_domain & ~pioneer_mt
                if non_pioneer_p.sum() > 20:
                    rec['p_domain_s9_enrichment_pioneer'] = (
                        np.mean(s9[pioneer_mt]) / np.mean(s9[non_pioneer_p])
                        if np.mean(s9[non_pioneer_p]) > 0 else np.nan
                    )
                else:
                    rec['p_domain_s9_enrichment_pioneer'] = np.nan
            else:
                rec['p_domain_pioneer_s9_mean'] = np.nan
                rec['p_domain_pioneer_pearson'] = np.nan
                rec['p_domain_s9_enrichment_pioneer'] = np.nan
        else:
            rec['p_domain_pioneer_mt_frac'] = np.nan
            rec['p_domain_pioneer_s9_mean'] = np.nan
            rec['p_domain_pioneer_pearson'] = np.nan
            rec['p_domain_s9_enrichment_pioneer'] = np.nan

        results.append(rec)
        log(f"    P-domain: Pearson={rec['p_domain_pearson_s9_tub']:.3f}, "
            f"Pioneer frac={rec['p_domain_pioneer_mt_frac']:.3f}, "
            f"Pioneer S9 enrich={rec.get('p_domain_s9_enrichment_pioneer', 0):.2f}")

    except Exception as e:
        log(f"    ERROR: {e}")

    del mip; garbage_collect.collect()

# ── Save results ──
rdf = pd.DataFrame(results)
rdf.to_csv(OUT / 'gc_subdomain_metrics.csv', index=False)
log(f"\nSaved {len(rdf)} cells to gc_subdomain_metrics.csv")

ctrl = rdf[rdf.condition == 'Control']
chir = rdf[rdf.condition == 'CHIR']
log(f"  Control: {len(ctrl)}, CHIR: {len(chir)}")

# ── Statistical comparison ──
log("\n" + "="*80)
log("STATISTICAL COMPARISON: CHIR vs CONTROL")
log("="*80)

stat_results = []
metrics_to_test = [
    ('gc_pearson_s9_tub', 'GC (whole) Pearson S9-AcTub'),
    ('c_domain_pearson_s9_tub', 'C-domain Pearson S9-AcTub'),
    ('t_zone_pearson_s9_tub', 'T-zone Pearson S9-AcTub'),
    ('p_domain_pearson_s9_tub', 'P-domain Pearson S9-AcTub'),
    ('shaft_pearson_s9_tub', 'Shaft Pearson S9-AcTub'),
    ('gc_manders_s9_on_tub', 'GC Manders S9→AcTub'),
    ('c_domain_manders_s9_on_tub', 'C-domain Manders S9→AcTub'),
    ('p_domain_manders_s9_on_tub', 'P-domain Manders S9→AcTub'),
    ('gc_manders_tub_on_s9', 'GC Manders AcTub→S9'),
    ('c_domain_manders_tub_on_s9', 'C-domain Manders AcTub→S9'),
    ('p_domain_manders_tub_on_s9', 'P-domain Manders AcTub→S9'),
    ('gc_icq_s9_tub', 'GC ICQ S9-AcTub'),
    ('p_domain_icq_s9_tub', 'P-domain ICQ S9-AcTub'),
    ('p_domain_pioneer_mt_frac', 'P-domain Pioneer MT fraction'),
    ('p_domain_pioneer_pearson', 'P-domain Pioneer Pearson S9-AcTub'),
    ('p_domain_s9_enrichment_pioneer', 'P-domain S9 enrichment on pioneers'),
    ('c_domain_frac', 'C-domain area fraction'),
    ('p_domain_frac', 'P-domain area fraction'),
    ('p_domain_s9_mean', 'P-domain S9 mean intensity'),
    ('p_domain_tub_mean', 'P-domain AcTub mean intensity'),
]

print(f"\n{'Metric':50s} {'Ctrl':>10s} {'CHIR':>10s} {'Δ':>8s} {'d':>8s} {'p':>10s} {'Sig':>5s}")
print("-"*100)

for col, label in metrics_to_test:
    if col not in rdf.columns:
        continue
    c = ctrl[col].dropna()
    h = chir[col].dropna()
    if len(c) < 3 or len(h) < 3:
        continue

    _, norm_c = stats.shapiro(c) if len(c) < 50 else (0, 0.01)
    _, norm_h = stats.shapiro(h) if len(h) < 50 else (0, 0.01)

    if norm_c > 0.05 and norm_h > 0.05:
        t_stat, p_val = stats.ttest_ind(c, h, equal_var=False)
        test = 'Welch t'
    else:
        t_stat, p_val = stats.mannwhitneyu(c, h, alternative='two-sided')
        test = 'MWU'

    pooled_std = np.sqrt(((len(c)-1)*c.std()**2 + (len(h)-1)*h.std()**2) / (len(c)+len(h)-2))
    d = (h.mean() - c.mean()) / pooled_std if pooled_std > 0 else 0

    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'

    print(f"  {label:48s} {c.mean():10.4f} {h.mean():10.4f} {h.mean()-c.mean():+8.4f} {d:+8.3f} {p_val:10.4f} {sig:>5s}")

    stat_results.append({
        'metric': label, 'col': col,
        'ctrl_mean': c.mean(), 'chir_mean': h.mean(),
        'delta': h.mean() - c.mean(), 'cohens_d': d,
        'p_value': p_val, 'test': test, 'sig': sig
    })

sdf = pd.DataFrame(stat_results)
sdf.to_csv(OUT / 'gc_subdomain_stats.csv', index=False)

# ── Rank CHIR cells by P-domain phenotype ──
log("\n\n" + "="*80)
log("TOP CHIR CELLS — P-DOMAIN S9-AcTub COLOCALIZATION")
log("="*80)

chir_cells = rdf[rdf.condition == 'CHIR'].copy()
# Composite score: Pearson + Manders S9→AcTub + S9 enrichment on pioneers
for c_name in ['p_domain_pearson_s9_tub', 'p_domain_manders_s9_on_tub', 'p_domain_s9_enrichment_pioneer']:
    if c_name in chir_cells.columns:
        vals = chir_cells[c_name].fillna(0)
        vmin, vmax = vals.min(), vals.max()
        chir_cells[f'{c_name}_norm'] = (vals - vmin) / (vmax - vmin + 1e-10)

score_cols = [c for c in chir_cells.columns if c.endswith('_norm')]
if score_cols:
    chir_cells['composite_score'] = chir_cells[score_cols].mean(axis=1)
else:
    chir_cells['composite_score'] = chir_cells.get('p_domain_pearson_s9_tub', 0)

chir_ranked = chir_cells.sort_values('composite_score', ascending=False)

log("\nRanking (top 10):")
for i, (_, r) in enumerate(chir_ranked.head(10).iterrows()):
    log(f"  {i+1}. {r['id']:>8s}: P-Pearson={r.get('p_domain_pearson_s9_tub',0):.3f}, "
        f"Pioneer enrich={r.get('p_domain_s9_enrichment_pioneer',0):.2f}, "
        f"Score={r['composite_score']:.3f}")

# Also rank controls for median reference
ctrl_cells = rdf[rdf.condition == 'Control'].copy()
ctrl_cells['p_pearson'] = ctrl_cells.get('p_domain_pearson_s9_tub', 0)
ctrl_median_idx = (ctrl_cells['p_pearson'] - ctrl_cells['p_pearson'].median()).abs().idxmin()
median_ctrl = ctrl_cells.loc[ctrl_median_idx]
log(f"\n  Median Control: {median_ctrl['id']} (P-Pearson={median_ctrl['p_pearson']:.3f})")

# ── FIGURE 1: Statistics overview ──
log("\nGenerating figures...")

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle('GC Subdomain S9–AcTub Colocalization: CHIR vs Control', fontsize=14, fontweight='bold')

# Color scheme
C_CTRL = '#4477AA'
C_CHIR = '#EE6677'

def boxplot_pair(ax, c_vals, h_vals, title, ylabel, show_pts=True):
    """Boxplot with individual points and stats."""
    c_vals = c_vals.dropna()
    h_vals = h_vals.dropna()
    bp = ax.boxplot([c_vals, h_vals], labels=['Control', 'CHIR'],
                    patch_artist=True, showfliers=False, widths=0.5)
    bp['boxes'][0].set_facecolor(C_CTRL); bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor(C_CHIR); bp['boxes'][1].set_alpha(0.4)
    bp['medians'][0].set_color(C_CTRL); bp['medians'][0].set_linewidth(2)
    bp['medians'][1].set_color(C_CHIR); bp['medians'][1].set_linewidth(2)

    if show_pts:
        jitter = 0.08
        ax.scatter(np.ones(len(c_vals)) + np.random.uniform(-jitter, jitter, len(c_vals)),
                   c_vals, c=C_CTRL, s=20, alpha=0.6, zorder=5, edgecolors='white', linewidths=0.3)
        ax.scatter(np.ones(len(h_vals))*2 + np.random.uniform(-jitter, jitter, len(h_vals)),
                   h_vals, c=C_CHIR, s=20, alpha=0.6, zorder=5, edgecolors='white', linewidths=0.3)

    # Stats
    _, p = stats.mannwhitneyu(c_vals, h_vals, alternative='two-sided')
    ps = np.sqrt(((len(c_vals)-1)*c_vals.std()**2 + (len(h_vals)-1)*h_vals.std()**2)/(len(c_vals)+len(h_vals)-2))
    d = (h_vals.mean()-c_vals.mean())/ps if ps > 0 else 0
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

    ymax = max(c_vals.max(), h_vals.max())
    ymin = min(c_vals.min(), h_vals.min())
    margin = (ymax - ymin) * 0.15

    ax.plot([1, 1, 2, 2], [ymax+margin*0.3, ymax+margin*0.5, ymax+margin*0.5, ymax+margin*0.3],
            'k-', lw=1)
    ax.text(1.5, ymax+margin*0.6, f'{sig}\np={p:.4f}\nd={d:+.2f}',
            ha='center', va='bottom', fontsize=8)

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Row 1: Pearson by domain
for i, (col, title) in enumerate([
    ('c_domain_pearson_s9_tub', 'C-domain\n(MT core)'),
    ('t_zone_pearson_s9_tub', 'T-zone\n(transition)'),
    ('p_domain_pearson_s9_tub', 'P-domain\n(peripheral)'),
    ('shaft_pearson_s9_tub', 'Shaft'),
]):
    boxplot_pair(axes[0, i], ctrl[col], chir[col],
                 f'Pearson S9–AcTub: {title}', 'Pearson r')

# Row 2: Pioneer MT and enrichment metrics
plot_pairs = [
    ('p_domain_manders_s9_on_tub', 'P-domain\nManders S9→AcTub', 'Manders M1'),
    ('p_domain_pioneer_mt_frac', 'P-domain\nPioneer MT fraction', 'Fraction'),
    ('p_domain_s9_enrichment_pioneer', 'P-domain\nS9 enrichment on pioneers', 'Fold enrichment'),
    ('p_domain_icq_s9_tub', 'P-domain\nICQ S9-AcTub', 'ICQ'),
]
for i, (col, title, ylabel) in enumerate(plot_pairs):
    if col in rdf.columns:
        boxplot_pair(axes[1, i], ctrl[col], chir[col], title, ylabel)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / '01_gc_subdomain_stats.png', dpi=200, bbox_inches='tight')
plt.close()
log("  Saved 01_gc_subdomain_stats.png")

# ── FIGURE 2: Domain-gradient plot ──
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('S9–AcTub Colocalization Gradient Across GC Subdomains', fontsize=14, fontweight='bold')

domain_order = ['C-domain', 'T-zone', 'P-domain']
pearson_cols = ['c_domain_pearson_s9_tub', 't_zone_pearson_s9_tub', 'p_domain_pearson_s9_tub']
manders_cols = ['c_domain_manders_s9_on_tub', 't_zone_manders_s9_on_tub', 'p_domain_manders_s9_on_tub']
icq_cols = ['c_domain_icq_s9_tub', 't_zone_icq_s9_tub', 'p_domain_icq_s9_tub']

for ax, cols, ylabel, title in [
    (axes2[0], pearson_cols, 'Pearson r', 'Pearson S9-AcTub'),
    (axes2[1], manders_cols, 'Manders M1', 'Manders S9→AcTub'),
    (axes2[2], icq_cols, 'ICQ', 'ICQ S9-AcTub'),
]:
    ctrl_means = [ctrl[c].mean() for c in cols]
    ctrl_sems = [ctrl[c].sem() for c in cols]
    chir_means = [chir[c].mean() for c in cols]
    chir_sems = [chir[c].sem() for c in cols]

    x = np.arange(3)
    ax.errorbar(x - 0.05, ctrl_means, yerr=ctrl_sems, fmt='o-', color=C_CTRL,
                markersize=8, capsize=5, label='Control', lw=2)
    ax.errorbar(x + 0.05, chir_means, yerr=chir_sems, fmt='s-', color=C_CHIR,
                markersize=8, capsize=5, label='CHIR', lw=2)

    # Add significance stars
    for i, col in enumerate(cols):
        c_v = ctrl[col].dropna()
        h_v = chir[col].dropna()
        if len(c_v) > 2 and len(h_v) > 2:
            _, p = stats.mannwhitneyu(c_v, h_v, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            if sig:
                ypos = max(ctrl_means[i]+ctrl_sems[i], chir_means[i]+chir_sems[i]) + 0.02
                ax.text(i, ypos, sig, ha='center', fontsize=12, fontweight='bold', color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(domain_order, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
fig2.savefig(OUT / '02_gc_domain_gradient.png', dpi=200, bbox_inches='tight')
plt.close()
log("  Saved 02_gc_domain_gradient.png")

# ── FIGURE 3: Top 3 showcase ──
log("\nCreating top 3 P-domain showcase...")

top3_ids = list(chir_ranked.head(3)['id'].values)
med_ctrl_id = str(median_ctrl['id'])
CELLS_SHOW = [(cid, 'CHIR', f'#{i+1} CHIR') for i, cid in enumerate(top3_ids)]
CELLS_SHOW.append((med_ctrl_id, 'Control', 'Median Ctrl'))

fig3 = plt.figure(figsize=(28, 22))
gs = GridSpec(4, 7, figure=fig3, hspace=0.25, wspace=0.12)
fig3.suptitle('Top 3 CHIR: S9–AcTub in GC Peripheral Domain\n(Pioneer MTs + SEPTIN9 in P-domain)',
              fontsize=15, fontweight='bold', y=0.99)

col_titles3 = ['Actin (647)', 'SEPTIN9 (561)', 'AcetylTub (488)',
               'S9(G)+AcTub(M)', 'GC Subdomains', 'P-domain\nS9+AcTub', 'P-domain\nPioneer MTs']

for row_idx, (cell_id, condition, label) in enumerate(CELLS_SHOW):
    log(f"  Loading {cell_id} ({condition})...")
    fp = find_actub_file(cell_id, condition)
    if fp is None:
        log(f"    FILE NOT FOUND")
        continue

    mip = load_mip(fp)
    mask = segment_neurite(mip)
    gc_mask, shaft_mask, edt = segment_gc_shaft(mip, mask)
    c_domain, t_zone, p_domain = segment_gc_subdomains(mip, gc_mask, edt)

    act = mip[CH_ACT]; s9 = mip[CH_S9]; tub = mip[CH_TUB]

    # Crop to neurite bbox
    ys, xs = np.where(mask)
    if len(ys) == 0:
        del mip; garbage_collect.collect(); continue
    pad = 30
    y0, y1 = max(0, ys.min()-pad), min(mask.shape[0], ys.max()+pad)
    x0, x1 = max(0, xs.min()-pad), min(mask.shape[1], xs.max()+pad)
    sl = (slice(y0, y1), slice(x0, x1))

    act_c = act[sl]; s9_c = s9[sl]; tub_c = tub[sl]
    mask_c = mask[sl]; gc_c = gc_mask[sl]; sh_c = shaft_mask[sl]
    cd_c = c_domain[sl]; tz_c = t_zone[sl]; pd_c = p_domain[sl]

    bg = 0.08
    act_d = np.where(mask_c, act_c, act_c*bg)
    s9_d = np.where(mask_c, s9_c, s9_c*bg)
    tub_d = np.where(mask_c, tub_c, tub_c*bg)

    # Get metrics for this cell
    cell_row = rdf[rdf.id == cell_id]
    pp = cell_row['p_domain_pearson_s9_tub'].values[0] if len(cell_row) > 0 else 0
    pe = cell_row['p_domain_s9_enrichment_pioneer'].values[0] if len(cell_row) > 0 else 0

    row_label = f'{label}\n{cell_id}\nP-Pear={pp:.3f}'
    color = C_CHIR if condition == 'CHIR' else C_CTRL

    # Col 0: Actin
    ax = fig3.add_subplot(gs[row_idx, 0])
    ax.imshow(act_d, cmap='hot', vmin=0, vmax=0.8)
    ax.set_ylabel(row_label, fontsize=10, fontweight='bold', color=color)
    ax.set_xticks([]); ax.set_yticks([])
    if row_idx == 0: ax.set_title(col_titles3[0], fontsize=9)

    # Col 1: S9
    ax = fig3.add_subplot(gs[row_idx, 1])
    ax.imshow(s9_d, cmap='Greens', vmin=0, vmax=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    if row_idx == 0: ax.set_title(col_titles3[1], fontsize=9)

    # Col 2: AcTub
    ax = fig3.add_subplot(gs[row_idx, 2])
    ax.imshow(tub_d, cmap='cool', vmin=0, vmax=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    if row_idx == 0: ax.set_title(col_titles3[2], fontsize=9)

    # Col 3: S9(green) + AcTub(magenta) merge
    ax = fig3.add_subplot(gs[row_idx, 3])
    merge = np.zeros((*s9_d.shape, 3))
    merge[:,:,0] = np.clip(tub_d * 1.3, 0, 1)
    merge[:,:,1] = np.clip(s9_d * 1.5, 0, 1)
    merge[:,:,2] = np.clip(tub_d * 1.3, 0, 1)
    ax.imshow(np.clip(merge, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    if row_idx == 0: ax.set_title(col_titles3[3], fontsize=9)

    # Col 4: GC subdomain segmentation overlay
    ax = fig3.add_subplot(gs[row_idx, 4])
    seg = np.zeros((*act_d.shape, 3))
    seg[cd_c] = [0.2, 0.4, 1.0]   # blue = C-domain
    seg[tz_c] = [0.8, 0.8, 0.2]   # yellow = T-zone
    seg[pd_c] = [1.0, 0.3, 0.3]   # red = P-domain
    seg[sh_c] = [0.4, 0.4, 0.4]   # gray = shaft
    # Blend with actin
    gray_bg = np.stack([act_d*0.4]*3, axis=-1)
    display = np.where(mask_c[:,:,None], 0.5*seg + 0.5*gray_bg, gray_bg*0.2)
    ax.imshow(np.clip(display, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    if row_idx == 0: ax.set_title(col_titles3[4], fontsize=9)
    # Legend
    if row_idx == 0:
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=[0.2,0.4,1.0], label='C-domain'),
                          Patch(facecolor=[0.8,0.8,0.2], label='T-zone'),
                          Patch(facecolor=[1.0,0.3,0.3], label='P-domain'),
                          Patch(facecolor=[0.4,0.4,0.4], label='Shaft')]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=6,
                  framealpha=0.8)

    # Col 5: P-domain only — S9+AcTub
    ax = fig3.add_subplot(gs[row_idx, 5])
    p_merge = np.zeros((*s9_d.shape, 3))
    p_merge[:,:,0] = np.clip(tub_d * 1.5, 0, 1)  # magenta
    p_merge[:,:,1] = np.clip(s9_d * 1.8, 0, 1)   # green
    p_merge[:,:,2] = np.clip(tub_d * 1.5, 0, 1)
    # Mask to P-domain only (dim everything else)
    p_display = np.where(pd_c[:,:,None], p_merge, p_merge * 0.1)
    # Add P-domain boundary
    from skimage.segmentation import find_boundaries
    p_bound = find_boundaries(pd_c, mode='inner')
    p_display[p_bound] = [1, 1, 0]  # yellow boundary
    ax.imshow(np.clip(p_display, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    if row_idx == 0: ax.set_title(col_titles3[5], fontsize=9)

    # Col 6: Pioneer MTs in P-domain
    ax = fig3.add_subplot(gs[row_idx, 6])
    tub_sm = ndimage.gaussian_filter(tub_c, sigma=1.0)
    ridges = filters.sato(tub_sm, sigmas=[1.0, 1.5, 2.0], black_ridges=False)

    # Normalize ridges
    rmax = np.percentile(ridges[gc_c] if gc_c.sum() > 0 else ridges, 99)
    ridges_n = np.clip(ridges / (rmax + 1e-10), 0, 1)

    pioneer_img = np.zeros((*act_d.shape, 3))
    # Show AcTub tubeness in cyan
    pioneer_img[:,:,1] = np.clip(ridges_n * 1.5, 0, 1) * 0.5
    pioneer_img[:,:,2] = np.clip(ridges_n * 1.5, 0, 1)
    # Show S9 in green
    pioneer_img[:,:,1] += np.clip(s9_d * 1.0, 0, 1) * 0.5
    # Highlight P-domain pioneers (high tubeness + S9 in P-domain)
    gc_tub_vals = ridges[gc_c] if gc_c.sum() > 0 else ridges[mask_c]
    if len(gc_tub_vals) > 0:
        pioneer_mask = pd_c & (ridges > np.percentile(gc_tub_vals, 50))
        # Mark pioneer MTs bright white
        pioneer_img[pioneer_mask] = [1, 1, 0.5]

    # Dim outside P-domain
    pioneer_display = np.where(pd_c[:,:,None], pioneer_img, pioneer_img * 0.15)
    # Add P-domain boundary
    pioneer_display[p_bound] = [1, 0.5, 0]  # orange boundary
    ax.imshow(np.clip(pioneer_display, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])
    if row_idx == 0: ax.set_title(col_titles3[6], fontsize=9)

    # Scale bar
    bar_px = 5.0 / PIXEL_UM
    ax.plot([10, 10+bar_px], [act_d.shape[0]-15]*2, 'w-', lw=3)
    ax.text(10+bar_px/2, act_d.shape[0]-25, '5µm', color='white',
            ha='center', fontsize=7, fontweight='bold')

    del mip, mask, gc_mask, shaft_mask
    garbage_collect.collect()
    log(f"    Done")

plt.savefig(OUT / '03_top3_pdomain_showcase.png', dpi=200, bbox_inches='tight')
plt.close()
log("  Saved 03_top3_pdomain_showcase.png")

# ── FINAL SUMMARY ──
log("\n" + "="*80)
log("SUMMARY")
log("="*80)

sig_metrics = sdf[sdf.sig.isin(['*', '**', '***'])]
log(f"\n  {len(sig_metrics)} significant metrics out of {len(sdf)} tested:")
for _, r in sig_metrics.iterrows():
    log(f"    {r['metric']:50s} d={r['cohens_d']:+.3f} p={r['p_value']:.4f} {r['sig']}")

log(f"\n  Top 3 CHIR cells for P-domain S9-AcTub:")
for i, (_, r) in enumerate(chir_ranked.head(3).iterrows()):
    log(f"    {i+1}. {r['id']}: P-Pearson={r.get('p_domain_pearson_s9_tub',0):.3f}, "
        f"Pioneer S9 enrich={r.get('p_domain_s9_enrichment_pioneer',0):.2f}")

log(f"\n  Median control: {med_ctrl_id}: "
    f"P-Pearson={median_ctrl.get('p_domain_pearson_s9_tub',0):.3f}")

log("\nDone!")
