"""
Triple colocalization: GSKp-488 × SEPT9-561 × Tubulin-640
Conditions: CONTL (n=12) vs CHIR (n=12)
Single-plane deconvolved crops

Channel order (confirmed from metadata):
  Ch0 = Tubulin  640nm
  Ch1 = SEPT9    561nm
  Ch2 = GSKp     488nm
  Ch3 = Actin    405nm  (not used here)

Metrics per cell (GC + Shaft separately):
  1. Costes thresholds for each pair
  2. Thresholded Manders (tM1, tM2) for all 3 pairs
  3. Triple overlap fraction  = pixels above threshold in ALL 3 channels
  4. tM_S9_triple   = SEPT9 colocalizing with BOTH GSKp AND Tubulin
  5. tM_GSKp_triple = GSKp colocalizing with BOTH SEPT9 AND Tubulin
  6. Pearson for all 3 pairs
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import nd2
from scipy import ndimage, stats
from scipy.ndimage import gaussian_filter, binary_dilation
from skimage import morphology, filters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import _config as C
OUTDIR = C.out("triple_coloc")
os.makedirs(OUTDIR, exist_ok=True)

CH_TUB=C.CH4_TUB; CH_S9=C.CH4_S9; CH_GSKp=C.CH4_GSK; CH_ACT=C.CH4_ACT

FILES = {
    'CONTL': [
        r'01326-CONTL-00002 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00003 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00004 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00005 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00006 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00007 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00008 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00009.nd2',
        r'01326-CONTL-00010 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00012 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CONTL-00013 - Denoised.nd2 - Deconvolved, Type Automatic_crop.nd2',
        r'01326-CONTL-00014 - Denoised.nd2 - Deconvolved, Type Automatic_crop.nd2',
    ],
    'CHIR': [
        r'01326-CHIR-00 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CHIR-00001 - Denoised.nd2 - Deconvolved, Type Automatic_crop.nd2',
        r'01326-CHIR-00002 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CHIR-00003 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CHIR-00004 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CHIR-00005 - Denoised.nd2 - Deconvolved, Type Automatic_crop.nd2',
        r'01326-CHIR-00006 - Denoised - Deconvolved, Type Automatic.nd2',
        r'01326-CHIR-00007 - Denoised.nd2 - Deconvolved, Type Automatic_crop.nd2',
        r'01326-CHIR-00008 - Denoised.nd2 - Deconvolved, Type Automatic_crop.nd2',
        r'01326-CHIR-00011 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CHIR-00012 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
        r'01326-CHIR-00013 - Denoised.nd2 - Deconvolved, Type Automatic.nd2',
    ],
}

# Resolve basenames against the 4-channel single-plane crops folder (PGSK_IMAGES2)
_IMG2 = C.require_images2()
FILES = {cond: [os.path.join(_IMG2, name) for name in names] for cond, names in FILES.items()}

COND_COLORS = {'CONTL': '#4878CF', 'CHIR': '#D65F5F'}

# ── Helpers ────────────────────────────────────────────────────────────────────
def norm01(img, plow=0.5, phigh=99.5):
    lo, hi = np.percentile(img, plow), np.percentile(img, phigh)
    return np.clip((img.astype(float)-lo)/(hi-lo+1e-9), 0, 1)

def get_mask(tub, sigma=1.5, min_size=500):
    sm    = gaussian_filter(tub.astype(float), sigma)
    mask  = sm > filters.threshold_otsu(sm)*0.6
    mask  = binary_dilation(mask, iterations=3)
    mask  = ndimage.binary_fill_holes(mask)
    mask  = morphology.remove_small_objects(mask.astype(bool), min_size=min_size)
    return mask.astype(bool)

def segment_gc_shaft(mask):
    if mask.sum() == 0:
        return mask.copy(), mask.copy()
    dist   = ndimage.distance_transform_edt(mask)
    n_er   = max(2, int(np.percentile(dist[mask], 25)*0.8))
    core   = ndimage.binary_erosion(mask, iterations=n_er)
    core   = morphology.remove_small_objects(core.astype(bool), min_size=200)
    shaft  = binary_dilation(core, iterations=n_er+2) & mask
    gc     = mask & ~shaft
    if gc.sum() < 200:
        cy, cx = ndimage.center_of_mass(mask)
        ys, xs = np.where(mask)
        d = np.sqrt((ys-cy)**2+(xs-cx)**2)
        gc = np.zeros_like(mask); gc[ys[d>np.median(d)*0.6]] = True
        shaft = mask & ~gc
    return gc.astype(bool), shaft.astype(bool)

def costes_threshold(a_flat, b_flat, n_steps=256):
    a, b = a_flat.astype(float), b_flat.astype(float)
    slope, intercept, *_ = stats.linregress(a, b)
    step = (a.max()-a.min()) / n_steps
    t1   = a.max()
    for _ in range(n_steps):
        t1 -= step
        t2  = slope*t1 + intercept
        below = (a < t1) & (b < t2)
        if below.sum() < 10: continue
        r, _ = stats.pearsonr(a[below], b[below])
        if r <= 0: break
    return float(t1), float(slope*t1+intercept)

def tM(ch1, ch2, mask, t1, t2):
    a, b = ch1[mask].astype(float), ch2[mask].astype(float)
    return (a[b>t2].sum()/(a.sum()+1e-12),
            b[a>t1].sum()/(b.sum()+1e-12))

# ── Per-file analysis ──────────────────────────────────────────────────────────
def analyze(fpath, cond):
    rows = []
    try:
        with nd2.ND2File(fpath) as f:
            arr = f.asarray()
    except Exception as e:
        print(f'    ERROR: {e}'); return rows

    if arr.ndim != 3 or arr.shape[0] < 3: return rows

    tub  = norm01(arr[CH_TUB])
    s9   = norm01(arr[CH_S9])

    # GSKp background subtraction before Costes — rolling-ball σ=40px removes
    # diffuse cytoplasmic background that inflates Costes threshold (user note)
    gskp_raw = arr[CH_GSKp].astype(float)
    gskp_bg  = gaussian_filter(gskp_raw, sigma=40)
    gskp     = norm01(np.clip(gskp_raw - gskp_bg, 0, None))

    mask = get_mask(tub)
    if mask.sum() < 300: return rows
    gc, shaft = segment_gc_shaft(mask)

    fname = os.path.basename(fpath)

    # Costes thresholds (on full mask)
    t_s9_tub   = costes_threshold(s9[mask],   tub[mask])
    t_gsk_tub  = costes_threshold(gskp[mask], tub[mask])
    t_gsk_s9   = costes_threshold(gskp[mask], s9[mask])

    for rname, rmask in [('GC', gc), ('Shaft', shaft)]:
        if rmask.sum() < 100: continue
        row = {'file': fname, 'condition': cond, 'region': rname,
               'n_pixels': int(rmask.sum())}

        # Pearson
        for n1,n2,a,b in [('S9','Tub',s9,tub),('GSKp','Tub',gskp,tub),('GSKp','S9',gskp,s9)]:
            av,bv = a[rmask].astype(float), b[rmask].astype(float)
            row[f'PCC_{n1}_{n2}'], _ = stats.pearsonr(av, bv) if len(av)>5 else (np.nan,None)

        # Thresholded Manders — pairwise
        for (n1,n2,a,b),(t1,t2) in [
            (('S9','Tub',s9,tub),   t_s9_tub),
            (('GSKp','Tub',gskp,tub), t_gsk_tub),
            (('GSKp','S9',gskp,s9),  t_gsk_s9),
        ]:
            m1, m2 = tM(a, b, rmask, t1, t2)
            row[f'tM1_{n1}_{n2}'] = m1
            row[f'tM2_{n1}_{n2}'] = m2
            row[f't1_{n1}_{n2}']  = t1
            row[f't2_{n1}_{n2}']  = t2

        # ── Triple colocalization ──────────────────────────────────────────────
        ts9, ttub, tgsk = t_s9_tub[0], t_s9_tub[1], t_gsk_tub[0]

        above_s9  = s9[rmask]   > ts9
        above_tub = tub[rmask]  > ttub
        above_gsk = gskp[rmask] > tgsk

        # Triple overlap fraction (all 3 above threshold / total masked pixels)
        triple_mask = above_s9 & above_tub & above_gsk
        row['triple_overlap_frac'] = float(triple_mask.sum() / rmask.sum())

        # tM_triple: fraction of each channel that overlaps with BOTH others
        s9_vals  = s9[rmask].astype(float)
        tub_vals = tub[rmask].astype(float)
        gsk_vals = gskp[rmask].astype(float)

        row['tM_S9_triple']   = s9_vals[above_tub & above_gsk].sum()  / (s9_vals.sum()  + 1e-12)
        row['tM_Tub_triple']  = tub_vals[above_s9  & above_gsk].sum() / (tub_vals.sum() + 1e-12)
        row['tM_GSKp_triple'] = gsk_vals[above_s9  & above_tub].sum() / (gsk_vals.sum() + 1e-12)

        # Mean intensities
        for name, ch in [('S9',s9),('Tub',tub),('GSKp',gskp)]:
            row[f'mean_{name}'] = float(ch[rmask].mean())

        rows.append(row)
    return rows

# ── Run ────────────────────────────────────────────────────────────────────────
print('=== Triple Colocalization: GSKp × SEPT9 × Tubulin ===\n')
all_rows = []
for cond, fpaths in FILES.items():
    print(f'{cond} ({len(fpaths)} files):')
    for fp in fpaths:
        fname = os.path.basename(fp)[:55]
        rows  = analyze(fp, cond)
        all_rows.extend(rows)
        print(f'  {fname}... {len(rows)} regions')

df = pd.DataFrame(all_rows)
csv_path = os.path.join(OUTDIR, 'triple_coloc_results.csv')
df.to_csv(csv_path, index=False)
print(f'\n{len(df)} rows saved → {csv_path}')

# ── Summary ────────────────────────────────────────────────────────────────────
KEY_METRICS = [
    ('triple_overlap_frac', 'Triple overlap\n(GSKp∩SEPT9∩Tub / total px)'),
    ('tM_S9_triple',        'tM SEPT9→(GSKp+Tub)\nSEPT9 on both others'),
    ('tM_GSKp_triple',      'tM GSKp→(SEPT9+Tub)\nGSKp on both others'),
    ('tM_Tub_triple',       'tM Tub→(SEPT9+GSKp)\nTub on both others'),
    ('tM1_S9_Tub',          'tM1 SEPT9→Tub\n(Costes)'),
    ('tM1_GSKp_Tub',        'tM1 GSKp→Tub\n(Costes)'),
    ('tM1_GSKp_S9',         'tM1 GSKp→SEPT9\n(Costes)'),
    ('PCC_S9_Tub',          'Pearson\nSEPT9 × Tub'),
    ('PCC_GSKp_Tub',        'Pearson\nGSKp × Tub'),
    ('PCC_GSKp_S9',         'Pearson\nGSKp × SEPT9'),
]

print('\n=== KEY RESULTS ===')
for metric, label in KEY_METRICS[:4]:
    print(f'\n{label.replace(chr(10)," — ")}:')
    for region in ['GC','Shaft']:
        sub = df[df.region==region]
        for cond in ['CONTL','CHIR']:
            vals = sub[sub.condition==cond][metric].dropna()
            print(f'  {cond} {region}: {vals.median():.3f} ± {vals.std():.3f}  (n={len(vals)})')
        c_v = sub[sub.condition=='CONTL'][metric].dropna().values
        ch_v= sub[sub.condition=='CHIR'][metric].dropna().values
        if len(c_v)>=3 and len(ch_v)>=3:
            _, p = stats.mannwhitneyu(c_v, ch_v, alternative='two-sided')
            print(f'  {region} CONTL vs CHIR p={p:.4f}')

# ── Figures ────────────────────────────────────────────────────────────────────
nrows = (len(KEY_METRICS)+2)//3
fig, axes = plt.subplots(nrows, 3, figsize=(18, nrows*4.5), facecolor='white')
fig.suptitle('Triple Colocalization: GSKp-488 × SEPT9-561 × Tubulin-640\nCONTL vs CHIR',
             fontsize=14, fontweight='bold', y=1.01)

for idx, (metric, label) in enumerate(KEY_METRICS):
    ax = axes.flat[idx]
    for region, offset, marker in [('GC',-0.15,'o'),('Shaft',0.15,'s')]:
        sub = df[df.region==region]
        for i, cond in enumerate(['CONTL','CHIR']):
            vals = sub[sub.condition==cond][metric].dropna().values
            if not len(vals): continue
            x   = i + offset
            med = np.median(vals)
            sem = vals.std()/np.sqrt(len(vals))
            col = COND_COLORS[cond]
            ax.errorbar(x, med, yerr=sem, fmt=marker, color=col, markersize=8,
                        capsize=4, lw=2,
                        markerfacecolor=col if region=='Shaft' else 'white',
                        markeredgewidth=2)
            jit = np.random.uniform(-0.05,0.05,len(vals))
            ax.scatter(np.full(len(vals),x)+jit, vals, color=col, alpha=0.4, s=18, zorder=3)

    # p-value (GC)
    gc_sub = df[df.region=='GC']
    c_v  = gc_sub[gc_sub.condition=='CONTL'][metric].dropna().values
    ch_v = gc_sub[gc_sub.condition=='CHIR'][metric].dropna().values
    if len(c_v)>=3 and len(ch_v)>=3:
        _, p = stats.mannwhitneyu(c_v, ch_v, alternative='two-sided')
        pstr = f'GC p={p:.3f}' if p>=0.001 else 'GC p<0.001'
    else: pstr=''
    ax.set_title(f'{label}\n{pstr}', fontsize=9)
    ax.set_xticks([0,1]); ax.set_xticklabels(['CONTL','CHIR'])
    ax.set_ylim(0, min(1.05, df[metric].max()*1.15 if df[metric].max()>0 else 1))
    ax.axhline(0.5,color='gray',lw=0.5,ls=':',alpha=0.4)
    ax.spines[['top','right']].set_visible(False)

# Legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0],[0],marker='o',color=COND_COLORS['CONTL'],mfc=COND_COLORS['CONTL'],label='CONTL Shaft'),
    Line2D([0],[0],marker='o',color=COND_COLORS['CONTL'],mfc='white',mew=2,label='CONTL GC'),
    Line2D([0],[0],marker='o',color=COND_COLORS['CHIR'], mfc=COND_COLORS['CHIR'], label='CHIR Shaft'),
    Line2D([0],[0],marker='o',color=COND_COLORS['CHIR'], mfc='white',mew=2,label='CHIR GC'),
]
axes.flat[len(KEY_METRICS)-1].legend(handles=handles,fontsize=8,loc='upper right')

# Hide empty panels
for idx in range(len(KEY_METRICS), nrows*3):
    axes.flat[idx].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUTDIR,'triple_coloc_figure.png'), dpi=180, bbox_inches='tight')
plt.close()
print(f'\nFigure saved → {os.path.join(OUTDIR,"triple_coloc_figure.png")}')
print('=== Done ===')
