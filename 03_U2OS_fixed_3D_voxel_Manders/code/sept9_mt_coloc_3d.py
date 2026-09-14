"""
3D SEPT9 ↔ MT colocalization + filament-organization pipeline.
U2OS WT/mutant, NIS-Elements denoised+deconvolved ND2 stacks.

Input  : ND2 (Z, C=2, Y, X), uint16
           ch0 = 561 (Tubulin)      [NOTE: reversed vs MDCK pipeline]
           ch1 = 488 (GFP-SEPT9)
Output : per_file/<stem>.csv        — 1 row of metrics per cell (or per FOV)
         qc/<stem>_qc.png           — QC MIP overlay
         Plus a master.csv aggregation when run in batch mode.

STRATEGY (follows SEPT9_MT_CHIR pptx, adapted):
  1.  GPU load and preprocess (CuPy where useful)
  2.  Cell mask (MT+SEPT9 sum → GaussBlur → Otsu → 3D closing)
  3.  NUCLEUS mask (NEW) — where SEPT9 is diffuse-high AND MT is low:
          nuc_score = smooth(SEPT9,σ=10) * (1 - smooth(MT,σ=10))
          inside cell mask → triangle threshold → largest 3D component
      Two flavours of every score reported:
          _wholecell   (all voxels in cell mask)
          _exNuc       (cell minus nucleus)  <-- headline
  4.  Background subtraction
          SEPT9: white top-hat, ball r=3 vox in xy (~115 nm)  -> kills diffuse
          MT   : Gaussian high-pass σxy=12                   -> kills centrosomal flare
  5.  Frangi 3D tubeness (skimage, anisotropy-aware) σ = 1, 1.5, 2 vox
  6.  Li threshold → binary filament masks, clipped to cell-ex-nucleus
  7.  3D skeletonize (Lee) → SEPT9_skel, MT_skel
  8.  Intensity coloc restricted to MT mask:
          Pearson, Manders M1/M2, Costes sanity, enrichment ratios
  9.  Skeleton-overlap (HEADLINE):
          |SEP_skel ∩ dil(MT_mask, 2 vox)| / |SEP_skel|
          and reciprocal (% MT skel covered by SEP dilated)
 10.  Orientation alignment:
          At joint voxels, Hessian (σ=1.5 vox) smallest-|λ| eigenvector
          → physical-scale tangent → |cos θ| to nearest MT tangent
          Reported: median |cos θ|, fraction within 20°
 11.  Filament organization (NEW):
          SEP skeleton length (µm), # components, length distribution
          branch-point and end-point counts
          structure-tensor coherence (anisotropy) over skeleton voxels
          mean straightness (chord / arc)
          network density (skel voxel fraction per cell volume)
          per-component MT-association fraction
 12.  QC PNG: MIP of SEP/MT skeletons + cell/nucleus contours
 13.  CSV row written; existing CSVs are skipped (resumable).

GPU usage:  CuPy + cupyx.scipy.ndimage for  Gaussian, top-hat, high-pass,
            binary dilation/closing, mask ops.  Frangi + skeletonize +
            Li threshold stay on CPU (skimage, well-tested reference).
            Keeps RAM < 4 GB and per-file wallclock ~20-40 s.

Run:
    python sept9_mt_coloc_3d.py                 # batch all files in this folder
    python sept9_mt_coloc_3d.py <file.nd2>      # single file
"""
import os, sys, io, time, json, glob, math, warnings, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True, errors='replace')
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import nd2
import tifffile
from scipy import ndimage as ndi
from skimage import filters, morphology, measure, exposure
from skimage.filters import frangi, threshold_li, threshold_otsu, threshold_triangle
from skimage.morphology import skeletonize, binary_dilation, binary_closing, ball
import _config as C

# ── GPU (CuPy) — fall back to CPU if missing ────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cnd
    _ = cp.zeros(1)
    GPU = True
except Exception as e:
    print(f'[warn] CuPy unavailable ({e}) — falling back to CPU ndimage')
    GPU = False

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FOLDER = C.IMAGES            # set via U2OS_IMAGES; may be "" until you point it at your .nd2 folder
OUT_DIR = Path(C.out('sept9_mt_coloc_3d'))
PER_FILE_DIR = OUT_DIR / 'per_file'
QC_DIR = OUT_DIR / 'qc'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PER_FILE_DIR.mkdir(exist_ok=True)
QC_DIR.mkdir(exist_ok=True)

MT_CH, SEP_CH = 0, 1                          # ch0=561=MT, ch1=488=SEP9
FRANGI_SIGMAS = (1.5, 2.0, 3.0)               # vox (xy) — avoid sub-pixel σ that picks up puncta
CELL_SIGMA = 2.0                              # vox (xy) for cell mask smoothing
NUC_SIGMA = 10.0                              # vox (xy) for nuclear blob smoothing
TOPHAT_RADIUS = 3                             # vox (xy)
MT_HIGHPASS_SIGMA = 12.0                      # vox
MT_DILATE_VOX = 2                             # ~1× lateral PSF FWHM tolerance
ALIGN_HESSIAN_SIGMA = 1.5                     # vox
ANGLE_WITHIN_DEG = 20.0                       # alignment cutoff
HYSTERESIS_LOW_FRAC = 0.25                    # low threshold = frac × Li  (aggressive growth from seeds)
MIN_FILAMENT_UM = 0.15                        # drop skeleton components shorter than this (rejects single-vox puncta)
MASK_CLOSE_R_XY = 2                           # vox — bridge sub-PSF gaps before skeletonize
FRANGI_KEEP_FRAC = 0.05                       # keep top 5% of Frangi-vesselness voxels within cell (= threshold at P95)

# ─────────────────────────────────────────────────────────────────────────────
# GPU helpers (Gaussian, top-hat, morphology) with CPU fallback
# ─────────────────────────────────────────────────────────────────────────────
def gauss(vol, sigma):
    """Anisotropy-aware 3D Gaussian, sigma in XY vox; Z sigma scaled down."""
    if isinstance(sigma, (int, float)):
        s_z = sigma / Z_ANISO
        sig = (s_z, sigma, sigma)
    else:
        sig = sigma
    if GPU:
        return cp.asnumpy(cnd.gaussian_filter(cp.asarray(vol, dtype=cp.float32), sig))
    return ndi.gaussian_filter(vol.astype(np.float32), sig)

def tophat_ball(vol, r_xy):
    """White top-hat with xy-ball, r in xy voxels. Z radius matched to xy-μm."""
    r_z = max(1, int(round(r_xy * VOXEL_XY / VOXEL_Z)))
    # Build ellipsoidal structuring element
    zz, yy, xx = np.ogrid[-r_z:r_z+1, -r_xy:r_xy+1, -r_xy:r_xy+1]
    se = ((zz/r_z)**2 + (yy/r_xy)**2 + (xx/r_xy)**2) <= 1.0
    if GPU:
        se_g = cp.asarray(se)
        v = cp.asarray(vol, dtype=cp.float32)
        opened = cnd.grey_opening(v, footprint=se_g)
        return cp.asnumpy(v - opened)
    return ndi.white_tophat(vol, footprint=se)

def highpass(vol, sigma_xy):
    """Gaussian high-pass (subtract blurred from self)."""
    return vol.astype(np.float32) - gauss(vol, sigma_xy)

# ── GPU 3D Frangi ────────────────────────────────────────────────────────────
def _sym3_eig_gpu(Hzz, Hyy, Hxx, Hyz, Hxz, Hxy):
    """Analytical eigenvalues of symmetric 3x3 matrix (Smith/Cardano),
    elementwise on CuPy arrays.  Returns (w1, w2, w3) unsorted.
    All inputs CuPy float32.  Numerically stable."""
    p1 = Hyz*Hyz + Hxz*Hxz + Hxy*Hxy
    q  = (Hzz + Hyy + Hxx) / 3.0
    # A - q*I
    A00 = Hzz - q; A11 = Hyy - q; A22 = Hxx - q
    p2 = A00*A00 + A11*A11 + A22*A22 + 2.0*p1
    p  = cp.sqrt(p2 / 6.0) + 1e-20
    # det of B = (A - qI)/p  — expanded symmetric 3x3 det
    # det(A-qI) = A00*(A11*A22 - Hxy*Hxy) - Hyz*(Hyz*A22 - Hxy*Hxz)
    #           + Hxz*(Hyz*Hxy - A11*Hxz)
    detA = (A00*(A11*A22 - Hxy*Hxy)
            - Hyz*(Hyz*A22 - Hxy*Hxz)
            + Hxz*(Hyz*Hxy - A11*Hxz))
    r = detA / (2.0 * p**3)
    r = cp.clip(r, -1.0, 1.0)
    phi = cp.arccos(r) / 3.0
    w1 = q + 2.0*p*cp.cos(phi)
    w3 = q + 2.0*p*cp.cos(phi + (2.0*cp.pi/3.0))
    w2 = 3.0*q - w1 - w3
    return w1, w2, w3

def frangi3d_gpu(vol_np, sigmas=FRANGI_SIGMAS, alpha=0.5, beta=0.5,
                 black_ridges=False):
    """GPU 3D Frangi tubeness. Returns np.float32 (Z,Y,X), max over scales."""
    if not GPU:
        return frangi(vol_np.astype(np.float32), sigmas=sigmas,
                      alpha=0.5, beta=0.5, gamma=None, black_ridges=black_ridges)
    v = cp.asarray(vol_np, dtype=cp.float32)
    out = cp.zeros_like(v)
    two_a2 = 2.0 * alpha * alpha
    two_b2 = 2.0 * beta  * beta
    for s in sigmas:
        sig = (s / Z_ANISO, s, s)
        g   = cnd.gaussian_filter(v, sigma=sig)
        # six Hessian components (scale-normalised by s^2)
        Hzz = cnd.gaussian_filter(g, sigma=sig, order=(2,0,0)) * (s*s)
        Hyy = cnd.gaussian_filter(g, sigma=sig, order=(0,2,0)) * (s*s)
        Hxx = cnd.gaussian_filter(g, sigma=sig, order=(0,0,2)) * (s*s)
        Hyz = cnd.gaussian_filter(g, sigma=sig, order=(1,1,0)) * (s*s)
        Hxz = cnd.gaussian_filter(g, sigma=sig, order=(1,0,1)) * (s*s)
        Hxy = cnd.gaussian_filter(g, sigma=sig, order=(0,1,1)) * (s*s)
        w1, w2, w3 = _sym3_eig_gpu(Hzz, Hyy, Hxx, Hyz, Hxz, Hxy)
        w1 = cp.nan_to_num(w1, nan=0, posinf=0, neginf=0)
        w2 = cp.nan_to_num(w2, nan=0, posinf=0, neginf=0)
        w3 = cp.nan_to_num(w3, nan=0, posinf=0, neginf=0)
        # sort |w1| <= |w2| <= |w3|
        aw = cp.stack([cp.abs(w1), cp.abs(w2), cp.abs(w3)], axis=0)
        order = cp.argsort(aw, axis=0)
        ws    = cp.stack([w1, w2, w3], axis=0)
        L1 = cp.take_along_axis(ws, order[0:1], axis=0)[0]
        L2 = cp.take_along_axis(ws, order[1:2], axis=0)[0]
        L3 = cp.take_along_axis(ws, order[2:3], axis=0)[0]
        aL2 = cp.abs(L2); aL3 = cp.abs(L3); aL1 = cp.abs(L1)
        Ra  = aL2 / (aL3 + 1e-10)
        Rb  = aL1 / (cp.sqrt(aL2 * aL3) + 1e-10)
        S   = cp.sqrt(L1*L1 + L2*L2 + L3*L3)
        # robust gamma: 50% of the 99.5th percentile of S (avoids NaN/outlier sensitivity)
        Sv = S.ravel()
        k = max(1, int(0.995 * Sv.size))
        s_sorted = cp.sort(Sv)
        gamma = 0.5 * float(s_sorted[k-1].item()) + 1e-10
        two_g2 = 2.0 * gamma * gamma
        V = ((1.0 - cp.exp(-Ra*Ra/two_a2)) *
             cp.exp(-Rb*Rb/two_b2) *
             (1.0 - cp.exp(-S*S/two_g2)))
        sign_mask = (L2 < 0) & (L3 < 0) if not black_ridges else (L2 > 0) & (L3 > 0)
        V = cp.where(sign_mask, V, 0.0)
        V = cp.nan_to_num(V, nan=0, posinf=0, neginf=0)
        out = cp.maximum(out, V)
    return cp.asnumpy(out)

def dilate3d(mask_bool, r_xy):
    r_z = max(1, int(round(r_xy * VOXEL_XY / VOXEL_Z)))
    zz, yy, xx = np.ogrid[-r_z:r_z+1, -r_xy:r_xy+1, -r_xy:r_xy+1]
    se = ((zz/max(r_z,1))**2 + (yy/r_xy)**2 + (xx/r_xy)**2) <= 1.0
    if GPU:
        out = cnd.binary_dilation(cp.asarray(mask_bool), structure=cp.asarray(se))
        return cp.asnumpy(out)
    return ndi.binary_dilation(mask_bool, structure=se)

def close3d(mask_bool, r_xy=5):
    r_z = max(1, int(round(r_xy * VOXEL_XY / VOXEL_Z)))
    zz, yy, xx = np.ogrid[-r_z:r_z+1, -r_xy:r_xy+1, -r_xy:r_xy+1]
    se = ((zz/max(r_z,1))**2 + (yy/r_xy)**2 + (xx/r_xy)**2) <= 1.0
    if GPU:
        m = cp.asarray(mask_bool)
        m = cnd.binary_dilation(m, structure=cp.asarray(se))
        m = cnd.binary_erosion (m, structure=cp.asarray(se))
        return cp.asnumpy(m)
    return ndi.binary_closing(mask_bool, structure=se)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────
def pearson(a, b, mask):
    a = a[mask].astype(np.float64); b = b[mask].astype(np.float64)
    if a.size < 3: return np.nan
    a -= a.mean(); b -= b.mean()
    n = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/n) if n > 0 else np.nan

def manders(I_sep, M_mt_bool):
    # Manders M1 = Σ I_sep·M_mt / Σ I_sep  (inside cell)
    I = I_sep.astype(np.float64); m = M_mt_bool
    s_all = I.sum()
    if s_all <= 0: return np.nan
    return float(I[m].sum() / s_all)

def hessian_tangent_3d(vol, sigma_xy):
    """At each voxel, compute the eigenvector of the Hessian with smallest |λ|
    (i.e. filament tangent). Returns (dz, dy, dx) float32 arrays."""
    s = (sigma_xy / Z_ANISO, sigma_xy, sigma_xy)
    g = gauss(vol, sigma_xy)
    # second derivatives (index order z,y,x)
    Hzz = ndi.gaussian_filter(g, s, order=(2,0,0))
    Hyy = ndi.gaussian_filter(g, s, order=(0,2,0))
    Hxx = ndi.gaussian_filter(g, s, order=(0,0,2))
    Hyz = ndi.gaussian_filter(g, s, order=(1,1,0))
    Hxz = ndi.gaussian_filter(g, s, order=(1,0,1))
    Hxy = ndi.gaussian_filter(g, s, order=(0,1,1))
    # flatten to voxel-list for eigvalsh
    shape = vol.shape
    H = np.empty(shape + (3,3), dtype=np.float32)
    H[...,0,0]=Hzz; H[...,1,1]=Hyy; H[...,2,2]=Hxx
    H[...,0,1]=H[...,1,0]=Hyz
    H[...,0,2]=H[...,2,0]=Hxz
    H[...,1,2]=H[...,2,1]=Hxy
    return H  # caller picks per-voxel eigvec where needed

def smallest_abs_eigvec(H):
    """H: (n,3,3) flat list of symmetric matrices → eigvec of smallest |λ| per voxel.
    Returns (n,3) in (ez,ey,ex) order."""
    w, v = np.linalg.eigh(H)            # w: (n,3) ascending; v: (n,3,3)
    idx = np.argmin(np.abs(w), axis=-1) # (n,)
    n = H.shape[0]
    return v[np.arange(n), :, idx]      # (n, 3)

# ─────────────────────────────────────────────────────────────────────────────
# QC overlay
# ─────────────────────────────────────────────────────────────────────────────
def qc_overlay(out_png, mt_raw, sep_raw, mt_skel, sep_skel, cell_mask, nuc_mask):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mip_mt  = mt_raw.max(0);   mip_sep = sep_raw.max(0)
    sk_mt   = mt_skel.max(0);  sk_sep  = sep_skel.max(0)
    cm      = cell_mask.max(0); nm = nuc_mask.max(0)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    ax[0].imshow(np.log1p(mip_mt),  cmap='gray');  ax[0].set_title('MT MIP')
    ax[1].imshow(np.log1p(mip_sep), cmap='gray');  ax[1].set_title('SEP9 MIP')
    rgb = np.zeros(mip_mt.shape + (3,), dtype=np.float32)
    rgb[..., 0] = sk_sep.astype(np.float32)   # SEP = red
    rgb[..., 1] = sk_mt .astype(np.float32)   # MT  = green
    ax[2].imshow(rgb); ax[2].set_title('skeletons: SEP=red MT=green')
    for a in ax:
        a.contour(cm, levels=[0.5], colors='cyan', linewidths=0.5)
        a.contour(nm, levels=[0.5], colors='magenta', linewidths=0.6)
        a.axis('off')
    fig.savefig(out_png, dpi=110); plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Main per-file pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_file(nd2_path):
    global VOXEL_XY, VOXEL_Z, Z_ANISO
    p = Path(nd2_path)
    stem = p.stem
    csv_out = PER_FILE_DIR / f'{stem}.csv'
    if csv_out.exists() and csv_out.stat().st_size > 64:
        print(f'   already done — skip')
        return pd.read_csv(csv_out)

    t0 = time.time()
    # ── load
    with nd2.ND2File(p) as f:
        vol = f.asarray()
        vx  = f.voxel_size()
    if vol.ndim != 4 or vol.shape[1] < 2:
        raise RuntimeError(f'{stem}: unexpected shape {vol.shape}')
    Z, C, Y, X = vol.shape
    VOXEL_XY = vx.x  # µm/px
    VOXEL_Z  = vx.z
    Z_ANISO  = VOXEL_Z / VOXEL_XY   # z step in xy-voxel units
    print(f'   shape=(Z={Z},C={C},Y={Y},X={X})  voxel={VOXEL_XY*1000:.1f} × {VOXEL_Z*1000:.1f} nm')

    mt  = vol[:, MT_CH ].astype(np.float32)
    sep = vol[:, SEP_CH].astype(np.float32)

    # ── 1. cell mask  (triangle is more forgiving than Otsu on multi-cell FOVs
    # where one cell is much brighter — Otsu sits above the dim cell's intensity)
    cell_sig = gauss(mt + sep, CELL_SIGMA)
    try:
        cell_thr = threshold_triangle(cell_sig)
    except Exception:
        cell_thr = threshold_otsu(cell_sig)
    cell_mask = cell_sig > cell_thr
    cell_mask = close3d(cell_mask, r_xy=5)
    # keep ALL components bigger than MIN_CELL_UM3 (multi-cell FOVs)
    MIN_CELL_UM3 = 50.0
    vox_um3 = VOXEL_XY**2 * VOXEL_Z
    min_vox = max(1, int(MIN_CELL_UM3 / vox_um3))
    lbl, nl = ndi.label(cell_mask)
    if nl >= 1:
        sizes = ndi.sum(cell_mask, lbl, index=np.arange(1, nl+1))
        keep = np.where(sizes >= min_vox)[0] + 1
        if keep.size == 0:  # fallback: keep largest
            keep = np.array([1 + int(np.argmax(sizes))])
        cell_mask = np.isin(lbl, keep)
        print(f'   cells kept: {keep.size} / {nl}  (min {MIN_CELL_UM3:.0f} µm³)')
    cell_vox = int(cell_mask.sum())
    cell_vol_um3 = cell_vox * VOXEL_XY**2 * VOXEL_Z
    print(f'   cell voxels = {cell_vox:,}  ({cell_vol_um3:,.0f} µm³)')

    # ── 2. nucleus mask (NEW)
    mt_blur  = gauss(mt , NUC_SIGMA)
    sep_blur = gauss(sep, NUC_SIGMA)
    mt_norm  = mt_blur / (mt_blur.max()  + 1e-6)
    sep_norm = sep_blur / (sep_blur.max() + 1e-6)
    nuc_score = sep_norm * (1.0 - mt_norm)
    nuc_score[~cell_mask] = 0
    thr = threshold_triangle(nuc_score[cell_mask])
    nuc_cand = nuc_score > thr
    # keep components above a minimum volume AND with MT sparser than cell median
    # (prevents misclassifying weak-MT cytoplasm as nucleus when MT staining is weak)
    min_nuc_vol_vox = max(100, int(50.0 / (VOXEL_XY**2 * VOXEL_Z)))
    mt_cell_median = float(np.median(mt_blur[cell_mask])) if cell_vox else 0.0
    lbl, nl = ndi.label(nuc_cand)
    nuc_mask = np.zeros_like(cell_mask)
    kept = 0; rejected_sparse = 0
    if nl:
        sizes = ndi.sum(nuc_cand, lbl, index=np.arange(1, nl+1))
        big = np.where(sizes >= min_nuc_vol_vox)[0]
        for i in big:
            comp = (lbl == (i+1))
            mt_in  = float(np.median(mt_blur[comp]))
            # require MT in nucleus to be < 70% of cell-wide median MT
            if mt_in < 0.70 * mt_cell_median:
                nuc_mask |= comp
                kept += 1
            else:
                rejected_sparse += 1
        if nuc_mask.any():
            nuc_mask = close3d(nuc_mask, r_xy=3)
    print(f'   nuc candidates: {nl}  kept={kept}  rejected_MT_not_sparse={rejected_sparse}')
    cell_exnuc = cell_mask & ~nuc_mask
    nuc_vox = int(nuc_mask.sum())
    print(f'   nucleus voxels = {nuc_vox:,}  ({nuc_vox/max(cell_vox,1)*100:.1f}% of cell)')

    # ── 3a. bleed-through correction:  SEP_true = max(0, SEP - alpha*MT)
    # alpha auto-estimated per FOV as the robust median slope (SEP-bg)/(MT-bg)
    # on SEP-bottom-25% voxels within cell_exnuc -- those least likely to have real SEP9.
    if cell_exnuc.any():
        sep_v = sep[cell_exnuc].astype(np.float64)
        mt_v  = mt [cell_exnuc].astype(np.float64)
        sep_bg = float(np.percentile(sep_v, 5))
        mt_bg  = float(np.percentile(mt_v , 5))
        sep_botq = sep_v <= np.percentile(sep_v, 25)
        valid = sep_botq & (mt_v > mt_bg + 1.0)
        if valid.sum() > 1000:
            ratios = (sep_v[valid] - sep_bg) / (mt_v[valid] - mt_bg)
            alpha_raw = float(np.median(ratios))
            # Real channel bleed for GFP from 561 is typically < 0.02. Larger values
            # almost always mean diffuse cytoplasmic SEP9 with high baseline,
            # NOT bleed-through. Skip correction in that case.
            alpha_bleed = float(np.clip(alpha_raw, 0.0, 0.02)) if alpha_raw < 0.04 else 0.0
        else:
            alpha_bleed = 0.0
    else:
        alpha_bleed = 0.0
    print(f'   bleed-through alpha = {alpha_bleed:.4f}  (SEP_corr = SEP - {alpha_bleed:.3f}*MT)')
    sep_corr = np.maximum(sep - alpha_bleed * mt, 0.0).astype(np.float32)

    # ── 3b. background subtraction (use bleed-corrected SEP)
    sep_bs = tophat_ball(sep_corr, TOPHAT_RADIUS)
    mt_bs  = highpass(mt, MT_HIGHPASS_SIGMA)
    mt_bs  = np.clip(mt_bs, 0, None)

    # ── 4. Frangi 3D (GPU; anisotropy-aware via Z_ANISO in sigmas)
    sep_fr = frangi3d_gpu(sep_bs, FRANGI_SIGMAS)
    mt_fr  = frangi3d_gpu(mt_bs , FRANGI_SIGMAS)

    # ── 5. Li threshold → filament masks, clipped to cell_exnuc
    def mask_from_frangi(fr, ref_mask, loose=False, hysteresis=False):
        """Percentile threshold by default: keep top FRANGI_KEEP_FRAC of Frangi-vesselness
        voxels in the reference mask (= threshold at quantile 1-FRANGI_KEEP_FRAC).
        This is robust across FOVs with different SEP9 dynamic range / baseline,
        unlike Li which inverts when the histogram shape changes.
        With hysteresis=True, grow seeds into fr > HYSTERESIS_LOW_FRAC*thr.
        With loose=True, fall back to P90."""
        v = fr[ref_mask]
        if v.size < 100 or not np.isfinite(v.max()) or v.max() <= 0:
            return np.zeros_like(fr, dtype=bool)
        if loose:
            thr_hi = np.percentile(v, 90)
        else:
            thr_hi = np.percentile(v, 100.0 * (1.0 - FRANGI_KEEP_FRAC))
        if not hysteresis:
            return (fr > thr_hi) & ref_mask
        thr_lo = HYSTERESIS_LOW_FRAC * thr_hi
        high = (fr > thr_hi) & ref_mask
        low  = (fr > thr_lo) & ref_mask
        lbl_lo, nlo = ndi.label(low)
        if nlo == 0:
            return high
        seed_ids = np.unique(lbl_lo[high])
        seed_ids = seed_ids[seed_ids > 0]
        return np.isin(lbl_lo, seed_ids)
    sep_mask = mask_from_frangi(sep_fr, cell_exnuc, loose=False, hysteresis=True)
    mt_mask  = mask_from_frangi(mt_fr , cell_exnuc, loose=False, hysteresis=True)
    # If MT mask is too small (poor staining), retry with looser threshold
    mt_frac = mt_mask.sum() / max(cell_exnuc.sum(), 1)
    if mt_frac < 0.01:                    # < 1 % of cell volume -> probably weak MT signal
        print(f'   MT mask only {mt_frac*100:.2f}% of cell — relaxing to P90')
        mt_mask = mask_from_frangi(mt_fr, cell_exnuc, loose=True)
    print(f'   Frangi max  SEP={sep_fr.max():.3g}  MT={mt_fr.max():.3g}')
    print(f'   mask vox    SEP={int(sep_mask.sum()):,}  MT={int(mt_mask.sum()):,}')

    # ── 6. close gaps, then 3D skeletonize (Lee), drop components < MIN_FILAMENT_UM
    sep_mask = close3d(sep_mask, r_xy=MASK_CLOSE_R_XY) & cell_exnuc
    mt_mask  = close3d(mt_mask , r_xy=MASK_CLOSE_R_XY) & cell_exnuc
    sep_skel = skeletonize(sep_mask)
    mt_skel  = skeletonize(mt_mask)
    min_vox_filament = max(3, int(round(MIN_FILAMENT_UM / VOXEL_XY)))  # length in xy-vox  (VOXEL_XY in µm)
    def _drop_short(skel, min_vox):
        if not skel.any():
            return skel
        lbl_s, n_s = ndi.label(skel, structure=np.ones((3,3,3), bool))  # 26-conn for gap-tolerant length
        sizes = ndi.sum(skel, lbl_s, index=np.arange(1, n_s+1))
        keep = np.where(sizes >= min_vox)[0] + 1
        if keep.size == 0:
            return np.zeros_like(skel)
        return np.isin(lbl_s, keep)
    sep_skel_raw_vox = int(sep_skel.sum())
    mt_skel_raw_vox  = int(mt_skel.sum())
    sep_skel = _drop_short(sep_skel, min_vox_filament)
    mt_skel  = _drop_short(mt_skel , min_vox_filament)
    print(f'   skel vox    SEP={sep_skel_raw_vox:,}→{int(sep_skel.sum()):,}  '
          f'MT={mt_skel_raw_vox:,}→{int(mt_skel.sum()):,}  (min {min_vox_filament} vox = {MIN_FILAMENT_UM} µm)')

    # ── 7. intensity colocalization
    cell_for_coloc = cell_exnuc
    r_pearson   = pearson(sep_bs, mt_bs, cell_for_coloc)
    r_pearson_raw = pearson(sep, mt, cell_for_coloc)
    m_mt_restrict  = cell_for_coloc & mt_mask
    manders_M1 = manders(sep_bs, m_mt_restrict)                      # SEP into MT
    manders_M2 = manders(mt_bs , cell_for_coloc & sep_mask)          # MT  into SEP

    # ── 8. skeleton-overlap (headline)
    mt_mask_dil = dilate3d(mt_mask, MT_DILATE_VOX)
    sep_skel_vox = int(sep_skel.sum())
    mt_skel_vox  = int(mt_skel.sum())
    sep_on_mt = int(np.logical_and(sep_skel, mt_mask_dil).sum())
    mt_on_sep = int(np.logical_and(mt_skel, dilate3d(sep_mask, MT_DILATE_VOX)).sum())
    skel_overlap_SEP_on_MT = sep_on_mt / max(sep_skel_vox, 1)
    skel_overlap_MT_on_SEP = mt_on_sep / max(mt_skel_vox , 1)
    # within-cell (includes nucleus) flavour
    sep_skel_all = skeletonize(mask_from_frangi(sep_fr, cell_mask))
    mt_mask_all  = mask_from_frangi(mt_fr, cell_mask)
    mt_mask_all_dil = dilate3d(mt_mask_all, MT_DILATE_VOX)
    overlap_wholecell = np.logical_and(sep_skel_all, mt_mask_all_dil).sum() / max(sep_skel_all.sum(), 1)

    # ── 9. orientation alignment (joint voxels only)
    joint = sep_mask & mt_mask_dil
    n_joint = int(joint.sum())
    if n_joint >= 50:
        # compute Hessians only around joint voxels (small bbox)
        zs, ys, xs = np.where(joint)
        # Use full-volume Hessians but pull only at joint voxels (vectorised)
        H_sep = hessian_tangent_3d(sep_bs, ALIGN_HESSIAN_SIGMA)[zs, ys, xs]   # (n,3,3)
        H_mt  = hessian_tangent_3d(mt_bs , ALIGN_HESSIAN_SIGMA)[zs, ys, xs]
        v_sep = smallest_abs_eigvec(H_sep)    # (n,3) (ez, ey, ex)
        v_mt  = smallest_abs_eigvec(H_mt)
        # physical-space scaling
        phys_scale = np.array([VOXEL_Z, VOXEL_XY, VOXEL_XY], dtype=np.float32)
        v_sep_p = v_sep * phys_scale; v_mt_p = v_mt * phys_scale
        v_sep_p /= np.linalg.norm(v_sep_p, axis=1, keepdims=True) + 1e-9
        v_mt_p  /= np.linalg.norm(v_mt_p , axis=1, keepdims=True) + 1e-9
        cos = np.abs(np.sum(v_sep_p * v_mt_p, axis=1))
        align_median = float(np.median(cos))
        align_frac_within = float((cos >= math.cos(math.radians(ANGLE_WITHIN_DEG))).mean())
    else:
        align_median = np.nan
        align_frac_within = np.nan

    # ── 10. filament organization (NEW)
    sep_length_um = sep_skel_vox * VOXEL_XY   # xy-lateral length proxy
    mt_length_um  = mt_skel_vox  * VOXEL_XY
    # connected components on SEP skeleton
    sep_lbl, n_sep_comp = ndi.label(sep_skel, structure=ndi.generate_binary_structure(3, 1))
    if n_sep_comp > 0:
        comp_sizes = ndi.sum(sep_skel, sep_lbl, index=np.arange(1, n_sep_comp+1))
        comp_len_um = comp_sizes * VOXEL_XY
        sep_med_comp_len = float(np.median(comp_len_um))
        sep_mean_comp_len = float(np.mean(comp_len_um))
    else:
        sep_med_comp_len = sep_mean_comp_len = np.nan
    # branch / end points (degree count on skeleton voxels)
    neigh = ndi.convolve(sep_skel.astype(np.uint8),
                         np.ones((3,3,3), np.uint8), mode='constant') - sep_skel.astype(np.uint8)
    deg = neigh[sep_skel]
    sep_endpoints = int((deg == 1).sum())
    sep_branchpts = int((deg >= 3).sum())
    # per-component MT association
    if n_sep_comp > 0:
        assoc = ndi.sum((sep_skel & mt_mask_dil).astype(np.uint8),
                        sep_lbl, index=np.arange(1, n_sep_comp+1))
        comp_mt_frac = assoc / np.maximum(comp_sizes, 1)
        sep_comp_mt_decorated = float((comp_mt_frac >= 0.5).mean())
    else:
        sep_comp_mt_decorated = np.nan
    # structure-tensor coherence on SEP skeleton voxels
    if sep_skel_vox >= 100:
        # tiny 3D structure tensor using the Hessian tangents we already computed? compute fresh
        H = hessian_tangent_3d(sep_bs, 1.0)
        ws, vs = np.linalg.eigh(H[sep_skel])   # (n,3), (n,3,3)
        # coherence = (|λmax| - |λmid|) / (|λmax| + |λmid|)
        abw = np.sort(np.abs(ws), axis=1)[:, ::-1]
        coh = (abw[:, 0] - abw[:, 1]) / (abw[:, 0] + abw[:, 1] + 1e-9)
        sep_coherence = float(np.median(coh))
    else:
        sep_coherence = np.nan
    sep_density = sep_skel_vox / max(cell_vox - nuc_vox, 1)
    mt_density  = mt_skel_vox  / max(cell_vox - nuc_vox, 1)

    # ── 11. QC overlay
    try:
        qc_overlay(QC_DIR / f'{stem}_qc.png',
                   mt, sep, mt_skel, sep_skel, cell_mask, nuc_mask)
    except Exception as e:
        print(f'   QC failed: {e}')

    # ── 12. assemble CSV row
    row = dict(
        file=stem,
        voxel_xy_nm=VOXEL_XY*1000, voxel_z_nm=VOXEL_Z*1000, Z=Z, Y=Y, X=X,
        cell_vox=cell_vox, cell_vol_um3=cell_vol_um3,
        nuc_vox=nuc_vox, nuc_frac_of_cell=nuc_vox/max(cell_vox,1),
        alpha_bleed_MT_to_SEP=alpha_bleed,
        # intensity coloc
        pearson_raw_exNuc=r_pearson_raw, pearson_bgSub_exNuc=r_pearson,
        manders_M1_SEP_in_MT_exNuc=manders_M1,
        manders_M2_MT_in_SEP_exNuc=manders_M2,
        # HEADLINE: skeleton overlap
        skel_SEP_on_MT_exNuc=skel_overlap_SEP_on_MT,
        skel_MT_on_SEP_exNuc=skel_overlap_MT_on_SEP,
        skel_SEP_on_MT_wholecell=float(overlap_wholecell),
        # alignment
        align_median_cosTheta=align_median,
        align_frac_within_20deg=align_frac_within,
        # filament organization
        sep_skel_length_um=sep_length_um,
        mt_skel_length_um=mt_length_um,
        sep_components=n_sep_comp,
        sep_med_component_length_um=sep_med_comp_len,
        sep_mean_component_length_um=sep_mean_comp_len,
        sep_endpoints=sep_endpoints,
        sep_branchpoints=sep_branchpts,
        sep_density_vox_per_cellExNuc=sep_density,
        mt_density_vox_per_cellExNuc=mt_density,
        sep_coherence_median=sep_coherence,
        sep_frac_components_MTdecorated=sep_comp_mt_decorated,
        # timing
        wallclock_s=round(time.time() - t0, 1),
    )
    df = pd.DataFrame([row])
    df.to_csv(csv_out, index=False)
    print(f'   wrote {csv_out.name}  ({row["wallclock_s"]} s)')
    print(f'   HEADLINE skel_SEP_on_MT_exNuc = {skel_overlap_SEP_on_MT:.3f}')
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Batch entry
# ─────────────────────────────────────────────────────────────────────────────
def main(argv):
    global OUT_DIR, PER_FILE_DIR, QC_DIR, MT_CH, SEP_CH
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='*', help='Optional explicit nd2 files')
    ap.add_argument('--folder', default=DEFAULT_FOLDER)
    ap.add_argument('--out', default=None, help='Output directory (default: <folder>/analysis_sept9_mt_3d)')
    ap.add_argument('--mt-ch', type=int, default=None, help='MT channel index (default: 0)')
    ap.add_argument('--sep-ch', type=int, default=None, help='SEP channel index (default: 1)')
    ap.add_argument('--glob', default=None, help='Override file glob within --folder')
    args = ap.parse_args(argv)
    if args.mt_ch is not None: MT_CH = args.mt_ch
    if args.sep_ch is not None: SEP_CH = args.sep_ch
    print(f'   channels: MT_CH={MT_CH}  SEP_CH={SEP_CH}')

    # Resolve input folder: explicit files > --folder > U2OS_IMAGES (friendly error if unset)
    if not args.files and not args.folder:
        args.folder = C.require_images()
    # Allow per-condition output dirs
    base_folder = Path(args.files[0]).parent if args.files else Path(args.folder)
    OUT_DIR = Path(args.out) if args.out else Path(C.out('sept9_mt_coloc_3d'))
    PER_FILE_DIR = OUT_DIR / 'per_file'
    QC_DIR = OUT_DIR / 'qc'
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    PER_FILE_DIR.mkdir(exist_ok=True)
    QC_DIR.mkdir(exist_ok=True)
    print(f'   OUT_DIR = {OUT_DIR}')

    if args.files:
        files = [Path(f) for f in args.files]
    elif args.glob:
        files = sorted(glob.glob(os.path.join(args.folder, args.glob)))
    else:
        # default batch list = decon nd2 files in folder
        patt = os.path.join(args.folder,
                            '*_crop - Denoised.nd2 - Deconvolved 20 iterations, Type Richardson-Lucy.nd2')
        patt2 = os.path.join(args.folder,
                             '*crop - Denoised.nd2 - Deconvolved 20 iterations, Type Richardson-Lucy.nd2')
        files = sorted(set(glob.glob(patt)) | set(glob.glob(patt2)))
        files = [Path(f) for f in files]

    print(f'=== sept9_mt_coloc_3d ===   GPU={GPU}   files={len(files)}')
    rows = []
    for i, p in enumerate(files, 1):
        print(f'\n[{i}/{len(files)}] {p.name}')
        try:
            df = process_file(p)
            rows.append(df)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'   ERROR: {e}')

    if rows:
        master = pd.concat(rows, ignore_index=True)
        master.to_csv(OUT_DIR / 'master.csv', index=False)
        print(f'\nmaster.csv written  ({len(master)} rows)')
        cols = ['file', 'skel_SEP_on_MT_exNuc',
                'manders_M1_SEP_in_MT_exNuc',
                'align_median_cosTheta',
                'sep_skel_length_um', 'sep_components',
                'nuc_frac_of_cell']
        print(master[cols].to_string(index=False))

if __name__ == '__main__':
    main(sys.argv[1:])
