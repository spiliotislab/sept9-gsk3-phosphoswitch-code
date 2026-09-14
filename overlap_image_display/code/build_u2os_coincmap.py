"""
U2OS FIXED (paper's system): SEPT9 (filament+puncta) INTERSECT actin STRESS FIBERS, z-slices 1-3
(basal), Control vs CHIR. nd2 ch: idx0=640=actin, idx1=561=SEPT9, idx2=488=MT. Voxel 0.0384um/px.
SEPT9=box white-tophat filament+puncta (nucleus-spared); SF=faint-straight actin ridge.
Coincidence M1/M2 (1-voxel tol). Smooth upscaled render.  Output: u2os_sept9_sf_coincmap.png
NOTE z-slices: 0-based indices [2,3]; adjust ZLO/ZHI if 'slice 2-3' means 1-based.
"""
import os, numpy as np, nd2, tifffile
import _config as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import white_tophat, square, remove_small_objects
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
HERE=os.path.dirname(os.path.abspath(__file__))
def resolve(name):     # search OVERLAP_IMAGES, then OVERLAP_IMAGES/nd2, then ./nd2 beside the script
    for c in [os.path.join(C.IMAGES,name), os.path.join(C.IMAGES,"nd2",name), os.path.join(HERE,"nd2",name)]:
        if os.path.isfile(c): return c
    return os.path.join(HERE,"nd2",name)
FILES={"Control":resolve("U2OS_Cntrl_3.nd2"),"CHIR":resolve("CHIR_theOne006.nd2")}   # CHIR example swapped
CH_A,CH_M,CH_S=0,2,1; ZLO,ZHI=0,3          # FIXED U2OS (user-confirmed correct): 640=actin(0), 561=SEPT9(1), 488=MT(2); SLICES 1-3 (1-based) = python [0:3]
SLL=f"z{ZLO+1}-{ZHI}"                       # 1-based slice label for titles/filenames
BOXK=25; SIGMAS=[1.0,1.5,2.5,4.0,6.0]       # add 1.0 px -> catch THIN faint fibers (0.0384um/px)
def frangi2d(img):
    best=np.zeros_like(img,np.float32)
    for s in SIGMAS:
        g=ndi.gaussian_filter(img.astype(np.float32),s)
        Hxx=np.pad(g[:,2:]-2*g[:,1:-1]+g[:,:-2],((0,0),(1,1)));Hyy=np.pad(g[2:,:]-2*g[1:-1,:]+g[:-2,:],((1,1),(0,0)))
        gx=np.pad((g[:,2:]-g[:,:-2])/2,((0,0),(1,1)));Hxy=np.pad((gx[2:,:]-gx[:-2,:])/2,((1,1),(0,0)))
        Hxx*=s*s;Hyy*=s*s;Hxy*=s*s;tmp=np.sqrt((Hxx-Hyy)**2+4*Hxy**2+1e-12)
        l1=(Hxx+Hyy+tmp)/2;l2=(Hxx+Hyy-tmp)/2;sw=np.abs(l1)>np.abs(l2)
        a1=np.where(sw,l2,l1);a2=np.where(sw,l1,l2)
        Rb=np.abs(a1)/(np.abs(a2)+1e-9);S=np.sqrt(a1**2+a2**2+1e-12);c=0.5*float(S.max())+1e-9
        best=np.maximum(best,np.where(a2<0,np.exp(-Rb**2/0.5)*(1-np.exp(-S**2/(2*c*c))),0))
    return best
def ridge(img,fg,k,minpx):
    v=frangi2d(img);nz=v[fg&(v>0)]
    if nz.size<50: return np.zeros_like(fg)
    med=np.median(nz);mad=np.median(np.abs(nz-med));return remove_small_objects((v>med+k*1.4826*mad)&fg,minpx)

def analyze(path):
    a=nd2.imread(path)[ZLO:ZHI].astype(np.float32)          # (2,C,Y,X)
    A=a[:,CH_A].max(0); M=a[:,CH_M].max(0); S=a[:,CH_S].max(0)
    fg=ndi.gaussian_filter(A+S,2)>threshold_otsu(ndi.gaussian_filter(A+S,2))   # cells foreground
    fg=ndi.binary_fill_holes(fg)
    # SEPT9 = discrete FIBERS + PUNCTA only; DIFFUSE removed (esp. CHIR). Detect each explicitly:
    sridge=ridge(S,fg,1.1,4)                        # FIBERS: Frangi ridge, LOWER thr -> catch dim fiber-aligned SEPT9 (not diffuse)
    smn=ndi.gaussian_filter(S,6); nucb=remove_small_objects((smn>np.percentile(smn[fg],92))&fg,20000)
    nuc=nucb&~ndi.binary_dilation(sridge,iterations=2)
    th=white_tophat(S,square(25))                   # remove broad background
    thr=np.percentile(th[fg],92)                    # PUNCTA: keep brightest ~8% (INTENSITY-based -> drops DIM diffuse)
    puncta=remove_small_objects((th>thr)&fg,4)
    S9=remove_small_objects((sridge|(puncta&~nuc)),4)  # bright fibers + puncta; dim diffuse excluded
    S9=ndi.binary_closing(S9,structure=np.ones((3,3)))
    # actin fibers: subtract fuzzy haze, then ridge — MORE INCLUSIVE (catch shorter/curved fibers,
    # not just long straight ones); relaxed shape filter drops only round blobs; close gaps.
    A_bs=white_tophat(A,square(41))
    aR=ridge(A_bs,fg,0.5,10); SF=np.zeros_like(fg); lbl=label(aR)   # lower thr -> THIN/faint fibers (esp. control)
    for rp in regionprops(lbl):
        mn=rp.minor_axis_length
        if rp.major_axis_length>=14 and (rp.eccentricity>=0.85 or (mn>0 and rp.major_axis_length/mn>=1.7)): SF[lbl==rp.label]=True
    SF=ndi.binary_closing(SF,structure=np.ones((3,3)))            # reconnect fiber gaps
    d=lambda m:ndi.binary_dilation(m,iterations=1); COINC=S9&d(SF)
    m1=COINC.sum()/(S9.sum()+1e-9); m2=(SF&d(S9)).sum()/(SF.sum()+1e-9)
    return dict(A=A,M=M,S=S,fg=fg,S9=S9,SF=SF,COINC=COINC,m1=m1,m2=m2)

R={k:analyze(p) for k,p in FILES.items()}
for k,r in R.items(): print(f"{k}: SEPT9vox={int(r['S9'].sum())} SFvox={int(r['SF'].sum())} coinc={int(r['COINC'].sum())} M1(SEPT9onSF)={r['m1']:.3f} M2(SFwithSEPT9)={r['m2']:.3f}",flush=True)

UP=2
def up(x): return ndi.zoom(x.astype(np.float32),UP,order=1)
def glow(m,s=1.4): return np.clip(ndi.gaussian_filter(up(m),s),0,1)
def n01(x,fg): lo,hi=np.percentile(x[fg],(2,99.5)); return np.clip((x-lo)/(hi-lo+1e-9),0,1)
COLT=["actin(640) — verify stress fibers","SEPT9(561) — verify puncta/filaments",
      "SF mask (red)","SEPT9 mask (green)","coincidence SEPT9∩SF (yellow)"]
fig,ax=plt.subplots(2,5,figsize=(21,8.6))
for i,(k,r) in enumerate(R.items()):
    ai=n01(r["A"],r["fg"]); si=n01(r["S"],r["fg"])
    ax[i,0].imshow(ndi.gaussian_filter(up(ai),0.8),cmap="gray",interpolation="bilinear")
    ax[i,1].imshow(ndi.gaussian_filter(up(si),0.8),cmap="gray",interpolation="bilinear")
    o=np.dstack([up(ai)*.6]*3); rr=glow(r["SF"]); o=o*(1-rr[...,None])+np.array([1,.15,.15])*rr[...,None]; ax[i,2].imshow(np.clip(o,0,1),interpolation="bilinear")
    o=np.dstack([up(si)*.6]*3); gg=glow(r["S9"]); o=o*(1-gg[...,None])+np.array([.1,.9,.2])*gg[...,None]; ax[i,3].imshow(np.clip(o,0,1),interpolation="bilinear")
    bg=up(ai)*.32; o=np.dstack([bg,bg,bg]); cg=glow(r["COINC"],1.2); o=o*(1-cg[...,None])+np.array([1,.95,.12])*cg[...,None]
    ax[i,4].imshow(np.clip(o,0,1),interpolation="bilinear")
    ax[i,0].set_ylabel(f"{k}\nM1(SEPT9 on SF)={r['m1']:.2f}\nM2(SF with SEPT9)={r['m2']:.2f}",fontsize=10)
    for j in range(5): ax[i,j].set_xticks([]);ax[i,j].set_yticks([])
for j,c in enumerate(COLT): ax[0,j].set_title(c,fontsize=10)
fig.suptitle(f"U2OS FIXED — SEPT9 ∩ actin stress fibers, slices {ZLO+1}-{ZHI} (basal): Control vs CHIR",fontsize=13)
plt.tight_layout(); plt.savefig(C.out("u2os_sept9_sf_coincmap.png"),dpi=140,bbox_inches="tight"); plt.close(); print("saved u2os_sept9_sf_coincmap.png")

# ---- ImageJ-editable TIFFs (raw channels + editable masks), per condition ----
IJ=C.out("imagej"); os.makedirs(IJ,exist_ok=True)
for k,r in R.items():
    raw=np.stack([r["A"],r["S"],r["M"]]).astype(np.uint16)              # actin, SEPT9, MT
    tifffile.imwrite(os.path.join(IJ,f"{k}_raw_actin-SEPT9-MT_{SLL}.tif"),raw,imagej=True,
                     metadata={"axes":"CYX","mode":"composite","Labels":["actin(640)","SEPT9(561)","MT(488)"]})
    masks=(np.stack([r["SF"],r["S9"],r["COINC"]])*255).astype(np.uint8) # editable binary masks
    tifffile.imwrite(os.path.join(IJ,f"{k}_masks_SF-SEPT9-coinc_{SLL}.tif"),masks,imagej=True,
                     metadata={"axes":"CYX","mode":"composite","Labels":["SF mask","SEPT9 mask","coincidence"]})
    print(f"saved ImageJ TIFFs for {k}: raw(3ch 16-bit) + masks(3ch 8-bit 0/255) -> {IJ}")
