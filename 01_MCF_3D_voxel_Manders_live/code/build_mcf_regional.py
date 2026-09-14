"""
3D voxel-Manders colocalization for ONE XY_4 cell given as two z-substacks:
  ROI-21-27 = BASAL (ventral stress fibers)   ROI-7-20 = APICAL (MT, nucleus)
SEPT9 = filaments+puncta only (box-K25 white top-hat removes diffuse; nucleus excluded).
Masks (3D, per-z Frangi): SEPT9 (filament+puncta), stress fibers (bright elongated actin),
MT (Frangi network). 3D Manders M1 coloc (1-voxel tolerance): SEPT9 on MT and on stress fibers.
Channels ch0=actin ch1=MT ch2=SEPT9. Voxel 0.108xy/0.30z.
Outputs: xy4_coloc_QC.png (mask check FIRST), xy4_coloc_metrics.txt
"""
import os, glob, numpy as np, tifffile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import white_tophat, square, disk, remove_small_objects, skeletonize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

import _config as C
HERE=os.path.dirname(os.path.abspath(__file__))
DD=os.environ.get("MCF_XY4_DIR", C.IMAGES or os.path.join(HERE,"crops"))   # hand-cropped BASAL/APICAL pairs
FRAME=os.environ.get("FRAME","003")                       # CHIR timepoint (003=early, 028=later)
ba=glob.glob(os.path.join(DD,f"BASAL-*-{FRAME}_decon*.tif")); ap=glob.glob(os.path.join(DD,f"APICAL-*-{FRAME}_decon*.tif"))
assert ap and ba,(FRAME,ap,ba)
bp=os.path.basename(ba[0]).split("-Jul")[0].strip("-")
STACKS={"BASAL (stress fibers)":ba[0], "APICAL (MT)":ap[0]}
print(f"FRAME {FRAME}: basal={os.path.basename(ba[0])}  apical={os.path.basename(ap[0])}")
METRIC=("METRIC = 3D object/voxel Manders M1: frac of SEPT9 filament+puncta VOXELS within 1 voxel of the "
 "MT / stress-fiber binary mask. SEPT9=box-K25 tophat (filament+puncta; diffuse+smooth-nucleus removed, "
 "sub-nuclear fibers KEPT); SF=faint-straight EXCLUSIVE actin (not near MT); MT=Frangi. Transfected-cell "
 "mask from 488. MANUAL basal/apical ROI z-crops. *** NOT comparable to prior ELIAS metrics: coverage-ratio "
 "P (heterogeneity), intensity-enrichment P3d (partition_3d), or sf_mt intensity-fraction — CHIR result is "
 "metric-dependent; keep this time course internally consistent (this metric only). ***")
MTAG="voxel-Manders-M1 (obj-based, 1vox tol) — NOT coverage-P / enrich-P3d / sf_mt"
CH_A,CH_M,CH_S=0,1,2; BOXK=25; SIGMAS=[1.0,1.5,2.5,4.0]

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
def ridge3d(vol,cellv,k,minpx):  # per-z frangi -> 3D mask
    out=np.zeros(vol.shape,bool)
    for z in range(vol.shape[0]):
        v=frangi2d(vol[z]); cv=cellv[z]; nz=v[cv&(v>0)]
        if nz.size<20: continue
        med=np.median(nz);mad=np.median(np.abs(nz-med)); out[z]=remove_small_objects((v>med+k*1.4826*mad)&cv,minpx)
    return out

def analyze(path):
    a=tifffile.imread(path).astype(np.float32); Z=a.shape[0]
    A,M,S=a[:,CH_A],a[:,CH_M],a[:,CH_S]
    # TRANSFECTED-cell mask from SEPT9(488): untransfected neighbors are dark in 488 -> excluded
    sm=ndi.gaussian_filter(S,(1,3,3)); cellv=sm>threshold_otsu(sm)
    cellv=ndi.binary_closing(cellv,structure=np.ones((1,3,3)),iterations=2)
    cellv=np.stack([ndi.binary_fill_holes(cellv[z]) for z in range(Z)])
    lab,n=ndi.label(cellv)                                   # keep largest 3D component = the transfected cell
    if n>1:
        sizes=ndi.sum(np.ones_like(lab),lab,range(1,n+1)); cellv=lab==(1+int(np.argmax(sizes)))
    cellv=ndi.binary_dilation(cellv,structure=np.ones((1,3,3)),iterations=1)
    # SEPT9 filament ridge (per z) — used to PROTECT sub-nuclear fibers from nucleus removal
    s9ridge=np.zeros(S.shape,bool)
    for z in range(Z):
        v=frangi2d(S[z]); cv=cellv[z]; nz=v[cv&(v>0)]
        if nz.size>=20:
            med=np.median(nz);mad=np.median(np.abs(nz-med)); s9ridge[z]=(v>med+2.5*1.4826*mad)&cv
    # FIX1: nucleus body = smooth bright SEPT9, EXCLUDING filaments (SEPT9 fibers run UNDER nucleus -> keep them)
    sm=ndi.gaussian_filter(S,(1,4,4)); nucblob=remove_small_objects((sm>np.percentile(sm[cellv],90))&cellv,4000)
    nuc=nucblob&~ndi.binary_dilation(s9ridge,iterations=1)
    # SEPT9 filaments+puncta: box-K25 top-hat; exclude smooth nucleus but KEEP fibers under it
    S9=np.zeros(S.shape,bool); se=square(BOXK)
    for z in range(Z):
        th=white_tophat(S[z],se); cv=cellv[z]; vals=th[cv]
        if vals.size<20: continue
        med=np.median(vals);mad=np.median(np.abs(vals-med)); cand=(th>med+3*1.4826*mad)&cv
        S9[z]=remove_small_objects((cand&~nuc[z])|(cand&s9ridge[z]),4)   # sub-nuclear fibers survive
    MT=ridge3d(M,cellv,2.0,15)   # MT network
    # FIX2 + prior EXCLUSIVE: faint-straight-capable SF = low-threshold actin ridge, NOT near MT, straight
    aR=ridge3d(A,cellv,1.2,20); SF=np.zeros(A.shape,bool)
    for z in range(Z):
        excl=aR[z]&~ndi.binary_dilation(MT[z],iterations=1)      # actin-only (exclusive of MT), prior method
        lbl=label(excl)
        for rp in regionprops(lbl):
            mn=rp.minor_axis_length
            if rp.major_axis_length>=20 and (rp.eccentricity>=0.95 or (mn>0 and rp.major_axis_length/mn>=2.5)):
                SF[z][lbl==rp.label]=True                        # faint-but-straight qualifies
    return dict(A=A,M=M,S=S,cellv=cellv,nuc=nuc,S9=S9,SF=SF,MT=MT)

def manders(S9,target,S9img):
    d=ndi.binary_dilation(target,iterations=1)   # 1-voxel tolerance
    ov=S9&d
    m_vox=ov.sum()/(S9.sum()+1e-9)               # frac of SEPT9 voxels on target
    m_int=S9img[ov].sum()/(S9img[S9].sum()+1e-9) # intensity-weighted Manders M1
    cov=ov.sum()/(target.sum()+1e-9)             # frac of target covered by SEPT9
    return m_vox,m_int,cov

R={name:analyze(p) for name,p in STACKS.items()}
lines=[f"XY_4 cell — frame {FRAME}  (DAY2 DISH1 20min, CHIR)",METRIC,""]
tot_onSF=tot_onMT=tot_S9=0
for name,r in R.items():
    mSFv,mSFi,covSF=manders(r["S9"],r["SF"],r["S"]); mMTv,mMTi,covMT=manders(r["S9"],r["MT"],r["S"])
    both=(r["S9"]&ndi.binary_dilation(r["SF"],iterations=1)&ndi.binary_dilation(r["MT"],iterations=1)).sum()
    part=mMTv/(mSFv+1e-9); r["md"]={"MT":mMTv,"SF":mSFv,"covMT":covMT,"covSF":covSF,"part":part}   # store for bar panel
    tot_onSF+=(r["S9"]&ndi.binary_dilation(r["SF"],iterations=1)).sum(); tot_onMT+=(r["S9"]&ndi.binary_dilation(r["MT"],iterations=1)).sum(); tot_S9+=r["S9"].sum()
    lines+=[f"== {name} ==",
      f"  SEPT9 voxels={int(r['S9'].sum())}  SF voxels={int(r['SF'].sum())}  MT voxels={int(r['MT'].sum())}",
      f"  frac SEPT9 on SF = {mSFv:.3f} (M1 intensity {mSFi:.3f})   SF covered by SEPT9 = {covSF:.3f}",
      f"  frac SEPT9 on MT = {mMTv:.3f} (M1 intensity {mMTi:.3f})   MT covered by SEPT9 = {covMT:.3f}",
      f"  SEPT9 on BOTH = {both/(r['S9'].sum()+1e-9):.3f}",""]
lines+=["== WHOLE CELL (both stacks) ==",
  f"  frac SEPT9 on SF = {tot_onSF/tot_S9:.3f}   frac SEPT9 on MT = {tot_onMT/tot_S9:.3f}"]
txt="\n".join(lines); print(txt); open(C.out(f"xy4_{FRAME}_coloc_metrics.txt"),"w").write(txt)

# QC render: per stack, raw + masks + coincidence
def n01(x,cv): lo,hi=np.percentile(x[cv],(2,99.3)); return np.clip((x-lo)/(hi-lo+1e-9),0,1)
fig,ax=plt.subplots(2,6,figsize=(23.5,8))
COLS=["SEPT9 raw + transfected-cell mask (green)","SEPT9 mask filament+puncta\n(nucleus excl, cyan)","stress fibers (red)","MT (yellow)","coincidence: SEPT9∩SF(orange) / ∩MT(green)"]
for i,(name,r) in enumerate(R.items()):
    cv=r["cellv"].max(0); s9mp=n01(r["S"].max(0),cv)
    ax[i,0].imshow(s9mp,cmap="gray"); ax[i,0].contour(cv,levels=[0.5],colors="#00ff88",linewidths=1.2)
    o=np.dstack([s9mp*.5]*3); o[r["S9"].max(0)]=[0,1,1]; o[r["nuc"].max(0)]=[.5,.2,.2]; ax[i,1].imshow(o)
    o2=np.dstack([n01(r["A"].max(0),cv)*.6]*3); o2[r["SF"].max(0)]=[1,.15,.15]; ax[i,2].imshow(o2)
    o3=np.dstack([n01(r["M"].max(0),cv)*.6]*3); o3[r["MT"].max(0)]=[1,1,0]; ax[i,3].imshow(o3)
    d=lambda m:ndi.binary_dilation(m,iterations=1)
    base=np.dstack([s9mp*.4]*3); s9m=r["S9"].max(0)
    base[s9m&d(r["SF"]).max(0)]=[1,.6,.05]; base[s9m&d(r["MT"]).max(0)]=[.1,1,.2]; ax[i,4].imshow(base)
    axb=ax[i,5]; xb=np.arange(2); wb=0.36; md=r["md"]
    axb.bar(xb-wb/2,[md["MT"],md["SF"]],wb,color=["#2ca02c","#d94801"])           # M1 (dark): frac SEPT9 on MT / on SF
    axb.bar(xb+wb/2,[md["covMT"],md["covSF"]],wb,color=["#a6d96a","#fdae61"])      # M2 (light): network covered by SEPT9
    for kk,(m1,m2) in enumerate([(md["MT"],md["covMT"]),(md["SF"],md["covSF"])]):
        axb.text(kk-wb/2,m1+.01,f"{m1:.2f}",ha="center",fontsize=7); axb.text(kk+wb/2,m2+.01,f"{m2:.2f}",ha="center",fontsize=7)
    axb.set_xticks(xb); axb.set_xticklabels(["MT","SF"],fontsize=9); axb.set_ylim(0,0.85); axb.set_title("Manders M1(dark)/M2(light)",fontsize=9)
    ax[i,0].set_ylabel(name,fontsize=10)
    for j in range(5): ax[i,j].set_xticks([]);ax[i,j].set_yticks([])
    if i==0:
        for j,c in enumerate(COLS): ax[i,j].set_title(c,fontsize=9)
fig.suptitle(f"XY_4 frame {FRAME} — one-cell 3D coloc QC (SEPT9 filament+puncta incl sub-nuclear fibers / SF / MT)",fontsize=12)
fig.text(0.5,0.005,MTAG,ha="center",fontsize=9,style="italic",color="#555")
plt.tight_layout(rect=[0,0.02,1,1]); plt.savefig(C.out(f"xy4_{FRAME}_coloc_QC.png"),dpi=140,bbox_inches="tight"); plt.close(); print(f"\nsaved xy4_{FRAME}_coloc_QC.png")
