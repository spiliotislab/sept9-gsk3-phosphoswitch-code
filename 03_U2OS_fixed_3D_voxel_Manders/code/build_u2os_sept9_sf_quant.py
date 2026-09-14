"""
Quantify the user's observation: in Control, SEPT9 on stress fibers is BRIGHTER and more CONTINUOUS.
Reuses saved ImageJ TIFFs (raw + masks) — no Frangi recompute.
Metrics per condition:
  BRIGHTNESS: mean SEPT9 filament signal (top-hat) within SF mask, and enrichment vs whole cell.
  CONTINUITY: along the SF skeleton, frac decorated by SEPT9, and mean run-length of continuous
              SEPT9-positive stretches (longer = more continuous), + segments per 100px SF.
Output: u2os_sept9_on_sf_quant.png (+ printed numbers). n=1 cell/condition -> DESCRIPTIVE.
"""
import os, numpy as np, tifffile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import white_tophat, square, skeletonize
from skimage.filters import threshold_otsu
from skimage.measure import label
import _config as C
# imagej TIFFs (raw channels + masks) are produced by build_u2os_coincmap.py in outputs/imagej/
IJ=C.out("imagej"); HERE=os.path.dirname(os.path.abspath(__file__)); SLL="z1-3"  # matches build_u2os_coincmap.py output
CROP=(750,1150,650,1050)   # zoom window (y0,y1,x0,x1)
res={}
for cond in ["Control","CHIR"]:
    raw=tifffile.imread(os.path.join(IJ,f"{cond}_raw_actin-SEPT9-MT_{SLL}.tif")).astype(np.float32)  # (3,Y,X) actin,SEPT9,MT
    mk=tifffile.imread(os.path.join(IJ,f"{cond}_masks_SF-SEPT9-coinc_{SLL}.tif"))                      # (3,Y,X) SF,SEPT9,coinc
    S9=raw[1]; SF=mk[0]>0; S9m=mk[1]>0
    fg=ndi.gaussian_filter(raw[0]+S9,2)>threshold_otsu(ndi.gaussian_filter(raw[0]+S9,2))
    th=white_tophat(S9,square(25))                       # SEPT9 filament/puncta signal (bg removed)
    bright_SF=float(th[SF].mean()); bright_cell=float(th[fg].mean()); enrich=bright_SF/(bright_cell+1e-9)
    skel=skeletonize(SF); dec=skel&ndi.binary_dilation(S9m,iterations=1)
    cov=dec.sum()/(skel.sum()+1e-9)
    lbl=label(dec); segs=ndi.sum(np.ones_like(lbl),lbl,range(1,lbl.max()+1)) if lbl.max()>0 else np.array([0])
    mean_run=float(np.mean(segs)); nseg_per100=100.0*lbl.max()/(skel.sum()+1e-9)
    res[cond]=dict(bright_SF=bright_SF,enrich=enrich,cov=cov,mean_run=mean_run,nseg_per100=nseg_per100,
                   S9=S9,SF=SF,S9m=S9m,fg=fg)
    print(f"{cond}: SEPT9-bright-on-SF={bright_SF:.1f} (enrich vs cell {enrich:.2f}) | SFlen-decorated={cov:.3f} "
          f"mean-run={mean_run:.1f}px segs/100px={nseg_per100:.2f}",flush=True)

C,H=res["Control"],res["CHIR"]
fig=plt.figure(figsize=(15,8));
# bar metrics
ax=fig.add_axes([0.06,0.12,0.40,0.78])
labels=["SEPT9 brightness\non SF","enrichment\n(SF/cell)","SF length\ndecorated","mean run-length\n(px, continuity)"]
cv=[C["bright_SF"],C["enrich"],C["cov"],C["mean_run"]]; hv=[H["bright_SF"],H["enrich"],H["cov"],H["mean_run"]]
# normalize each metric to Control for a fair grouped view
cvn=[1,1,1,1]; hvn=[h/(c+1e-9) for h,c in zip(hv,cv)]
x=np.arange(4); w=0.38
ax.bar(x-w/2,cvn,w,label="Control",color="#2171b5"); ax.bar(x+w/2,hvn,w,label="CHIR",color="#d94801")
for i in range(4):
    ax.text(x[i]-w/2,cvn[i]+0.02,f"{cv[i]:.1f}" if i in(0,3) else f"{cv[i]:.2f}",ha="center",fontsize=8)
    ax.text(x[i]+w/2,hvn[i]+0.02,f"{hv[i]:.1f}" if i in(0,3) else f"{hv[i]:.2f}",ha="center",fontsize=8)
ax.axhline(1,ls=':',c='k'); ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
ax.set_ylabel("relative to Control"); ax.set_title("SEPT9 on stress fibers: brightness & continuity (n=1 each)"); ax.legend()
# zoom overlays
y0,y1,x0,x1=CROP
for i,(cond,r) in enumerate(res.items()):
    axz=fig.add_axes([0.52,0.52-0.44*i,0.44,0.40])
    s9=r["S9"][y0:y1,x0:x1]; lo,hi=np.percentile(s9,(2,99.5)); s9n=np.clip((s9-lo)/(hi-lo+1e-9),0,1)
    ov=np.dstack([np.zeros_like(s9n),s9n,np.zeros_like(s9n)])          # SEPT9 green
    sf=r["SF"][y0:y1,x0:x1]; ov[sf]=np.clip(ov[sf]+[0.6,0,0],0,1)      # SF red tint
    axz.imshow(ov,interpolation="bilinear"); axz.set_title(f"{cond} zoom — SEPT9(green) on SF(red)",fontsize=10)
    axz.set_xticks([]);axz.set_yticks([])
fig.suptitle("U2OS: is SEPT9 brighter & more continuous on stress fibers in Control vs CHIR?",fontsize=13)
plt.savefig(C.out("u2os_sept9_on_sf_quant.png"),dpi=140,bbox_inches="tight"); plt.close(); print("saved u2os_sept9_on_sf_quant.png")
