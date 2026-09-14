"""
Publication-quality, ImageJ-ADJUSTABLE coincidence TIFFs for the XY_4 2-ch split crops.
VERIFIED channels: ch0=SEPT9, ch1=network (filename's 1st token = ACTIN or MT). Per crop we build a
3-channel ImageJ COMPOSITE (max-proj + full z-stack) with baked LUTs:
  C1 network (gray)   C2 SEPT9 (green)   C3 coincidence SEPT9∩network (yellow, = SEPT9 intensity where coincident)
so you adjust B&C / LUT per layer in Fiji. Masks match our metric family: SEPT9 = box-K25 white-tophat + ridge;
network = per-z Frangi ridge; coincidence = SEPT9 ∩ dilate(network,1vox). M1/M2 stamped in coinc_metrics.txt.
Slices are 1-based in the source filenames (z-range in name). Output -> ./coincidence_tiffs/
"""
import os, glob, numpy as np, tifffile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import white_tophat, square, remove_small_objects
from skimage.filters import threshold_otsu
import _config as C
DD=os.environ.get("MCF_XY4_DIR", C.IMAGES or "")   # set MCF_XY4_DIR to the XY_4 crop folder
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=C.out("coincidence_tiffs"); os.makedirs(OUT,exist_ok=True)
CH_S,CH_N=0,1; BOXK=25; SIGMAS=[0.8,1.2,1.8]   # thin-filament scales; network ridge k2.8/minpx10

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
def ridge3d(vol,cellv,k,minpx):
    out=np.zeros(vol.shape,bool)
    for z in range(vol.shape[0]):
        v=frangi2d(vol[z]); cv=cellv[z]; nz=v[cv&(v>0)]
        if nz.size<20: continue
        med=np.median(nz);mad=np.median(np.abs(nz-med)); out[z]=remove_small_objects((v>med+k*1.4826*mad)&cv,minpx)
    return out
def masks(S,N):
    Z=S.shape[0]
    # foreground from SEPT9 ALONE so the SEPT9 mask is identical across a cell's ACTIN & MT crops
    sm=ndi.gaussian_filter(S,(1,3,3)); cellv=sm>threshold_otsu(sm)
    cellv=ndi.binary_dilation(cellv,structure=np.ones((1,3,3)),iterations=1)
    # SEPT9 filament+puncta (box-K25 tophat)
    S9=np.zeros(S.shape,bool); se=square(BOXK)
    for z in range(Z):
        th=white_tophat(S[z],se); cv=cellv[z]; vals=th[cv]
        if vals.size<20: continue
        med=np.median(vals);mad=np.median(np.abs(vals-med)); S9[z]=remove_small_objects((th>med+3*1.4826*mad)&cv,4)
    NET=ridge3d(N,cellv,2.8,10)                       # network filaments (Frangi, thin)
    coinc=S9&ndi.binary_dilation(NET,iterations=1)    # 1-voxel tolerance
    return S9,NET,coinc

def lut(r,g,b):  # (3,256) uint8 ramp scaled by rgb weights
    ramp=np.arange(256,dtype=np.uint8)
    return np.stack([(ramp*r).astype(np.uint8),(ramp*g).astype(np.uint8),(ramp*b).astype(np.uint8)])
LUTS=[lut(1,1,1),lut(0,1,0),lut(1,1,0)]   # gray network, green SEPT9, yellow coincidence

lines=["XY_4 coincidence TIFFs — M1=frac SEPT9 on network, M2=frac network covered by SEPT9 (1vox tol)",
       "channels verified ch0=SEPT9 ch1=network; masks: SEPT9 box-K25 tophat, network Frangi ridge",""]
for f in sorted(glob.glob(os.path.join(DD,"XY-4-*SEPTIN9-*.tif"))):
    a=tifffile.imread(f).astype(np.float32); S=a[:,CH_S]; N=a[:,CH_N]
    S9,NET,coinc=masks(S,N)
    m1=coinc.sum()/(S9.sum()+1e-9); m2=coinc.sum()/(NET.sum()+1e-9)
    tag=os.path.splitext(os.path.basename(f))[0]
    net_name="ACTIN" if "ACTIN" in tag else "MT"
    # SEPT9 shown = FILAMENTOUS + PUNCTATE only (masked intensity; diffuse removed), NOT raw
    Smask=(S*S9).astype(np.uint16)
    coincI=(S*coinc).astype(np.uint16)                 # coincidence keeps SEPT9 intensity (adjustable)
    # --- max-projection composite (the publication panel) ---
    mp=np.stack([N.max(0).astype(np.uint16), Smask.max(0), coincI.max(0)])   # (C,Y,X)
    tifffile.imwrite(os.path.join(OUT,f"{tag}_coinc_MAXPROJ.tif"), mp, imagej=True,
                     metadata={"mode":"composite","axes":"CYX","LUTs":LUTS,
                               "Labels":[f"network({net_name})","SEPT9 filament+punctate","coincidence"]})
    # --- full z-stack composite (adjust z / re-project) ---
    zs=np.stack([N.astype(np.uint16), Smask, coincI],axis=1)                 # (Z,C,Y,X)
    tifffile.imwrite(os.path.join(OUT,f"{tag}_coinc_ZSTACK.tif"), zs, imagej=True,
                     metadata={"mode":"composite","axes":"ZCYX","LUTs":LUTS,
                               "Labels":[f"network({net_name})","SEPT9 filament+punctate","coincidence"]})
    # --- flattened RGB preview (QC / drop-in figure) ---
    def n01(x): x=x.astype(np.float32); lo,hi=np.percentile(x,(2,99.5)); return np.clip((x-lo)/(hi-lo+1e-9),0,1)
    Nn=n01(N.max(0)); Sn=n01(Smask.max(0)); cm=coinc.max(0)
    rgb=np.dstack([Nn*0.6, Nn*0.6+Sn*0.9, Nn*0.6]); rgb[cm]=[1,1,0]     # gray network + green SEPT9 + yellow coinc
    plt.figure(figsize=(7,4.5)); plt.imshow(np.clip(rgb,0,1)); plt.axis("off")
    plt.title(f"{net_name} {'BASAL' if 'BASAL' in tag else 'APICAL'} f{tag.split('-')[2]}  M1={m1:.2f} M2={m2:.2f}",fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,f"{tag}_preview.png"),dpi=130,bbox_inches="tight"); plt.close()
    lines.append(f"{tag:42s}  M1(SEPT9->net)={m1:.3f}  M2(net covered)={m2:.3f}  [SEPT9 {int(S9.sum())} / net {int(NET.sum())} / coinc {int(coinc.sum())} vox]")
    print(lines[-1],flush=True)
open(os.path.join(OUT,"coinc_metrics.txt"),"w").write("\n".join(lines))
print("\nsaved ->",OUT)
