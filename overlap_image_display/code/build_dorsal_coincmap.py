"""
Plain single coincidence map (network=magenta / SEPT9=green / coincidence=yellow) from the DORSAL-MOST
N z-slices of an apical stack. N is DIALABLE via env NLAST (default 3). Also configurable: NET (MT/ACTIN),
FRAMES, ZRANGE, PLANE. Sensitive masks (tophat+faint recovery, SEPT9 filament+punctate). 2ch ch0=SEPT9, ch1=network.
  py build_dorsal_coincmap.py           -> dorsal 3 z, apical MT, frames 003 & 028
  NLAST=5 NET=MT py build_dorsal_coincmap.py   -> dorsal 5 z
Output -> ./coincidence_tiffs/xy4_{NET}_{PLANE}_dorsal{N}z_{frame}_coincmap.png
"""
import os, numpy as np, tifffile
import _config as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import white_tophat, square, remove_small_objects
from skimage.filters import threshold_otsu
DD=C.require_images()  # set OVERLAP_IMAGES to the XY_4 crop folder
OUT=C.out("coincidence_tiffs"); os.makedirs(OUT,exist_ok=True)
NLAST=int(os.environ.get("NLAST","3")); NET=os.environ.get("NET","MT"); PLANE=os.environ.get("PLANE","APICAL")
ZRANGE=os.environ.get("ZRANGE","7-20"); FRAMES=os.environ.get("FRAMES","003,028").split(",")
netnm={"MT":"MT","ACTIN":"actin"}[NET]; SIGMAS=[0.8,1.2,1.8]; UP=2
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
def cellfg(S): sm=ndi.gaussian_filter(S,(1,3,3)); return ndi.binary_dilation(sm>threshold_otsu(sm),structure=np.ones((1,3,3)),iterations=1)
def net_mask(vol,cellv):
    out=np.zeros(vol.shape,bool)
    for z in range(vol.shape[0]):
        v=frangi2d(white_tophat(vol[z],square(41))); cv=cellv[z]; nz=v[cv&(v>0)]
        if nz.size<20: continue
        med=np.median(nz);mad=np.median(np.abs(nz-med)); out[z]=remove_small_objects((v>med+1.2*1.4826*mad)&cv,8)
    return out
def sept9_mask(S,cellv):
    out=np.zeros(S.shape,bool)
    for z in range(S.shape[0]):
        th=white_tophat(S[z],square(25)); cv=cellv[z]; vals=th[cv]
        if vals.size<20: continue
        med=np.median(vals);mad=np.median(np.abs(vals-med)); out[z]=remove_small_objects((th>med+2.0*1.4826*mad)&cv,4)
    return out
def n01(x,m): x=x.astype(np.float32); lo,hi=np.percentile(x[m] if m.any() else x,(2,99.5)); return np.clip((x-lo)/(hi-lo+1e-9),0,1)
def up(x): return ndi.zoom(x.astype(np.float32),UP,order=1)
zlo=int(ZRANGE.split("-")[1])-NLAST+1; zhi=int(ZRANGE.split("-")[1])
for fr in FRAMES:
    a=tifffile.imread(os.path.join(DD,f"XY-4-{fr}-{ZRANGE}-{NET}-SEPTIN9-{PLANE}.tif")).astype(np.float32)
    a=a[a.shape[0]-NLAST:]                       # dorsal-most N z-slices
    S=a[:,0]; N=a[:,1]; cellv=cellfg(S); S9=sept9_mask(S,cellv); NETm=net_mask(N,cellv)
    coinc=S9&ndi.binary_dilation(NETm,iterations=1); m1=coinc.sum()/(S9.sum()+1e-9); m2=coinc.sum()/(NETm.sum()+1e-9)
    cvm=cellv.max(0); Np=up(n01(N.max(0),cvm)); Sp=up(n01((S*S9).max(0),cvm)); cg=np.clip(ndi.gaussian_filter(up(coinc.max(0).astype(float)),1.0),0,1)
    img=np.zeros((*Np.shape,3),np.float32); img[...,0]=Np; img[...,2]=Np; img[...,1]=np.maximum(img[...,1],Sp)
    img=img*(1-cg[...,None])+np.array([1,1,0.2])*cg[...,None]
    fig,ax=plt.subplots(figsize=(11,7)); ax.imshow(np.clip(img,0,1),interpolation="bilinear"); ax.axis("off")
    ax.set_title(f"XY_4 {fr} {PLANE.lower()} — SEPT9 ∩ {netnm}, DORSAL {NLAST} z (z{zlo}-{zhi})\n{netnm}=magenta SEPT9=green coincidence=yellow | M1={m1:.2f}, M2={m2:.2f}",fontsize=12)
    plt.tight_layout(); out=os.path.join(OUT,f"xy4_{NET}_{PLANE}_dorsal{NLAST}z_{fr}_coincmap.png")
    plt.savefig(out,dpi=150,bbox_inches="tight"); plt.close(); print(f"{fr}: M1={m1:.3f} M2={m2:.3f} -> {os.path.basename(out)}",flush=True)
