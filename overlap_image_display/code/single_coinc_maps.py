"""Single-panel coincidence MAPS from the new basal z-stack (XY-4-003-17-27, 3ch): one for SEPT9∩actin,
one for SEPT9∩MT. Clean publication render: network=magenta, SEPT9(filament+punctate)=green,
coincidence=bright white/yellow. Smooth upscaled (UP=2, bilinear glow). ch0=actin,ch1=MT,ch2=SEPT9.
Output -> ./coincidence_tiffs/xy4_003_BASAL_{ACTIN,MT}_coincmap.png"""
import os, numpy as np, tifffile
import _config as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.morphology import white_tophat, square, remove_small_objects
from skimage.filters import threshold_otsu
F=os.path.join(C.require_images(),"XY-4-003-17-27-ACTIN-MT-SEPTIN9-BASAL.tif")  # set OVERLAP_IMAGES
OUT=C.out("coincidence_tiffs"); os.makedirs(OUT,exist_ok=True)
CH_A,CH_M,CH_S=0,1,2; BOXK=25; SIGMAS=[0.8,1.2,1.8]; UP=2
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
def n01(x,m): x=x.astype(np.float32); lo,hi=np.percentile(x[m] if m.any() else x,(2,99.5)); return np.clip((x-lo)/(hi-lo+1e-9),0,1)
def up(x): return ndi.zoom(x.astype(np.float32),UP,order=1)
a=tifffile.imread(F).astype(np.float32); A=a[:,CH_A]; M=a[:,CH_M]; S=a[:,CH_S]
sm=ndi.gaussian_filter(S,(1,3,3)); cellv=ndi.binary_dilation(sm>threshold_otsu(sm),structure=np.ones((1,3,3)),iterations=1)
S9=np.zeros(S.shape,bool); se=square(BOXK)
for z in range(S.shape[0]):
    th=white_tophat(S[z],se); cv=cellv[z]; vals=th[cv]
    if vals.size<20: continue
    med=np.median(vals);mad=np.median(np.abs(vals-med)); S9[z]=remove_small_objects((th>med+3*1.4826*mad)&cv,4)
cvm=cellv.max(0); Sp=up(n01((S*S9).max(0),cvm)); s9glow=np.clip(ndi.gaussian_filter(up(S9.max(0).astype(float)),1.0),0,1)
for net,vol,netnm in [("ACTIN",A,"actin"),("MT",M,"MT")]:
    NET=ridge3d(vol,cellv,2.8,10); coinc=S9&ndi.binary_dilation(NET,iterations=1)
    m1=coinc.sum()/(S9.sum()+1e-9); m2=coinc.sum()/(NET.sum()+1e-9)
    Np=up(n01(vol.max(0),cvm)); cg=np.clip(ndi.gaussian_filter(up(coinc.max(0).astype(float)),1.0),0,1)
    # network magenta + SEPT9 green; coincidence forced bright yellow/white on top
    img=np.zeros((*Np.shape,3),np.float32)
    img[...,0]=Np; img[...,2]=Np                 # magenta network
    img[...,1]=np.maximum(img[...,1],Sp)         # green SEPT9
    img=img*(1-cg[...,None])+np.array([1,1,0.2])*cg[...,None]   # yellow coincidence
    fig,ax=plt.subplots(figsize=(11,7)); ax.imshow(np.clip(img,0,1),interpolation="bilinear"); ax.axis("off")
    ax.set_title(f"XY_4 003 basal — SEPT9 ∩ {netnm} coincidence map\n{netnm}=magenta  SEPT9=green  coincidence=yellow   |   M1={m1:.2f} (SEPT9 on {netnm}), M2={m2:.2f}",fontsize=12)
    plt.tight_layout(); plt.savefig(os.path.join(OUT,f"xy4_003_BASAL_{net}_coincmap.png"),dpi=150,bbox_inches="tight"); plt.close()
    print(f"saved xy4_003_BASAL_{net}_coincmap.png  M1={m1:.2f} M2={m2:.2f}",flush=True)
