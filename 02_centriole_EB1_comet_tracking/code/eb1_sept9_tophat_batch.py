"""
Regenerate the SEPT9 channel for all 74 EB1/SEPT9 timelapses using median-3 + white
top-hat (disk~12), reading RAW SEPT9 from the ND2s (N2V2 destroyed the dim diffuse signal).
Output 2ch TCYX tiff: ch0 = EB1 (561) RAW, ch1 = SEPT9 (488) median3+top-hat.
GPU-accelerated (median via unfold, top-hat via box opening). -> EB1_IMAGES (outputs/eb1_s9tophat/)
"""
import os, glob, time, numpy as np, nd2, tifffile, torch
import torch.nn.functional as F
import _config as CFG
RAW=CFG.require_raw()
OUT=CFG.images_out()
DEV="cuda" if torch.cuda.is_available() else "cpu"; K=25   # box ~ disk radius 12
def proc_s9(vol):   # (T,Y,X) raw SEPT9 -> median3 + white top-hat, uint16
    out=np.empty(vol.shape,np.uint16)
    for t in range(vol.shape[0]):
        x=torch.from_numpy(vol[t].astype(np.float32)).to(DEV)[None,None]      # 1,1,Y,X
        p=F.pad(x,(1,1,1,1),mode='reflect'); med=F.unfold(p,3).median(1).values.view(x.shape)  # 3x3 median
        ero=-F.max_pool2d(-med,K,1,K//2); opn=F.max_pool2d(ero,K,1,K//2)      # opening
        out[t]=torch.clamp(med-opn,0,65535).squeeze().cpu().numpy().astype(np.uint16)
    return out
files=sorted(glob.glob(os.path.join(RAW,"*.nd2")))
done=0
for f in files:
    base=os.path.splitext(os.path.basename(f))[0]; outp=os.path.join(OUT,base+"_denoised.tif")
    if os.path.exists(outp): print("skip(done)",base,flush=True); continue
    with nd2.ND2File(f) as n:
        sizes=dict(n.sizes)
        if 'T' not in sizes: print("skip(not-timelapse)",base,flush=True); continue
        a=n.asarray()                          # T,C,Y,X  C=[561,488]
    if a.ndim==3: a=a[None]
    t0=time.time()
    out=np.empty((a.shape[0],2,a.shape[2],a.shape[3]),np.uint16)
    out[:,0]=np.clip(a[:,CFG.CH_EB1],0,65535).astype(np.uint16)     # EB1 raw
    out[:,1]=proc_s9(a[:,CFG.CH_SEPT9])                             # SEPT9 median3+tophat
    tifffile.imwrite(outp,out,imagej=True,metadata={"axes":"TCYX"})
    torch.cuda.empty_cache(); done+=1
    print(f"  {base}: {a.shape} ({time.time()-t0:.1f}s)",flush=True)
print(f"EB1 SEPT9-TOPHAT BATCH DONE ({done} processed) -> {OUT}",flush=True)
