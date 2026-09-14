"""
073126 ALL-comet (not outward-restricted) per-aster septin_along + lifetime, in USER-MARKED loops.
Matches the deck slide (n=26/28 asters). Per loop: track EB1; keep MOVING comets (net>=MOVE_PX);
septin_along = pct-rank of tophat SEPT9 in loop along track. Dump per-comet for violins.
Out -> eb1_analysis\\aster_allcomet_persistence_comets.csv
"""
import os,glob,warnings,numpy as np,tifffile,pandas as pd,trackpy as tp,roifile
from skimage.filters import gaussian
from matplotlib.path import Path as MPath
import _config as CFG
warnings.filterwarnings("ignore"); tp.quiet()
SRC=CFG.require_images(); AST=CFG.require_rois(); OUT=CFG.OUTDIR
PX=0.1076; DT=1.09; DIAM=7; SEP=5; SEARCH=6; MEMORY=1; MINLEN=4; MASS_PCTL=93; MOVE_PX=3.0; OUT_PX=2.0
def geno(b): return "WT" if "WT" in b else "AA"
def base_of(z):
    b=os.path.basename(z)
    for s in ["_denoisedt.zip","_denoised.zip",".zip"]:
        if b.endswith(s): return b[:-len(s)]
    return b
def polys(z):
    rr=roifile.roiread(z); rr=rr if isinstance(rr,list) else [rr]; P=[];Q=[]
    for r in rr:
        c=np.asarray(r.coordinates(),float)
        if (int(r.roitype)==10 or len(c)<=2): Q.append(c.mean(0) if len(c)>1 else c[0])
        else: P.append(c)
    return P,Q
COMETS=[]
for z in sorted(glob.glob(os.path.join(AST,"*.zip"))):
    base=base_of(z); g=geno(base); src=os.path.join(SRC,base+"_denoised.tif")
    if not os.path.exists(src): continue
    tf=tifffile.TiffFile(src); s=tf.series[0]; C=s.shape[s.axes.index('C')]; T=s.shape[s.axes.index('T')]; H,W=s.shape[-2:]
    if (H,W)!=(2304,2304): tf.close(); continue
    _polys,_pts=polys(z)
    for ai,poly in enumerate(_polys):
        pth=MPath(poly); x0=max(0,int(poly[:,0].min())-8);x1=min(W,int(poly[:,0].max())+8);y0=max(0,int(poly[:,1].min())-8);y1=min(H,int(poly[:,1].max())+8)
        Hc,Wc=y1-y0,x1-x0; yy,xx=np.mgrid[y0:y1,x0:x1]; loop=pth.contains_points(np.c_[xx.ravel(),yy.ravel()]).reshape(Hc,Wc)
        if loop.sum()<80: continue
        pin=[p for p in _pts if pth.contains_point((p[0],p[1]))]; hx,hy=(pin[0]-[x0,y0] if pin else poly.mean(0)-[x0,y0])
        s9=np.stack([tf.pages[t*C+CFG.CH_SEPT9].asarray()[y0:y1,x0:x1] for t in range(T)]).astype(np.float32)
        eb1=np.stack([tf.pages[t*C+CFG.CH_EB1].asarray()[y0:y1,x0:x1] for t in range(T)]).astype(np.float32)
        eb1=np.clip(eb1-np.median(eb1,axis=0)[None],0,None)   # SOLUTION (Mac deck S8): per-pixel temporal-median subtraction removes stationary aster-arm features so the tracker can't follow them all movie -> physiological comet lifetimes
        septm=gaussian(s9.mean(0),1.0); pct=np.zeros_like(septm); vals=septm[loop]; pct[loop]=vals.argsort().argsort()/(max(len(vals)-1,1))
        mm=float(np.percentile(eb1[:,loop][eb1[:,loop]>0],MASS_PCTL)); fl=[]
        for t in range(T):
            f=tp.locate(eb1[t],DIAM,minmass=mm,separation=SEP,engine="python")
            if len(f):
                ins=loop[np.clip(f.y.round().astype(int),0,Hc-1),np.clip(f.x.round().astype(int),0,Wc-1)]; f=f[ins]
                if len(f): fl.append(f.assign(frame=t))
        if len(fl)<3: continue
        tr=tp.filter_stubs(tp.link(pd.concat(fl,ignore_index=True),search_range=SEARCH,memory=MEMORY,adaptive_stop=2,adaptive_step=0.9),MINLEN).reset_index(drop=True)
        for pid,gg in tr.groupby("particle"):
            gg=gg.sort_values("frame"); xy=gg[["x","y"]].to_numpy()
            if np.hypot(*(xy[-1]-xy[0]))<MOVE_PX: continue
            yi=np.clip(xy[:,1].round().astype(int),0,Hc-1); xi=np.clip(xy[:,0].round().astype(int),0,Wc-1)
            path=np.hypot(*np.diff(xy,axis=0).T).sum(); spd=path*PX/max((len(xy)-1)*DT,1e-6)*1000
            d0=np.hypot(xy[0,0]-hx,xy[0,1]-hy); dL=np.hypot(xy[-1,0]-hx,xy[-1,1]-hy)   # radial vs marked hub
            COMETS.append(dict(base=base,geno=g,aster=ai,septin_along=round(float(np.mean(pct[yi,xi])),3),lifetime_s=round(len(xy)*DT,2),speed_nm_s=round(spd,1),outward=int((dL-d0)>=OUT_PX)))
    tf.close(); print(f"{g} {base[9:]:22s} done",flush=True)
pd.DataFrame(COMETS).to_csv(os.path.join(OUT,"aster_allcomet_persistence_comets.csv"),index=False)
print("DONE ->",os.path.join(OUT,"aster_allcomet_persistence_comets.csv"),"n_comets",len(COMETS),flush=True)
