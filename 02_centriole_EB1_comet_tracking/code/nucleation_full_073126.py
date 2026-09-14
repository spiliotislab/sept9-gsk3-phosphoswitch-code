"""
073126 nucleation at the USER-MARKED CENTRIOLE (hub point in each rois_aster loop), 3 complementary readouts:
 (A) SPATIAL enrichment: are comet BIRTHS clustered near the centriole vs uniform over the aster region?
     Poisson z = (obs births within R_NUC - exp under uniform)/sqrt(exp). z>0=center-concentrated, ~0=uniform.
 (B) RATE: comet births within R_NUC of centriole PER MINUTE (raw per-aster; + density-matched for comet count).
 (C) TIME-RESOLVED: births-near-centriole per minute across 10 movie-fraction windows -> nucleation-over-time.
Temporal-median EB1 detection. dt=1.09s, 0.1076um/px, R_NUC=10px(~1um). Out -> eb1_analysis\\nucleation_full_073126.{csv,npz}
"""
import os,glob,warnings,numpy as np,tifffile,pandas as pd,trackpy as tp,roifile
from matplotlib.path import Path as MPath
import _config as CFG
warnings.filterwarnings("ignore"); tp.quiet()
SRC=CFG.require_images(); AST=CFG.require_rois(); OUT=CFG.OUTDIR
PX=CFG.PX; DT=CFG.DT; DIAM=7; SEP=5; SEARCH=6; MEMORY=1; MINLEN=3; MASS_PCTL=93; R_NUC=10.0; NW=10
def geno(b): return "WT" if "WT" in b else "AA"
def base_of(z):
    b=os.path.basename(z)
    for s in ["_denoisedt.zip","_denoised.zip",".zip"]:
        if b.endswith(s): return b[:-len(s)]
    return b
def load(z):
    rr=roifile.roiread(z); rr=rr if isinstance(rr,list) else [rr]; P=[];Q=[]
    for r in rr:
        c=np.asarray(r.coordinates(),float)
        if int(r.roitype)==10 or len(c)<=2: Q.append(c.mean(0) if len(c)>1 else c[0])
        else: P.append(c)
    return P,Q
rows=[]; prof={}   # per-aster time-resolved rate profiles
for z in sorted(glob.glob(os.path.join(AST,"*.zip"))):
    base=base_of(z); g=geno(base); src=os.path.join(SRC,base+"_denoised.tif")
    if not os.path.exists(src): continue
    polys,pts=load(z); tf=tifffile.TiffFile(src); s=tf.series[0]; C=s.shape[s.axes.index('C')]; T=s.shape[s.axes.index('T')]; H,W=s.shape[-2:]
    if (H,W)!=(2304,2304): tf.close(); continue
    minutes=T*DT/60.0
    for ai,poly in enumerate(polys):
        pth=MPath(poly); x0=max(0,int(poly[:,0].min())-8);x1=min(W,int(poly[:,0].max())+8);y0=max(0,int(poly[:,1].min())-8);y1=min(H,int(poly[:,1].max())+8)
        Hc,Wc=y1-y0,x1-x0; yy,xx=np.mgrid[y0:y1,x0:x1]; loop=pth.contains_points(np.c_[xx.ravel(),yy.ravel()]).reshape(Hc,Wc)
        if loop.sum()<80: continue
        pin=[p for p in pts if pth.contains_point((p[0],p[1]))]; hx,hy=(pin[0]-[x0,y0] if pin else poly.mean(0)-[x0,y0])
        eb1=np.stack([tf.pages[t*C+CFG.CH_EB1].asarray()[y0:y1,x0:x1] for t in range(T)]).astype(np.float32)
        eb1=np.clip(eb1-np.median(eb1,axis=0)[None],0,None)   # temporal-median detection
        mm=float(np.percentile(eb1[:,loop][eb1[:,loop]>0],MASS_PCTL)); fl=[]
        for t in range(T):
            f=tp.locate(eb1[t],DIAM,minmass=mm,separation=SEP,engine="python")
            if len(f):
                ins=loop[np.clip(f.y.round().astype(int),0,Hc-1),np.clip(f.x.round().astype(int),0,Wc-1)]; f=f[ins]
                if len(f): fl.append(f.assign(frame=t))
        if len(fl)<3: continue
        tr=tp.filter_stubs(tp.link(pd.concat(fl,ignore_index=True),search_range=SEARCH,memory=MEMORY,adaptive_stop=2,adaptive_step=0.9),MINLEN).reset_index(drop=True)
        bf=[]; bx=[]; by=[]
        for pid,gg in tr.groupby("particle"):
            gg=gg.sort_values("frame"); bf.append(int(gg.frame.values[0])); bx.append(gg.x.values[0]); by.append(gg.y.values[0])
        bf=np.array(bf); bx=np.array(bx); by=np.array(by); nb=len(bf)
        if nb<8: continue
        d=np.hypot(bx-hx,by-hy); near=d<=R_NUC; nnear=int(near.sum())
        # spatial enrichment z
        ly,lx=np.where(loop); frac=float((np.hypot(lx-hx,ly-hy)<=R_NUC).mean()); exp=nb*frac
        zval=(nnear-exp)/np.sqrt(exp) if exp>0.5 else np.nan
        # time-resolved rate of near-births (fraction-of-movie windows)
        win_min=(T/NW)*DT/60.0; wr=np.zeros(NW)
        for k in range(NW):
            lo=k*T/NW; hi=(k+1)*T/NW; wr[k]=np.sum(near&(bf>=lo)&(bf<hi))/win_min
        prof[f"{base}|{ai}|{g}"]=wr
        rows.append(dict(base=base,geno=g,aster=ai,n_births=nb,n_near=nnear,rate_near_min=round(nnear/minutes,2),
            birth_enrich_z=round(zval,2) if zval==zval else np.nan))
    tf.close(); print(f"{g} {base[9:]:22s} done",flush=True)
D=pd.DataFrame(rows); D.to_csv(os.path.join(OUT,"nucleation_full_073126.csv"),index=False)
np.savez(os.path.join(OUT,"nucleation_timeresolved_073126.npz"),**prof)
from scipy.stats import mannwhitneyu
print("\n(A) SPATIAL enrichment z:  WT %.2f  AA %.2f  p=%.3f"%(D[D.geno=='WT'].birth_enrich_z.median(),D[D.geno=='AA'].birth_enrich_z.median(),mannwhitneyu(D[D.geno=='WT'].birth_enrich_z.dropna(),D[D.geno=='AA'].birth_enrich_z.dropna()).pvalue))
print("(B) RATE near-centriole /min (raw):  WT %.2f  AA %.2f  p=%.3f"%(D[D.geno=='WT'].rate_near_min.median(),D[D.geno=='AA'].rate_near_min.median(),mannwhitneyu(D[D.geno=='WT'].rate_near_min,D[D.geno=='AA'].rate_near_min).pvalue))
lo=max(D[D.geno=='WT'].n_births.min(),D[D.geno=='AA'].n_births.min()); hi=min(D[D.geno=='WT'].n_births.max(),D[D.geno=='AA'].n_births.max()); mm2=D[(D.n_births>=lo)&(D.n_births<=hi)]
print("    density-matched (n_births %d-%d): WT %.2f AA %.2f p=%.3f"%(lo,hi,mm2[mm2.geno=='WT'].rate_near_min.median(),mm2[mm2.geno=='AA'].rate_near_min.median(),mannwhitneyu(mm2[mm2.geno=='WT'].rate_near_min,mm2[mm2.geno=='AA'].rate_near_min).pvalue))
print("DONE",flush=True)
