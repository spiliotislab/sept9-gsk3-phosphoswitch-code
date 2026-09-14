"""
FAITHFUL coincidence across ALL Day-2 cells (both dishes), rim-excluded SF mask.
Per cell-timepoint (2D max-proj; flat cells):
  - SEPT9 filament skeleton (top-hat->Frangi->skeletonize, nucleus excluded)
  - MT skeleton (Frangi->skeletonize)
  - stress fibers = bright elongated actin (maj/min>=2.5) IN THE CELL INTERIOR
    (cell footprint eroded by RIM_BAND px -> cortical rim removed)
Metrics: frac of SEPT9 filament on MT, frac on SF, frac of MT decorated by SEPT9 (per length).
Per-cell early->late + paired bootstrap CI (pooled + by dish).  Cells: >=3 non-border tp.
Outputs: faithful_percell.csv, faithful_estimation.png, cell5_SF_rimfix_QC.png
"""
import os, glob, csv, numpy as np, tifffile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.morphology import white_tophat, disk, remove_small_objects, skeletonize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import _config as C

HERE=os.path.dirname(os.path.abspath(__file__))
DENROOT=C.require_images()                                  # set MCF_IMAGES to your stack folder
SIGMAS=[1.0,1.5,2.5,4.0,6.0]; CH_ACTIN,CH_MT,CH_S9=C.CH_ACTIN,C.CH_MT,C.CH_S9
MIN_CELL_AREA=8000; TOPHAT_R=12; IOU_GATE=0.15; PAD=25; RIM_BAND=12
CELL5_AREA=np.array([43252,95598,103243,113419,112403,113565,113036,56476,55879.],float)

def frangi2d(img):
    best=np.zeros_like(img,np.float32)
    for s in SIGMAS:
        g=ndi.gaussian_filter(img.astype(np.float32),s)
        Hxx=np.pad(g[:,2:]-2*g[:,1:-1]+g[:,:-2],((0,0),(1,1)))
        Hyy=np.pad(g[2:,:]-2*g[1:-1,:]+g[:-2,:],((1,1),(0,0)))
        gx=np.pad((g[:,2:]-g[:,:-2])/2,((0,0),(1,1))); Hxy=np.pad((gx[2:,:]-gx[:-2,:])/2,((1,1),(0,0)))
        Hxx*=s*s;Hyy*=s*s;Hxy*=s*s;tmp=np.sqrt((Hxx-Hyy)**2+4*Hxy**2+1e-12)
        l1=(Hxx+Hyy+tmp)/2;l2=(Hxx+Hyy-tmp)/2;sw=np.abs(l1)>np.abs(l2)
        a1=np.where(sw,l2,l1);a2=np.where(sw,l1,l2)
        Rb=np.abs(a1)/(np.abs(a2)+1e-9);S=np.sqrt(a1**2+a2**2+1e-12);c=0.5*float(S.max())+1e-9
        V=np.exp(-Rb**2/0.5)*(1-np.exp(-S**2/(2*c*c)));best=np.maximum(best,np.where(a2<0,V,0))
    return best
def ridge(img,fp,k,minpx):
    v=frangi2d(img); nz=v[fp&(v>0)]
    if nz.size==0: return np.zeros_like(fp)
    med=np.median(nz);mad=np.median(np.abs(nz-med)); return remove_small_objects((v>med+k*1.4826*mad)&fp,minpx)

def metrics(mpA,mpM,mpS,fp):
    sm=ndi.gaussian_filter(mpS,4); nuc=remove_small_objects(ndi.binary_fill_holes((sm>np.percentile(sm[fp],92))&fp),3000)
    nuc=ndi.binary_dilation(nuc,iterations=4)
    s9skel=skeletonize(ridge(white_tophat(mpS,disk(TOPHAT_R)),fp,3.0,12)&~nuc)
    mtskel=skeletonize(ridge(mpM,fp,2.5,15))
    interior=ndi.binary_erosion(fp,iterations=RIM_BAND)          # remove cortical rim band
    sfb=remove_small_objects((mpA>np.percentile(mpA[fp],88))&interior,40); lbl=label(sfb); sf=np.zeros_like(fp)
    for rp in regionprops(lbl):
        mn=rp.minor_axis_length
        if mn>0 and rp.major_axis_length/mn>=2.5 and rp.area>=40: sf[lbl==rp.label]=True
    d=lambda m:ndi.binary_dilation(m,iterations=2)
    mt_len=int(mtskel.sum()); s9_len=int(s9skel.sum()); sf_amt=int(sf.sum())
    if mt_len<40 or s9_len<40: return None
    return dict(mt_len=mt_len,s9_len=s9_len,sf_amt=sf_amt,
        fr_s9_mt=(s9skel&d(mtskel)).sum()/s9_len, fr_s9_sf=(s9skel&d(sf)).sum()/s9_len if sf_amt>0 else 0.0,
        mt_dec=(mtskel&d(s9skel)).sum()/mt_len,
        _s9=s9skel,_mt=mtskel,_sf=sf,_int=interior,_A=mpA,_fp=fp)

REGIONS=sorted(set(os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(DENROOT,"*_denoised.tif"))))
percell={}; percell_tp=[]; cell5_qc={}
for region in REGIONS:
    files=sorted(glob.glob(os.path.join(DENROOT,f"{region}__*_denoised.tif")))
    dish="DISH1(20m)" if region.endswith("_20min") else "DISH2(30m)"; T=[i*(20 if dish[4]=="1" else 30) for i in range(len(files))]
    foots=[]; MP=[]
    for f in files:
        a=tifffile.imread(f); tot=ndi.gaussian_filter(a.sum(1).astype(np.float32),2)
        foots.append(label(remove_small_objects(ndi.binary_fill_holes((tot>threshold_otsu(tot)).max(0)),MIN_CELL_AREA)))
        MP.append((a[:,CH_ACTIN].max(0).astype(np.float32),a[:,CH_MT].max(0).astype(np.float32),a[:,CH_S9].max(0).astype(np.float32)))
        del a
    tracked=[np.zeros_like(x) for x in foots]; nid=1
    for rp in regionprops(foots[0]): tracked[0][foots[0]==rp.label]=nid; nid+=1
    for fi in range(1,len(foots)):
        prev,cur=tracked[fi-1],foots[fi]; pl=[r.label for r in regionprops(prev)]; cl=[r.label for r in regionprops(cur)]
        if pl and cl:
            iou=np.zeros((len(pl),len(cl)))
            for i,pv in enumerate(pl):
                pm=prev==pv
                for j,cv in enumerate(cl):
                    cm=cur==cv;u=(pm|cm).sum();iou[i,j]=(pm&cm).sum()/u if u else 0
            ri,cj=linear_sum_assignment(-iou); asg=set()
            for i,j in zip(ri,cj):
                if iou[i,j]>=IOU_GATE: tracked[fi][cur==cl[j]]=pl[i]; asg.add(cl[j])
            for j,cv in enumerate(cl):
                if cv not in asg: tracked[fi][cur==cv]=nid; nid+=1
        else:
            for cv in cl: tracked[fi][cur==cv]=nid; nid+=1
    # cell #5 identity in XY_5_20min
    c5=None
    if region=="Jul28_XY_5_20min":
        ids=sorted(set(np.unique(np.concatenate([t.ravel() for t in tracked])))-{0}); bb=None
        for lid in ids:
            ar=np.array([(t==lid).sum() if (t==lid).any() else np.nan for t in tracked],float); ok=np.isfinite(ar)&np.isfinite(CELL5_AREA)
            if ok.sum()>=5:
                c=np.corrcoef(ar[ok],CELL5_AREA[ok])[0,1]
                if bb is None or c>bb[1]: bb=(lid,c)
        c5=bb[0]
    labels=sorted(set(np.unique(np.concatenate([t.ravel() for t in tracked])))-{0})
    for lid in labels:
        rows=[]
        for fi in range(len(foots)):
            m=tracked[fi]==lid
            if not m.any(): continue
            ys,xs=np.where(m); mnr,mnc,mxr,mxc=ys.min(),xs.min(),ys.max(),xs.max()
            border=mnr<3 or mnc<3 or mxr>2303-3 or mxc>2303-3
            if border: continue
            r0,r1=max(0,mnr-PAD),min(2304,mxr+PAD); c0,c1=max(0,mnc-PAD),min(2304,mxc+PAD)
            fp=m[r0:r1,c0:c1]; A,Mt,S=[X[r0:r1,c0:c1] for X in MP[fi]]
            mm=metrics(A,Mt,S,fp)
            if mm is None: continue
            rows.append((T[fi],mm))
            if lid==c5 and fi in (0,4,8): cell5_qc[T[fi]]=mm
        if len(rows)<3: continue
        rows.sort(); k=max(1,len(rows)//3)
        def el(key): v=np.array([r[1][key] for r in rows]); return v[:k].mean(),v[-k:].mean()
        e_mt,l_mt=el("fr_s9_mt"); e_sf,l_sf=el("fr_s9_sf"); e_de,l_de=el("mt_dec")
        uid=f"{region}#{lid}"
        percell[uid]=dict(dish=dish,n=len(rows),
            e_mt=e_mt,l_mt=l_mt,e_sf=e_sf,l_sf=l_sf,e_de=e_de,l_de=l_de)
        for tmin,mm in rows:                       # per-timepoint rows feed the figure scripts
            percell_tp.append((region,dish,uid,tmin,mm["fr_s9_mt"],mm["fr_s9_sf"]))
    del MP,foots,tracked
    print(f"{region}: cells so far {len(percell)}",flush=True)

cells=list(percell.values()); print(f"\nTOTAL cells: {len(cells)}")
with open(C.out("faithful_percell.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["uid","dish","n","early_fracS9onMT","late_fracS9onMT",
        "early_fracS9onSF","late_fracS9onSF","early_MTdecorated","late_MTdecorated"])
    for uid,c in percell.items():
        w.writerow([uid,c["dish"],c["n"],f"{c['e_mt']:.3f}",f"{c['l_mt']:.3f}",
                    f"{c['e_sf']:.3f}",f"{c['l_sf']:.3f}",f"{c['e_de']:.3f}",f"{c['l_de']:.3f}"])
# per-timepoint table in the per_cell_sf_mt schema -> feed straight into the figure scripts
#   (set MCF_DATA to this file, then run reproduce_figures.py on YOUR data)
with open(C.out("per_cell_sf_mt_computed.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["region","dish","uid","t_min","frac_MT","frac_SF"])
    for reg,dsh,uid,tmin,fmt,fsf in percell_tp:
        w.writerow([reg,dsh,uid,tmin,f"{fmt:.3f}",f"{fsf:.3f}"])
print(f"wrote per_cell_sf_mt_computed.csv ({len(percell_tp)} cell-timepoints) -> set MCF_DATA to it to plot your own data")

def boot(d,n=10000,seed=0):
    rng=np.random.default_rng(seed); bs=np.array([rng.choice(d,len(d),True).mean() for _ in range(n)]); return d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),bs
def wil(a,b):
    try:
        from scipy.stats import wilcoxon; return wilcoxon(a,b).pvalue
    except Exception: return float('nan')
DCOL={"DISH1(20m)":"#2171b5","DISH2(30m)":"#d94801"}
METRICS=[("fr_s9 on MT","e_mt","l_mt"),("fr_s9 on SF","e_sf","l_sf"),("MT decorated by SEPT9","e_de","l_de")]
fig,axes=plt.subplots(2,3,figsize=(16,9))
for col,(name,ek,lk) in enumerate(METRICS):
    axL=axes[0,col]
    for c in cells: axL.plot([0,1],[c[ek],c[lk]],'-',color=DCOL[c["dish"]],alpha=.3,lw=.9)
    for g in DCOL:
        sub=[c for c in cells if c["dish"]==g]
        axL.plot([0,1],[np.mean([c[ek] for c in sub]),np.mean([c[lk] for c in sub])],'-o',color=DCOL[g],lw=3,ms=7,label=f"{g} n={len(sub)}")
    axL.set_xlim(-.3,1.3);axL.set_xticks([0,1]);axL.set_xticklabels(["early","late"]);axL.set_ylabel(name);axL.set_title(name,fontsize=10)
    if col==0: axL.legend(fontsize=8)
    axR=axes[1,col]; grps=[("DISH1(20m)",[c for c in cells if c['dish']=='DISH1(20m)']),("DISH2(30m)",[c for c in cells if c['dish']=='DISH2(30m)']),("ALL",cells)]
    print(f"\n{name}:")
    for i,(g,sub) in enumerate(grps):
        dd=np.array([c[lk]-c[ek] for c in sub]); m,lo,hi,bs=boot(dd); p=wil([c[lk] for c in sub],[c[ek] for c in sub])
        print(f"  {g:12} n={len(dd):2d} Δ={m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] p={p:.3f}")
        vp=axR.violinplot(bs,positions=[i],vert=False,showextrema=False,widths=.8)
        for b in vp['bodies']: b.set_facecolor(DCOL.get(g,"#555"));b.set_alpha(.5)
        axR.plot([lo,hi],[i,i],'k-',lw=1.5);axR.plot(m,i,'ko',ms=6);axR.text(hi,i,f" Δ={m:+.3f}[{lo:+.3f},{hi:+.3f}]",va='center',fontsize=8)
    axR.axvline(0,ls=':',c='k');axR.set_yticks(range(3));axR.set_yticklabels([g for g,_ in grps]);axR.set_xlabel(f"paired Δ {name}")
fig.suptitle("FAITHFUL coincidence (filament skeleton overlap, per-MT-length normalized, rim-excluded SF) — all Day-2 cells",fontsize=12)
plt.tight_layout(); plt.savefig(C.out("faithful_estimation.png"),dpi=140); plt.close(); print("\nsaved faithful_estimation.png")

# cell5 SF rim-fix QC
if cell5_qc:
    ts=sorted(cell5_qc); fig,ax=plt.subplots(1,len(ts),figsize=(5*len(ts),5))
    if len(ts)==1: ax=[ax]
    for j,t in enumerate(ts):
        mm=cell5_qc[t]; g=mm["_A"]; lo,hi=np.percentile(g[mm["_fp"]],(2,99)); base=np.dstack([np.clip((g-lo)/(hi-lo),0,1)*0.6]*3)
        base[mm["_sf"]]=[1,0.15,0.15]; base[mm["_fp"]&~mm["_int"]]=np.clip(base[mm["_fp"]&~mm["_int"]]+[0,0,0.25],0,1)  # rim band tinted blue
        ax[j].imshow(base); ax[j].set_title(f"t={t}  (red=SF interior only; blue tint=excluded rim)",fontsize=9); ax[j].set_xticks([]);ax[j].set_yticks([])
    fig.suptitle("Cell #5 SF mask after cortical-rim exclusion",fontsize=12)
    plt.tight_layout(); plt.savefig(C.out("cell5_SF_rimfix_QC.png"),dpi=140); plt.close(); print("saved cell5_SF_rimfix_QC.png")
