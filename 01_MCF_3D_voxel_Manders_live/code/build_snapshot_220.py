"""Optimal-timepoint SNAPSHOT: SEPT9 redistribution at ~220 min post-CHIR (maximum-effect timepoint).
n = responder cells that tracked to the last frame (real timepoint >=205 min).
Manders M1 on MT (solid, up) and on actin stress fibers (dashed, down), t0 -> ~220 min,
per cell + bold mean + Wilcoxon; XY_4 003 example (whole-cell) overlaid.
Source: per_cell_sf_mt.csv (3D voxel-Manders). -> snapshot_220min.png"""
import pandas as pd, numpy as np
import _config as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon
d=pd.read_csv(C.DATA_CSV)
REALEND={"20min":220.0,"30min":192.5}; maxnom={}
for uid,g in d.groupby("uid"):
    dur="30min" if "30min" in uid else "20min"; maxnom[dur]=max(maxnom.get(dur,0),g.t_min.max())
cells=[]
for uid,g in d.groupby("uid"):
    g=g.sort_values("t_min")
    if len(g)<7: continue
    e,l=g.iloc[0],g.iloc[-1]
    if (l.frac_MT-e.frac_MT)-(l.frac_SF-e.frac_SF)<=0: continue
    dur="30min" if "30min" in uid else "20min"
    g=g.assign(real=g.t_min/maxnom[dur]*REALEND[dur])
    late=g[g.real>=205]
    if late.empty: continue                     # only cells reaching ~220 min
    lp=late.iloc[(late.real-220).abs().argmin()]
    cells.append(dict(uid=uid,MT_e=float(g.iloc[0].frac_MT),MT_l=float(lp.frac_MT),
        SF_e=float(g.iloc[0].frac_SF),SF_l=float(lp.frac_SF),tmin=float(lp.real)))
df=pd.DataFrame(cells); n=len(df)
def wp(a,b):
    try: return wilcoxon(a,b).pvalue
    except: return np.nan
pMT=wp(df.MT_e,df.MT_l); pSF=wp(df.SF_e,df.SF_l)
lo,hi=df.tmin.min(),df.tmin.max()
BLUE="#1f77b4"; ORANGE="#e8820c"
fig,A=plt.subplots(figsize=(6.6,6.4))
for _,r in df.iterrows():
    A.plot([0,1],[r.MT_e,r.MT_l],"-",color=BLUE,alpha=0.3,lw=1.2,marker="o",ms=4)
    A.plot([0,1],[r.SF_e,r.SF_l],"--",color=ORANGE,alpha=0.3,lw=1.2,marker="s",ms=3.5)
A.plot([0,1],[df.MT_e.mean(),df.MT_l.mean()],"-o",color="#08306b",lw=4,ms=11,zorder=10)
A.plot([0,1],[df.SF_e.mean(),df.SF_l.mean()],"--s",color="#7f2704",lw=4,ms=11,zorder=10)
# XY_4 003 example (bold black) at UNIFIED 220 min — WHOLE-CELL (reliable, no z-split): 003->044
A.plot([0,1],[0.413,0.716],"-o",color="k",lw=3,ms=8,zorder=11)
A.plot([0,1],[0.285,0.125],"--s",color="k",lw=3,ms=7,zorder=11)
A.text(1.04,df.MT_l.mean(),f"+{df.MT_l.mean()-df.MT_e.mean():.02f}\np={pMT:.3f}",color="#08306b",va="center",fontsize=10,weight="bold")
A.text(1.04,df.SF_l.mean(),f"{df.SF_l.mean()-df.SF_e.mean():+.02f}\np={pSF:.3f}",color="#7f2704",va="center",fontsize=10,weight="bold")
A.set_xticks([0,1]); A.set_xticklabels(["t0","~220 min"]); A.set_xlim(-0.15,1.4); A.set_ylim(0,None)
A.set_ylabel("Manders M1 (fraction of SEPT9 on network)")
A.set_title("SEPT9 on microtubules vs actin stress fibers")
A.legend(handles=[Line2D([0],[0],color="#08306b",lw=3,marker="o",label="on microtubules (MT)"),
                  Line2D([0],[0],color="#7f2704",lw=3,ls="--",marker="s",label="on actin stress fibers"),
                  Line2D([0],[0],color="k",lw=3,marker="o",label="XY_4 003 example @220 min (whole-cell)")],
         loc="center left",fontsize=8.5,frameon=False)
fig.suptitle(f"Optimal timepoint snapshot — SEPT9 redistribution at ~220 min post-CHIR (n={n} late responders)",fontsize=12)
plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(C.out("snapshot_220min.png"),dpi=150,bbox_inches="tight")
print(f"n={n} late responders @ {lo:.0f}-{hi:.0f}min")
print(f"  on-MT {df.MT_e.mean():.3f}->{df.MT_l.mean():.3f} p={pMT:.4f}")
print(f"  on-SF {df.SF_e.mean():.3f}->{df.SF_l.mean():.3f} p={pSF:.4f}")
print("  cells:",", ".join(df.uid))
