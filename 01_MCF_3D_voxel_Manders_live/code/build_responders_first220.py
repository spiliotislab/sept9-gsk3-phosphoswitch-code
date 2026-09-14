"""Manders M1 endpoint figure: responder cells whose last frame is ~220 min (drop the 30-min-dish
cells at ~192 and the short track at 165). Keep the XY_4 003 apical example (bold, illustrative).
Single 220-min endpoint: SEPT9 on MT (solid) + on actin stress fibers (dashed).
Source: per_cell_sf_mt.csv (3D voxel-Manders). -> responders_first220_PUB.png/.pdf/.svg"""
import pandas as pd, numpy as np
import _config as C
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    real_late=float(l.t_min)/maxnom[dur]*REALEND[dur]
    if real_late<205: continue                      # KEEP ONLY 220-min cells (drop 165 & 192 endpoints)
    cells.append(dict(uid=uid,MT_e=float(e.frac_MT),MT_l=float(l.frac_MT),
        SF_e=float(e.frac_SF),SF_l=float(l.frac_SF)))
cells.sort(key=lambda c:(c["MT_l"]-c["MT_e"])-(c["SF_l"]-c["SF_e"]),reverse=True)
n=len(cells); colors=plt.cm.turbo(np.linspace(0.05,0.95,n))
EX_MT=(0.47,0.63); EX_SF=(0.21,0.12)                 # XY_4 003 apical example (unchanged)
fig,A=plt.subplots(figsize=(6.4,6.4))
for i,c in enumerate(cells):
    A.plot([0,1],[c["MT_e"],c["MT_l"]],"-",color=colors[i],alpha=0.45,lw=1.4,marker="o",ms=4,mfc=colors[i],mec="none")
    A.plot([0,1],[c["SF_e"],c["SF_l"]],"--",color=colors[i],alpha=0.45,lw=1.2,marker="s",ms=3.5,mfc=colors[i],mec="none")
A.plot([0,1],EX_MT,"-o",color="k",lw=3.6,ms=9,zorder=10)
A.plot([0,1],EX_SF,"--s",color="k",lw=3.0,ms=8,zorder=10)
A.set_xticks([0,1]); A.set_xticklabels(["early\n(0 min)","~220 min"]); A.set_xlim(-0.15,1.2)
A.set_ylim(0,None); A.set_ylabel("Manders M1  (fraction of SEPT9 on network)")
A.set_title("SEPT9 on microtubules vs actin stress fibers (M1)")
A.legend(handles=[Line2D([0],[0],color="k",lw=3,marker="o",label="on microtubules (MT)"),
                  Line2D([0],[0],color="k",lw=2.4,ls="--",marker="s",label="on actin stress fibers")],
         loc="center left",fontsize=9,frameon=False,title="bold = XY_4 003 apical example")
fig.suptitle(f"SEPT9 redistribution per responder cell at 220 min (n={n}); bold = XY_4 003 apical example",fontsize=12)
plt.tight_layout(rect=[0,0,1,0.96])
import matplotlib as mpl; mpl.rcParams["pdf.fonttype"]=42; mpl.rcParams["svg.fonttype"]="none"
for ext in ["png","pdf","svg"]:
    plt.savefig(f'{C.out("responders_first220_PUB")}.{ext}',dpi=300,bbox_inches="tight")
print(f"n={n} cells (220-min only):",", ".join(c["uid"] for c in cells))
print("saved responders_first220_PUB.png/.pdf/.svg")
