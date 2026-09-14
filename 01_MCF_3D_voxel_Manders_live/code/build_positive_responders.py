"""Positive-responder panel to accompany the XY_4 live single-cell example.
Source: Day-2 3D voxel-Manders per_cell_sf_mt.csv (frac SEPT9 on MT / on SF), NOT the 2D max-Z full pass.
Select POSITIVE responders: redistribution score = (dfrac_MT)-(dfrac_SF) > 0, t0->last, and >=7 tracked
timepoints (drops short-track artifacts). Paired plot: SEPT9 fraction ON MT (rises) and ON SF (falls),
early->late; faint per cell + bold mean. Wilcoxon paired.
-> positive_responders.png + positive_responders.csv"""
import pandas as pd, numpy as np
import _config as C
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=pd.read_csv(C.DATA_CSV)
rows=[]
for uid,g in d.groupby("uid"):
    g=g.sort_values("t_min")
    if len(g)<7: continue
    e,l=g.iloc[0],g.iloc[-1]
    score=float((l.frac_MT-e.frac_MT)-(l.frac_SF-e.frac_SF))
    if score<=0: continue
    rows.append(dict(uid=uid,dur="30min" if "30min" in uid else "20min",n=len(g),
        MT_e=float(e.frac_MT),MT_l=float(l.frac_MT),SF_e=float(e.frac_SF),SF_l=float(l.frac_SF),
        score=round(score,3),is_xy4=("XY_4" in uid)))
df=pd.DataFrame(rows).sort_values("score",ascending=False)
df.to_csv(C.out("positive_responders.csv"),index=False)
def wp(a,b):
    try: return wilcoxon(a,b).pvalue
    except: return np.nan
print(f"POSITIVE RESPONDERS (score>0, n>=7): {len(df)} cells")
for _,r in df.iterrows(): print(f"  {r.uid:26s} MT {r.MT_e:.2f}->{r.MT_l:.2f}  SF {r.SF_e:.2f}->{r.SF_l:.2f}  score {r.score:+.2f}  n={r.n}")
print(f"\n  on-MT:  {df.MT_e.mean():.3f} -> {df.MT_l.mean():.3f}  p={wp(df.MT_e,df.MT_l):.4f}")
print(f"  on-SF:  {df.SF_e.mean():.3f} -> {df.SF_l.mean():.3f}  p={wp(df.SF_e,df.SF_l):.4f}")
# ---- figure: pooled Manders panel (SEPT9 onto MT, off stress fibers) ----
BLUE="#1f77b4"; ORANGE="#ff7f0e"
fig,A=plt.subplots(figsize=(6.6,6.2))
for _,r in df.iterrows():
    A.plot([0,1],[r.MT_e,r.MT_l],"-o",color=BLUE,alpha=.30,lw=1.1,ms=4,zorder=2)
    A.plot([0,1],[r.SF_e,r.SF_l],"-s",color=ORANGE,alpha=.30,lw=1.1,ms=4,zorder=2)
A.plot([0,1],[df.MT_e.mean(),df.MT_l.mean()],"-o",color="#08306b",lw=4,ms=11,zorder=10,label="on MT (mean)")
A.plot([0,1],[df.SF_e.mean(),df.SF_l.mean()],"-s",color="#7f2704",lw=4,ms=11,zorder=10,label="on stress fibers (mean)")
A.text(1.06,df.MT_l.mean()+0.025,f"+{df.MT_l.mean()-df.MT_e.mean():.02f}",color="#08306b",va="center",ha="left",fontsize=11,weight="bold")
A.text(1.06,df.SF_l.mean()-0.025,f"{df.SF_l.mean()-df.SF_e.mean():+.02f}",color="#7f2704",va="center",ha="left",fontsize=11,weight="bold")
A.set_xticks([0,1]); A.set_xticklabels(["early","late"]); A.set_xlim(-0.15,1.35); A.set_ylim(0,None)
A.set_ylabel("fraction of cell SEPT9 on network"); A.set_title(f"Positive responders (n={len(df)})\nSEPT9 shifts onto MT, off stress fibers")
A.legend(loc="center left",fontsize=9,frameon=False)
pMT=wp(df.MT_e,df.MT_l); pSF=wp(df.SF_e,df.SF_l)
A.text(0.5,0.02,f"paired Wilcoxon: MT p={pMT:.3f}   SF p={pSF:.3f}",transform=A.transAxes,ha="center",fontsize=9,color="#444")
fig.suptitle("SEPT9 positive responders (Day-2 3D voxel-Manders) — companion to XY_4 single-cell live figure",fontsize=12)
plt.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(C.out("positive_responders.png"),dpi=150,bbox_inches="tight")
print("\nsaved positive_responders.png + positive_responders.csv")
