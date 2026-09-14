"""Single Manders M1 panel (Panel 1 only) with stats: SEPT9 on MT (solid) + on actin-SF (dashed),
per responder cell reaching 220 min (n=8) + XY_4 003 apical example (bold, illustrative).
Wilcoxon matched-pairs p annotated per network (computed on the n=8 population, example excluded).
-> manders_panel_stats_PUB.png/.pdf/.svg"""
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
    if float(l.t_min)/maxnom[dur]*REALEND[dur]<205: continue
    cells.append(dict(MT_e=float(e.frac_MT),MT_l=float(l.frac_MT),SF_e=float(e.frac_SF),SF_l=float(l.frac_SF)))
df=pd.DataFrame(cells); n=len(df); colors=plt.cm.turbo(np.linspace(0.05,0.95,n))
EX_MT=(0.47,0.63); EX_SF=(0.21,0.12)
pMT=wilcoxon(df.MT_e,df.MT_l).pvalue; pSF=wilcoxon(df.SF_e,df.SF_l).pvalue
upMT=int((df.MT_l>df.MT_e).sum()); dnSF=int((df.SF_l<df.SF_e).sum())
fig,A=plt.subplots(figsize=(6.2,6.6))
for i,c in df.iterrows():
    A.plot([0,1],[c.MT_e,c.MT_l],"-",color=colors[i],alpha=0.5,lw=1.5,marker="o",ms=4,mfc=colors[i],mec="none")
    A.plot([0,1],[c.SF_e,c.SF_l],"--",color=colors[i],alpha=0.5,lw=1.3,marker="s",ms=3.5,mfc=colors[i],mec="none")
A.plot([0,1],EX_MT,"-o",color="k",lw=3.4,ms=9,zorder=10)
A.plot([0,1],EX_SF,"--s",color="k",lw=3.0,ms=8,zorder=10)
A.set_xticks([0,1]); A.set_xticklabels(["early\n(0 min)","~220 min"]); A.set_xlim(-0.15,1.25); A.set_ylim(0,0.82)
A.set_ylabel("Manders M1  (fraction of SEPT9 on network)")
A.set_title("SEPT9 on microtubules vs actin stress fibers")
def sigbar(y,p,upn,txt):
    A.plot([0,0,1,1],[y-0.013,y,y,y-0.013],color="#333",lw=1.1)
    A.text(0.5,y+0.006,f"{txt}: Wilcoxon p = {p:.3f}  ({upn}/{n})",ha="center",va="bottom",fontsize=9,weight="bold")
sigbar(0.755,pMT,f"{upMT} up","on MT")
sigbar(0.245,pSF,f"{dnSF} down","on actin-SF")
A.legend(handles=[Line2D([0],[0],color="#888",lw=2.5,marker="o",label="on microtubules (MT)  — solid"),
                  Line2D([0],[0],color="#888",lw=2.2,ls="--",marker="s",label="on actin stress fibers  — dashed")],
         loc="center left",bbox_to_anchor=(0.02,0.44),fontsize=9,frameon=False)
A.text(0.5,-0.155,f"n = {n} responder cells; Wilcoxon matched-pairs signed-rank.\nbold black = XY_4 003 example (illustrative, excluded from stats).",
       transform=A.transAxes,ha="center",va="top",fontsize=8.5,color="#555")
plt.tight_layout()
import matplotlib as mpl; mpl.rcParams["pdf.fonttype"]=42; mpl.rcParams["svg.fonttype"]="none"
for ext in ["png","pdf","svg"]:
    for base in [C.out("manders_panel_stats_PUB"),
                 C.out("manders_panel_stats_PUB")]:
        plt.savefig(f"{base}.{ext}",dpi=300,bbox_inches="tight")
print(f"n={n}  on-MT median {df.MT_e.median():.3f}->{df.MT_l.median():.3f} p={pMT:.4f} ({upMT}/{n} up)")
print(f"      on-SF median {df.SF_e.median():.3f}->{df.SF_l.median():.3f} p={pSF:.4f} ({dnSF}/{n} down)")
print("saved manders_panel_stats_PUB.png/.pdf/.svg")
