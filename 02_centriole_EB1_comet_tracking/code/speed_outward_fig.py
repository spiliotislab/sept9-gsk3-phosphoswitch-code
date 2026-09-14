"""
Speed slide restricted to OUTWARD (astral) comets = the ones growing OUT of the aster (2026-08-15).
Per aster: MEAN speed of outward comets (>=3), ALL asters. x = total comets (crowding proxy).
Overwrites outputs/reviewer_speed_mean.png (the reviewer-response figure picks it up).
"""
import os,numpy as np,pandas as pd,shutil
from scipy.stats import mannwhitneyu,spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _config as CFG
OUT=CFG.OUTDIR
cc=pd.read_csv(os.path.join(OUT,"aster_allcomet_persistence_comets.csv")); cc["unit"]=cc.base+"|"+cc.aster.astype(str)
tot=cc.groupby("unit").size().rename("ncomet")
ow=cc[cc.outward==1].groupby(["unit","geno"]).speed_nm_s.mean().reset_index().rename(columns={"speed_nm_s":"mean_out"})
now=cc[cc.outward==1].groupby("unit").size().rename("nout")
g=ow.merge(tot,on="unit").merge(now,on="unit"); g=g[g.nout>=3]
C={"WT":"#4C9F70","AA":"#E1655B"}
def st(p): return "n.s." if p>=.05 else ("*" if p>=.01 else ("**" if p>=.001 else "***"))
wt=g[g.geno=='WT'].mean_out; aa=g[g.geno=='AA'].mean_out; p=mannwhitneyu(wt,aa).pvalue; rho,pr=spearmanr(g.ncomet,g.mean_out)
fig,ax=plt.subplots(1,2,figsize=(12,5))
for gg in ["WT","AA"]:
    d=g[g.geno==gg]; ax[0].plot(d.ncomet,d.mean_out,'o',ms=6,color=C[gg],mec="k",mew=.3,alpha=.8,label=f"{gg} (n={len(d)})")
ax[0].set_xlabel("comets per aster (crowding)"); ax[0].set_ylabel("per-aster MEAN speed, OUTWARD comets (nm/s)")
ax[0].legend(fontsize=9); ax[0].set_title(f"(A) MEAN speed of OUTWARD (astral) comets vs comet count\nSpearman rho={rho:.2f} (p={pr:.1g}); AA upper-right",fontsize=10); ax[0].grid(alpha=.15)
parts=ax[1].violinplot([wt,aa],positions=[0,1],showextrema=False,widths=.8)
for b,gg in zip(parts['bodies'],["WT","AA"]): b.set_facecolor(C[gg]); b.set_alpha(.45); b.set_edgecolor("k"); b.set_linewidth(.6)
for i,(d,gg) in enumerate([(wt,"WT"),(aa,"AA")]): ax[1].plot(np.random.normal(i,.05,len(d)),d,'o',ms=6,color=C[gg],mec="k",mew=.4,alpha=.8); ax[1].plot([i-.28,i+.28],[d.mean()]*2,'k-',lw=2.4)
top=max(wt.max(),aa.max()); yb=top*0.92; h=top*0.03; ax[1].plot([0,0,1,1],[yb,yb+h,yb+h,yb],'k',lw=1.1); ax[1].text(.5,yb+h*1.4,f"{st(p)}  p={p:.2g}",ha="center",fontsize=10)
ax[1].set_xticks([0,1]); ax[1].set_xticklabels([f"WT\nn={len(wt)}",f"AA\nn={len(aa)}"]); ax[1].set_ylabel("MEAN speed, OUTWARD comets (nm/s)")
ax[1].set_ylim(min(wt.min(),aa.min())*0.85,top*1.12); ax[1].set_title(f"(B) OUTWARD (astral) comets, ALL asters\nWT {wt.mean():.0f} vs AA {aa.mean():.0f} nm/s",fontsize=10); ax[1].grid(alpha=.15,axis='y')
fig.suptitle("073126: MEAN speed of OUTWARD (astral) comets — AA faster than WT (all asters, raw)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94]); f=os.path.join(OUT,"reviewer_speed_mean.png"); fig.savefig(f,dpi=160,facecolor="white")
print(f"OUTWARD mean speed WT {wt.mean():.0f} AA {aa.mean():.0f} p={p:.2g}  rho(speed,count)={rho:.2f}")
