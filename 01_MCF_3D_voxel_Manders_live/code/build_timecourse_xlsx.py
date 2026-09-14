"""Excel time-course of the 13 responder cells: per-timepoint Manders (SEPT9 on MT, on actin-SF)
and redistribution vs t0 — with the MAX-RESPONSE timepoint highlighted per cell.
Source: per_cell_sf_mt.csv (3D voxel-Manders). -> SEPT9_timecourse_13responders.xlsx (package + local)."""
import os, pandas as pd, numpy as np
import _config as C
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
d=pd.read_csv(C.DATA_CSV)
REALEND={"20min":220.0,"30min":192.5}; maxnom={}
for uid,g in d.groupby("uid"):
    dur="30min" if "30min" in uid else "20min"; maxnom[dur]=max(maxnom.get(dur,0),g.t_min.max())
# select 13 responders
resp=[]
for uid,g in d.groupby("uid"):
    g=g.sort_values("t_min")
    if len(g)<7: continue
    e,l=g.iloc[0],g.iloc[-1]
    sc=(l.frac_MT-e.frac_MT)-(l.frac_SF-e.frac_SF)
    if sc>0: resp.append((uid,g))
resp.sort(key=lambda x:-(x[1].iloc[-1].frac_MT-x[1].iloc[0].frac_MT-(x[1].iloc[-1].frac_SF-x[1].iloc[0].frac_SF)))
HDR=Font(bold=True,color="FFFFFF"); HFILL=PatternFill("solid",fgColor="2F5597")
TITLE=Font(bold=True,size=11,color="1F3864"); PEAK=PatternFill("solid",fgColor="FFE699"); PB=Font(bold=True)
thin=Side(style="thin",color="BFBFBF"); BORD=Border(left=thin,right=thin,top=thin,bottom=thin)
COLS=["t (min, nominal)","~elapsed (min)","SEPT9 voxels","SEPT9 on MT (M1)","SEPT9 on actin-SF (M1)","redistribution vs t0"]
wb=Workbook(); ws=wb.active; ws.title="timecourse_13cells"
r=1
ws.cell(r,1,"SEPT9 Manders time-course — 13 responder cells (peak-response row highlighted)").font=Font(bold=True,size=13); r+=2
summary=[]
for uid,g in resp:
    dur="30min" if "30min" in uid else "20min"
    fMT0=g.iloc[0].frac_MT; fSF0=g.iloc[0].frac_SF
    g=g.assign(real=(g.t_min/maxnom[dur]*REALEND[dur]).round().astype(int),
               redist=((g.frac_MT-fMT0)-(g.frac_SF-fSF0)))
    ipk=g.redist.values.argmax(); pk=g.iloc[ipk]
    ws.cell(r,1,f"{uid}   (dur {dur}, n={len(g)} timepoints, peak response +{pk.redist:.2f} @ {int(pk.real)} min)").font=TITLE; r+=1
    for ci,h in enumerate(COLS,1):
        c=ws.cell(r,ci,h); c.font=HDR; c.fill=HFILL; c.alignment=Alignment(horizontal="center",wrap_text=True); c.border=BORD
    r+=1; peakrow=None
    for _,row in g.iterrows():
        vals=[int(row.t_min),int(row.real),int(row.S9_total),round(row.frac_MT,3),round(row.frac_SF,3),
              round(row.redist,3)]
        for ci,v in enumerate(vals,1):
            c=ws.cell(r,ci,v); c.border=BORD; c.alignment=Alignment(horizontal="center")
            if ci in (4,5,6): c.number_format="0.000"
        if abs(row.redist-pk.redist)<1e-9:  # peak-response row
            for ci in range(1,7): ws.cell(r,ci).fill=PEAK; ws.cell(r,ci).font=PB
            peakrow=r
        r+=1
    r+=1  # blank between cells
    summary.append(dict(cell=uid,dur=dur,n=len(g),
        MT_t0=round(fMT0,3),SF_t0=round(fSF0,3),
        peak_min=int(pk.real),peak_MT=round(pk.frac_MT,3),peak_SF=round(pk.frac_SF,3),peak_redist=round(pk.redist,3),
        late_MT=round(g.iloc[-1].frac_MT,3),late_SF=round(g.iloc[-1].frac_SF,3),
        final_redist=round(g.iloc[-1].redist,3)))
for col in ws.columns:
    m=max((len(str(c.value)) for c in col if c.value is not None),default=10); ws.column_dimensions[col[0].column_letter].width=min(max(m+2,11),22)
# summary sheet
s=wb.create_sheet("peak_summary")
sdf=pd.DataFrame(summary)
s.append(list(sdf.columns))
for c in s[1]: c.font=HDR; c.fill=HFILL
for _,row in sdf.iterrows(): s.append(list(row.values))
# highlight the peak_redist column
pk_col=list(sdf.columns).index("peak_redist")+1  # highlight the peak_redist column
for rr in range(2,s.max_row+1):
    s.cell(rr,pk_col).fill=PEAK; s.cell(rr,pk_col).font=PB
for col in s.columns:
    m=max((len(str(c.value)) for c in col if c.value is not None),default=10); s.column_dimensions[col[0].column_letter].width=min(max(m+2,10),16)
s.freeze_panes="A2"
outs=[C.out("SEPT9_timecourse_13responders.xlsx"),
      C.out("SEPT9_timecourse_13responders.xlsx")]
for o in outs: wb.save(o)
print(f"{len(resp)} cells written. peak-response highlighted. saved:\n  "+"\n  ".join(outs))
for row in summary: print(f"  {row['cell']:26s} peak +{row['peak_redist']:.2f}@{row['peak_min']}min  MT {row['MT_t0']:.2f}->{row['peak_MT']:.2f}  SF {row['SF_t0']:.2f}->{row['peak_SF']:.2f}")
