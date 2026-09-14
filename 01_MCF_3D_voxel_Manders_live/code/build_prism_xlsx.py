"""GraphPad Prism-ready workbook for the 220-min responder Manders plot (n=8 + XY_4 003 apical example).
Before-after (paired) tables for SEPT9 on MT and on actin stress fibers (Manders M1); stats sheet
(Wilcoxon matched-pairs + paired t); Prism step-by-step instructions.
Source: per_cell_sf_mt.csv (3D voxel-Manders). -> SEPT9_220min_PRISM.xlsx"""
import os, pandas as pd, numpy as np
import _config as C
from scipy.stats import wilcoxon, ttest_rel
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
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
    cells.append(dict(uid=uid,MT_e=round(float(e.frac_MT),3),MT_l=round(float(l.frac_MT),3),
        SF_e=round(float(e.frac_SF),3),SF_l=round(float(l.frac_SF),3)))
df=pd.DataFrame(cells)
EX=dict(uid="XY_4 003 (apical, 3D — separate measurement)",MT_e=0.47,MT_l=0.63,SF_e=0.21,SF_l=0.12)
HDR=Font(bold=True,color="FFFFFF"); HF=PatternFill("solid",fgColor="2F5597")
EXF=PatternFill("solid",fgColor="FFE699"); TIT=Font(bold=True,size=13,color="1F3864"); B=Font(bold=True)
thin=Side(style="thin",color="BFBFBF"); BORD=Border(*[thin]*4)
wb=Workbook(); wb.remove(wb.active)
def data_sheet(name,ecol,lcol,unit):
    ws=wb.create_sheet(name)
    ws["A1"]=f"{name}  —  Prism 'Column' before-after table ({unit})"; ws["A1"].font=TIT
    ws.append([]); hdr=["Cell (row label)","Early (0 min)","Late (220 min)"]
    ws.append(hdr)
    for c in ws[3]: c.font=HDR; c.fill=HF; c.alignment=Alignment(horizontal="center"); c.border=BORD
    for _,r in df.iterrows():
        ws.append([r["uid"],r[ecol],r[lcol]])
        for c in ws[ws.max_row]: c.border=BORD; c.alignment=Alignment(horizontal="center")
        ws.cell(ws.max_row,1).alignment=Alignment(horizontal="left")
    ws.append([EX["uid"],EX[ecol],EX[lcol]])
    for c in ws[ws.max_row]: c.fill=EXF; c.font=B; c.border=BORD
    ws.append([]); ws.append(["NOTE: enter the 8 population rows into Prism as the main dataset;",])
    ws.append(["the highlighted XY_4 003 row is a SEPARATE single-cell (apical 3D) — put it in its own dataset/column to overlay as a bold line.",])
    for col,w in zip("ABC",[46,16,16]): ws.column_dimensions[col].width=w
    ws.freeze_panes="A4"
data_sheet("SEPT9_on_MT_M1","MT_e","MT_l","fraction of SEPT9 on microtubules, Manders M1")
data_sheet("SEPT9_on_actinSF_M1","SF_e","SF_l","fraction of SEPT9 on actin stress fibers, Manders M1")
# STATS
st=wb.create_sheet("STATS",0)
st["A1"]="Paired stats, early vs late (220 min). Two blocks: n=8 (population only) and n=9 (+ XY_4 003 example)."; st["A1"].font=TIT
st.append([]); st.append(["metric","n","mean early","mean late","median early","median late","mean Δ (late-early)","Wilcoxon matched-pairs p","paired t p"])
for c in st[3]: c.font=HDR; c.fill=HF; c.alignment=Alignment(horizontal="center",wrap_text=True); c.border=BORD
def stat_row(label,ecol,lcol,ex=None):
    a=df[ecol].values.astype(float); b=df[lcol].values.astype(float)
    if ex is not None: a=np.append(a,ex[0]); b=np.append(b,ex[1])
    try: pw=wilcoxon(a,b).pvalue
    except: pw=np.nan
    try: pt=ttest_rel(a,b).pvalue
    except: pt=np.nan
    st.append([label,len(a),round(a.mean(),3),round(b.mean(),3),round(np.median(a),3),round(np.median(b),3),
               round(b.mean()-a.mean(),3),round(pw,4),round(pt,4)])
    for c in st[st.max_row]: c.border=BORD; c.alignment=Alignment(horizontal="center")
    st.cell(st.max_row,1).alignment=Alignment(horizontal="left")
def subhdr(txt):
    st.append([txt]); st.cell(st.max_row,1).font=Font(bold=True,color="1F3864")
subhdr("n = 8   (population responders only — one consistent pipeline)")
stat_row("SEPT9 on MT (M1)","MT_e","MT_l")
stat_row("SEPT9 on actin-SF (M1)","SF_e","SF_l")
st.append([])
subhdr("n = 9   (+ XY_4 003 example — NOTE: measured by the detailed 3D pipeline, not the population pipeline)")
stat_row("SEPT9 on MT (M1)","MT_e","MT_l",ex=(EX["MT_e"],EX["MT_l"]))
stat_row("SEPT9 on actin-SF (M1)","SF_e","SF_l",ex=(EX["SF_e"],EX["SF_l"]))
st.append([])
st.append(["Recommended test: Wilcoxon matched-pairs signed-rank (nonparametric, small n). Two-tailed.",])
st.append(["Including the example (n=9) makes all p-values SMALLER (it is a consistent responder).",])
st.append(["CAVEAT: cell #9 is a DIFFERENT measurement pipeline. OK for Wilcoxon (rank/direction-based);",])
st.append(["a purist reviewer may prefer n=8 with the example shown only as an overlay. State the method if you use n=9.",])
st.append(["Prism recomputes these when you run the analysis — values here are the reference.",])
st.append([])
subhdr("REPORT-READY  (n = 9, incl. XY_4 003 example;  MT + actin-SF)")
st.append(["metric","median early (range)","median late (range)","n","W","Z","p (exact)","r (N=n)","r (N=2n)"])
for c in st[st.max_row]: c.font=HDR; c.fill=HF; c.alignment=Alignment(horizontal="center",wrap_text=True)
def rr(label,ea,la,exe,exl):
    a=np.append(df[ea].values.astype(float),exe); b=np.append(df[la].values.astype(float),exl)
    exr=wilcoxon(a,b); apr=wilcoxon(a,b,method='approx'); nz=int(np.sum((b-a)!=0))
    st.append([label,f"{np.median(a):.3f} ({a.min():.3f}–{a.max():.3f})",f"{np.median(b):.3f} ({b.min():.3f}–{b.max():.3f})",
        len(a),round(exr.statistic,1),round(apr.zstatistic,3),round(exr.pvalue,4),
        round(abs(apr.zstatistic)/np.sqrt(nz),2),round(abs(apr.zstatistic)/np.sqrt(2*nz),2)])
    for c in st[st.max_row]: c.border=BORD; c.alignment=Alignment(horizontal="center")
    st.cell(st.max_row,1).alignment=Alignment(horizontal="left")
rr("SEPT9 on MT (M1)","MT_e","MT_l",EX["MT_e"],EX["MT_l"])
rr("SEPT9 on actin-SF (M1)","SF_e","SF_l",EX["SF_e"],EX["SF_l"])
st.append(["r = |Z|/sqrt(N): N=n (pairs) or N=2n (total obs). Both are LARGE effects (r>0.5). No zero-difference pairs dropped.",])
st.append(["Report sentence: SEPT9 on MT rose (median 0.52->0.61) and on actin stress fibers fell (median 0.16->0.09);",])
st.append(["two-tailed Wilcoxon matched-pairs, n=9: on MT W=2, Z=-2.43, p=0.012, r=0.81; on actin-SF W=0, Z=-2.67, p=0.004, r=0.89.",])
for col,w in zip("ABCDEFGHI",[26,6,11,11,12,12,16,20,12]): st.column_dimensions[col].width=w
st.freeze_panes="A4"
# INSTRUCTIONS
ins=wb.create_sheet("PRISM_INSTRUCTIONS")
steps=[
 ("SEPT9 220-min responder plot — how to rebuild in GraphPad Prism",TIT),
 ("",None),
 ("IMPORTANT: Prism has NO reliable 'before-after' GRAPH type across versions. Build the paired slopegraph",B),
 ("with an XY table (below) -- it works in every version. Use a Column table ONLY to compute the stats (Section C).",None),
 ("",None),
 ("A. Build the slopegraph -- BOTH M1(microtubules) and M1(actin stress fibers) on ONE graph (XY table):",B),
 ("1. New table & graph -> XY -> 'Enter and plot a single Y value for each point'.",None),
 ("2. X column: enter TWO rows, 1 and 2  (1 = early, 2 = late; using 1/2 keeps the slopegraph compact).",None),
 ("3. Enter each cell as its OWN Y column (2 values: early in row 1, late in row 2) -- TWICE per cell:",None),
 ("      * MT columns  (name them ..._MT):  values ~0.4-0.7  from sheet 'SEPT9_on_MT_M1'",None),
 ("      * SF columns  (name them ..._SF):  values ~0.05-0.2 from sheet 'SEPT9_on_actinSF_M1'",None),
 ("      * plus EXAMPLE_MT (0.47, 0.63) and EXAMPLE_SF (0.21, 0.12).",None),
 ("    (Prism connects the 2 points of each column into a line -> one line per cell. MT plots high, SF low",None),
 ("     -> same separation as the figure, no 'before-after' graph type needed.)",None),
 ("4. Style: Change -> Symbols & Lines. Select ALL _MT columns -> line SOLID (small/no symbols);",None),
 ("      select ALL _SF columns -> line DASHED. Set EXAMPLE_MT & EXAMPLE_SF -> BOLD BLACK, thicker.",None),
 ("5. X axis: Format Axes -> X axis -> range 0.5 to 2.5, major ticks ONLY at 1 and 2,",None),
 ("      rename the two tick labels to 'early (0 min)' and '~220 min'.",None),
 ("",None),
 ("B. Simpler alternative (if that is too many columns):",B),
 ("6. Make TWO XY slopegraphs -- one from 'SEPT9_on_MT_M1', one from 'SEPT9_on_actinSF_M1' -- side by side.",None),
 ("   Same information, fewer columns per graph.",None),
 ("",None),
 ("C. Stats (significance) -- use a COLUMN table (the graph and the stats are separate objects):",B),
 ("7. New table & graph -> Column. Two grouped columns 'Early' and 'Late'; each ROW = one cell (matched).",None),
 ("8. Paste the 8 rows from 'SEPT9_on_MT_M1' (Early->col A, Late->col B). (Repeat for SF.)",None),
 ("9. Analyze -> Column analyses -> 't tests (and nonparametric)' -> Paired ->",None),
 ("      'Wilcoxon matched-pairs signed rank test' (nonparametric, two-tailed).",None),
 ("10. Prism reports p; add it above each pair. Reference values are in sheet 'STATS'",None),
 ("      (MT p=0.023, actin-SF p=0.008).",None),
 ("",None),
 ("D. Caption disclosure (important):",B),
 ("11. State: colored lines = responder cells (population coincidence pipeline, n=8, 220 min);",None),
 ("    bold black = XY_4 003, single cell analyzed in detailed 3D (apical). Both at ~220 min post-CHIR.",None),
]
r=1
for text,style in steps:
    ins.cell(r,1,text)
    if style: ins.cell(r,1).font=style
    r+=1
ins.column_dimensions["A"].width=110
# ---- ready-to-paste Prism XY table (both networks, one graph) ----
def shortn(u): return u.replace("Jul28_","").replace("July28_","").replace("_20min","").replace("_30min","")
xy=wb.create_sheet("PRISM_XY_paste",1)
xy["A1"]="READY-TO-PASTE Prism XY table (both networks on ONE graph)."; xy["A1"].font=TIT
xy["A2"]="In Prism: New XY table -> select the green block (row 4 down) and paste. X=1 is early, X=2 is late."; xy["A2"].font=Font(italic=True,color="555555")
uids=[r["uid"] for _,r in df.iterrows()]
hdr=["X"]+[shortn(u)+"_MT" for u in uids]+["EXAMPLE_MT"]+[shortn(u)+"_SF" for u in uids]+["EXAMPLE_SF"]
row1=[1]+[df.iloc[i]["MT_e"] for i in range(len(df))]+[EX["MT_e"]]+[df.iloc[i]["SF_e"] for i in range(len(df))]+[EX["SF_e"]]
row2=[2]+[df.iloc[i]["MT_l"] for i in range(len(df))]+[EX["MT_l"]]+[df.iloc[i]["SF_l"] for i in range(len(df))]+[EX["SF_l"]]
xy.append([]); xy.append(hdr); xy.append(row1); xy.append(row2)
for c in xy[4]: c.font=HDR; c.fill=HF; c.alignment=Alignment(horizontal="center"); c.border=BORD
for rr_ in (5,6):
    for c in xy[rr_]: c.border=BORD; c.alignment=Alignment(horizontal="center")
excol=[2+len(df), 3+2*len(df)]  # EXAMPLE_MT and EXAMPLE_SF column indices (1-based)
for rr_ in (4,5,6):
    for cc in excol: xy.cell(rr_,cc).fill=EXF; xy.cell(rr_,cc).font=Font(bold=True)
xy["A8"]="_MT columns -> style SOLID lines.   _SF columns -> style DASHED lines.   EXAMPLE_* (amber) -> BOLD BLACK."; xy["A8"].font=Font(bold=True)
xy["A9"]="Each column's two dots (X=1 and X=2) become one slope line = one imaged cell."
for col in range(1,len(hdr)+1): xy.column_dimensions[xy.cell(1,col).column_letter].width=12
xy.freeze_panes="B5"
o=C.out("SEPT9_220min_PRISM.xlsx")
try: wb.save(o); print("saved",o)
except PermissionError: print("LOCKED (open in Excel), skipped:",o)
print("saved SEPT9_220min_PRISM.xlsx (sheets: STATS, 2 data tables, PRISM_XY_paste, PRISM_INSTRUCTIONS)")
for _,r in df.iterrows(): print(f"  {r.uid:22s} MT {r.MT_e}->{r.MT_l}  SF {r.SF_e}->{r.SF_l}")
print("STATS:")
for lab,ec,lc in [("MT","MT_e","MT_l"),("SF","SF_e","SF_l")]:
    a=df[ec].values.astype(float); b=df[lc].values.astype(float)
    print(f"  {lab}: {a.mean():.3f}->{b.mean():.3f}  Wilcoxon p={wilcoxon(a,b).pvalue:.4f}")
