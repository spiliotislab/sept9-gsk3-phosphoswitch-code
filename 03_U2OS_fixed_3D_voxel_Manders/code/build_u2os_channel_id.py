"""
Re-verify U2OS channel identities: render all 3 channels (640/561/488) at z2-3 with a zoomed
crop, so we can tell which is the fine MT NETWORK vs SEPT9 (punctate/bundled) vs actin(SF).
User flagged 488 looks MT-like. Output: u2os_channel_id.png  (+ Frangi network-energy per ch).
"""
import os, numpy as np, nd2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import _config as C
HERE=os.path.dirname(os.path.abspath(__file__)); IMG=C.require_images()
# Point U2OS_IMAGES at the folder holding these .nd2 files; edit the filenames for your own data.
FILES={"Control":os.path.join(IMG,"U2OS_Cntrl_3.nd2"),"CHIR":os.path.join(IMG,"U2OS_CHIR_11.nd2")}
LAB=["640 (assumed actin)","561 (assumed MT)","488 (assumed SEPT9)"]; ZLO,ZHI=2,4
CROP=(700,1100,700,1100)  # y0,y1,x0,x1 zoom window
def netE(img):  # crude ridge/network energy (mean |Laplacian of Gaussian|)
    g=ndi.gaussian_filter(img.astype(np.float32),2); return float(np.mean(np.abs(ndi.laplace(g))))
fig,ax=plt.subplots(4,3,figsize=(15,19))
for fi,(k,p) in enumerate(FILES.items()):
    a=nd2.imread(p)[ZLO:ZHI].astype(np.float32)   # (2,C,Y,X)
    for c in range(3):
        mp=a[:,c].max(0); lo,hi=np.percentile(mp,(2,99.6)); disp=np.clip((mp-lo)/(hi-lo+1e-9),0,1)
        r=2*fi
        ax[r,c].imshow(disp,cmap="gray"); ax[r,c].set_title(f"{k} — {LAB[c]}  (netE={netE(mp):.1f})",fontsize=10)
        y0,y1,x0,x1=CROP; cz=disp[y0:y1,x0:x1]
        ax[r+1,c].imshow(cz,cmap="gray"); ax[r+1,c].set_title(f"{k} zoom — {LAB[c].split('(')[0]}",fontsize=9)
        ax[r,c].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,ec="yellow",fc="none",lw=1))
        for rr in (r,r+1): ax[rr,c].set_xticks([]);ax[rr,c].set_yticks([])
        print(f"{k} ch idx{c} ({LAB[c]}): netE={netE(mp):.2f} mean={mp.mean():.0f} max={mp.max():.0f}",flush=True)
fig.suptitle("U2OS channel ID re-check — which is the fine MT NETWORK vs SEPT9 (punctate/bundled) vs actin (SF)?",fontsize=13)
plt.tight_layout(); plt.savefig(C.out("u2os_channel_id.png"),dpi=130,bbox_inches="tight"); plt.close(); print("saved u2os_channel_id.png")
