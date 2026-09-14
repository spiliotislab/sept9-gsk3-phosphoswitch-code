# overlap_image_display — binary coincidence-mask TIFFs for Fig 1F and Fig 1H

Code for the paper's **Overlap image display** (Methods). SEPT9 and network (F-actin or microtubule)
masks are built in Python and exported as ImageJ composite TIFFs; in Fiji the coincidence channel is
shown as a **binary yellow mask** (uniform intensity, no grayscale gradient) over the two proteins.
This artificial overlap signal is **for visualization only** — no quantification uses it.

Prior to masking, each channel is contrast-adjusted by linear rescaling between fixed intensity
percentiles applied identically across compared conditions/time-points (common scaling). SEPT9 is
displayed as its filamentous+punctate (white-top-hat) signal; diffuse cytosolic and smooth nuclear
pools are removed. Overlap = signals within one pixel of spatial coincidence.

## Scripts (code/)
- **build_u2os_coincmap.py** — Fig 1F (fixed U2OS): SEPT9 & F-actin masks + SEPT9-cap-SF coincidence; exports ImageJ composite TIFFs (raw + binary SF/SEPT9/coincidence masks) and a coincidence map; Manders M1/M2 (basal z).
- **build_xy4_coincidence_tiff.py** — Fig 1H (live MCF7): XY_4 exemplar; exports 3-ch ImageJ composite TIFFs (max-proj + z-stack) network / SEPT9(top-hat) / coincidence(1-vox tol), for assembly in Fiji with a solid-yellow overlap LUT.
- **single_coinc_maps.py** — Fig 1H render variant: single-panel basal coincidence maps (SEPT9-cap-actin and SEPT9-cap-MT) for XY_4 003, coincidence forced to uniform yellow (no gradient).
- **build_dorsal_coincmap.py** — Fig 1H render variant: dorsal/medial-most N z-slices (dialable), NET=MT or ACTIN, uniform-yellow overlap; used for ventral-actin / medial-MT panels.

## Panel mapping
- **Figure 1F** (fixed U2OS, single ventral section, SEPT9 &cap; F-actin) -> `build_u2os_coincmap.py`.
- **Figure 1H** (live MCF7-GFP-SEPT9_i1, max-proj of 4 ventral + 4 medial sections; SEPT9 &cap; actin and SEPT9 &cap; MT) -> `build_xy4_coincidence_tiff.py` (TIFF export for Fiji); `single_coinc_maps.py` and `build_dorsal_coincmap.py` produce the equivalent uniform-yellow render panels.

## Notes
- These scripts are also duplicated in their analysis arms (`build_u2os_coincmap.py` in 03; `build_xy4_coincidence_tiff.py` in 01); this folder collects the full overlap-display set in one place.
- `build_xy4_coincidence_tiff.py` writes the coincidence channel as SEPT9 intensity where coincident so it stays LUT-adjustable in Fiji; the published panels set it to a solid yellow (binary) display.
- Scripts hard-code acquisition paths (D:/E: drives); update them for your environment. Raw/deconvolved stacks are deposited separately (Data Availability).