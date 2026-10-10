# Pre-specification of reading G-C1c

Stamped 10 October 2026, 11:32 BST, before the reading procedure below was run on the published figure.

## Question

Gate G-C1a compared the elastic contact model with $S_{j,ini}$ of Table 8 of Girão Coelho, Bijlaard and Simões da Silva (2004). That value comes from a regression on the unloading branch after a first loading, and the paper does not plot that branch. The plotted curves (Fig. 14) are the reloading to collapse. Do those curves, at low moments, agree with the model or with Table 8?

## Why this is a reading and not a gate

- Fig. 14 is a 200 dpi raster image with a rotation axis to 120 mrad: 3.9 pixels per mrad.
- At 40 kNm the model's rotation is about 1 mrad, four pixels from the y-axis line. Curves steeper than about 60 kNm/mrad merge with that line and cannot be read.
- The two tests of a series cannot be told apart near the origin.
- The pixel uncertainty is therefore of the order of 25% at the levels that can be read, so no pass or fail is attached. The comparison rule below is fixed in advance all the same.
- The decisive data are the raw records. A request to the authors has been drafted for the project owner to send.

## What was seen before the stamp

- The independent audit of G-C1a read Fig. 11(b) by eye for FS1a (a secant of 30 to 45 kNm/mrad below 60 kNm) and Fig. 14 for FS3 (about 1.7 times softer than the model at 30 kNm).
- While the method was being designed, the pixels of panel (a) (FS1) near the origin were printed once, to locate the axes and ticks.
- The procedure was then tested only on synthetic figures drawn on the same pixel grid (`checks/gc1c_synthetic.py`). The reading interval contained the true secant in about 95% of readable cases.

## Procedure

Fixed in `checks/gc1c_read.py`; summary:

1. **Image.** The embedded image of Fig. 14 (page 14 of the article PDF; 602 × 1573 pixels, 200 dpi, greyscale JPEG), extracted losslessly with `pdfimages -j`.
2. **Axes.** For each panel, the 13 ticks below the x-axis (0 to 120 mrad) and the 9 ticks left of the y-axis (0 to 240 kNm) are located. Linear least-squares maps are fitted, and their residuals are reported.
3. **Levels.** $M$ = 40 and 50 kNm, chosen to lie between the dotted gridlines at 30 and 60 kNm. At the two pixel rows bracketing each level, runs of dark pixels (grey < 128) right of the y-axis line and left of 9 mrad are curve crossings. Runs touching the axis line are unreadable.
4. **Rotation.** Each crossing gives $\phi$ at the run centre, $\pm$ (half the run width + 1 pixel).
5. **Secant.** $S = M/\phi$, with the interval that follows from the $\phi$ interval.
6. **Comparison rule.** At each level, a series is *consistent with the model* if the model's $S_{FE}$ lies inside $[S_{min}, S_{max}]$. Here $S_{FE}$ is from G-C1a, fine mesh: 37.26, 53.22, 64.40 and 35.42 kNm/mrad for FS1 to FS4. $[S_{min}, S_{max}]$ is the span of all secant intervals read for that series at that level. The same test is reported for the two Table 8 values of the series.

## Configuration

| File | SHA-256 |
|---|---|
| `checks/gc1c_read.py` | `fbf8e636a982ac4d504852b3137fa3fb7582abfddf70aebcbb3e216b53dfb8b0` |
| Fig. 14 image | `dfdb8e1d9c7a049d82be27b98f4770288b0378573ccab7dccf8db3c338cc90a1` |

## References

- Girão Coelho, A.M., Bijlaard, F.S.K., Simões da Silva, L. (2004). Experimental assessment of the ductility of extended end plate connections. *Engineering Structures* 26(9), 1185-1206. doi:10.1016/j.engstruct.2000.09.001
