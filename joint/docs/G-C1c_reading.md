# Reading G-C1c: low-moment secants of the published joint curves

Pre-specified in `docs/PRESPEC_G-C1c.md`; the pre-specification was saved at 11:31:44 BST on 10 October 2026 and the readings at 11:31:52 (the stamp line says 11:32, rounded). Generated from `checks/gc1c_readings.json`. The independent audit of 10 October 2026 reproduced the axis calibration and the crossings pixel for pixel.

Source: Fig. 14 of Girão Coelho, Bijlaard and Simões da Silva (2004), the embedded 200 dpi image (SHA-256 `dfdb8e1d9c7a049d82be27b98f4770288b0378573ccab7dccf8db3c338cc90a1`). About 3.9 pixels per mrad and 1.2 pixels per kNm; tick-fit residuals under 0.9 pixel.

| Series | Level (kNm) | Secants read (kNm/mrad), with the pixel interval | Model, G-C1a fine | Table 8 | Model inside the span | Table 8 inside the span |
|---|---|---|---|---|---|---|
| FS1 | 40 | 34.6 (26.0 to 51.6); 18.4 (15.7 to 22.3); 38.9 (31.2 to 51.6); 19.5 (15.7 to 26.0) | 37.26 | 18.19, 16.84 | yes | yes, yes |
| FS1 | 50 | 30.0 (24.4 to 38.9); 17.8 (15.1 to 21.7) | 37.26 | 18.19, 16.84 | yes | yes, yes |
| FS2 | 40 | 26.4 (19.8 to 39.7); 24.4 (19.8 to 31.7) | 53.22 | 23.39, 22.0 | no | yes, yes |
| FS2 | 50 | 24.7 (19.8 to 33.0) | 53.22 | 23.39, 22.0 | no | yes, yes |
| FS3 | 40 | 29.1 (21.2 to 46.4) | 64.4 | 23.23, 21.56 | no | yes, yes |
| FS3 | 50 | 28.4 (23.3 to 36.4); 30.7 (23.3 to 44.7) | 64.4 | 23.23, 21.56 | no | no, no |
| FS4 | 40 | 21.5 (17.9 to 27.1) | 35.42 | 16.18, 17.15 | no | no, no |
| FS4 | 50 | 19.1 (16.7 to 22.3); 20.1 (16.7 to 25.2) | 35.42 | 16.18, 17.15 | no | no, yes |

Two crossings at a level are the two tests of the series; they cannot be told apart near the origin. The two pixel rows that bracket a level often give the same crossing, listed once.

**Reading.**
- For FS2, FS3 and FS4 the loading curves at 40 and 50 kNm lie much nearer Table 8 than the model. The secants are 19 to 31 kNm/mrad, 9% to 30% above Table 8, against the model's 35 to 64.
- FS1 splits. The stiffer curve (FS1a, as identified by the G-C1a audit) is near the model at 40 kNm and about 30 at 50 kNm; the softer curve is near Table 8.
- The pixel uncertainty is about 25%, so no pass or fail is attached. The raw records would settle it.
