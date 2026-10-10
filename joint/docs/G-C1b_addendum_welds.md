# Gate G-C1b: addendum on how the welds were drawn

Written on 10 October 2026, after G-C1b's verdict and after the audit of G-C1a2. **Not pre-specified. The verdict of G-C1b stands as stamped.**

## Why

- The audit of G-C1a2 found that the stair-step rule for fillet welds keeps or drops, by round-off, the cells whose centroids lie exactly on the weld face. The joint's flange welds lost all of them.
- The T-stub model of G-C1b uses the same rule: the line that defines `weld` in `build()` of `fejoint/tstub_hf.py`.
- On G-C1b's own meshes (`probes/tstub_weld_check.py`, mesh building only):
  - **Fine mesh, the primary.** In the 12 series with 5 mm welds, the weld reaches 6.43 mm up the flange instead of 7.07 mm, and the full 7.07 mm up the web: 0.959 of the fillet's volume. The 3 mm weld falls short up the flange (1.061 of the volume), the 7 mm weld on both faces (0.942), and the 8 mm weld (WT1f) up the web only (0.994).
  - **Medium mesh.** The 5 mm welds reach the full leg up the flange and fall short up the web.
- This explains G-C1b's mesh flags. From the medium to the fine mesh the stamped model softened by 1.7% to 2.1% in the 5 mm series (2.4% in WT2A, 3 mm), while the baseline stiffened by 1.4% to 2.1%; nine series were flagged.

## What was run

- G-C1b's model with the welds drawn the same way everywhere, both ways:
  - **kept**: every cell on the weld face kept. The weld contains the exact fillet, reaches the full leg on both faces, and has 1.06 to 1.14 of its volume on the fine mesh.
  - **dropped**: every such cell dropped. The weld lies inside the exact fillet, falls one cell short on both faces, and has 0.86 to 0.94 of its volume.
- The stamped `fejoint/tstub_hf.py` is unchanged on disk. The script checks its stamped hash and then changes one line in memory: `<= g.leg` becomes `<= g.leg + 1e-6` or `<= g.leg - 1e-6` (`probes/gc1b_weld_bracket.py`).
- All 15 series on both meshes, with G-C1b's load, solver settings and reading. G-C1b's own judging function, taken from `checks/gc1b_evaluate.py`, is applied to each variant.

## Results

**Stiffness.** On the fine mesh, against the stamped run:
- kept welds stiffen the model by 0.7% to 3.1%, and by 2.0% to 2.7% in the 5 mm series;
- dropped welds soften it by up to 2.3% (WT1f), and by 0.1% to 0.2% in the 5 mm series.

In every series the stiffness rises with the weld material: dropped, then stamped, then kept.

**Mesh change.** With the welds drawn the same way everywhere, the change from the medium to the fine mesh is −1.1% to −1.4% in every series (kept) and −0.5% to +0.2% (dropped). No series would be flagged.

**G-C1b's criteria** (fine mesh):

| Welds | Geometric mean of $R$ | Tests inside [0.80, 1.25], 17 needed | Would-be verdict |
|---|---|---|---|
| Stamped (the verdict) | 1.104 | 18 | PASS |
| Kept | 1.131 | 18 | PASS |
| Dropped | 1.101 | 18 | PASS |

- With kept welds the set inside changes: WT1d leaves ($R$ 1.27) and WT57_M20 enters (0.81).
- With dropped welds the set inside is the stamped one.
- 17 of the 25 tests lie inside the band at both ends.

**Exact fillets.** The kept weld contains the exact fillet and the dropped weld lies inside it.
- That the stiffness rises with the weld material is a theorem only for the compliance under the applied load. For $K = F/\Delta$, read at the ends of the flanges, it was observed in every series here, not proven.
- A model with exact fillets needs a mesh fitted to the weld face. Its discretisation error would be of order 1%: the bracket runs change by up to 1.4% from the medium to the fine mesh, about the half-width of the bracket in the 5 mm series.
- So on this family of meshes a model with exact fillets would likely fall between the two ends, up to a discretisation error of about 1%. On that assumption:
  - at least 17 tests would lie inside the band, the number needed;
  - the geometric mean would lie between 1.101 and 1.131;
  - C2 would hold, since its orders hold across the whole bracket;
  - G-C1b would pass, with no guaranteed margin.

## What it means

- **The verdict stands**, with the reading G-C1b gave it: the model matches the welded T-stubs within the band on average, and narrowly.
- **The guaranteed margin is gone.** Only 17 tests are inside at both ends of the bracket, exactly the number needed.
  - With kept welds, 6% to 14% more than the fillet, the geometric mean rises from 1.10 to 1.13. With exact fillets it would lie between 1.10 and 1.13.
  - The count with exact fillets could be 17, 18 or 19. WT1d stays inside if WT1 lies in the lower 52% of its bracket; WT57_M20 enters if it lies in the upper 33% of its own.
- **G-C1b's robustness.** G-C1b's audit combined plate thickness, plain shank and loaded-end grip in 12 ways: 14 to 19 tests inside, 4 combinations failing.
  - Kept welds raise $R$ by 0.7% to 3.1%. Exact fillets would change it by between −2.3% and +3.1%, and by −0.2% to +2.7% in the 5 mm series.
  - That could move about one test across the top of the band in some combinations. Those combinations were not rerun.
- **G-C1a2.** The joint's flange welds were one cell short too. With full-size welds that model would be stiffer, so its fail would not change.
- **The code.** Any later version should replace the line with a rule that does not depend on round-off. Keeping the cells on the weld face is what the comment in `fejoint/joint_hf.py` (`_weld_volume_ratio`) implies. The stamped files stay as they are.

## Numbers

<!-- generated by probes/gc1b_weld_bracket_tables.py -->

### Welds as meshed (fine mesh; leg and reach in mm)

| Series | throat | leg | up the flange: stamped | kept | dropped | volume: stamped | kept | dropped |
|---|---|---|---|---|---|---|---|---|
| WT1 | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT1f | 8 | 11.31 | 11.31 | 11.31 | 10.69 | 0.994 | 1.056 | 0.944 |
| WT2A | 3 | 4.24 | 3.64 | 4.24 | 3.64 | 1.061 | 1.143 | 0.857 |
| WT2B | 7 | 9.90 | 9.24 | 9.90 | 9.24 | 0.942 | 1.067 | 0.933 |
| WT4A | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT51 | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT53C | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT53D | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT53E | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT57_M12 | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT57_M16 | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT57_M20 | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT7_M12 | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT7_M16 | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |
| WT7_M20 | 5 | 7.07 | 6.43 | 7.07 | 6.43 | 0.959 | 1.091 | 0.909 |

### Stiffness of the model (kN/mm)

| Series | stamped medium | kept medium | dropped medium | stamped fine | kept fine | dropped fine | kept / stamped, fine |
|---|---|---|---|---|---|---|---|
| WT1 | 151.67 | 154.28 | 148.52 | 148.56 | 152.50 | 148.35 | 1.027 |
| WT1f | 180.03 | 187.78 | 179.36 | 183.91 | 185.21 | 179.64 | 1.007 |
| WT2A | 135.89 | 136.55 | 132.28 | 132.60 | 135.09 | 132.07 | 1.019 |
| WT2B | 166.42 | 174.34 | 166.42 | 166.76 | 171.98 | 166.74 | 1.031 |
| WT4A | 194.95 | 197.92 | 191.30 | 191.18 | 195.70 | 190.94 | 1.024 |
| WT51 | 147.08 | 149.60 | 143.98 | 144.00 | 147.84 | 143.80 | 1.027 |
| WT53C | 146.63 | 149.14 | 143.55 | 143.57 | 147.39 | 143.38 | 1.027 |
| WT53D | 147.05 | 149.57 | 143.95 | 143.98 | 147.83 | 143.79 | 1.027 |
| WT53E | 146.20 | 148.70 | 143.12 | 143.15 | 146.97 | 142.95 | 1.027 |
| WT57_M12 | 81.60 | 82.71 | 80.28 | 80.18 | 81.81 | 80.06 | 1.020 |
| WT57_M16 | 97.58 | 99.04 | 95.89 | 95.87 | 97.99 | 95.72 | 1.022 |
| WT57_M20 | 121.52 | 123.56 | 119.26 | 118.93 | 121.78 | 118.70 | 1.024 |
| WT7_M12 | 82.38 | 83.49 | 81.08 | 80.97 | 82.59 | 80.86 | 1.020 |
| WT7_M16 | 99.96 | 101.47 | 98.25 | 98.24 | 100.40 | 98.08 | 1.022 |
| WT7_M20 | 123.59 | 125.66 | 121.34 | 121.02 | 123.88 | 120.79 | 1.024 |

### Mesh change, fine over medium

| Series | stamped | kept | dropped |
|---|---|---|---|
| WT1 | -2.1% | -1.2% | -0.1% |
| WT1f | +2.2% | -1.4% | +0.2% |
| WT2A | -2.4% | -1.1% | -0.2% |
| WT2B | +0.2% | -1.4% | +0.2% |
| WT4A | -1.9% | -1.1% | -0.2% |
| WT51 | -2.1% | -1.2% | -0.1% |
| WT53C | -2.1% | -1.2% | -0.1% |
| WT53D | -2.1% | -1.2% | -0.1% |
| WT53E | -2.1% | -1.2% | -0.1% |
| WT57_M12 | -1.7% | -1.1% | -0.3% |
| WT57_M16 | -1.7% | -1.1% | -0.2% |
| WT57_M20 | -2.1% | -1.4% | -0.5% |
| WT7_M12 | -1.7% | -1.1% | -0.3% |
| WT7_M16 | -1.7% | -1.1% | -0.2% |
| WT7_M20 | -2.1% | -1.4% | -0.5% |

### G-C1b's criteria applied to each variant

| Run | geometric mean R | inside [0.80, 1.25] (17 needed) | outside | C2 | would-be verdict |
|---|---|---|---|---|---|
| stamped, fine (the verdict) | 1.104 | 18 | WT1a, WT1b, WT1f, WT2Ba, WT4Aa, WT53D, WT57_M20 | True | PASS |
| kept fine | 1.131 | 18 | WT1a, WT1b, WT1d, WT1f, WT2Ba, WT4Aa, WT53D | True | PASS |
| dropped fine | 1.101 | 18 | WT1a, WT1b, WT1f, WT2Ba, WT4Aa, WT53D, WT57_M20 | True | PASS |
| kept medium | 1.144 | 17 | WT1a, WT1b, WT1d, WT1f, WT2Ba, WT4Aa, WT51a, WT53D | True | PASS |
| dropped medium | 1.103 | 18 | WT1a, WT1b, WT1f, WT2Ba, WT4Aa, WT53D, WT57_M20 | True | PASS |

Tests inside the band at both ends of the bracket (fine mesh): 17 of 25.


### R per test (fine mesh)

| Test | stamped | kept | dropped |
|---|---|---|---|
| WT57_M20 | 0.788 | 0.807 | 0.786 |
| WT7_M16 | 0.846 | 0.865 | 0.845 |
| WT57_M16 | 0.868 | 0.887 | 0.867 |
| WT7_M20 | 0.879 | 0.900 | 0.877 |
| WT7_M12 | 0.888 | 0.906 | 0.887 |
| WT57_M12 | 0.935 | 0.954 | 0.933 |
| WT1h | 1.009 | 1.036 | 1.008 |
| WT2Aa | 1.031 | 1.050 | 1.027 |
| WT2Bb | 1.046 | 1.078 | 1.045 |
| WT2Ab | 1.072 | 1.092 | 1.068 |
| WT1g | 1.083 | 1.112 | 1.082 |
| WT4Ab | 1.099 | 1.125 | 1.098 |
| WT53E | 1.104 | 1.134 | 1.103 |
| WT1e | 1.107 | 1.136 | 1.105 |
| WT53C | 1.118 | 1.147 | 1.116 |
| WT1c | 1.155 | 1.186 | 1.153 |
| WT51b | 1.164 | 1.195 | 1.163 |
| WT51a | 1.208 | 1.240 | 1.206 |
| WT1d | 1.234 | 1.266 | 1.232 |
| WT4Aa | 1.273 | 1.303 | 1.272 |
| WT2Ba | 1.312 | 1.353 | 1.311 |
| WT1b | 1.352 | 1.388 | 1.350 |
| WT53D | 1.361 | 1.397 | 1.359 |
| WT1a | 1.543 | 1.584 | 1.541 |
| WT1f | 1.552 | 1.563 | 1.516 |
