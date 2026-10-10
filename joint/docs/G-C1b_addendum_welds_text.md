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
