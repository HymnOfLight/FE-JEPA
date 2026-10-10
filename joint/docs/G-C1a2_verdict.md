# Gate G-C1a2: verdict

**What was pre-specified**
- `docs/PRESPEC_G-C1a2.md`, configuration SHA-256 `585c7736b1132639a6017158b8ce6e895093c9ba50f926375e4a09aeb56e3dfd`.
- The records give the order:
  - pre-specification and configuration saved at 13:48:53 BST on 10 October 2026 (the stamp line says 13:49);
  - run started at 13:49:00;
  - run ended at 14:36:37.

**Where the numbers come from**
- The criteria, the alternative definitions, the sensitivities and the numerical checks come from `checks/gc1a2_evaluate.py` and `checks/gc1a2_write_verdict.py`. The tables are reproduced at the end.
- The figures for the model without welds against G-C1a are computed from the same records; they are marked where they appear.

**Independent audit** (10 October 2026):
- the hashes, the criteria and the inputs check out;
- a rerun of FS1 on the medium mesh from the stamped code reproduced the result bit for bit;
- no major findings. Its minor findings are taken up below: the flange welds came out one cell short of the specified leg, the numbers script is not under the stamp, and several sentences were tightened.

## Run record

- The stamped script ran all 20 runs in one session, with no interruption. No stamped file changed. The code hashes recorded at the start equal those in the stamp.
- `checks/gc1a2_write_verdict.py`, which only formats the evaluated numbers, is not under the stamp. It was saved at 13:49:22, after the run had started and before any result existed.
  - After the audit, only its docstring was corrected; it had said the script was written after the run.
  - No copy was kept from before the correction. The corrected script (SHA-256 `b1aa35a783c48b44b456a3eb6a54b78fe33a1b7651c6a76b04f5e1d2a1c488fa`) regenerates the tables below byte for byte.
- Order of the runs:
  - the four medium-mesh primary runs;
  - the sensitivities S1 to S3;
  - the four fine-mesh primary runs.
- The fine mesh has about 369,000 unknowns and took 4 to 6 minutes a run, with a peak memory of 4,353 MiB (about 4.6 GB).

## Verdict

**FAIL.**
- **C1 fails for all eight tests:** the model is 2.11 to 3.04 times stiffer than $S_{j,ini}$.
- **C2 holds:** FS2 and FS3 are stiffer than FS1 and FS4, and FS1/FS4 is 1.05.
- The mesh changes $S_{FE}$ by less than 1% from the medium to the fine mesh. No series is flagged.

This is the outcome the pre-specification expected. The model that matched the isolated welded T-stubs within the band on average, and narrowly (G-C1b), is still 2.1 to 3.0 times too stiff for the joints.

**Against G-C1a.** The model is 1.9% to 3.3% stiffer than G-C1a's model. Two changes nearly cancel (medium mesh, computed from the records):
- The holes, the solid heads on their bearing faces and the springs of the bolt below the plate soften the joint by 4.5% to 11% against G-C1a. Those springs use Agerskov's length, and their bending springs are about a third of G-C1a's. This is the model without welds (S1) against G-C1a's medium-mesh values.
- The welds, which G-C1a left out, stiffen it by 7% to 17% (S1 against the primary).
- Without welds, the smallest $R$ is 1.83 (FS1a, medium mesh). So the fail does not depend on the welds; its size does.

**Other definitions of the rotation and the column** (reported):

| Definition | Range of $R$ |
|---|---|
| Pre-specified | 2.11 to 3.04 |
| No beam subtraction on either side | 1.95 to 2.30 |
| The model's chord subtracted on both sides | 2.63 to 3.91 |
| Measured column flexibility in series | 1.86 to 2.46 |

None comes near the band.

**Gap at DT9 per rotation.** The tests show 0.54 to 0.69 of the model's value, as in G-C1a.

**Sensitivities** (medium mesh):

| Code | Change | Effect on $S_{FE}$ |
|---|---|---|
| S1 | No welds | −6% to −15% |
| S2 | Bending springs of the bolt below the plate × 4 | +0.6% to +3.3% |
| S3 | EN 1993-1-8 bolt length | +1.0% to +1.9% |

**Numerical checks** over all runs:
- optimality violation $3.9 \times 10^{-10}$;
- equilibrium residuals $2 \times 10^{-10}$;
- no bearing face dropped;
- the meshed welds are 0.89 to 0.94 of the fillet volume, by the code's own formula.

**Size of the welds** (found by the audit):
- The flange welds lose to round-off the row of cells whose centroids lie on the weld face: 0.875 of their volume on the medium mesh and 0.90 on the fine.
- They reach 7.1 to 7.3 mm up the plate instead of 8.1 mm. The fillet they reach has a throat of 5.0 to 5.2 mm (5.4 to 5.5 mm by volume), against the 5.75 mm intended.
- The web welds are 1.15 of their fillet on the medium mesh and 0.92 on the fine.
- The code's comment that the cell rule overstates the volume is wrong here. Its exact volume also counts the corner between web and flange twice: against the exact union, FS1 on the medium mesh is 0.95, not 0.94.
- Both are left as stamped. With the welds at full size the model would be stiffer on the fine mesh, where all its welds are short, so the fail would not change.
- The T-stub model of G-C1b draws its welds by the same rule. How much that matters there is checked after the fact in `docs/G-C1b_addendum_welds.md`; G-C1b's verdict is not changed.

## What this means

1. **The gap is at the level of the joint.**
   - On the isolated welded T-stubs of the same laboratory, the same modelling is 10% too stiff on average (G-C1b), and 0.79 to 1.55 times the measured stiffness test by test.
   - For the joints it is 1.8 to 3.9 times too stiff. That range spans the definitions of the rotation, the column variant and the no-weld variant.
   - For FS2 to FS4 the published loading curves lie 9% to 30% above Table 8, much nearer to it than the model (reading G-C1c).
   - The cause is still not established. The candidates are those listed in G-C1b's verdict:
     - initial gaps at the interface between end plate and column flange;
     - the bolts (a 50 to 60 mm grip, a second batch from another manufacturer, nut stripping in 4 of the 8 joints);
     - the measurement of the rotation;
     - the parts the T-stub tests do not cover.
2. **The welds are modelled as fused steel.**
   - In G-C1b the model was more sensitive to the weld throat than the tests. This suggests, without showing, that fused-steel welds stiffen the joint more than the welds as built do.
   - The modelled flange welds are, however, smaller than specified.
   - Both variants fail: $R$ is 2.11 to 3.04 with welds (fine mesh) and 1.83 to 2.87 without (medium mesh).

## Consequences (as pre-specified)

- **Reference model.** As pre-specified, the model becomes the reference for stage C1.3 under option A, with the gap to the joint tests stated as unexplained.
  - After the run, an exploratory check found this model too large for the surrogate as it is built: at least 23,690 nodes on any mesh tried. The idealised model of G-C1a comes within about 1% of its own fine mesh with about 4,100 nodes (`docs/C1.3_preparation_note.md`).
  - Which model C1.3 trains and is judged against is therefore left to C1.3's pre-specification, for the project owner to approve. It will state the distance between the two models.
- **The raw records** remain the way to locate the gap. The request to the authors has been drafted for the project owner.
- **Gate names.** The C plan of 9 October 2026 named the training gate of stage C1.3 "G-C1b". That name now belongs to the T-stub gate, so the training gate will be called G-C1.3.

## Numbers

<!-- generated by checks/gc1a2_write_verdict.py; config 585c7736b1132639a6017158b8ce6e895093c9ba50f926375e4a09aeb56e3dfd -->

### Main result (fine mesh, the primary; kNm/mrad)

| Series | S_FE fine | S_FE medium | mesh change | G-C1a S_FE | change from G-C1a | test a | test b | R a | R b |
|---|---|---|---|---|---|---|---|---|---|
| FS1 | 38.42 | 38.66 | -0.6% | 37.26 | +3.1% | 18.19 | 16.84 | 2.11 | 2.28 |
| FS2 | 54.59 | 54.88 | -0.5% | 53.22 | +2.6% | 23.39 | 22.0 | 2.33 | 2.48 |
| FS3 | 65.65 | 65.96 | -0.5% | 64.40 | +1.9% | 23.23 | 21.56 | 2.83 | 3.04 |
| FS4 | 36.59 | 36.84 | -0.7% | 35.42 | +3.3% | 16.18 | 17.15 | 2.26 | 2.13 |

C1 = False; C2 order = True; FS1 / FS4 = 1.050; C2 = True; verdict = **FAIL**. Mesh flags: none.

### Reported: other definitions of the rotation, and the column

| Test | R (pre-specified) | R, no beam subtraction | R, model chord on both sides | R, column flexibility in series |
|---|---|---|---|---|
| FS1a | 2.11 | 1.95 | 2.63 | 1.86 |
| FS1b | 2.28 | 2.05 | 2.80 | 2.00 |
| FS2a | 2.33 | 2.02 | 3.06 | 1.95 |
| FS2b | 2.48 | 2.09 | 3.21 | 2.07 |
| FS3a | 2.83 | 2.21 | 3.69 | 2.29 |
| FS3b | 3.04 | 2.30 | 3.91 | 2.46 |
| FS4a | 2.26 | 2.10 | 2.80 | 2.00 |
| FS4b | 2.13 | 2.02 | 2.67 | 1.89 |

Model chord subtraction (mrad/kNm): FS1 0.0186, FS2 0.0187, FS3 0.0188, FS4 0.0173.

### Reported: end plate gap at DT9 per rotation (mm/mrad)

| Test | Fig. 20 at about 60 kNm | model (fine) | test / model |
|---|---|---|---|
| FS1b | 0.155 | 0.225 | 0.69 |
| FS2a | 0.125 | 0.196 | 0.64 |
| FS3b | 0.097 | 0.180 | 0.54 |
| FS4b | 0.145 | 0.225 | 0.64 |

### Sensitivities (medium mesh, change of S_FE)

| Series | S1 no welds | S2 bending springs x4 | S3 EN 1993-1-8 bolt length |
|---|---|---|---|
| FS1 | -14.0% | +3.1% | +1.1% |
| FS2 | -8.8% | +1.3% | +1.5% |
| FS3 | -6.2% | +0.6% | +1.9% |
| FS4 | -14.8% | +3.3% | +1.0% |

### Numerical checks over all runs

- largest optimality violation: 3.9e-10
- largest equilibrium residual (vertical or horizontal), relative: 2.0e-10
- bearing faces dropped by the coverage check: 0
- meshed weld volume over the exact fillet volume: 0.89 to 0.94
- fine mesh: 368793 to 369723 unknowns, 247 to 332 s per run, peak memory 4353 MB
- bolt forces per row on the half model at P = 10 kN (N), rows 1, 2, 3: FS1 9877, 11079, 454; FS2 8982, 10564, 225; FS3 7797, 10380, 160; FS4 9758, 11143, 518
