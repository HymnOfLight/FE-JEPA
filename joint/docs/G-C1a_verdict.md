# Gate G-C1a: verdict

Pre-specified in `docs/PRESPEC_G-C1a.md`, stamped 10 October 2026 at 00:05 BST, configuration SHA-256 `5c07343e45110058acf2fba72bcf44169c728391de386698a03af57a5da76b87`. Every number below is generated from the raw records by `checks/gc1a_evaluate.py` and `checks/gc1a_write_verdict.py`; the tables are reproduced at the end. An independent audit (10 October 2026) checked integrity, criteria, inputs and claims; its corrections are incorporated and acknowledged where they apply.

## Run record

- 00:05 to 00:26 BST: the stamped script ran the medium-mesh primary runs and all pre-specified sensitivities (24 runs).
- The machine was restarted during the first fine-mesh run, and that process was lost.
- `checks/gc1a_resume_fine.py` then completed the four fine-mesh primary runs, by 02:22. Before running, it:
  - loaded a copy of the stamped code, checking each file against the archive in `checks/gc1a_stamped_code`, whose hashes match the stamp;
  - checked the configuration hash;
  - appended to the same results file.
- Nothing in the model, the criteria or the evaluation changed.
- The first 24 records are identical to `checks/gc1a_results_before_resume.json`.
- The audit re-ran one medium-mesh case from the stamped code and obtained a bit-identical result.

## Verdict

**FAIL.** C1 fails for all eight tests: the model is 2.05 to 2.99 times stiffer than the measured $S_{j,ini}$. C2 passes. As pre-specified, no training is started on this target.

The failure does not depend on how the beam's own deformation is removed from the measured rotation. The definition of that subtraction turned out to matter, because the paper subtracts more than the Euler-Bernoulli chord used in the pre-specification (finding of the audit, Fig. 11(b)). Every consistent treatment still fails C1:

| Treatment | Range of $R$ |
|---|---|
| Pre-specified | 2.05 to 2.99 |
| No beam subtraction on either side | 1.91 to 2.28 |
| The model's chord subtraction on both sides | 2.55 to 3.84 |

C2 holds only qualitatively. The model stiffens more with plate thickness than the tests do: FS3/FS1 is 1.73 in the model and 1.28 in the tests. So $R$ rises with thickness.

## What is identified, and how much

Each item below changes $S_{FE}$ with everything else fixed.

**Model side**

| Item | Change in $S_{FE}$ |
|---|---|
| Bolt holes cut from the plate | −7% (FS1), −3% (FS3) |
| Bearing radius | −1% to −3% |
| Bolt bending stiffness | −1% to −5% |
| Bolt length | about −1% to −2% |
| Welds at the plate-flange junction, not modelled | would stiffen the model |

**Readings of the paper**

| Item | Change in $S_{FE}$ |
|---|---|
| Distances measured from the beam-side face of the plate instead of the contact face | −4% to −10% |
| Column flexibility measured in Fig. 12(b) (about 0.0036 mrad/kNm), in series | −11% to −19% |
| Vertical displacement of the end plate, read from Fig. 13 | about 2% of the rotation |

The paper calls the last item negligible. The earlier scenario with the bolt shear stiffness divided by ten is not supported by Fig. 13 and is not used.

Taken together, these items bring the smallest $R$ down to about 1.5. Expressed as compliance, the tests are softer than the model by $1/S_{test} - 1/S_{FE} = 0.024$ to $0.034$ mrad/kNm. That excess is roughly the same for all eight tests, whatever the plate thickness. The identified items account for between a quarter (FS3a) and two fifths (FS1a) of it.

## What remains is not explained; two observations bear on it

**1. The reference model is probably stiffer than a higher-fidelity model of the same components** (post hoc).
- The same machinery was applied to the non-preloaded T-stub pairs of Bursi and Jaspart (1997). It is stiffer than their tests:
  - 1.15 to 1.26 times the T1 secant at 100 kN, and up to 1.9 times the tangent;
  - 1.3 to 1.5 times the T2 secant.
- Their own model followed those test curves. It modelled the bolt holes, solid bolts with heads and washers, penalty contact, friction under the heads, and plasticity.
- So our idealisations most likely make the joint model too stiff by an amount of the same order.

**2. On the published loading curve of FS1a, the low-moment stiffness is close to the model's** (post hoc; two independent readings of Fig. 11(b); large uncertainty at small rotations).
- Below about 60 kNm, the secant of the connection-rotation curve reads 30 to 45 kNm/mrad. The model gives 37.3.
- The secant drops to about 17 to 24 kNm/mrad by 70 to 80 kNm, where the curve turns into its knee.
- Table 8's $S_{j,ini}$ of 18.2 is regressed on the unloading branch after a first loading to two thirds of the design resistance, which the paper does not plot.
- The published data alone cannot settle which of the two describes the elastic response: the low-moment loading branch, or Table 8's unloading regression.
- For FS3, a rough reading by the audit still gives a ratio of about 1.7 at 30 kNm.
- The raw test records would decide this. The authors may be able to share them.

Neither observation alone closes the gap, and the two are not simply additive. The verdict therefore records the cause as **not established**. Earlier drafts of this reading made three claims that the audit showed were not supported, and they are withdrawn:
- that the model agrees with the Eurocode 3 component estimate for a rigid column. The paper's $k_{eq}$ contains column terms; without them, the model is softer than Eurocode 3.
- that the isolated T-stubs agree with the model. A wrong gauge base was used for T2.
- that the gap lies in the test arrangement.

## Consequences and options for the project owner

The gate did its job: it stopped training on a target that the reference model does not reproduce. Four options:

**A.** Keep the idealised model as the truth for stage C1.3, and judge the surrogate against the model. State the model-test gap as unexplained wherever results are shown.

**B.** Raise the fidelity of the reference model first, and test it under a new pre-specified gate on component tests already in hand: the Bursi and Jaspart T-stubs. Changes to make:
- bolt holes;
- deformable bolts with heads and washers;
- the bolt length of their model.

The same authors' isolated welded T-stubs (Girão Coelho, Bijlaard, Gresnigt, Simões da Silva, *J. Constr. Steel Res.* 60(2) (2004) 269-311, doi:10.1016/j.jcsr.2003.08.008) would add the exact plates and bolts of FS1, if the paper can be obtained.

**C.** Change the target. Compare the model with the low-moment secant of the measured loading curves under a newly pre-specified gate.
- Ideally use the authors' raw records; any request to them goes through the project owner.
- Otherwise use digitised curves, with the uncertainty stated.

This tests whether the elastic contact model describes the response before the non-linearity sets in.

**D.** Choose another joint test series.

**Recommendation:** B, then C. Both run on a CPU in days. Then A for C1.3.

## Numbers

The tables follow in `docs/G-C1a_verdict_numbers.md`; they are appended below in the assembled verdict.

<!-- generated by checks/gc1a_write_verdict.py; config 5c07343e45110058acf2fba72bcf44169c728391de386698a03af57a5da76b87 -->

### Main result (fine mesh, the primary)

| Series | S_FE fine | S_FE medium | change | test a | test b | R a | R b |
|---|---|---|---|---|---|---|---|
| FS1 | 37.26 | 37.16 | +0.3% | 18.19 | 16.84 | 2.05 | 2.21 |
| FS2 | 53.22 | 53.56 | -0.6% | 23.39 | 22.0 | 2.28 | 2.42 |
| FS3 | 64.40 | 64.82 | -0.7% | 23.23 | 21.56 | 2.77 | 2.99 |
| FS4 | 35.42 | 35.40 | +0.0% | 16.18 | 17.15 | 2.19 | 2.07 |

C1 = False; C2 order = True; FS1 / FS4 = 1.052; C2 = True; verdict = **FAIL**. No mesh flag. Largest optimality violation over all runs: 3.8e-09.

### Pre-specified sensitivities (medium mesh, change of S_FE)

| Series | S1 bearing 18.5 mm | S2 bending EI/L | S3 Agerskov length | S4 bonded | S7 rigid washer | S5: S_a / S_paper |
|---|---|---|---|---|---|---|
| FS1 | +10.0% | -4.9% | -1.3% | x4.1 | +21.4% | 1.27 |
| FS2 | +3.9% | -2.4% | -1.6% | x2.9 | +17.1% | 1.43 |
| FS3 | +2.1% | -1.3% | -1.9% | x2.4 | +11.8% | 1.52 |
| FS4 | +10.3% | -4.9% | -1.2% | x4.1 | +21.0% | 1.27 |

S6 (column rotation, S / (1 + c)): smallest R 1.95 (c = 0.05), 1.86 (c = 0.1), 1.78 (c = 0.15).

### Secondary check: end plate gap per unit rotation at DT9 (mm/mrad)

| Test | read from Fig. 20 at 60 kNm | model (fine) | test / model |
|---|---|---|---|
| FS1b | 0.155 | 0.224 | 0.69 |
| FS2a | 0.125 | 0.193 | 0.65 |
| FS3b | 0.097 | 0.174 | 0.56 |
| FS4b | 0.145 | 0.225 | 0.64 |

### Post hoc: other consistent definitions of the rotation (fine mesh)

Gross: no beam subtraction on either side. Chord: the model's Euler-Bernoulli chord subtraction on both sides (the paper's own subtraction, 0.032 mrad/kNm read from Fig. 11(b), is replaced by the chord value).

| Test | R pre-specified | R gross | R chord |
|---|---|---|---|
| FS1a | 2.05 | 1.91 | 2.55 |
| FS1b | 2.21 | 2.01 | 2.71 |
| FS2a | 2.28 | 1.99 | 2.98 |
| FS2b | 2.42 | 2.06 | 3.12 |
| FS3a | 2.77 | 2.19 | 3.62 |
| FS3b | 2.99 | 2.28 | 3.84 |
| FS4a | 2.19 | 2.06 | 2.71 |
| FS4b | 2.07 | 1.98 | 2.58 |

Model chord subtraction per series (mrad/kNm): FS1 0.0186, FS2 0.0187, FS3 0.0188, FS4 0.0173.

### Post hoc: extra compliance of the tests over the model, 1/S_test - 1/S_FE (mrad/kNm)

| FS1a | FS1b | FS2a | FS2b | FS3a | FS3b | FS4a | FS4b |
|---|---|---|---|---|---|---|---|
| 0.0281 | 0.0325 | 0.0240 | 0.0267 | 0.0275 | 0.0309 | 0.0336 | 0.0301 |

Measured column flexibility (Fig. 12(b), FS1a): about 0.0036 mrad/kNm. In series with the model it lowers S_FE by FS1 -11.7%, FS2 -16.0%, FS3 -18.7%, FS4 -11.2%.

### Post hoc: model omissions and readings (medium mesh unless stated)

| Change | FS1 | FS2 | FS3 | FS4 |
|---|---|---|---|---|
| bolt holes cut from the plate | -7.1% | | -2.6% | |
| bearing radius 12.5 mm | -2.6% | | -1.1% | |
| distances from the plate's beam-side face | -4.0% | -6.9% | -10.1% | -3.7% |
| bolt shear springs / 10 (not supported by Fig. 13) | -16.8% | | -25.9% | |

Beam share of the rotation in the model (end plate held fixed): FS1 20%, FS2 28%, FS3 32%, FS4 20% (medium mesh).

### Post hoc: T-stub pairs of Bursi and Jaspart (1997), non-preloaded (model, fine mesh, kN/mm)

| Specimen | model | test, read from their figure | model / test |
|---|---|---|---|
| T1 (gauge 300 mm) | 139 | secant at 100 kN 111 to 121; tangent 50 to 150 kN 75 to 100 | 1.15 to 1.26 (secant); 1.4 to 1.9 (tangent) |
| T2 (gauge 70 mm) | 330 to 347 | secant at 100 kN 232 to 251; tangent 50 to 120 kN about 186 | 1.32 to 1.50 (secant) |
