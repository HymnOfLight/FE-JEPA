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
