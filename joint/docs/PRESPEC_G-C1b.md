# Pre-specification of gate G-C1b

Stamped 10 October 2026, 11:10 BST, before any specimen of Girão Coelho, Bijlaard, Gresnigt and Simões da Silva (2004) was solved.

## Question

Gate G-C1a failed: the elastic contact model of the joint was 2.05 to 2.99 times stiffer than the eight measured $S_{j,ini}$, and the cause was not established. G-C1b asks the same question one level down. Does the elastic contact model reproduce the initial stiffness of the isolated welded T-stubs that the same laboratory tested with the same procedure?

These tests are the right ones for that question because they share with the joints:
- the laboratory (Delft University of Technology) and the authors;
- the definition of the measured stiffness: a linear regression on the unloading branch after a first loading to two thirds of the design resistance (T-stub paper Secs 2.3 and 3.2.1; joint paper Sec. 3.2);
- 10 mm plates of the same steel grades, 45 degree fillet welds and snug-tight bolts of grade 8.8;
- the geometry of the joints' top bolt row: specimen WT7_M20 is that row (joint paper Sec. 2.2).

The answer locates the gap of G-C1a:
- if the model reproduces these tests, the gap lies at the level of the joint, not in the end plate and its bolts;
- if it does not, the gap is already present at the level of the components.

This is a finite element question, not a machine learning one. The tests are used for checking only.

## What was seen before the stamp

- The T-stub paper was read in full on 10 October 2026, including the measured stiffnesses (Table 9) and the Eurocode 3 predictions (Table 8). The measured values are therefore known; the model's values are not.
- The baseline machinery had already been compared with the T-stubs of Bursi and Jaspart (1997): stiffer than those tests by 1.15 to 1.5 (post hoc to G-C1a).
- The higher-fidelity model was developed and verified on two synthetic T-stubs only, VT1 and VT2, which are not specimens of this gate (`probes/tstub_hf_verification.json`, `tests/test_tstub_hf.py`).
- No model was solved on any of the 15 specimen geometries before the stamp. Their meshes were built once, without a solve, to count the unknowns.

## Authority

- On 10 October 2026 at 09:58 BST the project owner instructed work to proceed as recommended in the G-C1a verdict: option B, a higher-fidelity reference model tested under a new pre-specified gate on component tests.
- The band $[0.80, 1.25]$ is the one the owner adopted for G-C1a.
- The owner may still change the thresholds. Any change made after this stamp will be reported as a change after results.

## Configuration

The configuration is held in `checks/gc1b_config.json`.
- Its canonical JSON (sorted keys, no whitespace) has SHA-256 `f021f3000a11af72fc97a7648ae8d72776083c3c4294c4bc81facb10ff86deac`.
- The run script refuses to start unless it is given this hash, and resumes from its results file if interrupted.

SHA-256 of the code at stamping:

| File | SHA-256 |
|---|---|
| `checks/gc1b_run.py` | `90ea492b248a5524cbbcbbdfb861c9108234c89caf99389bbe9b99cb9c2e63e0` |
| `checks/gc1b_evaluate.py` | `9f307d0df57f0eabfc3434298ad4df3fbc80cad775d30810e8607967c6fe3bed` |
| `fejoint/tstub_hf.py` | `ed7f8b8f966a0ae2b7908c20e414799484e93c8ae290d312d1a8ff17c52c1c4f` |
| `fejoint/tstub.py` | `0d09c29d09d8a7b0318a1b40860a068fecd34ff04a74e1563cceeacc00b51afd` |
| `fejoint/delft_tstubs.py` | `9763ba9c022968afc9c5cf102acc4aa318370f77e646f6983d3b4c6a48ec56be` |
| `fejoint/joint.py` | `23b7fb48f9a587384c797b597408e85b8c8e368dad69631af90c16c564a5c044` |
| `fejoint/contact.py` | `75885802990155b4634a8be58035658ead51510f0758ad1ce2d2778b5750db8a` |
| `fejoint/hex8i.py` | `27e4964ef755c1055fd9159e860db2f7542952fac9e1237053ef3cdd1babc0f7` |
| `fejoint/linsolve.py` | `a829f4f5d6becd7c1081db542f3ade923dc95ed7725527a2ad9f948a6eb6739a` |

A copy of these files is kept in `checks/gc1b_stamped_code`.

## Tests

The 25 unstiffened tests with parallel T-stubs in Table 9 of the T-stub paper, in 15 series of distinct inputs:

| Series | Tests | Plate | Bolt | Rows | Weld throat (mm) |
|---|---|---|---|---|---|
| WT1 | WT1a, b, c, d, e, g, h | S355 | M12 8.8 ST | 2 | 5 |
| WT1f | WT1f | S355 | M12 8.8 ST | 2 | 8 (welded so by mistake, Sec. 2.4) |
| WT2A | WT2Aa, b | S355 | M12 8.8 ST | 2 | 3 |
| WT2B | WT2Ba, b | S355 | M12 8.8 ST | 2 | 7 |
| WT4A | WT4Aa, b | S355 | M12 8.8 ST | 2, length 150 mm | 5 |
| WT51 | WT51a, b | S690 | M12 8.8 ST | 2 | 5 |
| WT53C, D, E | one test each | S690 | M12 8.8 FT, 10.9 ST, 10.9 FT | 2 | 5 |
| WT7_M12, M16, M20 | one test each | S355 | M12 ST, M16 FT, M20 FT, all 8.8 | 1 | 5 |
| WT57_M12, M16, M20 | one test each | S690 | M12, M16, M20 FT, all 8.8 | 1 | 5 |

Left out:
- the stiffened specimens (WT61, WT64A, WT64C), because the transverse stiffener is not modelled;
- the specimens with T-stubs at right angles (WT4B, WT64B), whose transducers did not measure a flange gap of the same kind.

The T-stubs of Bursi and Jaspart (1997) are not used. Their stiffness can only be read by eye from loading curves, so it is not defined as the targets here are.

## Models, fixed in advance

All inputs come from the T-stub paper unless marked otherwise; `fejoint/delft_tstubs.py` gives the source of each.

**Geometry**
- Averaged measured $b$, $p$, $e$, $w$, $n$ from Table 1.
- Nominal weld throat; 45 degree fillet welds of leg $a_w\sqrt{2}$, stair-stepped on the mesh.
- Plate thickness: flange and web 10.0 mm, nominal. The paper does not report the measured thickness (assumed).
- Two-row specimens: quarter model, symmetric about the web mid-plane and the mid-length plane. The bolt row sits at $(b - p)/2$ from the end.
- One-row specimens: half model about the web mid-plane, over the whole length. The row sits at $e$ from the end $y = 0$, the end of transducer HP1, "the shorter edge side" (Sec. 3.2.4).
- One T-stub of the pair is modelled. The plane between the flanges is a rigid frictionless support.

**Materials**
- Young's moduli from Table 4 (flange and web, by grade) and Table 2 (M12 bolts by grade and thread type).
- The M16 and M20 bolts were not tested (Sec. 2.2.1); the grade 8.8 fully threaded M12 value is used (assumed).
- Poisson's ratio 0.3 (assumed).

**Bolts** (sizes assumed, since the paper does not state the product standards):
- ISO 4014 and 4017 heads, ISO 4032 nuts;
- holes with normal clearance: 13, 18 and 22 mm for M12, M16 and M20;
- no washers: none is mentioned in the text or drawn in Fig. 2;
- no preload: snug tight, by hand and then a 45 degree turn of an ordinary spanner (Sec. 2.3).

**Higher-fidelity model** (`fejoint/tstub_hf.py`; **this is the model judged**)
- Bolt holes cut through the flange.
- Each half bolt is a solid body:
  - a round shank of tensile stress area $A_s$ through the hole, not touching its wall;
  - a round head of diameter $s$ (across flats) and height $k$.
- The head bears on the flange only on its bearing face, $d_0/2 \le r \le d_w/2$, unilaterally and without friction. No shear springs.
- The half bolt's axial compliance, from a uniform pressure on the bearing face to the symmetry plane, equals $L_{eff}/(2 E_b A_s)$. Here $L_{eff}$ is Agerskov's effective length as quoted by Bursi and Jaspart (1997, eqs 1 and 2), taken for a bolt threaded through the grip. The shank's Young's modulus is calibrated on each mesh to achieve it; this follows Bursi and Jaspart's own finite element model.
- Frictionless contact throughout, which keeps the model inside the energy framework of `docs/contact_energy_note.md`. Leaving friction out can only make the model softer.

**Baseline** (`fejoint/tstub.py`; reported, not judged)
- The G-C1a model's machinery carried over to the T-stub pair:
  - no hole;
  - a rigid head plane on the disc $r \le d_w/2$, free to tilt;
  - axial and bending springs with the EN 1993-1-8 bolt length;
  - shear springs.

**Reading**
- $\Delta = 2 u_x(0, y_{end}, 0)$: the gap between the flanges at the web centre line at the ends of the specimen, where the transducers sit (Sec. 2.3, Fig. 10).
- $\Delta$ is averaged over the two ends, as in Table 9.
- $K = F/\Delta$. The model is linear in the load, because a contact problem with zero initial gaps is positively homogeneous, so the load level is immaterial.

**Meshes**

| Model | Medium | Fine |
|---|---|---|
| Higher fidelity | in-plane spacing 2 mm; 3 cells across the bearing ring; 4 layers in the flange | 1.33 mm; 4.5 cells across the ring; 6 layers |
| Baseline | $h = 2.5$ mm, 4 layers | $h = 1.25$ mm, 6 layers |

**The fine mesh of the higher-fidelity model is the primary result.**

## Criteria

Let $R_i = K(\text{series of test } i)/k_{e.0,i}$ for the 25 tests, with $K$ from the higher-fidelity model on the fine mesh.

**C1 (level).** Both conditions must hold:
- **C1a:** the geometric mean of the 25 $R_i$ lies in $[0.80, 1.25]$;
- **C1b:** at least 17 of the 25 $R_i$ (two thirds) lie in $[0.80, 1.25]$.

The scatter within a series is large: WT1a to WT1h range from 96 to 147 kN/mm for nominally identical specimens. No model could place every test inside the band, so the band applies to the geometric mean, and to two thirds of the tests individually.

**C2 (trends).** Both conditions must hold:
- **C2a:** $K$ increases with the bolt diameter in both one-row series: M12 < M16 < M20 for WT7 and for WT57;
- **C2b:** $K(\text{WT4A}) > K(\text{WT1})$.

**Verdict.** G-C1b passes if and only if C1 and C2 hold.

**Mesh flag.** Any series with $|K_{fine}/K_{medium} - 1| > 0.02$ is flagged. The flag is reported but is not part of the verdict.

## Reported, not part of the verdict

- **C3:** $K(\text{M20})/K(\text{M12})$ of the model against the tests: 1.51 for WT7 and 1.76 for WT57. Eurocode 3 gives about 1.06.
- **Baseline:** the same statistics and the verdict the baseline would receive, on both meshes.
- **Eurocode 3:** Table 8 against Table 9, for context. The tests are stiffer than Eurocode 3 by a factor of about 1.25 to 1.5.
- **Reading position:** $K$ read on the web centre line at $x = t_f$, $t_f + a_w\sqrt{2}$, 20 and 30 mm, in case the transducers were clamped above the flange.
- **Series level:** the geometric mean over the series means as well as over the tests.

**Sensitivities** (higher-fidelity model, medium mesh):

| Code | Change | Series |
|---|---|---|
| S1 | Plate thickness 10.40 mm (S355) and 10.06 mm (S690): the measured 10 mm end plates of the joint tests (joint paper Table 2, FS1 and FS4) | all |
| S2 | EN 1993-1-8 bolt length instead of Agerskov's | WT1, WT4A, WT7_M12, WT7_M20, WT57_M20 |
| S3 | Web modelled 160 mm long instead of 80 mm | same five |

## Interpretation, fixed in advance

**PASS.** The higher-fidelity model reproduces the component tests. It becomes the reference model for the end plate and bolts. The gap of G-C1a then lies at the level of the joint: the column, the beam, the test arrangement or the measurement of the rotation. Next steps:
- carry the same changes into the joint model and run a new pre-specified joint gate against Table 8 of the joint paper, with the criteria of G-C1a;
- in parallel, option C (the low-moment part of the measured curves).

**FAIL with the geometric mean above 1.25.** The model is too stiff already for the components, by an amount to be compared with the 2 to 3 of G-C1a. Candidate causes, written before the run:
- welding distortion, which leaves gaps between the flanges;
- the unloading regression, if it takes in a soft foot near zero load;
- a plate thinner than nominal;
- thread and nut compliance beyond Agerskov's length.

Next steps: options C and A.

**FAIL with the geometric mean below 0.80.** The model is softer than the components. Then the joint gap, where the model is too stiff, cannot come from the end plate and its bolts. That points to the joint's test arrangement or measurement. Candidate causes of the softness, written before the run:
- friction under the heads and between the flanges, which the model leaves out;
- preload from the 45 degree turn;
- plates or welds larger than nominal (S1 tests the plates).

Next steps: options C and A, and the request for the raw records becomes more important.

**FAIL on C1b or C2 only.** The level is right but the scatter or a trend is not reproduced. This is reported as such.

## References

- Bursi, O.S., Jaspart, J.P. (1997). Benchmarks for finite element modelling of bolted steel connections. *Journal of Constructional Steel Research* 43(1-3), 17-42. doi:10.1016/S0143-974X(97)00031-X
- Girão Coelho, A.M., Bijlaard, F.S.K., Gresnigt, N., Simões da Silva, L. (2004). Experimental assessment of the behaviour of bolted T-stub connections made up of welded plates. *Journal of Constructional Steel Research* 60(2), 269-311. doi:10.1016/j.jcsr.2003.08.008
- Girão Coelho, A.M., Bijlaard, F.S.K., Simões da Silva, L. (2004). Experimental assessment of the ductility of extended end plate connections. *Engineering Structures* 26(9), 1185-1206. doi:10.1016/j.engstruct.2000.09.001
