# Gate G-C1b: verdict

**What was pre-specified**
- `docs/PRESPEC_G-C1b.md`, configuration SHA-256 `f021f3000a11af72fc97a7648ae8d72776083c3c4294c4bc81facb10ff86deac`.
- The stamp line says 11:10 BST, rounded. The records give the order:
  - pre-specification and configuration saved at 11:09:11;
  - run started at 11:09:18;
  - run ended at 12:30:57.

**Where the numbers come from**
- The criteria, the per-series and per-test values, the sensitivities and the numerical checks come from `checks/gc1b_evaluate.py` and `checks/gc1b_write_verdict.py`. The tables are reproduced at the end.
- The numbers marked post hoc come from the audit's probes in `checks/audit_gc1b/`.

**Independent audit** (10 October 2026):
- found the hashes, the criteria and every input checked against the paper in order;
- reproduced the stamped runs to round-off;
- found the pass marginal, and three of its interpretive claims overstated. Its corrections are made below.

## Run record

- The stamped script ran all 85 runs in one session, without interruption. The code hashes it recorded at the start equal those in the stamp.
- Order of the runs:
  - the medium meshes of both models;
  - the fine mesh of the baseline;
  - the fine mesh of the higher-fidelity model (the primary result);
  - the sensitivities S1 to S3.
- **One change after the stamp.** The evaluation script stopped in a block that is reported but not judged: the reading taken higher up the web. Its readings were keyed by height in millimetres, and one of the heights, $t_f + a_w\sqrt{2}$, differs between series. The readings are now matched by position.
  - The diff is in `checks/gc1b_evaluate_post_stamp.diff`.
  - The code of the criteria is unchanged.
  - The stamped copy is kept in `checks/gc1b_stamped_code`.

## Verdict

**PASS, narrowly, as pre-specified.**
- **C1a:** the geometric mean of $R$ over the 25 tests is 1.10, inside $[0.80, 1.25]$.
- **C1b:** 18 of the 25 tests lie inside the band, against 17 required. It passes by one test.
- **C2:** $K$ increases with the bolt diameter in WT7 and in WT57, and WT4A is stiffer than WT1.

Within the band, the model matches the 25 tests on average. Two qualifications follow, both found by the audit.

**Two-row and one-row specimens differ.**

| Specimens | Tests | Geometric mean of $R$ |
|---|---|---|
| two-row | 19 | 1.19 |
| one-row | 6 | 0.87 |

**1. The one-row agreement depends on the boundary condition at the loaded end** (post hoc).
- The pre-specified model applies a uniform traction to a free web end, which may tilt.
- The tests clamped the webs in the machine (T-stub paper Sec. 2.3).
- For the off-centre single rows, this matters:
  - In the model, the far end opens 3.7 to 4.4 times as much as the near end.
  - In the tests, the two ends deform alike: HP2/HP1 at the maximum load is 1.06 to 1.14 (Table 9).
- The audit reran the medium mesh with a rigid grip, in which all nodes of the loaded end share one displacement:

| Specimen | Change in $K$ |
|---|---|
| WT7_M12 | +12.9% |
| WT7_M20 | +10.7% |
| WT57_M20 | +10.2% |
| WT1 | +0.06% |

- With the grip, the far-to-near ratio falls to about 1.1, as in the tests, and the one-row geometric mean rises to about 0.97.
- Which end the one-row $k_{e.0}$ was read at is ambiguous. Table 9 gives no two-end average for the one-row tests, and Fig. 22 plots HP1 alone.
  - Under the pre-specified free end, a reading at HP1 alone would fail the gate: geometric mean 1.38, 13 of 25 inside.
  - Under a grip the two ends nearly agree, so the question no longer matters.

**2. The thread length of the short-threaded bolts was assumed** (post hoc).
- Every bolt was modelled as threaded through the 20 mm grip. The short-threaded bolts (18 of the 25 tests) may have plain shank there; their lengths are not reported.
- A probe on WT1 gives:
  - 10 mm of plain shank: +2.1%;
  - 20 mm of plain shank: +4.5%.
- Applied to the fine results, 17 and 16 tests lie inside the band, so with 20 mm of plain shank C1b fails.
- Across twelve plausible combinations, C1b ranges from 14 to 19 and four of the combinations fail; the geometric mean stays between 1.10 and 1.24. The combinations are:
  - nominal or S1 plate thickness;
  - 0, 10 or 20 mm of plain shank;
  - a free or a gripped end.

**The seven tests outside the band**

| Test | $R$ |
|---|---|
| WT1a | 1.54 |
| WT1b | 1.35 |
| WT1f | 1.55 |
| WT2Ba | 1.31 |
| WT4Aa | 1.27 |
| WT53D | 1.36 |
| WT57_M20 | 0.79 |

- WT1a, WT1b, WT2Ba and WT4Aa belong to series whose other tests lie inside the band. The scatter within a series is of the same size: WT1a to WT1h range from 96 to 147 kN/mm for one geometry.
- WT53D has the geometry of WT53C and WT53E, which lie inside.
- WT1f was welded with an 8 mm throat, which the model includes. The model makes WT1f 24% stiffer than WT1, but the test is no stiffer than the WT1 tests. More generally, the model is more sensitive to the weld throat than the tests are:

| Throat (mm) | 3 | 5 | 7 | 8 |
|---|---|---|---|---|
| Model (kN/mm) | 133 | 149 | 167 | 184 |
| Tests, geometric mean (kN/mm) | 126 | 124 | 142 | 119 |

  The paper too reports "little variation of stiffness" with the throat (Sec. 3.2.2).

**Bolt diameter (reported, C3).** The ratio $K(\text{M20})/K(\text{M12})$ is:

| Series | Model | Tests | Eurocode 3 |
|---|---|---|---|
| WT7 | 1.49 | 1.51 | 1.06 |
| WT57 | 1.48 | 1.76 | 1.06 |

- The Eurocode 3 stiffness of the flange in bending ($k_5 = 0.9\, l_{eff} t^3/m^3$) does not depend on the bolt size. Its 1.06 comes from the bolt term $k_{10}$ alone.
- The model also has a bearing face that grows with the bolt. The run did not separate the two mechanisms.

**Mesh**
- From the medium to the fine mesh, $K$ changes by −2.4% to +2.2%.
- Nine series exceed the 2% flag, by at most 0.4 percentage points. The flag is reported, not judged.

**Baseline** (the G-C1a machinery)
- It is 8% to 18% stiffer than the higher-fidelity model, series by series.
- It would fail C1b: 15 of 25 inside, geometric mean 1.24.

**Eurocode 3** is softer than the tests: geometric mean 0.84.

**Sensitivities**

| Code | Change | Effect on $K$ (medium mesh) |
|---|---|---|
| S1 | Plate thickness 10.40 mm (S355) or 10.06 mm (S690) | +7% to +9% (S355), +1% (S690). Applied to the fine result: geometric mean 1.17, 17 of 25 inside |
| S2 | EN 1993-1-8 bolt length | 0.3% or less. This length differs from Agerskov's by only 0.5% to 1.8%, so it does not probe the plain shank above |
| S3 | Web modelled 160 mm long | under 0.1% |
| — | Gap read higher up the web | changes the geometric mean by at most 2% |

**Numerical checks** over all runs:
- largest optimality violation $1.5 \times 10^{-10}$;
- equilibrium residual $3 \times 10^{-12}$;
- bolt calibration error $2 \times 10^{-13}$;
- no bearing face dropped.

**Notes on the sources**
- In Table 9, WT2Aa's $k_{e.0}$ of 128.63 equals WT1c's to the last digit, and its $k_{pl.0}$ equals WT2Ba's. This could be a transcription duplication in the paper. WT2Aa lies inside the band unless its true value is below 106 or above 166 kN/mm.
- Table 8 has no Eurocode 3 value for WT1f; the context column reuses WT1's.
- The pre-specification says the tests share 10 mm plates with the joints. Only FS1 and FS4 have 10 mm plates; FS2 and FS3 have 15 and 20 mm plates.

## What the pass means, read with G-C1a and the reading G-C1c

**1. What this covers.**
- Unstiffened pairs of 10 mm welded T-stubs are reproduced within the band.
- The model is too stiff on the two-row tests, by 19% on average. That is the same sign as the joint gap, much smaller.
- Not covered:
  - the stiffened inner row of the end plate (WT61 and WT64 were excluded);
  - the 15 and 20 mm plates of FS2 and FS3, where G-C1a's $R$ is highest (2.28 to 2.99);
  - a plate bearing on a 40 mm column flange rather than on a mirror plate;
  - the joints' bolt grip of 50 to 60 mm, against 20 mm here.

**2. The joint gap is largely not a matter of the definition of $S_{j,ini}$** (reading G-C1c).
- At 40 and 50 kNm the published loading curves of FS2, FS3 and FS4 give secants of about 19 to 31 kNm/mrad.
- That is much nearer Table 8 than the model's 35 to 64, although 9% to 30% above Table 8.
- So for FS2 to FS4 the unloading regression explains at most a minor part of the gap.
- FS1 is different:
  - its stiffer curve, which the G-C1a audit identified as FS1a, lies near the model at 40 kNm (secant 35 to 39) and softens to about 30 at 50 kNm;
  - its Table 8 value is about half its low-moment loading secant;
  - for FS1a the definition may explain most of the gap.

**3. Where the extra flexibility shows** (post hoc, partial).
- The gap between end plate and column flange at DT9, per unit moment, is 1.3 to 1.7 times the model's. This combines Fig. 20's gap-to-rotation ratio, read at about 60 kNm on the loading curve, with Table 8's unloading value of $S_{j,ini}$. With the Fig. 14 secants at 60 kNm instead, it is about 1.2 to 1.6.
- The rotation per unit moment, taken from Table 8, is 2 to 3 times the model's.
- The paper's own check (joint paper Sec. 3.3.1, Fig. 21, FS1a) computes the rotation from the horizontal transducers in the tension and compression zones of the end plate. It agrees closely with the rotation from the beam. Two caveats:
  - the figure's scale, 100 mrad, cannot resolve the first few mrad;
  - the paper takes the column rotation as negligible against the beam rotation (Fig. 12, a ratio of about 0.05 to 0.12 for FS1a in the elastic range); G-C1a put its flexibility in series as a correction of 11% to 19%.
- Both point to the interface between end plate and column flange: the tension side opening more than the model's, and the compression side closing, which the model represents as rigid flat contact.

**4. The cause is not established.** Candidates, none tested:
- **Initial gaps at the interface.** Welding distortion could leave the end plate out of flat against the column flange, and the gaps then close under load. The T-stubs are welded too (T-stub paper Sec. 2.4). The joints differ in having the plate welded all round and bearing on a rigid flat counterpart.
- **The bolts.** The grip is 50 to 60 mm, against 20 mm in the T-stubs. A second batch "from another manufacturer" was used. Nuts stripped in 4 of the 8 joints, against 1 of the 32 T-stubs. The bolt elongation records would test this.
- **The measurement of the rotation.** In G-C1a, the treatment of the beam's own deformation alone moves $R$ from 1.91 to 3.84.
- **The parts the T-stub tests do not cover.** These are listed in item 1.

The raw records would separate these candidates: the transducers on the compression side, the bolt elongations, the first loading and the unloading branch.

## Consequences

**As pre-specified for a pass:**
- The higher-fidelity model becomes the reference model for the end plate and its bolts.
- Next, its changes are carried into the joint model, and a new pre-specified joint gate is run against Table 8 with the criteria of G-C1a.
- The audit's finding should be carried over: load and supports are to be modelled as the test held them.
- At the level of the T-stub, the changes soften the model by 7% to 15%. This is far less than the factor of 2 to 3 of G-C1a, so that gate is expected to fail. It is still the reference model that stage C1.3 would train against.

**Option A, for stage C1.3**
- Train the surrogate against the higher-fidelity elastic contact model.
- State the gap at the level of the joint, as far as it is understood.

**Raw records.** A request to the authors has been drafted for the project owner to send.

## Numbers


<!-- generated by checks/gc1b_write_verdict.py; config f021f3000a11af72fc97a7648ae8d72776083c3c4294c4bc81facb10ff86deac -->

### Main result: the 25 tests (kN/mm)

Higher-fidelity model on the fine mesh (the primary), the baseline (G-C1a machinery) on the fine mesh, and Eurocode 3 (Table 8) for context. R = model / measured.

| Test | measured k_e.0 | model | R | baseline | R baseline | EC3 / measured |
|---|---|---|---|---|---|---|
| WT1a | 96.28 | 148.6 | 1.54 (out) | 165.0 | 1.71 | 1.14 |
| WT1b | 109.88 | 148.6 | 1.35 (out) | 165.0 | 1.50 | 1.00 |
| WT1c | 128.63 | 148.6 | 1.15 | 165.0 | 1.28 | 0.85 |
| WT1d | 120.42 | 148.6 | 1.23 | 165.0 | 1.37 | 0.91 |
| WT1e | 134.25 | 148.6 | 1.11 | 165.0 | 1.23 | 0.82 |
| WT1f | 118.46 | 183.9 | 1.55 (out) | 198.1 | 1.67 | 0.92 |
| WT1g | 137.16 | 148.6 | 1.08 | 165.0 | 1.20 | 0.80 |
| WT1h | 147.17 | 148.6 | 1.01 | 165.0 | 1.12 | 0.74 |
| WT2Aa | 128.63 | 132.6 | 1.03 | 149.3 | 1.16 | 0.69 |
| WT2Ab | 123.65 | 132.6 | 1.07 | 149.3 | 1.21 | 0.72 |
| WT2Ba | 127.15 | 166.8 | 1.31 (out) | 184.5 | 1.45 | 1.02 |
| WT2Bb | 159.49 | 166.8 | 1.05 | 184.5 | 1.16 | 0.81 |
| WT4Aa | 150.15 | 191.2 | 1.27 (out) | 211.7 | 1.41 | 1.21 |
| WT4Ab | 173.91 | 191.2 | 1.10 | 211.7 | 1.22 | 1.05 |
| WT51a | 119.24 | 144.0 | 1.21 | 160.1 | 1.34 | 0.78 |
| WT51b | 123.67 | 144.0 | 1.16 | 160.1 | 1.29 | 0.75 |
| WT53C | 128.46 | 143.6 | 1.12 | 159.5 | 1.24 | 0.75 |
| WT53D | 105.79 | 144.0 | 1.36 (out) | 160.0 | 1.51 | 0.93 |
| WT53E | 129.63 | 143.2 | 1.10 | 159.1 | 1.23 | 0.74 |
| WT57_M12 | 85.78 | 80.2 | 0.93 | 89.9 | 1.05 | 0.92 |
| WT57_M16 | 110.43 | 95.9 | 0.87 | 111.9 | 1.01 | 0.75 |
| WT57_M20 | 150.96 | 118.9 | 0.79 (out) | 140.4 | 0.93 | 0.56 |
| WT7_M12 | 91.18 | 81.0 | 0.89 | 90.9 | 1.00 | 0.95 |
| WT7_M16 | 116.09 | 98.2 | 0.85 | 114.7 | 0.99 | 0.77 |
| WT7_M20 | 137.70 | 121.0 | 0.88 | 143.1 | 1.04 | 0.67 |

**C1a:** geometric mean of R = 1.104 (band 0.80 to 1.25): True. **C1b:** 18 of 25 inside the band (at least 17 needed): True. **C2a** (bolt diameter order in WT7 and WT57): True. **C2b** (WT4A stiffer than WT1): True. **Verdict: PASS.**

R ranges from 0.79 to 1.55. Outside the band: WT1a, WT1b, WT1f, WT2Ba, WT4Aa, WT53D, WT57_M20. Geometric mean over the series means instead of the tests: 1.055.

The baseline would receive: geometric mean 1.237, 15 of 25 inside, verdict FAIL. Eurocode 3 over the tests: geometric mean 0.836.

### Per series (kN/mm)

| Series | model medium | model fine | mesh change | baseline medium | baseline fine | model / baseline (fine) |
|---|---|---|---|---|---|---|
| WT1 | 151.7 | 148.6 | -2.1% | 162.4 | 165.0 | 0.900 |
| WT1f | 180.0 | 183.9 | +2.2% | 194.9 | 198.1 | 0.928 |
| WT2A | 135.9 | 132.6 | -2.4% | 151.4 | 149.3 | 0.888 |
| WT2B | 166.4 | 166.8 | +0.2% | 181.2 | 184.5 | 0.904 |
| WT4A | 195.0 | 191.2 | -1.9% | 208.6 | 211.7 | 0.903 |
| WT51 | 147.1 | 144.0 | -2.1% | 157.5 | 160.1 | 0.900 |
| WT53C | 146.6 | 143.6 | -2.1% | 157.0 | 159.5 | 0.900 |
| WT53D | 147.0 | 144.0 | -2.1% | 157.4 | 160.0 | 0.900 |
| WT53E | 146.2 | 143.2 | -2.1% | 156.5 | 159.1 | 0.900 |
| WT57_M12 | 81.6 | 80.2 | -1.7% | 88.7 | 89.9 | 0.891 |
| WT57_M16 | 97.6 | 95.9 | -1.7% | 110.0 | 111.9 | 0.857 |
| WT57_M20 | 121.5 | 118.9 | -2.1% | 137.8 | 140.4 | 0.847 |
| WT7_M12 | 82.4 | 81.0 | -1.7% | 89.5 | 90.9 | 0.891 |
| WT7_M16 | 100.0 | 98.2 | -1.7% | 112.3 | 114.7 | 0.857 |
| WT7_M20 | 123.6 | 121.0 | -2.1% | 140.5 | 143.1 | 0.846 |

Mesh flags (change above 2%): WT1, WT1f, WT2A, WT51, WT53C, WT53D, WT53E, WT57_M20, WT7_M20.

### Reported: effect of the bolt diameter, K(M20) / K(M12)

| Series | model | tests | Eurocode 3 |
|---|---|---|---|
| WT7 | 1.49 | 1.51 | 1.06 |
| WT57 | 1.48 | 1.76 | 1.06 |

### Reported: the reading taken higher up the web centre line (geometric mean of R)

| x (mm from the plane between the flanges) | 0 (primary) | x = t_f | x = t_f + leg | x = 20 mm | x = 30 mm |
|---|---|---|---|---|---|
| geometric mean of R | 1.104 | 1.109 | 1.122 | 1.123 | 1.114 |

### Sensitivities (higher-fidelity model, medium mesh, change of K)

| Series | S1 plate 10.40 / 10.06 mm | S2 EN 1993-1-8 bolt length | S3 web 160 mm |
|---|---|---|---|
| WT1 | +8.2% | -0.3% | +0.0% |
| WT1f | +8.2% |  |  |
| WT2A | +8.6% |  |  |
| WT2B | +8.0% |  |  |
| WT4A | +8.0% | -0.3% | +0.0% |
| WT51 | +1.1% |  |  |
| WT53C | +1.1% |  |  |
| WT53D | +1.1% |  |  |
| WT53E | +1.1% |  |  |
| WT57_M12 | +1.1% |  |  |
| WT57_M16 | +1.0% |  |  |
| WT57_M20 | +1.0% | -0.1% | +0.0% |
| WT7_M12 | +7.9% | -0.3% | +0.0% |
| WT7_M16 | +7.7% |  |  |
| WT7_M20 | +7.4% | -0.1% | +0.0% |

S1 applied to the fine result: geometric mean of R 1.168, 17 of 25 inside, would-be verdict PASS.

### Numerical checks over all runs

- largest optimality violation: 1.5e-10
- largest equilibrium residual (applied + prying - bolt), relative: 3.2e-12
- largest bolt calibration error, relative: 1.8e-13
- bearing faces dropped by the coverage check: 0
- fine mesh: 192321 to 267977 unknowns, 109 to 225 s per run; shank modulus over bolt modulus 1.14 to 1.29
- prying force over applied force (fine): 0.40 to 0.67
