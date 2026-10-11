# wp9 tables (generated from records/wp9/w9_verdict.json by scripts/make_w9_paper_material.py; do not edit)

### H1 (2D): does a larger unlabelled pool lower the in-band energy gap at a fixed budget of label-free AR steps?

| Arm | Pool | Epochs | Seeds | Energy gap per seed | Energy gap mean ± SD | Validation split | IB |
|---|---|---|---|---|---|---|---|
| E1's states | 1,024 | 200 | 0-2 | 0.0386 / 0.0394 / 0.0373 | 0.0385 ± 0.0011 | 0.0383 ± 0.0019 | 0.0385 ± 0.0010 |
| AR | 25,600 | 8 | 0-2 | 0.0368 / 0.0370 / 0.0386 | 0.0375 ± 0.0010 | 0.0376 ± 0.0012 | 0.0375 ± 0.0010 |
| AR, fresh seeds | 1,024 | 200 | 3-5 | 0.0371 / 0.0373 / 0.0372 | 0.0372 ± 0.0001 | 0.0372 ± 0.0001 | 0.0372 ± 0.0001 |

- Verdict: NOT SUPPORTED.
- N_max against 1,024: change −2.5%, threshold 10.0% (max(10%, 2 SE_rel)); within the guard.
- N_max against 1,024, beside the verdict (no criterion): per-seed medians −0.7% (threshold 10.0%; within the guard); Welch 95% interval −8.7% to +3.7%; instance-resampling 95% interval −4.1% to −1.1%.
- Relative energy gap in band: per seed, the mean over E1's validation split (256 instances) and the in-band holdout IB (2,048 fresh training-family instances) together, the verdict's quantity; the last two columns are the two sets' seed means ± SD. 204,800 label-free AR steps per seed in every arm; the 1,024 arm evaluates E1's base states; the fresh baseline is E1's configuration trained again on seeds 3-5, a coarse check of that reuse (PREREG_W9 Sec. 2, 7).

### C1 (2D): relative energy gap of every pool size on every evaluation set.

| Pool | Seeds | In band (val + IB) | In band tail p90 | Validation split | IB in band | F1 4-6 holes | F2 slender | F3 hole size | F4 Poisson | F5 fine mesh | R remesh |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1,024 | 0-2 | 0.0385 ± 0.0011 | 0.0908 ± 0.0010 | 0.0383 ± 0.0019 | 0.0385 ± 0.0010 | 0.223 ± 0.174 | 26.9 ± 10.1 | 5.84 ± 4.75 | 0.0406 ± 0.0008 | 0.132 ± 0.014 | 0.0961 ± 0.0621 |
| 1,024 | 3-5 | 0.0372 ± 0.0001 | 0.0897 ± 0.0002 | 0.0372 ± 0.0001 | 0.0372 ± 0.0001 | 9.04 ± 15.46 | 24.1 ± 28.6 | 2.89 ± 4.85 | 0.631 ± 1.025 | 1.28 ± 2.00 | 0.0607 ± 0.0006 |
| 4,096 | 0-2 | 0.0369 ± 0.0001 | 0.0893 ± 0.0003 | 0.0369 ± 0.0001 | 0.0369 ± 0.0001 | 0.109 ± 0.001 | 1.57 ± 1.10 | 0.0887 ± 0.0008 | 0.0385 ± 0.0001 | 0.112 ± 0.012 | 0.0561 ± 0.0063 |
| 25,600 | 0-2 | 0.0375 ± 0.0010 | 0.0901 ± 0.0016 | 0.0376 ± 0.0012 | 0.0375 ± 0.0010 | 0.110 ± 0.003 | 98.1 ± 139.9 | 0.0903 ± 0.0048 | 0.0394 ± 0.0015 | 0.116 ± 0.016 | 0.0560 ± 0.0063 |

- Relative energy gap: seed mean ± sample SD of the per-seed means over each set's instances; the tail is the per-seed 90th percentile of the per-instance energy gap.
- In band: E1's validation split (256) and IB (2,048 fresh training-family instances) together, H1's set; F1-F5: OOD-2D v1 (256 instances each); R: 16 geometries at 5 mesh sizes. Evaluation only; secondary readings (PREREG_W9 Sec. 7).

### C1 (2D): relative displacement error of every pool size on every evaluation set.

| Pool | Seeds | In band (val + IB) | Validation split | IB in band | F1 4-6 holes | F2 slender | F3 hole size | F4 Poisson | F5 fine mesh | R remesh |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,024 | 0-2 | 0.0633 ± 0.0025 | 0.0634 ± 0.0031 | 0.0633 ± 0.0024 | 0.139 ± 0.008 | 0.366 ± 0.136 | 0.178 ± 0.028 | 0.0667 ± 0.0036 | 0.257 ± 0.019 | 0.115 ± 0.010 |
| 1,024 | 3-5 | 0.0613 ± 0.0003 | 0.0615 ± 0.0006 | 0.0613 ± 0.0003 | 0.152 ± 0.022 | 0.374 ± 0.048 | 0.143 ± 0.015 | 0.0657 ± 0.0008 | 0.287 ± 0.006 | 0.119 ± 0.002 |
| 4,096 | 0-2 | 0.0610 ± 0.0003 | 0.0609 ± 0.0001 | 0.0610 ± 0.0004 | 0.140 ± 0.003 | 0.282 ± 0.024 | 0.137 ± 0.002 | 0.0638 ± 0.0003 | 0.228 ± 0.023 | 0.102 ± 0.008 |
| 25,600 | 0-2 | 0.0637 ± 0.0046 | 0.0644 ± 0.0054 | 0.0637 ± 0.0045 | 0.135 ± 0.002 | 0.422 ± 0.141 | 0.140 ± 0.010 | 0.0662 ± 0.0036 | 0.247 ± 0.036 | 0.110 ± 0.017 |

- Relative displacement error: seed mean ± sample SD of the per-seed means over each set's instances.
- In band: E1's validation split (256) and IB (2,048 fresh training-family instances) together, H1's set; F1-F5: OOD-2D v1 (256 instances each); R: 16 geometries at 5 mesh sizes. Evaluation only; secondary readings (PREREG_W9 Sec. 7).

### H2 (2D): a mesh-independent decode scale and load input (S) against the freshly trained baseline.

| Arm | Seeds | In-band energy gap | In-band displacement | F5 displacement | F5 / in-band displacement | Median c* on F5 |
|---|---|---|---|---|---|---|
| AR, fresh seeds | 3-5 | 0.0372 ± 0.0001 | 0.0615 ± 0.0006 | 0.287 ± 0.006 | 4.67 ± 0.09 | 1.28 ± 0.03 |
| AR with S, fresh seeds | 3-5 | 0.0365 ± 0.0002 | 0.0570 ± 0.0004 | 0.0689 ± 0.0011 | 1.21 ± 0.01 | 0.986 ± 0.004 |

- Verdict: SUPPORTED.
- (i) F5 displacement: change −76.0%, threshold 10.0% (max(10%, 2 SE_rel)); lower.
- (i) F5 displacement, beside the verdict (no criterion): per-seed medians −82.8% (threshold 10.0%; lower); Welch 95% interval −81.3% to −70.7%; instance-resampling 95% interval −78.3% to −73.9%.
- (ii) F5 / in-band ratio: change −74.1%, threshold 10.0% (max(10%, 2 SE_rel)); lower.
- (ii) F5 / in-band ratio, beside the verdict (no criterion): per-seed medians −80.3% (threshold 10.0%; lower); Welch 95% interval −78.5% to −69.7%; instance-resampling 95% interval −76.6% to −71.7%.
- (iii) in-band energy gap: change −1.9%, threshold 10.0% (max(10%, 2 SE_rel)); within the guard.
- (iii) in-band energy gap, beside the verdict (no criterion): per-seed medians −2.9% (threshold 10.0%; within the guard); Welch 95% interval −2.8% to −1.0%; instance-resampling 95% interval −2.4% to −1.5%.
- (iii) in-band displacement: change −7.3%, threshold 10.0% (max(10%, 2 SE_rel)); within the guard.
- (iii) in-band displacement, beside the verdict (no criterion): per-seed medians −12.9% (threshold 10.0%; lower); Welch 95% interval −9.3% to −5.4%; instance-resampling 95% interval −9.3% to −5.5%.
- S: decode scale 1/64 of the battery's summed |F| and per-node load densities; the baseline is E1's configuration on the same fresh seeds (PREREG_W9 Sec. 2).

### S against the fresh baseline on every evaluation set (exploratory).

| Set | Baseline energy gap | S energy gap | Change | Baseline displacement | S displacement | Change  |
|---|---|---|---|---|---|---|
| In band (val + IB) | 0.0372 ± 0.0001 | 0.0365 ± 0.0002 | −1.8% | 0.0613 ± 0.0003 | 0.0577 ± 0.0006 | −5.8% |
| Validation split | 0.0372 ± 0.0001 | 0.0365 ± 0.0002 | −1.9% | 0.0615 ± 0.0006 | 0.0570 ± 0.0004 | −7.3% |
| IB in band | 0.0372 ± 0.0001 | 0.0365 ± 0.0002 | −1.8% | 0.0613 ± 0.0003 | 0.0578 ± 0.0006 | −5.7% |
| F1 4-6 holes | 9.04 ± 15.46 | 0.109 ± 0.000 | −98.8% | 0.152 ± 0.022 | 0.132 ± 0.006 | −12.8% |
| F2 slender | 24.1 ± 28.6 | 88.6 ± 79.9 | +267.8% | 0.374 ± 0.048 | 1.28 ± 1.57 | +243.0% |
| F3 hole size | 2.89 ± 4.85 | 0.0876 ± 0.0026 | −97.0% | 0.143 ± 0.015 | 0.118 ± 0.005 | −17.5% (lower) |
| F4 Poisson | 0.631 ± 1.025 | 0.0383 ± 0.0001 | −93.9% | 0.0657 ± 0.0008 | 0.0626 ± 0.0013 | −4.7% |
| F5 fine mesh | 1.28 ± 2.00 | 0.0521 ± 0.0004 | −95.9% | 0.287 ± 0.006 | 0.0689 ± 0.0011 | −76.0% (lower) |
| R remesh | 0.0607 ± 0.0006 | 0.0375 ± 0.0004 | −38.2% (lower) | 0.119 ± 0.002 | 0.0544 ± 0.0003 | −54.2% (lower) |

- Exploratory (PREREG_W9 Sec. 7): each change is S against the fresh baseline with the Sec. 6 guard computed (lower / worse beyond max(10%, 2 SE_rel)); no verdict.

### S against the fresh baseline with the energy-optimal amplitude c* on every evaluation set (exploratory).

| Set | Baseline median c* | S median c* | Energy gap after c*: change | Displacement after c*: change |
|---|---|---|---|---|
| Validation split | 0.999 ± 0.001 | 0.999 ± 0.000 | −1.1% | −1.8% |
| IB in band | 0.998 ± 0.000 | 0.999 ± 0.000 | −1.1% | −1.7% |
| F1 4-6 holes | 1.00 ± 0.00 | 0.998 ± 0.005 | −2.6% | −3.3% |
| F2 slender | 0.735 ± 0.130 | 0.206 ± 0.152 | +33.2% (worse) | +29.1% (worse) |
| F3 hole size | 1.01 ± 0.00 | 1.00 ± 0.00 | −0.2% | −1.6% |
| F4 Poisson | 0.998 ± 0.003 | 1.00 ± 0.00 | −2.2% | −2.0% |
| F5 fine mesh | 1.28 ± 0.03 | 0.986 ± 0.004 | −25.9% (lower) | −16.2% (lower) |
| R remesh | 1.02 ± 0.00 | 0.994 ± 0.001 | −19.8% (lower) | −12.9% (lower) |

- c*: the energy-optimal amplitude of a load case's prediction, from the energy alone (no reference solution); the median over a set's load cases, then seed mean ± sample SD. c* is the solution's energy norm over the prediction's, times the cosine between the two in the energy inner product: where the cosine is near 1, c* above 1 means the prediction's level is too small and below 1 too large, while a small c* can also mean a poor shape. After c*: the relative energy gap and the displacement error once each load case's prediction is scaled by its c*, which removes the level and leaves the shape (the energy gap after c* is one minus the squared cosine).
- Exploratory (PREREG_W9 Sec. 7): each change is S against the fresh baseline with the Sec. 6 guard computed (lower / worse beyond max(10%, 2 SE_rel)); no verdict.

### Remesh set R (2D): the label-free amplitude c_b on fixed geometries and loads meshed at five mesh sizes.

| Arm | Pool | h 0.12 | h 0.085 | h 0.05 | h 0.035 | h 0.025 |
|---|---|---|---|---|---|---|
| E1's states | 1,024 | 1.00 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.07 ± 0.03 | 1.23 ± 0.06 |
| AR, fresh seeds | 1,024 | 1.00 ± 0.01 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.07 ± 0.01 | 1.28 ± 0.06 |
| AR | 4,096 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.04 ± 0.02 | 1.19 ± 0.09 |
| AR | 25,600 | 1.00 ± 0.01 | 1.00 ± 0.01 | 1.00 ± 0.00 | 1.06 ± 0.04 | 1.22 ± 0.07 |
| AR with S, fresh seeds | 1,024 | 0.99 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 0.99 ± 0.00 | 0.99 ± 0.00 |

- c_b: the battery-level energy-optimal amplitude (label-free); median over the 16 geometries, then seed mean ± sample SD. Training mesh sizes 0.05-0.12.
- E1's states (1,024): predicted energy norm relative to the coarsest mesh, 1.00 / 1.01 / 0.999 / 0.940 / 0.801.
- AR, fresh seeds (1,024): predicted energy norm relative to the coarsest mesh, 1.00 / 1.00 / 0.998 / 0.940 / 0.774.
- AR (4,096): predicted energy norm relative to the coarsest mesh, 1.00 / 1.00 / 1.00 / 0.961 / 0.829.
- AR (25,600): predicted energy norm relative to the coarsest mesh, 1.00 / 0.994 / 0.990 / 0.945 / 0.813.
- AR with S, fresh seeds (1,024): predicted energy norm relative to the coarsest mesh, 1.00 / 1.00 / 1.01 / 1.01 / 1.02.
- The exact solution's own energy norm relative to the coarsest mesh: 1.00 / 1.00 / 1.01 / 1.01 / 1.01.
- E1's states (1,024): displacement error at the finest mesh over that at h 0.12 / 0.085 / 0.05: 4.70 ± 0.67 / 4.68 ± 0.52 / 3.52 ± 0.23.
- AR, fresh seeds (1,024): displacement error at the finest mesh over that at h 0.12 / 0.085 / 0.05: 5.59 ± 0.19 / 5.62 ± 0.16 / 4.40 ± 0.21.
- AR (4,096): displacement error at the finest mesh over that at h 0.12 / 0.085 / 0.05: 4.41 ± 0.70 / 4.54 ± 0.62 / 3.58 ± 0.50.
- AR (25,600): displacement error at the finest mesh over that at h 0.12 / 0.085 / 0.05: 4.48 ± 0.91 / 4.42 ± 0.58 / 3.65 ± 0.45.
- AR with S, fresh seeds (1,024): displacement error at the finest mesh over that at h 0.12 / 0.085 / 0.05: 1.35 ± 0.05 / 1.23 ± 0.02 / 1.10 ± 0.01.
- S against the fresh baseline, growth over h 0.12 (exploratory): change −75.8%, threshold 10.0% (max(10%, 2 SE_rel)); lower.
- S against the fresh baseline, growth over h 0.085 (exploratory): change −78.1%, threshold 10.0% (max(10%, 2 SE_rel)); lower.
- S against the fresh baseline, growth over h 0.05 (exploratory): change −75.1%, threshold 10.0% (max(10%, 2 SE_rel)); lower.

Deviations recorded by the adjudication: none. Deviations recorded by hand, which the adjudication cannot see: DEVIATIONS_W9.md.
