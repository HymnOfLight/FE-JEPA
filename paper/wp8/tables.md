# wp8 E-series tables (generated from records/wp8 by scripts/make_wp8_paper_material.py; do not edit)

### E1 (2D): SIGReg latent shaping of the label-free AR objective. Accuracy of the AR cell and separation readings of the pooled latents.

| Arm | Seeds | Displacement error | Energy gap | S | Probe R² | 1-NN | SIGReg monitor (tokens) |
|---|---|---|---|---|---|---|---|
| AR | 3 | 0.0634 ± 0.0031 | 0.0383 ± 0.0019 | −0.039 ± 0.003 | 0.957 ± 0.010 | 0.40 ± 0.05 | 0.18 ± 0.02 |
| AR + SIGReg (head) | 3 | 0.0604 ± 0.0019 | 0.0365 ± 0.0008 | −0.031 ± 0.004 | 0.976 ± 0.002 | 0.44 ± 0.03 | 0.19 ± 0.01 |
| AR + SIGReg (raw tokens) | 1 | 0.0626 | 0.0384 | −0.049 | 0.939 | 0.38 | 0.0021 |
| Untrained, descriptor input | 3 | – | – | 0.336 ± 0.024 | 1.000 ± 0.000 | 0.97 ± 0.00 | 0.31 ± 0.02 |
| Untrained, descriptor weights zeroed | 3 | – | – | −0.031 ± 0.002 | 0.198 ± 0.010 | 0.28 ± 0.03 | 0.32 ± 0.01 |
| Untrained, no descriptor input | 3 | – | – | −0.032 ± 0.001 | 0.210 ± 0.004 | 0.26 ± 0.02 | 0.31 ± 0.00 |

- K1 (accuracy parity, shaped vs AR): displacement −4.7% (resolution 10%), energy gap −4.6% (resolution 10%): not triggered.
- Effect (S of the shaped arm minus S of the AR arm, per seed): +0.009 / +0.014 / +0.002; the pre-registered floor is 0.02 in every seed: not met. Verdict: NO-GO.
- 256 validation instances; seed mean ± sample SD. S: silhouette of the pooled latents over geometry bins; probe R²: linear probe of the geometry descriptor; 1-NN: leave-one-out bin accuracy, chance 0.25 with 4 bins of 64.
- The geometry descriptor is also a per-node model input, so S, the probe and 1-NN largely read the input back; they are not evidence of learned geometry. Untrained rows (post-hoc reading 4a, same instances): models built from the AR arm's configuration at initialisation with the descriptor input; the same models with the descriptor's input weights set to zero; models built without the descriptor input.
- The raw-token ablation ran one seed and is reported only.

### E2 (3D): token bottleneck against the point-token transformer under the same label-free AR objective, corpus, split, seeds and schedule.

| Architecture | Tokens | In-band displacement | In-band energy gap | Fine displacement (zero-shot) | Step (ms), 12,318 nodes | Step (ms), 41,367 nodes |
|---|---|---|---|---|---|---|
| Point-token transformer | one per node | 0.0300 ± 0.0028 | 0.00912 ± 0.00228 | 0.258 ± 0.045 | 1,415 | 14,795 |
| Token bottleneck | 512 | 0.106 ± 0.051 | 0.0348 ± 0.0226 | 0.463 ± 0.082 | 27.8 | 49.2 |
| Token bottleneck | 1,024 | 0.0892 ± 0.0096 | 0.0258 ± 0.0051 | 0.574 ± 0.059 | 36.9 | 47.5 |

- M = 512: K1 in-band energy gap +280.9% (resolution 287%), fine displacement +79.1% (resolution 42%, fires); K2 (speed) fine step 0.049 s against the 2.0 s line: not triggered. Verdict: KILLED.
- M = 1,024: K1 in-band energy gap +183.3% (resolution 70%, fires), fine displacement +122.0% (resolution 33%, fires); K2 (speed) fine step 0.047 s against the 2.0 s line: not triggered. Verdict: KILLED.
- Seeds per architecture: 3; seed mean ± sample SD. In-band: the 256 Phase-2b validation instances (lc 0.0579–0.0906; the bench's instances at these two ends have 12,318 and 3,774 nodes); fine: 256 instances at lc 0.0374 (41,367 nodes in the bench), never trained on. Errors are relative L2 displacement and relative energy gap.
- Step time: one label-free AR training step on one instance of the stated size, NVIDIA GeForce RTX 5090, TF32 on, from the E2 bench (bottleneck: set-up-free, median of 3 differential pairs of 10 and 110 steps; transformer: one timed call per phase, its set-up included, and the mean of its two bench measurements, 1,408 / 1,422 ms and 14,793 / 14,798 ms).
- Resolution: the pre-registered max(10%, 2 × SE_rel) of the difference of seed means.

### Post-hoc (3D): errors of the label-free AR cells before and after scaling each prediction by its energy-optimal amplitude c*, in-band and on the fine mesh (zero-shot).

| Architecture | Tokens | In-band displacement | In-band with c* | Fine displacement (zero-shot) | Fine with c* | Fine energy gap (median) | Fine with c* | Median c* (fine) |
|---|---|---|---|---|---|---|---|---|
| Point-token transformer | one per node | 0.0300 ± 0.0028 | 0.0142 ± 0.0032 | 0.258 ± 0.045 | 0.0832 ± 0.0158 | 0.0911 ± 0.0177 | 0.0418 ± 0.0031 | 1.23 ± 0.08 |
| Token bottleneck | 512 | 0.106 ± 0.051 | 0.0393 ± 0.0243 | 0.463 ± 0.082 | 0.124 ± 0.039 | 0.258 ± 0.075 | 0.0904 ± 0.0405 | 1.73 ± 0.42 |
| Token bottleneck | 1,024 | 0.0892 ± 0.0096 | 0.0292 ± 0.0049 | 0.574 ± 0.059 | 0.126 ± 0.037 | 0.263 ± 0.037 | 0.0717 ± 0.0326 | 1.77 ± 0.28 |

- c* = F^T u / (u^T K u), per load case: the energy-optimal amplitude of the prediction u on the evaluated mesh, from that mesh's stiffness K and load F (no labels, no solve). Scaling by c* never increases the energy error (it did not in any of the 4,608 instance-seed rows measured). In-band median c*: 1.00 (transformer), 1.00 (M = 512), 1.00 (M = 1,024).
- Seeds per architecture: 3; seed mean ± sample SD of the per-seed mean over instances (energy gap: per-seed median). In-band: the 256 Phase-2b validation instances; fine: 256 instances at lc 0.0374, never trained on. The columns without c* reproduce the runs' own per-instance arrays: the transformer's exactly, the bottleneck's to a relative 0.011 at most; the bottleneck run's own two evaluations of the same states (E8, P3; in-band) differ by up to 0.003, the transformer run's not at all.
- Mean fine energy gap over instances (a heavy tail), as trained → with c*: transformer: 0.306 → 0.0670; M = 512: 0.295 → 0.106; M = 1,024: 0.590 → 0.0984.
- Post-hoc reading 4e (E-series runbook, Sec. 4): reported only; no verdict is revisited.

### Post-hoc (3D): the label-free amplitude c_b on fixed geometries and loads meshed at four mesh sizes lc.

| Architecture | Tokens | lc 0.0906 | lc 0.0742 | lc 0.0579 | lc 0.0374 (fine) |
|---|---|---|---|---|---|
| Point-token transformer | one per node | 0.98 ± 0.00 | 1.00 ± 0.01 | 1.01 ± 0.00 | 1.27 ± 0.08 |
| Token bottleneck | 512 | 0.91 ± 0.04 | 1.00 ± 0.02 | 1.14 ± 0.09 | 1.96 ± 0.46 |
| Token bottleneck | 1,024 | 0.94 ± 0.01 | 1.01 ± 0.01 | 1.14 ± 0.02 | 1.99 ± 0.35 |

- 16 fresh geometries, each meshed at the 4 lc with identical geometry and loads, so only the mesh changes; every mesh evaluated by every seed. c_b: the battery-level energy-optimal amplitude (label-free); median over the geometries, then seed mean ± sample SD. Training range: lc 0.0579–0.0906.
- fscale, the battery's largest nodal load (both architectures multiply their output by it), relative to lc 0.0906, at lc 0.0742 / 0.0579 / 0.0374: 0.705 / 0.430 / 0.186; for comparison (lc / 0.0906)²: 0.671 / 0.408 / 0.170.
- Predicted energy norm relative to lc 0.0906 (label-free), seed means, lc 0.0742 / 0.0579 / 0.0374: transformer: 0.99 / 0.98 / 0.75; M = 512: 0.90 / 0.82 / 0.50; M = 1,024: 0.93 / 0.83 / 0.49.
- Post-hoc reading 4e (E-series runbook, Sec. 4): reported only.
