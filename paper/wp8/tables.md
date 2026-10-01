# wp8 E-series tables (generated from records/wp8 by scripts/make_wp8_paper_material.py; do not edit)

### E1 (2D): SIGReg latent shaping of the label-free AR objective. Accuracy of the AR cell and separation readings of the pooled latents.

| Arm | Seeds | Displacement error | Energy gap | S | Probe R² | 1-NN | SIGReg monitor (tokens) |
|---|---|---|---|---|---|---|---|
| AR | 3 | 0.0634 ± 0.0031 | 0.0383 ± 0.0019 | −0.039 ± 0.003 | 0.957 ± 0.010 | 0.40 ± 0.05 | 0.18 ± 0.02 |
| AR + SIGReg (head) | 3 | 0.0604 ± 0.0019 | 0.0365 ± 0.0008 | −0.031 ± 0.004 | 0.976 ± 0.002 | 0.44 ± 0.03 | 0.19 ± 0.01 |
| AR + SIGReg (raw tokens) | 1 | 0.0626 | 0.0384 | −0.049 | 0.939 | 0.38 | 0.0021 |

- K1 (accuracy parity, shaped vs AR): displacement −4.7% (resolution 10%), energy gap −4.6% (resolution 10%): not triggered.
- Effect (S of the shaped arm minus S of the AR arm, per seed): +0.009 / +0.014 / +0.002; the pre-registered floor is 0.02 in every seed: not met. Verdict: NO-GO.
- 256 validation instances; seed mean ± sample SD. S: silhouette of the pooled latents over geometry bins; probe R²: linear probe of the geometry descriptor; 1-NN: leave-one-out bin accuracy.
- The geometry descriptor is also a per-node model input, so S, the probe and 1-NN largely read the input back; they are not evidence of learned geometry (untrained-model reference: post-hoc reading 4a, pending).
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
