# CMAME manuscript material (generated from the records by scripts/make_cmame_material.py; do not edit)

### Two-dimensional plates, every metric at the largest label budget. The supervised networks were trained on 1,024 labelled instances; the label-free network on the same 1,024 instances without their solutions.

| Model | Displacement error | Relative energy gap | von Mises error | Peak stress error | Critical-region recall | Worse than zero field |
|---|---|---|---|---|---|---|
| Label-free | 0.0634 ± 0.0031 | 0.0383 ± 0.0019 | 0.123 ± 0.002 | 0.128 ± 0.001 | 0.753 ± 0.001 | 0/768 |
| Supervised | 0.0505 ± 0.0023 | 0.133 ± 0.014 | 0.235 ± 0.015 | 0.141 ± 0.007 | 0.668 ± 0.015 | 0/768 |
| Supervised, stiffness norm | 0.0547 ± 0.0094 | 0.0290 ± 0.0125 | 0.106 ± 0.020 | 0.0947 ± 0.0356 | 0.804 ± 0.055 | 0/768 |
| Graph network, supervised | 0.0499 ± 0.0003 | 0.825 ± 0.020 | 0.578 ± 0.009 | 0.321 ± 0.011 | 0.517 ± 0.006 | 187/768 |
| Nearest-neighbour field | 0.404 | 0.959 | 0.735 | 0.643 | 0.515 | 94/256 |
| Scaled polynomial | 1.42 | 873 | 11.4 | 30.8 | 0.159 | 256/256 |
| Zero field | 1.00 | 1.00 | 1.00 | 1.00 | – | – |

- 256 validation instances, never trained on; four load cases each. Errors are relative to the reference finite-element solution, per load case, then averaged over the load cases; the relative energy gap of the zero field is 1. Trained networks: mean ± sample standard deviation over 3 seeds; the nearest-neighbour field and the scaled polynomial are deterministic.
- Worse than zero field: instance-seed pairs whose relative energy gap exceeds 1, that is, predictions further from the solution in the energy norm than the zero field.
- Critical-region recall: the share of the 10% most stressed elements of the reference solution that are also among the 10% most stressed of the prediction (higher is better; in every other column lower is better).
- Run of 9 October 2026, with the later code (\cref{sec:methods:networks}), whose decoder multiplies the output by $F_{\max}$ as in the three-dimensional runs. The label-free row evaluates, without retraining them, the networks trained on the same 1,024 instances in the run of 29 September 2026 (\ref{app:e1}). Supervised, stiffness norm: the supervised transformer trained on $\mathcal{L}_K$ \eqref{eq:lk} instead of $\mathcal{L}_D$ \eqref{eq:ld}, with everything else unchanged.

### Two-dimensional plates, the run of 16 July 2026 with the earlier code (\cref{sec:results2d:july}): every metric at the largest label budget, on the instances of \cref{tab:2d}.

| Model | Displacement error | Relative energy gap | von Mises error | Peak stress error | Critical-region recall | Worse than zero field |
|---|---|---|---|---|---|---|
| Label-free | 0.166 ± 0.002 | 0.0817 ± 0.0040 | 0.210 ± 0.003 | 0.228 ± 0.001 | 0.739 ± 0.003 | 3/768 |
| Supervised | 0.160 ± 0.003 | 0.418 ± 0.037 | 0.433 ± 0.031 | 0.334 ± 0.036 | 0.584 ± 0.027 | 52/768 |
| Supervised + energy term | 0.161 ± 0.005 | 0.166 ± 0.040 | 0.257 ± 0.007 | 0.246 ± 0.022 | 0.708 ± 0.012 | 14/768 |
| Label-free, then supervised | 0.166 ± 0.006 | 0.236 ± 0.032 | 0.336 ± 0.020 | 0.301 ± 0.019 | 0.645 ± 0.008 | 21/768 |
| Graph network, supervised | 0.167 ± 0.000 | 2.57 ± 0.25 | 1.10 ± 0.07 | 1.32 ± 0.13 | 0.322 ± 0.027 | 481/768 |
| Nearest-neighbour field | 0.404 | 0.959 | 0.735 | 0.643 | 0.515 | 94/256 |
| Scaled polynomial | 1.42 | 873 | 11.4 | 30.8 | 0.159 | 256/256 |
| Zero field | 1.00 | 1.00 | 1.00 | 1.00 | – | – |

- 256 validation instances, never trained on; four load cases each. Errors are relative to the reference finite-element solution, per load case, then averaged over the load cases; the relative energy gap of the zero field is 1. Trained networks: mean ± sample standard deviation over 3 seeds; the nearest-neighbour field and the scaled polynomial are deterministic.
- Worse than zero field: instance-seed pairs whose relative energy gap exceeds 1, that is, predictions further from the solution in the energy norm than the zero field.
- Critical-region recall: the share of the 10% most stressed elements of the reference solution that are also among the 10% most stressed of the prediction (higher is better; in every other column lower is better).
- Run of 16 July 2026. Its code did not multiply the load scale back onto the output (\cref{sec:methods:networks}); we attribute the narrow spread of the displacement errors to this; no experiment isolated it (\cref{sec:results2d:julyaccuracy}).

### Three-dimensional solids, every metric at the largest label budget. The supervised networks were trained on 1,024 labelled instances; the label-free network on the same 1,024 instances without their solutions.

| Model | Displacement error | Relative energy gap | von Mises error | Peak stress error | Critical-region recall | Worse than zero field |
|---|---|---|---|---|---|---|
| Label-free | 0.0300 ± 0.0028 | 0.00912 ± 0.00228 | 0.0554 ± 0.0045 | 0.0807 ± 0.0134 | 0.793 ± 0.022 | 0/768 |
| Supervised | 0.0470 ± 0.0038 | 31.2 ± 53.5 | 0.370 ± 0.140 | 0.784 ± 0.807 | 0.619 ± 0.005 | 18/768 |
| Supervised + energy term | 0.0470 ± 0.0122 | 0.116 ± 0.050 | 0.128 ± 0.008 | 0.115 ± 0.026 | 0.730 ± 0.006 | 2/768 |
| Graph network, supervised | 0.0168 ± 0.0001 | 0.447 ± 0.084 | 0.296 ± 0.010 | 0.488 ± 0.011 | 0.679 ± 0.009 | 28/768 |
| Nearest-neighbour field | 0.382 | 1.22 | 0.717 | 1.23 | 0.509 | 88/256 |
| Scaled polynomial | 1.56 | 4,362 | 14.9 | 54.8 | 0.137 | 256/256 |
| Zero field | 1.00 | 1.00 | 1.00 | 1.00 | – | – |

- 256 validation instances, never trained on; four load cases each. Errors are relative to the reference finite-element solution, per load case, then averaged over the load cases; the relative energy gap of the zero field is 1. Trained networks: mean ± sample standard deviation over 3 seeds; the nearest-neighbour field and the scaled polynomial are deterministic.
- Worse than zero field: instance-seed pairs whose relative energy gap exceeds 1, that is, predictions further from the solution in the energy norm than the zero field.
- Critical-region recall: the share of the 10% most stressed elements of the reference solution that are also among the 10% most stressed of the prediction (higher is better; in every other column lower is better).
- Run of 28 September 2026.

### Label efficiency: relative displacement error / relative energy gap against the number of labelled training instances (seed means).

| Run | Model | 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|---|
| 2D | Supervised | 0.459 / 20.8 | 0.165 / 3.06 | 0.0712 / 0.430 | 0.0505 / 0.133 |
| 2D | Supervised, stiffness norm | 0.362 / 0.254 | 0.0889 / 0.0505 | 0.0667 / 0.0383 | 0.0547 / 0.0290 |
| 2D | Graph network, supervised | – | 0.370 / 149 | – | 0.0499 / 0.825 |
| 2D | Nearest-neighbour field | 0.687 / 6.50 | 0.466 / 1.65 | 0.408 / 1.03 | 0.404 / 0.959 |
| 2D | Label-free (no labels) | 0.0634 / 0.0383 | 0.0634 / 0.0383 | 0.0634 / 0.0383 | 0.0634 / 0.0383 |
| 2D, July | Supervised | 0.272 / 6.26 | 0.193 / 2.72 | 0.176 / 1.26 | 0.160 / 0.418 |
| 2D, July | Supervised + energy term | 0.410 / 2.56 | 0.184 / 0.579 | 0.182 / 0.154 | 0.161 / 0.166 |
| 2D, July | Label-free, then supervised | 0.207 / 1.32 | 0.189 / 0.201 | 0.181 / 0.302 | 0.166 / 0.236 |
| 2D, July | Graph network, supervised | 0.684 / 92.6 | 0.425 / 80.6 | 0.181 / 7.55 | 0.167 / 2.57 |
| 2D, July | Label-free (no labels) | 0.166 / 0.0817 | 0.166 / 0.0817 | 0.166 / 0.0817 | 0.166 / 0.0817 |
| 3D | Supervised | 0.395 / 18.5 | 0.288 / 5.95 | 0.0726 / 0.799 | 0.0470 / 31.2 |
| 3D | Supervised + energy term | 0.369 / 0.621 | 0.256 / 0.854 | 0.0833 / 0.964 | 0.0470 / 0.116 |
| 3D | Graph network, supervised | – | 0.149 / 18.3 | – | 0.0168 / 0.447 |
| 3D | Nearest-neighbour field | 0.648 / 5.74 | 0.477 / 2.24 | 0.400 / 1.20 | 0.382 / 1.22 |
| 3D | Label-free (no labels) | 0.0300 / 0.00912 | 0.0300 / 0.00912 | 0.0300 / 0.00912 | 0.0300 / 0.00912 |

- Seed means over 3 seeds (the nearest-neighbour field is deterministic, and the same in both 2D runs). A dash marks a budget at which the model was not trained (the graph network was trained at 64 and 1,024 labels only, except in the 2D run of July 2026). 2D: the run of 9 October 2026 (\cref{tab:2d}); 2D, July: the run of 16 July 2026, with the earlier code (\cref{tab:2djuly}). The label-free row repeats its single value, which uses the same 1,024 instances without labels.

### Three-dimensional solids on the finer mesh (lc = 0.0374), never trained on. Zero-shot rows use the networks trained in-band; few-shot rows train, with labels, on the first 16 or on all of the 64 fine-mesh instances reserved for few-shot training.

| Model | Displacement error | Relative energy gap | von Mises error | Peak stress error | Critical-region recall |
|---|---|---|---|---|---|
| Label-free, zero-shot | 0.258 ± 0.045 | 0.306 ± 0.220 | 0.305 ± 0.063 | 0.769 ± 0.667 | 0.721 ± 0.006 |
| Supervised (1,024 labels), zero-shot | 0.257 ± 0.048 | 0.985 ± 0.267 | 0.537 ± 0.040 | 2.56 ± 1.39 | 0.462 ± 0.028 |
| Graph network (1,024 labels), zero-shot | 0.459 ± 0.010 | 16.3 ± 9.6 | 2.40 ± 0.84 | 11.6 ± 7.8 | 0.0610 ± 0.0083 |
| Nearest-neighbour field | 0.401 | 4.98 | 1.55 | 3.61 | 0.232 |
| Scaled polynomial | 1.95 | 540 | 18.7 | 134 | 0.102 |
| Label-free, then 16 fine-mesh labels | 0.0531 ± 0.0030 | 0.0702 ± 0.0112 | 0.166 ± 0.016 | 0.389 ± 0.061 | 0.715 ± 0.004 |
| Supervised from scratch, 16 fine-mesh labels | 0.799 ± 0.037 | 2.34 ± 1.25 | 1.07 ± 0.19 | 2.58 ± 1.23 | 0.116 ± 0.014 |
| Label-free, then 64 fine-mesh labels | 0.0298 ± 0.0020 | 0.0520 ± 0.0088 | 0.126 ± 0.018 | 0.272 ± 0.068 | 0.731 ± 0.014 |
| Supervised from scratch, 64 fine-mesh labels | 0.193 ± 0.076 | 13.2 ± 5.3 | 2.57 ± 0.47 | 8.94 ± 4.65 | 0.128 ± 0.043 |

- 256 fine-mesh evaluation instances (median 36,216 nodes), disjoint from the 64 instances reserved for few-shot training. Mean ± sample standard deviation over 3 seeds; the naive baselines are deterministic and built from the 1,024 in-band labelled instances.
- In-band, the label-free network's displacement error is 0.0300; the zero-shot ratio is therefore 8.62.
- Few-shot training runs 50 epochs at learning rate 1.5 × 10⁻³ from the trained label-free network of the same seed, or from random initialisation (scratch); the few-shot comparison was pre-registered as reported only.

### Every pre-registered criterion of the runs reported here, with its outcome.

| Run | Criterion | Measured | Outcome |
|---|---|---|---|
| 2D, 16 July 2026 | G1′ (a), energy-term sub-experiment (separately trained networks): supervised + energy term at least 3 times better than the zero field in displacement, and better than the 3 naive baselines fitted on 1,024 labelled instances, at every budget | 4.5–6.1 times; naive baselines beaten at every budget | met |
|  | G1′ (b), \cref{tab:2djuly,tab:labeleff}, 64 labels: the energy term lowers the relative energy gap by ≥ 50% and the von Mises error by ≥ 25% | 78.7%, 65.4% | met |
|  | G1′ (c), \cref{tab:2djuly,tab:labeleff}, 64 labels: label-free, then supervised beats supervised by ≥ 10% in relative energy gap or ≥ 5% in displacement | 92.6%, 2.0% | met |
|  | Gate G1′ = (a) and (b) and (c) |  | GO |
|  | K1, energy-term sub-experiment: the better of the fixed and the gradient-scaled energy term lowers the relative energy gap by < 25% at every budget | 98.0% / 94.4% / 90.6% / 74.0% | not triggered |
|  | K2, \cref{tab:2djuly}: label-free displacement error > 30% above supervised, 1,024 labels | +3.6% | not triggered |
|  | C1-advantage, \cref{tab:labeleff}: label-free relative energy-gap advantage over supervised < 40% at every budget | 98.7% / 97.0% / 93.5% / 80.4% | not triggered |
|  | K4: a latent-space regulariser raises the standardised effective rank of the latent vectors by ≤ 1.5 times | 1.70 times | not triggered |
|  | K5: conjugate gradients from the prediction of a label-free network (64 validation instances) save < 20% of the iterations from zero | −0.9% | triggered |
|  | K6: a cross-resolution invariance term shrinks the transfer gap by < 10% at coarsening 2.5 | −23.7% | triggered |
|  | E6 (alignment): within-geometry rank correlation of latent and solution distances < 0.3 | 0.730 | not triggered |
|  | C5: any pre-registered check of the conditioning bound, the modewise identity or the conjugate-gradient bound violated | none | not triggered |
| 2D, 31 July 2026 | K3: a self-supervised joint-embedding term added to label-free pretraining changes the displacement error and the relative energy gap of the transformer, fine-tuned with 16, 64 and 256 labels, by less than 3% at every budget (if triggered, the term is dropped) | change with the term, 16 / 64 / 256 labels: displacement −13.5% / −0.5% / +2.6%; relative energy gap −48.4% / +92.5% / −16.4% (positive: more accurate) | not triggered |
| 3D, 21 September 2026 | Initial run: gate G2 and kill conditions as registered; the label-free objective was defective (deviation D14) | (a), (b), (c) not met; KP1, KP2, KP4 triggered | NO-GO |
| 3D, 22 September 2026 | Pilot of the amendment: label-free displacement error after 20 epochs < 0.90 | 0.309 | met |
| 3D, 28 September 2026 | G2 (a), budgets ≥ 64 (amendment), \cref{tab:labeleff}: supervised + energy term at least 3 times better than the zero field in displacement, and better than both naive baselines | 3.9 / 12.0 / 21.3 times; naive beaten | met |
|  | G2 (a), every budget (the registered form, reported for reference) | 2.71 times at 16 labels | not met |
|  | G2 (b), \cref{tab:labeleff,tab:3d}: label-free displacement error within +10% of supervised at 1,024 labels, and relative energy-gap advantage ≥ 40% at every budget | −36.3%; 99.95% / 99.85% / 98.86% / 99.97% | met |
|  | G2 (c), \cref{tab:transfer}: finer mesh, zero-shot displacement error ≤ 1.25 times in-band, and better than both naive baselines | 8.62 times; best naive 0.401 against 0.258 | not met |
|  | Gate G2 = (a) and ((b) or (c)), amended |  | GO |
|  | Gate G2, registered form (reference) |  | NO-GO |
|  | KP1: supervised (1,024 labels) better than label-free by > 10% in displacement | −36.3% | not triggered |
|  | KP2: relative energy-gap advantage < 40% at any budget | minimum 98.9% | not triggered |
|  | KP3: the energy term lowers the relative energy gap by < 25% at every budget | +96.6% / +85.7% / −20.7% / +99.6% | not triggered |
|  | KP4: finer mesh, zero-shot displacement error > 1.5 times in-band, or a naive baseline better | 8.62 times | triggered |
|  | KP5: any pre-registered check of the conditioning bound, the modewise identity or the conjugate-gradient bound violated | none | not triggered |
|  | KP6 (alignment): within-geometry rank correlation of latent and solution distances < 0.3 | 0.827 | not triggered |
| 2D, 29 September 2026 | Latent regularisation, K1: relative change of the displacement error or of the relative energy gap beyond the resolution | within the resolution | not triggered |
|  | Latent regularisation, K2: the separation does not rise in any seed | rose in every seed | not triggered |
|  | Latent regularisation, GO: separation raised by ≥ 0.02 in every seed | +0.009 / +0.014 / +0.002 | NO-GO |
| 3D, 30 September 2026 | Token bottleneck, M = 512, K1: in-band relative energy gap or finer-mesh displacement error worse than the transformer's by more than the resolution | +280.9% (resolution 287.2%), +79.1% (resolution 41.9%) | triggered |
|  | Token bottleneck, M = 512, K2: finer-mesh training step longer than 2 s | 0.049 s | not triggered |
|  | Token bottleneck, M = 512: outcome |  | dropped |
|  | Token bottleneck, M = 1,024, K1: in-band relative energy gap or finer-mesh displacement error worse than the transformer's by more than the resolution | +183.3% (resolution 70.5%), +122.0% (resolution 33.3%) | triggered |
|  | Token bottleneck, M = 1,024, K2: finer-mesh training step longer than 2 s | 0.047 s | not triggered |
|  | Token bottleneck, M = 1,024: outcome |  | dropped |
| 2D, 9 October 2026 | H1, \cref{tab:2d,tab:cm2d}: label-free transformer lower than the supervised transformer ($\mathcal{L}_D$) in relative energy gap beyond the noise guard, 1,024 labels | −71.3% (guard 12.4%) | supported |
|  | H2a: stiffness-norm transformer ($\mathcal{L}_K$) lower than the supervised transformer ($\mathcal{L}_D$) in relative energy gap beyond the guard | −78.2% (guard 16.4%) | supported |
|  | H2b: stiffness-norm transformer lower than the supervised transformer in von Mises error beyond the guard | −54.9% (guard 12.3%) | supported |
|  | H3 (a reading, no criterion): stiffness-norm transformer against the label-free transformer in relative energy gap | −24.2% (guard 38.1%) | no difference shown |
|  | Reuse: the label-free networks of the run of 29 September 2026 reproduce that run's per-instance relative energy gaps and displacement errors (largest relative deviation at most $10^{-4}$) | largest relative deviation 0 | passed |

- Labels are those of the pre-registration documents, each of which numbers its own criteria. Gates (G1′, G2) decide whether the study proceeds to its next stage. Kill conditions withdraw a claim, or abandon a component, when triggered: in two dimensions K1–K6, C1-advantage (the claimed advantage in relative energy gap), C5 (the pre-registered checks of \cref{sec:theory:checks}) and E6 (latent alignment); in three dimensions KP1–KP6, with KP5 the counterpart of C5. The latent-regularisation and token-bottleneck runs number their criteria afresh, with the meanings given here. A criterion reads the trained networks of the tables it names; the energy-term sub-experiment of the 2D run of 16 July 2026 trained its own supervised networks with and without the energy term, with the same configuration. K4, E6 and KP6 concern latent representations, which this paper does not use. Standardised effective rank: the participation ratio of the eigenvalues of the covariance of the latent vectors, each dimension standardised. The 3D amendment, the deviations and the joint-embedding term are described below, the latent regularisation in \ref{app:e1} and the token bottleneck in \cref{sec:cost:training}. The hypotheses H1–H3 of the run of 9 October 2026 compare a network A with a reference B over three seeds each by a noise guard: with rel = mean(A)/mean(B) − 1 and SE_rel = $(s_A^2/3 + s_B^2/3)^{1/2}$ divided by mean(B), where $s_A$ and $s_B$ are the sample standard deviations over the seeds, A is lower beyond the guard when rel is below minus the guard, max(10%, 2 SE_rel), and higher beyond it when rel exceeds the guard; within it no difference is shown, which is not equivalence (\cref{tab:cm2d}). $\mathcal{L}_D$ and $\mathcal{L}_K$ are the losses \eqref{eq:ld} and \eqref{eq:lk}.

### The pre-registered comparisons of the two-dimensional run of 9 October 2026, with 1,024 labelled instances, as its adjudicator recorded them, and the readings its pre-registration required beside each.

|  | H1 | H2a | H2b | H3 |
|---|---|---|---|---|
| Network A | Label-free | Supervised, stiffness norm | Supervised, stiffness norm | Supervised, stiffness norm |
| Reference B | Supervised | Supervised | Supervised | Label-free |
| Metric | Relative energy gap | Relative energy gap | von Mises error | Relative energy gap |
| A, seeds 0, 1, 2 | 0.0374, 0.0405, 0.0371 | 0.0370, 0.0146, 0.0355 | 0.119, 0.0829, 0.116 | 0.0370, 0.0146, 0.0355 |
| B, seeds 0, 1, 2 | 0.132, 0.148, 0.120 | 0.132, 0.148, 0.120 | 0.232, 0.251, 0.222 | 0.0374, 0.0405, 0.0371 |
| rel | −71.3% | −78.2% | −54.9% | −24.2% |
| SE_rel | 6.2% | 8.2% | 6.1% | 19.0% |
| Guard | 12.4% | 16.4% | 12.3% | 38.1% |
| Reading | supported | supported | supported | no difference shown |
| Medians: rel (guard) | −78.7% (10.0%) | −83.3% (11.6%) | −53.6% (11.8%) | −21.4% (33.7%) |
| Welch 95% interval | −97% to −45% | −101% to −55% | −73% to −37% | −103% to +54% |
| Instance resampling 95% | −74.3% to −68.3% | −80.3% to −76.3% | −57.2% to −52.8% | −28.0% to −21.9% |

- rel = mean(A)/mean(B) − 1 over the three seeds' values, each the mean over the 256 validation instances and their four load cases; SE_rel = $(s_A^2/3 + s_B^2/3)^{1/2}$ divided by mean(B), where $s_A$ and $s_B$ are the sample standard deviations over the seeds; the guard is max(10%, 2 SE_rel). A is lower beyond the guard when rel is below minus the guard; within the guard no difference is shown, which is not equivalence.
- Medians: the same reading on each seed's median over the instances instead of its mean. Welch: rel ± t × SE_rel, with t the 97.5% quantile of Student's distribution at the Welch–Satterthwaite degrees of freedom, a symmetric interval that can extend below −100%. Instance resampling: the 2.5% and 97.5% percentiles of rel over 2,000 resamplings of the 256 validation instances, the same instances drawn for both networks and every seed, the seeds kept.
- H3 is a reading without a criterion. With the roles of A and B exchanged, rel was +31.9% against a guard of 50.2%.

## In-text numbers

- `numAmpFineCstar`: 1.23
- `numAmpFineDisp`: 0.258
- `numAmpFineDispC`: 0.0832
- `numAmpFineGapMedian`: 0.0911
- `numAmpFineGapMedianC`: 0.0418
- `numAmpFineRedMax`: 69%
- `numAmpFineRedMin`: 66%
- `numAmpIdentity`: $2 \times 10^{-12}$
- `numAmpInCstar`: 1.00
- `numAmpInDisp`: 0.0300
- `numAmpInDispC`: 0.0142
- `numAmpInGap`: 0.00912
- `numAmpInGapC`: 0.00740
- `numAmpRatioBoth`: 5.9
- `numAmpRatioFineOnly`: 2.8
- `numAmpRows`: 4,608
- `numAuditItems`: 22 of 22
- `numChebThree`: 0.055
- `numChebTwo`: 0.043
- `numCmAdvMax`: 99.8%
- `numCmAdvMin`: 71.3%
- `numCmDispSpread`: 1.27
- `numCmFreeDisp`: 0.0634
- `numCmFreeDispExcessAbs`: 25.6%
- `numCmFreeDispExcessMax`: 37%
- `numCmFreeDispExcessMin`: 15%
- `numCmFreeDispFour`: 0.0634
- `numCmFreeGap`: 0.0383
- `numCmFreeGapMedian`: 0.0251
- `numCmFreeGapSeeds`: 0.0374, 0.0405 and 0.0371
- `numCmFreePeak`: 0.128
- `numCmFreeRecall`: 0.753
- `numCmFreeVm`: 0.123
- `numCmFreeWorse`: 0
- `numCmKnormDisp`: 0.0547
- `numCmKnormGap`: 0.0290
- `numCmKnormGapMedian`: 0.0172
- `numCmKnormGapSeeds`: 0.0370, 0.0146 and 0.0355
- `numCmKnormPeak`: 0.0947
- `numCmKnormRecall`: 0.804
- `numCmKnormVm`: 0.106
- `numCmKnormWorse`: 0
- `numCmKnormWorseSixteen`: 11
- `numCmKnormWorseSixtyFour`: 1
- `numCmKnormWorseTwoFiveSix`: 0
- `numCmLabDisp`: 0.0505
- `numCmLabDispFour`: 0.0505
- `numCmLabGap`: 0.133
- `numCmLabGapMedian`: 0.117
- `numCmLabGapSixteen`: 20.8
- `numCmLabOverFreeGap`: 3.5
- `numCmLabOverFreeVm`: 1.9
- `numCmLabOverKnormGap`: 4.60
- `numCmLabPeak`: 0.141
- `numCmLabRecall`: 0.668
- `numCmLabVm`: 0.235
- `numCmLabWorse`: 0
- `numCmLabWorseSixteen`: 768
- `numCmLabWorseSixtyFour`: 532
- `numCmLabWorseTwoFiveSix`: 53
- `numCmMgnBelowLabDisp`: 1%
- `numCmMgnDisp`: 0.0499
- `numCmMgnGap`: 0.825
- `numCmMgnGapMedian`: 0.521
- `numCmMgnOverFreeVm`: 4.7
- `numCmMgnOverLabGap`: 6.2
- `numCmMgnPeak`: 0.321
- `numCmMgnRecall`: 0.517
- `numCmMgnVm`: 0.578
- `numCmMgnWorse`: 187
- `numCmMgnWorseSixtyFour`: 766
- `numCmPairs`: 768
- `numCmSecondary`: 80
- `numCmSolves`: 5,120
- `numCmSolvesTrain`: 4,096
- `numCmSolvesVal`: 1,024
- `numCmVal`: 256
- `numCoarsenHigh`: 2.5
- `numCoarsenHighCoarse`: 0.174
- `numCoarsenHighFine`: 0.198
- `numCoarsenHighReduction`: −23.7%
- `numCoarsenHighWiden`: 23.7%
- `numCoarsenLow`: 1.8
- `numCoarsenLowCoarse`: 0.182
- `numCoarsenLowFine`: 0.189
- `numCoarsenLowReduction`: −17.6%
- `numCoarsenLowWiden`: 17.6%
- `numCondThree`: 0.012
- `numCondTwo`: 0.022
- `numCrossThree`: $7.43 \times 10^{-3}$
- `numCrossTwo`: $8.39 \times 10^{-3}$
- `numDfourteenAlpha`: $3.6 \times 10^{-4}$
- `numDfourteenDisp`: 0.99964
- `numDfourteenGap`: 0.99928
- `numDiagFixedMax`: 0.22
- `numDiagFixedMin`: 0.20
- `numDiagFixedSeeds`: six
- `numDiagScaledMax`: 0.52
- `numDiagScaledMin`: 0.22
- `numDiagSeeds`: six
- `numEOneBaseDisp`: 0.0634
- `numEOneBaseGap`: 0.0383
- `numEOneDeltaMax`: 0.014
- `numEOneDeltaMin`: 0.002
- `numEOneFloor`: 0.02
- `numEtwoFineMax`: 2.2
- `numEtwoFineMin`: 1.8
- `numEtwoGapMax`: 3.8
- `numEtwoGapMin`: 2.8
- `numEtwoNodesFine`: 41,367
- `numEtwoNodesIn`: 12,318
- `numEtwoStepMax`: 51
- `numEtwoStepMin`: 38
- `numEtwoTransformerFineStep`: 14.8
- `numFewFactorMax`: 5.7
- `numFewFactorMin`: 2.3
- `numFewSixteenDisp`: 0.0531
- `numFewSixteenGap`: 0.0702
- `numFewSixteenPeak`: 0.389
- `numFewSixteenScratchDisp`: 0.799
- `numFewSixteenVm`: 0.166
- `numFewSixtyFourDisp`: 0.0298
- `numFewSixtyFourGap`: 0.0520
- `numFewSixtyFourPeak`: 0.272
- `numFewSixtyFourScratchDisp`: 0.193
- `numFewSixtyFourVm`: 0.126
- `numFieldBoundMax`: 0.85
- `numFieldLoadWorseFree`: 1
- `numFieldLoadWorseLab`: 126
- `numFieldLoadWorseMgn`: 106
- `numFieldLoads`: 3,072
- `numFieldRqAboveLab`: 89.0%
- `numFieldRqAboveMgn`: 99.9%
- `numFieldRqFree`: 11.9
- `numFieldRqLab`: 132
- `numFieldRqMgn`: 1,026
- `numFieldVmElemFree`: 0.0508
- `numFieldVmElemLab`: 0.257
- `numFieldVmElemMgn`: 0.226
- `numFieldVmVolFree`: 0.0474
- `numFieldVmVolLab`: 0.247
- `numFieldVmVolMgn`: 0.209
- `numFigFineLoad`: downward body force
- `numFigMedianLoad`: downward body force
- `numFigTwoDDispMax`: 0.033
- `numFigTwoDDispMin`: 0.026
- `numFigTwoDGapMax`: 0.66
- `numFigTwoDGapMin`: 0.031
- `numFigTwoDLoad`: shear traction on the top edge
- `numFigWorstFreeC`: 0.78
- `numFigWorstFreeDisp`: 0.22
- `numFigWorstFreeDispC`: 0.045
- `numFigWorstLoad`: downward body force
- `numFigWorstTwiceFree`: 12%
- `numFigWorstTwiceLab`: 67%
- `numFigWorstTwiceMgn`: 52%
- `numFineFreeDisp`: 0.258
- `numFineFreeGap`: 0.306
- `numFineFreeVm`: 0.305
- `numFineKill`: 1.5
- `numFineKnnDisp`: 0.401
- `numFineLabDisp`: 0.257
- `numFineLabGap`: 0.985
- `numFineLabVm`: 0.537
- `numFineMgnDisp`: 0.459
- `numFineMgnGap`: 16.3
- `numFineMgnVm`: 2.40
- `numFineRatio`: 8.6
- `numFineWin`: 1.25
- `numGOneFtDisp`: 2.0%
- `numGOneFtGap`: 92.6%
- `numGOneGapReduction`: 78.7%
- `numGOneRetired`: 5.8%
- `numGOneSanityMax`: 6.1
- `numGOneSanityMin`: 4.5
- `numGOneSanityTableSixteen`: 2.4
- `numGOneVmReduction`: 65.4%
- `numGTwoAdvMin`: 98.9%
- `numGTwoFloor`: 64
- `numGTwoParity`: −36.3%
- `numGTwoSanityMax`: 21.3
- `numGTwoSanitySixteen`: 2.71
- `numGTwoSanitySixtyFour`: 3.9
- `numGTwoSanityX`: 3.0
- `numHOneRel`: 71.3%
- `numHOneTau`: 12.4%
- `numHThreeExRel`: 31.9%
- `numHThreeExTau`: 50.2%
- `numHThreeRel`: 24.2%
- `numHThreeResHigh`: 28.0%
- `numHThreeResLow`: 21.9%
- `numHThreeSeedRel`: 1%, 64% and 4%
- `numHThreeSeedsZeroTwo`: 2.7%
- `numHThreeTau`: 38.1%
- `numHTwoARel`: 78.2%
- `numHTwoATau`: 16.4%
- `numHTwoBRel`: 54.9%
- `numHTwoBTau`: 12.3%
- `numJulyOverCmDispMax`: 3.4
- `numJulyOverCmDispMin`: 2.6
- `numKappaTauTwo`: $1.5 \times 10^{-7}$
- `numKappaThreeHigh`: $5.2 \times 10^{5}$
- `numKappaThreeLow`: $2.9 \times 10^{4}$
- `numKappaTwoHigh`: $1.5 \times 10^{5}$
- `numKappaTwoLow`: $4.8 \times 10^{3}$
- `numKfiveEval`: 64
- `numKfiveFiveGap`: 0.0430
- `numKfiveFree`: 242.5
- `numKfiveNaive`: 261.3
- `numKfiveNaiveSaving`: −8.7%
- `numKfiveSaving`: −0.9%
- `numKfiveStartGap`: 0.0722
- `numKfiveTol`: $10^{-6}$
- `numKfiveTwentyGap`: 0.0260
- `numKfiveZero`: 240.4
- `numKthreeBand`: 3%
- `numKthreeBaseRef`: 0.201 ± 0.033
- `numKthreeBaseSeeds`: 0.811, 6.99 and 1.66
- `numKthreeGain`: 92.5%
- `numKthreeOutside`: four
- `numKthreeWorse`: three
- `numMgnDispLowerMax`: 92.6%
- `numMgnDispLowerMin`: 84.4%
- `numMgnGapHigherMin`: 99.2%
- `numMgnLowerDispHigherVm`: 685
- `numMgnPeakLowerMax`: 29%
- `numMgnPeakLowerMin`: 17%
- `numMgnVmHigher`: 767
- `numModesPhrase`: $2 \times 10^{-15}$ in both dimensions
- `numModesThree`: $2 \times 10^{-15}$
- `numModesTwo`: $2 \times 10^{-15}$
- `numNodesFineMax`: 71,749
- `numNodesFineMedian`: 36,216
- `numNodesFineMin`: 15,175
- `numNodesInMax`: 17,692
- `numNodesInMedian`: 5,617
- `numNodesInMin`: 2,195
- `numPairCmFreeKnormGapTwoFiveSix`: 72.4%
- `numPairCmKnormFreeDisp`: 68.1%
- `numPairCmKnormFreeGap`: 87.6%
- `numPairCmKnormFreePeak`: 84.9%
- `numPairCmKnormFreeRecall`: 74.1%
- `numPairCmKnormFreeVm`: 83.3%
- `numPairCmKnormLabDisp`: 49.7%
- `numPairCmKnormLabGap`: 99.9%
- `numPairCmKnormLabPeak`: 84.6%
- `numPairCmKnormLabRecall`: 98.6%
- `numPairCmKnormLabVm`: 99.9%
- `numPairCmLabDisp`: 29.9%
- `numPairCmLabGap`: 99.9%
- `numPairCmLabPeak`: 62.6%
- `numPairCmLabRecall`: 96.7%
- `numPairCmLabVm`: 99.9%
- `numPairCmMgnLabDisp`: 55.7%
- `numPairCmMgnLabGap`: none
- `numPairCmMgnLabPeak`: 41.0%
- `numPairCmMgnLabRecall`: 9.8%
- `numPairCmMgnLabVm`: none
- `numPairCmOpposite`: 69.9%
- `numPairThreeAncDisp`: 81.2%
- `numPairThreeAncGap`: 99.7%
- `numPairThreeAncPeak`: 75.3%
- `numPairThreeAncRecall`: 95.7%
- `numPairThreeAncVm`: 99.7%
- `numPairThreeLabDisp`: 78.9%
- `numPairThreeLabGap`: 99.9%
- `numPairThreeLabPeak`: 87.4%
- `numPairThreeLabRecall`: all
- `numPairThreeLabVm`: 99.9%
- `numPairTwoAncDisp`: 36.2%
- `numPairTwoAncGap`: 91.8%
- `numPairTwoAncPeak`: 44.8%
- `numPairTwoAncRecall`: 76.4%
- `numPairTwoAncVm`: 75.1%
- `numPairTwoLabDisp`: 39.1%
- `numPairTwoLabGap`: all
- `numPairTwoLabPeak`: 63.2%
- `numPairTwoLabRecall`: 98.8%
- `numPairTwoLabVm`: 99.3%
- `numParamsAux`: 0.6
- `numParamsGraph`: 3.9
- `numParamsTransformer`: 6.7
- `numPilotDisp`: 0.309
- `numPilotLimit`: 0.90
- `numPremiseThree`: 0.462
- `numPremiseTwo`: 0.464
- `numRemeshFine`: 1.27
- `numRemeshGeometries`: 16
- `numRemeshInDev`: 2%
- `numRhoThree`: 0.827
- `numRhoTwo`: 0.730
- `numSpecBoundMax`: 0.90
- `numSpecFigIndex`: 221
- `numSpecFreeMax`: 3,622
- `numSpecFreeMin`: 348
- `numSpecIeeeLabKnormGap`: 11.9
- `numSpecIeeeLabKnormRq`: 9.30
- `numSpecIeeeRqFree`: 9.12
- `numSpecIeeeRqKnorm`: 9.42
- `numSpecIeeeRqLab`: 114
- `numSpecIeeeRqMgn`: 558
- `numSpecKnormFreeAbove`: 49.1%
- `numSpecKnormFreeDisp`: 0.64
- `numSpecKnormFreeDispSeeds`: 0.81, 0.39 and 0.81
- `numSpecKnormFreeGap`: 0.66
- `numSpecKnormFreeRq`: 1.04
- `numSpecLabKnormAbove`: 94.9%
- `numSpecLabKnormAboveSeedGm`: 98.8%
- `numSpecLabKnormDisp`: 1.26
- `numSpecLabKnormDispLower`: 50.0%
- `numSpecLabKnormDispSeeds`: 0.96, 1.65 and 1.26
- `numSpecLabKnormGap`: 9.85
- `numSpecLabKnormRq`: 7.80
- `numSpecLabKnormRqSeeds`: 8.26, 9.46 and 6.07
- `numSpecLoadCases`: 1,024
- `numSpecLoadWorseFree`: 1
- `numSpecLoadWorseKnorm`: 0
- `numSpecLoadWorseLab`: 1
- `numSpecLoadWorseMgn`: 724
- `numSpecLoads`: 3,072
- `numSpecMgnLabAbove`: 93.4%
- `numSpecMgnLabDisp`: 1.00
- `numSpecMgnLabGap`: 4.34
- `numSpecMgnLabRq`: 4.33
- `numSpecRqFree`: 10.3
- `numSpecRqKnorm`: 11.3
- `numSpecRqLab`: 115
- `numSpecRqMgn`: 553
- `numSpecRqStarOverMin`: 1.27
- `numSpecShareEucFree`: 7.2%
- `numSpecShareEucKnorm`: 6.6%
- `numSpecShareEucLab`: 24%
- `numSpecShareEucMgn`: 49%
- `numSpecShareEucStar`: 0.2%
- `numSpecShareKFree`: 69%
- `numSpecShareKKnorm`: 73%
- `numSpecShareKLab`: 95%
- `numSpecShareKMgn`: 99%
- `numSpecShareKStar`: 1.5%
- `numSpecSpectralShare`: 90%
- `numSpecSpectralShareMax`: 102%
- `numSpecSpectralShareMin`: 82%
- `numSpecTfFree`: 2.7%
- `numSpecTfKnorm`: 2.9%
- `numSpecTfLab`: 0.7%
- `numSpecTfMgn`: 0.05%
- `numSpecTfStiffFree`: 41%
- `numSpecTfStiffKnorm`: 36%
- `numSpecTfStiffLoads`: 670
- `numSpecTfStiffN`: 2,010
- `numSpecVmAreaFree`: 0.105
- `numSpecVmAreaKnorm`: 0.0896
- `numSpecVmAreaLab`: 0.227
- `numSpecVmAreaMgn`: 0.462
- `numSpecVmAreaRatio`: 2.26
- `numSpecVmElemFree`: 0.112
- `numSpecVmElemKnorm`: 0.0964
- `numSpecVmElemLab`: 0.231
- `numSpecVmElemMgn`: 0.474
- `numSpecVmElemRatio`: 2.22
- `numSqrtKappaMax`: 723
- `numSqrtKappaMin`: 69
- `numTauSqKappaTwo`: $6 \times 10^{-18}$
- `numTauSquared`: $10^{-12}$
- `numThreeAncDisp`: 0.0470
- `numThreeAncGap`: 0.116
- `numThreeAncGapMedian`: 0.108
- `numThreeAncOverFreeGap`: 12.7
- `numThreeAncPeak`: 0.115
- `numThreeAncRecall`: 0.730
- `numThreeAncVm`: 0.128
- `numThreeAncWorse`: 2
- `numThreeAncWorseTwoFiveSix`: 4
- `numThreeEvalSolves`: 2,048
- `numThreeFewSolves`: 256
- `numThreeFreeDisp`: 0.0300
- `numThreeFreeDispAdvantage`: 36.3%
- `numThreeFreeDispFour`: 0.0300
- `numThreeFreeGap`: 0.00912
- `numThreeFreeGapMedian`: 0.00763
- `numThreeFreeLoadWorse`: 1
- `numThreeFreeLoads`: 3,072
- `numThreeFreeOverMgnDisp`: 1.8
- `numThreeFreePeak`: 0.0807
- `numThreeFreeRecall`: 0.793
- `numThreeFreeVm`: 0.0554
- `numThreeFreeWorse`: 0
- `numThreeLabDisp`: 0.0470
- `numThreeLabDispFour`: 0.0470
- `numThreeLabGap`: 31.2
- `numThreeLabGapMedian`: 0.252
- `numThreeLabGapMs`: 31.2 ± 53.5
- `numThreeLabGapSeeds`: 0.297, 93.0, 0.365
- `numThreeLabPeak`: 0.784
- `numThreeLabRecall`: 0.619
- `numThreeLabVm`: 0.370
- `numThreeLabWorse`: 18
- `numThreeLabWorseTwoFiveSix`: 132
- `numThreeLabWorst`: 23,727
- `numThreeLabWorstSeed`: 1
- `numThreeMedianRatioMax`: 33
- `numThreeMedianRatioMin`: 14
- `numThreeMgnDisp`: 0.0168
- `numThreeMgnGap`: 0.447
- `numThreeMgnGapMedian`: 0.208
- `numThreeMgnOverFreePeak`: 6.0
- `numThreeMgnOverFreeVm`: 5.3
- `numThreeMgnPeak`: 0.488
- `numThreeMgnRecall`: 0.679
- `numThreeMgnVm`: 0.296
- `numThreeMgnWorse`: 28
- `numThreePairs`: 768
- `numThreeSolves`: 6,400
- `numThreeTrainSolves`: 4,096
- `numThreeVal`: 256
- `numTimeBottleneckSeedMax`: 2.5
- `numTimeBottleneckSeedMin`: 2.5
- `numTimeBottleneckSpeedup`: 11
- `numTimeThreeSeed`: 27.3
- `numTimeTwoParallel`: 2 h 58 min
- `numTwoAdvMax`: 98.7%
- `numTwoAdvMin`: 80.4%
- `numTwoAncDisp`: 0.161
- `numTwoAncGap`: 0.166
- `numTwoAncGapMedian`: 0.0709
- `numTwoAncPeak`: 0.246
- `numTwoAncRecall`: 0.708
- `numTwoAncSixteenSeeds`: 0.225, 0.458, 0.546
- `numTwoAncVm`: 0.257
- `numTwoAncWorse`: 14
- `numTwoDispSpread`: 1.05
- `numTwoFreeDisp`: 0.166
- `numTwoFreeDispExcess`: +3.6%
- `numTwoFreeDispExcessAbs`: 3.6%
- `numTwoFreeDispFour`: 0.1658
- `numTwoFreeGap`: 0.0817
- `numTwoFreeGapMedian`: 0.0493
- `numTwoFreePeak`: 0.228
- `numTwoFreeRecall`: 0.739
- `numTwoFreeVm`: 0.210
- `numTwoFreeWorse`: 3
- `numTwoFtDisp`: 0.166
- `numTwoFtGap`: 0.236
- `numTwoFtGapMedian`: 0.156
- `numTwoFtPeak`: 0.301
- `numTwoFtRecall`: 0.645
- `numTwoFtVm`: 0.336
- `numTwoFtWorse`: 21
- `numTwoGapSpread`: 31
- `numTwoLabDisp`: 0.160
- `numTwoLabDispFour`: 0.1600
- `numTwoLabGap`: 0.418
- `numTwoLabGapMedian`: 0.274
- `numTwoLabGapSixteen`: 6.26
- `numTwoLabOverFreeGap`: 5.1
- `numTwoLabPeak`: 0.334
- `numTwoLabRecall`: 0.584
- `numTwoLabVm`: 0.433
- `numTwoLabWorse`: 52
- `numTwoMgnDisp`: 0.167
- `numTwoMgnGap`: 2.57
- `numTwoMgnGapMedian`: 1.35
- `numTwoMgnPeak`: 1.32
- `numTwoMgnRecall`: 0.322
- `numTwoMgnVm`: 1.10
- `numTwoMgnWorse`: 481
- `numTwoPairs`: 768
- `numTwoRuns`: nine
- `numTwoRunsDisp`: 0.1658 ± 0.0023
- `numTwoRunsGap`: 0.0820 ± 0.0026
- `numTwoSolves`: 7,168
- `numTwoSolvesOffline`: 5,120
- `numTwoSolvesRun`: 2,048
- `numTwoVal`: 256
- `numWarmFractionSeven`: 15%
- `numWarmFractionSeventeen`: 6%
- `numWarmFractionTwelve`: 9%
- `numWithinThree`: $6.76 \times 10^{-2}$
- `numWithinTwo`: $7.56 \times 10^{-2}$
- `numWorstFreeDisp`: 0.071
- `numWorstFreeGap`: 0.0349
- `numWorstFreeVm`: 0.10
- `numWorstIndex`: 107
- `numWorstLabDisp`: 0.032
- `numWorstLabGap`: 2.90
- `numWorstLabVm`: 0.73
- `numWorstMgnDisp`: 0.033
- `numWorstMgnGap`: 27.0
- `numWorstMgnVm`: 2.4
- `numWorstSeedLabDisp`: 0.55
- `numWorstSeedLabPeak`: 361
- `numWorstSeedLabVm`: 65
