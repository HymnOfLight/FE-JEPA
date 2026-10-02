# CPU pilot of S (wp9 Stage 0c, 2 October 2026)

Exploratory, toy scale, run in the sandbox before PREREG_W9 is stamped. **Not
evidence for H2**, which is decided on the box at E1's scale, seeds and
evaluation sets. Nothing in PREREG_W9 depends on it; the S factor (1/64) stays
fixed by its Sec. 2 measurement (`records/wp9/scale_factor.json`).

## Design (pre-declared in `scripts/w9_pilot_s.py` before it ran)

The questions and red flags below were written into the script's docstring
before the pilot ran; the script and its results were first committed
together, so this record is the only attestation of that order.

- Questions: (a) does an S model train stably for thousands of steps; (b) on
  meshes finer than training, does the max-scaled baseline's amplitude deficit
  (c* above 1, growing as h shrinks) shrink under S?
- Red flags (reported to the PI before the stamp; no other decision is taken on
  these numbers): any of S's readings non-finite; S's in-band energy gap more
  than twice the baseline's (seed means).
- Data, its own draws only: a gmsh training-family corpus (n = 288, seed 920;
  validation 32, pool 256), an F5-like set (32 instances at h = 0.025, seed
  92005), an R-like remesh set (8 geometries at the five R mesh sizes, seed
  92006). None of PREREG_W9's sets (seeds 91001-91006; IB's seed 91007 came
  later) was generated or read.
- Model and training: E1's base configuration with dim 32 / depth 2 / heads 4
  (E1: 256 / 8 / 8); AR on the pool of 256 for 12 epochs (3,072 steps per seed;
  E1: 204,800); seeds 0 and 1 for both arms; CPU, TF32 off. Arms: `base` (E1's
  configuration) and `s` (`decode_scale = "l1"`, `decode_scale_factor = 1/64`,
  `features.load_density = true`).

## As run

- Code: Stage 0b's commit (`b190733`) plus Stage 0c work then in progress,
  not committed (the reports' git field names a sandbox-only commit of Stage
  0b's tree, never pushed, with local changes); Python 3.11.15, numpy 2.4.4,
  torch 2.14.0 (CPU).
- Attempt 1 (two workers) trained both baseline seeds and saved their states;
  one worker was then killed by the sandbox's memory limit (5.8 GiB) while
  evaluating, and the pool ended the run as designed. Cause: on CPU, PyTorch's
  inference path of `nn.MultiheadAttention` materializes the attention matrices
  (4 load cases x 4 heads x N^2 floats, about 3.9 GB at the largest F5-like mesh,
  N = 7,760; measured: 574 MB peak with the fast path off, 2,387 MB with it on at
  2 heads). On CUDA the same path calls memory-efficient attention: the box
  evaluated Phase 2's fine 3D meshes (about 4e4 nodes) with this encoder, which
  materialized attention could not have held. No box run evaluates on CPU.
- Attempt 2: `--workers 1 --reuse-states` (scheduling only): the saved baseline
  states were evaluated and S was trained and evaluated in the same process
  (about 19 min in all). The summary step was then fixed to read only the sets the
  pilot has (it had asked for F1) and rerun on the two reports.
- SHA-256 (sandbox files, not committed): base report `d6161e44...`, S report
  `1c52c44a...`; states base s0 `12bf3e10...`, s1 `0974f0ae...`; S s0
  `3df0d325...`, s1 `e7d6e532...`.

## Result (`summary.json`; seed 0 / seed 1)

| Reading | Baseline | S |
|---|---|---|
| In-band energy gap | 0.204 / 0.189 | 0.169 / 0.126 |
| In-band displacement error | 0.477 / 0.422 | 0.393 / 0.299 |
| F5-like displacement error | 0.695 / 0.679 | 0.399 / 0.336 |
| F5-like / in-band displacement | 1.46 / 1.61 | 1.02 / 1.13 |
| Median c* on F5-like | 2.41 / 2.67 | 0.94 / 0.98 |
| F5-like displacement after c* | 0.332 / 0.256 | 0.249 / 0.237 |

R-like set, seed means, h = 0.12 / 0.085 / 0.05 / 0.035 / 0.025:

| Reading | Baseline | S | Exact solution |
|---|---|---|---|
| c_b (median over geometries) | 0.889 / 1.091 / 1.519 / 1.908 / 2.419 | 1.040 / 1.016 / 0.988 / 0.975 / 0.959 | -- |
| Energy norm relative to h = 0.12 | 1 / 0.780 / 0.527 / 0.405 / 0.311 | 1 / 1.001 / 1.022 / 1.025 / 1.035 | 1 / 1.005 / 1.011 / 1.013 / 1.014 |

Battery scales on the R-like set relative to h = 0.12 (median over the
geometries): max|F| 0.700 / 0.416 / 0.294 / 0.211 at h = 0.085 / 0.05 / 0.035 /
0.025 (h itself: 0.708 / 0.417 / 0.292 / 0.208); sum|F| 1.000 / 0.999 / 0.999 /
0.999.

- Red flags: none (every reading of S finite; S's in-band energy gap is below
  the baseline's).
- Reading (exploratory): at this scale the baseline's decoded field shrinks with
  max|F| on finer meshes (its energy norm falls to 0.31 at h = 0.025 while the
  exact one stays at 1.01; median c* 2.4-2.7); S's follows the exact solution
  (c* 0.94-0.98) and its F5-like / in-band ratio is near 1. The toy model is far
  from E1's (32 against 256 channels, 3,072 against 204,800 steps), so neither
  the size of the baseline's deficit nor S's gain carries over; session 1 reads
  E1's own deficit on F5 (rule 1).
