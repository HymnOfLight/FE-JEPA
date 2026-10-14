# records/cmame -- cmame-paper records

## Default path against wp8-lejepa (Stage 2, run again at Stages 9 and 11)

Stage 2 changed `src` (opt-in additions only: the stiffness-norm loss of the
supervised trainer, its E8 row, an E8 run that trains a supervised grid beside
reused label-free states, and their step count), and
`scripts/regress_against_branch.py` ran in the sandbox on 6 October 2026 (CPU,
one thread, each side in its own process importing its own source tree; a
fresh pass and a restart pass) with this side's `src` = tree
`2da09d0593669f50149f4fb546c01861ac0d5149`, the code CM2D ran.

Stage 9 changed `src` again: `fe/tet3d.py`'s structured tetrahedral mesh, which
split each hexahedral cell into five tetrahedra covering 5/6 of it, now uses
the conforming six-tetrahedron Kuhn split (no run reported in the manuscript
and no record here used this mesh; only the three WP7-S1 smoke configurations,
`scripts/bench3d_scale.py` and `scripts/amp_compile_check.py` did). The
regression ran again in the sandbox on 10 October 2026, with the same library
versions, on the same two miniatures as `records/wp9/`;
`regression_wp8_e1_2d_base.json` and `regression_wp8_phase2b.json` are now its
summaries, identical to Stage 2's apart from this side's tree. Stage 11 added
two modules that no run imports (`fejepa.analysis.holdout` and
`fejepa.analysis.adjudicate_holdout`, PREREG_HOLDOUT's evaluation and
adjudication); the regression ran again on 14 October 2026 in the same way,
with the same result (Stage 9's tree was `7cc8ac1b6b16673f14609a422f7e95479d968df0`;
the runs took 60 s and 147 s):

| | source tree (`git rev-parse <commit>:src`) |
|---|---|
| this side | the Stage 11 code: `src` = tree `b719ea99d4063cf01fd183395745e0e87eba9591` |
| the other side | `wp8-lejepa` at `5f8e2df`: `src` = tree `5be6357b912cad058121092f274e6a322169b9f6` |

- `regression_wp8_e1_2d_base.json`: a miniature of `configs/e1_2d_base.json`
  (E1's stamped base) -- 162 numbers and 2 state files, identical (0
  differences; every state file byte-identical).
- `regression_wp8_phase2b.json`: a miniature of `configs/phase2b_v1.json` (E8
  with labels, labels_anchor, AR and MGN; P3; WP6; E6; gate G2) -- 2,590
  numbers and 6 state files, identical.

The two summaries equal `records/wp9/`'s except for this side's tree (wp9's
ran the Stage 0c code, tree `710ccb32...`). Each run took a few minutes (Stage
9: 43 s and 100 s). `tests/test_w9_records.py` checks
that a committed `src` without local changes is the tree named by the newest
regression record (this file on this branch): a change to `src` needs the
regression run again and this table updated. The miniatures exercise the
default paths only; the new opt-in paths are covered by
`tests/test_cm2d_training.py`. Bitwise identity is a same-machine,
same-library statement (sandbox: Python 3.11.15, torch 2.14.0+cu130 on CPU,
numpy 2.4.4, scipy 1.17.1).

## The decision rule's operating characteristics and the run's wall time (PREREG_CM2D Sec. 4, 7)

`cm2d_sims.json`: `scripts/cm2d_sims.py` (fixed seed; seconds), the source of
every simulated number PREREG_CM2D quotes. The E-series noise guard on three
seeds per arm with log-normal per-seed values: false and true "lower" and
"worse" readings at equal and unequal seed spreads (`cells`), with the
reference at July's labels-only spread in the relative energy gap, 9%
(`july_reference`), and with per-instance blow-ups (one validation instance
at 20-40 times an arm's level, about one per seed) in one arm or both
(`blowups`); and, from cmame-paper Stage 5 (PREREG_CM2D r3), with the
reference at that row's spread in the von Mises error, 7.1%
(`july_reference_vm`), drawn after all the others so that every earlier value
is unchanged. `schedule`: the 30 units of `configs/cm2d_v1.json` in the order
the run submits them on three workers, at the measured 52 ms per supervised
step plus a minute per unit, for the graph network's (unmeasured) step at
0.5-3 times the transformer's (2.5 and 3 added in Stage 5). The script
regenerates the file byte for byte (`--check`; tested).

## A functional check of the stiffness-norm loss (PREREG_CM2D Sec. 4)

`cm2d_funccheck.json`: `scripts/cm2d_funccheck.py`, run in the sandbox on 7
October 2026 (CPU, one thread; about 15 minutes) with the Stage 2 code (`src`
tree `2da09d05...`), before PREREG_CM2D's stamp. 72 instances of the 2D
training family drawn by the gmsh generator with its own seed (777; not E1's
corpus), 48 for training and 24 for validation; a transformer of width 32 and
depth 2 trained for 25 epochs with L_D, with L_K and with the label-free
objective, seeds 0 and 1. Recorded per loss and seed: the metric suite, the
gradient norms before clipping and the share of clipped steps; and the L_K
value the trainer computes against Lemma 1 over 200 steps. PREREG_CM2D Sec. 4
quotes its energy gaps and von Mises errors (tested); it is not evidence for
any of PREREG_CM2D's hypotheses. The record carries the script's SHA-256 and
the tree of the `src` it ran (named only when `src` had no local changes);
rerunning the script on the same machine and library versions gives the same
numbers apart from the wall-clock seconds.

## Sec. A return: inference timing and field export (8 October 2026)

`timing/` holds RUNBOOK_CMAME Sec. A as it ran on the box (RTX 5090, driver
595.71.05, torch 2.12.1+cu130, Python 3.12.3; a 16-CPU allotment of a
two-socket Xeon Gold 6459C host) on commit `61f018f` (tree
`ec2ecbf2057f28a97a94b1fe867ad2c9eb89a4ca`, clean), copied byte for byte
from `cmame_timing_return_20261008.tgz`
(SHA-256 `81d6f079bd271691c3b2474e3434d8a59f06aeae7072b840e15fe06836f79523`,
as the operator sent it and as recomputed on receipt). The suite there: 505
passed, 2 deselected (the two LaTeX compile tests).

| file | SHA-256 |
|---|---|
| `fields/energies_val.npz` | `504498e4030ef2645166baf2cf7cc12633272245bd638ed97dcd4d8a954c5297` |
| `fields/fields.json` | `c25bc4507639ced8158cf327b248a29a82c4078cfe2dd51a639b3e0ec65f1583` |
| `fields/fig5.npz` | `594556cc327987e17083a1fb967a53dbe3b6736679f7746d032385d852c5a4a5` |
| `fields/fig6.npz` | `b4ffede43f0029882da911d3da102e225e468679f20c9700fbdd4255a646771e` |
| `fields/fig7.npz` | `241fd0d22dc2b2040661d84227d952bfdad027ccd2cc3a785c641504107c484c` |
| `fields_3d.log` | `d7add2c407fa9adf47576387b97e89f032ced142efc34213604dc60b8f0bf2dc` |
| `machine.txt` | `2789deea33efcbedacbd75c449969f5ab6bf6386a5f76f5131f809256297fda0` |
| `precheck.txt` | `0bd95b80755017f80a0495437677a22e28a2847a6314d2339da393f2bfa91359` |
| `pytest.log` | `127543191e7db8581ba7a07fffd0cb6de79455d8074823fdd227a25f3d4c53f8` |
| `status.txt` | `f4f1843700b60f100953be2ced1b05e59966a7cd1be169674d1a4ad845e3b8ae` |
| `timing_2d.json` | `573543b13c54e85e1eaa275abef878f38c42c6590dc1e17035004aac62914997` |
| `timing_2d.log` | `2d33b63c7bf2ae6162afc6c368ae0160627d058ea0bf04446f53614e6abba13b` |
| `timing_3d.json` | `8eb5e4f506ca9c9864b297e02a372cbd1dccea795b1155fcf14066f7d1b1e871` |
| `timing_3d.log` | `02fdd2bcb822b72bbb94d18c6d28df0d0e33ee0a7de39c117021683c343bfe23` |

- `timing_2d.json`, `timing_3d.json` and their logs:
  `scripts/time_inference_vs_solve.py` on 32 validation instances of E1's 2D
  corpus with E1's label-free state of seed 0, and on 32 validation and 8
  fine-mesh instances of Phase-2b's 3D corpus with Phase-2b's label-free
  state of seed 0. Per instance: the surrogate's inference on the GPU (first
  and warm call) against, on the CPU, a direct solve and conjugate gradients
  from zero, from the prediction, and from zero until they are as accurate in
  the energy norm as the prediction or as the prediction rescaled by c* (a
  stop decided by the labels, so a lower bound), assembly and archive
  reading excluded.
- `fields/`: `scripts/export_fields.py` on Phase-2b's nine states of the
  1,024-instance budget (label-free, trained without labels; labels only;
  graph network; seeds 0-2): the per-load energies and error measures of the
  256 validation instances (`energies_val.npz`), the counts, checks and
  diagnostics (`fields.json`), and the three figure files (`fig5.npz`,
  `fig6.npz`, `fig7.npz`: validation instances 107 and 154, fine-set
  instance 77, chosen by the rules `fields.json` records).
- `status.txt` (the three steps, exit 0), `machine.txt`, `precheck.txt`
  (the GPU idle and no other run before the timing) and `pytest.log`.

`tests/test_cmame_timing_records.py` checks that:
- the files are these; the reports the steps read, which carry their
  configurations, are the committed records; and the states they evaluated
  are the runs' own: the label-free ones those the reports record
  (`d9_restart`), the supervised ones those of
  `records/wp9/phase2_supervised_states.json`;
- every solver solved every instance, and no conjugate-gradient solve fell
  back to the direct one; the direct solutions agree with the stored labels
  to round-off and the full conjugate-gradient ones to within 1e-8 (observed
  below 1e-11); every matching solve met its target; and the timed model
  reproduces its report's five per-instance arrays on every timed instance
  (the fine ones against the report's transfer arrays);
- the field export used all nine states and left none out, and its
  diagnostics lie in the runbook's ranges;
- its counts agree with the manuscript's numbers, which come from other
  records: instance-seed pairs whose mean relative energy gap exceeds 1
  (label-free 0, labels only 18, graph network 28 of 768) and label-free
  load cases with a relative energy gap above 1 (1 of 3,072);
- on every load case of every network and seed in `energies_val.npz`, the
  relative energy gap, computed from the energies, equals the squared
  relative energy-norm error, computed from the error (Lemma "Exactness");
  the energy is positive exactly when that error exceeds the zero field's,
  and the energies of two networks for the same load case rank them as
  their errors do (Corollary "Zero-field test and ranking"); the rescaling
  by c* never increases the gap and leaves no prediction worse than the zero
  field (Proposition "Energy-optimal amplitude"); and the volume-weighted
  von Mises error obeys the bound of Proposition "Energy gap and stress
  error";
- the per-instance means of these arrays reproduce the report's relative
  energy gaps, displacement errors and von Mises errors (the transformers'
  to round-off; the graph network's within the export's tolerance of 1e-3,
  observed median deviations below 1e-4 and largest 6.5e-4: its CUDA
  scatter reductions are not bitwise reproducible); the
  counts of `fields.json` follow from them; the figure rules, applied to the
  report's arrays, select the instances exported; and the two validation
  figures carry their instance's per-load arrays.

## The threads of the solvers: RUNBOOK_W9 Sec. 4 (11 October 2026)

`threads/` holds RUNBOOK_W9 Sec. 4 as it ran on the box on 11 October 2026,
copied byte for byte from `wp9_threads_return_20261011.tgz` (SHA-256
`cef9d17f23069ed5baa508559cd4dcaba54ab0609f07fe0dde9d0778d565ea81`, as the
operator sent it and as recomputed on receipt); branch wp9-pool commits the
same files as `records/wp9/threads/`. The cost table's timing above recorded
the threads of its environment (`OMP_NUM_THREADS` and `MKL_NUM_THREADS` 16)
but not the threads the solvers used. This return times the same direct solve
and conjugate gradients on the first 8 validation and 2 fine-mesh instances
of Phase-2b's corpus with the label-free state of seed 0, in the box's
environment, with one thread, and in the environment again, and reads the
BLAS pools that numpy and SciPy load (`scripts/blas_threads.py`). It ran on
another host than the cost table: a Xeon Platinum 8470Q with 25 CPUs
allotted and NVIDIA driver 580.95.05, where the environment sets the threads
to 25 (wp9-pool's `DEVIATIONS_W9.md`, W1 and W3).

| file | SHA-256 |
|---|---|
| `blas.txt` | `75e6e1c6c29958ba8de261e03bfcbd6734ebbb57e499031d3e4f3ed8b6ffa3d2` |
| `blas_env.log` | `36c5f0f861e8ec84fe9fc5f6a1c3d6e84fe7a4092cec46567ba64dc48b2af94e` |
| `blas_one.log` | `2857509aeb14a114e0183c5210417dfa6742a60fbac1927efa59b16268e0db5d` |
| `machine.txt` | `4436f8d70e59373d3b0790f18329fe26ea316d5933f516376971037d620fe128` |
| `precheck.txt` | `cc74ce67bac1b3053b4649d766e6e127f07255aca07ddc9632a9bdc35a528cf6` |
| `status.txt` | `7effaf50153a333b35fdc992387e8d6ede92695a9b34fd685f9a21d6d5225f57` |
| `timing_2d_env.json` | `5f77cac2c5bfc638c25f278b5beeac6aa1ff9591f39aa37094ef83e0b635eb08` |
| `timing_2d_env.log` | `8009fdbf4bc5e8b69912aac1ce3d4862b52a35f4434e7da71ab8ea325f0fa7d8` |
| `timing_2d_one.json` | `0e87c30cac6d58c2997124209de0b22459a55e5b22429b62fc87c07d9f5c5a49` |
| `timing_2d_one.log` | `93b171606edce5d0ed4856f6fc034f6856896385dd1c9d21d184b619b028b8a8` |
| `timing_3d_env.json` | `eaf238953e2cf7eb7ac127d24a6112f1b388979198ae52bccb107582e7d19c6f` |
| `timing_3d_env.log` | `44f67f1e7d44e26f9d861c326f9fc78f601b1d46a8e752797ed07db705feef0e` |
| `timing_3d_env2.json` | `38b4e98fd17c1cb864d228d9d8bb2c2cc27cf83e47eef3e2dd0870cd58bf43ef` |
| `timing_3d_env2.log` | `0370ff3fe199a27d5f7b25da74b2007d8ace6898338efe1a343df026143a3563` |
| `timing_3d_one.json` | `6d17ce7a1963303302d23423abb3d43e1660f478fe220a5f4a43b100207baa37` |
| `timing_3d_one.log` | `64a443e75405ba348121dbd33559a512eb100dfed6bf4a0db506a2d43d35d82f` |

The readings, which `scripts/make_cmame_material.py` turns into the cost
table's note on threads and `tests/test_cmame_material.py` recomputes: both
BLAS pools (two OpenBLAS builds) ran the environment's 25 threads, and 1 with
one thread; the direct solve took the same time with one thread as with 25;
conjugate gradients took less with one thread; the repeat in the environment
agreed with the first run. Every solution agreed with the stored labels, and
the timed model reproduced its report's arrays.

## B0 readiness (8 October 2026)

`cm2d_ready.log` (SHA-256 `d1ca946eee99f272e8924275329cff2634a454bf134fbdc265c304c57546a28e`):
RUNBOOK_CMAME B0 on the same commit, run after Sec. A by the operator's
account, as the operator sent it beside the tarball. Every check `ok` or
`info`, last line `GO`: E1's three states reproduced E1's report on CUDA
with largest relative deviation 0, one epoch of stiffness-norm training on
two pool instances gave a finite relative energy gap (0.995), and
PREREG_CM2D was unstamped, as it must be before the stamp. Checked by the
same test file.

## CM2D: the run of 8-9 October 2026 and its verdict (PREREG_CM2D)

`cm2d/return/` holds RUNBOOK_CMAME Sec. B as it ran on the box on the stamped
commit `094c804` (tag `prereg-cm2d`, tree
`cd3ec0c5baada05cf86eb6d12d64153d28930e10`), copied byte for byte from
`cm2d_return_20261009-1433.tgz` (SHA-256
`6cd07ef487cfab7c5d12f7d63a8f0d4887f535c495d2c04e444e29aa46ada2b6`, as the
operator sent it and as recomputed on receipt). The run started at
2026-10-08T15:24:06Z and exited 0 after one attempt; its supervised grid of
30 units took 12.8 h on three workers (16 CPUs allotted; RTX 5090, driver
595.71.05, torch 2.12.1+cu130). The suite there: 514 passed, 2 deselected.

| file | SHA-256 |
|---|---|
| `return/RESULTS.md` | `0afd779dd48c422def1cac3059ab588205b5f6deef34e8f0520e0a2419da320e` |
| `return/figure1_energy_gap.png` | `42274c7a2cc4e5a284ead107fda38a4c1ff2affe0124993bd3dccb659db24a0e` |
| `return/precheck.json.used` | `f69a8a177c8d6ffdf76679e72a3dab90fd2b8589e9f025e6071380b74b5da377` |
| `return/precheck.log.20261009-142644` | `7ed83d52a513f85d193c69926e77e1d058b26f9fdff44f4054f08a3480910e30` |
| `return/provenance.txt` | `cfc46f91c0828f21533a340ee7719462549a6e86e1e57b9640b49e203b6001ff` |
| `return/pytest.log` | `3c89394354f70ca22b45234465f944446a1d320368580095a414217622ae4a28` |
| `return/report.json` | `a72c007e1a612986af463084572f1794b574e9861cce233006534c51e1a6e5ec` |
| `return/run.log` | `75c403a76a8d6d1cdfb2ad26f6d691b85a1d80147af974cec5d86be161f06b16` |
| `return/status.txt` | `4687ec091e1ad57d59367ee8d7044d4c9742fc46630e1a396c9989cb997e5c11` |

- `report.json`, `RESULTS.md`, `figure1_energy_gap.png`: the run's outputs.
  `status.txt`: one start, exit 0. `run.log`: the run's console.
  `provenance.txt`: the commit, the tree, `git describe` (both forms
  `prereg-cm2d`), torch, the SHA-256 of the report and of the kept states
  (the states stay on the box), the GPU and the memory events (`oom_kill 0`);
  its `git status` lists one untracked file at the repository's root, the
  operator's manual of 3 September (already on the box at E1's run), which
  `git describe --dirty`, B1c and the precheck ignore by design. The
  container's hostname differs from E1's (`records/wp8/e1/provenance.txt`):
  the instance's host changed, as PREREG_CM2D Sec. 7 discloses; the GPU,
  driver and torch are E1's.
  `pytest.log`: the suite before the run. `precheck.json.used`: the GO the run
  used. `precheck.log.20261009-142644`: the log of that precheck (GO), renamed
  after the run (below).
- `cm2d/verdict.json` (SHA-256
  `008b26550dcde4880f5d1baf0e949c4d3909c7e8079807becf7a2f7d648b4db9`):
  `scripts/adjudicate_cm2d.py`, run on 9 October 2026 at the stamped commit
  in a clone of its own, with the tag fetched by a forced refspec and only
  `prereg-cm2d` pointing at that commit (RUNBOOK_CMAME Sec. C), as
  `python scripts/adjudicate_cm2d.py --return records/cmame/cm2d/return
  --e1-report records/wp8/e1/e1_2d_base/report.json --prereg PREREG_CM2D.md
  --july-report records/phase1/report_rec8_v2.json --out
  records/cmame/cm2d/verdict.json`. Run again on the same inputs, it writes
  the same file byte for byte. H1, H2a and H2b: SUPPORTED; H3: no difference
  shown (within the guard). The label-free row passed its reuse checks
  (largest relative deviation from E1's values 0); the adjudicator recorded
  no deviation and found its code identical to the stamped version.

Departures from the runbook's operations, recorded under PREREG_CM2D Sec. 8;
none touches a file the adjudicator reads or a criterion:
- B1b's direct fetch from GitHub timed out on the box. The branch and the tag
  were fetched from a git bundle of the stamped commit and its annotated tag,
  one of the two routes of the supplement to the operator's instruction of 8
  October (the other: AutoDL's network route); the operator's report,
  relayed on 9 October, recorded at Stage 7. The precheck
  (`precheck.json.used`) and the provenance file show the tag, the commit,
  the tree and a clean checkout.
- At 14:26:44 China Standard Time (06:26:44 UTC) on 9 October, after the run
  had ended (its report is dated 04:10:26 UTC), B2's line that sets an
  earlier precheck log aside ran again at the box's terminal and renamed
  `precheck.log`. No other line of B2 that writes a file ran: the return
  holds no new precheck log and no new GO file, and `precheck.json.used` is
  the GO the run used. Arrow keys had been pressed in the run's window during
  the run (a screenshot the operator sent shows them), so the line may have
  been recalled from the shell history; how it came to run is not recorded.

`tests/test_cm2d_records.py` checks that:
- the files are these, and the verdict's recorded inputs are these files,
  E1's committed report, the stamped PREREG_CM2D.md and July's report;
- the run was one attempt on the stamped commit and tree, exited 0, with
  the GPU idle and the suite green before it, no out-of-memory event, and
  the GO it used was the precheck of the stamped checkout;
- the report is the stamped configuration's (its canonical SHA-256 is
  PREREG_CM2D's CONFIG_SHA256[cm2d_v1]), on E1's corpus and seeds, with a
  solve ledger of 0 and no restart;
- the verdicts follow from the report's per-instance arrays by PREREG_CM2D
  Sec. 4's rule, recomputed in the test, and the label-free row reproduces
  E1's per-instance values; the secondary readings the manuscript may quote
  (the 80 comparisons, the label efficiency, the instance-seed pairs, the
  values worse than the zero field, the graph network's reading) follow from
  the same arrays, recomputed in the test;
- everything in `verdict.json` that does not depend on git (the hypotheses
  with their medians, Welch and resampling intervals, the reuse checks, all
  secondary readings, the deviations and the run block) is what
  `scripts/adjudicate_cm2d.py` computes in process from the committed
  files.

## CM2D's error spectra: the export of 9 October 2026 (RUNBOOK_CMAME Sec. D)

`spectra/` holds RUNBOOK_CMAME Sec. D as it ran on the box on Stage 7's
commit `8b5d443` (tree `e288619b0338a6fa497dfabba6c953e8b3a4e553`, clean),
copied byte for byte from `cm2d_spectra_return_20261010-0023.tgz` (SHA-256
`8c9a4c1f3830607579d3fd6d6f08aa62e2c76f223db8885074f0973dfe263b63`, as the
operator sent it and as recomputed on receipt; the name carries the box's
China Standard Time, 16:23 UTC on 9 October). The suite there: 535 passed, 2
deselected. The export exited 0; its timed part (the 256 instances, the
summaries and the figure file) took 70 s, of which 31 s went to the
eigendecompositions and 14 s to the second pass with TF32 off (RTX 5090,
driver 595.71.05, torch 2.12.1+cu130; 16 CPUs allotted). Post hoc: specified
after CM2D's verdict and reported only; no verdict reads it.

| file | SHA-256 |
|---|---|
| `export/fig2d.npz` | `1444977bceba4466b37a3b0f96b2eed78869826b499618e63ee24b349a7b4084` |
| `export/spectra.json` | `a494aa49c299a83f614d3b8e8bd3aaf81d6e3cf76181e66f6abbb7fff863dc85` |
| `export/spectra_val.npz` | `fed09ab55afbf2866427e54920ee7655df565a9704e6c5f098615e4f935116ac` |
| `machine.txt` | `adb55605b33d8c5569390244d1cf9c67c1204d94c256cf21b7126bc32ae03f9c` |
| `pytest.log` | `ad78a837322b7120785b0c73b829010fb8ba5a5efe47786a24ac00f03cfb5a37` |
| `spectra.log` | `26c1e419d8e806e16860686a828dc2722e35458cc63090b56cc108bc04f6a780` |
| `status.txt` | `fe4aefe67ea7930edbd135c51a60065bd5d4cbf7258a14bc67925de5a7da1b6b` |

- `export/`: `scripts/cm2d_spectra.py` on CM2D's kept states at 1,024 labels
  (L_D, row `labels`; L_K, row `labels_knorm`; the graph network, row `mgn`;
  seeds 0-2) and E1's three label-free states (row `ar`), on the run's 256
  validation instances: the per-load arrays (`spectra_val.npz`), the inputs
  and their hashes, the content checks, the counts and the summaries
  (`spectra.json`), and the figure instance chosen by the script's rule
  (`fig2d.npz`: validation index 221, `instance_27356.npz`, 547 nodes).
  Every state matched CM2D's provenance file, the label-free ones also the
  report's d9_restart record. Every transformer reproduced the report's five
  per-instance arrays exactly, the graph network with median relative
  deviations up to 1.7e-5 in the displacement error and 1.4e-4 in the energy
  gap, the two the export gates (up to 5.2e-4 in the peak von Mises error):
  its CUDA scatter reductions are not bitwise reproducible. The TF32 policy
  was in force after the second pass as before it.
- `status.txt` (exit 0), `machine.txt` (the commit, the tree, a clean
  checkout, the GPU and the CPU allotment), `spectra.log` (the export's
  console, ending with its summary line) and `pytest.log` (the suite before
  the export).

`tests/test_cm2d_spectra_records.py` checks that:
- the files are these; the export ran once on Stage 7's commit and tree,
  exited 0 after a green suite, kept the TF32 policy, and printed the summary
  of the JSON it wrote; the machine, the texts and the timings are those
  quoted here;
- its inputs were CM2D's committed report and provenance file, the stamped
  configuration, the twelve states the provenance file lists (the label-free
  ones also those of the report's d9_restart record) and the run's own
  validation split;
- every model reproduced the report's per-instance arrays (the export's
  content check, and the same comparison recomputed from the per-load
  arrays: the transformers' to 1e-12, and the graph network's recorded
  median and largest deviations);
- on every load case of every row and seed, Lemma "Exactness", Corollary
  "Zero-field test and ranking", Proposition "Energy-optimal amplitude"
  (with the gap of c* u from the stored energies) and the plane-stress
  identity and bound of Proposition "Energy gap and stress error" (with
  gamma* from the stored integrals) hold; the spectra sum to the norms (the
  errors' and the solution's in both binnings, the IEEE predictions' errors'
  and the rounding's in the log binning); in every occupied bin the
  stiffness-weighted mass over the Euclidean mass lies within the bin's
  eigenvalue range, and the outermost log bins are empty; every Rayleigh
  quotient (the errors', the IEEE predictions' errors', the rounding's and
  the solution's) lies between the extreme eigenvalues; and the rounding is
  of TF32's size on every prediction (the second pass was not the first
  again);
- the summaries, the counts and the eigenvalue ranges are what the script's
  functions compute from the committed arrays, and the readings the
  manuscript may quote follow from their definitions, written out again (per
  row the quotients, the shares above 10^k RQ*, the IEEE quotients and the
  rounding's shares; the solution's shares and quotient; per pair of rows
  the shares, the ratios of means and the geometric-mean ratios, overall,
  per seed and with TF32 off);
- end to end, the export's seed means of the relative energy gap and of the
  von Mises error are the verdict's, seed by seed, for H1, H2a, H2b and H3,
  and L_D's over L_K's ratios of means are H2a's and H2b's 4.596 and 2.215;
- the figure file is the instance the rule selects from the report and
  carries that instance's per-load arrays; with its stiffness matrix
  reassembled from its mesh and material by the generator's assembly, it has
  the stored eigenvalues, a U* that solves the stored loads, and every row's
  stored seed-0 coefficients, norms, energies, gaps, c*, spectra and von
  Mises stresses.

## The holdout sets: pool depths, exclusions and resampling (PREREG_HOLDOUT, 14 October 2026)

- `holdout/pool_depths.json`: `scripts/holdout_pool_depths.py` on the six
  branches of the repository as they stood on 14 October 2026 (the commits
  are listed under `scanned`), reading every JSON file of each from git
  objects. Fifteen reports: for each its corpus, split, the pool depth it
  records (`data_economy.unlabeled_pool_depth_used`; every one of the fifteen
  records it) and the depth its configuration implies
  (`fejepa.experiments.runner._pool_need`, from which the runner sets the
  recorded one, so that the two agree by construction). `history`: the scan
  of every JSON file reachable from the remote branches and tags it lists
  (`refs`), older commits included, finds the same fifteen reports.
  `other_reads`: the depths that
  the recorded tools outside the runs reached, from their records (E1's
  lambda pilot, the two-dimensional step timing, wp9's C0.6 memory audit and
  C0.4 reading, CM2D's precheck). The deepest prefix of `runs/data3d_phase2`
  is 1,024 and of `runs/data2d` 25,600.
- `holdout/overlap.json`: `scripts/holdout_overlap.py`. The replay of the
  draws of every three-dimensional set the repository records a generation
  of, without meshing (the cross-branch regression's miniature of
  `configs/phase2b_v1.json` and its fine set, run in the sandbox, and the
  geometries of the mesh-size calibration included), checked against the
  Poisson ratio, box and load scales of the field export's three figure
  instances; the sandbox's other sets of the two generators and seed 0,
  prefixes of the same streams (`sandbox`: `scripts/holdout_scan_sets.py`
  over /home/claude, /tmp and /mnt/user-data on 14 October 2026 at 05:37
  UTC; the script lists every set of the generators and seed per kind,
  copies of the corpus manifests, and stray instance files).
  Of `runs/data3d_phase2`, the regression's miniature drew instances 0 to
  79 again, and the fine-mesh set, drawn from the same seed, drew 28, 106
  and 294 of the tail and 198, 210 and 218 of the training pool (the fine
  evaluation instances 212, 224 and 232); the 34 tail instances among them
  (32 of 0 to 79, and 106 and 294) are excluded from h3d. Nine pairs share
  the box and the Poisson ratio only, with other cavities and load scales
  (`same_box_and_nu_only_3d`), and are not excluded. In two dimensions: the
  33 instances of the tail that wp9's load-scale measurement rebuilt (32 of
  the 256 it rebuilt; the regression's miniature of
  `configs/e1_2d_base.json`, instances 0 to 79, among them) or whose own
  file the authors' July 2026 archive holds (6, 9 and 10, rebuilt too, and
  29992), excluded from h2d (`records/wp9/scale_factor.json`); the backup
  archives of the whole corpus (`backups_2d`), from which no file is
  recorded as extracted outside the box, leave nothing out. `bounds`: a set
  of the generators and seed of at most 80 instances with a band of mesh
  sizes, 320 at one size or 256 in two dimensions shares with the tails only
  excluded instances (checked by the script), so that the regression's
  miniatures, drawn again whenever it runs, leave the sets as they are; the
  sandbox is scanned again before the stamp, which is not made unless every
  such set is within the bounds.
  PREREG_HOLDOUT's sets: 686 and 4,111 instances.
- `holdout/resample.json`: `scripts/holdout_resample.py`. The rules of
  PREREG_HOLDOUT Sec. 5 on 1,000 resamples of the validation instances of
  each holdout set's size, drawn directly and with the validation set's own
  sampling added, with the count of each claim's failures by the comparison
  that failed (`failing`); and the calibration of V on the validation
  instances alone (`V`: random halves of 128 and draws of the holdout sets'
  sizes, without a change and with the label-free transformer's four error
  metrics multiplied by 1.05 to 1.20 in both sets; and at the holdout sets'
  sizes each quantity alone made worse by 1.02 to 1.20). PREREG_HOLDOUT
  Sec. 5 quotes both.
- `tests/test_holdout_config.py` checks the listings against
  `configs/holdout_v1.json`, re-derives the pool depths and the other
  tools' depths from the listed commits when they are in the checkout
  (a checkout without them checks the record's consistency), re-derives
  the overlap record and its bounds, checks that the regression's
  miniatures are the sets it replays, and tests the scan script on sets of
  its own; `tests/test_holdout_rules.py` re-derives the resampling record.
