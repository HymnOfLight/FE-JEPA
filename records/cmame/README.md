# records/cmame -- cmame-paper records

## Default path against wp8-lejepa (Stage 2)

Stage 2 changed `src` (opt-in additions only: the stiffness-norm loss of the
supervised trainer, its E8 row, an E8 run that trains a supervised grid beside
reused label-free states, and their step count). `regression_wp8_e1_2d_base.json`
and `regression_wp8_phase2b.json` are the summaries
`scripts/regress_against_branch.py` wrote in the sandbox on 6 October 2026
(CPU, one thread, each side in its own process importing its own source tree;
a fresh pass and a restart pass), on the same two miniatures as
`records/wp9/`:

| | source tree (`git rev-parse <commit>:src`) |
|---|---|
| this side | the Stage 2 code: `src` = tree `2da09d0593669f50149f4fb546c01861ac0d5149` |
| the other side | `wp8-lejepa` at `5f8e2df`: `src` = tree `5be6357b912cad058121092f274e6a322169b9f6` |

- `regression_wp8_e1_2d_base.json`: a miniature of `configs/e1_2d_base.json`
  (E1's stamped base) -- 162 numbers and 2 state files, identical (0
  differences; every state file byte-identical).
- `regression_wp8_phase2b.json`: a miniature of `configs/phase2b_v1.json` (E8
  with labels, labels_anchor, AR and MGN; P3; WP6; E6; gate G2) -- 2,590
  numbers and 6 state files, identical.

The two summaries equal `records/wp9/`'s except for this side's tree (wp9's
ran the Stage 0c code, tree `710ccb32...`). `tests/test_w9_records.py` checks
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
reference at July's labels-only spread of 9% (`july_reference`), and with
per-instance blow-ups (one validation instance at 20-40 times an arm's level,
about one per seed) in one arm or both (`blowups`). `schedule`: the 30 units
of `configs/cm2d_v1.json` in the order the run submits them on three workers,
at the measured 52 ms per supervised step plus a minute per unit, for the
graph network's (unmeasured) step at 0.5-2 times the transformer's. The
script regenerates the file byte for byte (`--check`; tested).

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

## B0 readiness (8 October 2026)

`cm2d_ready.log` (SHA-256 `d1ca946eee99f272e8924275329cff2634a454bf134fbdc265c304c57546a28e`):
RUNBOOK_CMAME B0 on the same commit, run after Sec. A by the operator's
account, as the operator sent it beside the tarball. Every check `ok` or
`info`, last line `GO`: E1's three states reproduced E1's report on CUDA
with largest relative deviation 0, one epoch of stiffness-norm training on
two pool instances gave a finite relative energy gap (0.995), and
PREREG_CM2D was unstamped, as it must be before the stamp. Checked by the
same test file.
