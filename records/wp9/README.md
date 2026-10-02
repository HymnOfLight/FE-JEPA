# records/wp9 -- wp9-pool records

## Default path against wp8-lejepa (PREREG_W9 Sec. 9)

`regression_wp8_e1_2d_base.json` and `regression_wp8_phase2b.json` are the
summaries `scripts/regress_against_branch.py` wrote in the sandbox on 2 October
2026 (CPU, one thread, each side in its own process importing its own source
tree; a fresh pass and a restart pass), last on the Stage 0c code (the Stage
0b code, `src` tree `f300ac0c27ecaec63bca9fb0b32b2e1d4f124226`, gave the same
result):

| | source tree (`git rev-parse <commit>:src`) |
|---|---|
| this side | the Stage 0c code: `src` = tree `710ccb32a703bb99355d0e797895698fa2155300` |
| the other side | `wp8-lejepa` at `5f8e2df`: `src` = tree `5be6357b912cad058121092f274e6a322169b9f6` |

`tests/test_w9_records.py` checks that a committed `src` without local
changes is the tree named here: a change to `src` needs the regression run
again and this table updated.

- `regression_wp8_e1_2d_base.json`: a miniature of `configs/e1_2d_base.json`
  (E1's stamped base) -- 162 numbers and 2 state files, identical
  (0 differences; every state file byte-identical).
- `regression_wp8_phase2b.json`: a miniature of `configs/phase2b_v1.json`
  (E8 with labels, labels_anchor, AR and MGN; P3; WP6; E6; gate G2) -- 2,590
  numbers and 6 state files, identical.

Each summary names the two sides by these trees (`this_src_tree`,
`other_src_tree`: git tree ids computed from the files by
`regress_against_branch.src_tree`; Stage 0b's copies named the sandbox's paths
instead). The miniatures run their units inline (workers = 1); the
process-pool path is covered by the parallel-equals-inline tests
(`tests/test_experiments_smoke.py::test_e1_parallel_matches_serial`,
`tests/test_w9_parallel_kill.py::test_e8_ar_units_in_parallel_match_serial`).
Bitwise identity is a same-machine, same-library statement (sandbox: Python
3.11.15, torch 2.14.0+cu130 on CPU, numpy 2.4.4, scipy 1.17.1). This is what
supports evaluating E1's base states as C1's 1,024-instance arm instead of
retraining them.

## S's factor 1/64 (PREREG_W9 Sec. 2)

`scale_factor.json`: `scripts/w9_scale_factor.py` on the first 256 instances
of E1's training corpus (the generator's draws for seed 0, rebuilt with gmsh;
assembly only, no labels, no training), 2 October 2026. Per instance, the
battery's largest nodal force component max|F| (E1's decode scale) over the
sum of the absolute values of all its components sum|F| (S's): median 0.0151
(10th-90th percentile 0.0096-0.0219, i.e. 0.61-1.40 times 1/64), rank
correlation 0.67 with the instance's mesh size. 16 of these geometries meshed
again at R's mesh sizes (geometry, material and loads unchanged): from h =
0.12 to 0.025, max|F| scales by 0.213 (median over the geometries; range
0.204-0.225; h itself by 0.208), sum|F| by 1.000 (largest deviation 0.29%).

## Operating characteristics (PREREG_W9 Sec. 6-7)

`simulations.json`: `scripts/w9_sims.py` (fixed seeds; seconds on a laptop),
the source of every simulated number PREREG_W9 r3 quotes: H1 under r2's
log-normal model; H1 on E1's measured per-instance structure (E1's own
validation arrays, instance 233's blow-up included, as the reference's
validation part; per instance-seed jitter, seed noise and blow-ups elsewhere)
for the validation split alone and with IB; H2's three conditions; the
residual selection of rule 1 on F5's instances; and how often an exploratory
comparison without a true difference crosses the guard.

## The evaluation sets at full size (PREREG_W9 Sec. 3)

`ood2d_sandbox.json`: RUNBOOK_W9 step 1a (`scripts/w9_make_ood2d.py` with its
default sizes and seeds) run in the sandbox on 2 October 2026 on the Stage 0c
code: all seven sets generated and verified, no failed draw -- F1-F5 256
instances each, R 80 (16 geometries x 5 mesh sizes), IB 2,048; 3,408
instances, 5.5 min on one CPU, 0.83 GB (gmsh 4.15.2). The record session 1
writes, without the machine paths and the sandbox's git string. The instance
files are byte-reproducible on one machine (a draw regenerated hours later
was identical); session 1's manifests on the box can be compared with these
SHA-256 (equal with the same gmsh and numeric stack, else they show that the
meshes differ; either way the sets are the committed generator's draws for
these seeds, pinned by the box's own manifests).

## CPU pilot of S (Stage 0c, before the stamp)

`pilot_s_cpu/`: `scripts/w9_pilot_s.py` at toy scale on its own draws; its
README holds the pre-declared questions and red flags, what ran, and the
readings. Exploratory: nothing in PREREG_W9 depends on it.

## Phase-2's supervised states (Stage 0e)

`phase2_supervised_states.json`: the SHA-256 of the labels-only and
graph-network states at the largest budget (`labels_b1024_s{0,1,2}.pt`,
`mgn_b1024_s{0,1,2}.pt` in `runs/phase2/e8_states`), copied from
`e8_states_sha256.txt` in the Phase-2 return package of 22 September 2026
(the package's SHA-256 as received is recorded with them). Phase-2b reused
these units (`d9_restart.sup_units_from_cache`) and read these files as its
P3 shared checkpoints, but no report records their hashes. The field export
(`scripts/export_fields.py`) checks the states against this list and,
independently, by content against Phase-2b's per-instance arrays.

## Later

The adjudication of PREREG_W9 (`scripts/adjudicate_w9.py`) writes
`w9_verdict.json` here from the two box returns;
`scripts/make_w9_paper_material.py` turns it into `paper/wp9/`.
