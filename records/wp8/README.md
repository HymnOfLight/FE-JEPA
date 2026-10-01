# records/wp8 -- the E-series records: what the stamps rest on, and what the runs returned

All files are copied byte for byte from the box's return packages (host
`autodl-container-857b4c9493-2d948778`, RTX 5090, torch 2.12.1+cu130).

## Stamp records (Stage 1.34)

Returned on 29 September 2026 in `wp8_pilot_bench_return_20260930.tgz`
(SHA-256 `8ff3782853645297646314da5b5787a933e96959a84e9c8034c1ec7f12ded34d`;
code `cc9a168`, 281 tests passed there).

- `e1_pilot.json` -- the E1 lambda pilot (PREREG_E1 Sec. 3): lambda = 1.0, head width 6.
  Its SHA-256 is in PREREG_E1's parameter line (Sec. 2).
- `bench_e2_m512.json`, `bench_e2_m1024.json` -- the E2 bench (PREREG_E2 Sec. 3). Their
  SHA-256 are in PREREG_E2 Sec. 3; the E2 verdicts read these files.

`tests/test_e_series_records.py` keeps these files, the two pre-registrations and the
configurations in agreement.

## Run records (Stage 1.35)

Both experiments ran on the stamped commit `cdac731` (tags `prereg-e1`, `prereg-e2`;
286 tests passed on the box), one attempt each, no resumption, solve ledger 0.

- `e1/` -- from `wp8_e1_return_20260930.tgz` (SHA-256
  `7343dad3a98fe1a43343fca9fdf0c799dc05a79db5734e4e16f8f0fbcd42b7a0`): the three arms'
  reports and run logs (`e1_2d_base/`, `e1_2d_shaped/`, `e1_2d_raw_s0/`), the seven
  separation readings (`sep_*.json`), the verdict (`e1_verdict.json`: NO-GO) and the
  box provenance (`provenance.txt`, state SHA-256).
- `e2/` -- from `wp8_e2_return_20261001.tgz` (SHA-256
  `fabc65d377785ac485c163f66aef04284bab84d93d63edf9e8b4a30955b31e55`): the two
  bottleneck runs' reports and run logs (`e2_m512/`, `e2_m1024/`), the verdicts
  (`e2_verdict_M512.json`, `e2_verdict_M1024.json`: both KILLED by K1) and the box
  provenance.
- `e2/baseline/report_phase2b.json` -- the Phase-2b report E2 is judged against
  (SHA-256 `320b6db5060ecae9f4747327228c7705d30877bece67d09af2f68d6c466f2794`, as
  PREREG_E2 Sec. 2 records), copied from the Phase-2b return package of 28 September 2026.
- `restore_data2d.log` -- the restoration of the Phase-1 2D corpus from the data disk's
  archive before the pilot (manifest `3553396d...` unchanged).

`tests/test_e_series_verdicts.py` re-derives every verdict from these files with the
committed adjudicators, checks that each verdict was computed from exactly these files
(input SHA-256), and ties each report to its stamp line.

## Post-hoc records (Stage 1.38)

The post-hoc session (RUNBOOK_E_SERIES Sec. 4: inference and timing only, nothing
adjudicated) ran on 1 October 2026 (UTC) on the Stage 1.37 commit `0b3cd10` (tree
`9ae35919b1d22f5bd8c2398d000ad365be11fb65`; 318 tests passed there); all 12 steps exited 0.
Returned in `wp8_posthoc_return_20261002.tgz` (SHA-256
`b305be6a81956ab379f5585bb212c76e09868fb861adaaa51a6264ec6157c3c5`); `posthoc/` holds its
28 files:

- `probe_random_init.json` (4a): separation readings of untrained models on E1's
  validation split, three arms (descriptor input; its input weights zeroed; no descriptor
  input) x seeds 0-2.
- `profile_2d_head.json`, `profile_2d_v215.json` (4b): 2D step timing of this code (nine
  variants and a profiler table) and of the v2.1.5 code (four variants and a profiler
  table; a worktree at the tag, commit `a548825`).
- `id_phase2b_s{0,1,2}.json` (4c): intrinsic dimension of the Phase-2b AR states.
- `anat_{phase2b,e2_m512,e2_m1024}.json` (4d): error anatomy on 128 in-band and 32 fine
  instances.
- `amp_{phase2b,e2_m512,e2_m1024}.json` (4e): the energy-optimal amplitude on 256 in-band
  and 256 fine instances, and the remesh pass (16 geometries, each meshed at four lc).
- the step logs, `status.txt` (the 12 exit codes), `pytest.log`, `worktree.log` and
  `provenance.txt` (head, tree, the nine states' SHA-256, GPU, CPU count, cgroup memory
  events).

`tests/test_posthoc_records.py` checks the exit codes and the commit, ties every measured
state and report to the committed run records, and checks that the uncorrected amplitude
numbers reproduce the reports (the transformer's bitwise; the bottleneck's with per-seed
median deviations of the size its own run's two evaluations of the same states show, the
largest deviation 3.7x the run's own largest). The readings are summarised in
BRANCH_NOTES (Stage 1.38); the amplitude tables in `paper/wp8/` are generated from these
files.
