# records/wp9 -- wp9-pool records

## Default path against wp8-lejepa (PREREG_W9 Sec. 9)

`regression_wp8_e1_2d_base.json` and `regression_wp8_phase2b.json` are the
summaries `scripts/regress_against_branch.py` wrote in the sandbox on 2 October
2026 (CPU, one thread, each side in its own process importing its own source
tree; a fresh pass and a restart pass):

| | source tree (`git rev-parse <commit>:src`) |
|---|---|
| this side | the Stage 0b commit that adds these files: `src` = tree `f300ac0c27ecaec63bca9fb0b32b2e1d4f124226` |
| the other side | `wp8-lejepa` at `5f8e2df`: `src` = tree `5be6357b912cad058121092f274e6a322169b9f6` |

- `regression_wp8_e1_2d_base.json`: a miniature of `configs/e1_2d_base.json`
  (E1's stamped base) -- 162 numbers and 2 state files, identical
  (0 differences; every state file byte-identical).
- `regression_wp8_phase2b.json`: a miniature of `configs/phase2b_v1.json`
  (E8 with labels, labels_anchor, AR and MGN; P3; WP6; E6; gate G2) -- 2,590
  numbers and 6 state files, identical.

The paths inside the summaries are the sandbox's. The miniatures run their
units inline (workers = 1); the process-pool path is covered by the
parallel-equals-inline tests (`tests/test_experiments_smoke.py::
test_e1_parallel_matches_serial`, `tests/test_w9_parallel_kill.py::
test_e8_ar_units_in_parallel_match_serial`). Bitwise identity is a
same-machine, same-library statement (sandbox: Python 3.11.15, torch
2.14.0+cu130 on CPU, numpy 2.4.4, scipy 1.17.1). This is what licenses
evaluating E1's base states as C1's 1,024-instance arm instead of retraining
them.

## Later

The adjudication of PREREG_W9 (`scripts/adjudicate_w9.py`) writes
`w9_verdict.json` here from the two box returns.
