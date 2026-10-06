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
