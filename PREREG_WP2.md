# PRE-REGISTRATION WP2/E2 -- commit + `git tag prereg-wp2-e2` BEFORE the run

Run: `configs/wp2_e2_v2.json`, executed once. Post-hoc criterion changes are
prohibited. Both publication branches are pre-committed as publishable
(PREREG.md Sec.6: A = full FE-JEPA; B = "the anchor, not the SSL").

## 1. Question
Does the JEPA (SSL) term add measurable value over pure AR (energy-anchor)
pretraining under identical conditions? The 2026-07-16 deciding run
(Gate G1' = GO; config `62b26ad8...`) showed AR alone reaches displacement
parity with 1024-label supervision (0.1658 vs 0.1600) with a 5.1x better
energy gap; E2 is the pre-committed one-shot verdict on the SSL term
(PREREG.md Sec.6). E2 does NOT gate scale-up.

## 2. Decision rule (frozen; K3 verbatim from PREREG.md Sec.7)
K3 (E2): JEPA within 3% of AR everywhere (disp AND egap).
K3 triggered     => drop the SSL term; publication branch B.
K3 not triggered => the SSL term stands where it wins; publication branch A.
The verdict is taken at the E2 cells as written and cannot be re-litigated.

## 3. Statistics floor (as PREREG.md Sec.3)
n_val >= 256; >= 3 seeds; per-seed AND per-instance arrays persisted in the
report; comparisons computed from stored arrays.

## 4. Conditions held identical across arms
Same corpus (`runs/data2d`, manifest-pinned), same split (n_val 256, seed 1),
same model config, same pool_size 1024, same pre_epochs 200 / ft_epochs 200,
same seeds {0, 1, 2}, same device policy. The WP2 masking sweep
(ratios 0.2/0.4/0.6, 2000 steps) runs first as diagnostic context and carries
no verdict.

## 5. Provenance
As PREREG.md Sec.5. Additionally, this run SHOULD execute inside a git
checkout so the report's `git` field resolves (deciding-run lesson,
Finding F1 of the 2026-07-16 run report).

Signed-off config hash: run
`fejepa prereg configs/wp2_e2_v2.json --stamp --prereg-file PREREG_WP2.md`
and commit + tag before executing:

    CONFIG_SHA256 = 60088135b1fecb2768d882ad6fee37add0fc3463160decee98a34b511159cb2c
