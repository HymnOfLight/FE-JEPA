# FE-JEPA v2.0 -- RESULTS

Run `49574ab0f008` | 2026-10-10T15:20:34+00:00 | git `prereg-w9` | device=cuda tf32=True | workers=3
Pre-registration: verified `49574ab0f008` against `PREREG_W9.md`

## Gate G1'
**Verdict: NO-GO** (decision budget 64)
- (a) FAIL: E5' not run -- condition unmeasured, gate fails closed
- (b) FAIL: labels / labels_anchor cell missing at decision budget
- (c) FAIL: ar_ft cell missing at decision budget

## Kill conditions
| exp | condition | triggered | note |
|---|---|---|---|
| e8 | K2: not evaluated (ar_only run: no labels-only cells) | no | AR 0.0615 |
| e8 | C1-advantage: not evaluated (ar_only run) | no |  |

## E1' -- anchor as supervised auxiliary (P-A)
not run

## E5' -- repaired sanity
not run

## E8 -- regime grid (headline table)
**Displacement rel-L2**
| regime |  |
|---|

**Relative energy gap**
| regime |  |
|---|

AR (unlabeled pool 1024, 0 labels): disp 0.0615±0.0005, egap 0.0372±0.0001, vM 0.1213±0.0002
Label-efficiency AUC (disp): 

## E2 -- JEPA vs AR (P-B, one-shot verdict)
not run

## E3' -- collapse (standardized rank)
not run

## E4' -- cross-resolution invariance
not run

## E6 -- latent-physics alignment
not run

## E7 -- learned init + CG polish
not run

## WP2 -- region-mask ratio sweep (pre-E2)
not run

## WP6 -- theory numeric falsification pass
not run

## Data economy (WP5)
- labelled instances: 256 (val 256 + pool prefix 0), 4 solves each
- reference solves total: 0
- unlabeled pool depth used: 1024 (x1024.0 the labelled prefix)

## Solve ledger
- asis-preexisting-corpus: 0
- total: 0 (wall 0.0 s)

## Provenance
- runs/data2d: n=30000, backend=gmsh, manifest `3553396d1831`
