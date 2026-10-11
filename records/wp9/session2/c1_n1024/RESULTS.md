# FE-JEPA v2.0 -- RESULTS

Run `acccf57ec50e` | 2026-10-10T10:51:39+00:00 | git `prereg-w9` | device=cuda tf32=True | workers=3
Pre-registration: verified `acccf57ec50e` against `PREREG_W9.md`

## Gate G1'
**Verdict: NO-GO** (decision budget 64)
- (a) FAIL: E5' not run -- condition unmeasured, gate fails closed
- (b) FAIL: labels / labels_anchor cell missing at decision budget
- (c) FAIL: ar_ft cell missing at decision budget

## Kill conditions
| exp | condition | triggered | note |
|---|---|---|---|
| e8 | K2: not evaluated (ar_only run: no labels-only cells) | no | AR 0.0634 |
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

AR (unlabeled pool 1024, 0 labels): disp 0.0634±0.0025, egap 0.0383±0.0015, vM 0.1227±0.0020
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
