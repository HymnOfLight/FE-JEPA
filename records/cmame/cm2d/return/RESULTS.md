# FE-JEPA v2.0 -- RESULTS

Run `bf1f1143b397` | 2026-10-09T04:10:26+00:00 | git `prereg-cm2d` | device=cuda tf32=True | workers=3
Pre-registration: verified `bf1f1143b397` against `PREREG_CM2D.md`

## Gate G1'
**Verdict: NO-GO** (decision budget 64)
- (a) FAIL: E5' not run -- condition unmeasured, gate fails closed
- (b) FAIL: labels / labels_anchor cell missing at decision budget
- (c) FAIL: ar_ft cell missing at decision budget

## Kill conditions
| exp | condition | triggered | note |
|---|---|---|---|
| e8 | K2: AR (pool 1024) displacement > 30% worse than labels-only @ 1024 over >= 3 seeds | no | AR 0.0634 vs labels 0.0505 (+25.6%) |
| e8 | C1-advantage: AR energy-gap advantage < 40% at every budget | no | {16: 0.998, 64: 0.987, 256: 0.911, 1024: 0.713} |

## E1' -- anchor as supervised auxiliary (P-A)
not run

## E5' -- repaired sanity
not run

## E8 -- regime grid (headline table)
**Displacement rel-L2**
| regime | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| labels | 0.4587±0.0452 | 0.1647±0.0152 | 0.0712±0.0071 | 0.0505±0.0018 |
| labels_knorm | 0.3622±0.0684 | 0.0889±0.0043 | 0.0667±0.0015 | 0.0547±0.0077 |
| mgn | -- | 0.3703±0.1246 | -- | 0.0499±0.0002 |
| zero | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 |
| scale_aware_poly | 1.6562±0.0000 | 1.3679±0.0000 | 1.4121±0.0000 | 1.4219±0.0000 |
| knn_field | 0.6867±0.0000 | 0.4660±0.0000 | 0.4077±0.0000 | 0.4038±0.0000 |

**Relative energy gap**
| regime | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| labels | 20.8218±8.5323 | 3.0575±0.6594 | 0.4300±0.0936 | 0.1334±0.0116 |
| labels_knorm | 0.2539±0.0667 | 0.0505±0.0058 | 0.0383±0.0003 | 0.0290±0.0102 |
| mgn | -- | 148.6319±113.5723 | -- | 0.8250±0.0163 |
| zero | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 |
| scale_aware_poly | 5096.2611±0.0000 | 408.1311±0.0000 | 773.7779±0.0000 | 873.0812±0.0000 |
| knn_field | 6.5028±0.0000 | 1.6462±0.0000 | 1.0272±0.0000 | 0.9591±0.0000 |

AR (unlabeled pool 1024, 0 labels): disp 0.0634±0.0025, egap 0.0383±0.0015, vM 0.1227±0.0020
Label-efficiency AUC (disp): labels=0.1635, labels_knorm=0.1214, mgn=0.2101, zero=1.0000, scale_aware_poly=1.4397, knn_field=0.4730

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
- labelled instances: 1280 (val 256 + pool prefix 1024), 4 solves each
- reference solves total: 0
- unlabeled pool depth used: 1024 (x1.0 the labelled prefix)

## Solve ledger
- asis-preexisting-corpus: 0
- total: 0 (wall 0.0 s)

## Provenance
- runs/data2d: n=30000, backend=gmsh, manifest `3553396d1831`
