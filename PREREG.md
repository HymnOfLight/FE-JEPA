# PRE-REGISTRATION v2.0 (plan Sec.5) -- commit + `git tag prereg-v2.0` BEFORE the deciding run

Deciding run: `configs/phase1_rec8_v2.json`, executed once. Post-hoc criterion changes
are prohibited. Both publication branches are pre-committed as publishable.

## 1. Retirement, with evidence
The v1.0 criterion "displacement improvement >= 10% at 64 labels" is RETIRED as already
answered in the negative at proper scale (+1.08%, t=1.2, 3 seeds, n_val=256; June-17
record). It remains a reported number (`E1'.retired_criterion_report`), not a gate.

## 2. Gate G1' (GO to Phase 2 3D/SimJEB) iff ALL of:
(a) repaired sanity: >= 3x over the zero predictor on displacement AND beats every
    repaired naive baseline, at every budget (E5');
(b) physics value at the decision budget (64 labels): labels+anchor achieves relative
    energy-gap reduction >= 50% vs labels-only, with vM rel-L2 reduction >= 25% as
    co-primary (E8 cells);
(c) measured transfer: AR-pretrain->fine-tune at 64 labels beats scratch by >= 10% on
    the energy gap OR >= 5% on displacement (E8 cells; measured, never assumed).

## 3. Statistics floor
n_val >= 256; >= 3 seeds; per-seed AND per-instance arrays persisted in the report;
t-statistics computed from stored arrays.

## 4. Lambda policy
Pre-registered anchored configuration: gradient-balanced anchor (ratio 1.0). Fixed
lambda=1 reported alongside. The lambda grid is secondary; any grid-best number carries
`selection_bias: true`.

## 5. Provenance
Every report embeds git describe, config SHA-256, dataset manifest SHA-256s, seeds,
versions, timestamp, and the solve ledger. Reports without it are void (the writer
refuses).

## 6. E2 verdict timing
E2 runs only after the WP2 redesign (region masking + cross-attention predictor +
pooled regularizer -- shipped in this package); its one-shot verdict selects the paper
branch (A: full FE-JEPA; B: "the anchor, not the SSL") and cannot be re-litigated.
E2 does NOT gate scale-up.

## 7. Kill conditions (frozen)
K1 (E1'): energy-gap improvement < 25% at every budget.
K2 (E8): AR displacement > 30% worse than labels-only at the largest budget.
K3 (E2): JEPA within 3% of AR everywhere (disp AND egap).
K4 (E3'): best standardized-rank ratio (on/off) <= 1.5.
K5 (E7): < 20% CG-iteration savings vs zero init.
K6 (E4'): transfer-gap reduction < 10% at the largest coarsening.
C1-advantage (E8): AR energy-gap advantage < 40% at every budget.

Signed-off config hash: run `python -c "from fejepa.report import config_sha256; import json; print(config_sha256(json.load(open('configs/phase1_rec8_v2.json'))))"` and paste here before tagging:

    CONFIG_SHA256 = <fill before tagging>
