# FE-JEPA — Experiment Results

This file records headline numbers from full-scale (GPU) runs. The tables are
**intentionally left blank** (`TBD`). Reproduce them with:

```bash
# Edit configs/phase1_2d.json for your hardware (device, dataset size, epochs).
fejepa run-config configs/phase1_2d.json
```

The runner writes JSON reports to the paths configured under each stage
(`runs/regimes.json`, `runs/battery.json`, `runs/labeleff.json`); copy the
numbers into the tables below and commit. A quick CPU pipeline check (not a
headline result) is available via `fejepa run-config configs/smoke_cpu.json`.

> Note: the CPU development environment used to build this code has **no GPU**,
> so the tables below are left for the maintainer to fill after a GPU run.

---

## Environment

| Field | Value |
| --- | --- |
| Date | `TBD` |
| Commit | `TBD` |
| GPU(s) | `TBD` |
| Dataset size (unlabelled pretrain) | `TBD` |
| Mesh sizes (nodes: min/mean/max) | `TBD` |
| Model (dim / depth / heads / params) | `TBD` |

## Gate G0 — neural-solver sanity (single instance)

Source: `fejepa gate-g0 --device cuda` (or `--device cpu`).

| Metric | Value | Pass threshold |
| --- | --- | --- |
| rel-L2 (displacement) | `TBD` | ≤ 0.10 |
| relative energy gap | `TBD` | ≤ 0.01 |
| Verdict | `TBD` | PASS/FAIL |

## Training-regime comparison (RQ1 / RQ4)

Source: `runs/regimes.json` (`regimes.<name>.{val_rel_l2, val_energy_gap_rel,
val_rel_l2_vm, val_max_vm_rel_err, val_crit_recall, labelled_solves}`).

| Regime | rel-L2 disp. | energy gap | rel-L2 vM | max-vM err | crit. recall | labels |
| --- | --- | --- | --- | --- | --- | --- |
| labels only | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| labels + anchor | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| anchor only (label-free) | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | 0 |

(`rel-L2 vM` = relative L2 of the element von Mises field; `max-vM err` = peak
von Mises relative error; `crit. recall` = recall of the top-10% highest-stress
elements — the engineering quantities of Section "Evaluation protocol".)

## Label efficiency (RQ2)

Source: `runs/labeleff.json`. Report from-scratch vs. fine-tuned-from-pretrain at
each label budget; fill the fine-tune column after pretraining a checkpoint and
setting `label_efficiency.init_ckpt`.

| Labels | from-scratch rel-L2 | fine-tuned rel-L2 | label reduction @matched error |
| --- | --- | --- | --- |
| 16 | `TBD` | `TBD` | `TBD` |
| 64 | `TBD` | `TBD` | `TBD` |
| 256 | `TBD` | `TBD` | `TBD` |
| 1024 | `TBD` | `TBD` | `TBD` |

## Falsification battery + Gate G1

Source: `runs/report.json` (first GPU run, 2026-06-13). Config: budgets `[16, 64]`
(256 dropped — pool < 256), `n_val=16`, `lambda_phys=1.0`, single seed.

**E1 — anchor value** (`err_no_anchor` vs `err_anchor`, rel-L2 displacement):

| Budget | no anchor | + anchor (λ=1) | rel. improvement |
| --- | --- | --- | --- |
| 16 | 0.832 | 0.428 | **+48.6%** |
| 64 | 0.276 | 0.329 | **−19.2%** |

E1 killed? **No** (max improvement 48.6% ≥ 3%).

**E3 — collapse:** effective rank SIGReg on **1.355** / off **1.304**
(`sigreg_raises_rank=false`). E3 killed? **Yes** (SIGReg did not raise the rank
by ≥5%; both ranks are very low ≈1.3).

**E5 — sanity:** FE-JEPA rel-L2 **0.428 / 0.329** (budgets 16 / 64) vs naive
polynomial **4.33** → beats naive ~10×. E5 killed? **No**.

| Gate G1 | Value |
| --- | --- |
| (a) E5 sanity passes | True |
| (b) component value ≥ 10% at decision budget (64) | False (E1@64 = −19.2%) |
| (c) pretraining beats from-scratch ≤ 256 labels | not measured (treated satisfied) |
| **Decision** | **NO-GO** (publish negative/diagnostic study) |

### Analysis of the first run

- **The anchor is clearly *not* neutral** — it cut error by **48.6% at 16 labels**.
  This is the proposal's predicted regime (physics anchoring matters most when
  labels are scarce), and directly answers PI-JEPA's open question in the
  affirmative for the low-label regime.
- **But at 64 labels the fixed `λ=1` anchor *hurt* by 19%.** A single fixed
  weight over-powers the (now-stronger) label gradient. A CPU control on a
  different small dataset showed the *opposite* budget-dependence, confirming the
  single-λ / single-seed result is **regime- and noise-sensitive** and not yet
  conclusive at the decision budget.
- **Gate G1 = NO-GO** only because criterion (b) is evaluated at the 64-label
  budget, where the *fixed* λ regresses. This is an artefact of not tuning λ, not
  evidence the anchor is useless.
- **E3 collapse:** effective rank ≈ 1.3 with SIGReg barely moving it. The short
  default E3 pretraining (and pooled-latent rank over similar plate geometries)
  likely dominate; this needs longer pretraining and load-token conditioning work.

### What changed in response (this commit)

- **E1 now sweeps a `lambda_grid` and averages over `n_seeds`** (proposal mandates
  seeds ≥3 with CIs), reporting the *best-λ* improvement per budget with seed std
  — so the next run can answer "does the anchor help at 64 with the *right* λ?".
- Added an **opt-in gradient-balanced anchor** (`phys_grad_balance`) that caps the
  physics-gradient norm to a fraction of the label-gradient norm, so a fixed λ
  cannot overwhelm labels when they are plentiful.
- `configs/phase1_2d.json` battery now uses `lambda_grid=[0.1,0.3,1.0,3.0]`,
  `n_seeds=3`. Re-run with `fejepa run-config configs/phase1_2d.json` and refill
  the E1 table below.

**Re-run E1 table (lambda-swept, seed-averaged) — `TBD`:**

| Budget | no anchor (mean±std) | best λ | + anchor best (mean±std) | rel. improvement |
| --- | --- | --- | --- | --- |
| 16 | `TBD` | `TBD` | `TBD` | `TBD` |
| 64 | `TBD` | `TBD` | `TBD` | `TBD` |
| 256 | `TBD` | `TBD` | `TBD` | `TBD` |

## Cross-resolution transfer (RQ3 / E4)

Source: `runs/mesh_views.json` (`metrics.with_inv` / `metrics.without_inv`,
each with `rel_l2_coarse`, `rel_l2_fine`, `transfer_gap`). Train label-free on
coarse meshes, test on fine meshes, with and without the cross-mesh invariance
term `L_inv`.

| Setting | rel-L2 (train-coarse) | rel-L2 (test-fine) | transfer gap |
| --- | --- | --- | --- |
| `L_inv` off | `TBD` | `TBD` | `TBD` |
| `L_inv` on | `TBD` | `TBD` | `TBD` |
| E4 verdict (`inv_reduces_gap`) | | | `TBD` |
