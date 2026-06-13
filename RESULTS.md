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

Source: `runs/battery.json` (`results.E*`, `gate_g1`).

| ID | Metric | Value | Killed? |
| --- | --- | --- | --- |
| E1 | max rel. improvement from anchor | `TBD` | `TBD` |
| E3 | effective rank (SIGReg on / off) | `TBD` / `TBD` | `TBD` |
| E5 | FE-JEPA vs naive (beats at any budget) | `TBD` | `TBD` |

| Gate G1 | Value |
| --- | --- |
| (a) E5 sanity passes | `TBD` |
| (b) component value ≥ 10% at decision budget | `TBD` |
| (c) pretraining beats from-scratch ≤ 256 labels | `TBD` |
| **Decision** | `TBD` (GO / NO-GO) |

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
