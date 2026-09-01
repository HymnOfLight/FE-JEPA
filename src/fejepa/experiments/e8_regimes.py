"""E8 -- the regime grid, the paper's spine (plan Sec.6 E8).

Rows: labels / labels+anchor (gradient-balanced) / AR / AR->FT, an optional trained
MeshGraphNets column, and -- because E8 IS the headline table -- the plan Sec.4
mandatory naive rows: zero predictor, scale-aware poly, k-NN field, each fitted on
the same labelled pool prefix the supervised arms buy (deterministic, seedless).
Columns: label budgets {16, 64, 256, 1024}; AR's axis is *unlabeled pool size*,
reported as such and never conflated with the label budget (plan Sec.6 bottom).
Cells: the full frozen metric suite, >= 3 seeds, per-seed and per-instance arrays
persisted (plan Sec.5 item 3).

This experiment turns the June-17 single point (1024, seed 1) into the replicated
curve behind claim C1, and feeds gate conditions (b) and (c).

Execution: two unit phases through :mod:`~fejepa.experiments.parallel` --
(A) AR pretrainings (state dicts saved under ``cfg["state_dir"]``, evaluated on val),
(B) the supervised grid (AR->FT units load the phase-A states from disk).
``cfg["workers"]`` > 1 runs units concurrently with results identical to serial.

Kills:
  K2: "AR displacement > 30% worse than labels-only at the largest budget over
      >= 3 seeds" => drop C1's parity half.
  C1-advantage clause: "AR's energy-gap advantage < 40% at every budget" => drop
      C1's faithfulness half.
"""

from __future__ import annotations

from pathlib import Path

from ..baselines import (KNNFieldBaseline, ScaleAwarePolyBaseline,
                         zero_predictor)
from ..metrics import FIELD_KEYS, evaluate_model, label_efficiency_auc
from .parallel import cached_supervised_unit, map_units, pretrain_unit

POLICY_BALANCED_FROM = 64   # r8 Sec.6: fixed lambda=1 strictly below this budget


def _anchor_kw(budget: int) -> dict:
    """PREREG_PHASE2 r8 Sec.6 lambda policy: fixed lambda=1 below the decision
    budget (the measured stable low-budget carrier), balanced (ratio 1.0) at
    and above it."""
    if budget < POLICY_BALANCED_FROM:
        return dict(anchor_mode="fixed", lambda_phys=1.0)
    return dict(anchor_mode="balanced", balance_ratio=1.0)
from .protocol import (divergence_flags, kill, load_archs, mean_std, result,
                       seeds_list)

PLAN_REF = "plan v2.0 Sec.6 E8, C1, gate G1'(b,c), kills K2 + C1-advantage"


def _agg(seed_evals: list[dict]) -> dict:
    out = {k: mean_std([e[k] for e in seed_evals]) for k in FIELD_KEYS}
    out["per_seed_eval"] = seed_evals            # includes per-instance arrays (B6)
    out["divergence_flags"] = divergence_flags(seed_evals)   # r8 Sec.5; no exclusion
    return out


def naive_baseline_cells(pool_archs, val_archs, budgets) -> dict:
    """Plan Sec.4: 'baselines every headline table must contain' -- the zero
    predictor plus the repaired naives, as E8 rows.

    Deterministic and seedless: each cell wraps the single evaluation in the same
    ``_agg`` shape as the trained regimes (std 0, per_seed of length 1) so tables,
    AUC, and downstream readers treat every row uniformly. Fits consume only the
    labelled pool prefix the supervised arms buy (WP5 economy). The zero row is
    budget-independent; it is replicated per budget for table uniformity."""
    zero_cell = _agg([evaluate_model(zero_predictor, val_archs)])
    cells = {"zero": {b: zero_cell for b in budgets}}
    for name, cls in (("scale_aware_poly", ScaleAwarePolyBaseline),
                      ("knn_field", KNNFieldBaseline)):
        cells[name] = {b: _agg([evaluate_model(cls().fit(pool_archs[:b]).predict,
                                               val_archs)])
                       for b in budgets}
    return cells


def run_e8(model_cfg: dict, pool_files, val_files, cfg: dict) -> dict:
    budgets = [int(b) for b in cfg.get("budgets", [16, 64, 256, 1024])]
    pool_sizes = [int(p) for p in cfg.get("pool_sizes", [1024])]
    seeds = seeds_list(cfg.get("seeds", 3))
    device = cfg.get("device", "cpu")
    workers = int(cfg.get("workers", 1))
    tf32 = bool(cfg.get("tf32", True))
    state_dir = Path(cfg.get("state_dir", "runs/e8_states"))
    sup = dict(epochs=int(cfg.get("sup_epochs", 200)),
               lr=float(cfg.get("sup_lr", 1.5e-3)), device=device)
    pre = dict(epochs=int(cfg.get("ar_epochs", 100)),
               lr=float(cfg.get("ar_lr", 1e-3)), device=device)
    include_mgn = bool(cfg.get("include_mgn", False))
    include_ar_ft = bool(cfg.get("include_ar_ft", True))
    mgn_budgets = set(int(b) for b in cfg.get("mgn_budgets", budgets))
    reuse = bool(cfg.get("reuse_states", False))        # D9 restart mode
    cache_dir = str(state_dir / "unit_cache")   # always written; read only on reuse
    ft_pool = pool_sizes[0]

    if len(pool_files) < max(max(pool_sizes), max(budgets)):
        raise ValueError(f"E8: pool has {len(pool_files)} archives; "
                         f"needs {max(max(pool_sizes), max(budgets))}")
    val_str = [str(f) for f in val_files]
    regimes = (["labels", "labels_anchor"]
               + (["ar_ft"] if include_ar_ft else [])
               + (["mgn"] if include_mgn else []))

    # ---- phase A: AR pretrainings (states to disk, evaluated on val) ----------
    pre_keys, pre_payloads = [], []
    for s in seeds:
        for p in pool_sizes:
            pre_keys.append((s, p))
            pre_payloads.append({
                "kind": "fejepa", "model": model_cfg, "seed": s, "tf32": tf32,
                "compile": cfg.get("compile", False),
                    "precision": cfg.get("precision", "fp32"),
                "files": [str(f) for f in pool_files[:p]], "loss": "ar",
                "pre": dict(pre, desc=f"E8 AR pool{p} s{s}"),
                "state_path": str(state_dir / f"ar_p{p}_s{s}.pt"),
                "reuse_existing": reuse,
                "eval_val_files": val_str, "tag": f"AR pool{p} s{s}",
            })
    pre_out = dict(zip(pre_keys,
                       map_units(pretrain_unit, pre_payloads, workers,
                                 "E8 (AR pretrain)"), strict=True))
    raw = {"ar": {p: [pre_out[(s, p)]["val"] for s in seeds] for p in pool_sizes}}

    # ---- phase B: the supervised grid -----------------------------------------
    sup_keys, sup_payloads = [], []
    arm_kw = {"labels": dict(anchor_mode="none"),
              "ar_ft": dict(anchor_mode="none"),
              "mgn": dict(anchor_mode="none")}
    for s in seeds:
        for b in budgets:
            train_str = [str(f) for f in pool_files[:b]]
            for r in regimes:
                if r == "mgn" and int(b) not in mgn_budgets:
                    continue   # r10 L4: comparator trains on a budget subset
                sup_keys.append((r, b, s))
                sup_payloads.append({
                    "kind": "mgn" if r == "mgn" else "fejepa",
                    "model": model_cfg, "seed": s, "tf32": tf32,
                    "compile": cfg.get("compile", False),
                    "precision": cfg.get("precision", "fp32"),
                    "train_files": train_str, "val_files": val_str,
                    "sup": dict(sup, **(arm_kw[r] if r != "labels_anchor"
                                        else _anchor_kw(b)),
                                desc=f"E8 {r} b{b} s{s}"),
                    "pretrained_path": (pre_out[(s, ft_pool)]["state_path"]
                                        if r == "ar_ft" else None),
                    # P3 checkpoint sharing (PREREG_PHASE2 r8: identical
                    # configurations trained once): persist the b_max states of
                    # the labels and mgn arms for zero-shot transfer evaluation.
                    "state_path": (str(state_dir / f"{r}_b{b}_s{s}.pt")
                                   if (b == max(budgets) and r in ("labels", "mgn"))
                                   else None),
                    "tag": f"{r} b{b} s{s}",
                    "cache_dir": cache_dir, "reuse_existing": reuse,
                })
    sup_out = dict(zip(sup_keys,
                       map_units(cached_supervised_unit, sup_payloads, workers,
                                 "E8 (supervised grid)"), strict=True))
    for r in regimes:
        r_buds = [b for b in budgets
                  if not (r == "mgn" and int(b) not in mgn_budgets)]
        raw[r] = {b: [sup_out[(r, b, s)]["val"] for s in seeds] for b in r_buds}

    cells = {r: {k: _agg(v) for k, v in per.items()} for r, per in raw.items()}

    # ---- plan Sec.4 mandatory naive rows in the headline table -----------------
    baseline_rows = []
    if bool(cfg.get("include_naive_baselines", True)):
        pool_archs = load_archs(pool_files[:max(budgets)])
        val_archs = load_archs(val_files)
        cells.update(naive_baseline_cells(pool_archs, val_archs, budgets))
        baseline_rows = ["zero", "scale_aware_poly", "knn_field"]

    def _row_buds(r):
        return sorted((b for b in cells[r]), key=int)

    auc = {r: label_efficiency_auc(
        _row_buds(r),
        [cells[r][b]["disp_rel_l2"]["mean"] for b in _row_buds(r)])
        for r in regimes + baseline_rows}

    b_max, p_max = max(budgets), max(pool_sizes)
    ar_disp = cells["ar"][p_max]["disp_rel_l2"]["mean"]
    lab_disp = cells["labels"][b_max]["disp_rel_l2"]["mean"]
    k2_val = (ar_disp - lab_disp) / (lab_disp + 1e-30)
    k2 = kill(f"K2: AR (pool {p_max}) displacement > 30% worse than labels-only "
              f"@ {b_max} over >= {len(seeds)} seeds",
              triggered=bool(k2_val > 0.30),
              note=f"AR {ar_disp:.4f} vs labels {lab_disp:.4f} (+{k2_val:.1%})")

    ar_egap = cells["ar"][p_max]["energy_gap_rel"]["mean"]
    advantages = {b: (cells["labels"][b]["energy_gap_rel"]["mean"] - ar_egap)
                  / (cells["labels"][b]["energy_gap_rel"]["mean"] + 1e-30)
                  for b in budgets}
    adv_kill = kill("C1-advantage: AR energy-gap advantage < 40% at every budget",
                    triggered=bool(all(v < 0.40 for v in advantages.values())),
                    note=str({b: round(v, 3) for b, v in advantages.items()}))

    proto = {"budgets": budgets, "pool_sizes": pool_sizes,
             "ar_axis": "unlabeled pool size (never conflated with label budget)",
             "n_seeds": len(seeds), "sup": sup, "ar": pre,
             "ft_pool": ft_pool, "include_mgn": include_mgn,
             "naive_baseline_rows": baseline_rows,
             "workers": workers, "state_dir": str(state_dir),
             "anchored_policy": "gradient-balanced (ratio 1.0), plan Sec.5 item 4"}
    d9 = {"reuse_states": reuse,
          "ar_states": {f"s{s_}": {"reused": pre_out[(s_, ft_pool)].get("reused_state", False),
                                   "sha256": pre_out[(s_, ft_pool)].get("state_sha256")}
                        for s_ in seeds},
          "sup_units_from_cache": [" ".join(map(str, k)) for k, v in sup_out.items()
                                   if v.get("from_cache")],
          "units_resumed_from_epoch": {
              **{f"ar s{s_}": pre_out[(s_, ft_pool)]["resumed_from_epoch"]
                 for s_ in seeds if "resumed_from_epoch" in pre_out[(s_, ft_pool)]},
              **{" ".join(map(str, k)): v["resumed_from_epoch"]
                 for k, v in sup_out.items() if "resumed_from_epoch" in v}}}
    return result("E8", PLAN_REF, proto,
                  {"cells": cells, "label_efficiency_auc_disp": auc,
                   "ar_egap_advantage_by_budget": advantages,
                   "d9_restart": d9}, [k2, adv_kill])
