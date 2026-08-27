"""E1' -- the anchor as a supervised auxiliary (pipeline P-A).

Plan v2.0 Sec.6:
  "Anchor as auxiliary: gradient-balanced (primary) and fixed lambda=1; grid secondary;
   report displacement AND energy-gap improvement per budget; per-seed arrays."
Kill K1: "energy-gap improvement < 25% at every budget => anchor neutral even on its
home metric."

Selection-bias hygiene (plan B3): the grid-best number is emitted with
``selection_bias: true`` and never feeds the gate.

Execution: every training is one :func:`~fejepa.experiments.parallel.supervised_unit`
payload (archive paths, model config dict, seed); ``cfg["workers"]`` > 1 runs units
concurrently on the GPU with identical results to the serial path (tested).
"""

from __future__ import annotations

from .parallel import map_units, supervised_unit
from .protocol import (PIPELINE_PA, divergence_flags, kill, mean_std, result,
                       seeds_list, t_stat)

PLAN_REF = "plan v2.0 Sec.6 E1', Sec.5 item 4 (lambda policy), kill K1"


def _improvement(arm_out: dict, n_seeds: int, arm: str, key: str) -> dict:
    off, on = arm_out["none"][key], arm_out[arm][key]
    return {"value": (off["mean"] - on["mean"]) / (off["mean"] + 1e-30),
            "t": t_stat(off, on, n_seeds)}


def _arm_cfgs(cfg: dict) -> dict:
    base = dict(epochs=int(cfg.get("epochs", 200)), lr=float(cfg.get("lr", 1.5e-3)),
                device=cfg.get("device", "cpu"))
    arms = {
        "none": dict(anchor_mode="none", **base),
        "balanced": dict(anchor_mode="balanced",
                         balance_ratio=float(cfg.get("balance_ratio", 1.0)), **base),
        "fixed": dict(anchor_mode="fixed", lambda_phys=1.0, **base),
    }
    for lam in cfg.get("grid", []):
        arms[f"grid_{lam}"] = dict(anchor_mode="fixed", lambda_phys=float(lam), **base)
    return arms


def run_e1(model_cfg: dict, pool_files, val_files, cfg: dict) -> dict:
    budgets = [int(b) for b in cfg.get("budgets", [16, 64, 256, 1024])]
    seeds = seeds_list(cfg.get("seeds", 3))
    decision = int(cfg.get("decision_budget", 64))
    workers = int(cfg.get("workers", 1))
    tf32 = bool(cfg.get("tf32", True))
    arms = _arm_cfgs(cfg)
    val_str = [str(f) for f in val_files]

    keys, payloads = [], []
    for b in budgets:
        train_str = [str(f) for f in pool_files[:b]]
        for name, kw in arms.items():
            for s in seeds:
                keys.append((b, name, s))
                payloads.append({
                    "kind": "fejepa", "model": model_cfg, "seed": s, "tf32": tf32,
                    "compile": cfg.get("compile", False),
                    "precision": cfg.get("precision", "fp32"),
                    "train_files": train_str, "val_files": val_str,
                    "sup": dict(kw, desc=f"E1' b{b} {name} s{s}"),
                    "tag": f"b{b} {name} s{s}",
                })
    unit_out = dict(zip(keys, map_units(supervised_unit, payloads, workers, "E1'"),
                        strict=True))

    per_budget = []
    for b in budgets:
        arm_out = {}
        for name in arms:
            vals = [unit_out[(b, name, s)]["val"] for s in seeds]
            arm_out[name] = {
                "disp": mean_std([v["disp_rel_l2"] for v in vals]),
                "egap": mean_std([v["energy_gap_rel"] for v in vals]),
                "per_instance_by_seed": [v["per_instance"] for v in vals],
                "divergence_flags": divergence_flags(vals),   # r8 Sec.5
            }

        def improvement(arm: str, key: str, _arms=arm_out) -> dict:
            return _improvement(_arms, len(seeds), arm, key)

        entry = {
            "budget": b,
            "arms": arm_out,
            "improvements": {
                "balanced": {"disp": improvement("balanced", "disp"),
                             "egap": improvement("balanced", "egap")},
                "fixed": {"disp": improvement("fixed", "disp"),
                          "egap": improvement("fixed", "egap")},
            },
        }
        grid_arms = [a for a in arm_out if a.startswith("grid_")]
        if grid_arms:
            best = max(grid_arms, key=lambda a: improvement(a, "disp")["value"])
            entry["grid_best"] = {"arm": best, "selection_bias": True,
                                  "disp": improvement(best, "disp"),
                                  "egap": improvement(best, "egap")}
        per_budget.append(entry)

    best_egap_by_budget = [max(e["improvements"]["balanced"]["egap"]["value"],
                               e["improvements"]["fixed"]["egap"]["value"])
                           for e in per_budget]
    k1 = kill("K1: energy-gap improvement < 25% at every budget",
              triggered=bool(max(best_egap_by_budget) < 0.25),
              note=f"best per budget: {[round(x, 4) for x in best_egap_by_budget]}")

    retired = next((e for e in per_budget if e["budget"] == decision), None)
    metrics = {"per_budget": per_budget,
               "retired_criterion_report": {
                   "note": "plan Sec.5 item 1: displacement>=10%@decision is retired, "
                           "reported here for the record",
                   "fixed_disp_improvement_at_decision":
                       None if retired is None else retired["improvements"]["fixed"]["disp"],
               }}
    proto = {"pipeline": PIPELINE_PA, "budgets": budgets, "n_seeds": len(seeds),
             "epochs": int(cfg.get("epochs", 200)), "workers": workers,
             "primary_arm": "balanced (ratio=1.0)", "grid": list(cfg.get("grid", []))}
    return result("E1'", PLAN_REF, proto, metrics, [k1])
