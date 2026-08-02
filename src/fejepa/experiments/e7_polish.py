"""E7 -- the learned initializer + CG polish (plan C3 / Sec.6 E7).

Everything here is label-free at inference: the AR model and the (K, F) pair are all
CG needs. Measured quantities: iterations-to-tolerance from {zero, naive, learned}
inits; the frozen metric suite after exactly k polish steps for k in cfg.ks; and a
wall-clock ledger (plan Sec.4: "CG iterations-to-tolerance" is a primary metric).

Kill K5: "< 20% iteration savings vs. the zero init => drop C3."
"""

from __future__ import annotations

import numpy as np

from ..baselines import ScaleAwarePolyBaseline
from ..fe.solve import cg_iterations_to_tol
from ..polish import polish_battery
from ..metrics import evaluate_fields, torch_predictor
from ..train.pretrain import PretrainConfig, amortized_ritz
from ..progress import Task
from .protocol import kill, result, seeded_factory

PLAN_REF = "plan v2.0 Sec.6 E7, C3, kill K5, WP3 (consumes fejepa.polish)"


def run_e7(model_factory, pool_archs, val_archs, cfg: dict) -> dict:
    device = cfg.get("device", "cpu")
    tol = float(cfg.get("tol", 1e-6))
    ks = [int(k) for k in cfg.get("ks", [0, 5, 20])]
    n_eval = min(int(cfg.get("n_eval", 64)), len(val_archs))
    pool_size = min(int(cfg.get("pool_size", 256)), len(pool_archs))
    fit_budget = min(int(cfg.get("fit_budget", 256)), len(pool_archs))

    model = seeded_factory(model_factory, int(cfg.get("seed", 0)))
    amortized_ritz(model, pool_archs[:pool_size],
                   PretrainConfig(epochs=int(cfg.get("pre_epochs", 50)),
                                  lr=float(cfg.get("lr", 1e-3)),
                                  seed=int(cfg.get("seed", 0)), device=device,
                                  desc="E7 AR"))
    learned = torch_predictor(model, device)
    naive = ScaleAwarePolyBaseline().fit(pool_archs[:fit_budget]).predict

    iters = {"zero": [], "naive": [], "learned": []}
    wall = {"zero": 0.0, "naive": 0.0, "learned": 0.0}
    k_rows = {k: [] for k in ks}
    task = Task("E7 eval", total=n_eval)
    for a in val_archs[:n_eval]:
        U_l, U_n = learned(a), naive(a)
        for j in range(a.n_loads):
            for name, x0 in (("zero", None), ("naive", U_n[j]), ("learned", U_l[j])):
                it, _conv, dt = cg_iterations_to_tol(a.K, a.F[j], a.free_mask, x0, tol)
                iters[name].append(it)
                wall[name] += dt
        for k in ks:
            k_rows[k].append(evaluate_fields(polish_battery(a, U_l, k=k), a))
        task.step(f"instance {a.path.name if a.path else ''}")
    task.done()

    mean_iters = {n: float(np.mean(v)) for n, v in iters.items()}
    savings = 1.0 - mean_iters["learned"] / (mean_iters["zero"] + 1e-30)
    savings_naive = 1.0 - mean_iters["naive"] / (mean_iters["zero"] + 1e-30)
    polish = {k: {key: float(np.mean([r[key] for r in rows]))
                  for key in rows[0]} for k, rows in k_rows.items() if rows}

    k5 = kill("K5: iteration savings (learned vs zero init) < 20%",
              triggered=bool(savings < 0.20),
              note=f"savings learned={savings:.3f}, naive={savings_naive:.3f}")
    proto = {"tol": tol, "ks": ks, "n_eval": n_eval, "pool_size": pool_size,
             "cg": "unpreconditioned (interpretable counts)",
             "label_free_inference": True}
    metrics = {"iterations_to_tol_mean": mean_iters,
               "iterations_raw": {n: v for n, v in iters.items()},
               "savings_learned": savings, "savings_naive": savings_naive,
               "wall_clock_s": {n: round(v, 3) for n, v in wall.items()},
               "polish_at_k": polish}
    return result("E7", PLAN_REF, proto, metrics, [k5])
