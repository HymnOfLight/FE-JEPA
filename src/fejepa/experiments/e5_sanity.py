"""E5' -- repaired sanity floor (plan B2 / Sec.6 E5').

Every headline table must beat: the zero predictor by >= `margin` (default 3x, i.e.
anchored disp <= 1/3), AND every repaired naive (global poly kept as a diagnostic row,
scale-aware poly, k-NN field transport) at every budget. Failure at any budget is not a
verdict about the method -- it is a bug hunt (plan Sec.6).

The anchored errors are injected from E1's balanced arm when E1 ran (identical training
protocol, no duplicated compute); otherwise this module trains the pre-registered
anchored configuration itself.
"""

from __future__ import annotations

from ..baselines import (GlobalPolyBaseline, KNNFieldBaseline,
                         ScaleAwarePolyBaseline, zero_predictor)
from ..metrics import evaluate_model
from .protocol import kill, result, seeded_factory

PLAN_REF = "plan v2.0 Sec.6 E5', B2, gate G1'(a)"


def _train_anchored(model_factory, pool_archs, val_archs, budget: int, cfg: dict) -> dict:
    from ..train.supervised import SupervisedConfig, train_supervised

    res = train_supervised(seeded_factory(model_factory, 0),
                           pool_archs[:budget], val_archs,
                           SupervisedConfig(seed=0, anchor_mode="balanced",
                                            epochs=int(cfg.get("epochs", 200)),
                                            lr=float(cfg.get("lr", 1.5e-3)),
                                            device=cfg.get("device", "cpu"),
                                            desc=f"E5' fallback b{budget}"))
    return {"mean": res["val"]["disp_rel_l2"],
            "per_instance": res["val"]["per_instance"]["disp_rel_l2"]}


def run_e5(pool_archs, val_archs, cfg: dict, anchored: dict | None = None,
           model_factory=None) -> dict:
    budgets = [int(b) for b in cfg.get("budgets", [16, 64, 256, 1024])]
    fit_budget = int(cfg.get("fit_budget", max(budgets)))
    margin = float(cfg.get("margin", 3.0))
    fit = pool_archs[:fit_budget]

    baselines = {
        "zero": zero_predictor,
        "poly": GlobalPolyBaseline().fit(fit).predict,
        "scale_aware_poly": ScaleAwarePolyBaseline().fit(fit).predict,
        "knn_field": KNNFieldBaseline().fit(fit).predict,
    }
    base_eval = {name: evaluate_model(fn, val_archs) for name, fn in baselines.items()}
    naive_means = {n: base_eval[n]["disp_rel_l2"] for n in base_eval if n != "zero"}

    per_budget, failures = [], []
    for b in budgets:
        if anchored is not None and b in anchored:
            a = anchored[b]
        elif anchored is not None and str(b) in anchored:
            a = anchored[str(b)]
        else:
            if model_factory is None:
                raise ValueError("E5': no anchored errors injected and no model_factory")
            a = _train_anchored(model_factory, pool_archs, val_archs, b, cfg)
        beats_zero = a["mean"] <= 1.0 / margin
        beats_naive = a["mean"] < min(naive_means.values())
        passed = bool(beats_zero and beats_naive)
        if not passed:
            failures.append(b)
        per_budget.append({"budget": b, "anchored_disp": a["mean"],
                           "beats_zero_x": (1.0 / (a["mean"] + 1e-30)),
                           "beats_all_naive": beats_naive, "passed": passed})

    k = kill(f"E5' sanity: >= {margin}x over zero AND beat every repaired naive, "
             "at every budget",
             triggered=bool(failures),
             note=("bug hunt: failed at budgets " + str(failures)) if failures
             else "all budgets passed")
    metrics = {"baselines": {n: {"disp_rel_l2": base_eval[n]["disp_rel_l2"],
                                 "per_instance": base_eval[n]["per_instance"]["disp_rel_l2"]}
                             for n in base_eval},
               "per_budget": per_budget, "passed_all": not failures}
    proto = {"fit_budget": fit_budget, "margin": margin, "budgets": budgets,
             "anchored_source": "E1' balanced arm" if anchored else "trained in E5'"}
    return result("E5'", PLAN_REF, proto, metrics, [k])
