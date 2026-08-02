"""E2 -- the JEPA component on trial (pipeline P-B), run only after the WP2 redesign
(region masking + cross-attention predictor + pooled regularizer -- all shipped in
this package, see models/fejepa.py and models/regularizers.py).

Plan v2.0 Sec.6: "scratch vs. AR->FT vs. JEPA->FT, budgets {16,64,256}, same seeds/data."
Kill K3: "JEPA within 3% of AR everywhere => drop the SSL term, branch B."
The verdict is rendered once (plan Sec.5 item 6) and selects the paper branch; it does
NOT gate scale-up.
"""

from __future__ import annotations

import copy

from ..train.losses import JEPA_CONFIG
from ..train.pretrain import PretrainConfig, amortized_ritz, pretrain
from ..train.supervised import SupervisedConfig, train_supervised
from ..progress import Task
from .protocol import (PIPELINE_PB, kill, mean_std, result, seeded_factory,
                       seeds_list)

PLAN_REF = "plan v2.0 Sec.6 E2, C4, Sec.5 item 6, kill K3"


def run_e2(model_factory, pool_archs, val_archs, cfg: dict) -> dict:
    budgets = [int(b) for b in cfg.get("budgets", [16, 64, 256])]
    seeds = seeds_list(cfg.get("seeds", 3))
    pool_size = int(cfg.get("pool_size", 1024))
    decision = int(cfg.get("decision_budget", 64))
    pre = dict(epochs=int(cfg.get("pre_epochs", 100)),
               lr=float(cfg.get("pre_lr", 1e-3)), device=cfg.get("device", "cpu"))
    ft = dict(epochs=int(cfg.get("ft_epochs", 200)),
              lr=float(cfg.get("ft_lr", 1.5e-3)), device=cfg.get("device", "cpu"))
    if len(pool_archs) < pool_size:
        raise ValueError(f"E2: pool has {len(pool_archs)} < pool_size={pool_size}")
    pool = pool_archs[:pool_size]

    task = Task("E2", total=len(seeds) * (2 + 3 * len(budgets)))
    cells = {arm: {b: {"disp": [], "egap": []} for b in budgets}
             for arm in ("scratch", "ar_ft", "jepa_ft")}
    for s in seeds:
        m_ar = seeded_factory(model_factory, s)
        amortized_ritz(m_ar, pool, PretrainConfig(seed=s, desc=f"E2 AR s{s}", **pre))
        ar_state = copy.deepcopy(m_ar.state_dict())
        task.step(f"AR pretrain s{s}")
        m_j = seeded_factory(model_factory, s)
        pretrain(m_j, pool,
                 PretrainConfig(seed=s, loss=JEPA_CONFIG, desc=f"E2 JEPA s{s}", **pre))
        jepa_state = copy.deepcopy(m_j.state_dict())
        task.step(f"JEPA pretrain s{s}")
        for b in budgets:
            for arm, state in (("scratch", None), ("ar_ft", ar_state),
                               ("jepa_ft", jepa_state)):
                res = train_supervised(seeded_factory(model_factory, s),
                                       pool_archs[:b], val_archs,
                                       SupervisedConfig(seed=s, anchor_mode="none",
                                                        desc=f"E2 {arm} b{b} s{s}",
                                                        **ft),
                                       pretrained_state=state)
                cells[arm][b]["disp"].append(res["val"]["disp_rel_l2"])
                cells[arm][b]["egap"].append(res["val"]["energy_gap_rel"])
                task.step(f"{arm} b{b} s{s}")

    task.done()
    table = {arm: {b: {k: mean_std(v[k]) for k in ("disp", "egap")}
                   for b, v in per.items()} for arm, per in cells.items()}

    def impr(b: int, key: str) -> float:
        ar, je = table["ar_ft"][b][key]["mean"], table["jepa_ft"][b][key]["mean"]
        return (ar - je) / (ar + 1e-30)

    improvements = {b: {"disp": impr(b, "disp"), "egap": impr(b, "egap")}
                    for b in budgets}
    within3_everywhere = all(abs(improvements[b]["disp"]) < 0.03
                             and abs(improvements[b]["egap"]) < 0.03 for b in budgets)
    k3 = kill("K3: full JEPA within 3% of AR at every budget (disp AND egap) "
              "=> drop SSL term, branch B",
              triggered=bool(within3_everywhere),
              note=str({b: {k: round(v, 4) for k, v in d.items()}
                        for b, d in improvements.items()}))

    metrics = {"table": table, "jepa_vs_ar_improvements": improvements,
               "jepa_improvement_at_decision_budget": improvements.get(decision)}
    proto = {"pipeline": PIPELINE_PB, "pool_size": pool_size, "budgets": budgets,
             "n_seeds": len(seeds), "pretrain": pre, "finetune": ft,
             "run_after_wp2_redesign": True}
    return result("E2", PLAN_REF, proto, metrics, [k3])
