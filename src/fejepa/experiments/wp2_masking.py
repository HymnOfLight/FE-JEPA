"""WP2 -- region-mask ratio sweep {0.2, 0.4, 0.6} (pre-E2 design validation).

Plan v2.0 WP2 schedules the redesigned masking at three ratios. The E2 verdict is
one-shot (plan Sec.5 item 6), so the ratio must be chosen *before* E2 on signals
that never peek at E2's downstream comparison:

  * held-out masked-prediction MSE -- how well the cross-attention predictor
    reconstructs unseen-region latents at each ratio (lower is better);
  * pooled standardized effective rank on the hold-out -- a collapse sentinel
    (reported alongside; a ratio that wins MSE by collapsing is visibly suspect).

This is a design dial, not a falsification test: no kill condition. The
recommended ratio (argmin held-out MSE) feeds ``model.mask_frac`` for the E2 run.
Label-free throughout (WP5: consumes only the unlabeled pool).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from ..metrics import effective_rank
from ..models.fejepa import FEJEPAConfig, build_fejepa, mesh_adjacency
from ..progress import Task
from ..train.losses import JEPA_CONFIG
from ..train.pretrain import PretrainConfig, pretrain
from .protocol import result, seeded_factory

PLAN_REF = "plan v2.0 WP2 (mask ratios {0.2,0.4,0.6}), Sec.5 item 6 (pre-E2)"


def _holdout_pred_mse(model, archs, device, draws: int = 2) -> float:
    import torch

    model.eval()
    vals = []
    with torch.no_grad():
        for i, a in enumerate(archs):
            pack = model.prepare_instance(a, device)
            adj = mesh_adjacency(a.elements, a.n_nodes)
            for d in range(draws):
                rng = np.random.default_rng(10_000 + 7 * i + d)
                vals.append(float(model.masked_prediction(pack["feats"][0],
                                                          adj, rng)))
    model.train()
    return float(np.mean(vals))


def _pooled_std_rank(model, archs, device) -> float:
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for a in archs:
            pack = model.prepare_instance(a, device)
            rows.append(model.encode(pack["feats"][0]).mean(dim=0).cpu().numpy())
    model.train()
    return effective_rank(np.stack(rows), standardized=True)


def run_wp2(model_cfg: dict, pool_archs, cfg: dict) -> dict:
    ratios = [float(r) for r in cfg.get("ratios", [0.2, 0.4, 0.6])]
    steps = int(cfg.get("steps", 600))
    n_train = int(cfg.get("n_train", 32))
    n_holdout = int(cfg.get("n_holdout", 16))
    seed = int(cfg.get("seed", 0))
    device = cfg.get("device", "cpu")
    if len(pool_archs) < n_train + n_holdout:
        raise ValueError(f"WP2: pool has {len(pool_archs)} archives; "
                         f"needs n_train+n_holdout={n_train + n_holdout}")
    train = pool_archs[:n_train]
    holdout = pool_archs[n_train:n_train + n_holdout]
    epochs = max(1, math.ceil(steps / n_train))
    base = FEJEPAConfig.from_dict(model_cfg)

    task = Task("WP2 mask sweep", total=len(ratios))
    per_ratio = []
    for r in ratios:
        model = seeded_factory(
            lambda r=r: build_fejepa(replace(base, mask_frac=r)), seed)
        pretrain(model, train,
                 PretrainConfig(epochs=epochs, lr=float(cfg.get("lr", 1e-3)),
                                seed=seed, device=device, loss=JEPA_CONFIG,
                                desc=f"WP2 mask r={r}"))
        per_ratio.append({"ratio": r,
                          "holdout_pred_mse": _holdout_pred_mse(model, holdout,
                                                                device),
                          "pooled_std_rank": _pooled_std_rank(model, holdout,
                                                              device)})
        task.step(f"r={r}")
    task.done()

    recommended = min(per_ratio, key=lambda x: x["holdout_pred_mse"])["ratio"]
    proto = {"ratios": ratios, "steps": steps, "epochs": epochs,
             "n_train": n_train, "n_holdout": n_holdout,
             "selection": "argmin held-out masked-prediction MSE; standardized "
                          "rank reported alongside (design dial, no kill; must "
                          "conclude BEFORE E2's one-shot verdict)"}
    return result("WP2-mask", PLAN_REF, proto,
                  {"per_ratio": per_ratio, "recommended_ratio": recommended}, [])
