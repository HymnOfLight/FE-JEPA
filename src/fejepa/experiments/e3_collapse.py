"""E3' -- representation collapse with a repaired instrument (plan B4 / Sec.6 E3').

The audited defects being repaired here:
  - unstandardized participation-ratio rank is dominated by the largest-variance
    direction -> report *standardized* rank (primary) and raw (diagnostic);
  - mean-pooling is itself nearly degenerate (mean-pooled raw inputs read 1.21/6)
    -> three probes: mean, [mean, std], fixed-random-query attention pooling;
  - no reference point -> the input-feature rank floor is computed with the *same*
    probes on the raw features and printed alongside;
  - the probe model trained ~192 steps -> ``steps`` >= 2000 enforced by config.

Factors: regularizer mode in {none} + cfg.modes (sigreg | sigreg_pooled | vicreg_pooled)
x geometry-descriptor conditioning {off, on}.

Kill K4: "standardized rank with the best mechanism <= 1.5x the no-regularizer run
=> representation story fails; lean on the AR framing."
"""

from __future__ import annotations

import math

import numpy as np

from ..metrics import effective_rank
from ..models.features import FeatureSpec, build_features
from ..train.losses import LossConfig
from ..train.pretrain import PretrainConfig, pretrain
from ..progress import Task
from .protocol import kill, result, seeded_factory

PLAN_REF = "plan v2.0 Sec.6 E3', B4, kill K4"

PROBES = ("mean", "mean_std", "attn")


def _attn_pool(z: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fixed random-query attention pooling: parameter-free, deterministic, spatially
    selective (documented instrument; not a trained head)."""
    d = z.shape[1]
    q = np.random.default_rng(seed).standard_normal(d)
    w = z @ q / math.sqrt(d)
    w = np.exp(w - w.max())
    w = w / w.sum()
    return w @ z


def _probe_matrices(rows: list[np.ndarray]) -> dict:
    return {
        "mean": np.stack([z.mean(axis=0) for z in rows]),
        "mean_std": np.stack([np.concatenate([z.mean(axis=0), z.std(axis=0)])
                              for z in rows]),
        "attn": np.stack([_attn_pool(z) for z in rows]),
    }


def _ranks(mat: np.ndarray) -> dict:
    return {"raw": effective_rank(mat, standardized=False),
            "std": effective_rank(mat, standardized=True),
            "dim": int(mat.shape[1])}


def _collect_latents(model, probe_archs, device) -> list[np.ndarray]:
    import torch

    rows = []
    model.eval()
    with torch.no_grad():
        for a in probe_archs:
            pack = model.prepare_instance(a, device)
            z = model.encode(pack["feats"][0])            # load 0, (N, dim)
            rows.append(z.detach().cpu().numpy())
    model.train()
    return rows


def run_e3(model_factory, pool_archs, cfg: dict) -> dict:
    steps = int(cfg.get("steps", 2000))
    if steps < 2000:
        raise ValueError("plan E3': probe training >= 2000 steps")
    n_train = min(int(cfg.get("n_train", 32)), len(pool_archs))
    n_probe = min(int(cfg.get("n_probe", 64)), len(pool_archs))
    modes = list(cfg.get("modes", ["sigreg", "sigreg_pooled", "vicreg_pooled"]))
    conds = [bool(x) for x in cfg.get("geometry_conditions", [False, True])]
    device = cfg.get("device", "cpu")
    seed = int(cfg.get("seed", 0))
    epochs = max(1, math.ceil(steps / n_train))
    train, probe = pool_archs[:n_train], pool_archs[:n_probe]

    task = Task("E3'", total=len(conds) * (1 + len(modes)))
    conditions = []
    for geom in conds:
        spec = FeatureSpec(load_summary=True, geometry=geom)
        runs = {}
        for reg in ["none"] + modes:
            model = seeded_factory(lambda spec=spec: model_factory(features=spec),
                                   seed)
            loss = LossConfig(use_pred=True, use_phys=True, reg_mode=reg)
            pretrain(model, train,
                     PretrainConfig(epochs=epochs, seed=seed, device=device,
                                    loss=loss, lr=float(cfg.get("lr", 1e-3)),
                                    desc=f"E3' geom={geom} reg={reg}"))
            mats = _probe_matrices(_collect_latents(model, probe, device))
            runs[reg] = {"probes": {p: _ranks(m) for p, m in mats.items()}}
            task.step(f"geom={geom} reg={reg}")
        feat_rows = [build_features(a, 0, spec).astype(np.float64) for a in probe]
        floor = {p: _ranks(m) for p, m in _probe_matrices(feat_rows).items()}
        conditions.append({"geometry": geom, "runs": runs, "input_floor": floor})

    task.done()
    ratios = []
    for c in conditions:
        for reg in modes:
            for p in PROBES:
                on = c["runs"][reg]["probes"][p]["std"]
                off = c["runs"]["none"]["probes"][p]["std"]
                ratios.append({"geometry": c["geometry"], "mode": reg, "probe": p,
                               "std_ratio_on_over_off": on / (off + 1e-30)})
    best = max(r["std_ratio_on_over_off"] for r in ratios) if ratios else 0.0
    k4 = kill("K4: best standardized-rank ratio (reg on / off, matched condition & "
              "probe) <= 1.5",
              triggered=bool(best <= 1.5),
              note=f"best ratio = {best:.3f}")

    proto = {"steps": steps, "epochs": epochs, "n_train": n_train, "n_probe": n_probe,
             "modes": modes, "geometry_conditions": conds,
             "probes": list(PROBES), "rank": "standardized primary, raw diagnostic"}
    return result("E3'", PLAN_REF, proto,
                  {"conditions": conditions, "ratios": ratios,
                   "best_std_ratio": best}, [k4])
