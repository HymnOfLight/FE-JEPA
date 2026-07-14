"""E6 -- latent--physics alignment (the empirical face of Proposition 1, plan C5d).

Within-geometry (Prop. 1's scoped domain): per labelled probe instance, Spearman rho
between pairwise pooled-latent distances across the 4 loads (6 pairs) and the K-norm
distances ||U*_i - U*_j||_K; report the mean over geometries.
Cross-geometry (descriptor-conditioned variant): pooled latents at load 0 across
instances vs. geometry-descriptor distances (fields on different meshes are not
directly comparable; the descriptor ordering is the honest cross-geometry target).

Kill: "rho < 0.3 within-geometry => Prop. 1's scoping untenable; theory WP re-scopes."
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from ..models.features import geometry_descriptor
from ..train.pretrain import PretrainConfig, amortized_ritz
from .protocol import kill, result, seeded_factory

PLAN_REF = "plan v2.0 Sec.6 E6, C5(d)"


def _pooled_battery(model, arch, device) -> np.ndarray:
    import torch

    pack = model.prepare_instance(arch, device)
    model.eval()
    with torch.no_grad():
        z = model.encode(pack["feats"])                   # (L, N, dim)
        pooled = model.pooled(z)                          # (L, dim)
    model.train()
    return pooled.detach().cpu().numpy()


def _k_norm(d: np.ndarray, K) -> float:
    return float(np.sqrt(max(0.0, d @ (K @ d))))


def run_e6(model_factory, pool_archs, probe_archs, cfg: dict) -> dict:
    pool_size = min(int(cfg.get("pool_size", 256)), len(pool_archs))
    device = cfg.get("device", "cpu")
    model = seeded_factory(model_factory, int(cfg.get("seed", 0)))
    amortized_ritz(model, pool_archs[:pool_size],
                   PretrainConfig(epochs=int(cfg.get("pre_epochs", 50)),
                                  lr=float(cfg.get("lr", 1e-3)),
                                  seed=int(cfg.get("seed", 0)), device=device,
                                  desc="E6 AR"))

    within = []
    pooled0 = []
    for a in probe_archs:
        P = _pooled_battery(model, a, device)
        pooled0.append(P[0])
        lat, phys = [], []
        for i in range(a.n_loads):
            for j in range(i + 1, a.n_loads):
                lat.append(np.linalg.norm(P[i] - P[j]))
                phys.append(_k_norm(a.U_star[i] - a.U_star[j], a.K))
        res = spearmanr(lat, phys)
        rho = getattr(res, "statistic", getattr(res, "correlation", float("nan")))
        if np.isfinite(rho):
            within.append(float(rho))

    pooled0 = np.stack(pooled0)
    descs = np.stack([geometry_descriptor(a.meta) for a in probe_archs])
    n = pooled0.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = np.random.default_rng(0)
    if len(pairs) > int(cfg.get("max_pairs_cross", 2000)):
        idx = rng.choice(len(pairs), size=int(cfg.get("max_pairs_cross", 2000)),
                         replace=False)
        pairs = [pairs[k] for k in idx]
    lat_x = [np.linalg.norm(pooled0[i] - pooled0[j]) for i, j in pairs]
    dsc_x = [np.linalg.norm(descs[i] - descs[j]) for i, j in pairs]
    if pairs:
        res_x = spearmanr(lat_x, dsc_x)
        rho_cross = float(getattr(res_x, "statistic",
                                  getattr(res_x, "correlation", float("nan"))))
    else:
        rho_cross = float("nan")

    rho_within = float(np.mean(within)) if within else float("nan")
    k = kill("E6: within-geometry Spearman rho < 0.3",
             triggered=bool(not np.isfinite(rho_within) or rho_within < 0.3),
             note=f"mean within rho = {rho_within:.3f} over {len(within)} geometries")
    proto = {"pool_size": pool_size, "pre_epochs": int(cfg.get("pre_epochs", 50)),
             "n_probe": len(probe_archs), "pairs_per_geometry": 6,
             "cross_variant": "descriptor-conditioned"}
    return result("E6", PLAN_REF, proto,
                  {"rho_within_mean": rho_within, "rho_within_per_geometry": within,
                   "rho_cross_descriptor": rho_cross}, [k])
