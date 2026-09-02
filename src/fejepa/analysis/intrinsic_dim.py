"""Intrinsic dimension of latent tokens (wp8-lejepa Stage 0, the B1 gate).

LeWorldModel's recorded weakness: SIGReg pushes toward a d-dimensional
isotropic Gaussian and underperforms when the data's intrinsic dimension is
far below d. Before a latent-shaping arm is pre-registered we MEASURE:
TwoNN intrinsic dimension (Facco et al. 2017), PCA cumulative-variance
dimensions, the SIGReg baseline, and which module executes last in encode().
"""

from __future__ import annotations

import numpy as np

from .common import encode_instance, subsample_rows


def twonn(x: np.ndarray, discard_frac: float = 0.1) -> float:
    """TwoNN: fit -log(1-F(mu)) = d log(mu) through the origin on the ratios
    mu = r2/r1 of the two nearest-neighbour distances. The empirical CDF is
    taken over ALL points; the top `discard_frac` of ratios is excluded from
    the fit only (Facco et al.). Distances via |a|^2+|b|^2-2ab (memory-safe)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    r1, r2 = np.empty(n), np.empty(n)
    sq = (x * x).sum(1)
    chunk = 2048
    for s in range(0, n, chunk):
        blk = x[s:s + chunk]
        d = sq[s:s + chunk, None] + sq[None, :] - 2.0 * blk @ x.T
        np.maximum(d, 0.0, out=d)
        d[np.arange(blk.shape[0]), np.arange(s, s + blk.shape[0])] = np.inf
        part = np.partition(d, 1, axis=1)[:, :2]
        part.sort(axis=1)
        r1[s:s + chunk] = np.sqrt(part[:, 0])
        r2[s:s + chunk] = np.sqrt(part[:, 1])
    mu = r2 / np.maximum(r1, 1e-12)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1.0)])
    f = np.arange(1, mu.size + 1) / (mu.size + 1)
    keep = int((1.0 - discard_frac) * mu.size)
    xs, ys = np.log(mu[:keep]), -np.log(1.0 - f[:keep])
    return float((xs * ys).sum() / max((xs * xs).sum(), 1e-12))


def pca_dims(x: np.ndarray, levels=(0.90, 0.95, 0.99)) -> dict:
    xc = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    cum = np.cumsum(s ** 2 / (s ** 2).sum())
    return {f"pca_{int(l * 100)}": int(np.searchsorted(cum, l) + 1) for l in levels}


def last_executed_module(model, feats, pack=None) -> str:
    """Class name of the leaf module that fires LAST in encode() -- execution
    order via forward hooks, not registration order."""
    import torch

    fired, hooks = [], []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: fired.append((n, type(mod).__name__))))
    try:
        with torch.no_grad():
            if getattr(model, "needs_pack", False):
                model.encode(feats, pack)
            else:
                model.encode(feats)
    finally:
        for h in hooks:
            h.remove()
    return fired[-1][1] if fired else "none"


def last_module_is_layernorm(model, feats, pack=None) -> bool:
    return last_executed_module(model, feats, pack) == "LayerNorm"


def measure_intrinsic_dimension(model, archs, max_rows: int = 20000) -> dict:
    """Pool the token clouds of `archs`, subsample, and report the readings
    plus the pre-declared B1 rule (ID < 0.25 x latent dim => narrow head)."""
    import torch

    from ..train.sigreg import sigreg_monitor

    rows, first = [], None
    for arch in archs:
        z, pack = encode_instance(model, arch)
        rows.append(z.cpu())
        first = first or (pack["feats"], pack)
    z_all = subsample_rows(torch.cat(rows, 0), max_rows)
    x = z_all.numpy()
    res = {"n_rows": int(x.shape[0]), "latent_dim": int(x.shape[1]),
           "twonn_id": twonn(x), **pca_dims(x),
           "sigreg_monitor_raw": sigreg_monitor(z_all, n_proj=256),
           "encoder_ends_with_layernorm": last_module_is_layernorm(model, *first),
           "last_executed_module": last_executed_module(model, *first)}
    narrow = res["twonn_id"] < 0.25 * res["latent_dim"]
    res["suggested_head_width"] = int(round(2 * res["twonn_id"])) if narrow else 0   # 0 = full
    res["b1_reading"] = (
        "intrinsic dimension << latent dim: shape the latent on a projector head "
        "sized near the intrinsic dimension (LeWM Two-Room caveat)"
        if res["twonn_id"] < 0.25 * res["latent_dim"] else
        "intrinsic dimension comparable to latent dim: full-width SIGReg admissible")
    return res
