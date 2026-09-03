"""Cross-geometry latent separation (E1's adjudication statistic).

  1. pooled latent per instance: mean of the encoder output over load cases
     and nodes (or bottleneck tokens);
  2. geometry bins: quartiles of the first principal component of the
     per-instance 6-D geometry descriptor;
  3. S = mean silhouette of the bins in latent space (Euclidean).
Secondary readings: leave-one-out 1-NN bin accuracy, SIGReg monitors.
"""

from __future__ import annotations

import numpy as np

from .common import encode_instance


def quartile_bins(scalar: np.ndarray) -> np.ndarray:
    q = np.quantile(scalar, [0.25, 0.5, 0.75])
    return np.searchsorted(q, scalar, side="right")            # 0..3


def _pairwise(x: np.ndarray) -> np.ndarray:
    sq = (x * x).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * x @ x.T, 0.0))


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over samples whose bin has >= 2 members."""
    d = _pairwise(x)
    vals = []
    for i in range(x.shape[0]):
        own = labels == labels[i]
        if own.sum() < 2:
            continue
        a = d[i, own].sum() / (own.sum() - 1)
        b = min(d[i, labels == l].mean() for l in np.unique(labels) if l != labels[i])
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals)) if vals else float("nan")


def loo_1nn_accuracy(x: np.ndarray, labels: np.ndarray) -> float:
    d = _pairwise(x) ** 2
    np.fill_diagonal(d, np.inf)
    return float((labels[np.argmin(d, axis=1)] == labels).mean())


def measure_separation(model, archs, token_rows_for_monitor: int = 20000) -> dict:
    import torch

    from ..models.features import geometry_descriptor
    from ..train.sigreg import sigreg_monitor

    pooled, descs, tokens = [], [], []
    for arch in archs:
        z, _ = encode_instance(model, arch)
        pooled.append(z.mean(0).cpu().numpy())
        tokens.append(z.cpu())
        descs.append(np.asarray(geometry_descriptor(arch.meta), dtype=np.float64))
    X, G = np.stack(pooled), np.stack(descs)
    Gc = G - G.mean(0)
    pc1 = Gc @ np.linalg.svd(Gc, full_matrices=False)[2][0]
    bins = quartile_bins(pc1)
    tok = torch.cat(tokens, 0)[:token_rows_for_monitor]
    counts = np.bincount(bins, minlength=4)
    S = silhouette(X, bins)
    valid = bool(np.isfinite(S)) and bool((counts >= 2).all())
    return {"n_instances": int(X.shape[0]), "latent_dim": int(X.shape[1]),
            "bins": counts.tolist(),
            "S_valid": valid,
            "S_invalid_reason": (None if valid else
                                 "silhouette undefined: every bin needs >= 2 instances "
                                 f"(counts {counts.tolist()})"),
            "S_silhouette": S,
            "loo_1nn_bin_accuracy": loo_1nn_accuracy(X, bins),
            "sigreg_monitor_pooled": sigreg_monitor(torch.as_tensor(X, dtype=torch.float32),
                                                    n_proj=256),
            "sigreg_monitor_tokens": sigreg_monitor(tok, n_proj=256)}
