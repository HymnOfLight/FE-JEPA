"""Anti-collapse regularizers at the pooled (instance-level) granularity.

Plan v2.0 mapping: E3' modes {sigreg, sigreg_pooled, vicreg_pooled}. The audited lesson
(Sec.2.4/B4 and the v1 E3 kill): per-node SIGReg regularizes variation *within* one
instance, while collapse is measured on pooled latents *across* instances -- so the
pooled-granularity variants act on pooled vectors stacked with a ring buffer of recent
(detached) pooled latents, giving batch-1 training a cross-instance view.

``sigreg`` (per-node) is retained exactly so E3' can re-test the original placement
against the pooled placements under identical conditions.
"""

from __future__ import annotations

import numpy as np


def _torch():
    import torch

    return torch


def bhep_statistic(x):
    """Differentiable BHEP / Henze--Zirkler CF statistic of a 1-D sample vs N(0,1).

    Centred but NOT rescaled, so variance != 1 is penalized (the anti-collapse force).
    """
    torch = _torch()
    beta = 1.0
    x = x - x.mean()
    n = x.shape[0]
    d2 = (x.unsqueeze(0) - x.unsqueeze(1)) ** 2
    term1 = torch.exp(-0.5 * beta**2 * d2).mean() * n
    term2 = (2.0 / np.sqrt(1.0 + beta**2)) * torch.exp(
        -0.5 * beta**2 * x**2 / (1.0 + beta**2)).sum()
    term3 = n / np.sqrt(1.0 + 2.0 * beta**2)
    return (term1 - term2 + term3) / n


def _sigreg_2d(z, n_proj: int = 16, max_samples: int = 1024, generator=None):
    """Mean BHEP over random unit projections of a (M, d) sample."""
    torch = _torch()
    if z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z = z.index_select(0, idx)
    d = z.shape[1]
    dirs = torch.randn(n_proj, d, device=z.device, generator=generator)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)
    proj = z @ dirs.t()                                   # (M, n_proj)
    return torch.stack([bhep_statistic(proj[:, k]) for k in range(proj.shape[1])]).mean()


def sigreg_node(latents):
    """Original per-node placement: latents (L, N, d) flattened within the instance."""
    return _sigreg_2d(latents.reshape(-1, latents.shape[-1]))


class PooledBuffer:
    """Ring buffer of recent detached pooled latents (cross-instance memory)."""

    def __init__(self, size: int = 64):
        self.size = size
        self._items: list = []

    def push(self, pooled_rows) -> None:
        for row in pooled_rows.detach():
            self._items.append(row)
        self._items = self._items[-self.size:]

    def stacked_with(self, pooled_rows):
        torch = _torch()
        if not self._items:
            return pooled_rows
        buf = torch.stack(self._items, dim=0).to(pooled_rows.device)
        return torch.cat([pooled_rows, buf], dim=0)


def sigreg_pooled(pooled_rows, buffer: PooledBuffer):
    """SIGReg at the granularity E3 measures (plan WP0: the ~5-line variant)."""
    return _sigreg_2d(buffer.stacked_with(pooled_rows))


def vicreg_pooled(pooled_rows, buffer: PooledBuffer, var_target: float = 1.0,
                  cov_weight: float = 0.05):
    """VICReg-style variance hinge + covariance penalty on pooled latents."""
    torch = _torch()
    z = buffer.stacked_with(pooled_rows)
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + 1e-6)
    var_loss = torch.relu(var_target - std).mean()
    if z.shape[0] > 1:
        cov = (z.t() @ z) / (z.shape[0] - 1)
        off = cov - torch.diag(torch.diagonal(cov))
        cov_loss = (off ** 2).sum() / z.shape[1]
    else:
        cov_loss = z.sum() * 0.0
    return var_loss + cov_weight * cov_loss
