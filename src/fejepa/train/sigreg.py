"""SIGReg -- Sketched Isotropic Gaussian Regularization (wp8-lejepa, Stage 0).

After Balestriero & LeCun, "LeJEPA" (arXiv:2511.08544) and Maes et al.,
"LeWorldModel" (arXiv:2603.19312): push a set of embeddings toward an
isotropic standard Gaussian by (i) projecting onto M random unit directions
(Cramér–Wold: matching every 1-D marginal matches the joint) and (ii) applying
the Epps–Pulley normality statistic to each 1-D projection.

Two evaluators of the same statistic are provided:
  * `epps_pulley_knots` -- the empirical characteristic function integrated on
    a fixed grid of knots against the N(0,1) weight: O(N * K) per direction,
    the form that keeps SIGReg linear in the number of samples;
  * `epps_pulley_closed` -- the closed form O(N^2), used to cross-check the
    knot discretisation in the tests.
`sigreg` is the differentiable regulariser; `sigreg_monitor` evaluates it
without gradients as a collapse instrument (replacing the standardised
effective-rank probe whose 1e-12 floor was recorded as unreliable).

FE-JEPA usage note: our batch is one instance, so the sample set is the
per-node token set of that instance (N = nodes, d = dim). Mesh tokens are
spatially correlated, not i.i.d., so the statistic is a regulariser and a
relative monitor, not a calibrated hypothesis test.
"""

from __future__ import annotations

import math


def _knots(n_knots: int, t_max: float, device, dtype):
    import torch

    t = torch.linspace(-t_max, t_max, n_knots, device=device, dtype=dtype)
    w = torch.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)     # N(0,1) weight
    dt = (2.0 * t_max) / (n_knots - 1)
    return t, w, dt


def epps_pulley_knots(h, n_knots: int = 17, t_max: float = 5.0):
    """Epps–Pulley statistic for 1-D samples `h` of shape (N,) or (M, N)
    (one row per projection), integrated on `n_knots` knots in [-t_max, t_max]."""
    import torch

    h2 = h if h.dim() == 2 else h.unsqueeze(0)
    n = h2.shape[-1]
    t, w, dt = _knots(n_knots, t_max, h2.device, h2.dtype)
    th = h2.unsqueeze(-1) * t                                    # (M, N, K)
    re = torch.cos(th).mean(dim=1)                               # (M, K)
    im = torch.sin(th).mean(dim=1)
    phi0 = torch.exp(-0.5 * t * t)                               # target ECF
    diff2 = (re - phi0) ** 2 + im ** 2
    out = n * (diff2 * w).sum(dim=-1) * dt                       # (M,)
    return out if h.dim() == 2 else out[0]


def epps_pulley_closed(h):
    """Closed form of the same statistic (Gaussian weight):
    T = (1/N) sum_jk exp(-(x_j-x_k)^2/2) - sqrt(2) sum_j exp(-x_j^2/4) + N/sqrt(3)."""
    import torch

    x = h.reshape(-1)
    n = x.numel()
    d2 = (x.unsqueeze(0) - x.unsqueeze(1)) ** 2
    return (torch.exp(-0.5 * d2).sum() / n
            - math.sqrt(2.0) * torch.exp(-0.25 * x * x).sum()
            + n / math.sqrt(3.0))


def random_directions(d: int, n_proj: int, device, dtype, generator=None):
    import torch

    u = torch.randn(n_proj, d, device=device, dtype=dtype, generator=generator)
    return u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)


def sigreg(z, n_proj: int = 1024, n_knots: int = 17, t_max: float = 5.0,
           generator=None):
    """Differentiable SIGReg loss for embeddings `z` of shape (N, d):
    mean over `n_proj` random unit directions of the Epps–Pulley statistic of
    the projected samples, normalised by N so the scale is O(1)."""
    z2 = z.reshape(-1, z.shape[-1])
    u = random_directions(z2.shape[-1], n_proj, z2.device, z2.dtype, generator)
    h = z2 @ u.t()                                               # (N, M)
    stat = epps_pulley_knots(h.t(), n_knots=n_knots, t_max=t_max)   # (M,)
    return stat.mean() / z2.shape[0]


def sigreg_monitor(z, n_proj: int = 256, seed: int = 0) -> float:
    """Gradient-free SIGReg value (collapse / anisotropy instrument)."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        zc = z.detach().reshape(-1, z.shape[-1]).cpu()
        return float(sigreg(zc, n_proj=n_proj, generator=g))
