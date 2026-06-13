r"""SIGReg: sketched isotropic-Gaussian regularisation for collapse control.

Following LeJEPA, we enforce that embeddings are isotropic Gaussian by matching
the distribution of random 1D projections to a standard Gaussian via a
characteristic-function (CF) test.  We use the BHEP / Henze--Zirkler statistic,
which is the squared distance between the empirical CF of standardised data and
the Gaussian CF, integrated against a Gaussian kernel of bandwidth ``beta``:

.. math::

    T = \tfrac1n \sum_{j,k} e^{-\tfrac{\beta^2}{2}(x_j-x_k)^2}
        - \tfrac{2}{\sqrt{1+\beta^2}} \sum_j e^{-\tfrac{\beta^2 x_j^2}{2(1+\beta^2)}}
        + \tfrac{n}{\sqrt{1+2\beta^2}} .

Projections are centred but **not** rescaled, so the statistic also penalises
variance :math:`\ne 1` -- which is exactly the anti-collapse force (a collapsed
embedding has near-zero projected variance and is far from ``N(0,1)``).  This is
a linear-time (in projection count), single-hyperparameter, symmetric objective
that replaces EMA teachers and stop-gradients.

This is a faithful, self-contained implementation of the CF-test idea behind
SIGReg; it is not byte-for-byte the reference LeJEPA kernel.
"""

from __future__ import annotations

import math

import torch


def _bhep_statistic(x: torch.Tensor, beta: float) -> torch.Tensor:
    """BHEP normality statistic for a 1D sample ``x`` of shape ``(n,)``."""

    n = x.shape[0]
    x = x - x.mean()
    b2 = beta * beta

    diff = x.unsqueeze(0) - x.unsqueeze(1)  # (n, n)
    term1 = torch.exp(-0.5 * b2 * diff.pow(2)).mean()  # (1/n^2) sum * n -> use mean*... 
    # term1 above = (1/n^2) sum; multiply by n to match (1/n) sum_{j,k}
    term1 = term1 * n
    term2 = (2.0 / math.sqrt(1.0 + b2)) * torch.exp(
        -0.5 * b2 / (1.0 + b2) * x.pow(2)
    ).sum()
    term3 = n / math.sqrt(1.0 + 2.0 * b2)
    return (term1 - term2 + term3) / n


def sigreg_loss(
    z: torch.Tensor,
    n_proj: int = 64,
    beta: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """SIGReg loss for an embedding batch ``z`` of shape ``(n, dim)``.

    Samples ``n_proj`` random unit directions, projects, and averages the BHEP
    statistic.  Lower is closer to isotropic standard Gaussian.
    """

    if z.dim() != 2:
        raise ValueError("z must be (n, dim)")
    n, dim = z.shape
    if n < 4:
        return z.new_zeros(())

    dirs = torch.randn(dim, n_proj, generator=generator, device=z.device, dtype=z.dtype)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)
    proj = z @ dirs  # (n, n_proj)

    stats = torch.stack([_bhep_statistic(proj[:, k], beta) for k in range(n_proj)])
    return stats.mean()
