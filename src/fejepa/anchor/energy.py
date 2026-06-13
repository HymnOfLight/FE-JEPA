r"""Differentiable assembled-energy anchor.

Implements the discrete total potential energy

.. math::

    \Pi_h(\hat u) = \tfrac12 \hat u^\top K \hat u - F^\top \hat u

and its gradient :math:`\nabla \Pi_h(\hat u) = K\hat u - F = K(\hat u - U^\star)`
(Lemma 1).  The matrix-vector product with the sparse stiffness is the only
operation per step; **no linear solve is performed**.  Homogeneous Dirichlet
degrees of freedom are masked to zero, exactly as in the assembled reduced
system, so the unconstrained singularity of ``K`` never enters.

The custom autograd op returns the *analytic* gradient ``K u - F``, which by
Lemma 1 equals the gradient of the supervised energy-norm loss
:math:`\tfrac12\|\hat u - U^\star\|_K^2` -- label-free and supervised training
coincide at the level of gradients.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


class _EnergyAnchorFn(torch.autograd.Function):
    """Autograd op for ``Pi_h`` with a constant sparse ``K`` (numpy mat-vec)."""

    @staticmethod
    def forward(ctx, u: torch.Tensor, K_csr: sp.csr_matrix, F: torch.Tensor):
        u_np = u.detach().cpu().numpy()
        F_np = F.detach().cpu().numpy()
        # K @ u for both (ndof,) and (B, ndof) layouts.
        Ku_np = (K_csr @ u_np.T).T
        Ku = torch.as_tensor(np.ascontiguousarray(Ku_np), dtype=u.dtype, device=u.device)
        energy_val = 0.5 * (u * Ku).sum(dim=-1) - (F * u).sum(dim=-1)
        ctx.save_for_backward(Ku, F)
        return energy_val

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        Ku, F = ctx.saved_tensors
        grad_u = Ku - F  # = K u - F
        grad_u = grad_output.unsqueeze(-1) * grad_u
        return grad_u, None, None


def energy(u: torch.Tensor, K_csr: sp.csr_matrix, F: torch.Tensor) -> torch.Tensor:
    """Return ``Pi_h(u)`` (per row if ``u`` is batched over load cases)."""

    return _EnergyAnchorFn.apply(u, K_csr, F)


def energy_norm_sq(diff: np.ndarray, K_csr: sp.csr_matrix) -> np.ndarray:
    r"""Return :math:`\|diff\|_K^2 = diff^\top K\, diff` (numpy, per row)."""

    diff = np.atleast_2d(diff)
    return np.einsum("bi,bi->b", diff, (K_csr @ diff.T).T)


def energy_gap(
    u: np.ndarray, U_star: np.ndarray, K_csr: sp.csr_matrix, F: np.ndarray
) -> np.ndarray:
    r"""Return the energy gap :math:`\Pi_h(u) - \Pi_h(U^\star)` (numpy, per row).

    By Lemma 1 this equals :math:`\tfrac12\|u - U^\star\|_K^2`; the evaluation
    harness reports this FE-native metric on the labelled split.
    """

    u = np.atleast_2d(u)
    U_star = np.atleast_2d(U_star)
    F = np.atleast_2d(F)
    pi_u = 0.5 * np.einsum("bi,bi->b", u, (K_csr @ u.T).T) - np.einsum("bi,bi->b", F, u)
    pi_s = 0.5 * np.einsum("bi,bi->b", U_star, (K_csr @ U_star.T).T) - np.einsum(
        "bi,bi->b", F, U_star
    )
    return pi_u - pi_s


class EnergyAnchor(torch.nn.Module):
    """Energy anchor for one FE instance (shared ``K``, a battery of ``F``).

    Wraps the sparse stiffness, the load battery, and the Dirichlet mask.  The
    forward pass masks the constrained dofs to zero and returns ``Pi_h`` per load
    case, summed (the loss) or per-row (diagnostics).
    """

    def __init__(
        self,
        K_csr: sp.csr_matrix,
        F: np.ndarray,
        free_mask: np.ndarray,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.K = K_csr.tocsr()
        F = np.atleast_2d(np.asarray(F))
        self.register_buffer("F", torch.as_tensor(F, dtype=dtype))
        self.register_buffer(
            "free_mask", torch.as_tensor(np.asarray(free_mask, dtype=bool))
        )
        self.n_loads = F.shape[0]
        self.n_dof = F.shape[1]

    def _apply_bc(self, u: torch.Tensor) -> torch.Tensor:
        return u * self.free_mask.to(u.dtype)

    def per_load(self, u: torch.Tensor) -> torch.Tensor:
        """``Pi_h`` per load case. ``u`` is ``(ndof,)`` or ``(n_loads, ndof)``."""

        if u.dim() == 1:
            u = u.unsqueeze(0).expand(self.n_loads, -1)
        u = self._apply_bc(u)
        return energy(u, self.K, self.F)

    def forward(self, u: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        vals = self.per_load(u)
        if reduction == "mean":
            return vals.mean()
        if reduction == "sum":
            return vals.sum()
        if reduction == "none":
            return vals
        raise ValueError(f"unknown reduction {reduction!r}")
