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


def csr_to_torch_sparse(
    K_csr: sp.csr_matrix, dtype: torch.dtype = torch.float32, device="cpu"
) -> torch.Tensor:
    """Convert a SciPy CSR matrix to a coalesced torch sparse COO tensor.

    Using a torch sparse tensor keeps the mat-vec on the same device as the
    decoded field, so the anchor runs on CPU **or** CUDA without host round-trips.
    """

    coo = K_csr.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.as_tensor(coo.data, dtype=dtype)
    # The stiffness comes from a coalesced SciPy CSR, so invariants hold; opt out
    # of the check explicitly to avoid a noisy one-time UserWarning.
    with torch.sparse.check_sparse_tensor_invariants(False):
        K = torch.sparse_coo_tensor(indices, values, tuple(coo.shape))
    return K.coalesce().to(device)


def _spmv(K_sparse: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``K @ u`` for ``u`` of shape ``(ndof,)`` or ``(B, ndof)`` (rows)."""

    if u.dim() == 1:
        return torch.sparse.mm(K_sparse, u.unsqueeze(1)).squeeze(1)
    return torch.sparse.mm(K_sparse, u.t()).t()


class _EnergyAnchorFn(torch.autograd.Function):
    """Autograd op for ``Pi_h`` returning the analytic gradient ``K u - F``."""

    @staticmethod
    def forward(ctx, u: torch.Tensor, K_sparse: torch.Tensor, F: torch.Tensor):
        with torch.no_grad():
            Ku = _spmv(K_sparse, u)
        energy_val = 0.5 * (u * Ku).sum(dim=-1) - (F * u).sum(dim=-1)
        ctx.save_for_backward(Ku, F)
        return energy_val

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        Ku, F = ctx.saved_tensors
        grad_u = Ku - F  # = K u - F  (Lemma 1)
        grad_u = grad_output.unsqueeze(-1) * grad_u
        return grad_u, None, None


def energy(u: torch.Tensor, K_sparse: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Return ``Pi_h(u)`` (per row if ``u`` is batched over load cases).

    ``K_sparse`` is a torch sparse tensor (see :func:`csr_to_torch_sparse`).
    """

    return _EnergyAnchorFn.apply(u, K_sparse, F)


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
        device="cpu",
    ):
        super().__init__()
        F = np.atleast_2d(np.asarray(F))
        self.K_sparse = csr_to_torch_sparse(K_csr.tocsr(), dtype=dtype, device=device)
        self.register_buffer("F", torch.as_tensor(F, dtype=dtype, device=device))
        self.register_buffer(
            "free_mask",
            torch.as_tensor(np.asarray(free_mask, dtype=bool), device=device),
        )
        self.n_loads = F.shape[0]
        self.n_dof = F.shape[1]

    def to(self, *args, **kwargs):  # keep the sparse K in sync with .to()
        super().to(*args, **kwargs)
        device = self.F.device
        self.K_sparse = self.K_sparse.to(device=device)
        return self

    def _apply_bc(self, u: torch.Tensor) -> torch.Tensor:
        return u * self.free_mask.to(u.dtype)

    def per_load(self, u: torch.Tensor) -> torch.Tensor:
        """``Pi_h`` per load case. ``u`` is ``(ndof,)`` or ``(n_loads, ndof)``."""

        if u.dim() == 1:
            u = u.unsqueeze(0).expand(self.n_loads, -1)
        u = self._apply_bc(u)
        return energy(u, self.K_sparse, self.F)

    def forward(self, u: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        vals = self.per_load(u)
        if reduction == "mean":
            return vals.mean()
        if reduction == "sum":
            return vals.sum()
        if reduction == "none":
            return vals
        raise ValueError(f"unknown reduction {reduction!r}")
