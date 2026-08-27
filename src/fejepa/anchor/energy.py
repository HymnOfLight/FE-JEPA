"""The assembled-energy anchor: Lemma 1 made executable.

Plan v2.0 mapping: Sec.1 --

    Pi_h(u) = 0.5 u^T K u - F^T u,      grad Pi_h(u) = K u - F = K (u - U*),

so minimizing Pi_h is gradient-identical to supervised training in the energy norm,
without labels. The custom autograd op returns the *analytic* gradient (one sparse
mat-vec); Dirichlet dofs are masked before the mat-vec so the unconstrained kernel of
K never enters. Verified on the real corpus to <= 3.1e-11 (audit V1); preserved by
tests/test_anchor.py (gradient identity to machine precision).

Numpy-side identities (:func:`pi_h`, :func:`energy_gap`) feed the metric hierarchy
(plan Sec.4): gap(u) = Pi(u) - Pi(U*) = 0.5 ||u - U*||_K^2.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------- numpy side --

def pi_h(U: np.ndarray, K: sp.spmatrix, F: np.ndarray) -> np.ndarray:
    """Per-load assembled energy; U, F are (L, ndof) (or (ndof,))."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))
    F2 = np.atleast_2d(np.asarray(F, dtype=np.float64))
    KU = (sp.csr_matrix(K) @ U2.T).T
    return 0.5 * np.einsum("ld,ld->l", U2, KU) - np.einsum("ld,ld->l", F2, U2)


def energy_gap(U: np.ndarray, U_star: np.ndarray, K, F) -> np.ndarray:
    """Per-load Pi(U) - Pi(U*) = 0.5 ||U - U*||_K^2 >= 0."""
    return pi_h(U, K, F) - pi_h(U_star, K, F)


def pi_star_abs(arch) -> np.ndarray:
    """Per-load |Pi(U*)| = 0.5 ||U*||_K^2 -- the energy-gap normalizer (labels required)."""
    return np.abs(pi_h(arch.U_star, arch.K, arch.F))


# ---------------------------------------------------------------- torch side --

def _torch():
    import torch

    return torch


class EnergyAnchor:
    """Callable anchor for one instance; runs on CPU or CUDA without host round-trips."""

    def __init__(self, K: sp.spmatrix, F: np.ndarray, dirichlet_mask: np.ndarray,
                 device="cpu", dtype=None):
        torch = _torch()
        dtype = dtype or torch.float32
        Kc = sp.coo_matrix(K)
        idx = torch.as_tensor(np.vstack([Kc.row, Kc.col]), dtype=torch.long)
        val = torch.as_tensor(Kc.data, dtype=dtype)
        K_t = torch.sparse_coo_tensor(idx, val, size=Kc.shape,
                                      device=device).coalesce()
        # CSR spmm is the fast cuSPARSE path on CUDA (torch >= 2.x); COO stays the
        # portable CPU/float64 layout used by the machine-precision tests.
        self.K_t = K_t.to_sparse_csr() if str(device).startswith("cuda") else K_t
        self.F_t = torch.as_tensor(np.atleast_2d(F), dtype=dtype, device=device)
        self.free_t = torch.as_tensor(~np.asarray(dirichlet_mask, dtype=bool),
                                      device=device).to(dtype)
        self.device = device

    def energies(self, u):
        """Per-load Pi_h of a (L, ndof) or (ndof,) prediction; masks Dirichlet dofs."""
        import torch

        # Precision self-protection (r10): the anchor is the exact objective;
        # regardless of any surrounding autocast (bf16), energy math runs fp32.
        if u.dtype != self.K_t.dtype:
            u = u.to(self.K_t.dtype)
        with torch.autocast(device_type=u.device.type, enabled=False):
            return self._energies_fp32(u)

    def _energies_fp32(self, u):
        u2 = u if u.dim() == 2 else u.unsqueeze(0)
        u2 = u2 * self.free_t                      # autograd chains the mask
        return _EnergyFn.apply(u2, self.K_t, self.F_t)

    def __call__(self, u):
        return self.energies(u).mean()


def _make_energy_fn():
    torch = _torch()

    class EnergyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, K_t, F_t):
            with torch.no_grad():
                # .contiguous() keeps the CSR/cuSPARSE fast path (u.t() is a view)
                Ku = (K_t @ u.t().contiguous()).t()
            ctx.save_for_backward(Ku, F_t)
            return 0.5 * (u * Ku).sum(-1) - (F_t * u).sum(-1)

        @staticmethod
        def backward(ctx, grad_out):
            Ku, F_t = ctx.saved_tensors
            return grad_out.unsqueeze(-1) * (Ku - F_t), None, None

    return EnergyFn


class _Lazy:
    _fn = None

    @classmethod
    def apply(cls, *args):
        if cls._fn is None:
            cls._fn = _make_energy_fn()
        return cls._fn.apply(*args)


_EnergyFn = _Lazy


class AnchorCache:
    """One on-device anchor per instance, reused across epochs (verified asset)."""

    def __init__(self, device="cpu", dtype=None):
        self.device, self.dtype = device, dtype
        self._store: dict = {}

    def get(self, arch) -> EnergyAnchor:
        key = str(arch.path) if arch.path is not None else id(arch)
        if key not in self._store:
            self._store[key] = EnergyAnchor(arch.K, arch.F, arch.dirichlet_mask,
                                            device=self.device, dtype=self.dtype)
        return self._store[key]
