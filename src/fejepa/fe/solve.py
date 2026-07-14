"""Exact FE solves and the solve ledger.

Plan v2.0 mapping:
  - Sec.2.5 asset "Exact CG/direct solver". Its assembly-level guarantee (Gate G0 in the
    old code: solver reproduces U* to ~1e-10) is preserved as tests/test_solve.py rather
    than a runtime experiment -- plan v2.0's battery does not schedule G0.
  - E7 (CG polish): iterations-to-tolerance and k-step polish are built on these routines.
  - WP5 / B6 (solve ledger): every reference solve in the training economy is counted.

CG is deliberately *unpreconditioned* so E7's iteration counts are interpretable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class SolveLedger:
    """Counts labelled solves per pipeline stage (plan WP5: the data-asymmetry table)."""

    counts: dict = field(default_factory=dict)
    wall_clock_s: float = 0.0

    def add(self, stage: str, n: int = 1, seconds: float = 0.0) -> None:
        self.counts[stage] = self.counts.get(stage, 0) + int(n)
        self.wall_clock_s += float(seconds)

    @property
    def total(self) -> int:
        return int(sum(self.counts.values()))

    def as_dict(self) -> dict:
        return {"per_stage": dict(self.counts), "total": self.total,
                "wall_clock_s": round(self.wall_clock_s, 3)}


def _cg(A, b, x0=None, tol=1e-10, maxiter=None, count=False):
    """scipy CG across API generations (rtol vs tol); optionally count iterations."""
    it = 0

    def cb(_xk):
        nonlocal it
        it += 1

    kw = dict(x0=x0, maxiter=maxiter, callback=cb if count else None)
    try:
        x, info = spla.cg(A, b, rtol=tol, atol=0.0, **kw)
    except TypeError:                                     # scipy < 1.12
        x, info = spla.cg(A, b, tol=tol, atol=0.0, **kw)
    return x, info, it


def solve_fe_displacement(K: sp.csr_matrix, F: np.ndarray, free_mask: np.ndarray,
                          method: str = "cg", tol: float = 1e-10,
                          maxiter: int | None = None,
                          x0: np.ndarray | None = None,
                          ledger: SolveLedger | None = None,
                          stage: str = "solve"):
    """Solve K_ff u_f = F_f per load; zeros on Dirichlet dofs.

    ``x0``: optional (L, ndof) warm start for the CG path (plan WP3: the learned
    initializer); the direct path ignores it. Returns (U (L, ndof) float64, infos
    list of dicts). Falls back to a direct solve when CG fails to converge (this is
    exactness, not approximation -- Lemma 1).
    """
    F2 = np.atleast_2d(np.asarray(F, dtype=np.float64))
    free = np.asarray(free_mask, dtype=bool)
    X0 = None if x0 is None else np.atleast_2d(np.asarray(x0, dtype=np.float64))
    Kff = sp.csr_matrix(K)[free][:, free]
    lu = None
    U = np.zeros((F2.shape[0], F2.shape[1]), dtype=np.float64)
    infos = []
    t0 = time.perf_counter()
    for j in range(F2.shape[0]):
        b = F2[j][free]
        if method == "direct":
            if lu is None:
                lu = spla.splu(Kff.tocsc())
            uf, info = lu.solve(b), {"method": "direct"}
        else:
            xj = None if X0 is None else X0[j][free]
            x, code, it = _cg(Kff, b, x0=xj, tol=tol, maxiter=maxiter, count=True)
            if code != 0:                                  # non-convergence: exact fallback
                if lu is None:
                    lu = spla.splu(Kff.tocsc())
                uf, info = lu.solve(b), {"method": "cg->direct", "cg_iters": it}
            else:
                uf, info = x, {"method": "cg", "cg_iters": it}
        U[j][free] = uf
        infos.append(info)
    if ledger is not None:
        ledger.add(stage, n=F2.shape[0], seconds=time.perf_counter() - t0)
    return U, infos


def cg_iterations_to_tol(K, f: np.ndarray, free_mask: np.ndarray,
                         x0_full: np.ndarray | None, tol: float,
                         maxiter: int = 20000) -> tuple[int, bool, float]:
    """E7: CG iterations to relative tolerance from a full-vector initializer."""
    free = np.asarray(free_mask, dtype=bool)
    Kff = sp.csr_matrix(K)[free][:, free]
    x0 = None if x0_full is None else np.asarray(x0_full, dtype=np.float64)[free]
    t0 = time.perf_counter()
    _, code, it = _cg(Kff, np.asarray(f, dtype=np.float64)[free],
                      x0=x0, tol=tol, maxiter=maxiter, count=True)
    return it, code == 0, time.perf_counter() - t0


def cg_k_steps(K, f: np.ndarray, free_mask: np.ndarray,
               x0_full: np.ndarray | None, k: int) -> np.ndarray:
    """E7: the iterate after exactly k CG steps (or convergence, whichever first)."""
    free = np.asarray(free_mask, dtype=bool)
    if k <= 0:
        u = np.zeros(free.shape[0]) if x0_full is None else np.array(x0_full, dtype=np.float64)
        u[~free] = 0.0
        return u
    Kff = sp.csr_matrix(K)[free][:, free]
    x0 = None if x0_full is None else np.asarray(x0_full, dtype=np.float64)[free]
    x, _, _ = _cg(Kff, np.asarray(f, dtype=np.float64)[free], x0=x0, tol=1e-14, maxiter=k)
    u = np.zeros(free.shape[0], dtype=np.float64)
    u[free] = x
    return u
