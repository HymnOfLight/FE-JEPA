"""Sparse symmetric positive definite solves.

Backends, chosen per system size:
  "pardiso"  Intel MKL PARDISO through pypardiso (real symmetric positive definite, upper triangle),
             used for systems above PARDISO_MIN unknowns when pypardiso can be loaded;
  "direct"   SuperLU (scipy), for small systems, or up to DIRECT_MAX unknowns without PARDISO;
  "amg-cg"   conjugate gradients preconditioned by smoothed aggregation AMG, otherwise.
The environment variable FEJOINT_SOLVER = "legacy" restores the behaviour before PARDISO was added
(SuperLU up to DIRECT_MAX unknowns, AMG-CG above), which is what gate G-C1a ran with.
"""
from __future__ import annotations
import glob
import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DIRECT_MAX = 45000
PARDISO_MIN = 3000


def _load_pardiso():
    if "PYPARDISO_MKL_RT" not in os.environ:
        for d in (os.path.join(sys.prefix, "lib"), "/usr/local/lib", "/usr/lib"):
            hits = sorted(glob.glob(os.path.join(d, "libmkl_rt.so*")))
            if hits:
                os.environ["PYPARDISO_MKL_RT"] = hits[0]
                break
    try:
        import pypardiso
        return pypardiso
    except Exception:
        return None


_PARDISO = None if os.environ.get("FEJOINT_SOLVER", "auto") == "legacy" else _load_pardiso()


class SPDSolver:
    def __init__(self, A: sp.spmatrix, near_null: np.ndarray | None = None, rtol: float = 1e-12):
        self.A = A.tocsr()
        self.n = A.shape[0]
        self.rtol = rtol
        if _PARDISO is not None and self.n > PARDISO_MIN:
            self.kind = "pardiso"
            self.ps = _PARDISO.PyPardisoSolver(mtype=2)
            self.U = sp.triu(self.A, format="csr")
            self.U.sort_indices()
            self.ps.factorize(self.U)
        elif self.n <= DIRECT_MAX:
            self.kind = "direct"
            self.lu = spla.splu(self.A.tocsc())
        else:
            self.kind = "amg-cg"
            import pyamg
            B = near_null if near_null is not None else None
            self.ml = pyamg.smoothed_aggregation_solver(self.A, B=B, max_coarse=500)
            self.M = self.ml.aspreconditioner(cycle="V")
        self.last_iters = 0

    def solve(self, b: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        if self.kind == "pardiso":
            x = self.ps.solve(self.U, np.ascontiguousarray(b, dtype=float))
            r = b - self.A @ x                                  # one step of iterative refinement
            return x + self.ps.solve(self.U, r)
        if self.kind == "direct":
            return self.lu.solve(b)
        it = [0]
        def cb(_):
            it[0] += 1
        x, info = spla.cg(self.A, b, x0=x0, rtol=self.rtol, atol=0.0, M=self.M, maxiter=5000, callback=cb)
        self.last_iters = it[0]
        if info != 0:
            raise RuntimeError(f"CG did not converge (info={info})")
        return x

    def free(self):
        if self.kind == "pardiso":
            self.ps.free_memory(everything=True)
