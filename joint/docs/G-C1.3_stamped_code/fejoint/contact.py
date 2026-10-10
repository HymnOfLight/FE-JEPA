"""Frictionless unilateral contact with a rigid flat support (discrete Signorini problem).

    minimise  Pi(x) = 1/2 x'Ax - b'x   subject to   x[c] >= -g[c]

A is symmetric positive definite (Dirichlet conditions already removed). c indexes the
constrained unknowns: nodal normal displacements of candidate contact nodes (positive = away
from the support) or, after a linear change of variables, relative separations such as the gap
between a bolt head and the plate (docs/contact_energy_note.md, Section 7). g >= 0 is the
initial gap (zero for a joint that starts in contact). With g = 0 the feasible set is a closed
convex cone.

Solved with the primal-dual active set strategy (Hintermueller, Ito and Kunisch, 2002), which
is a semismooth Newton method for the complementarity conditions

    A x - b = lam,  lam[c] >= 0,  x[c] + g >= 0,  lam[c] * (x[c] + g) = 0,  lam[other] = 0.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .linsolve import SPDSolver


def solve_signorini(A: sp.spmatrix, b: np.ndarray, cidx: np.ndarray, g: np.ndarray | None = None,
                    near_null: np.ndarray | None = None, max_it: int = 100, active0: np.ndarray | None = None):
    A = A.tocsr()
    n = A.shape[0]
    cidx = np.asarray(cidx)
    g = np.zeros(cidx.size) if g is None else np.asarray(g, dtype=float)
    rho = float(A.diagonal()[cidx].mean())
    act = np.ones(cidx.size, bool) if active0 is None else np.asarray(active0, bool).copy()
    history = []
    for it in range(1, max_it + 1):
        fixed = np.zeros(n, bool); fixed[cidx[act]] = True
        x = np.zeros(n)
        x[cidx[act]] = -g[act]
        free = ~fixed
        rhs = b[free] - A[free][:, fixed] @ x[fixed]
        S = SPDSolver(A[free][:, free], near_null=None if near_null is None else near_null[free])
        x[free] = S.solve(rhs)
        S.free()
        lam = A @ x - b
        new = (lam[cidx] - rho * (x[cidx] + g)) > 0
        history.append(int(act.sum()))
        if np.array_equal(new, act):
            return _report(A, b, x, lam, cidx, g, act, it, history)
        act = new
    raise RuntimeError("active set did not settle within max_it iterations")


def _report(A, b, x, lam, cidx, g, act, it, history):
    """Optimality residuals, relative to the largest load and displacement components:
    min_gap (>= 0 wanted), min_lambda (>= 0), max_complementarity (= 0), max_lambda_inactive
    (multipliers on constrained unknowns that are not in contact, = 0) and
    max_residual_unconstrained (equilibrium residual on unknowns without a constraint, = 0)."""
    other = np.ones(A.shape[0], bool); other[cidx] = False
    scale_f = max(np.abs(b).max(), 1e-300)
    scale_u = max(np.abs(x).max(), 1e-300)
    inactive = cidx[~act]
    kkt = {"min_gap": float((x[cidx] + g).min() / scale_u),
           "min_lambda": float(lam[cidx].min() / scale_f),
           "max_complementarity": float(np.abs(lam[cidx] * (x[cidx] + g)).max() / (scale_f * scale_u)),
           "max_lambda_inactive": float(np.abs(lam[inactive]).max() / scale_f) if inactive.size else 0.0,
           "max_residual_unconstrained": float(np.abs(lam[other]).max() / scale_f) if other.any() else 0.0}
    return {"x": x, "lam": lam, "active": act, "iterations": it, "active_history": history, "kkt": kkt}


def solve_tied(A: sp.spmatrix, b: np.ndarray, cidx: np.ndarray, near_null: np.ndarray | None = None):
    """Bonded support: x[c] = 0 both ways (the 'tied' model)."""
    A = A.tocsr()
    n = A.shape[0]
    fixed = np.zeros(n, bool); fixed[cidx] = True
    free = ~fixed
    x = np.zeros(n)
    S = SPDSolver(A[free][:, free], near_null=None if near_null is None else near_null[free])
    x[free] = S.solve(b[free])
    return {"x": x, "lam": A @ x - b}
