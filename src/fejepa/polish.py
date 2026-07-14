"""Predict-then-polish inference: fe/solve wired into evaluation and inference
(plan WP3 / claim C3).

:func:`polished` upgrades ANY battery predictor -- a trained model wrapped by
:func:`fejepa.metrics.torch_predictor`, or a numpy baseline -- into a hybrid solver:
the prediction seeds unpreconditioned CG on the exact assembled ``(K, F)``.

Two modes, exactly one of which must be chosen:
  * ``k``   -- a fixed number of polish steps: the C3 trade of network quality for
               solver work at the Chebyshev rate on the energy gap;
  * ``tol`` -- solve to tolerance from the warm start: an *exact* answer whose cost
               is the E7-measured iterations-to-tolerance.

Everything is label-free at inference (only K, F, and the prediction are touched).
Ledger accounting: only the ``tol`` mode performs true reference-grade solves and is
counted (stage ``polish-inference``); fixed-``k`` passes are deliberately not counted
as solves.
"""

from __future__ import annotations

import numpy as np

from .fe.solve import SolveLedger, cg_k_steps, solve_fe_displacement


def polish_battery(arch, U0: np.ndarray, k: int | None = None,
                   tol: float | None = None,
                   ledger: SolveLedger | None = None) -> np.ndarray:
    """Polish a (L, ndof) prediction for one instance; returns (L, ndof) float64."""
    if (k is None) == (tol is None):
        raise ValueError("polish_battery: choose exactly one of k / tol")
    U0 = np.asarray(U0, dtype=np.float64)
    if tol is not None:
        U, _ = solve_fe_displacement(arch.K, arch.F, arch.free_mask, method="cg",
                                     tol=tol, x0=U0, ledger=ledger,
                                     stage="polish-inference")
        return U
    return np.stack([cg_k_steps(arch.K, arch.F[j], arch.free_mask, U0[j], k)
                     for j in range(arch.n_loads)])


def polished(base_predict_fn, k: int | None = None, tol: float | None = None,
             ledger: SolveLedger | None = None):
    """Wrap a predictor ``fn(arch) -> (L, ndof)`` into its polished version.

    Compatible with :func:`fejepa.metrics.evaluate_model`, so polished models drop
    into any table/experiment unchanged.
    """
    def fn(arch):
        return polish_battery(arch, base_predict_fn(arch), k=k, tol=tol,
                              ledger=ledger)

    return fn
