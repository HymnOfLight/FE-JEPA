"""Naive, solve-free baselines used as sanity references (E5).

The naive polynomial surrogate fits a single global ridge regression mapping
per-node features ``[x, y, dir_x, dir_y, fx, fy]`` (expanded to degree-2
polynomial features) to the nodal displacement, pooled over all labelled
training nodes.  It ignores the global elliptic coupling of elasticity entirely,
so any competent surrogate must beat it -- the SimJEB-style sanity check the
proposal's E5 falsification test demands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fejepa.data.archive import InstanceArchive
from fejepa.metrics import relative_l2


def _poly_features(x: np.ndarray, degree: int = 2) -> np.ndarray:
    """Polynomial feature expansion (bias + linear + up to ``degree`` terms)."""

    n, d = x.shape
    cols = [np.ones((n, 1)), x]
    if degree >= 2:
        # all pairwise products i<=j (includes squares)
        for i in range(d):
            cols.append(x[:, i : i + 1] * x[:, i:])
    return np.concatenate(cols, axis=1)


def _node_feature_matrix(arch: InstanceArchive, load_idx: int) -> np.ndarray:
    nodes = np.asarray(arch.nodes, dtype=np.float64)
    centre = nodes.mean(axis=0, keepdims=True)
    coords = nodes - centre
    scale = np.sqrt((coords**2).sum(axis=1).mean()) + 1e-8
    coords = coords / scale
    dmask = arch.dirichlet_mask.reshape(-1, 2).astype(np.float64)
    fscale = np.abs(arch.F).max() + 1e-12
    f = arch.F[load_idx].reshape(-1, 2) / fscale
    return np.concatenate([coords, dmask, f], axis=1)


@dataclass
class NaivePolynomialSurrogate:
    """A fitted global ridge-regression displacement predictor."""

    weights: np.ndarray  # (n_poly, 2)
    degree: int

    def predict(self, arch: InstanceArchive, load_idx: int) -> np.ndarray:
        X = _poly_features(_node_feature_matrix(arch, load_idx), self.degree)
        pred = X @ self.weights  # (n_nodes, 2)
        u = pred.reshape(-1) * arch.free_mask
        return u


def fit_naive_polynomial(
    train_archs: list[InstanceArchive], degree: int = 2, ridge: float = 1e-4
) -> NaivePolynomialSurrogate:
    Xs, Ys = [], []
    for arch in train_archs:
        for j in range(arch.n_loads):
            X = _poly_features(_node_feature_matrix(arch, j), degree)
            Y = arch.U_star[j].reshape(-1, 2)
            Xs.append(X)
            Ys.append(Y)
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    A = X.T @ X + ridge * np.eye(X.shape[1])
    W = np.linalg.solve(A, X.T @ Y)
    return NaivePolynomialSurrogate(weights=W, degree=degree)


def evaluate_naive(
    surrogate: NaivePolynomialSurrogate, val_archs: list[InstanceArchive]
) -> dict:
    rels = []
    for arch in val_archs:
        for j in range(arch.n_loads):
            pred = surrogate.predict(arch, j)
            rels.append(relative_l2(pred, arch.U_star[j]))
    return {"val_rel_l2_disp": float(np.mean(rels))}
