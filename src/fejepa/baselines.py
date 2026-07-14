"""Repaired sanity baselines (plan B2 / E5').

The audited defect: one global ridge map onto *raw* displacements across load cases
whose scales differ ~17x scores 3.7-3.8 -- worse than predicting zero (1.0) by 3.8x.
E5' therefore requires, in every headline table:

  - :func:`zero_predictor` -- rel-L2 == 1.0 identically, the honest floor;
  - :class:`GlobalPolyBaseline` -- the legacy naive, retained as a diagnostic row;
  - :class:`ScaleAwarePolyBaseline` -- the repaired naive: the poly supplies the field
    *direction*, a ridge model on legitimate case descriptors (load summary, geometry
    descriptor, load scale, mesh size) supplies the *norm*. No oracle information.
  - :class:`KNNFieldBaseline` -- nearest training geometry (descriptor space), field
    transported by scattered linear interpolation in normalized coordinates.

All are pure numpy/scipy and consume only labelled *pool-prefix* instances -- the same
labels the supervised arms buy (data-economy consistency, plan WP5).
"""

from __future__ import annotations

import numpy as np

from .models.features import geometry_descriptor, load_summary, normalized_coords


def zero_predictor(arch) -> np.ndarray:
    return np.zeros_like(arch.F)


# ------------------------------------------------------------- global poly ----

def _node_features(arch, j: int) -> np.ndarray:
    coords = normalized_coords(arch.nodes)
    dmask = arch.dirichlet_mask.reshape(-1, 2).astype(np.float64)
    fscale = np.abs(arch.F).max() + 1e-12
    f = arch.F[j].reshape(-1, 2) / fscale
    return np.concatenate([coords, dmask, f], axis=1)


def _poly(x: np.ndarray, degree: int = 2) -> np.ndarray:
    n, d = x.shape
    cols = [np.ones((n, 1)), x]
    if degree >= 2:
        for i in range(d):
            cols.append(x[:, i:i + 1] * x[:, i:])
    return np.concatenate(cols, axis=1)


class GlobalPolyBaseline:
    """Degree-2 ridge from node features to raw displacements (the legacy naive)."""

    def __init__(self, ridge: float = 1e-4):
        self.ridge = ridge
        self.W: np.ndarray | None = None

    def fit(self, train_archs) -> "GlobalPolyBaseline":
        X, Y = [], []
        for a in train_archs:
            for j in range(a.n_loads):
                X.append(_poly(_node_features(a, j)))
                Y.append(a.U_star[j].reshape(-1, 2))
        X, Y = np.concatenate(X), np.concatenate(Y)
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.W = np.linalg.solve(A, X.T @ Y)
        return self

    def predict(self, arch) -> np.ndarray:
        U = np.zeros_like(arch.F)
        for j in range(arch.n_loads):
            U[j] = (_poly(_node_features(arch, j)) @ self.W).reshape(-1)
        return U * arch.free_mask


# -------------------------------------------------------- scale-aware poly ----

def _case_descriptor(arch, j: int) -> np.ndarray:
    fscale = np.abs(arch.F).max() + 1e-12
    return np.concatenate([
        load_summary(arch.F, j),
        geometry_descriptor(arch.meta),
        [np.log(fscale), np.log(arch.n_nodes)],
    ])


class ScaleAwarePolyBaseline:
    """Repaired naive: poly direction x ridge-predicted log-norm from case descriptors."""

    def __init__(self, ridge: float = 1e-4):
        self.poly = GlobalPolyBaseline(ridge)
        self.ridge = ridge
        self.w_norm: np.ndarray | None = None

    def fit(self, train_archs) -> "ScaleAwarePolyBaseline":
        self.poly.fit(train_archs)
        X, y = [], []
        for a in train_archs:
            for j in range(a.n_loads):
                X.append(_case_descriptor(a, j))
                y.append(np.log(np.linalg.norm(a.U_star[j]) + 1e-30))
        X = np.concatenate([np.ones((len(X), 1)), np.stack(X)], axis=1)
        y = np.asarray(y)
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.w_norm = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, arch) -> np.ndarray:
        U = self.poly.predict(arch)
        for j in range(arch.n_loads):
            d = np.concatenate([[1.0], _case_descriptor(arch, j)])
            target = np.exp(float(d @ self.w_norm))
            U[j] *= target / (np.linalg.norm(U[j]) + 1e-30)
        return U


# ------------------------------------------------------------- k-NN field ----

class KNNFieldBaseline:
    """Nearest training geometry; field transported by scattered linear interpolation
    in normalized coordinates (nearest-neighbour fill outside the hull).

    The Delaunay triangulation of a neighbour's point set is what makes
    LinearNDInterpolator expensive; it depends only on the points, so it is built
    lazily once per stored instance and reused across loads, components, and queries
    (a large speedup for E5'/E7 with n_val=256)."""

    def __init__(self, k: int = 1):
        if k != 1:
            raise NotImplementedError("plan E5' specifies nearest-neighbour transport")
        self._store: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._tri: dict[int, object] = {}

    def fit(self, train_archs) -> "KNNFieldBaseline":
        for a in train_archs:
            self._store.append((geometry_descriptor(a.meta),
                                normalized_coords(a.nodes),
                                a.U_star.copy()))
        return self

    def _triangulation(self, i: int):
        if i not in self._tri:
            from scipy.spatial import Delaunay

            self._tri[i] = Delaunay(self._store[i][1])
        return self._tri[i]

    def predict(self, arch) -> np.ndarray:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        q = geometry_descriptor(arch.meta)
        i = int(np.argmin([np.linalg.norm(q - g) for g, _, _ in self._store]))
        _, pts, U_nb = self._store[i]
        tri = self._triangulation(i)
        xq = normalized_coords(arch.nodes)
        U = np.zeros_like(arch.F)
        for j in range(arch.n_loads):
            vals = U_nb[j].reshape(-1, 2)
            for c in range(2):
                out = LinearNDInterpolator(tri, vals[:, c])(xq)
                bad = np.isnan(out)
                if bad.any():
                    out[bad] = NearestNDInterpolator(pts, vals[:, c])(xq[bad])
                U[j, c::2] = out
        return U * arch.free_mask


BASELINE_FACTORIES = {
    "zero": None,  # handled directly: predict = zero_predictor
    "poly": GlobalPolyBaseline,
    "scale_aware_poly": ScaleAwarePolyBaseline,
    "knn_field": KNNFieldBaseline,
}
