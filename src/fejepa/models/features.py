"""Node features and conditioning channels (pure numpy; torch-free by design so the
baselines and tests can import it without the ``torch`` extra).

Plan v2.0 mapping:
  - Base 6-dim node features are the verified-asset design: RMS-normalized centred
    coordinates (2), Dirichlet flags (2), per-node consistent load normalized by the
    battery-wide max (2).
  - ``load_summary`` channel (4-dim, broadcast): existing v4 asset (plan Sec.2.5).
  - ``geometry`` channel (6-dim, broadcast): plan WP0 -- "geometry-descriptor channel
    (from arch.meta, broadcast like the load summary)"; toggled by E3'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASE_DIM = 6
LOAD_SUMMARY_DIM = 4
GEOMETRY_DIM = 6


@dataclass
class FeatureSpec:
    load_summary: bool = True
    geometry: bool = True

    @property
    def dim(self) -> int:
        return (BASE_DIM
                + (LOAD_SUMMARY_DIM if self.load_summary else 0)
                + (GEOMETRY_DIM if self.geometry else 0))

    def to_dict(self) -> dict:
        return {"load_summary": self.load_summary, "geometry": self.geometry}

    @classmethod
    def from_dict(cls, d: dict | None) -> "FeatureSpec":
        d = d or {}
        return cls(load_summary=bool(d.get("load_summary", True)),
                   geometry=bool(d.get("geometry", True)))


def normalized_coords(nodes: np.ndarray) -> np.ndarray:
    c = nodes - nodes.mean(axis=0, keepdims=True)
    scale = np.sqrt((c ** 2).sum(axis=1).mean()) + 1e-8
    return c / scale


def geometry_descriptor(meta: dict) -> np.ndarray:
    """(6,) static-normalized geometry summary: what E3 diagnosed as the missing
    identity channel (audit V10/V11)."""
    ex = meta["extra"]
    w, h = float(ex["width"]), float(ex["height"])
    holes = ex.get("holes", []) or []
    hole_area = float(sum(np.pi * r * r for _, _, r in holes))
    mean_r = float(np.mean([r for _, _, r in holes])) if holes else 0.0
    return np.array([
        w / 3.0,
        h / 1.5,
        float(meta["material"]["nu"]),
        len(holes) / 3.0,
        hole_area / (w * h),
        mean_r / min(w, h),
    ], dtype=np.float64)


def load_summary(F: np.ndarray, load_idx: int) -> np.ndarray:
    """(4,) per-load global descriptor (v4 asset): total |f|, net fx, net fy, loaded frac."""
    fscale = np.abs(F).max() + 1e-12
    f = F[load_idx].reshape(-1, 2)
    n = f.shape[0]
    mag = np.linalg.norm(f, axis=1)
    return np.array([
        mag.sum() / (n * fscale),
        f[:, 0].sum() / (n * fscale),
        f[:, 1].sum() / (n * fscale),
        float((mag > 1e-14 * fscale).mean()),
    ], dtype=np.float64)


def build_features(arch, load_idx: int, spec: FeatureSpec) -> np.ndarray:
    """(N, spec.dim) float32 node features for one load case."""
    n = arch.n_nodes
    coords = normalized_coords(arch.nodes)
    dmask = arch.dirichlet_mask.reshape(-1, 2).astype(np.float64)
    fscale = np.abs(arch.F).max() + 1e-12
    f = arch.F[load_idx].reshape(-1, 2) / fscale
    cols = [coords, dmask, f]
    if spec.load_summary:
        cols.append(np.broadcast_to(load_summary(arch.F, load_idx), (n, LOAD_SUMMARY_DIM)))
    if spec.geometry:
        cols.append(np.broadcast_to(geometry_descriptor(arch.meta), (n, GEOMETRY_DIM)))
    return np.concatenate(cols, axis=1).astype(np.float32)


def build_features_battery(arch, spec: FeatureSpec) -> np.ndarray:
    """(L, N, spec.dim) -- one batched tensor for the whole load battery."""
    return np.stack([build_features(arch, j, spec) for j in range(arch.n_loads)], axis=0)
