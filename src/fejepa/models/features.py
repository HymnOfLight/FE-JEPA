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

BASE_DIM = 6           # the 2D value of base_dim(); kept as the frozen constant
LOAD_SUMMARY_DIM = 4   # the 2D value of load_summary_dim(); kept as the frozen constant
GEOMETRY_DIM = 6

def battery_fscale(F: np.ndarray) -> float:
    """Battery-level load magnitude: normalises the load channels and, since WP7
    3D-P0.5, is multiplied back onto the decoded field (scale-equivariant decode).
    Assembly-level -- no solve involved."""
    return float(np.abs(F).max() + 1e-12)
       # dimension-independent by design (see geometry_descriptor)


def base_dim(spatial_dim: int = 2) -> int:
    """Coords + Dirichlet flags + load components: 3 * spatial_dim (2D: 6, 3D: 9)."""
    return 3 * int(spatial_dim)


def load_summary_dim(spatial_dim: int = 2) -> int:
    """Total |f| + per-component net sums + loaded fraction (2D: 4, 3D: 5)."""
    return int(spatial_dim) + 2


def spatial_dim_of(arch) -> int:
    """The instance's spatial dimension, read from the data itself (WP7 3D-P0)."""
    return int(arch.nodes.shape[1])


@dataclass
class FeatureSpec:
    load_summary: bool = True
    geometry: bool = True
    spatial_dim: int = 2   # WP7 3D-P0: 2 preserves every v2.1.5 shape and value

    @property
    def dim(self) -> int:
        return (base_dim(self.spatial_dim)
                + (load_summary_dim(self.spatial_dim) if self.load_summary else 0)
                + (GEOMETRY_DIM if self.geometry else 0))

    def to_dict(self) -> dict:
        return {"load_summary": self.load_summary, "geometry": self.geometry,
                "spatial_dim": int(self.spatial_dim)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "FeatureSpec":
        d = d or {}
        return cls(load_summary=bool(d.get("load_summary", True)),
                   geometry=bool(d.get("geometry", True)),
                   spatial_dim=int(d.get("spatial_dim", 2)))


def normalized_coords(nodes: np.ndarray) -> np.ndarray:
    c = nodes - nodes.mean(axis=0, keepdims=True)
    scale = np.sqrt((c ** 2).sum(axis=1).mean()) + 1e-8
    return c / scale


def geometry_descriptor(meta: dict) -> np.ndarray:
    """(6,) static-normalized geometry summary: what E3 diagnosed as the missing
    identity channel (audit V10/V11).

    WP7 3D-P0.2: dispatched on ``meta.extra.dim``. The 3D branch is the natural
    lift -- box extents (each normalized by its ``tet_instance`` sampling
    maximum, mirroring the 2D convention), Poisson ratio, cavity count, and
    cavity volume fraction (spherical cavities ``(x, y, z, r)``). The 2D
    mean-radius channel is exchanged for the depth extent so GEOMETRY_DIM
    stays 6 in both dimensions. The 2D branch is unchanged from v2.1.5."""
    ex = meta["extra"]
    if int(ex.get("dim", 2)) == 3:
        w, h, d = float(ex["width"]), float(ex["height"]), float(ex["depth"])
        holes = ex.get("holes", []) or []
        cav_vol = float(sum(4.0 / 3.0 * np.pi * r ** 3 for *_, r in holes))
        return np.array([
            w / 3.0,
            h / 1.5,
            d / 1.2,
            float(meta["material"]["nu"]),
            len(holes) / 3.0,
            cav_vol / (w * h * d),
        ], dtype=np.float64)
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


def load_summary(F: np.ndarray, load_idx: int, spatial_dim: int = 2) -> np.ndarray:
    """(spatial_dim + 2,) per-load global descriptor (v4 asset): total |f|, net
    per-component sums, loaded frac.

    WP7 3D-P0.3: the v2.1.5 body summed x/y only; the component sums now follow
    ``spatial_dim`` (z included in 3D). The default 2 makes every existing 2D
    call site return the v2.1.5 vector element-for-element."""
    sd = int(spatial_dim)
    fscale = battery_fscale(F)
    f = F[load_idx].reshape(-1, sd)
    n = f.shape[0]
    mag = np.linalg.norm(f, axis=1)
    return np.array([
        mag.sum() / (n * fscale),
        *(f[:, c].sum() / (n * fscale) for c in range(sd)),
        float((mag > 1e-14 * fscale).mean()),
    ], dtype=np.float64)


def build_features(arch, load_idx: int, spec: FeatureSpec) -> np.ndarray:
    """(N, spec.dim) float32 node features for one load case.

    WP7 3D-P0.3: base columns follow the instance's spatial dimension; the spec
    must agree (the encoder's input layer is sized from ``spec.dim`` before any
    data is seen, so a silent mismatch would train garbage -- fail loudly)."""
    n = arch.n_nodes
    sd = spatial_dim_of(arch)
    if int(spec.spatial_dim) != sd:
        raise ValueError(
            f"FeatureSpec.spatial_dim={spec.spatial_dim} but the instance is "
            f"{sd}D (nodes {arch.nodes.shape}); set model.features.spatial_dim="
            f"{sd} in the config (WP7 3D-P0.3)")
    coords = normalized_coords(arch.nodes)
    dmask = arch.dirichlet_mask.reshape(-1, sd).astype(np.float64)
    fscale = battery_fscale(arch.F)
    f = arch.F[load_idx].reshape(-1, sd) / fscale
    cols = [coords, dmask, f]
    if spec.load_summary:
        cols.append(np.broadcast_to(load_summary(arch.F, load_idx, sd),
                                    (n, load_summary_dim(sd))))
    if spec.geometry:
        cols.append(np.broadcast_to(geometry_descriptor(arch.meta), (n, GEOMETRY_DIM)))
    return np.concatenate(cols, axis=1).astype(np.float32)


def build_features_battery(arch, spec: FeatureSpec) -> np.ndarray:
    """(L, N, spec.dim) -- one batched tensor for the whole load battery."""
    return np.stack([build_features(arch, j, spec) for j in range(arch.n_loads)], axis=0)
