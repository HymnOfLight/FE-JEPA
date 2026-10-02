"""Node features and conditioning channels (pure numpy; torch-free by design so the
baselines and tests can import it without the ``torch`` extra).

Plan v2.0 mapping:
  - Base 6-dim node features are the verified-asset design: RMS-normalized centred
    coordinates (2), Dirichlet flags (2), per-node consistent load normalized by the
    battery-wide max (2).
  - ``load_summary`` channel (4-dim, broadcast): existing v4 asset (plan Sec.2.5).
  - ``geometry`` channel (6-dim, broadcast): plan WP0 -- "geometry-descriptor channel
    (from arch.meta, broadcast like the load summary)"; toggled by E3'.
  - wp9 S (``FeatureSpec.load_density``, default off): the load columns and the load
    summary in mesh-independent form -- per-node load DENSITIES (nodal force over the
    measure the node carries) and load-summary entries that are ratios of mesh-free
    totals. Off, every column is the v2.1.5 / wp8 value bit for bit.
"""

from __future__ import annotations

import math
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


def battery_l1(F: np.ndarray) -> float:
    """wp9 S: the battery's total load, sum over load cases and dofs of |F|. For
    consistent nodal loads it is the integral of |t_c| over the loaded boundary
    plus that of |b_c| over the body, summed over components and load cases, so
    it does not change with the mesh (the largest nodal force, `battery_fscale`,
    shrinks like the element size); like it, it is homogeneous of degree one in
    the loads. Assembly-level -- no solve involved."""
    return float(np.abs(F).sum() + 1e-12)


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
    load_density: bool = False   # wp9 S: mesh-independent load columns and summary

    @property
    def dim(self) -> int:
        return (base_dim(self.spatial_dim)
                + (load_summary_dim(self.spatial_dim) if self.load_summary else 0)
                + (GEOMETRY_DIM if self.geometry else 0))

    def to_dict(self) -> dict:
        d = {"load_summary": self.load_summary, "geometry": self.geometry,
             "spatial_dim": int(self.spatial_dim)}
        if self.load_density:           # only when on: the default dict is unchanged
            d["load_density"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> "FeatureSpec":
        d = d or {}
        return cls(load_summary=bool(d.get("load_summary", True)),
                   geometry=bool(d.get("geometry", True)),
                   spatial_dim=int(d.get("spatial_dim", 2)),
                   load_density=bool(d.get("load_density", False)))


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


# ------------------------------------------------- wp9 S: load densities --

def simplex_geometry(nodes: np.ndarray, elements: np.ndarray) -> tuple:
    """(cell measures (E,), boundary facets (B, d) as node indices, facet
    measures (B,)) of a simplicial mesh: triangles in 2D (areas; boundary edges
    and their lengths), tetrahedra in 3D (volumes; boundary triangles and their
    areas). A boundary facet belongs to exactly one cell (holes included)."""
    X = np.asarray(nodes, dtype=np.float64)
    el = np.asarray(elements, dtype=np.int64)
    sd, k = X.shape[1], el.shape[1]
    if k != sd + 1:
        raise ValueError(f"simplex_geometry: {k}-node cells in {sd}D (simplices only)")
    E = X[el[:, 1:]] - X[el[:, :1]]                       # (E, sd, sd)
    vol = np.abs(np.linalg.det(E)) / math.factorial(sd)
    fac = np.sort(np.concatenate([np.delete(el, j, axis=1) for j in range(k)]), axis=1)
    uniq, cnt = np.unique(fac, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    if sd == 2:
        fm = np.linalg.norm(X[bnd[:, 1]] - X[bnd[:, 0]], axis=1)
    else:
        fm = 0.5 * np.linalg.norm(np.cross(X[bnd[:, 1]] - X[bnd[:, 0]],
                                           X[bnd[:, 2]] - X[bnd[:, 0]]), axis=1)
    return vol, bnd, fm


def _lumped(n: int, simp: np.ndarray, meas: np.ndarray) -> np.ndarray:
    """Each simplex's measure shared equally by its vertices (the consistent
    P1 load of a uniform density)."""
    out = np.zeros(n, dtype=np.float64)
    if len(simp):
        k = simp.shape[1]
        np.add.at(out, simp.ravel(), np.repeat(meas / k, k))
    return out


@dataclass(frozen=True)
class LoadDensities:
    """wp9 S: per-node load densities of a battery (see `load_densities`)."""
    density: np.ndarray          # (L, N, sd): nodal force / the measure it carries
    measure: np.ndarray          # (L, N): that measure (0 off the loaded set)
    kind: tuple                  # per load case: "body", "traction" or "none"
    loaded_fraction: np.ndarray  # (L,): loaded measure / the whole of its kind
    dscale: float                # battery max |density| (+1e-12)
    l1: float                    # battery_l1(F)


def load_densities(arch) -> LoadDensities:
    """wp9 S: nodal loads as densities. A load case is a BODY load if it loads
    an interior node (gravity), else a TRACTION; its measure at a loaded node is
    the node's share of the cells (body) or boundary facets (traction) whose
    vertices are all loaded -- for a uniform density with P1 elements the
    consistent nodal force is exactly density x that share, so the density is
    exact and the same on every mesh. A loaded node with no such cell or facet
    (a point load) takes its share of every adjacent one of that kind.
    Assembly-level: mesh and load vector only."""
    X, el = arch.nodes, arch.elements
    sd, n, L = spatial_dim_of(arch), int(arch.n_nodes), int(arch.n_loads)
    F = np.asarray(arch.F, dtype=np.float64).reshape(L, n, sd)
    vol, bnd, fm = simplex_geometry(X, el)
    boundary = np.zeros(n, dtype=bool)
    boundary[bnd.ravel()] = True
    total = {"body": float(vol.sum()), "traction": float(fm.sum())}
    mag = np.abs(F).max(axis=2)                                  # (L, N)
    tol = 1e-12 * max(float(mag.max()) if mag.size else 0.0, 1e-300)
    dens, meas = np.zeros_like(F), np.zeros((L, n))
    kinds, frac = [], np.zeros(L)
    for li in range(L):
        S = mag[li] > tol
        if not S.any():
            kinds.append("none")
            continue
        kind = "body" if bool((S & ~boundary).any()) else "traction"
        simp, sm = (np.asarray(el, dtype=np.int64), vol) if kind == "body" else (bnd, fm)
        inside = S[simp].all(axis=1)
        m = _lumped(n, simp[inside], sm[inside])
        miss = S & ~(m > 0)
        if miss.any():
            m[miss] = _lumped(n, simp, sm)[miss]
        ok = S & (m > 0)
        dens[li][ok] = F[li][ok] / m[ok, None]
        meas[li] = np.where(S, m, 0.0)
        frac[li] = float(meas[li].sum() / total[kind]) if total[kind] > 0 else 0.0
        kinds.append(kind)
    return LoadDensities(density=dens, measure=meas, kind=tuple(kinds), loaded_fraction=frac,
                         dscale=float(np.abs(dens).max() + 1e-12), l1=battery_l1(arch.F))


def density_load_summary(F: np.ndarray, load_idx: int, dd: LoadDensities,
                         spatial_dim: int = 2) -> np.ndarray:
    """wp9 S: the load summary in mesh-free form, same length as `load_summary`
    -- this case's total |f| and its resultant components, each over the
    battery's total (battery_l1), and the loaded fraction of the boundary
    (traction) or of the body (body load)."""
    sd = int(spatial_dim)
    f = np.asarray(F, dtype=np.float64)[load_idx].reshape(-1, sd)
    return np.array([np.abs(f).sum() / dd.l1,
                     *(f[:, c].sum() / dd.l1 for c in range(sd)),
                     float(dd.loaded_fraction[load_idx])], dtype=np.float64)


def build_features(arch, load_idx: int, spec: FeatureSpec,
                   densities: LoadDensities | None = None) -> np.ndarray:
    """(N, spec.dim) float32 node features for one load case.

    WP7 3D-P0.3: base columns follow the instance's spatial dimension; the spec
    must agree (the encoder's input layer is sized from ``spec.dim`` before any
    data is seen, so a silent mismatch would train garbage -- fail loudly).
    wp9 S: with ``spec.load_density`` the load columns are the load densities
    over the battery's largest density and the load summary is
    `density_load_summary` (``densities``: the battery's, computed once)."""
    n = arch.n_nodes
    sd = spatial_dim_of(arch)
    if int(spec.spatial_dim) != sd:
        raise ValueError(
            f"FeatureSpec.spatial_dim={spec.spatial_dim} but the instance is "
            f"{sd}D (nodes {arch.nodes.shape}); set model.features.spatial_dim="
            f"{sd} in the config (WP7 3D-P0.3)")
    coords = normalized_coords(arch.nodes)
    dmask = arch.dirichlet_mask.reshape(-1, sd).astype(np.float64)
    if spec.load_density:
        dd = densities if densities is not None else load_densities(arch)
        f = dd.density[load_idx] / dd.dscale
    else:
        fscale = battery_fscale(arch.F)
        f = arch.F[load_idx].reshape(-1, sd) / fscale
    cols = [coords, dmask, f]
    if spec.load_summary:
        summ = (density_load_summary(arch.F, load_idx, dd, sd) if spec.load_density
                else load_summary(arch.F, load_idx, sd))
        cols.append(np.broadcast_to(summ, (n, load_summary_dim(sd))))
    if spec.geometry:
        cols.append(np.broadcast_to(geometry_descriptor(arch.meta), (n, GEOMETRY_DIM)))
    return np.concatenate(cols, axis=1).astype(np.float32)


def build_features_battery(arch, spec: FeatureSpec) -> np.ndarray:
    """(L, N, spec.dim) -- one batched tensor for the whole load battery."""
    dd = load_densities(arch) if spec.load_density else None
    return np.stack([build_features(arch, j, spec, dd) for j in range(arch.n_loads)], axis=0)
