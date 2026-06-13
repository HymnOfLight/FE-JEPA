"""Parametric generator of 2D linear-elasticity instances.

Samples rectangular plates with randomly placed circular holes, clamps the left
edge (homogeneous Dirichlet), and applies a small battery of load cases that all
share the same stiffness ``K`` (only ``F`` changes between cases -- cheap, as the
proposal notes).  This is the Phase-0 unlabelled corpus / labelled eval source.

The geometry/material/load parameters are sampled once and can be meshed at
multiple resolutions, which yields the *exact physical augmentation* used by the
cross-mesh invariance term: two meshes of the same boundary-value problem are
identical physics by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fejepa.fe.elasticity import FEProblem, LoadCase, Material, assemble_problem
from fejepa.fe.meshing import plate_with_holes_mesh


@dataclass
class GeneratorConfig:
    width_range: tuple[float, float] = (1.5, 3.0)
    height_range: tuple[float, float] = (0.8, 1.5)
    max_holes: int = 3
    hole_radius_frac: tuple[float, float] = (0.06, 0.14)
    mesh_size_frac: tuple[float, float] = (0.06, 0.11)
    E: float = 1.0
    nu_range: tuple[float, float] = (0.25, 0.38)
    plane: str = "stress"
    traction_scale: float = 0.05
    body_scale: float = 0.05


@dataclass
class InstanceParams:
    width: float
    height: float
    nu: float
    holes: list[tuple[float, float, float]]
    mesh_size: float
    load_specs: list[dict] = field(default_factory=list)
    E: float = 1.0
    plane: str = "stress"


def _sample_holes(rng: np.random.Generator, w: float, h: float, cfg: GeneratorConfig):
    n_holes = int(rng.integers(0, cfg.max_holes + 1))
    holes: list[tuple[float, float, float]] = []
    rmin = cfg.hole_radius_frac[0] * min(w, h)
    rmax = cfg.hole_radius_frac[1] * min(w, h)
    attempts = 0
    while len(holes) < n_holes and attempts < 50:
        attempts += 1
        r = float(rng.uniform(rmin, rmax))
        cx = float(rng.uniform(0.25 * w, 0.9 * w))
        cy = float(rng.uniform(0.2 * h, 0.8 * h))
        margin = 1.4 * r
        if cx - margin < 0 or cx + margin > w or cy - margin < 0 or cy + margin > h:
            continue
        ok = all(
            (cx - ox) ** 2 + (cy - oy) ** 2 > (1.3 * (r + orr)) ** 2
            for (ox, oy, orr) in holes
        )
        if ok:
            holes.append((cx, cy, r))
    return holes


def sample_params(rng: np.random.Generator, cfg: GeneratorConfig | None = None) -> InstanceParams:
    cfg = cfg or GeneratorConfig()
    w = float(rng.uniform(*cfg.width_range))
    h = float(rng.uniform(*cfg.height_range))
    nu = float(rng.uniform(*cfg.nu_range))
    mesh_size = float(rng.uniform(*cfg.mesh_size_frac)) * min(w, h)
    holes = _sample_holes(rng, w, h, cfg)

    ts, bs = cfg.traction_scale, cfg.body_scale
    load_specs = [
        {"name": "tip_down", "kind": "traction", "edge": "right", "vec": (0.0, -ts * rng.uniform(0.5, 1.5))},
        {"name": "tip_axial", "kind": "traction", "edge": "right", "vec": (ts * rng.uniform(0.5, 1.5), 0.0)},
        {"name": "top_shear", "kind": "traction", "edge": "top", "vec": (ts * rng.uniform(0.3, 1.0), 0.0)},
        {"name": "gravity", "kind": "body", "vec": (0.0, -bs * rng.uniform(0.5, 1.5))},
    ]
    return InstanceParams(
        width=w, height=h, nu=nu, holes=holes, mesh_size=mesh_size,
        load_specs=load_specs, E=cfg.E, plane=cfg.plane,
    )


def _build_loads(params: InstanceParams) -> list[LoadCase]:
    w, h = params.width, params.height
    edges = {
        "right": lambda x: np.isclose(x[0], w),
        "top": lambda x: np.isclose(x[1], h),
        "left": lambda x: np.isclose(x[0], 0.0),
        "bottom": lambda x: np.isclose(x[1], 0.0),
    }
    cases = []
    for spec in params.load_specs:
        if spec["kind"] == "traction":
            cases.append(
                LoadCase(spec["name"], tractions=[(edges[spec["edge"]], tuple(spec["vec"]))])
            )
        else:
            cases.append(LoadCase(spec["name"], body=tuple(spec["vec"])))
    return cases


def build_problem(
    params: InstanceParams, mesh_size: float | None = None, labelled: bool = True
) -> FEProblem:
    mesh_size = mesh_size or params.mesh_size
    mesh = plate_with_holes_mesh(params.width, params.height, params.holes, mesh_size=mesh_size)
    material = Material(E=params.E, nu=params.nu, plane=params.plane)
    dirichlet = lambda x: np.isclose(x[0], 0.0)  # noqa: E731  clamp left edge
    loads = _build_loads(params)
    meta = {
        "width": params.width,
        "height": params.height,
        "n_holes": len(params.holes),
        "holes": params.holes,
        "mesh_size": mesh_size,
    }
    return assemble_problem(mesh, material, dirichlet, loads, solve_reference=labelled, meta=meta)


def sample_instance(
    rng: np.random.Generator,
    cfg: GeneratorConfig | None = None,
    labelled: bool = True,
) -> FEProblem:
    """Sample a single FE instance (geometry + material + load battery)."""

    params = sample_params(rng, cfg)
    return build_problem(params, labelled=labelled)


def sample_instance_multires(
    rng: np.random.Generator,
    cfg: GeneratorConfig | None = None,
    coarsen: float = 1.6,
    labelled: bool = True,
) -> tuple[FEProblem, FEProblem]:
    """Sample one BVP meshed at two resolutions (fine, coarse).

    The two problems share geometry, material and loads exactly -- the exact
    physical augmentation for the cross-mesh invariance term.
    """

    params = sample_params(rng, cfg)
    fine = build_problem(params, mesh_size=params.mesh_size, labelled=labelled)
    coarse = build_problem(params, mesh_size=params.mesh_size * coarsen, labelled=labelled)
    return fine, coarse


def generate_dataset(
    out_dir: str | Path,
    n_instances: int,
    seed: int = 0,
    labelled: bool = True,
    cfg: GeneratorConfig | None = None,
    verbose: bool = True,
) -> Path:
    """Generate ``n_instances`` archives into ``out_dir`` with a manifest."""

    from fejepa.data.archive import save_problem, write_manifest

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = cfg or GeneratorConfig()

    records = []
    prob = None
    for i in range(n_instances):
        prob = sample_instance(rng, cfg, labelled=labelled)
        fname = f"instance_{i:05d}.npz"
        save_problem(out_dir / fname, prob)
        records.append(
            {
                "file": fname,
                "n_nodes": int(prob.n_nodes),
                "n_dof": int(prob.n_dof),
                "n_loads": int(prob.n_loads),
                "n_holes": prob.meta.get("n_holes"),
            }
        )
        if verbose and (i + 1) % max(1, n_instances // 10) == 0:
            print(f"  generated {i + 1}/{n_instances}")

    manifest = {
        "seed": seed,
        "labelled": labelled,
        "n_instances": n_instances,
        "load_names": prob.load_names if prob is not None else [],
        "instances": records,
    }
    write_manifest(out_dir, manifest)
    return out_dir
