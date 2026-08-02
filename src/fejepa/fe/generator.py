"""Parametric dataset generation (gmsh backend): plates with holes, load batteries,
multiresolution pairs, deterministic parallel generation.

Plan v2.0 mapping:
  - Sec.2.5 verified assets: "2D generator (plates+holes), 4-load battery sharing K,
    multires pairs, parallel seeded generation".
  - WP5 (data economy): default ``labelled='none'`` -- archives are written without U*;
    the runner's labelling stage solves exactly val + budget prefixes via the ledger.
    ``labelled='all'`` reproduces the Phase-1 corpus style (economy not demonstrated).
  - E4' needs two coarsening factors: :func:`generate_multires_dataset` takes ``coarsen``.

Determinism: instance i is generated from ``SeedSequence(seed).spawn(n)[i]`` so the
corpus is reproducible for a given seed and independent of worker scheduling (audited
property, preserved). Parallel output is valid but not bit-identical to serial output
of a *different* fejepa version.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np

from ..data.archive import InstanceArchive, save_instance, write_manifest
from .elasticity import LOAD_NAMES, assemble_plate
from .solve import SolveLedger, solve_fe_displacement

_GMSH_READY = False


def _gmsh():
    global _GMSH_READY
    import gmsh

    if not _GMSH_READY:
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        _GMSH_READY = True
    return gmsh


def mesh_plate(width: float, height: float, holes: list, target_h: float):
    """Mesh [0,w]x[0,h] minus circular holes; returns (nodes (N,2), tris (E,3))."""
    gmsh = _gmsh()
    gmsh.model.add(f"plate_{np.random.randint(1 << 30)}")
    occ = gmsh.model.occ
    rect = occ.addRectangle(0.0, 0.0, 0.0, width, height)
    tools = [(2, occ.addDisk(cx, cy, 0.0, r, r)) for cx, cy, r in holes]
    if tools:
        occ.cut([(2, rect)], tools, removeObject=True, removeTool=True)
    occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.4 * target_h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", target_h)
    gmsh.model.mesh.generate(2)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xy = coords.reshape(-1, 3)[:, :2]
    index = {int(t): i for i, t in enumerate(tags)}
    _, elem_tags = gmsh.model.mesh.getElementsByType(2)[:2]
    tris = np.array([[index[int(t)] for t in tri]
                     for tri in np.asarray(elem_tags).reshape(-1, 3)], dtype=np.int64)
    nodes = np.asarray(xy, dtype=np.float64)
    gmsh.model.remove()
    return nodes, tris


def sample_params(rng: np.random.Generator) -> dict:
    width = float(rng.uniform(1.5, 3.0))
    height = float(rng.uniform(0.8, 1.5))
    nu = float(rng.uniform(0.25, 0.38))
    n_holes = int(rng.integers(0, 4))
    holes = []
    for _ in range(n_holes):
        for _attempt in range(50):
            r = float(rng.uniform(0.06, 0.16) * min(width, height))
            cx = float(rng.uniform(0.2 * width, 0.8 * width))
            cy = float(rng.uniform(0.2 * height, 0.8 * height))
            if all((cx - hx) ** 2 + (cy - hy) ** 2 > (r + hr + 0.05) ** 2
                   for hx, hy, hr in holes):
                holes.append([cx, cy, r])
                break
    target_h = float(rng.uniform(0.05, 0.12))
    traction_scales = 0.05 * rng.uniform(0.5, 1.5, size=4)
    return dict(width=width, height=height, nu=nu, holes=holes,
                target_h=target_h, traction_scales=traction_scales.tolist())


def build_instance(params: dict, coarsen: float = 1.0) -> InstanceArchive:
    material = {"E": 1.0, "nu": params["nu"], "plane": "stress"}
    nodes, elements = mesh_plate(params["width"], params["height"], params["holes"],
                                 params["target_h"] * coarsen)
    K, F, dmask = assemble_plate(nodes, elements, material,
                                 params["width"], params["height"],
                                 np.asarray(params["traction_scales"]))
    meta = {"material": material,
            "extra": {"width": params["width"], "height": params["height"],
                      "n_holes": len(params["holes"]), "holes": params["holes"],
                      "target_h": params["target_h"] * coarsen, "backend": "gmsh"},
            "loads": LOAD_NAMES}
    return InstanceArchive(nodes=nodes, elements=elements, K=K, F=F,
                           dirichlet_mask=dmask, meta=meta)


def _label(arch: InstanceArchive, ledger: SolveLedger | None) -> None:
    arch.U_star, _ = solve_fe_displacement(arch.K, arch.F, arch.free_mask,
                                           method="direct", ledger=ledger,
                                           stage="generation-labels")


def _worker(args):
    i, seed_state, out, labelled = args
    rng = np.random.default_rng(seed_state)
    arch = build_instance(sample_params(rng))
    if labelled == "all":
        _label(arch, None)
    fn = f"instance_{i:05d}.npz"
    save_instance(arch, Path(out) / fn)
    return {"file": fn, "n_nodes": arch.n_nodes,
            "n_holes": arch.meta["extra"]["n_holes"], "labelled": arch.labelled}


def generate_dataset(out, n: int, seed: int, labelled: str = "none",
                     jobs: int = 0, ledger: SolveLedger | None = None) -> Path:
    """labelled: 'none' (economy default, plan WP5) | 'all' (legacy Phase-1 style)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    children = np.random.SeedSequence(seed).spawn(n)
    tasks = [(i, children[i], str(out), labelled) for i in range(n)]
    if jobs and jobs != 1:
        procs = None if jobs <= 0 else jobs
        with mp.get_context("spawn").Pool(processes=procs, maxtasksperchild=64) as pool:
            records = pool.map(_worker, tasks, chunksize=8)
    else:
        records = [_worker(t) for t in tasks]
    if labelled == "all" and ledger is not None:
        ledger.add("generation-labels", n=sum(r["labelled"] for r in records) * len(LOAD_NAMES))
    write_manifest(out, records, {"backend": "gmsh", "seed": seed,
                                  "labelled_policy": labelled, "load_names": LOAD_NAMES})
    return out


def generate_multires_dataset(out, n: int, seed: int, coarsen: float,
                              labelled: str = "none",
                              ledger: SolveLedger | None = None) -> Path:
    """Same BVP meshed at target_h and target_h*coarsen (plan E4', coarsen in {1.8, 2.5})."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    children = np.random.SeedSequence(seed).spawn(n)
    pairs = []
    for i in range(n):
        params = sample_params(np.random.default_rng(children[i]))
        fine = build_instance(params, coarsen=1.0)
        coarse = build_instance(params, coarsen=coarsen)
        if labelled == "all":
            _label(fine, ledger)
            _label(coarse, ledger)
        ff, cf = f"instance_{i:05d}_fine.npz", f"instance_{i:05d}_coarse.npz"
        save_instance(fine, out / ff)
        save_instance(coarse, out / cf)
        pairs.append({"fine": ff, "coarse": cf})
    write_manifest(out, [], {"backend": "gmsh", "seed": seed, "coarsen": coarsen,
                             "labelled_policy": labelled, "load_names": LOAD_NAMES,
                             "pairs": pairs})
    return out
