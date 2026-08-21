"""WP7 3D-G1: gmsh tetrahedral corpus backend (box with spherical cavities).

The envelope memo of 21 August 2026 fixed the corpus shape this module serves:
a training corpus of order 2,000 instances at 10k--30k dof and a
resolution-transfer set at ~100k dof, labelled economy-style (CG per the
measured crossover). This module supplies the geometry family:

  * OCC boxes (the ``tet_instance`` sampling ranges, so the 3D geometry
    descriptor's normalisation constants carry over verbatim) minus 0--3
    non-overlapping interior spherical cavities ``(x, y, z, r)`` -- exactly the
    cavity convention the P0.2 descriptor already encodes;
  * gmsh Delaunay tet meshing with a characteristic length ``lc`` controlling
    resolution (the transfer set is the same sampler at a smaller ``lc``);
  * the same four-load battery, clamp, assembly, labelling and manifest
    contract as :mod:`fejepa.fe.tet3d` -- assembly, gravity lumping and the
    recovery/metric family are reused from there unchanged.

Face tractions get a proper unstructured quadrature here: boundary triangles on
the loaded plane are extracted from the tet mesh and each triangle lumps
area/3 to its nodes -- exact for constant tractions on P1 (the structured
helper's tensor-grid trapezoidal weights do not apply to gmsh meshes).

Determinism: for a fixed gmsh version, options set here, parameters and ``lc``,
meshing is deterministic (tested); across gmsh versions meshes may differ --
the manifest pins what was actually built, which is the contract that matters.
"""

from __future__ import annotations

import numpy as np

from ..data.archive import InstanceArchive, save_instance, write_manifest
from .solve import SolveLedger, solve_fe_displacement
from .tet3d import LOAD_NAMES_3D, _gravity_3d, _tet_geometry, assemble_tet

_GMSH_READY = False


def _gmsh():
    """Process-lifetime gmsh handle (initialise once; never finalise in-loop)."""
    global _GMSH_READY
    import gmsh

    if not _GMSH_READY:
        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)   # determinism
        _GMSH_READY = True
    return gmsh


# ------------------------------------------------------------------ sampling --

def sample_params3d(rng: np.random.Generator) -> dict:
    """Box + cavity parameters (tet_instance ranges; descriptor-compatible)."""
    w = float(rng.uniform(1.5, 3.0))
    h = float(rng.uniform(0.8, 1.5))
    d = float(rng.uniform(0.6, 1.2))
    nu = float(rng.uniform(0.25, 0.38))
    n_cav = int(rng.integers(0, 4))
    holes: list[tuple[float, float, float, float]] = []
    guard = 0
    while len(holes) < n_cav and guard < 200:
        guard += 1
        r = float(rng.uniform(0.08, 0.16) * min(w, h, d))
        m = 1.6 * r                                   # wall + inter-cavity margin
        c = (float(rng.uniform(m, w - m)), float(rng.uniform(m, h - m)),
             float(rng.uniform(m, d - m)))
        if all((c[0]-x)**2 + (c[1]-y)**2 + (c[2]-z)**2 >= (1.25*(r+rr))**2
               for x, y, z, rr in holes):
            holes.append((*c, r))
    return {"width": w, "height": h, "depth": d, "nu": nu, "holes": holes}


# ------------------------------------------------------------------- meshing --

def mesh_box_with_cavities(params: dict, lc: float):
    """gmsh OCC box minus spheres -> (nodes (N,3) float64, tets (E,4) int64)."""
    g = _gmsh()
    g.clear()
    g.model.add("g1")
    occ = g.model.occ
    box = occ.addBox(0, 0, 0, params["width"], params["height"], params["depth"])
    tools = [(3, occ.addSphere(x, y, z, r)) for x, y, z, r in params["holes"]]
    if tools:
        occ.cut([(3, box)], tools, removeObject=True, removeTool=True)
    occ.synchronize()
    # Sizing model (E4-3D revision): drive the bulk from geometry-point sizes
    # at lc and cap with Max=lc; let curvature refine cavity surfaces down to
    # Min=lc/4. Without the point sizes, cavity-bearing geometries saturate the
    # sizing field and lc stops binding (observed: identical meshes at lc and
    # 2*lc), which breaks multires pairs.
    g.model.mesh.setSize(g.model.getEntities(0), lc)
    g.option.setNumber("Mesh.MeshSizeMin", lc / 4.0)
    g.option.setNumber("Mesh.MeshSizeMax", lc)
    g.option.setNumber("Mesh.MeshSizeFromCurvature", 8)
    g.option.setNumber("Mesh.Algorithm", 6)      # 2D frontal (surface, stable)
    g.option.setNumber("Mesh.Algorithm3D", 1)    # Delaunay
    g.option.setNumber("Mesh.Optimize", 1)
    g.model.mesh.generate(3)

    tags, coords, _ = g.model.mesh.getNodes()
    order = np.argsort(tags)
    nodes = np.asarray(coords, dtype=np.float64).reshape(-1, 3)[order]
    remap = {int(t): i for i, t in enumerate(np.asarray(tags)[order])}
    etypes, _, enodes = g.model.mesh.getElements(dim=3)
    assert list(etypes) == [4], f"expected linear tets only, got types {etypes}"
    conn = np.asarray(enodes[0], dtype=np.int64).reshape(-1, 4)
    tets = np.vectorize(remap.__getitem__)(conn).astype(np.int64)
    vol, _ = _tet_geometry(nodes, tets)
    assert (vol > 0).all(), "gmsh produced degenerate/inverted tets"
    return nodes, tets


# --------------------------------------------------------------- load battery --

def _boundary_tris(tets: np.ndarray) -> np.ndarray:
    """(T, 3) boundary triangles: tet faces that appear exactly once."""
    faces = np.concatenate([tets[:, idx] for idx in
                            ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))])
    key = np.sort(faces, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return faces[cnt[inv] == 1]


def _plane_tris(nodes, tris, axis: int, value: float) -> np.ndarray:
    on = np.isclose(nodes[:, axis], value, atol=1e-8)
    return tris[on[tris].all(axis=1)]


def _tri_area(nodes, tris) -> np.ndarray:
    a, b, c = nodes[tris[:, 0]], nodes[tris[:, 1]], nodes[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def plane_face_traction(nodes, tets, axis: int, value: float,
                        t: np.ndarray, tris_all=None) -> np.ndarray:
    """Area-lumped constant traction on the boundary plane ``x[axis]==value``:
    each boundary triangle scatters area/3 to its nodes (exact for P1)."""
    tris = _plane_tris(nodes, tris_all if tris_all is not None
                       else _boundary_tris(tets), axis, value)
    area = _tri_area(nodes, tris)
    f = np.zeros(3 * nodes.shape[0])
    for a in range(3):
        for c in range(3):
            np.add.at(f, 3 * tris[:, a] + c, area / 3.0 * t[c])
    return f


# ------------------------------------------------------------------ instance --

def gmsh3d_instance(rng: np.random.Generator, lc: float = 0.30,
                    labelled: bool = False, params: dict | None = None,
                    solve_method: str = "cg",
                    ledger: SolveLedger | None = None) -> InstanceArchive:
    """One unstructured 3D BVP in the standard archive schema (extra.dim == 3)."""
    p = params if params is not None else sample_params3d(rng)
    w, h, d = p["width"], p["height"], p["depth"]
    material = {"E": 1.0, "nu": p["nu"], "plane": "3d"}
    nodes, tets = mesh_box_with_cavities(p, lc)
    K = assemble_tet(nodes, tets, material)

    tris = _boundary_tris(tets)
    ts = 0.05 * rng.uniform(0.5, 1.5, size=4)
    F = np.stack([
        plane_face_traction(nodes, tets, 0, w, np.array([0.0, -ts[0], 0.0]), tris),
        plane_face_traction(nodes, tets, 0, w, np.array([ts[1], 0.0, 0.0]), tris),
        plane_face_traction(nodes, tets, 1, h, np.array([ts[2], 0.0, 0.0]), tris),
        _gravity_3d(nodes, tets, np.array([0.0, -ts[3], 0.0])),
    ])
    left = np.nonzero(np.isclose(nodes[:, 0], 0.0, atol=1e-8))[0]
    dmask = np.zeros(3 * nodes.shape[0], dtype=bool)
    for c in range(3):
        dmask[3 * left + c] = True

    meta = {"material": material,
            "extra": {"width": w, "height": h, "depth": d, "dim": 3,
                      "n_holes": len(p["holes"]),
                      "holes": [list(hh) for hh in p["holes"]],
                      "lc": float(lc), "backend": "gmsh3d"},
            "loads": LOAD_NAMES_3D}
    arch = InstanceArchive(nodes=nodes, elements=tets, K=K, F=F,
                           dirichlet_mask=dmask, meta=meta)
    if labelled:
        arch.U_star, _ = solve_fe_displacement(K, F, ~dmask, method=solve_method,
                                               ledger=ledger,
                                               stage="generation-labels")
    return arch


def generate_gmsh3d_dataset(out, n: int, seed: int, labelled: str = "none",
                            lc: float = 0.30, solve_method: str = "cg",
                            ledger: SolveLedger | None = None):
    """gmsh 3D corpus in the standard manifest contract (WP7 3D-G1).

    Labelled solves default to CG per the 21 August envelope memo (the
    no-factor-reuse crossover sits at ~7k dof). ``lc`` sets the resolution;
    the transfer set is this generator at a smaller ``lc``.
    """
    from pathlib import Path

    out = Path(out)
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        arch = gmsh3d_instance(rng, lc=lc, labelled=(labelled == "all"),
                               solve_method=solve_method, ledger=ledger)
        fn = f"instance_{i:05d}.npz"
        save_instance(arch, out / fn)
        records.append({"file": fn, "n_nodes": arch.n_nodes,
                        "n_holes": arch.meta["extra"]["n_holes"],
                        "labelled": arch.labelled})
    write_manifest(out, records, {"backend": "gmsh3d", "seed": seed,
                                  "labelled_policy": labelled, "lc": float(lc),
                                  "load_names": LOAD_NAMES_3D})
    return out


def generate_gmsh3d_multires(out, n: int, seed: int, coarsen: float,
                             lc: float = 0.30, labelled: str = "none",
                             solve_method: str = "cg",
                             ledger: SolveLedger | None = None):
    """Same unstructured 3D BVP meshed at ``lc`` and ``lc*coarsen``; pairs
    manifest in the 2D multires contract (WP7 E4-3D wiring). The rng-state
    reset makes geometry and traction scales identical across the pair;
    meshing consumes no randomness."""
    from pathlib import Path

    out = Path(out)
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n):
        state = rng.bit_generator.state
        fine = gmsh3d_instance(rng, lc=lc, labelled=(labelled == "all"),
                               solve_method=solve_method, ledger=ledger)
        rng.bit_generator.state = state
        coarse = gmsh3d_instance(rng, lc=lc * coarsen,
                                 labelled=(labelled == "all"),
                                 solve_method=solve_method, ledger=ledger)
        ff, cf = f"instance_{i:05d}_fine.npz", f"instance_{i:05d}_coarse.npz"
        save_instance(fine, out / ff)
        save_instance(coarse, out / cf)
        pairs.append({"fine": ff, "coarse": cf})
    write_manifest(out, [], {"backend": "gmsh3d", "seed": seed,
                             "coarsen": float(coarsen), "lc": float(lc),
                             "labelled_policy": labelled,
                             "load_names": LOAD_NAMES_3D, "pairs": pairs})
    return out
