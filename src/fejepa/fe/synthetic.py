"""Synthetic FE backend: structured-mesh CST assembly with no gmsh/skfem dependency.

Plan v2.0 mapping: supports Sec.9 (the measured cost model / ``fejepa bench`` needs
representative instances without the meshing stack), the smoke config (end-to-end
pipeline exercise on any machine), and the numpy test suite (tests/conftest.py) that
preserves the audited invariants of Sec.2.1 (K U* = F, energy identities) as unit tests.

The assembly here uses the same B-matrix and Lame conventions as :mod:`fejepa.fe.stress`,
so the "recovered strain energy == 0.5 u^T K u" invariant holds exactly by construction
and is enforced by tests against the independent recovery code path.

This backend generates rectangular plates *without holes* (holes need gmsh); the meta
schema is identical to the gmsh generator's so every downstream component is exercised.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..data.archive import InstanceArchive, save_instance, write_manifest
from .solve import SolveLedger, solve_fe_displacement
from .stress import lame

LOAD_NAMES = ["tip_down", "tip_axial", "top_shear", "gravity"]


def structured_mesh(w: float, h: float, nx: int, ny: int):
    xs = np.linspace(0.0, w, nx + 1)
    ys = np.linspace(0.0, h, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], axis=1)

    def nid(i, j):
        return i * (ny + 1) + j

    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
            tris.append([a, b, c])
            tris.append([a, c, d])
    return nodes.astype(np.float64), np.asarray(tris, dtype=np.int64)


def assemble_cst(nodes: np.ndarray, elements: np.ndarray, material: dict) -> sp.csr_matrix:
    """Node-major CST stiffness: dof 2*i + c is component c of node i."""
    lam, mu = lame(material["E"], material["nu"], material["plane"])
    D = np.array([[lam + 2 * mu, lam, 0.0],
                  [lam, lam + 2 * mu, 0.0],
                  [0.0, 0.0, mu]])
    p = nodes[elements]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    area = 0.5 * np.abs(det)
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / det[:, None]
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / det[:, None]
    E = elements.shape[0]
    B = np.zeros((E, 3, 6))
    for i in range(3):
        B[:, 0, 2 * i] = b[:, i]
        B[:, 1, 2 * i + 1] = c[:, i]
        B[:, 2, 2 * i] = c[:, i]
        B[:, 2, 2 * i + 1] = b[:, i]
    ke = np.einsum("e,eji,jk,ekl->eil", area, B, D, B)     # (E, 6, 6)
    dof = np.empty((E, 6), dtype=np.int64)
    dof[:, 0::2] = 2 * elements
    dof[:, 1::2] = 2 * elements + 1
    rows = np.repeat(dof, 6, axis=1).ravel()
    cols = np.tile(dof, (1, 6)).ravel()
    ndof = 2 * nodes.shape[0]
    K = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    return K


def _edge_lumped_traction(nodes, edge_node_ids, t: np.ndarray, along: int) -> np.ndarray:
    """Consistent linear traction on a straight boundary edge chain (half-length lumping)."""
    ndof = 2 * nodes.shape[0]
    f = np.zeros(ndof)
    ids = edge_node_ids[np.argsort(nodes[edge_node_ids, along])]
    seg = np.abs(np.diff(nodes[ids, along]))
    for k, ln in enumerate(seg):
        for nid in (ids[k], ids[k + 1]):
            f[2 * nid] += 0.5 * ln * t[0]
            f[2 * nid + 1] += 0.5 * ln * t[1]
    return f


def _gravity_load(nodes, elements, g: np.ndarray) -> np.ndarray:
    p = nodes[elements]
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    f = np.zeros(2 * nodes.shape[0])
    for i in range(3):
        np.add.at(f, 2 * elements[:, i], area / 3.0 * g[0])
        np.add.at(f, 2 * elements[:, i] + 1, area / 3.0 * g[1])
    return f


def synthetic_instance(rng: np.random.Generator, nx: int = 8, ny: int = 6,
                       labelled: bool = False,
                       ledger: SolveLedger | None = None) -> InstanceArchive:
    w = float(rng.uniform(1.5, 3.0))
    h = float(rng.uniform(0.8, 1.5))
    nu = float(rng.uniform(0.25, 0.38))
    material = {"E": 1.0, "nu": nu, "plane": "stress"}
    nodes, elements = structured_mesh(w, h, nx, ny)
    K = assemble_cst(nodes, elements, material)

    right = np.nonzero(np.isclose(nodes[:, 0], w))[0]
    top = np.nonzero(np.isclose(nodes[:, 1], h))[0]
    left = np.nonzero(np.isclose(nodes[:, 0], 0.0))[0]
    ts = 0.05 * rng.uniform(0.5, 1.5, size=4)
    F = np.stack([
        _edge_lumped_traction(nodes, right, np.array([0.0, -ts[0]]), along=1),
        _edge_lumped_traction(nodes, right, np.array([ts[1], 0.0]), along=1),
        _edge_lumped_traction(nodes, top, np.array([ts[2], 0.0]), along=0),
        _gravity_load(nodes, elements, np.array([0.0, -ts[3]])),
    ])

    dmask = np.zeros(2 * nodes.shape[0], dtype=bool)
    dmask[2 * left] = True
    dmask[2 * left + 1] = True

    meta = {"material": material,
            "extra": {"width": w, "height": h, "n_holes": 0, "holes": [],
                      "target_h": w / nx, "backend": "synthetic"},
            "loads": LOAD_NAMES}
    arch = InstanceArchive(nodes=nodes, elements=elements, K=K, F=F,
                           dirichlet_mask=dmask, meta=meta)
    if labelled:
        arch.U_star, _ = solve_fe_displacement(K, F, ~dmask, method="direct",
                                               ledger=ledger, stage="generation-labels")
    return arch


def generate_synthetic_dataset(out, n: int, seed: int, labelled: str = "none",
                               nx: int = 8, ny: int = 6,
                               ledger: SolveLedger | None = None):
    """Mirror of the gmsh generator's contract for smoke/tests (labelled: none|all)."""
    from pathlib import Path

    out = Path(out)
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        arch = synthetic_instance(rng, nx=nx, ny=ny,
                                  labelled=(labelled == "all"), ledger=ledger)
        fn = f"instance_{i:05d}.npz"
        save_instance(arch, out / fn)
        records.append({"file": fn, "n_nodes": arch.n_nodes,
                        "n_holes": 0, "labelled": arch.labelled})
    write_manifest(out, records, {"backend": "synthetic", "seed": seed,
                                  "labelled_policy": labelled,
                                  "load_names": LOAD_NAMES})
    return out


def generate_synthetic_multires(out, n: int, seed: int, coarsen: float,
                                labelled: str = "none", nx: int = 10, ny: int = 8,
                                ledger: SolveLedger | None = None):
    """Same BVP meshed fine (nx,ny) and coarse (nx,ny / coarsen); pairs manifest."""
    from pathlib import Path

    out = Path(out)
    rng = np.random.default_rng(seed)
    cx = max(2, int(round(nx / coarsen)))
    cy = max(2, int(round(ny / coarsen)))
    pairs = []
    for i in range(n):
        state = rng.bit_generator.state          # same params for both resolutions
        fine = synthetic_instance(rng, nx=nx, ny=ny,
                                  labelled=(labelled == "all"), ledger=ledger)
        rng.bit_generator.state = state
        coarse = synthetic_instance(rng, nx=cx, ny=cy,
                                    labelled=(labelled == "all"), ledger=ledger)
        ff, cf = f"instance_{i:05d}_fine.npz", f"instance_{i:05d}_coarse.npz"
        save_instance(fine, out / ff)
        save_instance(coarse, out / cf)
        pairs.append({"fine": ff, "coarse": cf})
    write_manifest(out, [], {"backend": "synthetic", "seed": seed, "coarsen": coarsen,
                             "labelled_policy": labelled, "load_names": LOAD_NAMES,
                             "pairs": pairs})
    return out
