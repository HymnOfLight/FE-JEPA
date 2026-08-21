"""WP7 foundation: 3D P1 tetrahedra under the SAME node-major contract.

Plan v2.0 WP7 is conditional on Gate G1' ("3D P1 tetrahedra (same node-major
contract), SimJEB with official splits, linear-attention encoder, AMP/compile
validation"). Building Phase 2 now would violate the gate; but the *contract
clause* -- that the node-major convention and everything downstream of it survive
the jump to 3D -- is checkable today at near-zero cost, and checking it now is
what de-risks WP7.

This module therefore provides exactly the contract proof and nothing more:

  * structured tetrahedral box meshes (5 tets per hex cell);
  * P1 linear-tetrahedron stiffness assembly, node-major dofs ``3*i + c``;
  * face tractions + gravity load battery, one clamped face;
  * 3D strain / stress / von-Mises / strain-energy recovery with the same Lame
    conventions as assembly (the 2D energy identity, one dimension up).

What is *proven* by tests/test_tet3d.py without touching any 2D code path:
archives round-trip, the exact solver reproduces U*, the anchor's gap identity
holds, warm starts and :mod:`fejepa.polish` work, and the WP6 theory checks
(conditioning / Chebyshev) pass -- all on 3D instances, unchanged.

Intentionally absent, still G1'-gated (PLAN_MAP): SimJEB ingestion and official
splits, 3D features/encoder (2D features hard-code 2 components by design),
linear attention, AMP/compile validation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..data.archive import InstanceArchive
from .solve import SolveLedger, solve_fe_displacement
from .stress import lame

LOAD_NAMES_3D = ["face_down", "face_axial", "top_shear", "gravity"]

# 5-tet decomposition of the unit hex (corner ordering: x fastest, then y, then z)
_HEX_TO_TETS = np.array([[0, 1, 3, 7], [0, 1, 7, 5], [0, 5, 7, 4],
                         [1, 3, 7, 2], [1, 2, 7, 6]], dtype=np.int64)


def structured_tet_mesh(w: float, h: float, d: float, nx: int, ny: int, nz: int):
    xs, ys, zs = (np.linspace(0, w, nx + 1), np.linspace(0, h, ny + 1),
                  np.linspace(0, d, nz + 1))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k),
                     nid(i, j + 1, k), nid(i, j, k + 1), nid(i + 1, j, k + 1),
                     nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                for t in _HEX_TO_TETS:
                    tets.append([c[t[0]], c[t[1]], c[t[2]], c[t[3]]])
    return nodes.astype(np.float64), np.asarray(tets, dtype=np.int64)


def _tet_geometry(nodes: np.ndarray, tets: np.ndarray):
    """Volumes and shape-function gradients grad N_a (E, 4, 3) for P1 tets."""
    p = nodes[tets]                                        # (E, 4, 3)
    J = p[:, 1:] - p[:, :1]                                # (E, 3, 3) edge matrix
    detJ = np.linalg.det(J)
    vol = np.abs(detJ) / 6.0
    Jinv = np.linalg.inv(J)                                # rows: d(xi)/d(x)
    grads = np.empty((tets.shape[0], 4, 3))
    grads[:, 1:, :] = np.transpose(Jinv, (0, 2, 1))        # grad N_a, a=1..3
    grads[:, 0, :] = -grads[:, 1:, :].sum(axis=1)
    return vol, grads


def _elastic_D(material: dict) -> np.ndarray:
    lam, mu = lame(material["E"], material["nu"], plane="strain")  # true 3D Lame
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2.0 * mu
    D[3:, 3:] = mu * np.eye(3)
    return D


def _B_matrices(grads: np.ndarray) -> np.ndarray:
    """Voigt strain-displacement matrices (E, 6, 12); order xx,yy,zz,xy,yz,zx."""
    E = grads.shape[0]
    B = np.zeros((E, 6, 12))
    for a in range(4):
        gx, gy, gz = grads[:, a, 0], grads[:, a, 1], grads[:, a, 2]
        c = 3 * a
        B[:, 0, c] = gx
        B[:, 1, c + 1] = gy
        B[:, 2, c + 2] = gz
        B[:, 3, c], B[:, 3, c + 1] = gy, gx
        B[:, 4, c + 1], B[:, 4, c + 2] = gz, gy
        B[:, 5, c], B[:, 5, c + 2] = gz, gx
    return B


def assemble_tet(nodes: np.ndarray, tets: np.ndarray, material: dict) -> sp.csr_matrix:
    """Node-major P1 stiffness: dof ``3*i + c`` is component c of node i."""
    vol, grads = _tet_geometry(nodes, tets)
    D = _elastic_D(material)
    B = _B_matrices(grads)
    ke = np.einsum("e,eji,jk,ekl->eil", vol, B, D, B)      # (E, 12, 12)
    dof = (3 * tets[:, :, None] + np.arange(3)).reshape(tets.shape[0], 12)
    rows = np.repeat(dof, 12, axis=1).ravel()
    cols = np.tile(dof, (1, 12)).ravel()
    ndof = 3 * nodes.shape[0]
    K = sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(ndof, ndof)).tocsr()
    K.sum_duplicates()
    return K


# ------------------------------------------------------------------- loads ----

def _face_traction(nodes, face_ids: np.ndarray, t: np.ndarray,
                   plane_axes: tuple[int, int]) -> np.ndarray:
    """Area-lumped traction on an axis-aligned rectangular boundary face."""
    ndof = 3 * nodes.shape[0]
    f = np.zeros(ndof)
    a0, a1 = plane_axes
    u = np.unique(nodes[face_ids, a0])
    v = np.unique(nodes[face_ids, a1])
    du, dv = np.diff(u).mean(), np.diff(v).mean()
    for nid in face_ids:
        edge0 = np.isclose(nodes[nid, a0], u[0]) or np.isclose(nodes[nid, a0], u[-1])
        edge1 = np.isclose(nodes[nid, a1], v[0]) or np.isclose(nodes[nid, a1], v[-1])
        wgt = du * dv * (0.25 if (edge0 and edge1) else 0.5 if (edge0 or edge1)
                         else 1.0)
        f[3 * nid:3 * nid + 3] += wgt * t
    return f


def _gravity_3d(nodes, tets, g: np.ndarray) -> np.ndarray:
    vol, _ = _tet_geometry(nodes, tets)
    f = np.zeros(3 * nodes.shape[0])
    for a in range(4):
        for c in range(3):
            np.add.at(f, 3 * tets[:, a] + c, vol / 4.0 * g[c])
    return f


def tet_instance(rng: np.random.Generator, nx: int = 4, ny: int = 3, nz: int = 3,
                 labelled: bool = False,
                 ledger: SolveLedger | None = None) -> InstanceArchive:
    """One 3D BVP in the standard archive schema (meta.extra.dim == 3)."""
    w = float(rng.uniform(1.5, 3.0))
    h = float(rng.uniform(0.8, 1.5))
    d = float(rng.uniform(0.6, 1.2))
    material = {"E": 1.0, "nu": float(rng.uniform(0.25, 0.38)), "plane": "3d"}
    nodes, tets = structured_tet_mesh(w, h, d, nx, ny, nz)
    K = assemble_tet(nodes, tets, material)

    right = np.nonzero(np.isclose(nodes[:, 0], w))[0]
    top = np.nonzero(np.isclose(nodes[:, 1], h))[0]
    left = np.nonzero(np.isclose(nodes[:, 0], 0.0))[0]
    ts = 0.05 * rng.uniform(0.5, 1.5, size=4)
    F = np.stack([
        _face_traction(nodes, right, np.array([0.0, -ts[0], 0.0]), (1, 2)),
        _face_traction(nodes, right, np.array([ts[1], 0.0, 0.0]), (1, 2)),
        _face_traction(nodes, top, np.array([ts[2], 0.0, 0.0]), (0, 2)),
        _gravity_3d(nodes, tets, np.array([0.0, -ts[3], 0.0])),
    ])
    dmask = np.zeros(3 * nodes.shape[0], dtype=bool)
    for c in range(3):
        dmask[3 * left + c] = True

    meta = {"material": material,
            "extra": {"width": w, "height": h, "depth": d, "dim": 3,
                      "n_holes": 0, "holes": [], "backend": "tet3d"},
            "loads": LOAD_NAMES_3D}
    arch = InstanceArchive(nodes=nodes, elements=tets, K=K, F=F,
                           dirichlet_mask=dmask, meta=meta)
    if labelled:
        arch.U_star, _ = solve_fe_displacement(K, F, ~dmask, method="direct",
                                               ledger=ledger,
                                               stage="generation-labels")
    return arch


def generate_tet3d_dataset(out, n: int, seed: int, labelled: str = "none",
                           nx: int = 4, ny: int = 3, nz: int = 3,
                           ledger: SolveLedger | None = None):
    """Structured-tet corpus in the standard manifest contract (WP7 3D-S1).

    Mirror of :func:`fejepa.fe.synthetic.generate_synthetic_dataset`, one
    dimension up (labelled: none|all); zero gmsh dependency by design, so the
    3D smoke run proves the pipeline end to end before any 3D-G1 corpus work.
    """
    from pathlib import Path

    from ..data.archive import save_instance, write_manifest

    out = Path(out)
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        arch = tet_instance(rng, nx=nx, ny=ny, nz=nz,
                            labelled=(labelled == "all"), ledger=ledger)
        fn = f"instance_{i:05d}.npz"
        save_instance(arch, out / fn)
        records.append({"file": fn, "n_nodes": arch.n_nodes,
                        "n_holes": 0, "labelled": arch.labelled})
    write_manifest(out, records, {"backend": "tet3d", "seed": seed,
                                  "labelled_policy": labelled,
                                  "load_names": LOAD_NAMES_3D})
    return out


def generate_tet3d_multires(out, n: int, seed: int, coarsen: float,
                            nx: int = 8, ny: int = 6, nz: int = 6,
                            labelled: str = "none",
                            ledger: SolveLedger | None = None):
    """Same 3D BVP meshed fine (nx,ny,nz) and coarse (dims/coarsen); pairs
    manifest in the 2D multires contract (WP7 E4-3D wiring)."""
    from pathlib import Path

    from ..data.archive import save_instance, write_manifest

    out = Path(out)
    rng = np.random.default_rng(seed)
    cx, cy, cz = (max(2, int(round(v / coarsen))) for v in (nx, ny, nz))
    pairs = []
    for i in range(n):
        state = rng.bit_generator.state          # same params for both meshes
        fine = tet_instance(rng, nx=nx, ny=ny, nz=nz,
                            labelled=(labelled == "all"), ledger=ledger)
        rng.bit_generator.state = state
        coarse = tet_instance(rng, nx=cx, ny=cy, nz=cz,
                              labelled=(labelled == "all"), ledger=ledger)
        ff, cf = f"instance_{i:05d}_fine.npz", f"instance_{i:05d}_coarse.npz"
        save_instance(fine, out / ff)
        save_instance(coarse, out / cf)
        pairs.append({"fine": ff, "coarse": cf})
    write_manifest(out, [], {"backend": "tet3d", "seed": seed,
                             "coarsen": float(coarsen),
                             "labelled_policy": labelled,
                             "load_names": LOAD_NAMES_3D, "pairs": pairs})
    return out


# --------------------------------------------------------------- 3D recovery --

def tet_strains(nodes, tets, u: np.ndarray) -> np.ndarray:
    """(E, 6) Voigt strains (engineering shears) of a node-major field."""
    _, grads = _tet_geometry(nodes, tets)
    B = _B_matrices(grads)
    dof = (3 * tets[:, :, None] + np.arange(3)).reshape(tets.shape[0], 12)
    return np.einsum("eij,ej->ei", B, u[dof])


def tet_stresses(nodes, tets, u, material: dict) -> np.ndarray:
    return tet_strains(nodes, tets, u) @ _elastic_D(material).T


def tet_von_mises(nodes, tets, u, material: dict) -> np.ndarray:
    s = tet_stresses(nodes, tets, u, material)
    sxx, syy, szz, sxy, syz, szx = (s[:, i] for i in range(6))
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def tet_strain_energy(nodes, tets, u, material: dict) -> float:
    """Recovered energy; equals 0.5 u^T K u (the 2D identity, one dimension up)."""
    vol, _ = _tet_geometry(nodes, tets)
    eps = tet_strains(nodes, tets, u)
    sig = tet_stresses(nodes, tets, u, material)
    return float(np.sum(0.5 * np.einsum("ei,ei->e", sig, eps) * vol))
