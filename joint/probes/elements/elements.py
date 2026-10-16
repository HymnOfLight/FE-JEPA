"""Isoparametric solid elements for the element study of the joint model (exploratory, after gate G-C1.3).

Kinds (shape functions from sympy, checked at import for the Kronecker property and partition of unity):
  H8    8-node hexahedron, trilinear, full 2 x 2 x 2 Gauss integration (conforming; locks in bending)
  H8I   8-node hexahedron with incompatible modes, as in the gates (fejoint.hex8i; assembled there)
  H20   20-node serendipity hexahedron (corner and mid-edge nodes), full 3 x 3 x 3 Gauss integration
  H20R  the same element with reduced 2 x 2 x 2 Gauss integration
  H27   27-node Lagrange hexahedron (corner, mid-edge, mid-face and centre nodes), 3 x 3 x 3 Gauss integration
  T4    4-node tetrahedron, linear, one point (exact)
  T10   10-node tetrahedron (corner and mid-edge nodes), quadratic, 4 points (exact for its stiffness)
Quadratic elements keep straight edges here (mid nodes at the edge, face and cell centres), which is exact for the
joint's box-shaped parts. Tetrahedra come from splitting every hexahedron of the structured mesh into six along its
main diagonal (Kuhn), which keeps the mesh conforming because all cells share one orientation.

Degrees of freedom are node-major, [u_x, u_y, u_z] per node; strains [e_xx, e_yy, e_zz, g_xy, g_yz, g_zx].
"""
from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp
import sympy as sy

from fejoint import hex8i as H8MOD

# ------------------------------------------------------------------ reference nodes
XI8 = H8MOD.XI.copy()
HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
HEX_FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
XI20 = np.vstack([XI8, [0.5 * (XI8[a] + XI8[b]) for a, b in HEX_EDGES]])
XI27 = np.vstack([XI20, [XI8[list(f)].mean(axis=0) for f in HEX_FACES], [[0.0, 0.0, 0.0]]])
T4REF = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
TET_EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]
T10REF = np.vstack([T4REF, [0.5 * (T4REF[a] + T4REF[b]) for a, b in TET_EDGES]])
KUHN = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]

_x, _y, _z = sy.symbols("x y z")


def _sym_shapes(kind: str):
    if kind == "H8":
        return [sy.Rational(1, 8) * (1 + _x * a) * (1 + _y * b) * (1 + _z * c) for a, b, c in XI8.astype(int)]
    if kind == "H20":
        out = []
        for a, b, c in XI20:
            a, b, c = sy.nsimplify(a), sy.nsimplify(b), sy.nsimplify(c)
            if 0 not in (a, b, c):
                out.append(sy.Rational(1, 8) * (1 + _x * a) * (1 + _y * b) * (1 + _z * c) * (_x * a + _y * b + _z * c - 2))
            elif a == 0:
                out.append(sy.Rational(1, 4) * (1 - _x ** 2) * (1 + _y * b) * (1 + _z * c))
            elif b == 0:
                out.append(sy.Rational(1, 4) * (1 + _x * a) * (1 - _y ** 2) * (1 + _z * c))
            else:
                out.append(sy.Rational(1, 4) * (1 + _x * a) * (1 + _y * b) * (1 - _z ** 2))
        return out
    if kind == "H27":
        def l1(t, ti):
            return {-1: t * (t - 1) / 2, 0: 1 - t ** 2, 1: t * (t + 1) / 2}[int(ti)]
        return [l1(_x, a) * l1(_y, b) * l1(_z, c) for a, b, c in XI27]
    if kind == "T4":
        return [1 - _x - _y - _z, _x, _y, _z]
    if kind == "T10":
        L = [1 - _x - _y - _z, _x, _y, _z]
        return [L[i] * (2 * L[i] - 1) for i in range(4)] + [4 * L[a] * L[b] for a, b in TET_EDGES]
    raise ValueError(kind)


REF = {"H8": XI8, "H20": XI20, "H27": XI27, "T4": T4REF, "T10": T10REF}
SHAPES, GRADS = {}, {}
for _k in REF:
    _N = _sym_shapes(_k)
    _G = [[sy.diff(n_, v) for v in (_x, _y, _z)] for n_ in _N]
    SHAPES[_k] = sy.lambdify((_x, _y, _z), _N, "numpy")
    GRADS[_k] = sy.lambdify((_x, _y, _z), _G, "numpy")


def shape_values(kind: str, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(pts)
    return np.array([np.broadcast_to(np.asarray(SHAPES[kind](*p), dtype=float), (len(REF[kind]),)) for p in pts])


def shape_grads(kind: str, pts: np.ndarray) -> np.ndarray:
    """dN/dxi at the points, shape (npts, nen, 3)."""
    pts = np.atleast_2d(pts)
    return np.array([np.array(GRADS[kind](*p), dtype=float).reshape(len(REF[kind]), 3) for p in pts])


def _check_shapes():
    for k, ref in REF.items():
        V = shape_values(k, ref)
        assert np.allclose(V, np.eye(len(ref)), atol=1e-12), f"{k}: not interpolating"
        rnd = np.random.default_rng(0).uniform(-0.9, 0.9, (5, 3)) if k[0] == "H" else np.array([[0.1, 0.2, 0.3]] * 5)
        assert np.allclose(shape_values(k, rnd).sum(axis=1), 1.0), f"{k}: no partition of unity"
        assert np.allclose(shape_grads(k, rnd).sum(axis=1), 0.0), f"{k}: gradients do not sum to zero"


_check_shapes()

# ------------------------------------------------------------------ quadrature
def gauss_hex(n: int):
    if n == 1:
        p, w = np.array([0.0]), np.array([2.0])
    elif n == 2:
        p, w = np.array([-1, 1]) / np.sqrt(3.0), np.array([1.0, 1.0])
    elif n == 3:
        p, w = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5, 8, 5]) / 9.0
    else:
        raise ValueError(n)
    pts = np.array(list(itertools.product(p, p, p)))
    wts = np.array([a * b * c for a, b, c in itertools.product(w, w, w)])
    return pts, wts


def gauss_tet(n: int):
    if n == 1:
        return np.array([[0.25, 0.25, 0.25]]), np.array([1.0 / 6.0])
    if n == 4:
        a, b = 0.5854101966249685, 0.1381966011250105
        pts = np.array([[b, b, b], [a, b, b], [b, a, b], [b, b, a]])
        return pts, np.full(4, 1.0 / 24.0)
    raise ValueError(n)


RULES = {"H8": ("H8", gauss_hex(2)), "H20": ("H20", gauss_hex(3)), "H20R": ("H20", gauss_hex(2)),
         "H27": ("H27", gauss_hex(3)), "T4": ("T4", gauss_tet(1)), "T10": ("T10", gauss_tet(4))}
NEN = {"H8": 8, "H8I": 8, "H20": 20, "H20R": 20, "H27": 27, "T4": 4, "T10": 10}


# ------------------------------------------------------------------ meshes
def _edge_nodes(nodes, conn, edges):
    """Mid nodes on the given local edges of every cell, shared between cells: returns (nodes, ids (E, n_edges))."""
    a = conn[:, [e[0] for e in edges]]; b = conn[:, [e[1] for e in edges]]
    key = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=-1).reshape(-1, 2)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    mid = 0.5 * (nodes[uniq[:, 0]] + nodes[uniq[:, 1]])
    ids = nodes.shape[0] + inv.ravel().reshape(conn.shape[0], len(edges))
    return np.vstack([nodes, mid]), ids


def to_H20(nodes, hexes):
    nodes2, ids = _edge_nodes(nodes, hexes, HEX_EDGES)
    return nodes2, np.hstack([hexes, ids])


def to_H27(nodes, hexes):
    nodes2, conn20 = to_H20(nodes, hexes)
    fc = np.stack([np.sort(hexes[:, list(f)], axis=1) for f in HEX_FACES], axis=1).reshape(-1, 4)
    uniq, inv = np.unique(fc, axis=0, return_inverse=True)
    fmid = nodes[uniq].mean(axis=1)
    fid = nodes2.shape[0] + inv.ravel().reshape(hexes.shape[0], 6)
    nodes3 = np.vstack([nodes2, fmid])
    cid = nodes3.shape[0] + np.arange(hexes.shape[0])
    nodes4 = np.vstack([nodes3, nodes[hexes].mean(axis=1)])
    return nodes4, np.hstack([conn20, fid, cid[:, None]])


def to_T4(nodes, hexes, part):
    tets = np.concatenate([hexes[:, list(t)] for t in KUHN], axis=0)
    tpart = np.concatenate([part] * len(KUHN))
    X = nodes[tets]
    vol = np.einsum("ij,ij->i", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
    neg = vol < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    assert (np.abs(vol) > 1e-12).all(), "degenerate tetrahedron"
    return tets, tpart


def to_T10(nodes, tets):
    nodes2, ids = _edge_nodes(nodes, tets, TET_EDGES)
    return nodes2, np.hstack([tets, ids])


def tet_conformity(tets, hexes):
    """Every triangle is shared by at most two tetrahedra, and the boundary has two triangles per boundary quad."""
    faces = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    _, ct = np.unique(faces, axis=0, return_counts=True)
    quads = np.sort(np.concatenate([hexes[:, list(f)] for f in HEX_FACES]), axis=1)
    _, cq = np.unique(quads, axis=0, return_counts=True)
    return {"max_share": int(ct.max()), "boundary_triangles": int((ct == 1).sum()), "boundary_quads": int((cq == 1).sum())}


def mesh_for(kind, nodes8, hexes, part):
    """Nodes, connectivity and the part of every cell for the element kind, from the structured hexahedral mesh."""
    if kind in ("H8", "H8I"):
        return nodes8, hexes, part
    if kind in ("H20", "H20R"):
        n, c = to_H20(nodes8, hexes); return n, c, part
    if kind == "H27":
        n, c = to_H27(nodes8, hexes); return n, c, part
    tets, tpart = to_T4(nodes8, hexes, part)
    if kind == "T4":
        return nodes8, tets, tpart
    n, c = to_T10(nodes8, tets); return n, c, tpart


# ------------------------------------------------------------------ assembly
def assemble(nodes, conn, kind, E_cell, nu, chunk=None):
    """Global stiffness (3n x 3n, node-major) for the element kind; E_cell is Young's modulus of each cell."""
    base, (pts, wts) = RULES[kind]
    dN = shape_grads(base, pts)                      # (nq, nen, 3)
    nen = dN.shape[1]
    D1 = H8MOD.elastic_D(1.0, nu)
    n = nodes.shape[0]
    chunk = chunk or max(200, int(4e7 // (9 * nen * nen * 8)))
    K = None
    for s in range(0, conn.shape[0], chunk):
        c = conn[s:s + chunk]
        X = nodes[c]                                  # (E, nen, 3)
        Ke = np.zeros((c.shape[0], 3 * nen, 3 * nen))
        for q in range(pts.shape[0]):
            J = np.einsum("ena,nb->eab", X, dN[q])     # J[a, b] = dx_a / dxi_b
            det = np.linalg.det(J)
            assert (det > 0).all(), f"{kind}: non-positive Jacobian"
            Jinv = np.linalg.inv(J)
            dNdx = np.einsum("nb,eba->ena", dN[q], Jinv)
            B = H8MOD._B(dNdx)                         # (E, 6, 3 nen)
            Ke += (wts[q] * det * E_cell[s:s + chunk])[:, None, None] * np.einsum("eki,kl,elj->eij", B, D1, B)
        dofs = (3 * c[:, :, None] + np.arange(3)).reshape(c.shape[0], 3 * nen)
        rows = np.repeat(dofs, 3 * nen, axis=1).ravel()
        cols = np.tile(dofs, (1, 3 * nen)).ravel()
        Kc = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(3 * n, 3 * n)).tocsr()
        K = Kc if K is None else (K + Kc)
    K = 0.5 * (K + K.T)
    return K.tocsr()


def assemble_kind(kind, nodes, conn, cell_part, E_by_part, nu):
    """Stiffness for any kind; H8I through fejoint.hex8i, part by part, as fejoint.joint.analyse does."""
    if kind == "H8I":
        groups = [ps for ps in ((0,), (1, 2), (3,)) if np.isin(cell_part, ps).any()]
        Ks = [H8MOD.assemble(nodes, conn[np.isin(cell_part, ps)], E_by_part[ps[0]], nu) for ps in groups]
        K = Ks[0]
        for k_ in Ks[1:]:
            K = K + k_
        return K.tocsr()
    E_cell = np.array([E_by_part[p] for p in cell_part], dtype=float)
    return assemble(nodes, conn, kind, E_cell, nu)


def strain_energy(K, u):
    return 0.5 * float(u @ (K @ u))
