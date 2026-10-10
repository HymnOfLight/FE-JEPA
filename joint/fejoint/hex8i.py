"""Eight-node hexahedron with incompatible (bubble) modes, condensed at element level.

Displacement u = sum_a N_a u_a + sum_k P_k alpha_k, with trilinear N_a and the three
bubbles P_k = 1 - xi_k^2 (Wilson et al. 1973). The bubble strains use the Jacobian at the
element centre, scaled by det J0 / det J (Taylor, Beresford and Wilson 1976), so that the
integral of the bubble strain over the element vanishes and the patch test is passed on
distorted meshes. The nine internal parameters are eliminated by static condensation,

    K_e = K_uu - K_ua K_aa^{-1} K_au,

which is symmetric positive semi-definite with the six rigid-body modes as its null space.
The assembled stiffness is therefore an ordinary symmetric positive definite matrix once the
rigid-body modes are removed, so Pi(u) - Pi(u*) = 1/2 ||u - u*||_K^2 holds unchanged.

Degrees of freedom are node-major: [u_x, u_y, u_z] for node 0, then node 1, and so on.
Node order in an element: (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1), then the same at +1.
Engineering strain order: [e_xx, e_yy, e_zz, g_xy, g_yz, g_zx].
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

XI = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
               [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
_G = 1.0 / np.sqrt(3.0)
GAUSS = np.array([[a, b, c] for c in (-_G, _G) for b in (-_G, _G) for a in (-_G, _G)])  # weights all 1


def elastic_D(E: float, nu: float) -> np.ndarray:
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def dN_dxi(p: np.ndarray) -> np.ndarray:
    """Derivatives of the eight trilinear shape functions at natural point p, shape (8, 3)."""
    x, e, z = p
    d = np.empty((8, 3))
    d[:, 0] = XI[:, 0] * (1 + e * XI[:, 1]) * (1 + z * XI[:, 2]) / 8.0
    d[:, 1] = XI[:, 1] * (1 + x * XI[:, 0]) * (1 + z * XI[:, 2]) / 8.0
    d[:, 2] = XI[:, 2] * (1 + x * XI[:, 0]) * (1 + e * XI[:, 1]) / 8.0
    return d


def dP_dxi(p: np.ndarray) -> np.ndarray:
    """Derivatives of the three bubbles P_k = 1 - xi_k^2, shape (3 modes, 3 directions)."""
    return np.diag(-2.0 * np.asarray(p, dtype=float))


def _B(dfdx: np.ndarray) -> np.ndarray:
    """Strain-displacement matrices from gradients dfdx (E, n, 3): returns (E, 6, 3n), node-major."""
    E_, n, _ = dfdx.shape
    B = np.zeros((E_, 6, 3 * n))
    B[:, 0, 0::3] = dfdx[:, :, 0]
    B[:, 1, 1::3] = dfdx[:, :, 1]
    B[:, 2, 2::3] = dfdx[:, :, 2]
    B[:, 3, 0::3] = dfdx[:, :, 1]
    B[:, 3, 1::3] = dfdx[:, :, 0]
    B[:, 4, 1::3] = dfdx[:, :, 2]
    B[:, 4, 2::3] = dfdx[:, :, 1]
    B[:, 5, 0::3] = dfdx[:, :, 2]
    B[:, 5, 2::3] = dfdx[:, :, 0]
    return B


def element_matrices(X: np.ndarray, D: np.ndarray, keep_B: bool = True):
    """Uncondensed blocks for elements with nodal coordinates X (E, 8, 3).

    Returns K_uu (E,24,24), K_ua (E,24,9), K_aa (E,9,9) and the per-Gauss-point data
    needed for strain recovery: Bu (G,E,6,24), Ba (G,E,6,9), detJ (G,E)."""
    X = np.asarray(X, dtype=float)
    E_ = X.shape[0]
    J0 = np.einsum("ai,eaj->eij", dN_dxi(np.zeros(3)), X)
    det0 = np.linalg.det(J0)
    invJ0 = np.linalg.inv(J0)
    Kuu = np.zeros((E_, 24, 24)); Kua = np.zeros((E_, 24, 9)); Kaa = np.zeros((E_, 9, 9))
    Bus, Bas, dets = [], [], []
    for g in GAUSS:
        dN = dN_dxi(g)
        J = np.einsum("ai,eaj->eij", dN, X)
        det = np.linalg.det(J)
        if np.any(det <= 0):
            raise ValueError("inverted or degenerate hexahedron (det J <= 0 at a Gauss point)")
        invJ = np.linalg.inv(J)
        dNdx = np.einsum("eji,ai->eaj", invJ, dN)
        dPdx = np.einsum("eji,ki->ekj", invJ0, dP_dxi(g)) * (det0 / det)[:, None, None]
        Bu, Ba = _B(dNdx), _B(dPdx)
        DBu = np.einsum("ij,ejk->eik", D, Bu)
        DBa = np.einsum("ij,ejk->eik", D, Ba)
        Kuu += np.einsum("eji,ejk->eik", Bu, DBu) * det[:, None, None]
        Kua += np.einsum("eji,ejk->eik", Bu, DBa) * det[:, None, None]
        Kaa += np.einsum("eji,ejk->eik", Ba, DBa) * det[:, None, None]
        if keep_B:
            Bus.append(Bu); Bas.append(Ba); dets.append(det)
    if not keep_B:
        return Kuu, Kua, Kaa, None, None, None
    return Kuu, Kua, Kaa, np.array(Bus), np.array(Bas), np.array(dets)


def condensed_stiffness(X: np.ndarray, D: np.ndarray) -> np.ndarray:
    Kuu, Kua, Kaa, *_ = element_matrices(X, D, keep_B=False)
    S = np.linalg.solve(Kaa, np.transpose(Kua, (0, 2, 1)))
    Ke = Kuu - Kua @ S
    return 0.5 * (Ke + np.transpose(Ke, (0, 2, 1)))


def element_dofs(hexes: np.ndarray) -> np.ndarray:
    hexes = np.asarray(hexes)
    return (3 * hexes[:, :, None] + np.arange(3)[None, None, :]).reshape(hexes.shape[0], 24)


def assemble(nodes: np.ndarray, hexes: np.ndarray, E: float, nu: float, chunk: int = 4000) -> sp.csr_matrix:
    """Global condensed stiffness (3n x 3n), node-major."""
    D = elastic_D(E, nu)
    n = nodes.shape[0]
    rows, cols, vals = [], [], []
    for s in range(0, hexes.shape[0], chunk):
        h = hexes[s:s + chunk]
        Ke = condensed_stiffness(nodes[h], D)
        dofs = element_dofs(h)
        rows.append(np.repeat(dofs, 24, axis=1).ravel())
        cols.append(np.tile(dofs, (1, 24)).ravel())
        vals.append(Ke.ravel())
    K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * n, 3 * n))
    K = K.tocsr()
    K = 0.5 * (K + K.T)
    return K.tocsr()


def gauss_strains(nodes: np.ndarray, hexes: np.ndarray, u: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Strains at the 8 Gauss points of every element, shape (E, 8, 6), bubbles recovered."""
    D = elastic_D(E, nu)
    Kuu, Kua, Kaa, Bu, Ba, _ = element_matrices(nodes[hexes], D)
    ue = u[element_dofs(hexes)]
    alpha = -np.linalg.solve(Kaa, np.einsum("eji,ej->ei", Kua, ue)[..., None])[..., 0]
    eps = np.einsum("geij,ej->gei", Bu, ue) + np.einsum("geij,ej->gei", Ba, alpha)
    return np.transpose(eps, (1, 0, 2))


def von_mises(eps: np.ndarray, E: float, nu: float) -> np.ndarray:
    s = eps @ elastic_D(E, nu).T
    sxx, syy, szz, sxy, syz, szx = (s[..., i] for i in range(6))
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))


def box_mesh(xs, ys, zs):
    """Structured hexahedral mesh on the tensor grid xs x ys x zs (node index i + nx*(j + ny*k))."""
    xs, ys, zs = (np.asarray(v, dtype=float) for v in (xs, ys, zs))
    nx, ny, nz = len(xs), len(ys), len(zs)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    nodes = np.stack([X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")], axis=1)
    idx = lambda i, j, k: i + nx * (j + ny * k)  # noqa: E731
    I, J, K = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    hexes = np.stack([idx(I, J, K), idx(I + 1, J, K), idx(I + 1, J + 1, K), idx(I, J + 1, K),
                      idx(I, J, K + 1), idx(I + 1, J, K + 1), idx(I + 1, J + 1, K + 1), idx(I, J + 1, K + 1)], axis=1)
    return nodes, hexes


def rigid_body_modes(nodes: np.ndarray) -> np.ndarray:
    """The six rigid-body modes as columns (3n x 6), for algebraic multigrid."""
    n = nodes.shape[0]
    c = nodes - nodes.mean(axis=0)
    B = np.zeros((3 * n, 6))
    B[0::3, 0] = 1; B[1::3, 1] = 1; B[2::3, 2] = 1
    B[0::3, 3] = -c[:, 1]; B[1::3, 3] = c[:, 0]
    B[1::3, 4] = -c[:, 2]; B[2::3, 4] = c[:, 1]
    B[2::3, 5] = -c[:, 0]; B[0::3, 5] = c[:, 2]
    return B
