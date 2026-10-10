"""Two-dimensional welded T-stub with unilateral contact, for the exploratory network probe of stage C1.3.

Half of one T-stub of a symmetric pair, in plane strain, per unit width out of plane. Units: mm, N, MPa.
- x is horizontal with x = 0 on the web centre line (symmetry: u_x = 0 there); y is vertical with y = 0 on the
  plane between the two flanges, which by symmetry acts as a rigid, frictionless foundation: u_y >= 0 on the
  underside of the flange (discrete Signorini condition, zero initial gap).
- The flange spans 0 <= x <= t_w/2 + m + n, 0 <= y <= t_f; the web spans 0 <= x <= t_w/2, t_f <= y <= t_f + H.
- The bolt is a spring k_b (N/mm per mm width) from the top of the flange at x = t_w/2 + m to the foundation.
- The web is pulled up by F/2 (N per mm width), as a uniform traction on its top edge.
The mesh has the same topology for every geometry; only the node coordinates change. Elements are four-node
quadrilaterals with incompatible modes (Wilson et al. 1973, with the correction of Taylor, Beresford and Wilson
1976), condensed at element level, as fejoint.hex8i in three dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse as sp

E_STEEL, NU = 210000.0, 0.3
XI = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
_G = 1.0 / np.sqrt(3.0)
GAUSS = np.array([[a, b] for b in (-_G, _G) for a in (-_G, _G)])          # weights all 1


def D_plane_strain(E: float = E_STEEL, nu: float = NU) -> np.ndarray:
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([[1 - nu, nu, 0.0], [nu, 1 - nu, 0.0], [0.0, 0.0, (1 - 2 * nu) / 2]])


def q4i_stiffness(xy: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Condensed 8 x 8 stiffness of a quadrilateral with incompatible modes; xy is 4 x 2, counter-clockwise.
    Unknowns are node-major: [u0x, u0y, u1x, u1y, ...]."""
    def grads(p):
        xi, eta = p
        dN = 0.25 * np.array([XI[:, 0] * (1 + eta * XI[:, 1]), XI[:, 1] * (1 + xi * XI[:, 0])])  # 2 x 4
        return dN
    J0 = grads((0.0, 0.0)) @ xy
    detJ0 = np.linalg.det(J0)
    J0inv = np.linalg.inv(J0)
    Kuu = np.zeros((8, 8)); Kua = np.zeros((8, 4)); Kaa = np.zeros((4, 4))
    for p in GAUSS:
        dN = grads(p)
        J = dN @ xy
        detJ = np.linalg.det(J)
        dNx = np.linalg.solve(J, dN)                                        # 2 x 4: d/dx, d/dy
        B = np.zeros((3, 8))
        B[0, 0::2] = dNx[0]; B[1, 1::2] = dNx[1]; B[2, 0::2] = dNx[1]; B[2, 1::2] = dNx[0]
        dP = np.array([[-2 * p[0], 0.0], [0.0, -2 * p[1]]])                 # rows: bubbles; cols: d/dxi, d/deta
        dPx = (J0inv @ dP.T).T * (detJ0 / detJ)                             # 2 bubbles x (d/dx, d/dy)
        Ba = np.zeros((3, 4))
        for k in range(2):
            Ba[0, 2 * k] = dPx[k, 0]; Ba[1, 2 * k + 1] = dPx[k, 1]
            Ba[2, 2 * k] = dPx[k, 1]; Ba[2, 2 * k + 1] = dPx[k, 0]
        Kuu += B.T @ D @ B * detJ; Kua += B.T @ D @ Ba * detJ; Kaa += Ba.T @ D @ Ba * detJ
    Ke = Kuu - Kua @ np.linalg.solve(Kaa, Kua.T)
    return 0.5 * (Ke + Ke.T)


@dataclass(frozen=True)
class TStub2D:
    t_f: float           # flange thickness
    m: float             # web face to bolt axis
    n: float             # bolt axis to flange tip
    k_b: float           # bolt spring, N/mm per mm width
    t_w: float = 10.0    # web thickness
    H: float = 40.0      # modelled web height
    F: float = 100.0     # total pull on the web of one T-stub, N per mm width (the half model carries F / 2)
    E: float = E_STEEL
    nu: float = NU


# divisions: web half-thickness, web face to bolt, bolt to tip, flange thickness, web height
DIV = dict(n_a=2, n_m=14, n_n=10, n_t=4, n_h=8)


def mesh(g: TStub2D, div: dict = DIV):
    """Nodes, elements and labelled node sets. The topology depends only on div."""
    xa = np.linspace(0.0, g.t_w / 2, div["n_a"] + 1)
    xm = np.linspace(g.t_w / 2, g.t_w / 2 + g.m, div["n_m"] + 1)[1:]
    xn = np.linspace(g.t_w / 2 + g.m, g.t_w / 2 + g.m + g.n, div["n_n"] + 1)[1:]
    xs = np.r_[xa, xm, xn]
    ys_f = np.linspace(0.0, g.t_f, div["n_t"] + 1)
    ys_w = np.linspace(g.t_f, g.t_f + g.H, div["n_h"] + 1)[1:]
    nx, nyf = xs.size, ys_f.size
    nodes = [(x, y) for y in ys_f for x in xs]                              # flange grid, row-major in y
    idf = np.arange(nx * nyf).reshape(nyf, nx)
    nwx = div["n_a"] + 1
    idw = np.zeros((ys_w.size + 1, nwx), int)
    idw[0] = idf[-1, :nwx]                                                  # shared with the flange top
    for j, y in enumerate(ys_w):
        for i in range(nwx):
            idw[j + 1, i] = len(nodes); nodes.append((xs[i], y))
    nodes = np.array(nodes)
    els = [[idf[j, i], idf[j, i + 1], idf[j + 1, i + 1], idf[j + 1, i]] for j in range(nyf - 1) for i in range(nx - 1)]
    els += [[idw[j, i], idw[j, i + 1], idw[j + 1, i + 1], idw[j + 1, i]] for j in range(ys_w.size) for i in range(nwx - 1)]
    els = np.array(els)
    sets = {"symmetry": np.nonzero(np.isclose(nodes[:, 0], 0.0))[0],
            "contact": idf[0],                                              # flange underside, x ascending
            "bolt": int(idf[-1, div["n_a"] + div["n_m"]]),                  # top of the flange at the bolt axis
            "load": idw[-1],                                                # web top edge, x ascending
            "centre": int(idf[0, 0])}                                       # underside at the web centre line
    return nodes, els, sets


def element_matrices(g: TStub2D, nodes: np.ndarray, els: np.ndarray) -> np.ndarray:
    D = D_plane_strain(g.E, g.nu)
    return np.stack([q4i_stiffness(nodes[e], D) for e in els])


def load_vector(g: TStub2D, nodes: np.ndarray, sets: dict) -> np.ndarray:
    f = np.zeros(2 * nodes.shape[0])
    top = sets["load"]
    q = (g.F / 2) / (g.t_w / 2)                                             # traction on the half web
    for a, b in zip(top[:-1], top[1:]):
        L = abs(nodes[b, 0] - nodes[a, 0])
        f[2 * a + 1] += q * L / 2; f[2 * b + 1] += q * L / 2
    return f


def assemble(g: TStub2D, nodes, els, sets, Ke=None):
    Ke = element_matrices(g, nodes, els) if Ke is None else Ke
    dofs = np.stack([2 * els[:, k] + c for k in range(4) for c in (0, 1)], axis=1)   # node-major per element
    r = np.repeat(dofs, 8, axis=1).ravel(); c = np.tile(dofs, (1, 8)).ravel()
    n = 2 * nodes.shape[0]
    K = sp.csr_matrix((Ke.reshape(len(els), 64).ravel(), (r, c)), shape=(n, n))
    b = 2 * sets["bolt"] + 1
    K = K + sp.csr_matrix(([g.k_b], ([b], [b])), shape=(n, n))
    return K.tocsr()


def solve(g: TStub2D, div: dict = DIV, constrained: bool = True):
    """Exact discrete solution: primal-dual active set (fejoint.contact.solve_signorini) on the unknowns left after
    the symmetry condition. constrained=False drops the contact condition (penetration allowed)."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from fejoint.contact import solve_signorini
    from fejoint.linsolve import SPDSolver
    nodes, els, sets = mesh(g, div)
    Ke = element_matrices(g, nodes, els)
    K = assemble(g, nodes, els, sets, Ke)
    f = load_vector(g, nodes, sets)
    n = K.shape[0]
    fixed = np.zeros(n, bool); fixed[2 * sets["symmetry"]] = True
    free = np.nonzero(~fixed)[0]
    pos = -np.ones(n, int); pos[free] = np.arange(free.size)
    Kr, fr = K[free][:, free], f[free]
    u = np.zeros(n)
    if constrained:
        cidx = pos[2 * sets["contact"] + 1]
        assert (cidx >= 0).all()
        o = solve_signorini(Kr, fr, cidx)
        u[free] = o["x"]; kkt = o["kkt"]; it = o["iterations"]
        lam = np.zeros(n); lam[free] = o["lam"]
    else:
        S = SPDSolver(Kr); u[free] = S.solve(fr); S.free(); kkt = None; it = 0; lam = np.zeros(n)
    Pi = 0.5 * u @ (K @ u) - f @ u
    return {"u": u, "Pi": float(Pi), "lam": lam, "kkt": kkt, "iterations": it, "nodes": nodes, "els": els,
            "sets": sets, "Ke": Ke, "f": f, "K": K}


def energy(K, f, u):
    return 0.5 * u @ (K @ u) - f @ u


def u_scale(g: TStub2D) -> float:
    """Label-free displacement scale: tip deflection of a unit-width cantilever of length m under F / 2,
    plus the stretch of the bolt spring under F / 2. Used only to scale the network's output and loss."""
    Ep = g.E / (1 - g.nu ** 2)
    return (g.F / 2) * (4 * g.m ** 3 / (Ep * g.t_f ** 3) + 1.0 / g.k_b)


RANGES = {"t_f": (8.0, 20.0), "m": (20.0, 45.0), "n": (15.0, 40.0), "log10_k_b": (np.log10(3e3), np.log10(3e4))}


def sample(n: int, seed: int) -> list[TStub2D]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(TStub2D(t_f=float(rng.uniform(*RANGES["t_f"])), m=float(rng.uniform(*RANGES["m"])),
                           n=float(rng.uniform(*RANGES["n"])), k_b=float(10 ** rng.uniform(*RANGES["log10_k_b"]))))
    return out


__all__ = ["TStub2D", "DIV", "mesh", "element_matrices", "load_vector", "assemble", "solve", "energy", "u_scale",
           "sample", "RANGES", "q4i_stiffness", "D_plane_strain", "asdict"]
