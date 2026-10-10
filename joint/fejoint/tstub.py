"""A pair of identical T-stubs bolted flange to flange, as in Bursi and Jaspart (1997) and in the isolated
T-stub tests of Girao Coelho et al. (J. Constr. Steel Res. 60 (2004) 269-311). Exploratory (post hoc to
gate G-C1a): checks whether the joint model's machinery reproduces a tension-zone component test. Gate
G-C1b uses it as the baseline model ("the G-C1a model carried over to a T-stub pair").

By symmetry one T-stub is modelled, on the plane between the flanges, which acts as a rigid frictionless
support (u_x >= 0 there). Quarter model: z = 0 is the web mid-plane, y = L/2 the mid-length plane; with
full_length = True the whole length y in [0, L] is modelled (for a single, off-centre bolt row); the shear
springs at the bolts then hold the T-stub along y.
Frame: x = 0 the contact plane, the flange in x in [0, t_f], the web from x = t_f to x = t_f + H.
The web-to-flange corner is filled either by a root radius (rolled section) or a 45 degree fillet weld
(welded plates), stair-stepped on the mesh. Each bolt: a rigid head plane in unilateral contact with the
flange's outer face (as model "head_contact" of fejoint.joint), free to tilt; half of the bolt is modelled,
so its axial stiffness is E A_s / (L_b / 2) and its bending stiffness factor * E I_s / (L_b / 2), with the
slope fixed at the symmetry plane (guided end): factor 1.
Load: total tension F on the pair (F / 4 on the quarter, F / 2 on the full-length half), uniform on the
web's end section. Readings:
  measure = "web"  relative displacement of two points on the webs at +-x_meas, Delta = 2 u_x(x_meas)
                   (area average over the web section), K = F / Delta;
  measure = "gap"  gap between the flanges at the web centreline at the ends of the specimen, where the
                   LVDTs of Girao Coelho et al. (2004, Sec. 2.3) sit: Delta = 2 u_x(0, y_end, 0), averaged
                   over the ends (y = 0, and y = L when the whole length is modelled), K = F / Delta.
Young's moduli: E for everything unless E_f (flange), E_w (web and corner) or E_b (bolt springs) is given.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse as sp

from . import hex8i as H
from .contact import solve_signorini
from .joint import _subdivide, _graded, _face_weights


@dataclass
class TStubPair:
    t_f: float; t_w: float; b: float; L: float      # flange and web thickness, flange width, length
    w: float                                         # bolt gauge (bolts at z = +-w/2)
    bolt_y: tuple                                    # bolt positions along y (in [0, L/2], or [0, L] if full_length)
    H: float                                         # web length modelled beyond the flange
    corner: str; r: float                            # "root" (radius r) or "weld" (45 degree fillet, leg r)
    d: float; As: float; head: float; nut: float; bearing_r: float; n_washers: int = 0; washer_t: float = 0.0
    E: float = 210000.0; nu: float = 0.3; x_meas: float = 150.0
    E_f: float | None = None; E_w: float | None = None; E_b: float | None = None
    full_length: bool = False; measure: str = "web"

    @property
    def L_b(self):                                   # whole bolt, EN 1993-1-8 Table 6.11
        return 2 * self.t_f + self.n_washers * self.washer_t + 0.5 * (self.head + self.nut)

    @property
    def Ef(self): return self.E if self.E_f is None else self.E_f
    @property
    def Ew(self): return self.E if self.E_w is None else self.E_w
    @property
    def Eb(self): return self.E if self.E_b is None else self.E_b
    @property
    def y_len(self): return self.L if self.full_length else self.L / 2


def build(g: TStubPair, h: float, n_tf: int, n_tw: int = 2, growth: float = 1.15, hx_max: float = 20.0,
          with_part: bool = False):
    xs = np.r_[np.linspace(0, g.t_f, n_tf + 1),
               _subdivide([g.t_f, g.t_f + g.r], h / 2)[1:],
               _graded(g.t_f + g.r, g.t_f + g.H, h, growth, hx_max, must=(g.x_meas,))[1:]]
    zs = _subdivide([0, g.t_w / 2, g.t_w / 2 + g.r, g.w / 2 - g.bearing_r, g.w / 2, g.w / 2 + g.bearing_r, g.b / 2], h,
                    exact=[(0, g.t_w / 2, n_tw)])
    zs = np.unique(np.r_[zs, _subdivide([g.t_w / 2, g.t_w / 2 + g.r], h / 2)])
    ys = _subdivide([0, g.y_len] + ([g.L / 2] if g.full_length else [])
                    + [y for yb in g.bolt_y for y in (yb - g.bearing_r, yb, yb + g.bearing_r)], h)
    nodes, hexes = H.box_mesh(xs, ys, zs)
    c = nodes[hexes].mean(axis=1)
    cx, cz = c[:, 0], c[:, 2]
    flange = cx < g.t_f
    web = (cz < g.t_w / 2) & (cx > g.t_f)
    du, dv = cx - g.t_f, cz - g.t_w / 2
    if g.corner == "root":
        corner = (du > 0) & (dv > 0) & (du < g.r) & (dv < g.r) & ((du - g.r) ** 2 + (dv - g.r) ** 2 >= g.r ** 2)
    else:
        corner = (du > 0) & (dv > 0) & (du + dv <= g.r)
    keep = flange | web | corner
    part = np.where(flange[keep], 0, 1)
    hexes = hexes[keep]
    used = np.unique(hexes)
    renum = -np.ones(nodes.shape[0], int); renum[used] = np.arange(used.size)
    if with_part:
        return nodes[used], renum[hexes], part
    return nodes[used], renum[hexes]


def analyse(g: TStubPair, h: float = 2.5, n_tf: int = 4, F: float = 10000.0, k_rot_factor: float = 1.0,
            reg: float = 1e-6, hx_max: float = 20.0):
    nodes, hexes, part = build(g, h, n_tf, hx_max=hx_max, with_part=True)
    n = nodes.shape[0]; ndof = 3 * n
    x, y, z = nodes.T
    if g.Ef == g.Ew:
        K = H.assemble(nodes, hexes, g.Ef, g.nu)
    else:
        K = (H.assemble(nodes, hexes[part == 0], g.Ef, g.nu) + H.assemble(nodes, hexes[part == 1], g.Ew, g.nu)).tocsr()
    face = np.nonzero(np.isclose(x, g.t_f))[0]
    patches = [face[np.hypot(y[face] - yb, z[face] - g.w / 2) <= g.bearing_r + 1e-9] for yb in g.bolt_y]
    assert all(p.size >= 3 for p in patches)
    nb = len(patches)
    Lh = g.L_b / 2                                     # half bolt
    k_ax = g.Eb * g.As / Lh
    d_s = np.sqrt(4 * g.As / np.pi); I_s = np.pi * d_s ** 4 / 64
    k_rot = k_rot_factor * g.Eb * I_s / Lh
    k_sh = 16.0 * g.d ** 2 * 800.0 / 16.0               # EN 1993-1-8 Table 6.11, per bolt
    diag = np.zeros(ndof)
    for p in patches:
        wts = _face_weights(nodes, p, (1, 2)); wts = wts / wts.sum()
        diag[3 * p + 1] += k_sh * wts; diag[3 * p + 2] += k_sh * wts
    # head contact: s_i = a + b (y - y_b) + c (z - w/2) - u_x,i >= 0 replaces the patch u_x
    nfull = ndof + 3 * nb
    is_pux = np.zeros(ndof, bool)
    for p in patches:
        is_pux[3 * p] = True
    ident = np.r_[np.nonzero(~is_pux)[0], ndof + np.arange(3 * nb)]
    rT, cT, vT = [ident], [ident], [np.ones(ident.size)]
    for k, (p, yb) in enumerate(zip(patches, g.bolt_y)):
        base = ndof + 3 * k
        rT += [3 * p] * 4
        cT += [np.full(p.size, base), np.full(p.size, base + 1), np.full(p.size, base + 2), 3 * p]
        vT += [np.ones(p.size), y[p] - yb, z[p] - g.w / 2, -np.ones(p.size)]
    T = sp.csr_matrix((np.concatenate(vT), (np.concatenate(rT), np.concatenate(cT))), shape=(nfull, nfull))
    head_k = np.tile([k_ax, k_rot, k_rot], nb)
    Kr = (T.T @ sp.block_diag([K + sp.diags(diag), sp.diags(head_k)]) @ T).tocsr()
    kreg = np.zeros(nfull)
    for p in patches:
        kreg[3 * p] = reg * k_ax / p.size
    Kr = (Kr + sp.diags(kreg)).tocsr()
    # load: F / 4 (quarter) or F / 2 (whole length) on the web end, uniform
    share = 0.5 if g.full_length else 0.25
    xe = x.max()
    endn = np.nonzero(np.isclose(x, xe))[0]
    wts = _face_weights(nodes, endn, (1, 2)); wts = wts / wts.sum()
    f = np.zeros(nfull); np.add.at(f, 3 * endn, share * F * wts)
    fixed = np.zeros(nfull, bool)
    fixed[3 * np.nonzero(np.isclose(z, 0))[0] + 2] = True            # web mid-plane
    if not g.full_length:                                             # (whole length: the shear springs
        fixed[3 * np.nonzero(np.isclose(y, g.L / 2))[0] + 1] = True  # hold the float along y) mid-length plane
    free = np.nonzero(~fixed)[0]
    pos = -np.ones(nfull, int); pos[free] = np.arange(free.size)
    A = Kr[free][:, free].tocsr(); b = f[free]
    contact = np.nonzero(np.isclose(x, 0))[0]
    cidx = np.r_[pos[3 * contact], pos[np.concatenate([3 * p for p in patches])]]
    assert (cidx >= 0).all()
    sol = solve_signorini(A, b, cidx)
    zr = np.zeros(nfull); zr[free] = sol["x"]
    u = (T @ zr)[:ndof].reshape(-1, 3)
    heads = (T @ zr)[ndof:].reshape(nb, 3)
    out = {"geometry": asdict(g), "mesh": {"h": h, "n_tf": n_tf, "nodes": int(n), "dof": int(A.shape[0])},
           "F_N": F, "bolt_force_total_N": float((2 if g.full_length else 4) * k_ax * heads[:, 0].sum()),
           "contact_fraction": float(sol["active"][:contact.size].mean()),
           "pdas_iterations": sol["iterations"], "kkt": sol["kkt"]}
    if g.measure == "web":
        sec = np.nonzero(np.isclose(x, g.x_meas))[0]
        wts = _face_weights(nodes, sec, (1, 2)); wts = wts / wts.sum()
        ux = float((u[sec, 0] * wts).sum())
        out.update({"K_kN_per_mm": F / (2 * ux) / 1000.0, "u_meas_mm": ux})
    elif g.measure == "gap":
        ends = [0.0, g.L] if g.full_length else [0.0]
        gaps = []
        for ye in ends:
            m = np.nonzero(np.isclose(x, 0) & np.isclose(y, ye) & np.isclose(z, 0))[0]
            assert m.size == 1
            gaps.append(2.0 * float(u[m[0], 0]))
        out.update({"K_kN_per_mm": F / float(np.mean(gaps)) / 1000.0, "gaps_mm": gaps})
    else:
        raise ValueError(g.measure)
    return out
