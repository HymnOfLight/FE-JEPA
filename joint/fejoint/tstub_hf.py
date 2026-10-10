"""Welded T-stub pair, higher-fidelity elastic contact model (gate G-C1b, docs/PRESPEC_G-C1b.md).

Differences from the baseline (fejoint.tstub, the G-C1a joint model's machinery carried over to a T-stub):
  bolt holes   cut through the flange, diameter d0, stair-stepped on the mesh;
  solid bolts  each half bolt is a body of its own: a round shank of tensile stress area A_s passing
               through the hole without touching its wall, and a round head of diameter s (across flats)
               and height k. The head's underside touches the flange only on the bearing face,
               r <= d_w / 2 (r from the bolt axis; the hole removes r < d0 / 2), unilaterally and without
               friction. No shear springs: nothing ties the flange to the bolt sideways;
  bolt length  the half bolt's axial compliance, from a uniform pressure on its bearing face to the
               symmetry plane, is L_eff / 2 / (E_b A_s), with L_eff after Agerskov as quoted by Bursi and
               Jaspart (1997, eqs 1 and 2), as their finite element model did; the shank's Young's modulus
               is calibrated on the same mesh to achieve it (the head keeps E_b).
Kept from the baseline: linear elasticity, small strain, frictionless contact between the flanges (the
mid-plane x = 0 of the pair acts as a rigid frictionless support, u_x >= 0), no preload, zero initial
gaps, 45 degree fillet welds of leg a_w * sqrt(2) stair-stepped on the mesh. Friction is left out on
purpose: it has no potential energy, and the reference model must stay inside the energy framework of
docs/contact_energy_note.md. Leaving it out can only make the model softer.

Symmetry: one T-stub and half of each bolt (about x = 0; the bolt's mid-section keeps u_x = 0 and is
free to slide, which is the mirror condition), half the width (about the web mid-plane z = 0) and, for
two bolt rows, half the length (about y = b / 2; the bolt row sits at y = (b - p) / 2). For one row the
whole length y in [0, b] is modelled, the row at y = e from the end y = 0 where LVDT HP1 sits, and the
float along y is removed by fixing u_y at one node of the loaded end, which carries no force.
The bolt bodies float sideways and spin about their axis at zero energy (no friction, no hole contact);
springs of total stiffness reg * E_b A_s / (L_eff / 2) on the lateral displacements of each shank's
mid-section remove that, and carry no force because no lateral load acts on a bolt.

Frame: x normal to the flanges, x = 0 the plane between them, flange x in [0, t_f], web from x = t_f to
x = t_f + H; z across, z = 0 the web mid-plane; y along the T-stub.
Reading (Girao Coelho et al. 2004, Sec. 2.3 and Table 9): the gap between the flanges at the web
centreline at both ends of the specimen, Delta = 2 u_x(0, y_end, 0), averaged; K = F / Delta, F the
total tension on the pair.

Contact formulation: for each node pair (flange node f on the bearing face, head node a at the same
place) the separation s = u_x,a - u_x,f >= 0 replaces u_x,f as an unknown (u_x,f = u_x,a - s). The change
of variables T is linear and its own inverse, so the constraints stay simple bounds and the feasible set
stays a closed convex cone (docs/contact_energy_note.md, Section 7). Units: mm, N, MPa.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse as sp

from . import hex8i as H
from .contact import solve_signorini
from .linsolve import SPDSolver
from .joint import _graded


@dataclass(frozen=True)
class BoltSize:
    d: float     # nominal diameter
    As: float    # tensile stress area (ISO 898-1)
    k: float     # head height (ISO 4014 / 4017)
    m: float     # nut height (ISO 4032)
    s: float     # width across flats
    dw: float    # bearing face diameter (ISO 4014 / 4017, product grade A, minimum)
    d0: float    # hole diameter, normal clearance (EN 1090-2: +1 mm up to M14, +2 mm M16 to M24)


BOLTS = {12: BoltSize(12.0, 84.3, 7.5, 10.8, 18.0, 16.63, 13.0),
         16: BoltSize(16.0, 157.0, 10.0, 14.8, 24.0, 22.49, 18.0),
         20: BoltSize(20.0, 245.0, 12.5, 18.0, 30.0, 28.19, 22.0)}


@dataclass
class WeldedTStub:
    name: str
    b: float; w: float; n: float; a_w: float      # length, gauge, bolt axis to flange tip, weld throat
    rows: tuple                                    # bolt rows along y from the end y = 0 (one or two)
    bolt: int                                      # nominal diameter, key of BOLTS
    E_f: float; E_w: float; E_b: float             # flange; web and welds; bolt
    t_f: float = 10.0; t_w: float = 10.0; nu: float = 0.3; H: float = 80.0
    bolt_length: str = "agerskov"                  # or "ec3": grip + (k + m) / 2
    d0: float | None = None                        # hole diameter, if not the normal clearance
    full: bool = False                             # model the whole length even with two rows (checks)

    @property
    def B(self): return BOLTS[self.bolt]
    @property
    def leg(self): return self.a_w * np.sqrt(2.0)
    @property
    def hole(self): return self.B.d0 if self.d0 is None else self.d0
    @property
    def symmetric(self): return len(self.rows) == 2 and not self.full
    @property
    def y_len(self): return self.b / 2 if self.symmetric else self.b
    @property
    def rows_model(self):
        if len(self.rows) == 2:                     # e, p, e with the measured b and p
            p = self.rows[1] - self.rows[0]
            return [(self.b - p) / 2] if self.symmetric else [(self.b - p) / 2, (self.b + p) / 2]
        assert len(self.rows) == 1
        return [self.rows[0]]

    @property
    def L_bolt(self):
        """Whole-bolt effective length. Agerskov (Bursi and Jaspart 1997, eqs 1, 2): E A_b / (K1 + 2 K4) =
        E A_s / L_eff, K1 = l_s + 1.43 l_t + 0.71 l_n, K4 = 0.1 l_n + 0.2 l_w; fully threaded in the grip
        (l_s = 0, l_t = 2 t_f), nut height l_n = m, no washers (l_w = 0). EC3: 2 t_f + (k + m) / 2."""
        bs = self.B
        if self.bolt_length == "ec3":
            return 2 * self.t_f + 0.5 * (bs.k + bs.m)
        A_b = np.pi * bs.d ** 2 / 4.0
        K1 = 1.43 * (2 * self.t_f) + 0.71 * bs.m
        K4 = 0.1 * bs.m
        return bs.As * (K1 + 2 * K4) / A_b


# h: in-plane spacing; n_b: cells across the bearing ring d0/2 <= r <= d_w/2 (sets the spacing h_b in the
# square around each bolt head); n_tf: layers through the flange; n_head: layers through the head
MESHES = {"coarse": dict(h=4.0, n_b=1.5, n_tf=2, n_head=2),
          "medium": dict(h=2.0, n_b=3.0, n_tf=4, n_head=3),
          "fine": dict(h=1.33, n_b=4.5, n_tf=6, n_head=4)}


def _zones_grid(breaks, h, zones=()):
    """Grid through all breakpoints; spacing <= h, and <= hz inside each zone (lo, hi, hz)."""
    b = np.unique(np.round(np.asarray(breaks, float), 9))
    out = [b[0]]
    for a, c in zip(b[:-1], b[1:]):
        hh = h
        mid = 0.5 * (a + c)
        for lo, hi, hz in zones:
            if lo - 1e-9 <= mid <= hi + 1e-9:
                hh = min(hh, hz)
        k = max(1, int(np.ceil((c - a) / hh - 1e-9)))
        out.extend(list(np.linspace(a, c, k + 1)[1:]))
    return np.array(out)


def _compact(nodes, hexes):
    used = np.unique(hexes)
    renum = -np.ones(nodes.shape[0], int); renum[used] = np.arange(used.size)
    return nodes[used], renum[hexes]


def _xface_areas(nodes, hexes, n):
    """Nodal shares (a quarter each) of the +x faces (local nodes 1, 2, 5, 6) of the given hexes."""
    w = np.zeros(n)
    fn = hexes[:, [1, 2, 5, 6]]
    dy = nodes[hexes[:, 2], 1] - nodes[hexes[:, 1], 1]
    dz = nodes[hexes[:, 5], 2] - nodes[hexes[:, 1], 2]
    np.add.at(w, fn.ravel(), np.repeat(np.abs(dy * dz) / 4.0, 4))
    return w


def build(g: WeldedTStub, h: float, n_b: float, n_tf: int, n_head: int = 3, n_tw: int = 2,
          growth: float = 1.2, hx_max: float = 10.0):
    bs = g.B
    R, rs, rh, rw = bs.s / 2, float(np.sqrt(bs.As / np.pi)), g.hole / 2, bs.dw / 2
    assert rs < rh < rw < R, "shank < hole < bearing face < head radius"
    h_b = min(h, (rw - rh) / n_b)
    zc, Ly, yr = g.w / 2, g.y_len, g.rows_model
    for y0 in yr:
        assert 0 < y0 - R and y0 + R < Ly, "head must lie inside the modelled length"
    assert zc + R < g.w / 2 + g.n and zc - R > g.t_w / 2 + g.leg, "head must clear the weld and the flange tip"
    radii = (rs, rh, rw, R)
    hw = min(h / 2, g.leg / 5)
    ybr = [0.0, Ly] + yr + [y0 + sg * r for y0 in yr for r in radii for sg in (-1, 1)]
    if len(yr) == 2:
        ybr.append(g.b / 2)                         # keeps the whole-length grid the mirror image of the half
    zbr = [0.0, g.t_w / 2, g.t_w / 2 + g.leg, zc, g.w / 2 + g.n] + [zc + sg * r for r in radii for sg in (-1, 1)]
    ys = _zones_grid(ybr, h, [(y0 - R, y0 + R, h_b) for y0 in yr])
    zs = _zones_grid(zbr, h, [(zc - R, zc + R, h_b), (g.t_w / 2, g.t_w / 2 + g.leg, hw), (0.0, g.t_w / 2, g.t_w / 2 / n_tw)])
    xs_f = np.linspace(0.0, g.t_f, n_tf + 1)
    xs = np.unique(np.round(np.r_[xs_f, _zones_grid([g.t_f, g.t_f + g.leg], hw),
                                  _graded(g.t_f + g.leg, g.t_f + g.H, hw, growth, hx_max)], 9))
    # flange, web and welds: one body
    nd, hx = H.box_mesh(xs, ys, zs)
    c = nd[hx].mean(axis=1)
    cx, cy, cz = c.T
    flange = cx < g.t_f
    for y0 in yr:
        flange &= np.hypot(cy - y0, cz - zc) >= rh
    web = (cx > g.t_f) & (cz < g.t_w / 2)
    weld = (cx > g.t_f) & (cz > g.t_w / 2) & ((cx - g.t_f) + (cz - g.t_w / 2) <= g.leg)
    keep = flange | web | weld
    mat = np.where(flange[keep], 0, 1)
    nd, hx = _compact(nd, hx[keep])
    nodes_l, hexes_l, mat_l, body_l = [nd], [hx], [mat], [np.zeros(nd.shape[0], int)]
    off = nd.shape[0]
    # one half bolt per modelled row: shank (material 2) and head (material 3)
    xs_b = np.r_[xs_f, g.t_f + np.linspace(0.0, bs.k, n_head + 1)[1:]]
    for i, y0 in enumerate(yr):
        ysb = ys[(ys >= y0 - R - 1e-9) & (ys <= y0 + R + 1e-9)]
        zsb = zs[(zs >= zc - R - 1e-9) & (zs <= zc + R + 1e-9)]
        nb, hb = H.box_mesh(xs_b, ysb, zsb)
        c = nb[hb].mean(axis=1)
        r = np.hypot(c[:, 1] - y0, c[:, 2] - zc)
        shank = (c[:, 0] < g.t_f) & (r < rs)
        head = (c[:, 0] > g.t_f) & (r < R)
        keepb = shank | head
        matb = np.where(shank[keepb], 2, 3)
        nb, hb = _compact(nb, hb[keepb])
        nodes_l.append(nb); hexes_l.append(hb + off); mat_l.append(matb); body_l.append(np.full(nb.shape[0], 1 + i))
        off += nb.shape[0]
    nodes = np.concatenate(nodes_l); hexes = np.concatenate(hexes_l); mat = np.concatenate(mat_l)
    body = np.concatenate(body_l)
    n = nodes.shape[0]
    # bearing faces: +x faces of top-layer flange cells whose centre is within r <= d_w / 2 of a bolt axis
    hx_f = hexes[mat == 0]
    top = np.isclose(nodes[hx_f[:, 1], 0], g.t_f)
    fc = nodes[hx_f[:, [1, 2, 5, 6]]].mean(axis=1)
    pairs = []
    kf = lambda j: (round(float(nodes[j, 1]), 6), round(float(nodes[j, 2]), 6))  # noqa: E731
    for i, y0 in enumerate(yr):
        bolt_nodes = np.nonzero((body == 1 + i) & np.isclose(nodes[:, 0], g.t_f))[0]
        key = {kf(j): j for j in bolt_nodes}
        sel = top & (np.hypot(fc[:, 1] - y0, fc[:, 2] - zc) <= rw + 1e-9)
        # keep only faces the head's underside covers completely (always so unless the mesh is very coarse)
        cover = np.array([all(kf(j) in key for j in face) for face in hx_f[:, [1, 2, 5, 6]][sel]], bool)
        idx = np.nonzero(sel)[0][cover]
        area = _xface_areas(nodes, hx_f[idx], n)
        fnodes = np.nonzero(area > 0)[0]
        anodes = np.array([key[kf(j)] for j in fnodes])
        pairs.append({"f": fnodes, "a": anodes, "area": area[fnodes], "faces_dropped": int((~cover).sum())})
    # loaded end: +x faces of the web cells at the far end
    hx_w = hexes[mat == 1]
    endc = np.isclose(nodes[hx_w[:, 1], 0], xs.max())
    load_area = _xface_areas(nodes, hx_w[endc], n)
    return {"nodes": nodes, "hexes": hexes, "mat": mat, "body": body, "pairs": pairs, "load_area": load_area,
            "grids": {"xs": xs, "ys": ys, "zs": zs, "xs_b": xs_b}, "radii": {"R": R, "rs": rs, "rh": rh, "rw": rw},
            "h_b": h_b,
            "rows_model": yr}


def _assemble_parts(g, msh):
    nodes, hexes, mat = msh["nodes"], msh["hexes"], msh["mat"]
    Ks = {}
    for m, E in ((0, g.E_f), (1, g.E_w), (2, 1.0), (3, g.E_b)):
        Ks[m] = H.assemble(nodes, hexes[mat == m], E, g.nu)
    return Ks


def calibrate_shank(g, msh, Ks, reg=1e-6, tol=1e-10):
    """Young's modulus of the shank that gives the half bolt the target axial compliance."""
    nodes, body = msh["nodes"], msh["body"]
    target = (g.L_bolt / 2) / (g.E_b * g.B.As)
    pr = msh["pairs"][0]
    bn = np.nonzero(body == 1)[0]
    dofs = (3 * bn[:, None] + np.arange(3)).ravel()
    bottom = bn[np.isclose(nodes[bn, 0], 0.0)]
    k_half = g.E_b * g.B.As / (g.L_bolt / 2)
    kreg = np.zeros(3 * nodes.shape[0])
    kreg[3 * bottom + 1] = kreg[3 * bottom + 2] = reg * k_half / bottom.size
    fixed = np.zeros(3 * nodes.shape[0], bool); fixed[3 * bottom] = True
    keep = dofs[~fixed[dofs]]
    f = np.zeros(3 * nodes.shape[0]); np.add.at(f, 3 * pr["a"], pr["area"] / pr["area"].sum())
    Ks1, Kh = Ks[2][keep][:, keep], (Ks[3] + sp.diags(kreg))[keep][:, keep]
    loc = -np.ones(3 * nodes.shape[0], int); loc[keep] = np.arange(keep.size)
    ia = loc[3 * pr["a"]]; wa = pr["area"] / pr["area"].sum()

    def comp(Es):
        S = SPDSolver((Es * Ks1 + Kh).tocsr())
        u = S.solve(f[keep]); S.free()
        return float(wa @ u[ia])

    e1, e2 = g.E_b, 0.5 * g.E_b
    c1, c2 = comp(e1), comp(e2)
    hist = [(e1, c1), (e2, c2)]
    for _ in range(20):                                   # secant on 1/E, c nearly affine in 1/E
        (ea, ca), (eb, cb) = hist[-2], hist[-1]
        slope = (cb - ca) / (1 / eb - 1 / ea)
        inv = 1 / eb + (target - cb) / slope
        if inv <= 0:
            raise RuntimeError("head compliance alone exceeds the target bolt compliance")
        en = 1 / inv
        cn = comp(en)
        hist.append((en, cn))
        if abs(cn / target - 1) < tol:
            break
    return {"E_shank": en, "target_mm_per_N": target, "achieved_mm_per_N": cn,
            "rel_error": cn / target - 1, "E_shank_over_E_b": en / g.E_b,
            "compliance_at_E_b_mm_per_N": c1}


def analyse(g: WeldedTStub, mesh: str | dict = "medium", F: float = 10000.0, reg: float = 1e-6,
            return_data: bool = False):
    mp = dict(MESHES[mesh]) if isinstance(mesh, str) else dict(mesh)
    msh = build(g, **mp)
    nodes, body = msh["nodes"], msh["body"]
    n = nodes.shape[0]; ndof = 3 * n
    x, y, z = nodes.T
    Ks = _assemble_parts(g, msh)
    cal = calibrate_shank(g, msh, Ks, reg=reg)
    # bolt mid-sections: u_x = 0, weak lateral springs
    fw = body == 0
    kreg = np.zeros(ndof)
    k_half = g.E_b * g.B.As / (g.L_bolt / 2)
    fixed = np.zeros(ndof, bool)
    for i in range(len(msh["rows_model"])):
        bot = np.nonzero((body == 1 + i) & np.isclose(x, 0.0))[0]
        fixed[3 * bot] = True
        kreg[3 * bot + 1] = kreg[3 * bot + 2] = reg * k_half / bot.size
    K = (Ks[0] + Ks[1] + cal["E_shank"] * Ks[2] + Ks[3] + sp.diags(kreg)).tocsr()
    # change of variables: the flange's u_x at a bearing node becomes the separation s >= 0
    fs = np.concatenate([p["f"] for p in msh["pairs"]]); as_ = np.concatenate([p["a"] for p in msh["pairs"]])
    rows_T = np.r_[np.arange(ndof), 3 * fs]
    cols_T = np.r_[np.arange(ndof), 3 * as_]
    vals_T = np.r_[np.ones(ndof), np.ones(fs.size)]
    vals_T[3 * fs] = -1.0
    T = sp.csr_matrix((vals_T, (rows_T, cols_T)), shape=(ndof, ndof))
    Kr = (T.T @ K @ T).tocsr()
    share = (0.25 if g.symmetric else 0.5)
    f = np.zeros(ndof); f[0::3] = share * F * msh["load_area"] / msh["load_area"].sum()
    fixed[3 * np.nonzero(fw & np.isclose(z, 0.0))[0] + 2] = True           # web mid-plane
    if g.symmetric:
        fixed[3 * np.nonzero(fw & np.isclose(y, g.y_len))[0] + 1] = True   # mid-length plane
    else:
        endn = np.nonzero(msh["load_area"] > 0)[0]
        anchor = endn[np.argmin(np.hypot(y[endn], z[endn]))]
        fixed[3 * anchor + 1] = True
    free = np.nonzero(~fixed)[0]
    pos = -np.ones(ndof, int); pos[free] = np.arange(free.size)
    A = Kr[free][:, free].tocsr(); b = f[free]
    bottom = np.nonzero(fw & np.isclose(x, 0.0))[0]
    c_bot, c_pair = pos[3 * bottom], pos[3 * fs]
    assert (c_bot >= 0).all() and (c_pair >= 0).all()
    cidx = np.r_[c_bot, c_pair]
    near_null = (T @ H.rigid_body_modes(nodes))[free]
    sol = solve_signorini(A, b, cidx, near_null=near_null)
    zr = np.zeros(ndof); zr[free] = sol["x"]
    u = (T @ zr).reshape(-1, 3)
    lam = sol["lam"]
    ends = [0.0] if g.symmetric else [0.0, g.b]
    gaps = []
    for ye in ends:
        m = np.nonzero(fw & np.isclose(x, 0.0) & np.isclose(y, ye) & np.isclose(z, 0.0))[0]
        assert m.size == 1
        gaps.append(2.0 * float(u[m[0], 0]))
    # the same reading taken higher up the web centre line (where a transducer might be clamped)
    profile = {}
    for xr in (g.t_f, g.t_f + g.leg, 20.0, 30.0):
        vals = []
        for ye in ends:
            line = np.nonzero(fw & np.isclose(y, ye) & np.isclose(z, 0.0))[0]
            line = line[np.argsort(x[line])]
            vals.append(2.0 * float(np.interp(xr, x[line], u[line, 0])))
        profile[f"{xr:.2f}"] = F / float(np.mean(vals)) / 1000.0
    B_model = float(lam[c_pair].sum())                   # head-to-flange force, the modelled bolt(s)
    Q_model = float(lam[c_bot].sum())                    # prying force on the modelled quarter / half
    resid = Kr @ zr - f
    shank_reaction = float(-resid[3 * np.nonzero((body > 0) & np.isclose(x, 0.0))[0]].sum())
    n_bolts_model = len(msh["rows_model"])
    out = {"name": g.name, "geometry": asdict(g),
           "mesh": {**mp, "h_b": msh["h_b"], "nodes": int(n), "dof": int(A.shape[0])},
           "K_kN_per_mm": F / float(np.mean(gaps)) / 1000.0, "gaps_mm": gaps, "F_N": F,
           "K_read_higher_kN_per_mm": profile,
           "bolt_force_over_F_per_bolt": B_model / (share * F / n_bolts_model) if n_bolts_model else None,
           "prying_over_F": Q_model / (share * F),
           "equilibrium_rel": (share * F + Q_model - B_model) / (share * F),
           "shank_reaction_rel": (shank_reaction - B_model) / max(B_model, 1e-300),
           "contact_fraction_flanges": float(sol["active"][:c_bot.size].mean()),
           "contact_fraction_heads": float(sol["active"][c_bot.size:].mean()),
           "pdas_iterations": sol["iterations"], "kkt": sol["kkt"], "calibration": cal,
           "L_bolt_mm": g.L_bolt, "n_bearing_nodes": int(fs.size),
           "bearing_faces_dropped": int(sum(p["faces_dropped"] for p in msh["pairs"]))}
    if return_data:
        out["_data"] = {"msh": msh, "u": u, "sol": sol, "T": T, "pos": pos, "K": K}
    return out
