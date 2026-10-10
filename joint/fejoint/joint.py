"""Half model of an extended end plate beam-to-column joint with a rigid column flange.

Frame: x along the beam, x = 0 at the column face (the end plate's contact face); y vertical,
y = 0 at the bottom of the end plate; z across, z = 0 on the symmetry plane through the beam web.
Parts: end plate (x in [0, tp]), beam flanges and web (x in [tp, tp + L_beam]), meshed as one
conforming structured hexahedral mesh. The column flange is rigid: the plate's face x = 0 may
separate from it (u_x >= 0) but not penetrate it; no friction. Each bolt joins a patch of the plate
face x = tp (the washer footprint) to the rigid column through its axial stiffness E A_s / L_b, with
a transverse (shear) spring spread over the patch. Four ways to connect the patch to the bolt:

  "rigid"        the patch u_x all equal one head displacement a (bonded, no tilt);
  "springs"      the axial stiffness spread over the patch nodes, area weighted (no head at all);
  "tilt_tied"    the patch follows a rigid head plane a + b (y - y_r) + c (z - z_r) (bonded), and the
                 head tilts b, c are resisted by bending springs k_rot = factor * E I_s / L_b;
  "head_contact" as "tilt_tied", but the plate may separate from the head plane, never pass it:
                 s_i = a + b dy_i + c dz_i - u_x,i >= 0. The s_i replace the patch u_x as unknowns,
                 a linear change of variables, so the constraints stay simple bounds and the feasible
                 set stays a closed convex cone. Without a bonded link the stiffness matrix is
                 singular: an axial float of plate and beam between column and heads, a rigid pitch
                 about a line on x = tp, and with k_rot_factor = 0 the head tilts, all cost no strain
                 energy. A weak bonded spring on every s_i (total reg * E A_s / L_b per bolt) removes
                 them all.

With k_rot_factor = None the head plane stays parallel to the plate (no tilt unknowns).
support = "tied" bonds the plate to the column flange (Dirichlet u_x = 0 on x = 0); the bolt
heads keep their own model, so "head_contact" stays unilateral at the heads.
Linear elasticity, small strain, no preload, zero initial gaps.

Units: mm, N, MPa. Moments in N mm; rotational stiffness reported in kNm/mrad (1 kNm/mrad = 1e9 N mm/rad).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import numpy as np
import scipy.sparse as sp

from . import hex8i as H
from .contact import solve_signorini
from .linsolve import SPDSolver


@dataclass
class JointGeometry:
    tp: float; hp: float; bp: float           # end plate thickness, height, width
    eX: float; p: float; p23: float; w: float  # bolt rows: top edge to row 1, row 1 to 2, row 2 to 3; gauge
    LX: float                                  # plate top to the beam's top face
    hb: float; bb: float; tfb: float; twb: float
    L_beam: float; L_load: float; x_dt1: float  # measured from the plate's beam-side face (x = tp)
    t_fc: float                                # column flange thickness (bolt grip only; the flange is rigid)
    d: float = 20.0; As: float = 245.0; head: float = 13.0; nut: float = 16.0; washer_r: float = 18.5
    n_washers: int = 0; washer_t: float = 3.0   # washers in the grip (count, thickness each)
    E: float = 210000.0; nu: float = 0.3
    # optional Young's moduli by part (None: use E); the bolt modulus enters the bolt springs only
    E_plate: float | None = None; E_flange: float | None = None; E_web: float | None = None
    E_bolt: float | None = None
    L_b_override: float | None = None           # bolt length for the axial and bending stiffness, if given
    hole_d: float | None = None                 # if given, bolt holes of this diameter are cut from the plate

    @property
    def Ep(self): return self.E if self.E_plate is None else self.E_plate
    @property
    def Ef(self): return self.E if self.E_flange is None else self.E_flange
    @property
    def Ew(self): return self.E if self.E_web is None else self.E_web
    @property
    def Eb(self): return self.E if self.E_bolt is None else self.E_bolt
    def beam_EI_half(self):
        """Bending stiffness of the half beam section about its centroidal z axis (no root radii)."""
        hw = self.hb - 2 * self.tfb
        I_f = 2 * (self.bb / 2 * self.tfb ** 3 / 12 + self.bb / 2 * self.tfb * ((self.hb - self.tfb) / 2) ** 2)
        I_w = self.twb / 2 * hw ** 3 / 12
        return self.Ef * I_f + self.Ew * I_w

    @property
    def y_top(self): return self.hp - self.LX
    @property
    def y_bot(self): return self.y_top - self.hb
    @property
    def rows(self):
        r1 = self.hp - self.eX
        return [r1, r1 - self.p, r1 - self.p - self.p23]
    @property
    def L_b(self):          # EN 1993-1-8 Table 6.11: grip (plates and washers) + (head + nut) / 2
        if self.L_b_override is not None:
            return self.L_b_override
        return self.tp + self.t_fc + self.n_washers * self.washer_t + 0.5 * (self.head + self.nut)
    @property
    def k_axial(self): return self.Eb * self.As / self.L_b
    @property
    def k_shear(self):                          # EN 1993-1-8 Table 6.11, bolts in shear: k11 = 16 d^2 f_ub / (E d_M16)
        return 16.0 * self.d ** 2 * 800.0 / (self.E * 16.0) * self.E


def _subdivide(breaks, h, exact=None):
    """Grid through all breakpoints with spacing <= h; intervals listed in `exact` get that many parts."""
    b = np.unique(np.round(np.asarray(breaks, float), 9))
    out = [b[0]]
    for a, c in zip(b[:-1], b[1:]):
        n = None
        if exact:
            for (lo, hi, k) in exact:
                if abs(a - lo) < 1e-9 and abs(c - hi) < 1e-9:
                    n = k
        if n is None:
            n = max(1, int(np.ceil((c - a) / h - 1e-9)))
        out.extend(list(np.linspace(a, c, n + 1)[1:]))
    return np.array(out)


def _graded(x0, x1, h0, growth, hmax, must=()):
    xs = [x0]
    hcur = h0
    must = sorted(m for m in must if x0 < m < x1)
    while xs[-1] < x1 - 1e-9:
        nxt = xs[-1] + hcur
        hit = [m for m in must if xs[-1] < m < nxt + 0.25 * hcur]
        if hit:
            nxt = hit[0]
        xs.append(min(nxt, x1))
        hcur = min(hcur * growth, hmax)
    return np.array(xs)


def build_mesh(g: JointGeometry, h: float, n_tp: int, n_tf: int, n_tw: int = 1,
               growth: float = 1.15, hx_max: float = 40.0):
    extra_y, extra_z = [], []
    if g.hole_d is not None:                    # grid lines on the hole and bearing radii (holes only)
        for rr in (g.hole_d / 2, g.washer_r):
            extra_y += [r + sgn * rr for r in g.rows for sgn in (-1, 1)]
            extra_z += [g.w / 2 - rr, g.w / 2 + rr]
    ys = _subdivide([0, g.hp, g.y_bot, g.y_bot + g.tfb, g.y_top - g.tfb, g.y_top] + g.rows + extra_y, h,
                    exact=[(g.y_bot, g.y_bot + g.tfb, n_tf), (g.y_top - g.tfb, g.y_top, n_tf)])
    zs = _subdivide([0, g.twb / 2, g.w / 2, g.bp / 2, g.bb / 2] + extra_z, h, exact=[(0, g.twb / 2, n_tw)])
    xs_p = np.linspace(0, g.tp, n_tp + 1)
    xs_b = _graded(g.tp, g.tp + g.L_beam, h, growth, hx_max, must=(g.tp + g.L_load, g.tp + g.x_dt1))
    blocks = [(xs_p, ys, zs[zs <= g.bp / 2 + 1e-9])]
    sel = lambda arr, lo, hi: arr[(arr >= lo - 1e-9) & (arr <= hi + 1e-9)]  # noqa: E731
    zf = sel(zs, 0, g.bb / 2)
    blocks.append((xs_b, sel(ys, g.y_top - g.tfb, g.y_top), zf))
    blocks.append((xs_b, sel(ys, g.y_bot, g.y_bot + g.tfb), zf))
    blocks.append((xs_b, sel(ys, g.y_bot + g.tfb, g.y_top - g.tfb), sel(zs, 0, g.twb / 2)))
    all_nodes, all_hex, part = [], [], []
    for bi, (bx, by, bz) in enumerate(blocks):
        nd, hx = H.box_mesh(bx, by, bz)
        all_hex.append(hx + sum(a.shape[0] for a in all_nodes))
        all_nodes.append(nd)
        part.append(np.full(hx.shape[0], bi))
    nodes = np.concatenate(all_nodes)
    hexes = np.concatenate(all_hex)
    key = np.round(nodes, 6)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    nodes, hexes, part = uniq.astype(float), inv[hexes], np.concatenate(part)
    if g.hole_d is not None:                    # cut the holes (stair-stepped) and drop orphaned nodes
        c = nodes[hexes].mean(axis=1)
        in_hole = np.zeros(hexes.shape[0], bool)
        for r in g.rows:
            in_hole |= (part == 0) & (np.hypot(c[:, 1] - r, c[:, 2] - g.w / 2) < g.hole_d / 2)
        hexes, part = hexes[~in_hole], part[~in_hole]
        used = np.unique(hexes)
        renum = -np.ones(nodes.shape[0], int); renum[used] = np.arange(used.size)
        nodes, hexes = nodes[used], renum[hexes]
    return {"nodes": nodes, "hexes": hexes, "part": part,
            "grids": {"ys": ys, "zs": zs, "xs_p": xs_p, "xs_b": xs_b}}


def _face_weights(nodes, idx, axes):
    """Tributary areas of the nodes idx on a structured face spanned by the two given axes."""
    a, b = axes
    pa, pb = nodes[idx, a], nodes[idx, b]
    ua, ub = np.unique(np.round(pa, 6)), np.unique(np.round(pb, 6))
    def half_spans(u):
        d = np.diff(u)
        left = np.r_[0, d] / 2; right = np.r_[d, 0] / 2
        return dict(zip(np.round(u, 6), left + right))
    wa, wb = half_spans(ua), half_spans(ub)
    return np.array([wa[round(x, 6)] * wb[round(y, 6)] for x, y in zip(pa, pb)])


def bolt_section(g: JointGeometry):
    """Diameter and second moment of area of a round section with the tensile stress area A_s."""
    d_s = float(np.sqrt(4.0 * g.As / np.pi))
    return d_s, float(np.pi * d_s ** 4 / 64.0)


def analyse(g: JointGeometry, h: float = 5.0, n_tp: int = 2, n_tf: int = 2, n_tw: int = 1, P_full: float = 10000.0,
            support: str = "contact", washer: str = "rigid", growth: float = 1.15, hx_max: float = 40.0,
            k_rot_factor: float | None = 4.0, reg: float = 1e-6, beam_segment: float | None = None):
    """Linear elastic response to a downward load P_full at x = tp + L_load (the half model carries P_full / 2).

    support:      "contact" (the plate may separate from the rigid column flange) or "tied" (bonded to it).
    washer:       bolt model, one of "rigid", "springs", "tilt_tied", "head_contact" (module docstring).
    k_rot_factor: head tilt stiffness as a multiple of E I_s / L_b (I_s from the stress area);
                  None keeps the head plane parallel to the plate. Used by "tilt_tied" and "head_contact".
    reg:          weak bonded spring of "head_contact", relative to E A_s / L_b per bolt.
    beam_segment: if given, mesh the beam only up to x = tp + beam_segment, make that end section rigid in
                  its plane (u_x = c0 - theta (y - y_ref), u_y = c1) and load it with the statically
                  equivalent shear and moment; the deflection at DT1 beyond the segment is completed with
                  beam theory (bending and shear). For smaller training meshes (stage C1.3).
    Returns (summary dict, data dict with the reduced system and the maps needed to check it)."""
    seg = beam_segment is not None
    if seg:
        assert beam_segment < g.L_load, "the segment must end before the load point"
    m = build_mesh(replace(g, L_beam=beam_segment) if seg else g, h, n_tp, n_tf, n_tw, growth, hx_max)
    nodes, hexes = m["nodes"], m["hexes"]
    n = nodes.shape[0]
    x, y, z = nodes.T
    part = m["part"]
    if g.E_plate is None and g.E_flange is None and g.E_web is None:
        K = H.assemble(nodes, hexes, g.E, g.nu)
    else:                                        # parts: 0 plate, 1 and 2 flanges, 3 web
        K = (H.assemble(nodes, hexes[part == 0], g.Ep, g.nu) + H.assemble(nodes, hexes[(part == 1) | (part == 2)], g.Ef, g.nu)
             + H.assemble(nodes, hexes[part == 3], g.Ew, g.nu)).tocsr()
    ndof = 3 * n
    # bolt patches on the plate face x = tp
    face = np.nonzero(np.isclose(x, g.tp))[0]
    patches = [face[np.hypot(y[face] - r, z[face] - g.w / 2) <= g.washer_r + 1e-9] for r in g.rows]
    allp = np.concatenate(patches)
    assert np.unique(allp).size == allp.size, "washer patches overlap"
    beam_nodes = np.unique(hexes[m["part"] > 0])
    assert not np.isin(allp, beam_nodes).any(), "a washer patch overlaps the beam's footprint on the plate"
    assert all(pt.size >= 3 for pt in patches), "washer patch with fewer than 3 nodes: refine the mesh"
    # transverse (shear) springs spread over each patch, area weighted
    diag = np.zeros(ndof)
    for pt in patches:
        wts = _face_weights(nodes, pt, (1, 2)); wts = wts / wts.sum()
        diag[3 * pt + 1] += g.k_shear * wts
        diag[3 * pt + 2] += g.k_shear * wts
    nb = len(patches)
    if washer == "rigid":
        # dof map: patch u_x -> a shared bolt-head dof; transformation T: (ndof + nb) -> reduced
        master = np.arange(ndof + nb)
        for bi, pt in enumerate(patches):
            master[3 * pt] = ndof + bi
        keep = np.unique(master)
        newidx = -np.ones(ndof + nb, int); newidx[keep] = np.arange(keep.size)
        T = sp.csr_matrix((np.ones(ndof + nb), (np.arange(ndof + nb), newidx[master])), shape=(ndof + nb, keep.size))
        Kfull = sp.block_diag([K + sp.diags(diag), sp.diags(np.full(nb, g.k_axial))]).tocsr()
        Kr = (T.T @ Kfull @ T).tocsr()
        dof_of = lambda d: newidx[master[d]]  # noqa: E731
    elif washer == "springs":
        for pt in patches:
            wts = _face_weights(nodes, pt, (1, 2)); wts = wts / wts.sum()
            diag[3 * pt] += g.k_axial * wts
        Kr = (K + sp.diags(diag)).tocsr()
        T = None
        dof_of = lambda d: d  # noqa: E731
    elif washer in ("tilt_tied", "head_contact"):
        tilt = k_rot_factor is not None
        nh = 3 if tilt else 1
        nfull = ndof + nh * nb
        is_pux = np.zeros(ndof, bool)
        for pt in patches:
            is_pux[3 * pt] = True
        if washer == "head_contact":            # same numbering: a patch u_x slot now holds its s_i
            red_of = np.arange(nfull)
        else:                                   # patch u_x eliminated
            keep = np.r_[np.nonzero(~is_pux)[0], ndof + np.arange(nh * nb)]
            red_of = -np.ones(nfull, int); red_of[keep] = np.arange(keep.size)
        nred = int(red_of.max()) + 1
        ident = np.r_[np.nonzero(~is_pux)[0], ndof + np.arange(nh * nb)]
        rT, cT, vT = [ident], [red_of[ident]], [np.ones(ident.size)]
        for bi, (pt, r) in enumerate(zip(patches, g.rows)):
            base = ndof + nh * bi
            rT.append(3 * pt); cT.append(np.full(pt.size, red_of[base])); vT.append(np.ones(pt.size))
            if tilt:
                rT.append(3 * pt); cT.append(np.full(pt.size, red_of[base + 1])); vT.append(y[pt] - r)
                rT.append(3 * pt); cT.append(np.full(pt.size, red_of[base + 2])); vT.append(z[pt] - g.w / 2)
            if washer == "head_contact":
                rT.append(3 * pt); cT.append(red_of[3 * pt]); vT.append(-np.ones(pt.size))
        T = sp.csr_matrix((np.concatenate(vT), (np.concatenate(rT), np.concatenate(cT))), shape=(nfull, nred))
        _, I_s = bolt_section(g)
        k_rot = (k_rot_factor * g.Eb * I_s / g.L_b) if tilt else 0.0
        head_k = np.tile([g.k_axial] + ([k_rot, k_rot] if tilt else []), nb)
        Kfull = sp.block_diag([K + sp.diags(diag), sp.diags(head_k)]).tocsr()
        Kr = (T.T @ Kfull @ T).tocsr()
        if washer == "head_contact":
            kreg = np.zeros(nred)
            for pt in patches:
                kreg[red_of[3 * pt]] = reg * g.k_axial / pt.size
            Kr = (Kr + sp.diags(kreg)).tocsr()
        dof_of = lambda d: red_of[d]  # noqa: E731
    else:
        raise ValueError(f"unknown washer model {washer!r}")
    T2 = None
    if seg:                                     # rigid end section: eliminate its u_x, u_y for three masters
        x_end = g.tp + beam_segment
        sec_e = np.nonzero(np.isclose(x, x_end, atol=1e-6))[0]
        ux_e, uy_e = dof_of(3 * sec_e), dof_of(3 * sec_e + 1)
        nr0 = Kr.shape[0]
        elim = np.zeros(nr0, bool); elim[ux_e] = True; elim[uy_e] = True
        keep2 = np.nonzero(~elim)[0]
        col2 = -np.ones(nr0, int); col2[keep2] = np.arange(keep2.size)
        mc0, mc1, mth = keep2.size, keep2.size + 1, keep2.size + 2
        y_ref = 0.5 * (g.y_top + g.y_bot)
        T2 = sp.csr_matrix((np.r_[np.ones(keep2.size), np.ones(ux_e.size), -(y[sec_e] - y_ref), np.ones(uy_e.size)],
                            (np.r_[keep2, ux_e, ux_e, uy_e],
                             np.r_[col2[keep2], np.full(ux_e.size, mc0), np.full(ux_e.size, mth), np.full(uy_e.size, mc1)])),
                           shape=(nr0, keep2.size + 3))
        Kr = (T2.T @ Kr @ T2).tocsr()
        dof_of1 = dof_of
        dof_of = lambda d: col2[dof_of1(d)]  # noqa: E731
    nr = Kr.shape[0]
    f = np.zeros(nr)
    if seg:                                     # statically equivalent end load: shear and the moment of P/2
        f[mc1] = -0.5 * P_full
        f[mth] = -0.5 * P_full * (g.L_load - beam_segment)
    else:                                       # P_full/2 downward over the beam section at x = tp + L_load
        xl = g.tp + g.L_load
        sec = np.nonzero(np.isclose(x, xl, atol=1e-6))[0]
        wts = _face_weights(nodes, sec, (1, 2)); wts = wts / wts.sum()
        np.add.at(f, dof_of(3 * sec + 1), -0.5 * P_full * wts)
    # supports: symmetry plane u_z = 0; for "tied", also u_x = 0 on the column face
    fixed = np.zeros(nr, bool)
    fixed[dof_of(3 * np.nonzero(np.isclose(z, 0))[0] + 2)] = True
    contact_nodes = np.nonzero(np.isclose(x, 0))[0]
    col_dofs = dof_of(3 * contact_nodes)
    if support == "tied":
        fixed[col_dofs] = True
    elif support == "rigid_plate":              # diagnostic: the whole plate held fixed (a rigid joint)
        assert washer == "springs", "rigid_plate is a diagnostic for the beam alone; use washer='springs'"
        plate_nodes = np.unique(hexes[part == 0])
        for c_ in range(3):
            fixed[dof_of(3 * plate_nodes + c_)] = True
        fixed[col_dofs] = True
    elif support != "contact":
        raise ValueError(f"unknown support {support!r}")
    free = np.nonzero(~fixed)[0]
    pos = -np.ones(nr, int); pos[free] = np.arange(free.size)
    A = Kr[free][:, free].tocsr(); b = f[free]
    c_col = pos[col_dofs] if support == "contact" else np.zeros(0, int)
    s_slots = (np.concatenate([dof_of(3 * pt) for pt in patches]) if washer == "head_contact" and support != "rigid_plate"
               else np.zeros(0, int))
    cidx = np.r_[c_col, pos[s_slots]].astype(int)
    n_column = c_col.size
    assert (cidx >= 0).all()
    if cidx.size:
        sol = solve_signorini(A, b, cidx)
    else:
        x_ = SPDSolver(A).solve(b)
        sol = {"x": x_, "lam": A @ x_ - b, "active": np.zeros(0, bool), "iterations": None, "kkt": None}
    ur = np.zeros(nr); ur[free] = sol["x"]
    ur2 = ur
    if seg:
        ur = T2 @ ur2
    ufull = T @ ur if T is not None else ur
    u = ufull[:ndof]
    heads = None
    if washer in ("tilt_tied", "head_contact"):
        hd = ufull[ndof:].reshape(nb, nh)
        heads = {"bolt_force_N": [float(g.k_axial * a_) for a_ in hd[:, 0]],
                 "head_tilt_rad": [[float(v_) for v_ in row[1:]] for row in hd] if tilt else None,
                 "k_rot_N_mm_per_rad": float(k_rot) if tilt else None}
        if washer == "head_contact":
            s_all = np.concatenate([ur2[dof_of(3 * pt)] for pt in patches])
            act = sol["active"][n_column:]
            heads["patch_nodes_separated_fraction"] = float(1 - act.mean())
            heads["s_min_mm"] = float(s_all.min())
    U = u.reshape(-1, 3)
    # rotation (a): section rotation from the flanges' mean axial displacement next to the plate
    xs_b = m["grids"]["xs_b"]
    xa = xs_b[1]
    def flange_mean(ylo, yhi, xx):
        idx = np.nonzero(np.isclose(x, xx) & (y >= ylo - 1e-9) & (y <= yhi + 1e-9) & (z <= g.bb / 2 + 1e-9))[0]
        idx = idx[(y[idx] >= ylo - 1e-9) & (y[idx] <= yhi + 1e-9)]
        wts_ = _face_weights(nodes, idx, (1, 2))
        return float((U[idx, 0] * wts_).sum() / wts_.sum()), float((y[idx] * wts_).sum() / wts_.sum())
    ut, yt = flange_mean(g.y_top - g.tfb, g.y_top, xa)
    ub, yb = flange_mean(g.y_bot, g.y_bot + g.tfb, xa)
    phi_a = (ut - ub) / (yt - yb)
    # rotation (b): mean deflection at the DT1 section, minus the beam's own elastic bending and shear
    # (cantilever clamped at x = tp, load at L_load) and minus the rigid translation at x = tp
    def section_v(xx):
        idx = np.nonzero(np.isclose(x, xx))[0]
        idx = idx[(y[idx] >= g.y_bot - 1e-9) & (y[idx] <= g.y_top + 1e-9)]
        wts_ = _face_weights(nodes, idx, (1, 2))
        return float((U[idx, 1] * wts_).sum() / wts_.sum())
    I_half = (g.bb / 2 * g.hb ** 3 - (g.bb / 2 - g.twb / 2) * (g.hb - 2 * g.tfb) ** 3) / 12.0
    G_ = g.E / (2 * (1 + g.nu))
    A_web_half = (g.hb - 2 * g.tfb) * g.twb / 2
    Ph, a, Lf = 0.5 * P_full, g.x_dt1, g.L_load
    d_eb = Ph * a ** 2 * (3 * Lf - a) / (6 * g.E * I_half)
    d_sh = Ph * a / (G_ * A_web_half)
    v0 = section_v(g.tp)
    if seg and g.x_dt1 > beam_segment:          # rigid-section kinematics plus the beam beyond the segment
        sd = g.x_dt1 - beam_segment; Lc = g.L_load - beam_segment
        Gw = g.Ew / (2 * (1 + g.nu))
        v_dt1 = (ur2[mc1] + ur2[mth] * sd - 0.5 * P_full * sd ** 2 * (3 * Lc - sd) / (6 * g.beam_EI_half())
                 - 0.5 * P_full * sd / (Gw * A_web_half))
    else:
        v_dt1 = section_v(g.tp + g.x_dt1)
    phi_b = ((-v_dt1) - (-v0) - d_eb - d_sh) / a
    M_full = P_full * g.L_load
    to_kNm_mrad = 1e-9
    # rotation (paper): the definition of Girao Coelho et al. (2004), eqs (2) to (5), with positions measured
    # from the plate's contact face x = 0: phi = arctan(delta_DT1 / a1) - theta_el, theta_el = delta_EB(a1) / a1,
    # delta_EB the Euler-Bernoulli deflection of a cantilever of length L1 clamped at x = 0 (shear deformation
    # and the plate's vertical displacement are NOT subtracted, as in the paper); M = P L1.
    a1, L1 = g.tp + g.x_dt1, g.tp + g.L_load
    d_eb1 = Ph * a1 ** 2 * (3 * L1 - a1) / (6 * g.beam_EI_half())
    phi_p = float(np.arctan(-v_dt1 / a1) - d_eb1 / a1)
    M_face = P_full * L1
    # end plate gap at the DT9 position: contact face, mid-thickness of the tension flange, plate edge
    cf_ = np.nonzero(np.isclose(x, 0))[0]
    target = np.array([0.0, g.y_top - g.tfb / 2, g.bp / 2])
    i9 = cf_[np.argmin(np.linalg.norm(nodes[cf_] - target, axis=1))]
    gap9 = float(U[i9, 0])
    # end plate gap on the contact face at the level of the tension flange and at the plate tip
    cf = contact_nodes
    tension_band = cf[(y[cf] >= g.y_top - g.tfb - 1e-9) & (y[cf] <= g.y_top + 1e-9)]
    tip_band = cf[np.isclose(y[cf], g.hp)]
    out = {
        "geometry": asdict(g), "mesh": {"h": h, "n_tp": n_tp, "n_tf": n_tf, "n_tw": n_tw, "nodes": int(n),
                                        "hexes": int(hexes.shape[0]), "dof_reduced": int(A.shape[0])},
        "support": support, "washer": washer, "P_full_N": P_full, "M_full_Nmm": M_full,
        "k_axial_N_per_mm": g.k_axial, "k_shear_N_per_mm": g.k_shear, "L_b_mm": g.L_b,
        "phi_a_section_rad": phi_a, "phi_b_dt1_rad": phi_b,
        "S_a_kNm_per_mrad": M_full / phi_a * to_kNm_mrad, "S_b_kNm_per_mrad": M_full / phi_b * to_kNm_mrad,
        "phi_paper_rad": phi_p, "M_face_Nmm": M_face, "S_paper_kNm_per_mrad": M_face / phi_p * to_kNm_mrad,
        "gap_DT9_mm": gap9, "gap_DT9_node_yz": [float(nodes[i9, 1]), float(nodes[i9, 2])],
        "gap_DT9_over_phi_mm_per_mrad": gap9 / (phi_p * 1e3),
        "gap_at_tension_flange_mm_max": float(U[tension_band, 0].max()),
        "gap_at_plate_tip_mm_max": float(U[tip_band, 0].max()),
        "contact_fraction": float(sol["active"][:n_column].mean()) if support == "contact" else 1.0,
        "bolt_heads": heads, "k_rot_factor": k_rot_factor if washer in ("tilt_tied", "head_contact") else None,
        "reg": reg if washer == "head_contact" else None,
        "pdas_iterations": sol.get("iterations"), "kkt": sol.get("kkt"),
    }
    data = {"u": u, "nodes": nodes, "hexes": hexes, "A": A, "b": b, "cidx": cidx, "sol": sol, "n_column": n_column,
            "patches": patches, "pos": pos, "T": T, "ndof": ndof}
    if washer in ("tilt_tied", "head_contact"):
        data.update({"red_of": red_of, "nh": nh, "head_slots": ndof + nh * np.arange(nb), "s_slots": s_slots,
                     "kreg": kreg if washer == "head_contact" else None, "head_k": head_k})
    return out, data
