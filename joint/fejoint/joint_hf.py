"""Extended end plate joint, higher-fidelity elastic contact model (gate G-C1a2, docs/PRESPEC_G-C1a2.md).

Changes from model "head_contact" of fejoint.joint (the model of gate G-C1a), carried over from the T-stub
model that passed gate G-C1b (fejoint.tstub_hf):
  bolt holes    cut through the end plate, diameter d0, stair-stepped on the mesh;
  solid bolts   each bolt's head (round, diameter s across flats, height k) and the part of its shank inside
                the end plate (round, tensile stress area A_s, passing through the hole without touching its
                wall) are one solid body. The head bears on the plate's beam-side face x = tp only on the
                bearing face, r <= d_w / 2 (the hole removes r < d0 / 2), unilaterally and without friction;
  rest of bolt  the bolt below the plate's contact face, inside the rigid column flange and down to the nut,
                is represented at the shank's section x = 0, which moves as a plane (u_x = a + b (y - y_r)
                + c (z - z_r)): an axial spring on a and bending springs on b and c, to the column.
                The axial spring is calibrated on the same mesh so that the whole bolt's compliance, from a
                uniform pressure on the bearing face to the column, is L_eff / (E_b A_s), with L_eff after
                Agerskov (Bursi and Jaspart 1997, eqs 1 and 2) for the grip t_p + t_fc, threaded through
                the grip, no washers. The bending springs are E_b I_s / L_rest, L_rest = t_fc + m / 2: the
                bolt below the plate as a cantilever from the nut, free to move sideways in its hole;
  shear         the vertical shear has no frictionless path to the column except the bolts: as in G-C1a,
                the EN 1993-1-8 bolt shear springs, here spread over the bearing ring of the plate face.
Kept from G-C1a: rigid column flange with frictionless unilateral contact on the plate's contact face (no
initial gap), no preload, linear elasticity, small strain, half model about the beam web, the load P/2 on the
beam section at x = tp + L_load, and the readings (the paper's rotation, eqs 2 to 5; the DT9 gap).
The loading device and supports are as in G-C1a: the column is rigid, the load is applied over a beam
section and the beam beyond it is free, as in the test (joint paper Sec. 2.4); no boundary condition here
lets a part tilt that the test held.

Frame and units as fejoint.joint: x along the beam (x = 0 the column face), y vertical, z across (z = 0 the
web's mid-plane); mm, N, MPa.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import numpy as np
import scipy.sparse as sp

from . import hex8i as H
from .contact import solve_signorini
from .linsolve import SPDSolver
from .joint import JointGeometry, _graded, _face_weights, bolt_section
from .tstub_hf import _zones_grid, _compact, _xface_areas, BOLTS

MESHES = {"coarse": dict(h=10.0, n_b=1.5, n_tp=2, n_tf=2, n_head=2),
          "medium": dict(h=5.0, n_b=3.0, n_tp=2, n_tf=2, n_head=3),
          "fine": dict(h=2.5, n_b=4.5, n_tp=4, n_tf=2, n_head=4)}


def L_eff_agerskov(g: JointGeometry) -> float:
    """Whole-bolt effective length, Agerskov as quoted by Bursi and Jaspart (1997), threaded through the grip."""
    A_b = np.pi * g.d ** 2 / 4.0
    grip = g.tp + g.t_fc + g.n_washers * g.washer_t
    K1 = 1.43 * grip + 0.71 * g.nut
    K4 = 0.1 * g.nut + 0.2 * g.n_washers * g.washer_t
    return g.As * (K1 + 2 * K4) / A_b


def build(g: JointGeometry, h: float, n_b: float, n_tp: int, n_tf: int, n_head: int = 3, n_tw: int = 1,
          growth: float = 1.15, hx_max: float = 40.0, a_flange: float | None = None, a_web: float | None = None):
    """a_flange, a_web: throats of the 45 degree fillet welds joining the beam flanges (both faces) and web
    (both faces) to the plate; None leaves the welds out, as in G-C1a."""
    bs = BOLTS[int(round(g.d))]
    d0 = g.hole_d if g.hole_d is not None else bs.d0
    R, rs, rh, rw = bs.s / 2, float(np.sqrt(g.As / np.pi)), d0 / 2, g.washer_r
    assert rs < rh < rw < R, "shank < hole < bearing face < head radius"
    h_b = min(h, (rw - rh) / n_b)
    zc, rows = g.w / 2, g.rows
    assert zc - R > g.twb / 2 and zc + R < g.bp / 2, "heads must clear the web and the plate edge"
    for r in rows:
        for lo, hi in ((g.y_top - g.tfb, g.y_top), (g.y_bot, g.y_bot + g.tfb)):
            assert r + R < lo or r - R > hi, "a head overlaps a beam flange"
        assert 0 < r - R and r + R < g.hp
    radii = (rs, rh, rw, R)
    lf = 0.0 if a_flange is None else a_flange * np.sqrt(2.0)       # weld legs
    lw = 0.0 if a_web is None else a_web * np.sqrt(2.0)
    welds = lf > 0 or lw > 0
    hwd = min(h / 4, min(v for v in (lf, lw) if v > 0) / 5) if welds else h   # refines with h
    if welds:
        for r in rows:                                                # heads clear of the weld toes
            for lo, hi in ((g.y_top - g.tfb - lf, g.y_top + lf), (g.y_bot - lf, g.y_bot + g.tfb + lf)):
                assert r + R < lo or r - R > hi, "a head overlaps a flange weld"
        assert zc - R > g.twb / 2 + lw, "a head overlaps a web weld"
    ybr = [0, g.hp, g.y_bot, g.y_bot + g.tfb, g.y_top - g.tfb, g.y_top] + list(rows) + \
          [r + sg * q for r in rows for q in radii for sg in (-1, 1)]
    zbr = [0, g.twb / 2, zc, g.bp / 2, g.bb / 2] + [zc + sg * q for q in radii for sg in (-1, 1)]
    yz = [(r - R, r + R, h_b) for r in rows] + [(g.y_bot, g.y_bot + g.tfb, g.tfb / n_tf),
                                                 (g.y_top - g.tfb, g.y_top, g.tfb / n_tf)]
    zz = [(zc - R, zc + R, h_b), (0.0, g.twb / 2, g.twb / 2 / n_tw)]
    if lf > 0:
        bands = [(g.y_top, g.y_top + lf), (g.y_top - g.tfb - lf, g.y_top - g.tfb),
                 (g.y_bot + g.tfb, g.y_bot + g.tfb + lf), (g.y_bot - lf, g.y_bot)]
        ybr += [v for bnd in bands for v in bnd]
        yz += [(lo, hi, hwd) for lo, hi in bands]
    if lw > 0:
        zbr += [g.twb / 2 + lw]
        zz += [(g.twb / 2, g.twb / 2 + lw, hwd)]
    ys = _zones_grid(ybr, h, yz)
    zs = _zones_grid(zbr, h, zz)
    xs_p = np.linspace(0, g.tp, n_tp + 1)
    if welds:
        lmax = max(lf, lw)
        xs_w = _zones_grid([g.tp, g.tp + lmax], hwd)
        xs_b = np.unique(np.round(np.r_[xs_w, _graded(g.tp + lmax, g.tp + g.L_beam, hwd, growth, hx_max,
                                                       must=(g.tp + g.L_load, g.tp + g.x_dt1))], 9))
    else:
        xs_b = _graded(g.tp, g.tp + g.L_beam, h, growth, hx_max, must=(g.tp + g.L_load, g.tp + g.x_dt1))
    sel = lambda arr, lo, hi: arr[(arr >= lo - 1e-9) & (arr <= hi + 1e-9)]  # noqa: E731
    zf = sel(zs, 0, g.bb / 2)
    blocks = [(xs_p, ys, sel(zs, 0, g.bp / 2)),
              (xs_b, sel(ys, g.y_top - g.tfb, g.y_top), zf),
              (xs_b, sel(ys, g.y_bot, g.y_bot + g.tfb), zf),
              (xs_b, sel(ys, g.y_bot + g.tfb, g.y_top - g.tfb), sel(zs, 0, g.twb / 2))]
    if welds:                                                         # part 4: weld candidates, cut below
        blocks.append((sel(xs_b, g.tp, g.tp + max(lf, lw)), ys, sel(zs, 0, min(g.bp, g.bb) / 2)))
    all_nodes, all_hex, part = [], [], []
    for bi, (bx, by, bz) in enumerate(blocks):
        nd, hx = H.box_mesh(bx, by, bz)
        all_hex.append(hx + sum(a.shape[0] for a in all_nodes))
        all_nodes.append(nd)
        part.append(np.full(hx.shape[0], bi))
    nodes = np.concatenate(all_nodes); hexes = np.concatenate(all_hex); part = np.concatenate(part)
    uniq, inv = np.unique(np.round(nodes, 6), axis=0, return_inverse=True)
    nodes, hexes = uniq.astype(float), inv.ravel()[hexes]
    c = nodes[hexes].mean(axis=1)
    in_hole = np.zeros(hexes.shape[0], bool)
    for r in rows:
        in_hole |= (part == 0) & (np.hypot(c[:, 1] - r, c[:, 2] - zc) < rh)
    drop = in_hole
    if welds:                                                         # keep weld cells inside a fillet triangle
        cx, cy, cz = c.T
        dx = cx - g.tp
        wl = np.zeros(hexes.shape[0], bool)
        if lf > 0:
            zin = cz < min(g.bp, g.bb) / 2
            wl |= zin & (cy > g.y_top) & (dx + (cy - g.y_top) <= lf)
            wl |= zin & (cz > g.twb / 2) & (cy < g.y_top - g.tfb) & (dx + (g.y_top - g.tfb - cy) <= lf)
            wl |= zin & (cz > g.twb / 2) & (cy > g.y_bot + g.tfb) & (dx + (cy - g.y_bot - g.tfb) <= lf)
            wl |= zin & (cy < g.y_bot) & (dx + (g.y_bot - cy) <= lf)
        if lw > 0:
            wl |= (cy > g.y_bot + g.tfb) & (cy < g.y_top - g.tfb) & (cz > g.twb / 2) & (dx + (cz - g.twb / 2) <= lw)
        # a weld candidate cell inside a beam part is already meshed by that part
        inside_beam = (((cy > g.y_top - g.tfb) & (cy < g.y_top)) | ((cy > g.y_bot) & (cy < g.y_bot + g.tfb))
                       | ((cz < g.twb / 2) & (cy > g.y_bot) & (cy < g.y_top)))
        drop = drop | ((part == 4) & (~wl | inside_beam))
    hexes, part = hexes[~drop], part[~drop]
    nodes, hexes = _compact(nodes, hexes)
    mat = np.where(part == 0, 0, np.where(part == 3, 2, np.where(part == 4, 4, 1)))  # 0 plate, 1 flanges, 2 web, 4 welds
    n_jt = nodes.shape[0]
    nodes_l, hexes_l, mat_l, body_l = [nodes], [hexes], [mat], [np.zeros(n_jt, int)]
    off = n_jt
    xs_bolt = np.r_[xs_p, g.tp + np.linspace(0.0, g.head, n_head + 1)[1:]]
    for i, r in enumerate(rows):
        ysb = ys[(ys >= r - R - 1e-9) & (ys <= r + R + 1e-9)]
        zsb = zs[(zs >= zc - R - 1e-9) & (zs <= zc + R + 1e-9)]
        nb, hb = H.box_mesh(xs_bolt, ysb, zsb)
        cb = nb[hb].mean(axis=1)
        rr = np.hypot(cb[:, 1] - r, cb[:, 2] - zc)
        shank = (cb[:, 0] < g.tp) & (rr < rs)
        head = (cb[:, 0] > g.tp) & (rr < R)
        kb = shank | head
        nb, hb = _compact(nb, hb[kb])
        nodes_l.append(nb); hexes_l.append(hb + off); mat_l.append(np.full(hb.shape[0], 3))
        body_l.append(np.full(nb.shape[0], 1 + i)); off += nb.shape[0]
    nodes = np.concatenate(nodes_l); hexes = np.concatenate(hexes_l); mat = np.concatenate(mat_l)
    body = np.concatenate(body_l)
    n = nodes.shape[0]
    # bearing faces: +x faces of the plate's top-layer cells within r <= d_w / 2, covered by the head
    hx_p = hexes[mat == 0]
    top = np.isclose(nodes[hx_p[:, 1], 0], g.tp)
    fc = nodes[hx_p[:, [1, 2, 5, 6]]].mean(axis=1)
    kf = lambda j: (round(float(nodes[j, 1]), 6), round(float(nodes[j, 2]), 6))  # noqa: E731
    pairs = []
    for i, r in enumerate(rows):
        head_under = np.nonzero((body == 1 + i) & np.isclose(nodes[:, 0], g.tp))[0]
        key = {kf(j): j for j in head_under}
        selc = top & (np.hypot(fc[:, 1] - r, fc[:, 2] - zc) <= rw + 1e-9)
        cover = np.array([all(kf(j) in key for j in face) for face in hx_p[:, [1, 2, 5, 6]][selc]], bool)
        idx = np.nonzero(selc)[0][cover]
        area = _xface_areas(nodes, hx_p[idx], n)
        fn = np.nonzero(area > 0)[0]
        pairs.append({"f": fn, "a": np.array([key[kf(j)] for j in fn]), "area": area[fn],
                      "faces_dropped": int((~cover).sum())})
    beam_nodes = np.unique(hexes[(mat == 1) | (mat == 2) | (mat == 4)])
    for p in pairs:
        assert not np.isin(p["f"], beam_nodes).any(), "a bearing face overlaps the beam's footprint"
    return {"nodes": nodes, "hexes": hexes, "mat": mat, "body": body, "pairs": pairs, "n_joint_nodes": n_jt,
            "grids": {"ys": ys, "zs": zs, "xs_p": xs_p, "xs_b": xs_b, "xs_bolt": xs_bolt}, "h_b": h_b,
            "radii": {"R": R, "rs": rs, "rh": rh, "rw": rw}, "d0": d0,
            "welds": {"leg_flange": float(lf), "leg_web": float(lw), "cells": int((mat == 4).sum()),
                      "volume_over_exact": _weld_volume_ratio(g, nodes, hexes, mat, lf, lw)}}


def _weld_volume_ratio(g, nodes, hexes, mat, lf, lw):
    """Meshed weld volume over the exact fillet volume (the centroid rule overstates it by about 1/n)."""
    if not (mat == 4).any():
        return None
    p = nodes[hexes[mat == 4]]
    vol = float(np.prod(p.max(axis=1) - p.min(axis=1), axis=1).sum())
    zmax = min(g.bp, g.bb) / 2
    exact = 0.5 * lf ** 2 * (2 * zmax + 2 * (zmax - g.twb / 2)) + 0.5 * lw ** 2 * (g.hb - 2 * g.tfb)
    return vol / exact


def _assemble(g, msh):
    nodes, hexes, mat = msh["nodes"], msh["hexes"], msh["mat"]
    K = sp.csr_matrix((3 * nodes.shape[0], 3 * nodes.shape[0]))
    for m, E in ((0, g.Ep), (1, g.Ef), (2, g.Ew), (3, g.Eb), (4, g.Ew)):    # welds: the web's modulus
        if (mat == m).any():
            K = K + H.assemble(nodes, hexes[mat == m], E, g.nu)
    return K.tocsr()


def analyse(g: JointGeometry, mesh: str | dict = "medium", P_full: float = 10000.0, reg: float = 1e-6,
            rot_factor: float = 1.0, a_flange: float | None = None, a_web: float | None = None,
            L_eff: float | None = None, return_data: bool = False):
    """rot_factor scales the bending springs of the bolt below the plate (1: E_b I_s / L_rest).
    a_flange, a_web: fillet weld throats (None: no welds). L_eff: whole-bolt length (None: Agerskov)."""
    mp = dict(MESHES[mesh]) if isinstance(mesh, str) else dict(mesh)
    msh = build(g, **mp, a_flange=a_flange, a_web=a_web)
    nodes, body, pairs = msh["nodes"], msh["body"], msh["pairs"]
    n = nodes.shape[0]; ndof = 3 * n
    x, y, z = nodes.T
    rows, zc = g.rows, g.w / 2
    nb = len(rows)
    K = _assemble(g, msh)
    # bolt bottoms: the section x = 0 of each shank moves as a plane, u_x = a + b (y - y_r) + c (z - z_r)
    bottoms = [np.nonzero((body == 1 + i) & np.isclose(x, 0.0))[0] for i in range(nb)]
    _, I_s = bolt_section(g)
    L_eff = L_eff_agerskov(g) if L_eff is None else L_eff
    L_rest = g.t_fc + 0.5 * g.nut
    k_rot = rot_factor * g.Eb * I_s / L_rest
    k_tot = g.Eb * g.As / L_eff
    # change of variables. Full vector: [node dofs (ndof), heads a, b, c (3 nb)]. Reduced: the same slots,
    # where a bearing flange u_x slot holds s (u_x,f = u_x,a - s) and a bottom u_x slot is eliminated
    nfull = ndof + 3 * nb
    fs = np.concatenate([p["f"] for p in pairs]); as_ = np.concatenate([p["a"] for p in pairs])
    bot_ux = np.concatenate([3 * b_ for b_ in bottoms])
    keep = np.ones(nfull, bool); keep[bot_ux] = False
    red_of = -np.ones(nfull, int); red_of[keep] = np.arange(int(keep.sum()))
    nred = int(keep.sum())
    rT, cT, vT = [], [], []
    ident = np.nonzero(keep)[0]
    vals = np.ones(ident.size)
    vals[np.isin(ident, 3 * fs)] = -1.0                       # u_x,f = -s + u_x,a
    rT.append(ident); cT.append(red_of[ident]); vT.append(vals)
    rT.append(3 * fs); cT.append(red_of[3 * as_]); vT.append(np.ones(fs.size))
    for i, (bt, r) in enumerate(zip(bottoms, rows)):
        base = ndof + 3 * i
        rT += [3 * bt, 3 * bt, 3 * bt]
        cT += [np.full(bt.size, red_of[base]), np.full(bt.size, red_of[base + 1]), np.full(bt.size, red_of[base + 2])]
        vT += [np.ones(bt.size), y[bt] - r, z[bt] - zc]
    T = sp.csr_matrix((np.concatenate(vT), (np.concatenate(rT), np.concatenate(cT))), shape=(nfull, nred))
    # springs: rest of the bolt (axial calibrated below), bending; shear springs on the bearing rings;
    # weak lateral springs on the bottoms (their float sideways and spin cost no energy otherwise)
    diag = np.zeros(ndof)
    for p in pairs:
        wts = p["area"] / p["area"].sum()
        diag[3 * p["f"] + 1] += g.k_shear * wts
        diag[3 * p["f"] + 2] += g.k_shear * wts
    for bt in bottoms:
        diag[3 * bt + 1] += reg * k_tot / bt.size
        diag[3 * bt + 2] += reg * k_tot / bt.size
    # calibration of the axial spring: compliance of head and shank with the bottom plane held (a = b = c = 0)
    cal = []
    k_ax = []
    for i, p in enumerate(pairs):
        bn = np.nonzero(body == 1 + i)[0]
        dofs = np.r_[3 * bn, 3 * bn + 1, 3 * bn + 2]
        dofs = dofs[~np.isin(dofs, bot_ux)]
        Kb = (K + sp.diags(diag))[dofs][:, dofs].tocsr()
        f_ = np.zeros(ndof); f_[3 * p["a"]] = p["area"] / p["area"].sum()
        S = SPDSolver(Kb); ub = S.solve(f_[dofs]); S.free()
        loc = -np.ones(ndof, int); loc[dofs] = np.arange(dofs.size)
        c_solid = float((p["area"] / p["area"].sum()) @ ub[loc[3 * p["a"]]])
        c_target = L_eff / (g.Eb * g.As)
        if c_solid >= c_target:
            raise RuntimeError("head and shank alone are more compliant than the whole bolt")
        k_ax.append(1.0 / (c_target - c_solid))
        cal.append({"c_solid_mm_per_N": c_solid, "c_target_mm_per_N": c_target, "k_rest_N_per_mm": k_ax[-1]})
    head_k = np.ravel([[k_ax[i], k_rot, k_rot] for i in range(nb)])
    Kfull = sp.block_diag([K + sp.diags(diag), sp.diags(head_k)]).tocsr()
    Kr = (T.T @ Kfull @ T).tocsr()
    # load: P_full / 2 downward over the beam section at x = tp + L_load
    f = np.zeros(nred)
    xl = g.tp + g.L_load
    jn = body == 0
    sec = np.nonzero(jn & np.isclose(x, xl, atol=1e-6))[0]
    wts = _face_weights(nodes, sec, (1, 2)); wts = wts / wts.sum()
    np.add.at(f, red_of[3 * sec + 1], -0.5 * P_full * wts)
    fixed = np.zeros(nred, bool)
    fixed[red_of[3 * np.nonzero(jn & np.isclose(z, 0))[0] + 2]] = True     # symmetry plane
    free = np.nonzero(~fixed)[0]
    pos = -np.ones(nred, int); pos[free] = np.arange(free.size)
    A = Kr[free][:, free].tocsr(); b = f[free]
    col = np.nonzero(jn & np.isclose(x, 0))[0]                             # plate's contact face
    c_col, c_pair = pos[red_of[3 * col]], pos[red_of[3 * fs]]
    assert (c_col >= 0).all() and (c_pair >= 0).all()
    cidx = np.r_[c_col, c_pair]
    sol = solve_signorini(A, b, cidx)
    ur = np.zeros(nred); ur[free] = sol["x"]
    ufull = T @ ur
    u = ufull[:ndof]; U = u.reshape(-1, 3)
    heads = ufull[ndof:].reshape(nb, 3)
    lam = sol["lam"]
    # readings, as fejoint.joint
    Ph = 0.5 * P_full
    xs_b = msh["grids"]["xs_b"]

    def section_v(xx):
        idx = np.nonzero(jn & np.isclose(x, xx))[0]
        idx = idx[(y[idx] >= g.y_bot - 1e-9) & (y[idx] <= g.y_top + 1e-9)]
        w_ = _face_weights(nodes, idx, (1, 2))
        return float((U[idx, 1] * w_).sum() / w_.sum())

    v_dt1 = section_v(g.tp + g.x_dt1)
    a1, L1 = g.tp + g.x_dt1, g.tp + g.L_load
    d_eb1 = Ph * a1 ** 2 * (3 * L1 - a1) / (6 * g.beam_EI_half())
    phi_p = float(np.arctan(-v_dt1 / a1) - d_eb1 / a1)
    phi_gross = float(np.arctan(-v_dt1 / a1))
    M_face = P_full * L1
    cf_ = col
    target = np.array([0.0, g.y_top - g.tfb / 2, g.bp / 2])
    i9 = cf_[np.argmin(np.linalg.norm(nodes[cf_] - target, axis=1))]
    gap9 = float(U[i9, 0])
    bolt_force = [float(lam[c_pair][np.isin(fs, p["f"])].sum()) for p in pairs]
    rest_force = [float(k_ax[i] * heads[i, 0]) for i in range(nb)]
    # vertical equilibrium: the load reaches the column only through the shear springs
    shear_y = float(sum((diag[3 * p["f"] + 1] * U[p["f"], 1]).sum() for p in pairs))
    col_force = float(lam[c_col].sum())
    out = {"geometry": asdict(g), "mesh": {**mp, "h_b": msh["h_b"], "nodes": int(n), "dof": int(A.shape[0])},
           "welds": msh["welds"], "rot_factor": rot_factor,
           "P_full_N": P_full, "M_face_Nmm": M_face, "L_eff_mm": L_eff, "L_rest_mm": L_rest,
           "k_rot_N_mm_per_rad": k_rot, "calibration": cal, "d0_mm": msh["d0"],
           "phi_paper_rad": phi_p, "S_paper_kNm_per_mrad": M_face / phi_p * 1e-9,
           "phi_gross_rad": phi_gross, "S_gross_kNm_per_mrad": M_face / phi_gross * 1e-9,
           "gap_DT9_mm": gap9, "gap_DT9_node_yz": [float(nodes[i9, 1]), float(nodes[i9, 2])],
           "gap_DT9_over_phi_mm_per_mrad": gap9 / (phi_p * 1e3),
           "bolt_force_N": bolt_force, "bolt_force_from_spring_N": rest_force,
           "vertical_equilibrium_rel": (shear_y + Ph) / Ph,
           "horizontal_equilibrium_rel": (col_force - sum(rest_force)) / max(sum(rest_force), 1e-300),
           "contact_fraction_column": float(sol["active"][:c_col.size].mean()),
           "contact_fraction_heads": float(sol["active"][c_col.size:].mean()),
           "pdas_iterations": sol["iterations"], "kkt": sol["kkt"],
           "bearing_faces_dropped": int(sum(p["faces_dropped"] for p in pairs))}
    if return_data:
        out["_data"] = {"msh": msh, "U": U, "sol": sol, "T": T, "pos": pos, "red_of": red_of, "Kr": Kr, "f": f,
                        "heads": heads, "k_ax": k_ax, "free": free, "K": K, "diag": diag, "head_k": head_k,
                        "bottoms": bottoms, "L_eff": L_eff}
    return out
