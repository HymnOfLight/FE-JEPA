"""Export the joint problem of fejoint.joint in nodal form for label-free training (stage C1.3).

The problem is the one fejoint.joint.analyse solves with support="contact" and washer="head_contact": the end
plate on a rigid, frictionless column flange; each bolt a rigid head plane a + b (y - y_r) + c (z - z_r) on its
bearing patch, which the plate may leave but not pass; the head's axial stiffness E A_s / L_b and tilt
stiffnesses k_rot; shear springs on the patches; a weak bonded spring of reg * E A_s / L_b per bolt on the
separations; symmetry u_z = 0 on z = 0; P_full / 2 downward on the beam section at the load point.

Nodal form. A network predicts three numbers per node, so the nine head unknowns (a, b, c for three bolts)
are condensed out exactly. With the unknowns split into nodal ones x and head ones h,

    Pi(x, h) = 1/2 x^T A_xx x + x^T A_xh h + 1/2 h^T A_hh h - f^T x      (no load acts on a head)
    min_h Pi = 1/2 x^T S x - f^T x,   S = A_xx - A_xh A_hh^-1 A_hx,   h(x) = -A_hh^-1 A_hx x.

The heads carry no constraint, so the condensed problem has the same solution x, the same minimum and the
same constraints, which stay simple bounds: x_i >= 0 on the column face (u_x) and on the patches, where the
x slot of a patch node holds the separation s_i from its head plane instead of u_x (u_x = a + b dy + c dz - s).

Beam segment (beam_segment = s). As fejoint.joint.analyse with the same option: the beam is meshed only up to
x = tp + s, its end section is rigid in its plane (u_x = c0 - theta (y - y_ref), u_y = c1), and it carries the
statically equivalent shear and moment; the deflection at DT1 beyond the segment is completed by beam theory.
The three section unknowns carry load, so condensing them leaves a nodal load and a constant:
    m(x) = A_mm^-1 (f_m - A_mx x),  S_c = A_xx - A_xm A_mm^-1 A_mx,  f_c = f_x - A_xm A_mm^-1 f_m,
    Pi_c(x) = 1/2 x^T S_c x - f_c^T x - 1/2 f_m^T A_mm^-1 f_m.
The u_x and u_y slots of the end section's nodes are then no unknowns of their own; they are marked Dirichlet
(masked to zero) and recovered from m(x).

Nothing here solves the problem; solve_exported() does, and is for labels and checks only.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import numpy as np
import scipy.sparse as sp

from . import hex8i as H
from .joint import JointGeometry, build_mesh, _face_weights, bolt_section
from .contact import solve_signorini


def export(g: JointGeometry, h: float = 10.0, n_tp: int = 1, n_tf: int = 1, n_tw: int = 1, P_full: float = 10000.0,
           k_rot_factor: float = 4.0, reg: float = 1e-6, growth: float = 1.15, hx_max: float = 40.0,
           beam_segment: float | None = None) -> dict:
    assert g.hole_d is None, "the idealised model: no holes"
    seg = beam_segment is not None
    if seg:
        assert beam_segment < g.L_load and beam_segment < g.x_dt1, "the segment must end before DT1 and the load"
    m = build_mesh(replace(g, L_beam=beam_segment) if seg else g, h, n_tp, n_tf, n_tw, growth, hx_max)
    nodes, hexes, part = m["nodes"], m["hexes"], m["part"]
    n = nodes.shape[0]
    x, y, z = nodes.T
    if g.E_plate is None and g.E_flange is None and g.E_web is None:
        K = H.assemble(nodes, hexes, g.E, g.nu)
    else:
        K = (H.assemble(nodes, hexes[part == 0], g.Ep, g.nu) + H.assemble(nodes, hexes[(part == 1) | (part == 2)], g.Ef, g.nu)
             + H.assemble(nodes, hexes[part == 3], g.Ew, g.nu)).tocsr()
    ndof = 3 * n
    face = np.nonzero(np.isclose(x, g.tp))[0]
    patches = [face[np.hypot(y[face] - r, z[face] - g.w / 2) <= g.washer_r + 1e-9] for r in g.rows]
    allp = np.concatenate(patches)
    assert np.unique(allp).size == allp.size, "washer patches overlap"
    assert not np.isin(allp, np.unique(hexes[part > 0])).any(), "a washer patch overlaps the beam's footprint"
    assert all(pt.size >= 3 for pt in patches), "washer patch with fewer than 3 nodes: refine the mesh"
    diag = np.zeros(ndof)
    for pt in patches:
        wts = _face_weights(nodes, pt, (1, 2)); wts = wts / wts.sum()
        diag[3 * pt + 1] += g.k_shear * wts
        diag[3 * pt + 2] += g.k_shear * wts
    nb, nh = len(patches), 3
    nfull = ndof + nh * nb
    # u_full = T x_full: identity except on the patch u_x rows, u_x = a + b dy + c dz - s (s in the u_x slot)
    is_pux = np.zeros(ndof, bool)
    for pt in patches:
        is_pux[3 * pt] = True
    ident = np.r_[np.nonzero(~is_pux)[0], ndof + np.arange(nh * nb)]
    rT, cT, vT = [ident], [ident], [np.ones(ident.size)]
    for bi, (pt, r) in enumerate(zip(patches, g.rows)):
        base = ndof + nh * bi
        rT += [3 * pt, 3 * pt, 3 * pt, 3 * pt]
        cT += [np.full(pt.size, base), np.full(pt.size, base + 1), np.full(pt.size, base + 2), 3 * pt]
        vT += [np.ones(pt.size), y[pt] - r, z[pt] - g.w / 2, -np.ones(pt.size)]
    T = sp.csr_matrix((np.concatenate(vT), (np.concatenate(rT), np.concatenate(cT))), shape=(nfull, nfull))
    _, I_s = bolt_section(g)
    k_rot = k_rot_factor * g.Eb * I_s / g.L_b
    head_k = np.tile([g.k_axial, k_rot, k_rot], nb)
    Kfull = sp.block_diag([K + sp.diags(diag), sp.diags(head_k)]).tocsr()
    A = (T.T @ Kfull @ T).tocsr()
    kreg = np.zeros(nfull)
    for pt in patches:
        kreg[3 * pt] = reg * g.k_axial / pt.size
    A = (A + sp.diags(kreg)).tocsr()
    # condense the heads
    hx = np.arange(ndof, nfull)
    Axx = A[:ndof][:, :ndof]
    Axh = A[:ndof][:, hx].toarray()
    Ahh = A[hx][:, hx].toarray()
    Ahh_inv_Ahx = np.linalg.solve(Ahh, Axh.T)                      # 9 x ndof, dense but few nonzero columns
    rows_h = np.nonzero(np.abs(Axh).sum(axis=1))[0]                # dofs coupled to a head
    corr = Axh[rows_h] @ Ahh_inv_Ahx[:, rows_h]
    S = (Axx - sp.csr_matrix((corr.ravel(), (np.repeat(rows_h, rows_h.size), np.tile(rows_h, rows_h.size))),
                             shape=(ndof, ndof))).tocsr()
    S = (0.5 * (S + S.T)).tocsr()
    dirichlet = np.zeros(ndof, bool); dirichlet[3 * np.nonzero(np.isclose(z, 0))[0] + 2] = True
    f = np.zeros(ndof)
    const = 0.0
    masters = None
    if not seg:                                 # P_full / 2 downward over the beam section at the load point
        sec = np.nonzero(np.isclose(x, g.tp + g.L_load, atol=1e-6))[0]
        wts = _face_weights(nodes, sec, (1, 2)); wts = wts / wts.sum()
        np.add.at(f, 3 * sec + 1, -0.5 * P_full * wts)
    else:                                       # rigid end section: three loaded unknowns, condensed
        sec = np.nonzero(np.isclose(x, g.tp + beam_segment, atol=1e-6))[0]
        y_ref = 0.5 * (g.y_top + g.y_bot)
        ux_e, uy_e = 3 * sec, 3 * sec + 1
        nz = ndof + 3
        keep = np.ones(ndof, bool); keep[ux_e] = False; keep[uy_e] = False
        idk = np.nonzero(keep)[0]
        T2 = sp.csr_matrix((np.r_[np.ones(idk.size), np.ones(sec.size), -(y[sec] - y_ref), np.ones(sec.size)],
                            (np.r_[idk, ux_e, ux_e, uy_e],
                             np.r_[idk, np.full(sec.size, ndof), np.full(sec.size, ndof + 2), np.full(sec.size, ndof + 1)])),
                           shape=(ndof, nz))
        A2 = (T2.T @ S @ T2).tocsr()
        mi = np.arange(ndof, nz)
        Amm = A2[mi][:, mi].toarray()
        Axm = A2[:ndof][:, mi].toarray()
        f_m = np.array([0.0, -0.5 * P_full, -0.5 * P_full * (g.L_load - beam_segment)])
        Amm_inv_Amx = np.linalg.solve(Amm, Axm.T)
        rows_m = np.nonzero(np.abs(Axm).sum(axis=1))[0]
        corr = Axm[rows_m] @ Amm_inv_Amx[:, rows_m]
        S = (A2[:ndof][:, :ndof] - sp.csr_matrix((corr.ravel(), (np.repeat(rows_m, rows_m.size),
                                                                 np.tile(rows_m, rows_m.size))), shape=(ndof, ndof))).tocsr()
        S = (0.5 * (S + S.T)).tocsr()
        Amm_inv_fm = np.linalg.solve(Amm, f_m)
        f = -(Axm @ Amm_inv_fm)
        const = -0.5 * float(f_m @ Amm_inv_fm)
        dirichlet[ux_e] = True; dirichlet[uy_e] = True
        masters = {"m0": Amm_inv_fm, "m_from_x": -Amm_inv_Amx, "f_m": f_m, "y_ref": y_ref, "section": sec}
    column = np.nonzero(np.isclose(x, 0))[0]
    nonneg = np.zeros(ndof, bool); nonneg[3 * column] = True; nonneg[3 * allp] = True
    # reading: deflection at DT1 and the constants of the paper's rotation (eqs 2 to 5)
    a1, L1 = g.tp + g.x_dt1, g.tp + g.L_load
    d_eb1 = 0.5 * P_full * a1 ** 2 * (3 * L1 - a1) / (6 * g.beam_EI_half())
    I_half = (g.bb / 2 * g.hb ** 3 - (g.bb / 2 - g.twb / 2) * (g.hb - 2 * g.tfb) ** 3) / 12.0
    if not seg:                                 # mean deflection of the DT1 section
        s1 = np.nonzero(np.isclose(x, g.tp + g.x_dt1))[0]
        s1 = s1[(y[s1] >= g.y_bot - 1e-9) & (y[s1] <= g.y_top + 1e-9)]
        w1 = _face_weights(nodes, s1, (1, 2)); w1 = w1 / w1.sum()
        reading = {"kind": "section", "dt1_nodes": s1, "dt1_weights": w1}
        # label-free displacement scale: tip deflection of the half beam as a cantilever from the column face
        u_scale = 0.5 * P_full * L1 ** 3 / (3 * g.E * I_half)
    else:                                       # rigid section's c1 and theta, then beam theory to DT1
        sd, Lc = g.x_dt1 - beam_segment, g.L_load - beam_segment
        A_web_half = (g.hb - 2 * g.tfb) * g.twb / 2
        Gw = g.Ew / (2 * (1 + g.nu))
        beyond = (-0.5 * P_full * sd ** 2 * (3 * Lc - sd) / (6 * g.beam_EI_half())
                  - 0.5 * P_full * sd / (Gw * A_web_half))
        s1 = np.zeros(0, int)
        reading = {"kind": "segment", "sd": sd, "beyond": beyond}
        # label-free displacement scale: the segment as a cantilever under the end shear and moment
        Ls = beam_segment
        u_scale = (0.5 * P_full * Ls ** 3 / (3 * g.E * I_half)
                   + 0.5 * P_full * (g.L_load - beam_segment) * Ls ** 2 / (2 * g.E * I_half))
    flags = np.zeros((n, 6), bool)
    flags[:, 0] = np.isclose(z, 0)                                  # symmetry plane
    flags[column, 1] = True                                         # column face, u_x >= 0
    flags[allp, 2] = True                                           # bearing patch, s >= 0 in the x slot
    flags[sec, 3] = True                                            # load section
    flags[s1, 4] = True                                             # DT1 section
    flags[np.unique(hexes[part == 0]), 5] = True                    # end plate
    return {"nodes": nodes, "hexes": hexes, "part": part, "K": S, "F": f, "dirichlet": dirichlet, "nonneg": nonneg,
            "node_flags": flags, "flag_names": ["symmetry", "column_face", "bearing_patch", "load_section",
                                                "dt1_section", "end_plate"],
            "heads_from_x": -Ahh_inv_Ahx, "patches": patches, "const": const, "masters": masters,
            "reading": {**reading, "a1": a1, "L1": L1, "d_eb1": d_eb1, "M_face": P_full * L1},
            "u_scale": u_scale,
            "meta": {"geometry": asdict(g), "mesh": {"h": h, "n_tp": n_tp, "n_tf": n_tf, "n_tw": n_tw,
                                                     "growth": growth, "hx_max": hx_max, "nodes": int(n),
                                                     "hexes": int(hexes.shape[0]), "beam_segment": beam_segment},
                     "model": {"support": "contact", "washer": "head_contact", "k_rot_factor": k_rot_factor,
                               "reg": reg, "P_full_N": P_full, "k_axial": g.k_axial, "k_rot": k_rot,
                               "k_shear": g.k_shear, "L_b": g.L_b}}}


def energy(ex: dict, u: np.ndarray) -> float:
    """Pi(u) = 1/2 u^T S u - f^T u + const with the Dirichlet unknowns zeroed (as the repository's anchor does;
    the anchor leaves out the constant, which moves no gradient and cancels in every energy gap)."""
    v = np.where(ex["dirichlet"], 0.0, u)
    return float(0.5 * v @ (ex["K"] @ v) - ex["F"] @ v + ex["const"])


def stiffness(ex: dict, u: np.ndarray) -> float:
    """S_paper in kNm/mrad from a nodal field: the paper's rotation at DT1 (eqs 2 to 5), as fejoint.joint."""
    r = ex["reading"]
    if r["kind"] == "section":
        v_dt1 = float(r["dt1_weights"] @ u[3 * r["dt1_nodes"] + 1])
    else:
        v = np.where(ex["dirichlet"], 0.0, u)
        c0, c1, th = ex["masters"]["m0"] + ex["masters"]["m_from_x"] @ v
        v_dt1 = float(c1 + th * r["sd"] + r["beyond"])
    phi = float(np.arctan(-v_dt1 / r["a1"]) - r["d_eb1"] / r["a1"])
    return r["M_face"] / phi * 1e-9


def physical_displacement(ex: dict, x: np.ndarray) -> np.ndarray:
    """The displacements of the nodes (3 per node) from an exported nodal vector x: the end section of a beam
    segment from its rigid motion, then the bearing patches' u_x = a + b dy + c dz - s from the heads."""
    u = np.where(ex["dirichlet"], 0.0, np.asarray(x, dtype=float)).copy()
    nodes = ex["nodes"]
    g = ex["meta"]["geometry"]
    if ex["masters"] is not None:
        c0, c1, th = ex["masters"]["m0"] + ex["masters"]["m_from_x"] @ u
        sec = ex["masters"]["section"]
        u[3 * sec] = c0 - th * (nodes[sec, 1] - ex["masters"]["y_ref"])
        u[3 * sec + 1] = c1
    h = ex["heads_from_x"] @ u
    r1 = g["hp"] - g["eX"]
    rows = [r1, r1 - g["p"], r1 - g["p"] - g["p23"]]
    for bi, (pt, r) in enumerate(zip(ex["patches"], rows)):
        a_, b_, c_ = h[3 * bi: 3 * bi + 3]
        u[3 * pt] = a_ + b_ * (nodes[pt, 1] - r) + c_ * (nodes[pt, 2] - g["w"] / 2) - u[3 * pt]
    return u


def solve_exported(ex: dict) -> dict:
    """Exact solution of the exported problem (active set method). For labels and checks, after the stamp."""
    free = np.nonzero(~ex["dirichlet"])[0]
    pos = -np.ones(ex["F"].size, int); pos[free] = np.arange(free.size)
    A = ex["K"][free][:, free].tocsr()
    cidx = pos[np.nonzero(ex["nonneg"])[0]]
    assert (cidx >= 0).all()
    sol = solve_signorini(A, ex["F"][free], cidx)
    u = np.zeros(ex["F"].size); u[free] = sol["x"]
    return {"u": u, "Pi": energy(ex, u), "S_paper": stiffness(ex, u), "kkt": sol["kkt"],
            "iterations": sol["iterations"], "active": sol["active"]}


__all__ = ["export", "energy", "stiffness", "physical_displacement", "solve_exported"]
