"""The joint model of fejoint.joint.analyse on any element kind of elements.py (exploratory, after gate G-C1.3).

Only the path the C1.3 reference uses: frictionless unilateral contact with the rigid column flange (u_x >= 0 on
every node of the face x = 0), bolt heads that bear unilaterally with tilt springs ("head_contact"), and a beam
segment whose end section is rigid in its plane and carries the statically equivalent shear and moment. The
operations follow fejoint.joint.analyse line by line; only the stiffness matrix and the node sets change with the
element kind (quadratic elements bring their mid nodes into the contact face, the bolt patches and the rigid end
section). With kind "H8I" the result must equal fejoint.joint.analyse to round-off; check_generic.py checks it.
"""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from fejoint.contact import solve_signorini
from fejoint.joint import JointGeometry, build_mesh, _face_weights, bolt_section
import elements as EL


def analyse_generic(g: JointGeometry, kind: str, h: float, n_tp: int, n_tf: int, n_tw: int = 1,
                    beam_segment: float = 300.0, P_full: float = 10000.0, k_rot_factor: float = 4.0,
                    reg: float = 1e-6, growth: float = 1.15, hx_max: float = 40.0, details: bool = False):
    """The joint's response for one element kind and mesh. With details=True it also returns the nodes, the
    contact nodes on the column face with their active set, the washer patches, and the number of nodes of the
    hexahedral mesh: every kind keeps those nodes first, so a lower index means a corner node."""
    t0 = time.time()
    m = build_mesh(replace(g, L_beam=beam_segment), h, n_tp, n_tf, n_tw, growth, hx_max)
    nodes, conn, cpart = EL.mesh_for(kind, m["nodes"], m["hexes"], m["part"])
    n = nodes.shape[0]
    x, y, z = nodes.T
    E_by_part = {0: g.Ep, 1: g.Ef, 2: g.Ef, 3: g.Ew}
    K = EL.assemble_kind(kind, nodes, conn, cpart, E_by_part, g.nu)
    t_asm = time.time() - t0
    ndof = 3 * n
    face = np.nonzero(np.isclose(x, g.tp))[0]
    patches = [face[np.hypot(y[face] - r, z[face] - g.w / 2) <= g.washer_r + 1e-9] for r in g.rows]
    allp = np.concatenate(patches)
    assert np.unique(allp).size == allp.size, "washer patches overlap"
    beam_nodes = np.unique(conn[cpart > 0])
    assert not np.isin(allp, beam_nodes).any(), "a washer patch overlaps the beam's footprint on the plate"
    assert all(pt.size >= 3 for pt in patches), "washer patch with fewer than 3 nodes: refine the mesh"
    diag = np.zeros(ndof)
    for pt in patches:
        wts = _face_weights(nodes, pt, (1, 2)); wts = wts / wts.sum()
        diag[3 * pt + 1] += g.k_shear * wts
        diag[3 * pt + 2] += g.k_shear * wts
    nb = len(patches)
    nh = 3
    nfull = ndof + nh * nb
    is_pux = np.zeros(ndof, bool)
    for pt in patches:
        is_pux[3 * pt] = True
    red_of = np.arange(nfull)
    nred = nfull
    ident = np.r_[np.nonzero(~is_pux)[0], ndof + np.arange(nh * nb)]
    rT, cT, vT = [ident], [red_of[ident]], [np.ones(ident.size)]
    for bi, (pt, r) in enumerate(zip(patches, g.rows)):
        base = ndof + nh * bi
        rT.append(3 * pt); cT.append(np.full(pt.size, red_of[base])); vT.append(np.ones(pt.size))
        rT.append(3 * pt); cT.append(np.full(pt.size, red_of[base + 1])); vT.append(y[pt] - r)
        rT.append(3 * pt); cT.append(np.full(pt.size, red_of[base + 2])); vT.append(z[pt] - g.w / 2)
        rT.append(3 * pt); cT.append(red_of[3 * pt]); vT.append(-np.ones(pt.size))
    T = sp.csr_matrix((np.concatenate(vT), (np.concatenate(rT), np.concatenate(cT))), shape=(nfull, nred))
    _, I_s = bolt_section(g)
    k_rot = k_rot_factor * g.Eb * I_s / g.L_b
    head_k = np.tile([g.k_axial, k_rot, k_rot], nb)
    Kfull = sp.block_diag([K + sp.diags(diag), sp.diags(head_k)]).tocsr()
    Kr = (T.T @ Kfull @ T).tocsr()
    kreg = np.zeros(nred)
    for pt in patches:
        kreg[red_of[3 * pt]] = reg * g.k_axial / pt.size
    Kr = (Kr + sp.diags(kreg)).tocsr()
    dof_of = lambda d: red_of[d]  # noqa: E731
    # rigid end section of the beam segment
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
    f[mc1] = -0.5 * P_full
    f[mth] = -0.5 * P_full * (g.L_load - beam_segment)
    fixed = np.zeros(nr, bool)
    fixed[dof_of(3 * np.nonzero(np.isclose(z, 0))[0] + 2)] = True
    contact_nodes = np.nonzero(np.isclose(x, 0))[0]
    col_dofs = dof_of(3 * contact_nodes)
    free = np.nonzero(~fixed)[0]
    pos = -np.ones(nr, int); pos[free] = np.arange(free.size)
    A = Kr[free][:, free].tocsr(); b = f[free]
    c_col = pos[col_dofs]
    s_slots = np.concatenate([dof_of(3 * pt) for pt in patches])
    cidx = np.r_[c_col, pos[s_slots]].astype(int)
    n_column = c_col.size
    assert (cidx >= 0).all()
    t1 = time.time()
    sol = solve_signorini(A, b, cidx)
    t_sol = time.time() - t1
    ur = np.zeros(nr); ur[free] = sol["x"]
    ur2 = ur
    ur = T2 @ ur2
    ufull = T @ ur
    u = ufull[:ndof]
    U = u.reshape(-1, 3)
    hd = ufull[ndof:].reshape(nb, nh)
    A_web_half = (g.hb - 2 * g.tfb) * g.twb / 2
    Ph = 0.5 * P_full
    sd = g.x_dt1 - beam_segment; Lc = g.L_load - beam_segment
    Gw = g.Ew / (2 * (1 + g.nu))
    v_dt1 = (ur2[mc1] + ur2[mth] * sd - 0.5 * P_full * sd ** 2 * (3 * Lc - sd) / (6 * g.beam_EI_half())
             - 0.5 * P_full * sd / (Gw * A_web_half))
    a1, L1 = g.tp + g.x_dt1, g.tp + g.L_load
    d_eb1 = Ph * a1 ** 2 * (3 * L1 - a1) / (6 * g.beam_EI_half())
    phi_p = float(np.arctan(-v_dt1 / a1) - d_eb1 / a1)
    M_face = P_full * L1
    act_col = sol["active"][:n_column]
    cf_ = contact_nodes
    target = np.array([0.0, g.y_top - g.tfb / 2, g.bp / 2])
    i9 = cf_[np.argmin(np.linalg.norm(nodes[cf_] - target, axis=1))]
    out = {"kind": kind, "mesh": {"h": h, "n_tp": n_tp, "n_tf": n_tf, "n_tw": n_tw, "beam_segment": beam_segment,
                                  "nodes": int(n), "cells": int(conn.shape[0]), "dof_reduced": int(A.shape[0]),
                                  "nnz": int(A.nnz), "contact_nodes": int(n_column),
                                  "patch_nodes": [int(pt.size) for pt in patches]},
           "S_paper_kNm_per_mrad": M_face / phi_p * 1e-9, "phi_paper_rad": phi_p,
           "contact_fraction_column_face": float(act_col.mean()),
           "gap_DT9_mm": float(U[i9, 0]), "bolt_force_N": [float(g.k_axial * a_) for a_ in hd[:, 0]],
           "pdas_iterations": sol["iterations"], "kkt": sol["kkt"],
           "seconds_assembly": round(t_asm, 2), "seconds_solve": round(t_sol, 2)}
    if details:
        return out, {"nodes": nodes, "contact_nodes": contact_nodes, "active_column": np.asarray(act_col, bool),
                     "patches": patches, "n_hex_nodes": int(m["nodes"].shape[0])}
    return out
