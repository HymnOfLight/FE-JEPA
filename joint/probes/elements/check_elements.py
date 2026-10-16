"""Checks of the element library: rigid-body modes, constant-strain patch test, tetrahedral conformity, and a
cantilever in bending against Timoshenko beam theory (exploratory, after gate G-C1.3).

    python -I probes/elements/check_elements.py OUT.json
"""
import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fejoint import hex8i as H8MOD                       # noqa: E402
from fejoint.linsolve import SPDSolver                   # noqa: E402
import elements as EL                                    # noqa: E402

KINDS = ["H8", "H8I", "H20", "H20R", "H27", "T4", "T10"]
E, NU = 210000.0, 0.3


def box(nx, ny, nz, L, Hh, B, jitter=0.0, seed=0):
    xs, ys, zs = np.linspace(0, L, nx + 1), np.linspace(0, Hh, ny + 1), np.linspace(0, B, nz + 1)
    nodes, hexes = H8MOD.box_mesh(xs, ys, zs)
    if jitter:
        rng = np.random.default_rng(seed)
        inner = ((nodes[:, 0] > 0) & (nodes[:, 0] < L) & (nodes[:, 1] > 0) & (nodes[:, 1] < Hh)
                 & (nodes[:, 2] > 0) & (nodes[:, 2] < B))
        h = np.array([L / nx, Hh / ny, B / nz])
        nodes = nodes.copy(); nodes[inner] += rng.uniform(-jitter, jitter, (inner.sum(), 3)) * h
    return nodes, hexes, np.zeros(hexes.shape[0], int)


def stiffness(kind, nodes8, hexes, part):
    nodes, conn, cpart = EL.mesh_for(kind, nodes8, hexes, part)
    K = EL.assemble_kind(kind, nodes, conn, cpart, {0: E, 1: E, 2: E, 3: E}, NU)
    return nodes, conn, K


def rigid_and_patch(kind):
    nodes8, hexes, part = box(3, 2, 2, 3.0, 2.0, 2.0, jitter=0.15)
    nodes, conn, K = stiffness(kind, nodes8, hexes, part)
    R = H8MOD.rigid_body_modes(nodes)
    rig = float(np.abs(K @ R).max() / np.abs(K).max())
    G = np.array([[1.0, 0.3, -0.2], [0.1, -0.5, 0.4], [0.2, 0.0, 0.7]]) * 1e-3
    eps_t = 0.5 * (G + G.T)
    u = (nodes @ G.T).ravel()
    eps = np.array([eps_t[0, 0], eps_t[1, 1], eps_t[2, 2], 2 * eps_t[0, 1], 2 * eps_t[1, 2], 2 * eps_t[2, 0]])
    U_exact = 0.5 * 3.0 * 2.0 * 2.0 * eps @ H8MOD.elastic_D(E, NU) @ eps
    U_fe = EL.strain_energy(K, u)
    # interior equilibrium of the linear field: no force on nodes away from the boundary
    f = K @ u
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    inner = np.all((nodes > lo + 1e-9) & (nodes < hi - 1e-9), axis=1)
    fin = float(np.abs(f.reshape(-1, 3)[inner]).max() / np.abs(f).max()) if inner.any() else 0.0
    return {"rigid_residual": rig, "patch_energy_rel_err": abs(U_fe / U_exact - 1), "interior_force_rel": fin}


def cantilever(kind, nx, ny, nz):
    """Clamped at x = 0 (all components), tip shear P over the end face; returns the mean tip deflection."""
    L, Hh, B, P = 200.0, 20.0, 10.0, 1000.0
    nodes8, hexes, part = box(nx, ny, nz, L, Hh, B)
    nodes, conn, K = stiffness(kind, nodes8, hexes, part)
    n = nodes.shape[0]
    fixed = np.zeros(3 * n, bool)
    clamp = np.nonzero(np.isclose(nodes[:, 0], 0))[0]
    for c in range(3):
        fixed[3 * clamp + c] = True
    tip = np.nonzero(np.isclose(nodes[:, 0], L))[0]
    # consistent tip load: P times the face integral of each shape function (exact for any element kind)
    f = np.zeros(3 * n)
    lin = np.zeros(3 * n); lin[1::3] = 1.0           # unit vertical field: work of a uniform traction
    face_load = _face_integrals(kind, nodes, conn, L)
    f[3 * face_load[0] + 1] = -P * face_load[1] / face_load[1].sum()
    free = ~fixed
    u = np.zeros(3 * n)
    u[free] = SPDSolver(K[free][:, free].tocsr()).solve(f[free])
    w_mean = float(-(f @ u) / P)                      # work-conjugate tip deflection
    I = B * Hh ** 3 / 12; A = B * Hh; G = E / (2 * (1 + NU))
    w_eb = P * L ** 3 / (3 * E * I)
    w_tim = w_eb + P * L / (5 / 6 * G * A)
    return {"mesh": [nx, ny, nz], "dof": int(free.sum()), "tip": w_mean, "tip_over_timoshenko": w_mean / w_tim,
            "timoshenko": w_tim}


def _face_integrals(kind, nodes, conn, xface):
    """Nodes on the face x = xface and the integral of their shape functions over it (any element kind)."""
    base, _ = EL.RULES[kind if kind != "H8I" else "H8"]
    ref = EL.REF[base]
    on = np.isclose(nodes[conn][:, :, 0], xface)
    acc = {}
    if base.startswith("H"):
        pts2, w2 = np.polynomial.legendre.leggauss(4)
        for e in np.nonzero(on.sum(axis=1) >= 4)[0]:
            loc = np.nonzero(on[e])[0]
            # the face is xi_a = +1 or -1 for the local axis whose reference coordinate is constant on loc
            ax = [a for a in range(3) if np.allclose(ref[loc[:4], a], ref[loc[0], a]) and abs(ref[loc[0], a]) == 1][0]
            others = [a for a in range(3) if a != ax]
            for i, s in enumerate(pts2):
                for j, t in enumerate(pts2):
                    p = np.zeros(3); p[ax] = ref[loc[0], ax]; p[others[0]] = s; p[others[1]] = t
                    N = EL.shape_values(base, p)[0]
                    dN = EL.shape_grads(base, p)[0]
                    J = nodes[conn[e]].T @ dN
                    dA = np.linalg.norm(np.cross(J[:, others[0]], J[:, others[1]]))
                    for k in loc:
                        acc[conn[e, k]] = acc.get(conn[e, k], 0.0) + w2[i] * w2[j] * N[k] * dA
    else:
        a = 1.0 / 6.0; b = 2.0 / 3.0
        tri = [(a, a), (b, a), (a, b)]                 # degree-2 rule on the unit triangle, weights 1/6
        for e in np.nonzero(on.sum(axis=1) >= 3)[0]:
            loc = np.nonzero(on[e])[0]
            corners = [k for k in loc if k < 4]
            if len(corners) != 3:
                continue
            free_c = [k for k in range(4) if k not in corners][0]
            # parametrise the face by two reference edges from the first corner
            c0, c1, c2 = (ref[k] for k in corners)
            for s, t in tri:
                p = c0 + s * (c1 - c0) + t * (c2 - c0)
                N = EL.shape_values(base, p)[0]
                dN = EL.shape_grads(base, p)[0]
                J = nodes[conn[e]].T @ dN
                dA = np.linalg.norm(np.cross(J @ (c1 - c0), J @ (c2 - c0)))
                for k in loc:
                    acc[conn[e, k]] = acc.get(conn[e, k], 0.0) + (1.0 / 6.0) * N[k] * dA
    ids = np.array(sorted(acc)); vals = np.array([acc[i] for i in ids])
    return ids, vals


def main(out):
    res = {"patch": {}, "cantilever": {}}
    t0 = time.time()
    for k in KINDS:
        res["patch"][k] = rigid_and_patch(k)
    nodes8, hexes, part = box(4, 3, 2, 4.0, 3.0, 2.0)
    tets, _ = EL.to_T4(nodes8, hexes, part)
    res["tet_conformity"] = EL.tet_conformity(tets, hexes)
    for k in KINDS:
        res["cantilever"][k] = [cantilever(k, nx, ny, nz) for nx, ny, nz in ((10, 1, 1), (20, 2, 1), (40, 4, 2), (80, 8, 4))]
    res["seconds"] = round(time.time() - t0, 1)
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    for k in KINDS:
        p = res["patch"][k]
        print(f"{k:5s} rigid {p['rigid_residual']:.1e}  patch {p['patch_energy_rel_err']:.1e}  interior {p['interior_force_rel']:.1e}  "
              + "  ".join(f"{c['mesh']}: {c['tip_over_timoshenko']:.3f}" for c in res["cantilever"][k]))
    print(res["tet_conformity"], res["seconds"], "s")


if __name__ == "__main__":
    main(sys.argv[1])
