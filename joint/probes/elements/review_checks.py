"""Three checks raised by the independent review of the element study (exploratory, after gate G-C1.3; not a gate).

clamp    The cantilever of check_elements.py with H20 on its two finest meshes: clamped as there (every component on
         the root face), with nu = 0.3 and with nu = 0, and with nu = 0.3 clamped in the section plane only
         (u_x = 0 on the root face, u_y = 0 on its line y = H/2, u_z = 0 on its line z = B/2). The full clamp also
         restrains the Poisson contraction of the root section, which beam theory leaves free; with nu = 0 there is
         none to restrain. The section-plane clamp still holds the line y = H/2 in y.
corners  FS1 on the 10 mm mesh with one layer: on the strip of the column face behind the compression flange
         (y_bot <= y <= y_bot + t_fb), the corner nodes and the mid nodes in contact, for H8I, H20, H20R, H27 and
         T10. The kinds bring different node sets into the contact face.
family   Members of the C1.3 family (fejoint.family.sample, seed 555) whose washer patches hold 10 nodes on the
         10 mm mesh: S as they are, and with washer_r just below the radius of the 10th node, so that 9 remain.

    python -I probes/elements/review_checks.py OUT.json
"""
import json
import pathlib
import platform
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np                                            # noqa: E402
import scipy                                                  # noqa: E402
from fejoint import family as F                               # noqa: E402
from fejoint import linsolve                                   # noqa: E402
from fejoint.joint import build_mesh                           # noqa: E402
from fejoint.linsolve import SPDSolver                         # noqa: E402
from fejoint.specimens import specimen                         # noqa: E402
import check_elements as CE                                    # noqa: E402
import elements as EL                                          # noqa: E402
from joint_generic import analyse_generic                      # noqa: E402


def clamp_check():
    L, Hh, B, P = 200.0, 20.0, 10.0, 1000.0
    I = B * Hh ** 3 / 12; A = B * Hh
    out = []
    for mesh in ((40, 4, 2), (80, 8, 4)):
        nodes8, hexes, part = CE.box(*mesh, L, Hh, B)
        nodes, conn, cpart = EL.mesh_for("H20", nodes8, hexes, part)
        n = nodes.shape[0]
        ids, vals = CE._face_integrals("H20", nodes, conn, L)
        f = np.zeros(3 * n); f[3 * ids + 1] = -P * vals / vals.sum()
        root = np.nonzero(np.isclose(nodes[:, 0], 0))[0]
        rec = {"kind": "H20", "mesh": list(mesh)}
        for nu, clamp in ((CE.NU, "full"), (CE.NU, "section_plane"), (0.0, "full")):
            K = EL.assemble_kind("H20", nodes, conn, cpart, {0: CE.E, 1: CE.E, 2: CE.E, 3: CE.E}, nu)
            G = CE.E / (2 * (1 + nu))
            w_tim = P * L ** 3 / (3 * CE.E * I) + P * L / (5 / 6 * G * A)
            fixed = np.zeros(3 * n, bool)
            if clamp == "full":
                for c in range(3):
                    fixed[3 * root + c] = True
            else:
                fixed[3 * root] = True
                fixed[3 * root[np.isclose(nodes[root, 1], Hh / 2)] + 1] = True
                fixed[3 * root[np.isclose(nodes[root, 2], B / 2)] + 2] = True
            fr = ~fixed
            u = np.zeros(3 * n); u[fr] = SPDSolver(K[fr][:, fr].tocsr()).solve(f[fr])
            rec[f"tip_over_timoshenko_{clamp}_nu{nu:g}"] = float(abs(f @ u) / P / w_tim)
        out.append(rec)
        print("clamp", rec, flush=True)
    return out


def corner_check():
    g = specimen("FS1")
    out = {}
    for kind in ("H8I", "H20", "H20R", "H27", "T10"):
        r, d = analyse_generic(g, kind, 10.0, 1, 1, 1, details=True)
        cn = d["contact_nodes"]; act = d["active_column"]
        y = d["nodes"][cn, 1]
        band = (y >= g.y_bot - 1e-9) & (y <= g.y_bot + g.tfb + 1e-9)
        corner = cn < d["n_hex_nodes"]
        out[kind] = {"S_paper_kNm_per_mrad": r["S_paper_kNm_per_mrad"],
                     "corner_nodes": int((band & corner).sum()), "corner_in_contact": int((band & corner & act).sum()),
                     "mid_nodes": int((band & ~corner).sum()), "mid_in_contact": int((band & ~corner & act).sum())}
        print("corners", kind, out[kind], flush=True)
    return out


def family_check(n_members=3, n_sample=60, seed=555):
    qs, _ = F.sample(n_sample, seed=seed)
    out = []
    for i, q in enumerate(qs):
        g = F.geometry(q)
        m = build_mesh(replace(g, L_beam=300.0), 10.0, 1, 1, 1, 1.15, 40.0)
        x, y, z = m["nodes"].T
        face = np.nonzero(np.isclose(x, g.tp))[0]
        ds = [np.sort(np.hypot(y[face] - r, z[face] - g.w / 2)) for r in g.rows]
        counts = [int((d <= g.washer_r + 1e-9).sum()) for d in ds]
        if counts != [10] * len(counts):
            continue
        r10 = min(d[9] for d in ds)
        a = analyse_generic(g, "H8I", 10.0, 1, 1, 1)
        b = analyse_generic(replace(g, washer_r=r10 - 1e-3), "H8I", 10.0, 1, 1, 1)
        assert a["mesh"]["patch_nodes"] == counts and b["mesh"]["patch_nodes"] == [9] * len(counts)
        rec = {"member": i, "washer_r": g.washer_r, "radius_10th_node": float(r10),
               "patches": a["mesh"]["patch_nodes"], "S": a["S_paper_kNm_per_mrad"],
               "patches_reduced": b["mesh"]["patch_nodes"], "S_reduced": b["S_paper_kNm_per_mrad"],
               "change": a["S_paper_kNm_per_mrad"] / b["S_paper_kNm_per_mrad"] - 1}
        out.append(rec)
        print("family", rec, flush=True)
        if len(out) == n_members:
            break
    return out


def main(path):
    res = {"clamp": clamp_check(), "corners": corner_check(), "family": family_check(),
           "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                   "solver": "PARDISO" if linsolve._PARDISO is not None else "legacy"}}
    pathlib.Path(path).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
