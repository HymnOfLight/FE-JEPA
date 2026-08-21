"""WP7 3D-B1: solve benchmark -> the costed compute envelope (RUN_PLAN_2026-08-05
sec 3.4; Manual sec 16.4 "bench before launch").

Sweeps structured tetrahedral boxes over increasing node counts and measures, per
scale: P1 assembly time; direct-path splu factorisation time, LU fill-in (nnz and
its ~16 B/nnz memory proxy) and RSS delta; per-RHS direct solve time; and
unpreconditioned CG@1e-10 iterations + time per RHS (unpreconditioned so the
counts stay interpretable against E7, per fe/solve.py). The corpus scale and
labelling budget for 3D-G1/3D-D1 are OUTPUTS of the envelope memo this feeds,
not inputs.

The envelope itself must be measured on the target box (Manual sec 16.4); a run
of this script anywhere else -- e.g. the development sandbox -- validates the
script, not the envelope.

Usage:
  PYTHONPATH=src python3 scripts/bench3d_scale.py \
      --sizes 6 8 10 12 16 20 24 --max-seconds 600 \
      --out runs/bench3d/bench3d_scale.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fejepa.fe.tet3d import assemble_tet, structured_tet_mesh


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def bench_one(n_side: int, tol: float = 1e-10) -> dict:
    w = h = d = 1.0
    t0 = time.perf_counter()
    nodes, tets = structured_tet_mesh(w, h, d, n_side, n_side, n_side)
    material = {"E": 1.0, "nu": 0.3, "plane": "3d"}
    K = assemble_tet(nodes, tets, material)
    t_asm = time.perf_counter() - t0

    ndof = 3 * nodes.shape[0]
    left = np.nonzero(np.isclose(nodes[:, 0], 0.0))[0]
    dmask = np.zeros(ndof, dtype=bool)
    for c in range(3):
        dmask[3 * left + c] = True
    free = ~dmask
    rng = np.random.default_rng(0)
    f = np.zeros(ndof)
    f[free] = rng.standard_normal(int(free.sum()))
    Kff = sp.csr_matrix(K)[free][:, free].tocsc()
    b = f[free]

    rss0 = _rss_mb()
    t0 = time.perf_counter()
    lu = spla.splu(Kff)
    t_fact = time.perf_counter() - t0
    rss_delta = _rss_mb() - rss0
    fill_nnz = int(lu.L.nnz + lu.U.nnz)
    t0 = time.perf_counter()
    x_dir = lu.solve(b)
    t_dsolve = time.perf_counter() - t0

    it = 0

    def cb(_):
        nonlocal it
        it += 1

    t0 = time.perf_counter()
    try:
        x_cg, info = spla.cg(Kff, b, rtol=tol, atol=0.0, callback=cb)
    except TypeError:                                     # scipy < 1.12
        x_cg, info = spla.cg(Kff, b, tol=tol, atol=0.0, callback=cb)
    t_cg = time.perf_counter() - t0

    res_dir = float(np.linalg.norm(Kff @ x_dir - b) / np.linalg.norm(b))
    res_cg = float(np.linalg.norm(Kff @ x_cg - b) / np.linalg.norm(b))
    return {
        "n_side": n_side, "n_nodes": int(nodes.shape[0]),
        "n_tets": int(tets.shape[0]), "ndof": ndof,
        "ndof_free": int(free.sum()), "K_nnz": int(Kff.nnz),
        "assembly_s": round(t_asm, 4),
        "splu_factor_s": round(t_fact, 4),
        "lu_fill_nnz": fill_nnz,
        "lu_fill_mem_mb_16B": round(fill_nnz * 16 / 1e6, 2),
        "rss_delta_mb": round(rss_delta, 1),
        "direct_solve_per_rhs_s": round(t_dsolve, 5),
        "cg_iters_1e-10": it, "cg_info": int(info),
        "cg_per_rhs_s": round(t_cg, 4),
        "residual_direct": res_dir, "residual_cg": res_cg,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[6, 8, 10, 12, 16, 20])
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--max-seconds", type=float, default=600.0,
                    help="stop the sweep once cumulative wall time exceeds this")
    ap.add_argument("--out", default="runs/bench3d/bench3d_scale.json")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, t_start = [], time.perf_counter()
    payload = {"purpose": "WP7 3D-B1 solve benchmark (envelope input)",
               "note": "envelope valid only when measured on the target box "
                       "(Manual sec 16.4); elsewhere this validates the script",
               "tol_cg": a.tol, "rows": rows}

    def _flush():
        out.write_text(json.dumps(payload, indent=1))

    for n in a.sizes:
        if time.perf_counter() - t_start > a.max_seconds:
            print(f"[bench3d] budget {a.max_seconds}s reached; stopping before "
                  f"n_side={n}", flush=True)
            break
        try:
            r = bench_one(n, tol=a.tol)
        except MemoryError:
            rows.append({"n_side": n, "memory_wall": True,
                         "note": "splu MemoryError -- direct-factor memory wall "
                                 "on this box at this size"})
            print(f"[bench3d] n_side={n}: MemoryError in splu -- memory wall "
                  f"recorded; stopping sweep", flush=True)
            _flush()
            break
        rows.append(r)
        _flush()                       # incremental: completed sizes never lost
        print(f"[bench3d] n_side={n:>3} nodes={r['n_nodes']:>7} "
              f"ndof={r['ndof']:>8} | asm {r['assembly_s']:.3f}s | "
              f"splu {r['splu_factor_s']:.3f}s fill {r['lu_fill_mem_mb_16B']:.1f}MB "
              f"(rss +{r['rss_delta_mb']:.0f}MB) | dir/rhs {r['direct_solve_per_rhs_s']:.4f}s | "
              f"CG {r['cg_iters_1e-10']} its {r['cg_per_rhs_s']:.3f}s/rhs", flush=True)

    _flush()
    print(f"[bench3d] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
