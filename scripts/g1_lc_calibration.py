"""WP7 3D-G1.1: lc calibration for the production corpus bands.

The envelope memo (21 August 2026) fixed the corpus in dof terms -- training
2,000 @ 10k-30k dof, transfer 256 @ ~100k dof; this script converts those
targets into gmsh characteristic lengths for the box-with-cavities family, on
the machine that will actually generate the corpus (gmsh sizing is
version-dependent, so the stamped numbers should come from a box run of this
script; anywhere else it calibrates the design).

Method: the SAME sampled geometries (seeded children) are meshed at every lc
in the sweep, giving a paired log-log fit dof ~ A * lc^b (b ~ -3 expected);
the fit is inverted at each dof target and reported with the per-geometry
spread. Rows are written incrementally (3D-B1.1 lesson) and an optional
--probe pass meshes the extrapolated transfer lc directly to validate it.

Usage:
  PYTHONPATH=src python3 scripts/g1_lc_calibration.py \
      --lcs 0.32 0.24 0.18 0.14 0.11 --samples 6 \
      --targets 10000 30000 100000 --probe --out runs/g1_cal/lc_calibration.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fejepa.fe.gmsh3d import mesh_box_with_cavities, sample_params3d


def _provenance() -> dict:
    """Best-effort provenance for measurement JSONs (git describe, versions)."""
    import subprocess
    prov = {}
    try:
        prov["git"] = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10).stdout.strip() or "unavailable"
    except Exception:                                     # noqa: BLE001
        prov["git"] = "unavailable"
    prov["numpy"] = np.__version__
    try:
        import gmsh
        prov["gmsh"] = ".".join(str(v) for v in gmsh.GMSH_API_VERSION.split("."))             if isinstance(getattr(gmsh, "GMSH_API_VERSION", None), str) else             str(getattr(gmsh, "GMSH_API_VERSION", "unknown"))
    except Exception:                                     # noqa: BLE001
        pass
    return prov


def _measure(params: dict, lc: float) -> dict:
    t0 = time.perf_counter()
    nodes, tets = mesh_box_with_cavities(params, lc)
    return {"lc": lc, "n_nodes": int(nodes.shape[0]), "dof": 3 * int(nodes.shape[0]),
            "n_tets": int(tets.shape[0]), "mesh_s": round(time.perf_counter() - t0, 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lcs", type=float, nargs="+",
                    default=[0.32, 0.24, 0.18, 0.14, 0.11])
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--targets", type=int, nargs="+",
                    default=[10_000, 30_000, 100_000])
    ap.add_argument("--probe", action="store_true",
                    help="mesh the extrapolated largest-target lc directly")
    ap.add_argument("--out", default="runs/g1_cal/lc_calibration.json")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    children = np.random.SeedSequence(a.seed).spawn(a.samples)
    geoms = [sample_params3d(np.random.default_rng(c)) for c in children]
    payload: dict = {"purpose": "WP7 3D-G1.1 lc calibration (envelope targets)",
                     "seed": a.seed, "samples": a.samples,
                     "provenance": _provenance(), "rows": []}

    def flush():
        out.write_text(json.dumps(payload, indent=1))

    for lc in sorted(a.lcs, reverse=True):
        for gi, p in enumerate(geoms):
            r = {"geom": gi, **_measure(p, lc)}
            payload["rows"].append(r)
            flush()
        ds = [r["dof"] for r in payload["rows"] if r["lc"] == lc]
        print(f"[g1-cal] lc={lc:<5} dof min/med/max = "
              f"{min(ds)}/{int(np.median(ds))}/{max(ds)}", flush=True)

    def curve(rows):
        """(log lc, log median dof) samples, sorted by lc; local slope varies
        (surface-dominated ~lc^-2 coarse, volume ~lc^-3 fine), so recommend by
        piecewise log-log interpolation, never a single global power law."""
        by = {}
        for r in rows:
            by.setdefault(r["lc"], []).append(r["dof"])
        lcs = np.array(sorted(by))
        med = np.array([np.median(by[l]) for l in lcs], dtype=float)
        return lcs, med

    def recommend(rows, targets):
        lcs, med = curve(rows)
        # interpolate log(lc) as a function of log(dof); med decreases with lc
        lg_l, lg_d = np.log(lcs)[::-1], np.log(med)[::-1]      # dof ascending
        rec = {}
        for t in targets:
            lt = np.log(t)
            if lg_d[0] <= lt <= lg_d[-1]:
                lc_t, mode = float(np.exp(np.interp(lt, lg_d, lg_l))), "interpolated"
            elif lt > lg_d[-1]:                    # finer than the swept range:
                sl = (lg_l[-1] - lg_l[-2]) / (lg_d[-1] - lg_d[-2])
                lc_t, mode = float(np.exp(lg_l[-1] + sl * (lt - lg_d[-1]))), "extrapolated"
            else:                                  # coarser than the swept range
                sl = (lg_l[1] - lg_l[0]) / (lg_d[1] - lg_d[0])
                lc_t, mode = float(np.exp(lg_l[0] + sl * (lt - lg_d[0]))), "extrapolated"
            rec[str(t)] = {"lc": round(lc_t, 4), "mode": mode}
        return rec

    dof_all = np.array([r["dof"] for r in payload["rows"]], dtype=float)
    lcs_all = np.array([r["lc"] for r in payload["rows"]])
    spread = float(np.exp((np.log(dof_all)
                           - np.interp(np.log(lcs_all),
                                       *map(np.log, curve(payload["rows"])))).std()))
    payload["geom_spread_x"] = round(spread, 3)
    rec = recommend(payload["rows"], a.targets)

    if a.probe:
        # refine any extrapolated target by meshing at its predicted lc,
        # folding the probe into the curve, and recommending again
        for _ in range(2):
            worst = [t for t, v in rec.items() if v["mode"] == "extrapolated"]
            if not worst:
                break
            probed = {round(r["lc"], 4) for r in payload["rows"]}
            for t in worst:
                lc_p = rec[t]["lc"]
                if round(lc_p, 4) in probed:
                    continue                       # already on the curve
                for gi, p in enumerate(geoms[: min(3, len(geoms))]):
                    r = {"geom": gi, "probe_for": int(t), **_measure(p, lc_p)}
                    payload["rows"].append(r)
                    print(f"[g1-cal] probe lc={lc_p}: geom {gi} dof={r['dof']} "
                          f"mesh {r['mesh_s']}s", flush=True)
                flush()
            rec = recommend(payload["rows"], a.targets)

    payload["recommended_lc"] = rec
    for t in a.targets:
        v = rec[str(t)]
        print(f"[g1-cal] target {t:>7} dof -> lc = {v['lc']:.4f} "
              f"({v['mode']}; per-geometry spread x{spread:.2f})")
    flush()
    print(f"[g1-cal] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
