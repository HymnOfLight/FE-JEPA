#!/usr/bin/env python3
"""wp9 Stage 0c: the measurement behind S's factor 1/64 (PREREG_W9 Sec. 2),
committed so that it can be re-run. Assembly only: no labels, no training.

  sample   the first `--n` instances of E1's training corpus as the generator
           draws them (configs/e1_2d_base.json: seed 0, SeedSequence(0).spawn,
           child i is instance i; the gmsh backend), rebuilt here
  reading  per instance the battery's largest nodal force component max|F|
           (E1's decode scale, `battery_fscale`) over the sum of the absolute
           values of all its nodal force components sum|F| (S's, `battery_l1`):
           its median and 10th / 90th percentiles, times 64, and its rank
           correlation with the instance's mesh size
  remesh   the first `--geometries` of those geometries meshed again at R's
           mesh sizes (geometry, material and loads unchanged): max|F| and
           sum|F| relative to the geometry's coarsest mesh (median over the
           geometries), beside h / 0.12

    python scripts/w9_scale_factor.py --out records/wp9/scale_factor.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y)[0])


def measure(n: int, geometries: int, seed: int) -> dict:
    from fejepa.fe.generator import build_instance, sample_params
    from fejepa.fe.ood2d import REMESH_H
    from fejepa.models.features import battery_fscale, battery_l1

    t0 = time.time()
    children = np.random.SeedSequence(seed).spawn(n)
    rows, params = [], []
    for i, child in enumerate(children):
        p = sample_params(np.random.default_rng(child))
        a = build_instance(p)
        mx, l1 = battery_fscale(a.F), battery_l1(a.F)
        rows.append({"instance": i, "target_h": p["target_h"], "n_nodes": a.n_nodes,
                     "max_abs_F": mx, "sum_abs_F": l1, "ratio": mx / l1})
        params.append(p)
    r = np.array([x["ratio"] for x in rows])
    h = np.array([x["target_h"] for x in rows])
    q10, q50, q90 = (float(np.percentile(r, q)) for q in (10, 50, 90))
    sample = {"n": n, "median": q50, "p10": q10, "p90": q90,
              "median_x64": 64 * q50, "p10_x64": 64 * q10, "p90_x64": 64 * q90,
              "spearman_ratio_vs_target_h": _spearman(r, h),
              "n_nodes_median": int(np.median([x["n_nodes"] for x in rows]))}
    hs = sorted(REMESH_H, reverse=True)
    per_geo = []
    for g in range(min(geometries, n)):
        vals = {}
        for hh in hs:
            a = build_instance(dict(params[g], target_h=float(hh)))
            vals[hh] = (battery_fscale(a.F), battery_l1(a.F), a.n_nodes)
        per_geo.append(vals)
    remesh = {}
    for hh in hs:
        remesh[repr(hh)] = {
            "h_over_coarsest": hh / hs[0],
            "max_abs_F_ratio_median": statistics.median(v[hh][0] / v[hs[0]][0] for v in per_geo),
            "sum_abs_F_ratio_median": statistics.median(v[hh][1] / v[hs[0]][1] for v in per_geo),
            "max_abs_F_ratio_range": [min(v[hh][0] / v[hs[0]][0] for v in per_geo),
                                      max(v[hh][0] / v[hs[0]][0] for v in per_geo)],
            "sum_abs_F_ratio_max_dev": max(abs(v[hh][1] / v[hs[0]][1] - 1) for v in per_geo),
            "n_nodes_median": int(statistics.median(v[hh][2] for v in per_geo))}
    return {"what": "wp9: max|F| / sum|F| on E1's training family (scripts/w9_scale_factor.py)",
            "corpus_seed": seed, "sample": sample, "remesh_geometries": len(per_geo),
            "remesh": remesh, "instances": rows, "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--geometries", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seed = int(json.loads((ROOT / "configs" / "e1_2d_base.json").read_text())["data"]["seed"])
    res = measure(a.n, a.geometries, seed)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("sample", "remesh")}, indent=1))


if __name__ == "__main__":
    main()
