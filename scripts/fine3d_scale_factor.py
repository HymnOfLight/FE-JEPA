#!/usr/bin/env python3
"""fine3d (PREREG_FINE3D Sec. 2, R2a): the measurement behind S's
three-dimensional factor, committed so that it can be re-run. Assembly only:
no labels, no training, no network.

  sample   `--n` instances of the three-dimensional training family drawn
           afresh: the gmsh3d sampler of the E2 corpus (configs/e2_m1024.json:
           lc uniform in its lc_range, drawn from the generator immediately
           before each instance, one generator for the whole sample, as
           `generate_gmsh3d_dataset` draws a corpus) from SAMPLE_SEED, a seed
           that no corpus, evaluation set or post-hoc pass has used -- the
           training corpus's own instances (among them R1's holdout) are not
           read
  reading  per instance the battery's largest nodal force component max|F|
           (`battery_fscale`, the bottleneck's decode scale) over the sum of
           the absolute values of all its nodal force components sum|F|
           (`battery_l1`, S's): its median and 10th / 90th percentiles, their
           ratios to the factor, and the rank correlation with the
           instance's lc. The factor is the power of two nearest the median
           on a log scale (`factor_of`), the rule that gives wp9's 1/64 from
           its two-dimensional median 0.0151
  remesh   `--geometries` geometries, each meshed at the in-band lc 0.0906 /
           0.0742 / 0.0579 and the fine lc 0.0374 with geometry and loads
           unchanged (the wp8 post-hoc remesh pass's construction:
           scripts/posthoc_amplitude.py, generator 90000 + g): max|F| and
           sum|F| relative to the geometry's coarsest mesh (median and range
           over the geometries), beside lc / 0.0906

    python scripts/fine3d_scale_factor.py --out records/wp9/fine3d/scale_factor_3d.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_SEED = 93000
"""The sample's generator seed: unused by every corpus and set of the
repository (corpora 0, the multi-resolution pairs 7919, the post-hoc remesh
geometries 90000-90015, OOD-2D 91001-91007)."""
BASE_CONFIG = "configs/e2_m1024.json"
REMESH_LC = (0.0906, 0.0742, 0.0579, 0.0374)
"""The post-hoc remesh pass's mesh sizes (scripts/posthoc_amplitude.py): the
in-band range's ends and middle, and the fine evaluation set's lc."""


def factor_of(median: float) -> float:
    """The power of two nearest `median` on a log scale (wp9: 0.0151 -> 1/64)."""
    return float(2.0 ** round(math.log2(float(median))))


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y)[0])


def sample_rows(n: int, seed: int, lc_range) -> list:
    """`n` fresh training-family instances' load readings (no labels)."""
    from fejepa.fe.gmsh3d import gmsh3d_instance
    from fejepa.models.features import battery_fscale, battery_l1

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lc = float(rng.uniform(*lc_range))
        a = gmsh3d_instance(rng, lc=lc, labelled=False)
        mx, l1 = battery_fscale(a.F), battery_l1(a.F)
        rows.append({"instance": i, "lc": lc, "n_nodes": int(a.n_nodes),
                     "n_holes": int(a.meta["extra"]["n_holes"]),
                     "max_abs_F": mx, "sum_abs_F": l1, "ratio": mx / l1})
    return rows


def remesh_rows(geometries: int) -> list:
    """Per geometry {lc: (max|F|, sum|F|, n_nodes)} at REMESH_LC."""
    from fejepa.fe.gmsh3d import gmsh3d_instance
    from fejepa.models.features import battery_fscale, battery_l1

    out = []
    for g in range(geometries):
        vals = {}
        for lc in REMESH_LC:
            a = gmsh3d_instance(np.random.default_rng(90000 + g), lc=lc, labelled=False)
            vals[lc] = (battery_fscale(a.F), battery_l1(a.F), int(a.n_nodes))
        out.append(vals)
    return out


def summarise(rows: list, per_geo: list) -> dict:
    r = np.array([x["ratio"] for x in rows])
    q10, q50, q90 = (float(np.percentile(r, q)) for q in (10, 50, 90))
    fac = factor_of(q50)
    sample = {"n": len(rows), "median": q50, "p10": q10, "p90": q90, "factor": fac,
              "log2_factor": round(math.log2(fac)),
              "median_over_factor": q50 / fac, "p10_over_factor": q10 / fac,
              "p90_over_factor": q90 / fac,
              "spearman_ratio_vs_lc": _spearman(r, [x["lc"] for x in rows]),
              "n_nodes_median": int(np.median([x["n_nodes"] for x in rows]))}
    remesh = {}
    base = REMESH_LC[0]
    for lc in REMESH_LC:
        mx = [v[lc][0] / v[base][0] for v in per_geo]
        l1 = [v[lc][1] / v[base][1] for v in per_geo]
        remesh[repr(lc)] = {
            "lc_over_coarsest": lc / base,
            "max_abs_F_ratio_median": statistics.median(mx),
            "max_abs_F_ratio_range": [min(mx), max(mx)],
            "sum_abs_F_ratio_median": statistics.median(l1),
            "sum_abs_F_ratio_max_dev": max(abs(x - 1) for x in l1),
            "n_nodes_median": int(statistics.median(v[lc][2] for v in per_geo))}
    return {"sample": sample, "remesh_geometries": len(per_geo), "remesh": remesh}


def measure(n: int, geometries: int, seed: int = SAMPLE_SEED) -> dict:
    t0 = time.time()
    cfg = json.loads((ROOT / BASE_CONFIG).read_text())
    lc_range = [float(x) for x in cfg["data"]["lc_range"]]
    rows = sample_rows(n, seed, lc_range)
    per_geo = remesh_rows(geometries)
    import gmsh

    res = {"what": "fine3d: max|F| / sum|F| on the 3D training family (scripts/"
                   "fine3d_scale_factor.py); assembly only",
           "sample_seed": seed, "lc_range": lc_range, "base_config": BASE_CONFIG,
           "gmsh_version": getattr(gmsh, "__version__", None),
           **summarise(rows, per_geo),
           "remesh_lc": list(REMESH_LC),
           "remesh_per_geometry": [{repr(lc): list(v[lc]) for lc in REMESH_LC}
                                   for v in per_geo],
           "instances": rows, "seconds": round(time.time() - t0, 1)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--geometries", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = measure(a.n, a.geometries)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("sample", "remesh", "seconds")}, indent=1))


if __name__ == "__main__":
    main()
