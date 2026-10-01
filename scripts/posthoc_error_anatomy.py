#!/usr/bin/env python3
"""wp8 Stage 1.36 post-hoc (item 3): where does a 3D model's error live?

For every seed's state of a run (verified against the run's report) on the
run's in-band validation split (and optionally its fine evaluation set), per
instance and load case, from the exact element decomposition of the error
energy (fejepa.analysis.posthoc):

* the share of the error energy within 3 mean edge lengths of a cavity
  surface (instances with cavities only), in elements carrying a loaded node
  (traction load cases only: a body force loads every node), and in elements
  carrying a Dirichlet node -- each beside the same elements' share of the
  reference solution's energy (enrichment = error share / reference share);
* for a bottleneck run, the same for elements near a token-cell boundary
  (mean relative seed margin < 0.1);
* the share of the error's L2 norm captured by a quadratic field (a global,
  smooth error reads near 1), beside the same share for the reference
  solution itself (a pure amplitude error is exactly as smooth as U*), and
  the ratio of the relative energy-norm error to the relative L2 error
  (roughness).

Reported only; no verdict.

    python scripts/posthoc_error_anatomy.py --report records/wp8/e2/e2_m1024/report.json \
        --states-dir runs/e2_m1024/e8_states --out runs/wp8/posthoc/anat_e2_m1024.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REGIONS = ("cavity", "load", "support", "token_boundary")


def mean_edge(nodes: np.ndarray, tets: np.ndarray) -> float:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    p = nodes[tets]
    return float(np.mean([np.linalg.norm(p[:, i] - p[:, j], axis=1).mean() for i, j in pairs]))


def anatomy(arch, U: np.ndarray, margin: np.ndarray | None) -> dict:
    from fejepa.analysis.posthoc import cavity_distance, element_energy, low_order_fraction, region_share

    nodes, tets = np.asarray(arch.nodes), np.asarray(arch.elements)
    if nodes.shape[1] != 3 or tets.shape[1] != 4:
        raise SystemExit("error anatomy is defined for 3D P1 tetrahedra")
    mat = arch.meta["material"]
    h = mean_edge(nodes, tets)
    cent = nodes[tets].mean(1)
    holes = (arch.meta.get("extra") or {}).get("holes", []) or []
    near_cav = cavity_distance(cent, holes) < 3.0 * h
    sup_node = np.asarray(arch.dirichlet_mask).reshape(-1, 3).any(1)
    near_sup = sup_node[tets].any(1)
    near_tok = (margin[tets].mean(1) < 0.1) if margin is not None else None
    from fejepa.fe.tet3d import _tet_geometry
    vol, _ = _tet_geometry(nodes, tets)
    out = {k: [] for k in ("err_cavity", "ref_cavity", "vol_cavity", "err_load", "ref_load",
                           "vol_load", "err_support", "ref_support", "vol_support",
                           "err_token_boundary", "ref_token_boundary", "vol_token_boundary",
                           "low_order", "ref_low_order", "rough")}
    for j in range(arch.n_loads):
        e = U[j] - arch.U_star[j]
        ee = element_energy(nodes, tets, e, mat)
        er = element_energy(nodes, tets, arch.U_star[j], mat)
        load_node = (np.abs(np.asarray(arch.F[j])).reshape(-1, 3) > 0).any(1)
        # a body force (gravity) loads every node: no "near the load" region
        near_load = load_node[tets].any(1) if load_node.mean() < 0.5 else None
        for name, m in (("cavity", near_cav if holes else None), ("load", near_load),
                        ("support", near_sup), ("token_boundary", near_tok)):
            if m is None:
                continue
            out[f"err_{name}"].append(region_share(ee, m))
            out[f"ref_{name}"].append(region_share(er, m))
            out[f"vol_{name}"].append(region_share(vol, m))
        out["low_order"].append(low_order_fraction(nodes, e.reshape(-1, 3), degree=2))
        out["ref_low_order"].append(low_order_fraction(nodes, np.asarray(arch.U_star[j])
                                                       .reshape(-1, 3), degree=2))
        rel_l2 = np.linalg.norm(e) / (np.linalg.norm(arch.U_star[j]) + 1e-30)
        rel_en = np.sqrt(ee.sum() / max(er.sum(), 1e-300))
        out["rough"].append(float(rel_en / max(rel_l2, 1e-300)))
    res = {k: float(np.mean(v)) for k, v in out.items() if v}
    res["n_holes"] = len(holes)
    res["n_nodes"] = int(arch.n_nodes)
    return res


def _summary(rows: list) -> dict:
    s = {}
    for name in REGIONS:
        e = [r[f"err_{name}"] for r in rows if f"err_{name}" in r]
        f = [r[f"ref_{name}"] for r in rows if f"ref_{name}" in r]
        s[f"{name}_n_instances"] = len(e)
        if e:
            s[name] = {"err_share_median": float(np.median(e)),
                       "ref_share_median": float(np.median(f)),
                       "vol_share_median": float(np.median([r[f"vol_{name}"] for r in rows
                                                            if f"vol_{name}" in r])),
                       "enrichment_median": float(np.median(np.asarray(e) / np.maximum(f, 1e-12)))}
    s["low_order_mean"] = float(np.mean([r["low_order"] for r in rows]))
    s["ref_low_order_mean"] = float(np.mean([r["ref_low_order"] for r in rows]))
    s["rough_median"] = float(np.median([r["rough"] for r in rows]))
    s["n_instances"] = len(rows)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--n", type=int, default=128, help="in-band validation instances")
    ap.add_argument("--n-fine", type=int, default=0, help="fine evaluation instances (0 = skip)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from fejepa.analysis.common import resolve_device, sha256_of, write_json
    from fejepa.analysis.posthoc import run_files, run_model, seed_margin, verified_states
    from fejepa.data.archive import LazyArchives
    from fejepa.report import _git_describe

    report = json.loads(Path(a.report).read_text())
    dev = resolve_device(a.device)
    states = verified_states(report, a.states_dir)
    sets = {"inband": run_files(report, "val", a.n)}
    if a.n_fine > 0:
        sets["fine"] = run_files(report, "fine", a.n_fine)
    kind = report["config"]["model"].get("kind", "fejepa")
    res = {"what": "post-hoc error anatomy (wp8 Stage 1.36 item 3); reported only",
           "git": _git_describe(), "report": a.report, "report_sha256": sha256_of(a.report),
           "model_kind": kind, "device": dev,
           "states": {f"s{s}": sha256_of(p) for s, p in states.items()}, "seeds": {}}
    for s, p in states.items():
        model = run_model(report, p, s, dev)
        res["seeds"][f"s{s}"] = {}
        for name, files in sets.items():
            t0, rows = time.time(), []
            for arch in LazyArchives(files):
                pack = model.prepare_instance(arch, dev)
                with torch.no_grad():
                    U = model.forward_instance(pack).detach().cpu().numpy()
                margin = (seed_margin(pack["nbr_rel"].detach().cpu().numpy())
                          if "nbr_rel" in pack else None)
                rows.append(dict(anatomy(arch, U, margin), file=Path(str(arch.path)).name))
            res["seeds"][f"s{s}"][name] = {"summary": dict(_summary(rows),
                                                          seconds=round(time.time() - t0, 1)),
                                           "per_instance": rows}
            print(json.dumps({"seed": s, "set": name,
                              **res["seeds"][f"s{s}"][name]["summary"]}), flush=True)
            write_json(a.out, res)                         # incremental: a late crash keeps this
        del model
    write_json(a.out, res)


if __name__ == "__main__":
    main()
