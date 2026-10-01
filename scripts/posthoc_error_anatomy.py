#!/usr/bin/env python3
"""wp8 post-hoc (wrap-up item 3; Stage 1.36, regions revised at Stage 1.37):
where does a 3D model's error live?

For every seed's state of a run (verified against the run's report) on the
run's in-band validation split (and optionally its fine evaluation set), per
instance and load case, from the exact element decomposition of the error
energy (fejepa.analysis.posthoc), the share of the error energy carried by a
set of elements, beside the same elements' share of the reference solution's
energy (enrichment = error share / reference share) and of the volume:

* mesh-layer regions (one to three element layers; their physical size
  shrinks with the mesh): `cavity` (centroid within 3 mean edge lengths of a
  cavity surface), `load` (elements carrying a loaded node; traction load
  cases only -- a body force loads every node), `support` (elements carrying
  a Dirichlet node);
* physical regions (the same region on every mesh, so in-band and fine
  readings are comparable): `cavity_phys` (within one cavity radius of a
  cavity surface), `load_phys` and `support_phys` (centroid within
  DELTA_PHYS of the loaded / supported face);
* for a bottleneck run: `token_straddle` (elements whose nodes are pooled
  into different tokens -- the hard cell boundaries, exactly) and
  `token_band` (centroid relative margin to its two nearest seeds < 0.1, a
  band relative to the token size);
* the share of the error's L2 norm captured by a quadratic field (a global,
  smooth error reads near 1), beside the same share for the reference
  solution itself, and the ratio of the relative energy-norm error to the
  relative L2 error (roughness).

Each instance is loaded, and its element operator and reference energies
built, once; every seed's model is then evaluated on it (Stage 1.37).

Reported only; no verdict.

    python scripts/posthoc_error_anatomy.py --report records/wp8/e2/e2_m1024/report.json \
        --states-dir runs/e2_m1024/e8_states --n-fine 32 --out runs/wp8/posthoc/anat_e2_m1024.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REGIONS = ("cavity", "cavity_phys", "load", "load_phys", "support", "support_phys",
           "token_straddle", "token_band")
DELTA_PHYS = 0.2
"""Physical band width (domain units) of load_phys / support_phys: about two
element sizes at the coarsest in-band lc (0.0906), five at the fine lc
(0.0374); the boxes are 1.5-3.0 x 0.8-1.5 x 0.6-1.2."""
TOKEN_BAND = 0.1


def mean_edge(nodes: np.ndarray, tets: np.ndarray) -> float:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    p = nodes[tets]
    return float(np.mean([np.linalg.norm(p[:, i] - p[:, j], axis=1).mean() for i, j in pairs]))


def instance_geometry(arch) -> dict:
    """Everything about an instance that does not depend on the model: the
    element operator, the static regions, the reference energies."""
    from fejepa.analysis.posthoc import (cavity_distance, cavity_shell, element_energies,
                                         element_operator, low_order_fraction, plane_band)

    nodes, tets = np.asarray(arch.nodes), np.asarray(arch.elements)
    if nodes.shape[1] != 3 or tets.shape[1] != 4:
        raise SystemExit("error anatomy is defined for 3D P1 tetrahedra")
    op = element_operator(nodes, tets, arch.meta["material"])
    cent = nodes[tets].mean(1)
    holes = (arch.meta.get("extra") or {}).get("holes", []) or []
    sup_node = np.asarray(arch.dirichlet_mask).reshape(-1, 3).any(1)
    static = {"support": sup_node[tets].any(1),
              "support_phys": plane_band(nodes[sup_node], cent, DELTA_PHYS)}
    if holes:
        static["cavity"] = cavity_distance(cent, holes) < 3.0 * mean_edge(nodes, tets)
        static["cavity_phys"] = cavity_shell(cent, holes, 1.0)
    per_load = []
    for j in range(arch.n_loads):
        load_node = (np.abs(np.asarray(arch.F[j])).reshape(-1, 3) > 0).any(1)
        if load_node.mean() < 0.5:                 # traction; a body force loads every node
            per_load.append({"load": load_node[tets].any(1),
                             "load_phys": plane_band(nodes[load_node], cent, DELTA_PHYS)})
        else:
            per_load.append({})
    U_star = np.asarray(arch.U_star, dtype=np.float64)
    return {"nodes": nodes, "tets": tets, "op": op, "static": static, "per_load": per_load,
            "er": element_energies(op, U_star), "U_star": U_star,
            "ref_low_order": [low_order_fraction(nodes, U_star[j].reshape(-1, 3), degree=2)
                              for j in range(arch.n_loads)],
            "n_holes": len(holes), "n_nodes": int(arch.n_nodes),
            "lc": float((arch.meta.get("extra") or {}).get("lc", float("nan")))}


def token_regions(pack, geom) -> dict:
    """The bottleneck's token regions for this instance (empty for FE-JEPA)."""
    from fejepa.analysis.posthoc import centroid_margin, token_straddle

    if "tok_idx" not in pack:
        return {}
    tok = pack["tok_idx"].detach().cpu().numpy()
    seeds = pack["seed_xyz"].detach().cpu().numpy()
    return {"token_straddle": token_straddle(tok, geom["tets"]),
            "token_band": centroid_margin(geom["nodes"], geom["tets"], seeds) < TOKEN_BAND}


def anatomy(geom: dict, U: np.ndarray, tok: dict) -> dict:
    """One instance, one model: region shares (mean over the load cases where
    the region is defined), low-order fractions and roughness."""
    from fejepa.analysis.posthoc import element_energies, low_order_fraction, region_share

    U = np.asarray(U, dtype=np.float64)
    E = U - geom["U_star"]
    ee, er, vol = element_energies(geom["op"], E), geom["er"], geom["op"]["vol"]
    acc: dict = {}
    for j in range(U.shape[0]):
        masks = {**geom["static"], **geom["per_load"][j], **tok}
        for name in REGIONS:
            m = masks.get(name)
            if m is None:
                continue
            for pre, w in (("err", ee[j]), ("ref", er[j]), ("vol", vol)):
                acc.setdefault(f"{pre}_{name}", []).append(region_share(w, m))
        acc.setdefault("low_order", []).append(
            low_order_fraction(geom["nodes"], E[j].reshape(-1, 3), degree=2))
        rel_l2 = np.linalg.norm(E[j]) / (np.linalg.norm(geom["U_star"][j]) + 1e-30)
        rel_en = np.sqrt(ee[j].sum() / max(er[j].sum(), 1e-300))
        acc.setdefault("rough", []).append(float(rel_en / max(rel_l2, 1e-300)))
    res = {k: float(np.mean(v)) for k, v in acc.items()}
    res["ref_low_order"] = float(np.mean(geom["ref_low_order"]))
    res.update(n_holes=geom["n_holes"], n_nodes=geom["n_nodes"], lc=geom["lc"])
    return res


def _summary(rows: list) -> dict:
    s = {}
    for name in REGIONS:
        e = [r[f"err_{name}"] for r in rows if f"err_{name}" in r]
        f = [r[f"ref_{name}"] for r in rows if f"ref_{name}" in r]
        s[f"{name}_n_instances"] = len(e)
        if e:
            e_, f_ = np.asarray(e, float), np.asarray(f, float)
            s[name] = {"err_share_median": float(np.median(e_)),
                       "ref_share_median": float(np.median(f_)),
                       "vol_share_median": float(np.median([r[f"vol_{name}"] for r in rows
                                                            if f"vol_{name}" in r])),
                       "enrichment_median": float(np.median(e_ / np.maximum(f_, 1e-12))),
                       # (err - ref) / (1 - ref): 0 = no concentration beyond the solution's
                       # own, 1 = all error energy there. Unlike the enrichment (capped at
                       # 1 / ref) it compares across meshes on which the region covers
                       # different shares (e.g. token_straddle in-band vs fine).
                       "norm_excess_median": float(np.median(
                           (e_ - f_) / np.maximum(1.0 - f_, 1e-12)))}
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
    from fejepa.analysis.posthoc import run_files, run_model, verified_states
    from fejepa.data.archive import LazyArchives
    from fejepa.report import _git_describe

    report = json.loads(Path(a.report).read_text())
    dev = resolve_device(a.device)
    states = verified_states(report, a.states_dir)
    sets = {"inband": run_files(report, "val", a.n)}
    if a.n_fine > 0:
        sets["fine"] = run_files(report, "fine", a.n_fine)
    kind = report["config"]["model"].get("kind", "fejepa")
    res = {"what": "post-hoc error anatomy (wp8 wrap-up item 3); reported only",
           "git": _git_describe(), "report": a.report, "report_sha256": sha256_of(a.report),
           "model_kind": kind, "device": dev, "delta_phys": DELTA_PHYS,
           "token_band": TOKEN_BAND,
           "states": {f"s{s}": sha256_of(p) for s, p in states.items()},
           "seeds": {f"s{s}": {} for s in states}}
    models = {s: run_model(report, p, s, dev) for s, p in states.items()}
    for name, files in sets.items():
        t0 = time.time()
        rows = {s: [] for s in models}
        for i, arch in enumerate(LazyArchives(files)):
            geom = instance_geometry(arch)
            # one preparation for all seeds: the pack depends on the configuration
            # and the instance, not on the weights; forward passes only read it
            pack = next(iter(models.values())).prepare_instance(arch, dev)
            tok = token_regions(pack, geom)
            for s, model in models.items():
                with torch.no_grad():
                    U = model.forward_instance(pack).detach().cpu().numpy()
                rows[s].append(dict(anatomy(geom, U, tok), file=Path(str(arch.path)).name))
            del geom, pack, tok
            if (i + 1) % 16 == 0 or i + 1 == len(files):     # incremental
                for s in models:
                    res["seeds"][f"s{s}"][name] = {
                        "summary": dict(_summary(rows[s]), seconds=round(time.time() - t0, 1),
                                        complete=(i + 1 == len(files))),
                        "per_instance": rows[s]}
                write_json(a.out, res)
        for s in models:
            print(json.dumps({"seed": s, "set": name,
                              **res["seeds"][f"s{s}"][name]["summary"]}), flush=True)
    write_json(a.out, res)


if __name__ == "__main__":
    main()
