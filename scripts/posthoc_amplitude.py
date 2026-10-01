#!/usr/bin/env python3
"""wp8 Stage 1.36 post-hoc (item 2): how much of the fine-mesh zero-shot error
is amplitude?

For every seed's state of a run (verified against the run's report), on the
run's in-band validation split and on its fine evaluation set, per instance
and load case: the energy-optimal amplitude c* = F^T u / (u^T K u) (label-
free, see fejepa.analysis.posthoc) and its battery-level form c_b, the energy
norms ||u||_K and ||U*||_K (= sqrt(F^T U*)), the K-cosine between u and U*
(c* = ||U*||_K / ||u||_K x cos), the label-based L2 amplitude, the battery
load scale `fscale` and the mesh size lc, and the displacement error and
relative energy gap before and after scaling the prediction by c*. The
uncorrected per-instance values are checked against the report's own
per-instance arrays (`reproduction`).

`--remesh N` adds the label-free dose-response test: N fresh geometries,
each meshed at the in-band lc 0.0906 / 0.0742 / 0.0579 and the fine lc
0.0374 with identical geometry and loads (gmsh3d corpora only); only the
mesh changes, so c*(lc) isolates the resolution dependence of the amplitude.
No solves are needed (c* uses K, F and u only).

Reported only; no verdict.

    python scripts/posthoc_amplitude.py --report records/wp8/e2/baseline/report_phase2b.json \
        --states-dir runs/phase2/e8_states --remesh 16 --out runs/wp8/posthoc/amp_phase2b.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REMESH_LC = (0.0906, 0.0742, 0.0579, 0.0374)


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _summary(rows: list, load_names: list) -> dict:
    if not rows:
        return {}
    cs = np.array([r["c_star"] for r in rows], dtype=float)          # (n, L)
    out = {}
    for k in ("disp", "disp_c", "egap", "egap_c"):
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(np.mean(v))
        out[k + "_median"] = float(np.median(v))
    fin = cs[np.isfinite(cs)]
    cb = np.array([r["c_battery"] for r in rows], dtype=float)
    out.update({
        "n_instances": len(rows),
        "c_star_median": float(np.median(fin)) if fin.size else float("nan"),
        "c_star_q10_q90": ([float(np.quantile(fin, 0.1)), float(np.quantile(fin, 0.9))]
                           if fin.size else [float("nan")] * 2),
        "frac_c_star_gt_1": float((fin > 1.0).mean()) if fin.size else float("nan"),
        "c_star_median_by_load": {n: float(np.nanmedian(cs[:, j])) for j, n in enumerate(load_names)},
        "c_battery_median": float(np.nanmedian(cb)),
        "cos_K_median": float(np.nanmedian(np.array([r["cos_K"] for r in rows], dtype=float))),
        "c_l2_median": float(np.nanmedian(np.array([r["c_l2"] for r in rows], dtype=float))),
        "disp_reduction": float(1.0 - out["disp_c"] / out["disp"]) if out["disp"] > 0
        else float("nan"),
        "spearman_c_battery_vs_lc": _spearman(cb, [r["lc"] for r in rows]),
        "spearman_c_battery_vs_fscale": _spearman(cb, [r["fscale"] for r in rows])})
    return out


def _predict(model, arch, device):
    import torch

    pack = model.prepare_instance(arch, device)
    with torch.no_grad():
        return model.forward_instance(pack).detach().cpu().numpy()


def measure(model, archs, device: str) -> list:
    from fejepa.analysis.posthoc import (amplitude_factor, apply_amplitude, battery_amplitude,
                                         energy_norms, free_mask, l2_amplitude)
    from fejepa.metrics import displacement_errors, energy_gap_rel
    from fejepa.models.features import battery_fscale

    rows = []
    for a in archs:
        if a.U_star is None:
            raise SystemExit(f"{a.path}: no labels (the evaluation sets are labelled)")
        U = _predict(model, a, device)
        free = free_mask(a)
        c = amplitude_factor(U, a.K, a.F, free)
        Uc = apply_amplitude(U, c)
        un, sn = energy_norms(U, a.K, free), energy_norms(a.U_star, a.K, free)
        fu = np.einsum("ld,ld->l", np.atleast_2d(a.F)[:, free], np.atleast_2d(U)[:, free])
        rows.append({"file": Path(str(a.path)).name, "c_star": c,
                     "c_battery": battery_amplitude(U, a.K, a.F, free),
                     "u_norm_K": un, "ustar_norm_K": sn,
                     "cos_K": fu / np.maximum(un * sn, 1e-300),
                     "c_l2": l2_amplitude(U, a.U_star),
                     "disp": float(displacement_errors(U, a).mean()),
                     "disp_c": float(displacement_errors(Uc, a).mean()),
                     "egap": float(energy_gap_rel(U, a).mean()),
                     "egap_c": float(energy_gap_rel(Uc, a).mean()),
                     "lc": float((a.meta.get("extra") or {}).get("lc", float("nan"))),
                     "fscale": float(battery_fscale(a.F)), "n_nodes": int(a.n_nodes)})
    return rows


def reproduction(rows: list, cell: dict | None, seed_index: int) -> dict:
    """Max relative deviation of the uncorrected per-instance values from the
    report's own per-instance arrays for that seed (None: not recorded)."""
    try:
        per = cell["per_seed_eval"][seed_index]["per_instance"]
    except (TypeError, KeyError, IndexError):
        return {"checked": False}
    out = {"checked": True}
    for k, rk in (("disp", "disp_rel_l2"), ("egap", "energy_gap_rel")):
        ref = np.asarray(per[rk][:len(rows)], dtype=float)
        got = np.asarray([r[k] for r in rows], dtype=float)
        out[f"{k}_max_rel_dev"] = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)))
    return out


def remesh_rows(models: dict, n: int, device: str):
    """Fresh gmsh3d geometries, each meshed at every REMESH_LC with the same
    geometry and loads (the rng state is reset per lc; the geometry and the
    traction scales are drawn before and independently of the mesh); label-free
    readings only. Each mesh is built ONCE and read by every seed's model
    (meshing at the fine lc dominates the cost). Yields, per geometry,
    {seed: row}."""
    from fejepa.analysis.posthoc import amplitude_factor, battery_amplitude, energy_norms, free_mask
    from fejepa.fe.gmsh3d import gmsh3d_instance
    from fejepa.models.features import battery_fscale

    for g in range(n):
        rng = np.random.default_rng(90000 + g)
        state = rng.bit_generator.state
        per = {s: [] for s in models}
        n_holes = None
        for lc in REMESH_LC:
            rng.bit_generator.state = state
            a = gmsh3d_instance(rng, lc=lc, labelled=False)
            free = free_mask(a)
            fs = float(battery_fscale(a.F))
            for s, model in models.items():
                U = _predict(model, a, device)
                per[s].append({"lc": lc, "n_nodes": int(a.n_nodes), "fscale": fs,
                               "c_battery": battery_amplitude(U, a.K, a.F, free),
                               "c_star": amplitude_factor(U, a.K, a.F, free).tolist(),
                               "u_norm_K": energy_norms(U, a.K, free).tolist()})
            n_holes = int(a.meta["extra"]["n_holes"])
            del a
        yield {s: {"geometry": g, "n_holes": n_holes, "by_lc": per[s]} for s in models}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="the run's report (the records copy)")
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--n-inband", type=int, default=256)
    ap.add_argument("--n-fine", type=int, default=256)
    ap.add_argument("--remesh", type=int, default=0,
                    help="fresh geometries meshed at every REMESH_LC (gmsh3d runs only)")
    ap.add_argument("--inband-dir", default=None, help="override the report's data dir")
    ap.add_argument("--fine-dir", default=None, help="override the report's transfer dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from fejepa.analysis.common import resolve_device, sha256_of, write_json
    from fejepa.analysis.posthoc import run_files, run_model, verified_states
    from fejepa.data.archive import LazyArchives
    from fejepa.report import _git_describe

    report = json.loads(Path(a.report).read_text())
    cfg = report["config"]
    dev = resolve_device(a.device)
    states = verified_states(report, a.states_dir)
    sets = {"inband": run_files(report, "val", a.n_inband, a.inband_dir)}
    if a.n_fine > 0 and cfg.get("data_transfer"):
        sets["fine"] = run_files(report, "fine", a.n_fine, a.fine_dir)
    if a.remesh and (cfg.get("data") or {}).get("backend") != "gmsh3d":
        raise SystemExit("--remesh is defined for gmsh3d corpora only")
    e8_ar = ((((report.get("results") or {}).get("e8") or {}).get("metrics") or {})
             .get("cells") or {}).get("ar") or {}
    cells = {"inband": e8_ar[max(e8_ar, key=int)] if e8_ar else None,
             "fine": ((((report["results"].get("p3_transfer") or {}).get("metrics") or {})
                       .get("ar") or {}).get("fine"))}
    seeds = [int(s) for s in report["provenance"]["seeds"]]
    load_names = None
    res = {"what": "post-hoc energy-optimal amplitude (wp8 Stage 1.36 item 2); reported only",
           "git": _git_describe(), "torch": torch.__version__, "report": a.report,
           "report_sha256": sha256_of(a.report),
           "config_sha256": report["provenance"]["config_sha256"],
           "model_kind": cfg["model"].get("kind", "fejepa"), "tf32": bool(cfg.get("tf32", True)),
           "device": dev, "states": {f"s{s}": sha256_of(p) for s, p in states.items()},
           "sets": {k: {"n": len(v), "first": Path(str(v[0])).name, "last": Path(str(v[-1])).name}
                    for k, v in sets.items()},
           "remesh_lc": list(REMESH_LC) if a.remesh else None, "seeds": {}}
    models = {}
    for s, p in states.items():
        model = run_model(report, p, s, dev)
        res["seeds"][f"s{s}"] = {}
        for name, files in sets.items():
            t0 = time.time()
            rows = measure(model, LazyArchives(files), dev)
            if load_names is None:
                from fejepa.data.archive import load_instance
                load_names = list(load_instance(files[0]).meta.get("loads")
                                  or [f"load{j}" for j in range(len(rows[0]["c_star"]))])
            res["seeds"][f"s{s}"][name] = {
                "summary": dict(_summary(rows, load_names), seconds=round(time.time() - t0, 1)),
                "reproduction": reproduction(rows, cells.get(name), seeds.index(s)),
                "per_instance": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in r.items()} for r in rows]}
            summ = res["seeds"][f"s{s}"][name]["summary"]
            print(json.dumps({"seed": s, "set": name, **{k: summ[k] for k in (
                "disp", "disp_c", "egap_median", "egap_c_median", "c_battery_median", "seconds")},
                "reproduction": res["seeds"][f"s{s}"][name]["reproduction"]}), flush=True)
            write_json(a.out, res)                         # incremental: a late crash keeps this
        if a.remesh:
            models[s] = model                     # kept for the shared remesh pass below
        else:
            del model
    if a.remesh:
        t0 = time.time()
        for s in models:
            res["seeds"][f"s{s}"]["remesh"] = {"rows": []}
        for g_rows in remesh_rows(models, a.remesh, dev):
            for s, row in g_rows.items():
                rem = res["seeds"][f"s{s}"]["remesh"]
                rem["rows"].append(row)
                rem["c_battery_median_by_lc"] = {
                    lc: float(np.nanmedian([g["by_lc"][i]["c_battery"] for g in rem["rows"]]))
                    for i, lc in enumerate(REMESH_LC)}
                rem["seconds"] = round(time.time() - t0, 1)
            write_json(a.out, res)                         # incremental, per geometry
        for s in models:
            print(json.dumps({"seed": s, "remesh_c_battery_median_by_lc":
                              res["seeds"][f"s{s}"]["remesh"]["c_battery_median_by_lc"]}),
                  flush=True)
        models.clear()
    for name in sets:
        per = [res["seeds"][k][name]["summary"] for k in res["seeds"]]
        res.setdefault("seed_means", {})[name] = {
            k: float(np.mean([q[k] for q in per]))
            for k in ("disp", "disp_c", "egap", "egap_c", "disp_median", "egap_median",
                      "egap_c_median", "c_star_median", "c_battery_median")}
    write_json(a.out, res)
    print(json.dumps(res["seed_means"]))


if __name__ == "__main__":
    main()
