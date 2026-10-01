#!/usr/bin/env python3
"""wp8 post-hoc (wrap-up item 2; Stage 1.36, restructured at Stage 1.37): how
much of the fine-mesh zero-shot error is amplitude?

For every seed's state of a run (verified against the run's report), on the
run's in-band validation split and on its fine evaluation set, per instance
and load case: the energy-optimal amplitude c* = F^T u / (u^T K u) (label-
free, see fejepa.analysis.posthoc) and its battery-level form c_b, the energy
norms ||u||_K and ||U*||_K, the K-cosine between u and U* (c* = ||U*||_K /
||u||_K x cos, so c* mixes amplitude and shape; the pure amplitude ratio
||U*||_K / ||u||_K is summarised as `amp_ratio_median`), the label-based L2
amplitude, the battery load scale `fscale` and the mesh size lc, and the
displacement error and relative energy gap before and after scaling the
prediction by c*. The uncorrected per-instance values are checked against the
report's own per-instance arrays (`reproduction`; expected: 0 for FE-JEPA
states, up to a few 1e-3 for the bottleneck, whose CUDA scatter-mean
accumulates atomically, in a nondeterministic order).

Each instance is loaded, prepared and its free stiffness block built once;
every seed's model is then evaluated on it.

`--remesh N` adds the label-free dose-response test: N fresh geometries,
each meshed at the in-band lc 0.0906 / 0.0742 / 0.0579 and the fine lc
0.0374 with identical geometry and loads (gmsh3d corpora only); only the
mesh changes. Per lc: c_b, the predicted energy norm relative to the
coarsest mesh (`u_norm_K_ratio`, label-free amplitude) and fscale. Meshes
are cached as archives (`--remesh-cache`, default next to --out) so the
three runs of the runbook mesh once; each row records the mesh's SHA-256.
No solves are needed (c* uses K, F and u only).

Reported only; no verdict.

    python scripts/posthoc_amplitude.py --report records/wp8/e2/baseline/report_phase2b.json \
        --states-dir runs/phase2/e8_states --remesh 16 --out runs/wp8/posthoc/amp_phase2b.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

REMESH_LC = (0.0906, 0.0742, 0.0579, 0.0374)


def _spearman(x, y) -> float:
    """Spearman rank correlation with average ranks for ties; NaN when either
    variable has fewer than two distinct finite values (e.g. lc on the fine
    set, which is one mesh size)."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])


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
    un = np.array([r["u_norm_K"] for r in rows], dtype=float)
    sn = np.array([r["ustar_norm_K"] for r in rows], dtype=float)
    amp = sn / np.where(un > 0, un, np.nan)
    out.update({
        "n_instances": len(rows),
        "c_star_median": float(np.median(fin)) if fin.size else float("nan"),
        "c_star_q10_q90": ([float(np.quantile(fin, 0.1)), float(np.quantile(fin, 0.9))]
                           if fin.size else [float("nan")] * 2),
        "frac_c_star_gt_1": float((fin > 1.0).mean()) if fin.size else float("nan"),
        "c_star_median_by_load": {n: float(np.nanmedian(cs[:, j])) for j, n in enumerate(load_names)},
        "c_battery_median": float(np.nanmedian(cb)),
        "amp_ratio_median": float(np.nanmedian(amp)),
        "cos_K_median": float(np.nanmedian(np.array([r["cos_K"] for r in rows], dtype=float))),
        "c_l2_median": float(np.nanmedian(np.array([r["c_l2"] for r in rows], dtype=float))),
        "disp_reduction": float(1.0 - out["disp_c"] / out["disp"]) if out["disp"] > 0
        else float("nan"),
        "spearman_c_battery_vs_lc": _spearman(cb, [r["lc"] for r in rows]),
        "spearman_c_battery_vs_fscale": _spearman(cb, [r["fscale"] for r in rows])})
    return out


def _shared_pack(models: dict, arch, device):
    """One instance preparation for all seeds: the pack (features, masks, load
    scale; for the bottleneck the token seeds and blend weights) depends on
    the configuration and the instance, not on the weights, and the forward
    pass only reads it (Stage 1.37; the reproduction check covers every seed)."""
    return next(iter(models.values())).prepare_instance(arch, device)


def _forward(model, pack):
    import torch

    with torch.no_grad():
        return model.forward_instance(pack).detach().cpu().numpy()


def _jsonable(row: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()}


def measure_instance(models: dict, a, device: str) -> dict:
    """{seed: row} for one labelled instance; the free stiffness block and the
    reference norms are built once and shared by the seeds."""
    from fejepa.analysis.posthoc import (amplitude_factor, apply_amplitude, battery_amplitude,
                                         energy_norms, free_block, free_mask, l2_amplitude)
    from fejepa.metrics import displacement_errors, energy_gap_rel
    from fejepa.models.features import battery_fscale

    if a.U_star is None:
        raise SystemExit(f"{a.path}: no labels (the evaluation sets are labelled)")
    free = free_mask(a)
    Kf = free_block(a.K, free)
    sn = energy_norms(a.U_star, a.K, free, Kf=Kf)
    F2 = np.atleast_2d(a.F)[:, free]
    common = {"file": Path(str(a.path)).name,
              "lc": float((a.meta.get("extra") or {}).get("lc", float("nan"))),
              "fscale": float(battery_fscale(a.F)), "n_nodes": int(a.n_nodes)}
    out = {}
    pack = _shared_pack(models, a, device)
    for s, model in models.items():
        U = _forward(model, pack)
        c = amplitude_factor(U, a.K, a.F, free, Kf=Kf)
        Uc = apply_amplitude(U, c)
        un = energy_norms(U, a.K, free, Kf=Kf)
        fu = np.einsum("ld,ld->l", F2, np.atleast_2d(U)[:, free])
        out[s] = _jsonable({**common, "c_star": c,
                            "c_battery": battery_amplitude(U, a.K, a.F, free, Kf=Kf),
                            "u_norm_K": un, "ustar_norm_K": sn,
                            "cos_K": fu / np.maximum(un * sn, 1e-300),
                            "c_l2": l2_amplitude(U, a.U_star),
                            "disp": float(displacement_errors(U, a).mean()),
                            "disp_c": float(displacement_errors(Uc, a).mean()),
                            "egap": float(energy_gap_rel(U, a).mean()),
                            "egap_c": float(energy_gap_rel(Uc, a).mean())})
    return out


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


def _mesh_sha(a) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(a.nodes, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(a.elements, dtype=np.int64).tobytes())
    return h.hexdigest()


def remesh_instance(g: int, lc: float, cache: Path | None):
    """Geometry g (rng 90000 + g) meshed at lc: from the cache when present,
    otherwise generated (and cached). The geometry and the traction scales come
    from the same rng state at every lc and do not depend on the mesh (the
    scales are drawn after meshing, but from the rng alone), so only the mesh
    changes between lc."""
    from fejepa.data.archive import load_instance, save_instance
    from fejepa.fe.gmsh3d import gmsh3d_instance

    path = cache / f"g{g:03d}_lc{lc:.4f}.npz" if cache is not None else None
    if path is not None and path.exists():
        try:
            return load_instance(path), True
        except Exception:                                  # noqa: BLE001 -- regenerate
            path.unlink(missing_ok=True)
    a = gmsh3d_instance(np.random.default_rng(90000 + g), lc=lc, labelled=False)
    if path is not None:
        save_instance(a, path)                             # atomic (temp + replace)
    return a, False


def remesh_rows(models: dict, n: int, device: str, cache: Path | None = None):
    """Yields, per geometry, {seed: row} (or {"error": ...} for a geometry that
    could not be meshed or evaluated -- the pass continues with the next)."""
    from fejepa.analysis.posthoc import amplitude_factor, battery_amplitude, energy_norms, free_block, free_mask
    from fejepa.models.features import battery_fscale

    for g in range(n):
        try:
            per = {s: [] for s in models}
            n_holes = None
            for lc in REMESH_LC:
                a, cached = remesh_instance(g, lc, cache)
                free = free_mask(a)
                Kf = free_block(a.K, free)
                fs, sha = float(battery_fscale(a.F)), _mesh_sha(a)
                pack = _shared_pack(models, a, device)
                for s, model in models.items():
                    U = _forward(model, pack)
                    per[s].append({"lc": lc, "n_nodes": int(a.n_nodes), "fscale": fs,
                                   "mesh_sha256": sha, "from_cache": cached,
                                   "c_battery": battery_amplitude(U, a.K, a.F, free, Kf=Kf),
                                   "c_star": amplitude_factor(U, a.K, a.F, free, Kf=Kf).tolist(),
                                   "u_norm_K": energy_norms(U, a.K, free, Kf=Kf).tolist()})
                n_holes = int(a.meta["extra"]["n_holes"])
                del a, Kf, pack
            yield {s: {"geometry": g, "n_holes": n_holes, "by_lc": per[s]} for s in models}
        except Exception as exc:                           # noqa: BLE001 -- keep going
            print(json.dumps({"remesh_geometry": g, "error": f"{type(exc).__name__}: {exc}"}),
                  flush=True)
            yield {s: {"geometry": g, "error": f"{type(exc).__name__}: {exc}"} for s in models}


def remesh_summary(rows: list) -> dict:
    ok = [r for r in rows if "by_lc" in r]
    out = {"n_geometries": len(ok), "n_failed": len(rows) - len(ok)}
    if not ok:
        return out
    cb = {lc: [] for lc in REMESH_LC}
    ur = {lc: [] for lc in REMESH_LC}
    fr = {lc: [] for lc in REMESH_LC}
    for r in ok:
        base = r["by_lc"][0]
        for i, lc in enumerate(REMESH_LC):
            b = r["by_lc"][i]
            cb[lc].append(b["c_battery"])
            u0, u1 = np.asarray(base["u_norm_K"], float), np.asarray(b["u_norm_K"], float)
            ur[lc] += list(u1 / np.where(u0 > 0, u0, np.nan))
            fr[lc].append(b["fscale"] / base["fscale"] if base["fscale"] > 0 else float("nan"))
    out.update({"c_battery_median_by_lc": {lc: float(np.nanmedian(v)) for lc, v in cb.items()},
                "u_norm_K_ratio_median_by_lc": {lc: float(np.nanmedian(v)) for lc, v in ur.items()},
                "fscale_ratio_median_by_lc": {lc: float(np.nanmedian(v)) for lc, v in fr.items()}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="the run's report (the records copy)")
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--n-inband", type=int, default=256)
    ap.add_argument("--n-fine", type=int, default=256)
    ap.add_argument("--remesh", type=int, default=0,
                    help="fresh geometries meshed at every REMESH_LC (gmsh3d runs only)")
    ap.add_argument("--remesh-cache", default=None,
                    help="directory of cached remesh archives (default: remesh_cache next to --out)")
    ap.add_argument("--inband-dir", default=None, help="override the report's data dir")
    ap.add_argument("--fine-dir", default=None, help="override the report's transfer dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from fejepa.analysis.common import resolve_device, sha256_of, write_json
    from fejepa.analysis.posthoc import run_files, run_model, verified_states
    from fejepa.data.archive import LazyArchives, load_instance
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
    first = load_instance(sets["inband"][0])
    load_names = list(first.meta.get("loads") or [f"load{j}" for j in range(first.n_loads)])
    del first
    res = {"what": "post-hoc energy-optimal amplitude (wp8 wrap-up item 2); reported only",
           "git": _git_describe(), "torch": torch.__version__, "report": a.report,
           "report_sha256": sha256_of(a.report),
           "config_sha256": report["provenance"]["config_sha256"],
           "model_kind": cfg["model"].get("kind", "fejepa"), "tf32": bool(cfg.get("tf32", True)),
           "device": dev, "states": {f"s{s}": sha256_of(p) for s, p in states.items()},
           "sets": {k: {"n": len(v), "first": Path(str(v[0])).name, "last": Path(str(v[-1])).name}
                    for k, v in sets.items()},
           "reproduction_expected": "0 for FE-JEPA states; up to a few 1e-3 for the bottleneck "
                                    "(atomic, order-nondeterministic CUDA scatter-mean)",
           "remesh_lc": list(REMESH_LC) if a.remesh else None,
           "seeds": {f"s{s}": {} for s in states}}
    models = {s: run_model(report, p, s, dev) for s, p in states.items()}
    for name, files in sets.items():
        t0 = time.time()
        rows = {s: [] for s in models}
        for i, arch in enumerate(LazyArchives(files)):
            for s, row in measure_instance(models, arch, dev).items():
                rows[s].append(row)
            done = i + 1 == len(files)
            if (i + 1) % 16 == 0 or done:                  # incremental
                for s in models:
                    res["seeds"][f"s{s}"][name] = {
                        "summary": dict(_summary(rows[s], load_names),
                                        seconds=round(time.time() - t0, 1), complete=done),
                        "reproduction": (reproduction(rows[s], cells.get(name), seeds.index(s))
                                         if done else {"checked": False, "partial": True}),
                        "per_instance": rows[s]}
                write_json(a.out, res)
        for s in models:
            summ = res["seeds"][f"s{s}"][name]["summary"]
            print(json.dumps({"seed": s, "set": name, **{k: summ[k] for k in (
                "disp", "disp_c", "egap_median", "egap_c_median", "c_battery_median",
                "amp_ratio_median", "seconds")},
                "reproduction": res["seeds"][f"s{s}"][name]["reproduction"]}), flush=True)
    for name in sets:                                       # before the remesh pass
        per = [res["seeds"][k][name]["summary"] for k in res["seeds"]]
        res.setdefault("seed_means", {})[name] = {
            k: float(np.mean([q[k] for q in per]))
            for k in ("disp", "disp_c", "egap", "egap_c", "disp_median", "egap_median",
                      "egap_c_median", "c_star_median", "c_battery_median", "amp_ratio_median")}
    write_json(a.out, res)
    print(json.dumps(res["seed_means"]), flush=True)
    if a.remesh:
        cache = Path(a.remesh_cache) if a.remesh_cache else Path(a.out).parent / "remesh_cache"
        cache.mkdir(parents=True, exist_ok=True)
        res["remesh_cache"] = str(cache)
        t0 = time.time()
        for s in models:
            res["seeds"][f"s{s}"]["remesh"] = {"rows": []}
        for g_rows in remesh_rows(models, a.remesh, dev, cache):
            for s, row in g_rows.items():
                rem = res["seeds"][f"s{s}"]["remesh"]
                rem["rows"].append(row)
                rem.update(remesh_summary(rem["rows"]), seconds=round(time.time() - t0, 1))
            write_json(a.out, res)                          # incremental, per geometry
        for s in models:
            rem = res["seeds"][f"s{s}"]["remesh"]
            print(json.dumps({"seed": s, "remesh": {k: v for k, v in rem.items()
                                                    if k != "rows"}}), flush=True)
    write_json(a.out, res)


if __name__ == "__main__":
    main()
