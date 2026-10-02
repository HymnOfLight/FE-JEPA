#!/usr/bin/env python3
"""wp9 Stages 0d and 0e: the cost table of the CMAME paper -- per instance, the wall
time of the surrogate's inference against exact solves of the same system,
on the same instances and the same machine, with the surrogate's accuracy on
those instances. Reported only; no verdict, and nothing in PREREG_W9 reads it.

For a run (its records copy of the report and its SHA-verified AR states) and
each instance set -- the run's in-band validation split (`val`), the run's
fine evaluation set (`fine`, 3D runs) and any OOD-2D family given with
`--family NAME=DIR` (manifest and files verified; a family that fails the
check is skipped and recorded, and the script then exits 3) -- per instance:

* surrogate, on the run's device under the run's TF32 policy, batch of one,
  every load case of the battery in one forward pass:
  - `prep_s`: `prepare_instance` (features, masks, load scale, transfer to
    the device);
  - `fwd_s`: `forward_instance`, run `--repeats` times, each synchronised;
    `fwd_first_s` is the first (a design is evaluated once; it includes any
    shape-dependent kernel selection), `fwd_warm_s` the median of the others;
  - `d2h_s`: the copy of the prediction to the host;
  - `surrogate_cold_s` = prep_s + fwd_first_s + d2h_s (the headline), and
    `surrogate_warm_s` = prep_s + fwd_warm_s + d2h_s;
* accuracy against the stored labels: the evaluation's own per-instance
  metrics (`fejepa.metrics.evaluate_fields`: displacement and von Mises
  relative errors, relative energy gap, peak stress error, critical-region
  recall);
* exact solves of the free system, every load case (`--solvers`):
  - `direct`: `solve_fe_displacement(method="direct")`, the SuperLU call
    that bought every label (`runner._label_one`, used by `fejepa label` and
    the in-run labelling; `fe/ood2d.py` for the OOD-2D sets): the labelling
    cost;
  - `cg`: the same function's unpreconditioned conjugate gradients to a
    relative residual of `--cg-tol`, from zero (exact to that tolerance); a
    load case that has not converged after `--cg-maxiter` iterations falls
    back to the direct solve (recorded as a fallback: such a time is not a
    CG time);
  - `cg_warm`: the same CG started from the surrogate's prediction (the
    polishing use; iterations against `cg` recorded);
  - `cg_match` (Stage 0e): the same CG from zero, stopped on each load case
    at the first step whose relative energy error ||x_k - U*||_K^2 /
    ||U*||_K^2 is at or below the surrogate's own relative energy gap on that
    load case (0 steps when the zero field already is, i.e. a gap of 1 or
    more): the time an exact solver needs to be as accurate as the
    surrogate, in the energy norm. The stop is an oracle (the labels decide
    it; a practical CG has no such rule), so this is a lower bound on any
    CG run to that accuracy. The steps are found first, untimed, from one
    trace per load case shared by both matching kinds (the same CG with a
    callback recording the energy error and the displacement error, stopped
    once the smaller target is met; `<kind>_trace_s` is each kind's share);
    then that many steps are timed without the callback, by the CG of `cg`
    (`<kind>_timed_iters` must equal `<kind>_iters`). CG's displacement error
    at the matched step is recorded (`<kind>_disp_at_match`), so that the
    matching can be read in displacement too. Ratios are to the surrogate's
    time;
  - `cg_match_cstar` (Stage 0e): the same, with the targets of the
    prediction rescaled per load case by the energy-optimal amplitude
    c* = F^T u / u^T K u (one mat-vec with the full K, no label: the
    prediction is zero on the Dirichlet dofs; `cstar_s`, the median of three
    runs, times it, and the ratios are to the surrogate's time plus
    `cstar_s`). Its accuracy is recorded as `energy_gap_rel_cstar` and
    `disp_rel_l2_cstar`;
  a solve that takes under one second is repeated (three runs, the median
  kept); each full solution is compared with the stored labels
  (`<solver>_label_max_rel_dev`; the direct solve reproduces them to round-
  off, which identifies the labels' solver).

Assembly is excluded from both sides: the archives hold K and F, and at
inference the surrogate needs no K. One seed's state is timed (the time does
not depend on the weights); its metrics on the first `--check` validation
instances are compared with the report's own per-instance arrays for that
seed, so the timed model is the evaluated one (0 is expected for these
transformer states). `--warmup` validation instances run first, untimed.
Each solver of a set starts no new solve once it has used `--solve-budget-s`
seconds (the solve in progress completes); the surrogate is timed on every
instance and ratios are formed where both exist. One progress line is
printed per instance; the output is rewritten after every instance.

    python scripts/time_inference_vs_solve.py --report records/wp8/e1/e1_2d_base/report.json \
        --states-dir runs/e1_2d_base/e8_states --family F5=runs/w9/ood2d/F5 \
        --out runs/w9/timing/timing_2d.json
    python scripts/time_inference_vs_solve.py --report records/wp8/e2/baseline/report_phase2b.json \
        --states-dir runs/phase2/e8_states --out runs/w9/timing/timing_3d.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SOLVERS = ("direct", "cg", "cg_warm", "cg_match", "cg_match_cstar")
MATCH = ("cg_match", "cg_match_cstar")
MATCH_TOL = 1e-14
"""Relative residual given to the timed CG of the matching kinds (at most 1e-4
times `--cg-tol`): never reached before the step count found by the trace
(which stops at `--cg-tol` at the latest), so the timed CG runs exactly that
many steps."""
SURROGATE = ("prep_s", "fwd_first_s", "fwd_warm_s", "d2h_s", "surrogate_cold_s",
             "surrogate_warm_s")
ACCURACY = ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err", "crit_recall",
            "energy_gap_rel_cstar", "disp_rel_l2_cstar", "cstar_s")
REPEAT_BELOW_S = 1.0
"""A solve faster than this is run three times and the median kept (one
sample of a millisecond-scale solve is noise)."""


def _sync(dev: str) -> None:
    if str(dev).startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _stats(vals) -> dict | None:
    v = [float(x) for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return None
    return {"median": float(np.median(v)), "mean": float(np.mean(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "min": min(v), "max": max(v), "n": len(v)}


def summarise(rows: list, solvers=SOLVERS) -> dict:
    """Statistics of every timed quantity, the accuracy and the sizes."""
    out = {"n_instances": len(rows)}
    for k in SURROGATE + ACCURACY:
        out[k] = _stats(r.get(k) for r in rows)
    for s in solvers:
        out[f"{s}_n_solved"] = sum(r.get(f"{s}_s") is not None for r in rows)
        for k in (f"{s}_s", f"{s}_over_surrogate_cold", f"{s}_over_surrogate_warm"):
            out[k] = _stats(r.get(k) for r in rows)
        devs = [r.get(f"{s}_label_max_rel_dev") for r in rows
                if r.get(f"{s}_s") is not None and r.get(f"{s}_label_max_rel_dev") is not None]
        out[f"{s}_label_max_rel_dev_max"] = max(devs) if devs else None
        if s in MATCH:
            out[f"{s}_unreached"] = sum(int(r.get(f"{s}_unreached") or 0) for r in rows)
            out[f"{s}_iters_mismatch"] = sum(int(r.get(f"{s}_iters_mismatch") or 0) for r in rows)
            out[f"{s}_disp_at_match"] = _stats(d for r in rows
                                               for d in (r.get(f"{s}_disp_at_match") or []))
        if s != "direct":
            out[f"{s}_iters_per_load"] = _stats(it for r in rows for it in (r.get(f"{s}_iters") or [])
                                                if it is not None)
            out[f"{s}_fallbacks"] = sum(bool(r.get(f"{s}_fallback")) for r in rows)
    if "cg" in solvers and "cg_warm" in solvers:
        out["cg_warm_iteration_saving"] = _stats(r.get("cg_warm_iteration_saving") for r in rows)
    for k in ("n_nodes", "n_free_dof"):
        v = [int(r[k]) for r in rows]
        out[k] = {"median": float(np.median(v)), "min": min(v), "max": max(v)} if v else None
    return out


def surrogate_time(model, a, dev: str, repeats: int) -> tuple:
    """(timings, prediction): see the module docstring."""
    import torch

    with torch.no_grad():
        _sync(dev)
        t0 = time.perf_counter()
        pack = model.prepare_instance(a, dev)
        _sync(dev)
        prep = time.perf_counter() - t0
        fwd = []
        for _ in range(max(1, int(repeats))):
            _sync(dev)
            t0 = time.perf_counter()
            U = model.forward_instance(pack)
            _sync(dev)
            fwd.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    pred = U.detach().cpu().numpy()
    d2h = time.perf_counter() - t0
    warm = statistics.median(fwd[1:]) if len(fwd) > 1 else fwd[0]
    return {"prep_s": prep, "fwd_s": fwd, "fwd_first_s": fwd[0], "fwd_warm_s": warm,
            "d2h_s": d2h, "surrogate_cold_s": prep + fwd[0] + d2h,
            "surrogate_warm_s": prep + warm + d2h}, pred


def solve_time(kind: str, a, pred: np.ndarray, cg_tol: float, cg_maxiter: int) -> dict:
    """One exact solve of every load case by `kind` (see SOLVERS), repeated
    when fast; the solution checked against the stored labels."""
    from fejepa.fe.solve import solve_fe_displacement

    kw = {"direct": {"method": "direct"},
          "cg": {"method": "cg", "tol": cg_tol, "maxiter": cg_maxiter},
          "cg_warm": {"method": "cg", "tol": cg_tol, "maxiter": cg_maxiter, "x0": pred}}[kind]
    runs = []
    while True:
        t0 = time.perf_counter()
        U, infos = solve_fe_displacement(a.K, a.F, a.free_mask, **kw)
        runs.append(time.perf_counter() - t0)
        if runs[0] >= REPEAT_BELOW_S or len(runs) == 3:
            break
    out = {f"{kind}_s": statistics.median(runs), f"{kind}_runs": runs,
           f"{kind}_label_max_rel_dev": None}
    if a.U_star is not None:
        ref = np.atleast_2d(np.asarray(a.U_star, dtype=np.float64))
        out[f"{kind}_label_max_rel_dev"] = float(
            np.max(np.abs(U - ref)) / max(float(np.max(np.abs(ref))), 1e-300))
    if kind != "direct":
        out[f"{kind}_iters"] = [inf.get("cg_iters") for inf in infos]
        out[f"{kind}_fallback"] = any(inf.get("method") == "cg->direct" for inf in infos)
    return out


class _Reached(Exception):
    """Raised by the trace's callback once the target is met (stops CG early)."""


def energy_trace(Kff, b: np.ndarray, ustar: np.ndarray, tol: float, maxiter: int,
                 stop_at: float | None = None) -> tuple:
    """(energy errors, displacement errors, end) after every step k of the
    `cg` solver's CG from zero (scipy CG, relative residual `tol`, atol 0):
    entry k - 1 holds step k's ||x_k - u*||_K^2 / ||u*||_K^2 and
    ||x_k - u*||_2 / ||u*||_2. `end`: "target" (stopped once the energy error
    is at or below `stop_at`), "converged" (CG reached `tol`) or "maxiter".
    Untimed (one extra mat-vec per step)."""
    import scipy.sparse.linalg as spla

    norm2 = float(ustar @ (Kff @ ustar))
    unorm = float(np.linalg.norm(ustar))
    errs, derrs = [], []

    def cb(xk):
        e = xk - ustar
        errs.append(float(e @ (Kff @ e)) / norm2)
        derrs.append(float(np.linalg.norm(e)) / unorm)
        if stop_at is not None and errs[-1] <= stop_at:
            raise _Reached

    kw = dict(x0=None, maxiter=maxiter, callback=cb)
    try:
        try:
            _x, info = spla.cg(Kff, b, rtol=tol, atol=0.0, **kw)
        except TypeError:                                 # scipy < 1.12
            errs.clear()
            derrs.clear()
            _x, info = spla.cg(Kff, b, tol=tol, atol=0.0, **kw)
        end = "converged" if info == 0 else "maxiter"
    except _Reached:
        end = "target"
    return np.asarray(errs, dtype=float), np.asarray(derrs, dtype=float), end


def match_steps(trace, target: float):
    """The first CG step whose relative energy error is at or below `target`:
    0 when the zero field already is (its relative error is 1), None when no
    step of the trace reaches it."""
    if not np.isfinite(target) or target >= 1.0:
        return 0
    hit = np.nonzero(np.asarray(trace) <= target)[0]
    return int(hit[0]) + 1 if hit.size else None


def match_steps_for(a, targets: dict, cg_tol: float, cg_maxiter: int) -> tuple:
    """({kind: {"steps", "end", "disp", "unreached"}}, seconds): for each kind
    of `targets` ({kind: per-load relative gaps}), per load case the first CG
    step at or below the kind's target, from one trace per load case stopped
    once every kind's target is met. `end` per load case: "zero" (0 steps),
    "target", or the trace's end when the target was not reached (then the
    steps are the whole trace: "converged", the full solve, or "maxiter",
    where `cg` falls back to the direct solve)."""
    import scipy.sparse as sp

    free = np.asarray(a.free_mask, dtype=bool)
    F2 = np.atleast_2d(np.asarray(a.F, dtype=np.float64))
    S2 = np.atleast_2d(np.asarray(a.U_star, dtype=np.float64))
    t0 = time.perf_counter()
    Kff = sp.csr_matrix(a.K)[free][:, free]
    out = {k: {"steps": [], "end": [], "disp": [], "unreached": 0} for k in targets}
    for j in range(F2.shape[0]):
        tj = {k: float(np.asarray(v)[j]) for k, v in targets.items()}
        live = [t for t in tj.values() if match_steps([], t) != 0]
        if live:
            errs, derrs, end = energy_trace(Kff, F2[j][free], S2[j][free], cg_tol, cg_maxiter,
                                            stop_at=min(live))
        else:
            errs, derrs, end = np.zeros(0), np.zeros(0), "none"
        for k, t in tj.items():
            m = match_steps(errs, t)
            if m == 0:
                out[k]["end"].append("zero")
            elif m is not None:
                out[k]["end"].append("target")
            else:
                m = len(errs)
                out[k]["unreached"] += 1
                out[k]["end"].append(end)
            out[k]["steps"].append(int(m))
            out[k]["disp"].append(float(derrs[m - 1]) if m > 0 else 1.0)
    return out, time.perf_counter() - t0


def match_time(kind: str, a, steps, cg_tol: float) -> dict:
    """The timed CG of `steps[l]` steps on each load case, from zero, by the
    `cg` solver's CG (free-block extraction included, as in `cg`), repeated
    when fast; the steps each run actually made are recorded."""
    import scipy.sparse as sp

    from fejepa.fe.solve import _cg

    free = np.asarray(a.free_mask, dtype=bool)
    F2 = np.atleast_2d(np.asarray(a.F, dtype=np.float64))
    tol = min(MATCH_TOL, 1e-4 * float(cg_tol))
    runs, made = [], []
    while True:
        t0 = time.perf_counter()
        Kff = sp.csr_matrix(a.K)[free][:, free]
        made = []
        for j, k in enumerate(steps):
            if k > 0:
                made.append(int(_cg(Kff, F2[j][free], x0=None, tol=tol, maxiter=k, count=True)[2]))
            else:
                made.append(0)
        runs.append(time.perf_counter() - t0)
        if runs[0] >= REPEAT_BELOW_S or len(runs) == 3:
            break
    return {f"{kind}_s": statistics.median(runs), f"{kind}_runs": runs,
            f"{kind}_iters": [int(k) for k in steps], f"{kind}_timed_iters": made,
            f"{kind}_iters_mismatch": int(sum(m != k for m, k in zip(made, steps))),
            f"{kind}_label_max_rel_dev": None}


def cstar_time(pred: np.ndarray, a) -> tuple:
    """(seconds, c*, rescaled prediction): c* = F^T u / u^T K u per load case
    with the full K (the prediction is zero on the Dirichlet dofs, so this is
    the free-dof value), timed three times and the median kept. A case with
    u^T K u = 0 keeps its prediction (c* NaN)."""
    import scipy.sparse as sp

    K = sp.csr_matrix(a.K)
    F2 = np.atleast_2d(np.asarray(a.F, dtype=np.float64))
    U = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        KU = (K @ U.T).T
        num = np.einsum("ld,ld->l", F2, U)
        den = np.einsum("ld,ld->l", U, KU)
        c = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        Uc = U * np.where(np.isfinite(c), c, 1.0)[:, None]
        runs.append(time.perf_counter() - t0)
    return statistics.median(runs), c, Uc


def time_instance(model, a, dev: str, repeats: int, solvers, budget_left, cg_tol: float,
                  cg_maxiter: int) -> tuple:
    """(row, prediction) for one instance. `budget_left(kind)` says whether
    `kind` may start a solve; skipped solves are recorded as such."""
    from fejepa.metrics import evaluate_fields

    row = {"file": Path(str(a.path)).name, "n_nodes": int(a.n_nodes),
           "n_free_dof": int(np.count_nonzero(a.free_mask)), "n_loads": int(a.n_loads)}
    timing, pred = surrogate_time(model, a, dev, repeats)
    row.update(timing)
    targets = {}
    if a.U_star is not None:
        from fejepa.metrics import displacement_errors, energy_gap_rel

        row.update(evaluate_fields(pred, a))
        row["cstar_s"], c, pred_c = cstar_time(pred, a)
        row["cstar"] = [float(x) for x in c]
        row["energy_gap_rel_cstar"] = float(energy_gap_rel(pred_c, a).mean())
        row["disp_rel_l2_cstar"] = float(displacement_errors(pred_c, a).mean())
        targets = {"cg_match": energy_gap_rel(pred, a), "cg_match_cstar": energy_gap_rel(pred_c, a)}
    need = [k for k in MATCH if k in solvers and k in targets and budget_left(k)]
    found = {}
    if need:
        found, trace_s = match_steps_for(a, {k: targets[k] for k in need}, cg_tol, cg_maxiter)
        for k in need:
            row[f"{k}_trace_s"] = trace_s / len(need)
            row[f"{k}_targets"] = [float(t) for t in targets[k]]
            row[f"{k}_trace_end"] = found[k]["end"]
            row[f"{k}_disp_at_match"] = found[k]["disp"]
            row[f"{k}_unreached"] = found[k]["unreached"]
    for kind in solvers:
        if kind in MATCH and kind not in targets:
            row[f"{kind}_s"] = None
            row[f"{kind}_skipped"] = "no labels"
            continue
        if not budget_left(kind) or (kind in MATCH and kind not in found):
            row[f"{kind}_s"] = None
            row[f"{kind}_skipped"] = "budget"
            continue
        if kind in MATCH:
            row.update(match_time(kind, a, found[kind]["steps"], cg_tol))
        else:
            row.update(solve_time(kind, a, pred, cg_tol, cg_maxiter))
        extra = row["cstar_s"] if kind == "cg_match_cstar" else 0.0
        row[f"{kind}_over_surrogate_cold"] = row[f"{kind}_s"] / (row["surrogate_cold_s"] + extra)
        row[f"{kind}_over_surrogate_warm"] = row[f"{kind}_s"] / (row["surrogate_warm_s"] + extra)
    if row.get("cg_iters") and row.get("cg_warm_iters") and None not in row["cg_iters"] \
            and None not in row["cg_warm_iters"] and sum(row["cg_iters"]) > 0:
        row["cg_warm_iteration_saving"] = 1.0 - sum(row["cg_warm_iters"]) / sum(row["cg_iters"])
    return row, pred


def check_against_report(report: dict, seed: int, rows: list) -> dict:
    """Max relative deviation of the timed model's per-instance displacement
    error and relative energy gap (rows of the first validation instances, in
    the report's order) from the report's own arrays for `seed`."""
    try:
        cells = report["results"]["e8"]["metrics"]["cells"]["ar"]
        per = cells[max(cells, key=int)]["per_seed_eval"][
            [int(s) for s in report["provenance"]["seeds"]].index(int(seed))]["per_instance"]
    except (KeyError, TypeError, ValueError, IndexError):
        return {"checked": False, "why": "the report records no per-instance arrays"}
    rows = [r for r in rows if r.get("disp_rel_l2") is not None]
    if not rows:
        return {"checked": False, "why": "no labelled validation instance was checked"}
    out = {"checked": True, "n": len(rows)}
    for rk in ("disp_rel_l2", "energy_gap_rel"):
        got = np.array([r[rk] for r in rows], dtype=float)
        ref = np.asarray(per[rk][:len(rows)], dtype=float)
        out[f"{rk}_max_rel_dev"] = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)))
    return out


def family_files(spec: str, n: int) -> tuple:
    """NAME=DIR: an evaluation family whose manifest names NAME and whose
    files match it; (name, files[:n], record). SystemExit if not."""
    from fejepa.data.archive import instance_files, load_manifest, manifest_sha256
    from fejepa.fe.ood2d import verify_manifest_files

    if "=" not in spec:
        raise SystemExit(f"--family {spec!r}: write it as NAME=DIR")
    name, d = spec.split("=", 1)
    d = Path(d)
    try:
        m = load_manifest(d)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{d}: no readable manifest ({type(exc).__name__})") from exc
    if m.get("family") != name:
        raise SystemExit(f"{d}: holds family {m.get('family')!r}, not {name!r}")
    bad = verify_manifest_files(d)
    if bad:
        raise SystemExit(f"{d}: {len(bad)} files differ from their manifest (first {bad[0]})")
    return name, instance_files(d)[:n], {"dir": str(d), "manifest_sha256": manifest_sha256(d),
                                         "family": name, "seed": m.get("seed"),
                                         "n_instances": m.get("n_instances")}


def _machine(dev: str) -> dict:
    import scipy
    import torch

    out = {"python": platform.python_version(), "numpy": np.__version__,
           "scipy": scipy.__version__, "torch": torch.__version__,
           "platform": platform.platform(), "cpu_count": os.cpu_count(),
           "torch_threads": torch.get_num_threads(),
           "env_threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                          "OPENBLAS_NUM_THREADS")},
           "device": dev}
    if str(dev).startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        out.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                   gpu_memory_free_at_start=int(free), gpu_memory_total=int(total))
    try:
        from threadpoolctl import threadpool_info

        out["threadpools"] = [{k: p.get(k) for k in ("internal_api", "num_threads", "version")}
                              for p in threadpool_info()]
    except ImportError:
        out["threadpools"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="the run's report (the records copy)")
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--seed", type=int, default=None, help="the state timed (default: the first)")
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--n-fine", type=int, default=8, help="instances of the run's fine set (3D)")
    ap.add_argument("--family", action="append", default=[], help="NAME=DIR, repeatable")
    ap.add_argument("--n-family", type=int, default=32)
    ap.add_argument("--solvers", nargs="+", default=list(SOLVERS), choices=SOLVERS)
    ap.add_argument("--cg-tol", type=float, default=1e-10)
    ap.add_argument("--cg-maxiter", type=int, default=20000,
                    help="per load case; beyond it the direct solve takes over (recorded)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--check", type=int, default=8)
    ap.add_argument("--solve-budget-s", type=float, default=900.0,
                    help="per set and solver: no new solve once this many seconds are used")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    solvers = [s for s in SOLVERS if s in a.solvers]

    import torch

    from fejepa.analysis.common import resolve_device, sha256_of, write_json
    from fejepa.analysis.posthoc import run_files, run_model, run_seeds, verified_states
    from fejepa.data.archive import LazyArchives, load_instance
    from fejepa.report import _git_describe

    report = json.loads(Path(a.report).read_text())
    cfg = report["config"]
    dev = resolve_device(a.device)
    seed = int(run_seeds(report)[0] if a.seed is None else a.seed)
    state = verified_states(report, a.states_dir, seeds=[seed])[seed]
    sets = {"val": (run_files(report, "val", a.n_val), {"from": "the run's validation split"})}
    if a.n_fine > 0 and cfg.get("data_transfer"):
        sets["fine"] = (run_files(report, "fine", a.n_fine), {"from": "the run's fine set"})
    skipped = {}
    for spec in a.family:
        try:
            name, files, rec = family_files(spec, a.n_family)
        except SystemExit as exc:                     # recorded; the other sets still run
            skipped[spec.split("=", 1)[0]] = str(exc)
            print(f"[timing] family {spec} skipped: {exc}", flush=True)
            continue
        if name in sets:
            raise SystemExit(f"--family {name}: the name is taken")
        sets[name] = (files, rec)
    dims = {k: int(load_instance(f[0]).nodes.shape[1]) for k, (f, _m) in sets.items() if f}
    if len(set(dims.values())) > 1:
        raise SystemExit(f"the sets differ in spatial dimension {dims}: a family of another "
                         "run was given")
    model = run_model(report, state, seed, dev)
    res = {"what": "wp9 Stage 0d: surrogate inference against exact solves of the same "
                   "system, per instance, with the surrogate's accuracy (CMAME cost table); "
                   "reported only",
           "git": _git_describe(), "machine": _machine(dev), "report": a.report,
           "report_sha256": sha256_of(a.report),
           "config_sha256": report["provenance"]["config_sha256"],
           "model_kind": (cfg.get("model") or {}).get("kind", "fejepa"),
           "tf32": bool(cfg.get("tf32", True)), "seed": seed, "state": str(state),
           "state_sha256": sha256_of(state),
           "settings": {"repeats": a.repeats, "warmup": a.warmup, "check": a.check,
                        "solvers": solvers, "cg_tol": a.cg_tol, "cg_maxiter": a.cg_maxiter,
                        "solve_budget_s": a.solve_budget_s, "repeat_below_s": REPEAT_BELOW_S,
                        "excluded": "assembly (the archives hold K and F) and archive reading"},
           "sets": {k: dict(meta, n=len(files)) for k, (files, meta) in sets.items()},
           "skipped_families": skipped, "results": {}}
    write_json(a.out, res)
    for arch in LazyArchives(sets["val"][0][:max(0, a.warmup)]):           # untimed warm-up
        surrogate_time(model, arch, dev, 1)
    for name, (files, _meta) in sets.items():
        if not files:
            res["results"][name] = {"skipped": "no instances"}
            write_json(a.out, res)
            continue
        if str(dev).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        rows, used, t_set = [], {s: 0.0 for s in solvers}, time.time()
        for i, arch in enumerate(LazyArchives(files)):
            row, _pred = time_instance(model, arch, dev, a.repeats, solvers,
                                       lambda s: used[s] < a.solve_budget_s, a.cg_tol,
                                       a.cg_maxiter)
            for s in solvers:
                if row.get(f"{s}_s") is not None:
                    used[s] += sum(row[f"{s}_runs"]) + float(row.get(f"{s}_trace_s") or 0.0)
            rows.append(row)
            if name == "val" and len(rows) == min(a.check, len(files)):
                res["check_against_report"] = check_against_report(report, seed, rows)
            res["results"][name] = {
                "summary": dict(summarise(rows, solvers), complete=i + 1 == len(files),
                                seconds=round(time.time() - t_set, 1), solve_seconds_used=used,
                                gpu_peak_bytes=(int(torch.cuda.max_memory_allocated())
                                                if str(dev).startswith("cuda") else None)),
                "per_instance": rows}
            write_json(a.out, res)
            print(f"[timing] {name} {i + 1}/{len(files)} dof {row['n_free_dof']} | surrogate "
                  f"{row['surrogate_cold_s']:.4f} s (warm {row['surrogate_warm_s']:.4f}) | "
                  + " | ".join(f"{s} " + (f"{row[f'{s}_s']:.4f} s" if row.get(f"{s}_s") is not None
                                          else "skipped") for s in solvers)
                  + f" | {time.time() - t_set:.0f} s", flush=True)
        s = res["results"][name]["summary"]
        print(json.dumps({"set": name, "n": s["n_instances"],
                          **{k: (s[k] or {}).get("median") for k in
                             ("surrogate_cold_s", "surrogate_warm_s", *[f"{v}_s" for v in solvers],
                              "disp_rel_l2", "energy_gap_rel")},
                          **{f"{v}_label_max_rel_dev": s[f"{v}_label_max_rel_dev_max"]
                             for v in solvers}}), flush=True)
    if "check_against_report" not in res:
        res["check_against_report"] = {"checked": False, "why": "--check 0" if a.check <= 0
                                       else "no validation instance"}
    write_json(a.out, res)
    print(json.dumps({"check_against_report": res["check_against_report"]}), flush=True)
    if skipped:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
