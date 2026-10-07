#!/usr/bin/env python3
"""PREREG_CM2D: the verdicts H1, H2a and H2b, the reading H3 and the secondary
readings, from the box return of RUNBOOK_CMAME Sec. B.

    python scripts/adjudicate_cm2d.py --return <return directory> \
        --out records/cmame/cm2d/verdict.json

The return must hold `report.json`, `status.txt` and `provenance.txt` (and the
run logs `run.log*`), and the report must have the SHA-256 the provenance file
lists; the commit and tree the provenance file records must be `prereg-cm2d`'s
in this checkout. E1's base report is the records copy, refused unless it is
PREREG_E1's stamped e1_2d_base run. PREREG_CM2D.md must be stamped (its
CONFIG_SHA256 line filled and its last line its own SHA-256).

It refuses to adjudicate at all (PREREG_CM2D Sec. 6) when the report is not a
guard-verified run of PREREG_CM2D under its CONFIG_SHA256 line, its
configuration is not the generator's (`scripts/make_cm2d_config.py` applied to
E1's configuration), it recorded another commit than `prereg-cm2d`, another
corpus or other seeds, or a row of the grid is missing or lacks one evaluation
per seed with its seed values and values for every validation instance. It
leaves H1 and H3 NOT EVALUATED, and still adjudicates H2a and H2b, when the
label-free row is not E1's states (by SHA-256, reused through E1's records copy
as a supervised grid, evaluation only) or does not reproduce E1's validation
arrays. The verdict file records every input's SHA-256 and the adjudicating
code's; adjudicating code that differs from its `prereg-cm2d` version is
recorded as a deviation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LABEL = "cm2d_v1"
PREREG_NAME = "PREREG_CM2D.md"
TAG = "prereg-cm2d"
E1_TAG = "prereg-e1"
E1_PREREG = "PREREG_E1.md"
E1_BASE_CONFIG_SHA256 = "4dfdea42e22c0cc45c113f11408723953e29e114ec8f94668b447c2a1740a52d"
METRICS = ("energy_gap_rel", "disp_rel_l2", "vm_rel_l2", "peak_vm_rel_err", "crit_recall")
HIGHER_IS_BETTER = ("crit_recall",)
PRIMARY = "energy_gap_rel"
HYPOTHESES = {"H1": ("ar", "labels", "energy_gap_rel",
                     "the label-free transformer against labels-only (reference)"),
              "H2a": ("labels_knorm", "labels", "energy_gap_rel",
                      "the stiffness-norm transformer against labels-only (reference)"),
              "H2b": ("labels_knorm", "labels", "vm_rel_l2",
                      "the stiffness-norm transformer against labels-only (reference)")}
"""Verdict: (new arm A, reference B, metric, what), PREREG_CM2D Sec. 4."""
NEEDS_REUSE = ("H1",)
REPRO_METRICS = ("energy_gap_rel", "disp_rel_l2")
"""The label-free row must reproduce E1's per-instance values of these (as
PREREG_W9's reuse check); the other three metrics' deviations are reported."""
REPRO_MAX = 1e-4
TRAINED = ("labels", "labels_knorm", "mgn")
NAIVE = ("zero", "scale_aware_poly", "knn_field")
RETURN_FILES = ("report.json", "status.txt", "provenance.txt")
PAIRS = (("ar", "labels"), ("ar", "labels_knorm"), ("labels_knorm", "labels"), ("mgn", "labels"))
ADJUDICATING_FILES = ("scripts/adjudicate_cm2d.py", "scripts/make_cm2d_config.py",
                      "scripts/stamp_prereg_cm2d.py", "src/fejepa/analysis/adjudicate_w9.py",
                      "src/fejepa/analysis/adjudicate.py", "src/fejepa/report.py")
"""The files whose code decides a verdict or a refusal."""


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


def _soft(fn, *args, **kw):
    """A secondary reading: computed, or recorded as not evaluated (it never
    blocks a verdict)."""
    try:
        return fn(*args, **kw)
    except Exception as exc:                              # noqa: BLE001
        return {"not_evaluated": f"{type(exc).__name__}: {exc}"}


def jsonable(x):
    """The verdict as strict JSON: non-finite floats become strings."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else str(float(x))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ---------------------------------------------------------------- cells --

def _at(d: dict, b):
    for k in (b, str(b)):
        if k in d:
            return d[k]
    raise KeyError(b)


def cell(rep: dict, row: str, b) -> dict:
    cells = rep["results"]["e8"]["metrics"]["cells"]
    if row not in cells:
        raise ValueError(f"adjudication refused: the report has no {row} row")
    try:
        return _at(cells[row], b)
    except KeyError:
        raise ValueError(f"adjudication refused: the {row} row has no cell at {b}") from None


def seed_values(c: dict, metric: str) -> list:
    return [float(e[metric]) for e in c["per_seed_eval"]]


def per_instance(c: dict, metric: str) -> list:
    return [np.asarray(e["per_instance"][metric], float) for e in c["per_seed_eval"]]


def grid(cfg: dict) -> dict:
    """The rows, budgets, seeds and sizes the configuration fixes."""
    e8 = cfg["experiments"]["e8"]
    budgets = [int(b) for b in e8["budgets"]]
    mgn = [int(b) for b in e8.get("mgn_budgets", budgets)] if e8.get("include_mgn") else []
    return {"budgets": budgets, "mgn_budgets": mgn, "pool": int(e8["pool_sizes"][0]),
            "seeds": [int(e8.get("seed_offset", 0) or 0) + s for s in range(int(e8["seeds"]))],
            "n_val": int(cfg["split"]["n_val"]), "decision": max(budgets)}


def row_cells(g: dict) -> list:
    """(row, budget) of every cell the grid must hold."""
    out = [("ar", g["pool"])]
    out += [(r, b) for r in ("labels", "labels_knorm", *NAIVE) for b in g["budgets"]]
    out += [("mgn", b) for b in g["mgn_budgets"]]
    return out


def _cell_at(rep: dict, row: str, g: dict, b=None) -> dict:
    return cell(rep, row, g["pool"] if row == "ar" else (g["decision"] if b is None else b))


# ------------------------------------------------------------- refusals --

def check_e1(e1: dict) -> None:
    """Refuse an E1 report that is not PREREG_E1's stamped e1_2d_base run."""
    from fejepa.analysis.adjudicate import require_stamped
    from fejepa.report import config_sha256

    if config_sha256(e1["config"]) != E1_BASE_CONFIG_SHA256:
        raise ValueError("adjudication refused: E1's report is not the stamped e1_2d_base run "
                         "(configuration SHA-256)")
    require_stamped(e1, "E1 report", E1_PREREG)
    if (e1.get("provenance") or {}).get("git") != E1_TAG:
        raise ValueError(f"adjudication refused: E1's report ran on "
                         f"{(e1.get('provenance') or {}).get('git')!r}, not {E1_TAG!r}")


def check_report(rep: dict, e1: dict, expected_cfg: dict, prereg_entries: dict | None,
                 expected_git: str | None) -> dict:
    """Refuse a report that is not PREREG_CM2D's run (ValueError); return the grid."""
    from fejepa.analysis.adjudicate import require_stamped
    from fejepa.analysis.adjudicate_w9 import config_diff

    require_stamped(rep, "CM2D run", PREREG_NAME)
    if prereg_entries is not None and rep["prereg"]["config_sha256"] != prereg_entries.get(LABEL):
        raise ValueError(f"adjudication refused: the run's verified hash is not "
                         f"{PREREG_NAME}'s CONFIG_SHA256[{LABEL}]")
    if rep["config"] != expected_cfg:
        diff = sorted(".".join(map(str, p)) for p in config_diff(rep["config"], expected_cfg))
        raise ValueError(f"adjudication refused: the run's configuration is not the "
                         f"generator's (differs in {diff})")
    prov, e1p = rep["provenance"], e1["provenance"]
    if expected_git is not None and prov.get("git") != expected_git:
        raise ValueError(f"adjudication refused: the run ran on {prov.get('git')!r}, "
                         f"not {expected_git!r}")
    if [d.get("manifest_sha256") for d in prov["datasets"]] != \
            [d.get("manifest_sha256") for d in e1p["datasets"]]:
        raise ValueError("adjudication refused: the run used another corpus than E1's")
    g = grid(expected_cfg)
    if [int(s) for s in prov.get("seeds") or []] != g["seeds"]:
        raise ValueError(f"adjudication refused: the run's seeds {prov.get('seeds')} are not "
                         f"{g['seeds']}")
    for row, b in row_cells(g):
        c = cell(rep, row, b)
        n = 1 if row in NAIVE else len(g["seeds"])       # the naive rows are seedless
        evs = c.get("per_seed_eval") or []
        if len(evs) != n:
            raise ValueError(f"adjudication refused: the {row} cell at {b} holds {len(evs)} "
                             f"evaluations, not {n}")
        for e in evs:
            per = e.get("per_instance") or {}
            if any(not isinstance(e.get(m), (int, float)) for m in METRICS):
                raise ValueError(f"adjudication refused: the {row} cell at {b} lacks a seed "
                                 f"value of a metric")
            if any(len(per.get(m) or ()) != g["n_val"] for m in METRICS):
                raise ValueError(f"adjudication refused: the {row} cell at {b} lacks values "
                                 f"for the {g['n_val']} validation instances")
    return g


def _repro(c: dict, ref: dict, metrics) -> float:
    """Largest relative deviation of a cell's per-instance arrays from a
    reference cell's (a non-finite value never reproduces)."""
    got, want = c["per_seed_eval"], ref["per_seed_eval"]
    if len(got) != len(want):
        return float("inf")
    dev = 0.0
    for a_, b_ in zip(got, want, strict=True):
        for m in metrics:
            a = np.asarray(a_["per_instance"][m], float)
            b = np.asarray(b_["per_instance"][m], float)
            if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                return float("inf")
            dev = max(dev, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))))
    return dev


def check_reuse(rep: dict, e1: dict, e1_report_sha256: str | None, pool: int) -> dict:
    """Is the label-free row E1's states, evaluated, reproducing E1's arrays?
    {"ok", "reasons", ...}; never raises on a failed check."""
    e8 = rep["results"]["e8"]
    d9 = e8["metrics"]["d9_restart"]
    want = {k: v["sha256"] for k, v in
            e1["results"]["e8"]["metrics"]["d9_restart"]["ar_states"].items()}
    got = {k: v.get("sha256") for k, v in d9["ar_states"].items()}
    reasons = []
    if not e8["protocol"].get("eval_only"):
        reasons.append("the label-free row was not evaluation only")
    if got != want or not all(v.get("reused") for v in d9["ar_states"].values()):
        reasons.append("the label-free row did not evaluate E1's states "
                       f"({ {k: str(v)[:12] for k, v in got.items()} })")
    rf = rep.get("reuse_from") or {}
    if e1_report_sha256 is not None and rf.get("report_sha256") != e1_report_sha256:
        reasons.append("the states were reused through another report than E1's records copy")
    if rf.get("supervised_grid") is not True:
        reasons.append("the run did not reuse the states as a supervised grid")
    ar, ref = cell(rep, "ar", pool), cell(e1, "ar", pool)
    dev = _repro(ar, ref, REPRO_METRICS)
    if not dev <= REPRO_MAX:
        reasons.append(f"the label-free row does not reproduce E1's validation arrays "
                       f"(largest relative deviation {dev:.3g} > {REPRO_MAX:g})")
    others = _repro(ar, ref, [m for m in METRICS if m not in REPRO_METRICS])
    return {"ok": not reasons, "reasons": reasons, "reproduction_max_rel_dev": dev,
            "other_metrics_max_rel_dev": others, "states_sha256": got}


# ------------------------------------------------------------- verdicts --

def reading(base_vals, new_vals) -> dict:
    """The guard (PREREG_CM2D Sec. 4) on two arms' per-seed values; a
    non-finite reference is reported, not raised."""
    from fejepa.analysis.adjudicate_w9 import guard

    try:
        return guard(base_vals, new_vals)
    except ValueError as exc:
        return {"not_evaluated": str(exc)}


def _direction(r: dict) -> str:
    return "lower" if r.get("lower") else ("worse" if r.get("worse") else "neither")


def robustness(base: dict, new: dict, metric: str, direction: str) -> dict:
    """Beside a primary comparison (no criterion): the guard on per-seed
    medians, rel's Welch interval, the instance-resampling interval; and
    whether the medians show the comparison's direction."""
    from fejepa.analysis.adjudicate_w9 import guard, instance_bootstrap, welch_interval

    pb, pa = per_instance(base, metric), per_instance(new, metric)
    med = _soft(guard, [float(np.median(x)) for x in pb], [float(np.median(x)) for x in pa])
    out = {"medians": med,
           "welch_95": _soft(welch_interval, seed_values(base, metric),
                             seed_values(new, metric)),
           "instance_resampling_95": _soft(instance_bootstrap, pb, pa)}
    out["medians_show_the_direction"] = (None if "not_evaluated" in med
                                         else _direction(med) == direction)
    return out


def _verdict(base: dict, new: dict, what: str, metric: str) -> dict:
    r = reading(seed_values(base, metric), seed_values(new, metric))
    if "not_evaluated" in r:
        return {"verdict": "NOT EVALUATED", "reason": f"the reference: {r['not_evaluated']}",
                "comparison": what, "metric": metric}
    out = {"comparison": what, "metric": metric, **r}
    if r["diverged"]:
        out["verdict"] = "NOT SUPPORTED (diverged)"
        out["robustness"] = None
        return out
    out["verdict"] = "SUPPORTED" if r["lower"] else "NOT SUPPORTED"
    if r["worse"]:
        out["note"] = "worse beyond the guard"
    out["robustness"] = _soft(robustness, base, new, metric, _direction(r))
    return out


def h3_reading(ar: dict, knorm: dict) -> dict:
    r = reading(seed_values(ar, PRIMARY), seed_values(knorm, PRIMARY))
    what = "the stiffness-norm transformer against the label-free transformer (reference)"
    if "not_evaluated" in r:
        return {"reading": "NOT EVALUATED", "reason": f"the reference: {r['not_evaluated']}",
                "comparison": what, "metric": PRIMARY}
    if r["diverged"]:
        label = "the stiffness-norm transformer diverged"
    elif r["lower"]:
        label = "the stiffness-norm transformer lower beyond the guard"
    elif r["worse"]:
        label = "the label-free transformer lower beyond the guard"
    else:
        label = "no difference shown (within the guard)"
    swapped = reading(seed_values(knorm, PRIMARY), seed_values(ar, PRIMARY))
    return {"reading": label, "comparison": what, "metric": PRIMARY, **r,
            "robustness": (None if r["diverged"] else
                           _soft(robustness, ar, knorm, PRIMARY, _direction(r))),
            "roles_exchanged": {"comparison": "the label-free transformer against the "
                                              "stiffness-norm transformer (reference)",
                                **swapped}}


# ---------------------------------------------------- secondary readings --

def _comparison(base: dict, new: dict, metric: str) -> dict:
    r = reading(seed_values(base, metric), seed_values(new, metric))
    if "not_evaluated" in r:
        return r
    higher = metric in HIGHER_IS_BETTER
    r["better_direction"] = "higher" if higher else "lower"
    r["new_better_beyond_guard"] = bool(r["worse"] if higher else r["lower"])
    r["new_worse_beyond_guard"] = bool(r["lower"] if higher else r["worse"])
    return r


def comparisons(rep: dict, g: dict) -> dict:
    """Every metric at every budget: each trained row against the label-free
    row, and the stiffness-norm and graph-network rows against labels-only."""
    ar = cell(rep, "ar", g["pool"])
    out = {}
    for m in METRICS:
        per_b = {}
        for b in g["budgets"]:
            rows = ["labels", "labels_knorm"] + (["mgn"] if b in g["mgn_budgets"] else [])
            lab = cell(rep, "labels", b)
            entry = {f"{r}_vs_label_free": _comparison(ar, cell(rep, r, b), m) for r in rows}
            entry["labels_knorm_vs_labels"] = _comparison(lab, cell(rep, "labels_knorm", b), m)
            if "mgn" in rows:
                entry["mgn_vs_labels"] = _comparison(lab, cell(rep, "mgn", b), m)
            per_b[str(b)] = entry
        out[m] = per_b
    return out


def seed_table(rep: dict, g: dict) -> dict:
    """Per row and budget: per-seed values and the seed mean of every metric."""
    out = {}
    for row, b in row_cells(g):
        c = cell(rep, row, b)
        out.setdefault(row, {})[str(b)] = {
            m: {"per_seed": seed_values(c, m), "mean": float(np.mean(seed_values(c, m)))}
            for m in METRICS}
    return out


def label_efficiency(rep: dict, g: dict) -> dict:
    ar = cell(rep, "ar", g["pool"])
    out = {}
    for r in ("labels", "labels_knorm"):
        adv, first = {}, {}
        for b in g["budgets"]:
            ga = float(np.mean(seed_values(ar, PRIMARY)))
            gx = float(np.mean(seed_values(cell(rep, r, b), PRIMARY)))
            adv[str(b)] = 1.0 - ga / gx if gx else None
        for m in ("energy_gap_rel", "disp_rel_l2"):
            a = float(np.mean(seed_values(ar, m)))
            below = [b for b in g["budgets"]
                     if float(np.mean(seed_values(cell(rep, r, b), m))) < a]
            first[m] = min(below) if below else None
        out[r] = {"label_free_advantage_in_energy_gap": adv,
                  "first_budget_below_label_free": first}
    return out


def pairing(rep: dict, g: dict) -> dict:
    """At the decision budget: on how many instance-seed pairs the first
    network of each pair has the lower value (ties apart)."""
    b = g["decision"]
    out = {}
    for x, y in PAIRS:
        if "mgn" in (x, y) and b not in g["mgn_budgets"]:
            continue
        cx, cy = _cell_at(rep, x, g), _cell_at(rep, y, g)
        res = {}
        for m in ("energy_gap_rel", "disp_rel_l2", "vm_rel_l2"):
            X, Y = np.concatenate(per_instance(cx, m)), np.concatenate(per_instance(cy, m))
            res[m] = {"first_lower": int(np.sum(X < Y)), "ties": int(np.sum(X == Y)),
                      "pairs": int(X.size)}
        out[f"{x}_vs_{y}"] = res
    return out


def worse_than_zero(rep: dict, g: dict) -> dict:
    """Per row and budget, the instance-seed values with a relative energy gap
    above 1 (a prediction further from the solution, in the energy norm, than
    the zero field); the naive rows, which have no seeds, count their
    instance values."""
    out = {}
    for row, b in row_cells(g):
        x = np.concatenate(per_instance(cell(rep, row, b), PRIMARY))
        out.setdefault(row, {})[str(b)] = {"count": int(np.sum(x > 1.0)), "of": int(x.size)}
    return out


def mgn_reading(rep: dict, g: dict) -> dict | None:
    b = g["decision"]
    if b not in g["mgn_budgets"]:
        return None
    disp = {r: float(np.mean(seed_values(_cell_at(rep, r, g), "disp_rel_l2")))
            for r in ("ar", "labels", "labels_knorm", "mgn")}
    gap = {r: float(np.mean(seed_values(_cell_at(rep, r, g), PRIMARY)))
           for r in ("ar", "labels", "mgn")}
    return {"budget": b, "disp_seed_means": disp, "energy_gap_seed_means": gap,
            "mgn_lowest_displacement": bool(disp["mgn"] == min(disp.values())),
            "mgn_gap_above_labels": bool(gap["mgn"] > gap["labels"]),
            "mgn_gap_above_label_free": bool(gap["mgn"] > gap["ar"])}


def july(rep: dict, g: dict, july_rep: dict | None) -> dict | None:
    """The labels-only row against the run of 16 July 2026 (other code;
    descriptive only)."""
    if july_rep is None:
        return None
    out = {}
    for b in g["budgets"]:
        try:
            old = cell(july_rep, "labels", b)
        except ValueError:
            continue
        new = cell(rep, "labels", b)
        out[str(b)] = {m: {"july": float(np.mean(seed_values(old, m))),
                           "now": float(np.mean(seed_values(new, m)))}
                       for m in ("energy_gap_rel", "disp_rel_l2", "vm_rel_l2")}
    return out


def builtin(rep: dict) -> dict:
    e8 = rep["results"]["e8"]
    return {"kills": e8.get("kills"),
            "gate_g1_prime_passed": (rep.get("gate_g1_prime") or {}).get("passed"),
            "divergence_flags": {r: {b: c.get("divergence_flags") for b, c in cells.items()}
                                 for r, cells in e8["metrics"]["cells"].items()
                                 if r not in NAIVE}}


# ------------------------------------------------------------ deviations --

def status_attempts(status_text: str | None) -> dict:
    """The run's attempts in the status file: the start lines
    (`run.log start <time>`) and the exit codes (`run.log exit=N`), in order."""
    t = status_text or ""
    return {"starts": len(re.findall(r"^run\.log start\b", t, re.M)),
            "exits": [int(m.group(1)) for m in
                      re.finditer(r"^run\.log exit=(-?\d+)\s*$", t, re.M)]}


def deviations(rep: dict, e1: dict, status_text: str | None, run_logs: int | None = None) -> list:
    out = []
    e8 = rep["results"]["e8"]
    d9 = e8["metrics"]["d9_restart"]
    if rep.get("d9_reuse_states"):
        out.append("the run was restarted with --reuse-states (D9)")
    if d9.get("sup_units_from_cache"):
        out.append(f"supervised units taken from the unit cache: {d9['sup_units_from_cache']}")
    if d9.get("units_resumed_from_epoch"):
        out.append(f"units resumed from an epoch checkpoint: {d9['units_resumed_from_epoch']}")
    ckpt = (rep.get("runtime_overrides") or {}).get("activation_checkpointing")
    if ckpt is False:
        out.append("activation checkpointing off (PREREG_CM2D Sec. 6: on)")
    if e8["protocol"].get("workers") != 3:
        out.append(f"workers {e8['protocol'].get('workers')} (PREREG_CM2D Sec. 6: 3)")
    total = int((rep.get("solve_ledger") or {}).get("total", 0))
    if total:
        out.append(f"the solve ledger reads {total}, not 0")
    for what, a, b in (("torch", (rep["provenance"].get("versions") or {}).get("torch"),
                        (e1["provenance"].get("versions") or {}).get("torch")),
                       ("GPU", (rep.get("runtime_policy") or {}).get("gpu"),
                        (e1.get("runtime_policy") or {}).get("gpu"))):
        if a != b:
            out.append(f"{what} {a!r}, E1's states were trained with {b!r}")
    if status_text is not None:
        att = status_attempts(status_text)
        if att["starts"] != 1 or att["exits"] != [0]:
            out.append(f"run attempts in the status file: {att['starts']} started, "
                       f"exit codes {att['exits']}")
    if run_logs is not None and run_logs != 1:
        out.append(f"{run_logs} run logs in the return (one per attempt)")
    return out


# --------------------------------------------------------------- the lot --

def adjudicate_cm2d(rep: dict, e1: dict, expected_cfg: dict, prereg_entries: dict | None,
                    expected_git: str | None, e1_report_sha256: str | None,
                    status_text: str | None = None, july_rep: dict | None = None,
                    run_logs: int | None = None) -> dict:
    """PREREG_CM2D Sec. 4-6 on one report. Raises ValueError on a refusal."""
    g = check_report(rep, e1, expected_cfg, prereg_entries, expected_git)
    reuse = check_reuse(rep, e1, e1_report_sha256, g["pool"])
    reason = "the label-free row: " + "; ".join(reuse["reasons"])
    out = {"what": "PREREG_CM2D verdicts (scripts/adjudicate_cm2d.py)",
           "budget": g["decision"]}
    for h, (a, b, metric, what) in HYPOTHESES.items():
        if h in NEEDS_REUSE and not reuse["ok"]:
            out[h] = {"verdict": "NOT EVALUATED", "reason": reason, "comparison": what,
                      "metric": metric}
        else:
            out[h] = _verdict(_cell_at(rep, b, g), _cell_at(rep, a, g), what, metric)
    out["H3"] = (h3_reading(_cell_at(rep, "ar", g), _cell_at(rep, "labels_knorm", g))
                 if reuse["ok"] else
                 {"reading": "NOT EVALUATED", "reason": reason, "metric": PRIMARY,
                  "comparison": "the stiffness-norm transformer against the label-free "
                                "transformer (reference)"})
    e8 = rep["results"]["e8"]
    out["reuse"] = reuse
    out["secondary"] = {
        "seed_table": _soft(seed_table, rep, g), "comparisons": _soft(comparisons, rep, g),
        "label_efficiency": _soft(label_efficiency, rep, g), "pairing": _soft(pairing, rep, g),
        "worse_than_zero": _soft(worse_than_zero, rep, g),
        "graph_network": _soft(mgn_reading, rep, g),
        "july_labels_only": _soft(july, rep, g, july_rep), "e8_builtin": _soft(builtin, rep)}
    out["deviations"] = deviations(rep, e1, status_text, run_logs)
    out["run"] = {"git": rep["provenance"].get("git"),
                  "config_sha256": rep["provenance"].get("config_sha256"),
                  "workers": e8["protocol"].get("workers"),
                  "runtime_overrides": rep.get("runtime_overrides"),
                  "torch": (rep["provenance"].get("versions") or {}).get("torch"),
                  "gpu": (rep.get("runtime_policy") or {}).get("gpu")}
    return out


# -------------------------------------------------------------- inputs --

def prereg_hash(path) -> dict:
    """The stamped entries of PREREG_CM2D.md; refuses an unstamped file or one
    whose last line is not its own SHA-256."""
    from fejepa.report import PREREG_PLACEHOLDER, read_prereg_entries

    entries = dict(read_prereg_entries(path))
    if entries.get(LABEL) in (None, PREREG_PLACEHOLDER):
        raise SystemExit(f"{path}: not stamped")
    if not _script("stamp_prereg_cm2d").self_hash_ok(Path(path).read_text(encoding="utf-8")):
        raise SystemExit(f"{path}: its last line is not its own SHA-256")
    return entries


def provenance_report_sha(text: str) -> str | None:
    """The SHA-256 the return's provenance file lists for runs/cm2d/report.json."""
    for line in text.splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+(?:\S*/)?runs/cm2d/report\.json$", line.strip())
        if m:
            return m.group(1)
    return None


def provenance_commit(text: str) -> dict:
    """The `HEAD <sha>` and `tree <sha>` lines of the provenance file."""
    out = {}
    for key in ("HEAD", "tree"):
        m = re.search(rf"^{key} ([0-9a-f]{{40}})\s*$", text, re.M)
        out[key] = m.group(1) if m else None
    return out


def _git(*args) -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                           timeout=30)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def check_commit(prov_text: str, tag: str = TAG) -> str | None:
    """Refuse a return whose recorded commit or tree is not `tag`'s in this
    checkout (SystemExit); return a deviation note when the tag is absent."""
    rec = provenance_commit(prov_text)
    want = {"HEAD": _git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"),
            "tree": _git("rev-parse", "--verify", "--quiet", f"{tag}^{{tree}}")}
    if want["HEAD"] is None:
        return f"the run's commit not compared with {tag} (no such tag in this checkout)"
    if rec != want:
        raise SystemExit(f"the return records HEAD {rec['HEAD']} and tree {rec['tree']}, "
                         f"{tag} is {want['HEAD']} with tree {want['tree']}")
    return None


def load_return(ret: Path) -> tuple:
    missing = [f for f in RETURN_FILES if not (ret / f).is_file()]
    if missing:
        raise SystemExit(f"{ret}: the return lacks {missing} (RUNBOOK_CMAME Sec. B)")
    prov = (ret / "provenance.txt").read_text()
    listed = provenance_report_sha(prov)
    got = file_sha256(ret / "report.json")
    if listed != got:
        raise SystemExit(f"{ret / 'report.json'}: not the report the box hashed "
                         f"(provenance.txt lists {listed}, the file is {got})")
    logs = sorted(p.name for p in ret.glob("run.log*"))
    return load_json(ret / "report.json"), got, (ret / "status.txt").read_text(), prov, logs


def code_against(ref: str) -> dict:
    """The adjudicating files in this checkout against their versions at `ref`."""
    out = {"ref": ref, "compared": False, "differs": []}
    try:
        ok = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet",
                             f"{ref}^{{commit}}"], capture_output=True, timeout=10).returncode == 0
        if not ok:
            out["note"] = f"no {ref} in this checkout"
            return out
        for p in ADJUDICATING_FILES:
            rc = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", ref, "--", p],
                                capture_output=True, timeout=30).returncode
            if rc == 1:
                out["differs"].append(p)
            elif rc != 0:
                raise RuntimeError(f"git diff exited {rc} on {p}")
        out["compared"] = True
    except Exception as exc:                              # noqa: BLE001
        out["differs"] = []
        out["note"] = f"comparison failed ({type(exc).__name__}: {exc})"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--return", dest="ret", required=True, help="the return directory")
    ap.add_argument("--e1-report", default=str(ROOT / "records/wp8/e1/e1_2d_base/report.json"))
    ap.add_argument("--prereg", default=str(ROOT / PREREG_NAME))
    ap.add_argument("--july-report", default=str(ROOT / "records/phase1/report_rec8_v2.json"))
    ap.add_argument("--expected-git", default=TAG, help="the git describe the run must record")
    ap.add_argument("--code-ref", default=TAG,
                    help="the commit the adjudicating code is compared with")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    e1 = load_json(a.e1_report)
    try:
        check_e1(e1)
    except ValueError as exc:
        raise SystemExit(f"{a.e1_report}: {exc}") from exc
    entries = prereg_hash(a.prereg)
    rep, rep_sha, status, prov, logs = load_return(Path(a.ret))
    commit_note = check_commit(prov, a.expected_git)
    expected = _script("make_cm2d_config").cm2d_config(e1["config"])
    july_rep = load_json(a.july_report) if a.july_report else None
    try:
        res = adjudicate_cm2d(rep, e1, expected, entries, a.expected_git,
                              file_sha256(a.e1_report), status, july_rep, len(logs))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if commit_note:
        res["deviations"].append(commit_note)
    res["inputs"] = {
        "report": {"path": str(Path(a.ret) / "report.json"), "sha256": rep_sha},
        **{f: {"path": str(Path(a.ret) / f), "sha256": file_sha256(Path(a.ret) / f)}
           for f in RETURN_FILES[1:]},
        "run_logs": {n: file_sha256(Path(a.ret) / n) for n in logs},
        "recorded_commit": provenance_commit(prov),
        "e1_report": {"path": a.e1_report, "sha256": file_sha256(a.e1_report)},
        "prereg": {"path": a.prereg, "sha256": file_sha256(a.prereg)},
        "july_report": ({"path": a.july_report, "sha256": file_sha256(a.july_report)}
                        if a.july_report else None)}
    code = code_against(a.code_ref)
    res["adjudicator"] = {"sha256": {p: file_sha256(ROOT / p) for p in ADJUDICATING_FILES},
                          "against_stamp": code}
    if not code["compared"]:
        res["deviations"].append(f"adjudicating code not compared with {a.code_ref} "
                                 f"({code.get('note')})")
    elif code["differs"]:
        res["deviations"].append(f"adjudicating code differs from {a.code_ref} in "
                                 f"{code['differs']} (its SHA-256 under adjudicator)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(jsonable(res), indent=1, allow_nan=False) + "\n")
    print(json.dumps({h: res[h]["verdict"] for h in HYPOTHESES} | {
        "H3": res["H3"]["reading"], "reuse_ok": res["reuse"]["ok"],
        "deviations": res["deviations"]}, indent=1))


if __name__ == "__main__":
    main()
