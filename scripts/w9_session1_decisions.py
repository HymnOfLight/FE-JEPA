#!/usr/bin/env python3
"""wp9 Stage 0a: the session-1 decision rules, as code, committed before the
session runs (BRANCH_NOTES_wp9-pool.md, "Pre-declared readings"). Reads the
session's outputs and writes the decisions they imply; nothing here is tuned
after the readings are seen.

  1. S enters the pre-registration iff, for the E1 base states, the F5 / in-band
     displacement ratio (seed means) is >= 1.5 (KP4's own line) AND the median
     c* on F5 (seed mean of per-seed medians) is >= 1.10.
  2. C1's largest pool keeps the design before the schedule: 25,600 with three
     workers if they fit, else 25,600 with one worker at a time (seeds
     sequential), else 12,800 with three workers, else with one; else neither
     (the pool design is revisited). k workers fit at pool size n iff
       host:   k x (one worker's residency at n + the host growth of the AR
               steps) <= 80% of the usable host memory, and
       device: k x (CUDA context + device residency at n + the extra memory of
               AR steps with activation checkpointing on) <= 80% of the
               device memory (the context: the larger of the readings before
               and after the steps).
     Residency at a pool size the audit did not reach is extrapolated linearly
     from its last two marks.
  3. wp9's 2D configurations turn activation checkpointing off iff the AR step
     in E1's worker set-up is >= 1.3x faster without it (worker3 against
     worker3_no_ckpt, ms per step per unit, both valid) AND rule 2's choice
     still fits with the extra memory of steps without it; otherwise it stays
     on. (Memory first: a faster step never costs the pool or the schedule.)
  4. C0.4 is a prediction only: training-set / validation-set energy gap < 0.8
     reads "the 1,024-instance arm overfits: C1 likely improves in-band";
     otherwise "near parity: C1's in-band gain likely small". C1's quantities
     do not depend on it.
  5. For C2 later (recorded only): the CG energy-drop estimate is usable as a
     score iff its seed-mean Spearman correlation with the true gap at k = 10
     is >= 0.9 (no reading at k = 10, or an undefined correlation: not decided).

Each rule is evaluated on its own: a missing or malformed input leaves the
rules that read it undecided (value null, with the error) and the others
decided.

    python scripts/w9_session1_decisions.py --dir runs/w9/session1 \
        --out runs/w9/session1/decisions.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

S_RATIO_MIN, S_CSTAR_MIN = 1.5, 1.10
CKPT_SPEEDUP_MIN = 1.3
POOLS = (25600, 12800)
MEM_FRACTION = 0.8
OVERFIT_RATIO = 0.8
C2_RHO_MIN, C2_K = 0.9, "10"
INPUTS = ("amp2d", "timing", "memory", "trainval", "val")


def at(memory: dict, n: int, key: str = "rss") -> float:
    """One worker's residency (bytes) at pool size n under `key` (host "rss"
    or device "gpu_reserved"): measured, or extrapolated linearly from the
    last two marks."""
    marks = sorted(memory["marks"], key=lambda m: m["n"])
    for m in marks:
        if m["n"] == n:
            return float(m[key])
    if len(marks) < 2:
        raise ValueError("fewer than two memory marks: cannot extrapolate")
    (n1, r1), (n2, r2) = [(m["n"], m[key]) for m in marks[-2:]]
    return float(r2 + (n - n2) * (r2 - r1) / (n2 - n1))


def rss_at(memory: dict, n: int) -> float:
    return at(memory, n, "rss")


def host_need(memory: dict, n: int) -> float:
    """One worker's host memory at pool size n: residency + the growth of the
    AR steps (the RSS after the step measurement over the last mark's)."""
    grow = 0.0
    if memory.get("rss_after_step") is not None and memory.get("marks"):
        last = max(memory["marks"], key=lambda m: m["n"])
        grow = max(0.0, float(memory["rss_after_step"]) - float(last["rss"]))
    return rss_at(memory, n) + grow


def gpu_need(memory: dict, n: int, ckpt_on: bool) -> float | None:
    """One worker's device memory at pool size n: CUDA context + residency +
    the extra memory of AR steps; None without a GPU reading."""
    g = memory.get("gpu")
    if not g:
        return None
    if "step" not in g:
        raise KeyError(f"no step measurement in the memory audit ({g.get('step_error')})")
    step = g["step"]["ckpt_on" if ckpt_on else "ckpt_off"]["reserved_overhead"]
    ctx = max(float(g["context_estimate"]), float(g.get("context_after_step") or 0))
    return ctx + at(memory, n, "gpu_reserved") + float(step)


def fits(memory: dict, n: int, k: int, ckpt_on: bool) -> bool:
    host = k * host_need(memory, n) <= MEM_FRACTION * float(memory["limits"]["usable"])
    g = gpu_need(memory, n, ckpt_on)
    dev = g is None or k * g <= MEM_FRACTION * float(memory["gpu"]["total"])
    return bool(host and dev)


def rule_s(amp2d: dict) -> dict:
    s = amp2d["summary"]
    ratio, cstar = float(s["ratio_disp_F5_over_inband"]), float(s["c_star_median_F5"])
    return {"value": bool(ratio >= S_RATIO_MIN and cstar >= S_CSTAR_MIN),
            "ratio_disp_F5_over_inband": ratio, "c_star_median_F5": cstar,
            "F5": {k: amp2d.get("F5", {}).get(k) for k in ("family", "seed", "n_instances")},
            "rule": f"ratio >= {S_RATIO_MIN} and c* >= {S_CSTAR_MIN}"}


def rule_pool(memory: dict) -> dict:
    if float(memory["limits"]["usable"]) <= 0:
        raise ValueError("the memory audit read no usable host memory")
    workers = int(memory.get("workers", 3))
    cap = {"n": None, "mode": None, "k": None}
    for n in POOLS:                          # the design first, then the schedule
        for mode, k in (("parallel", workers), ("sequential", 1)):
            if fits(memory, n, k, ckpt_on=True):
                cap = {"n": n, "mode": mode, "k": k}
                break
        if cap["n"]:
            break
    gpu = memory.get("gpu")
    return {"value": {"n": cap["n"], "mode": cap["mode"]}, "n": cap["n"], "mode": cap["mode"],
            "k": cap["k"], "workers": workers,
            "usable_host_bytes": float(memory["limits"]["usable"]),
            "usable_host_source": memory["limits"].get("usable_source"),
            "host_per_worker_bytes": {str(n): host_need(memory, n) for n in POOLS},
            "gpu_total_bytes": float(gpu["total"]) if gpu else None,
            "gpu_per_worker_bytes_ckpt_on": ({str(n): gpu_need(memory, n, True) for n in POOLS}
                                             if gpu else None),
            "rule": f"25,600 then 12,800; at each, {workers} workers then one; fit = host and "
                    f"device needs <= {MEM_FRACTION} x their limits, checkpointing on"}


def rule_ckpt(timing: dict, memory: dict, pool: dict) -> dict:
    v = timing.get("variants", {})
    on, off = v.get("worker3", {}), v.get("worker3_no_ckpt", {})
    try:
        a, b = on["ar"], off["ar"]
        valid = bool(a["valid"] and b["valid"])
        speedup = float(a["ms_per_step_per_unit"]) / float(b["ms_per_step_per_unit"])
    except (KeyError, TypeError, ZeroDivisionError):
        valid, speedup = False, float("nan")
    fast = bool(valid and speedup >= CKPT_SPEEDUP_MIN)
    if pool.get("value") is None:
        raise ValueError(f"rule 2 is undecided ({pool.get('error')})")
    room = bool(pool["n"] and fits(memory, pool["n"], pool["k"], ckpt_on=False))
    return {"value": bool(fast and room), "speedup": speedup, "timings_valid": valid,
            "fits_without": room,
            "gpu_per_worker_bytes_ckpt_off": ({str(n): gpu_need(memory, n, False) for n in POOLS}
                                              if memory.get("gpu") else None),
            "rule": f"worker3 / worker3_no_ckpt >= {CKPT_SPEEDUP_MIN} with both valid, and the "
                    "pool choice fits without it; else on"}


def rule_c04(trainval: dict) -> dict:
    r = float(trainval["summary"]["egap_train_over_val"])
    return {"value": bool(r < OVERFIT_RATIO), "egap_train_over_val": r,
            "reading": ("the 1,024-instance arm overfits: C1 likely improves in-band"
                        if r < OVERFIT_RATIO else
                        "near parity: C1's in-band gain likely small"),
            "rule": f"< {OVERFIT_RATIO}: overfit; otherwise near parity (prediction only)"}


def rule_c2(val: dict) -> dict:
    rho = (val["summary"].get("cg_estimates") or {}).get(C2_K, {}).get("spearman_seed_mean")
    rho = float("nan") if rho is None else float(rho)
    return {"value": bool(rho >= C2_RHO_MIN) if math.isfinite(rho) else None,
            "spearman_k10": rho,
            "rule": f"seed-mean Spearman at k = {C2_K} >= {C2_RHO_MIN} "
                    "(recorded for C2; no session-2 effect)"}


def _guard(fn, *args) -> dict:
    try:
        if any(x is None for x in args):
            raise FileNotFoundError("an input of this rule is missing or unreadable")
        return fn(*args)
    except Exception as exc:                              # noqa: BLE001 -- the others still run
        return {"value": None, "error": f"{type(exc).__name__}: {exc}"}


def decide(amp2d=None, timing=None, memory=None, trainval=None, val=None) -> dict:
    out = {"S_enters": _guard(rule_s, amp2d)}
    out["pool_max"] = _guard(rule_pool, memory)
    out["checkpointing_off"] = _guard(rule_ckpt, timing, memory, out["pool_max"])
    out["C0_4_prediction"] = _guard(rule_c04, trainval)
    out["C2_score_usable"] = _guard(rule_c2, val)
    return out


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/w9/session1")
    ap.add_argument("--amp2d", default="c0_amp2d.json")
    ap.add_argument("--timing", default="profile_2d_w9.json")
    ap.add_argument("--memory", default="c0_memory.json")
    ap.add_argument("--trainval", default="c0_trainval.json")
    ap.add_argument("--val", default="c0_val.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    data, sha, problems = {}, {}, {}
    for k in INPUTS:
        p = d / getattr(a, k)
        try:
            raw = p.read_bytes()
            sha[p.name] = hashlib.sha256(raw).hexdigest()
            data[k] = json.loads(raw)
        except (OSError, ValueError) as exc:
            data[k] = None
            problems[p.name] = f"{type(exc).__name__}: {exc}"
    try:
        from fejepa.report import _git_describe
        git = _git_describe()
    except Exception:                                     # noqa: BLE001
        git = "unavailable"
    res = {"what": "wp9 session-1 decisions from the pre-declared rules", "git": git,
           "rules_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "inputs_sha256": sha, "inputs_unreadable": problems,
           "decisions": decide(**data)}
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["decisions"], indent=1))
    undecided = [k for k, v in res["decisions"].items() if v.get("error")]
    if undecided:
        raise SystemExit(f"undecided: {undecided} (see {a.out})")


if __name__ == "__main__":
    main()
