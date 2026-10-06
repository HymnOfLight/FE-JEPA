#!/usr/bin/env python3
"""cmame-paper Stage 2: operating characteristics of PREREG_CM2D's decision
rule -- the E-series noise guard on three seeds per arm -- by simulation, and
the run's wall time by a simulation of its unit schedule.

Guard. Each arm's three per-seed values are log-normal around the arm's true
level with coefficient of variation c_ref (reference) or c_new (new arm); the
new arm's true level is r times the reference's. Per trial, as the
adjudication computes it: rel = mean(new)/mean(ref) - 1, SE_rel =
sqrt(s_new^2/3 + s_ref^2/3)/mean(ref) with sample standard deviations,
tau = max(10%, 2 x SE_rel); the new arm is "lower beyond the guard" iff
rel < -tau and "worse beyond the guard" iff rel > tau.

Blow-ups. The log-normal model has no per-instance outliers. In the
`blowups` cells a seed's value is additionally multiplied by
1 + sum_k (m_k - 1)/256: per seed a Poisson number (mean `rate`) of the
256 validation instances read m_k ~ U(20, 40) times the arm's level instead
of about one, in one arm only ("ref" or "new") or in both.

Schedule. The 30 supervised units of configs/cm2d_v1.json in the order the
run submits them (seed, budget, row), each taken by the first free of three
workers, at a supervised step of 52 ms for the transformers (measured on the
box with activation checkpointing, records/wp8/posthoc/profile_2d_head.json)
and a multiple of it for the graph network (not measured), plus one minute
per unit; the hours until the last unit ends.

    python scripts/cm2d_sims.py --out records/cmame/cm2d_sims.json
    python scripts/cm2d_sims.py --check records/cmame/cm2d_sims.json
"""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np

SEED = 20261006
TRIALS = 200_000
N_SEEDS = 3
N_VAL = 256
BAND, K = 0.10, 2.0
RATIOS = (0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
NOISE = ((0.05, 0.05), (0.10, 0.10), (0.20, 0.20), (0.30, 0.30), (0.05, 0.20), (0.20, 0.05))
"""(c_ref, c_new) pairs: equal seed spreads, and a quiet reference with a noisy
new arm and the reverse."""
JULY_REF = 0.09
"""The seed spread of July's labels-only transformer at 1,024 labels (8.9%)."""
JULY_NEW = (0.05, 0.09, 0.20, 0.30, 0.50)
JULY_RATIOS = (0.5, 0.8, 1.0)
BLOWUP = {"c": 0.05, "rate": 1.0, "low": 20.0, "high": 40.0}
STEP_S = 0.052
UNIT_OVERHEAD_S = 60.0
MGN_FACTORS = (0.5, 1.0, 1.5, 2.0)
BUDGETS, MGN_BUDGETS, EPOCHS, WORKERS = (16, 64, 256, 1024), (64, 1024), 200, 3


def _lognormal(rng, level: float, cv: float, size) -> np.ndarray:
    s = np.sqrt(np.log1p(cv * cv))
    return np.exp(rng.normal(np.log(level) - s * s / 2, s, size=size))


def _blowups(rng, size) -> np.ndarray:
    """Per-seed multipliers 1 + sum_k (m_k - 1)/N_VAL."""
    n = rng.poisson(BLOWUP["rate"], size=size)
    out = np.ones(size)
    for k in range(1, int(n.max()) + 1 if n.size else 1):
        m = rng.uniform(BLOWUP["low"], BLOWUP["high"], size=size)
        out += np.where(n >= k, (m - 1.0) / N_VAL, 0.0)
    return out


def _guard(ref: np.ndarray, new: np.ndarray) -> dict:
    mb, ma = ref.mean(axis=1), new.mean(axis=1)
    se = np.sqrt(ref.var(axis=1, ddof=1) / N_SEEDS + new.var(axis=1, ddof=1) / N_SEEDS) / mb
    tau = np.maximum(BAND, K * se)
    rel = ma / mb - 1
    return {"lower": round(float(np.mean(rel < -tau)), 5),
            "worse": round(float(np.mean(rel > tau)), 5),
            "tau_median": round(float(np.median(tau)), 4)}


def cell(rng, r: float, c_ref: float, c_new: float, trials: int = TRIALS) -> dict:
    return _guard(_lognormal(rng, 1.0, c_ref, (trials, N_SEEDS)),
                  _lognormal(rng, r, c_new, (trials, N_SEEDS)))


def blowup_cell(rng, where: str, trials: int = TRIALS) -> dict:
    c = BLOWUP["c"]
    ref = _lognormal(rng, 1.0, c, (trials, N_SEEDS))
    new = _lognormal(rng, 1.0, c, (trials, N_SEEDS))
    if where in ("ref", "both"):
        ref = ref * _blowups(rng, (trials, N_SEEDS))
    if where in ("new", "both"):
        new = new * _blowups(rng, (trials, N_SEEDS))
    return _guard(ref, new)


def schedule_hours(mgn_factor: float) -> float:
    units = [(r, b) for _s in range(3) for b in BUDGETS
             for r in ("labels", "labels_knorm", "mgn") if r != "mgn" or b in MGN_BUDGETS]
    free = [0.0] * WORKERS
    heapq.heapify(free)
    for r, b in units:
        t = heapq.heappop(free)
        step = STEP_S * (mgn_factor if r == "mgn" else 1.0)
        heapq.heappush(free, t + EPOCHS * b * step + UNIT_OVERHEAD_S)
    return round(max(free) / 3600.0, 2)


def simulate(trials: int = TRIALS) -> dict:
    rng = np.random.default_rng(SEED)
    out = {}
    for c_ref, c_new in NOISE:
        out[f"c_ref={c_ref:.2f},c_new={c_new:.2f}"] = {
            f"r={r:g}": cell(rng, r, c_ref, c_new, trials) for r in RATIOS}
    july = {}
    for c_new in JULY_NEW:
        july[f"c_ref={JULY_REF:.2f},c_new={c_new:.2f}"] = {
            f"r={r:g}": cell(rng, r, JULY_REF, c_new, trials) for r in JULY_RATIOS}
    blow = {w: blowup_cell(rng, w, trials) for w in ("ref", "new", "both")}
    return {"what": "PREREG_CM2D operating characteristics (scripts/cm2d_sims.py)",
            "seed": SEED, "trials_per_cell": trials, "seeds_per_arm": N_SEEDS,
            "guard": {"band": BAND, "k": K}, "model": "log-normal per-seed values",
            "cells": out, "july_reference": july,
            "blowups": {"model": BLOWUP, "r": 1.0, "cells": blow},
            "schedule": {"step_s": STEP_S, "unit_overhead_s": UNIT_OVERHEAD_S,
                         "workers": WORKERS,
                         "hours_by_mgn_step_factor": {f"{f:g}": schedule_hours(f)
                                                      for f in MGN_FACTORS}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--out")
    g.add_argument("--check", help="exit non-zero unless this file is the simulation's output")
    a = ap.parse_args()
    text = json.dumps(simulate(), indent=1) + "\n"
    if a.check:
        if Path(a.check).read_text() != text:
            raise SystemExit(f"{a.check}: not this simulation's output")
        return
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
