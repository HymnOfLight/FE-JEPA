#!/usr/bin/env python3
"""wp9 Stage 0c: the simulations behind PREREG_W9's operating characteristics
(Sec. 6-7), so that every number the file quotes can be re-derived. Writes
records/wp9/simulations.json; seconds on a laptop. Each section draws from
its own stream, numpy.random.default_rng([20261002, STREAMS[section]]).

The guard is PREREG_W9 Sec. 6's (three seeds per arm unless stated, sample
standard deviations, tau = max(10%, 2 SE_rel)); "support" means rel < -tau.

  h1_r2_model     H1 under r2's model: both arms log-normal across seeds
                  (coefficient of variation 4.9%, E1's sample value, or 10%).
  h1_instances    H1 on E1's measured per-instance structure: the reference's
                  validation part is E1's own arrays (instance 233's blow-up
                  included); per instance-seed jitter (log-sd 0.08, E1's
                  0.06-0.10), a seed-level factor (sd 0.5% or 2%), and
                  blow-ups (rate 0, 1 or 2 per 1,792 instance-seeds -- E1 had
                  1 in 7 x 256 -- value uniform 0.3-0.9) in every arm and set
                  not fixed by E1's arrays. Designs: the validation split
                  alone (r2) and the validation split with IB (2,048) (r3).
  h2              H2's three conditions (S against the fresh baseline, both
                  log-normal; F5 / in-band correlation 0.5; in-band
                  displacement CV 3%, energy-gap CV 2%).
  selection       the residual selection on F5's instances (rule 1 admits S
                  on the instances H2 reads again): rel given admission
                  against rel overall.
  exploratory     how often a comparison without a true difference crosses
                  the guard (each direction), at seed noise 10-20%.

    python scripts/w9_sims.py --out records/wp9/simulations.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E1_BASE = ROOT / "records" / "wp8" / "e1" / "e1_2d_base" / "report.json"
N_IB = 2048


def guard(b, a, band=0.10, k=2.0):
    """(lower, worse) beyond the guard, arrays of trials x seeds."""
    mb, ma = b.mean(-1), a.mean(-1)
    se = np.sqrt(b.var(-1, ddof=1) / b.shape[-1] + a.var(-1, ddof=1) / a.shape[-1]) / mb
    tau = np.maximum(band, k * se)
    rel = ma / mb - 1
    return rel < -tau, rel > tau


def lognormal(rng, cv, shape):
    s = np.sqrt(np.log(1 + cv ** 2))
    return np.exp(rng.normal(-s * s / 2, s, shape))


def h1_r2_model(rng, trials=400_000) -> dict:
    out = {}
    for cv in (0.0486, 0.10):
        row = {}
        for red in (0.0, 0.10, 0.15, 0.20):
            b = lognormal(rng, cv, (trials, 3))
            a = (1 - red) * lognormal(rng, cv, (trials, 3))
            row[f"{red:g}"] = float(guard(b, a)[0].mean())
        out[f"cv_{cv:g}"] = row
    return out


def _e1_arrays():
    r = json.loads(E1_BASE.read_text())
    c = r["results"]["e8"]["metrics"]["cells"]["ar"]
    cell = c[max(c, key=int)]
    return np.stack([np.asarray(e["per_instance"]["energy_gap_rel"], float)
                     for e in cell["per_seed_eval"]])


def h1_instances(rng, trials=40_000, jit=0.08) -> dict:
    """Per-seed means are drawn in aggregate: the clean part as the profile
    mean times the seed factor with the jitter's averaged noise, plus the
    blow-ups' excess over the clean values they replace."""
    ref = _e1_arrays()                                         # (3, 256)
    nv = ref.shape[1]
    prof = ref.mean(0).copy()
    prof[233] = np.median(ref[:, 233])                         # the instance's clean level
    keep = np.arange(nv) != 233
    ref_fac = ref[:, keep].mean(1) / prof[keep].mean()         # E1's own seed factors
    pm, p2 = prof.mean(), (prof ** 2).mean() / prof.mean() ** 2

    def seed_means(n, fac, rate, rfac, pmean):
        clean = pmean * rfac * fac * (1 + rng.normal(0, jit * np.sqrt(p2 / n), fac.shape))
        k = rng.poisson(rate * n, fac.shape)
        blow = np.zeros(fac.shape)
        for kk in range(1, int(k.max()) + 1):
            blow += np.where(k >= kk, rng.uniform(0.3, 0.9, fac.shape), 0.0)
        return clean + blow / n - k * pmean * rfac * fac / n

    out = {"reference_validation_seed_means": ref.mean(1).tolist(),
           "reference_without_233": ref[:, keep].mean(1).tolist()}
    for rate in (0, 1, 2):
        for sd in (0.005, 0.02):
            rows = {"val": {}, "val_IB": {}}
            for red in (0.0, 0.05, 0.075, 0.09, 0.10, 0.11, 0.125, 0.15):
                fac = np.exp(rng.normal(0, sd, (trials, 3)))
                nval = seed_means(nv, fac, rate / 1792, 1 - red, pm)
                rval = np.broadcast_to(ref.mean(1), (trials, 3))
                rows["val"][f"{red:g}"] = float(guard(rval, nval)[0].mean())
                dib = 1 + rng.normal(0, 0.022, (trials, 1))    # IB's profile mean vs val's
                nib = seed_means(N_IB, fac, rate / 1792, 1 - red, pm * dib)
                rib = seed_means(N_IB, np.broadcast_to(ref_fac, (trials, 3)), rate / 1792, 1.0,
                                 pm * dib)
                pool_r = (nv * rval + N_IB * rib) / (nv + N_IB)
                pool_n = (nv * nval + N_IB * nib) / (nv + N_IB)
                rows["val_IB"][f"{red:g}"] = float(guard(pool_r, pool_n)[0].mean())
            out[f"rate_{rate}_per_1792_seed_sd_{sd:g}"] = rows
    return out


def h2(rng, trials=200_000) -> dict:
    def corr_ln(cv1, cv2, rho, shape):
        s1, s2 = np.sqrt(np.log(1 + cv1 ** 2)), np.sqrt(np.log(1 + cv2 ** 2))
        z1 = rng.normal(size=shape)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.normal(size=shape)
        return np.exp(s1 * z1 - s1 * s1 / 2), np.exp(s2 * z2 - s2 * s2 / 2)

    def arm(mf, mi, me, cvf):
        f, i = corr_ln(cvf, 0.03, 0.5, (trials, 3))
        return mf * f, mi * i, me * lognormal(rng, 0.02, (trials, 3))

    def run(red_f, d_inband=0.0, cvf=0.15):
        bf, bi, be = arm(0.13, 0.0634, 0.0383, cvf)
        sf, si, se = arm(0.13 * (1 - red_f), 0.0634 * (1 + d_inband), 0.0383, cvf)
        c1 = guard(bf, sf)[0]
        c2 = guard(bf / bi, sf / si)[0]
        c3 = ~(guard(be, se)[1] | guard(bi, si)[1])
        return float((c1 & c2 & c3).mean())

    out = {}
    for cvf in (0.10, 0.15, 0.20):
        out[f"f5_cv_{cvf:g}"] = {f"{r:g}": run(r, cvf=cvf) for r in (0.0, 0.2, 0.3, 0.5)}
    out["inband_minus_0.1_f5_minus_0.3_cv_0.15"] = run(0.3, d_inband=-0.10)
    return out


def selection(rng, trials=20_000, n=256) -> dict:
    """Rule 1 admits S when F5's displacement ratio (against in band 0.0634)
    is >= 1.5 and the median c* >= 1.10; a baseline error per instance is
    sqrt(A^2 + s^2) with amplitude part A = 1 - 1/c*, S removes A."""
    out = {}
    for cmed, smed in ((1.10, 0.075), (1.12, 0.07), (1.15, 0.065)):
        c = np.exp(rng.normal(np.log(cmed), 0.12, (trials, n)))
        s = smed * np.exp(rng.normal(0, 0.7, (trials, n)))
        e = np.sqrt((1 - 1 / c) ** 2 + s ** 2)
        adm = (e.mean(1) / 0.0634 >= 1.5) & (np.median(c, 1) >= 1.10)
        rel = s.mean(1) / e.mean(1) - 1
        out[f"c_star_median_{cmed:g}"] = {"p_admit": float(adm.mean()),
                                          "bias_given_admitted": float(rel[adm].mean()
                                                                       - rel.mean())}
    return out


def exploratory(rng, trials=400_000) -> dict:
    out = {}
    for cv in (0.10, 0.15, 0.20):
        lo, wo = guard(lognormal(rng, cv, (trials, 3)), lognormal(rng, cv, (trials, 3)))
        out[f"cv_{cv:g}"] = {"lower": float(lo.mean()), "worse": float(wo.mean())}
    return out


STREAMS = {"h1_r2_model": 11, "h1_instances": 12, "h2": 2, "selection": 9, "exploratory": 5}
"""One random stream per section (the first four keys are the streams the
sections first ran with; "exploratory" shared h1_r2_model's until Stage 0c's
review)."""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "records" / "wp9" / "simulations.json"))
    a = ap.parse_args()
    res = {"what": "wp9 operating characteristics (scripts/w9_sims.py)", "seed": 20261002}
    for name, fn in (("h1_r2_model", h1_r2_model), ("h1_instances", h1_instances),
                     ("h2", h2), ("selection", selection), ("exploratory", exploratory)):
        res[name] = fn(np.random.default_rng([20261002, STREAMS[name]]))
        print(name, "done", file=sys.stderr, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
