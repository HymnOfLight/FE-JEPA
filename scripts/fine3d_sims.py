#!/usr/bin/env python3
"""fine3d (PREREG_FINE3D Sec. 6): properties of the R2a and R3 rules by
simulation, recorded before the stamp (records/wp9/fine3d/simulations.json).

Both rules compare seed means of three seeds per arm with the E-series
guard (rel = mean(new)/mean(ref) - 1 beyond tau = max(10%, 2 x SE_rel),
fejepa.analysis.adjudicate_w9.guard). Seed values are drawn log-normal
around each arm's true mean with the coefficients of variation given; the
reference's are the sample coefficients of variation of the runs the rule
compares against.

  r2a   E2's M = 1,024 run (records/wp8/e2/e2_m1024/report.json, seeds 0-2):
        P3's fine and in-band displacement errors and E8's in-band energy gap
        per seed; S's spread equal to the baseline's, or 1.5 times it. A
        seed's fine and in-band values are drawn independently; K1's in-band
        displacement is the draw that enters the ratio. Scenarios: S lowers
        the fine error by d (0-70%), with the in-band errors unchanged, 10%
        lower, or 10% higher. Reported: P(SUPPORTED), and P(R2b recommended)
        (SUPPORTED and the fine error's rel <= -50%).
  r3    the labelled few-shot fine-tuning at 16 (the Phase-2b report, P3,
        seeds 0-2): the fine energy gap (R3-E) and von Mises error (R3-V);
        both arms drawn afresh in every trial; the energy-fine-tuned arm's
        spread equal to the reference's, 1.5 times it, or 50% (the zero-shot
        AR states' fine energy gaps spread by 72%). Scenarios: a true
        reduction of d (0-50%). Reported: P(SUPPORTED).
  r2a_sensitivity  as r2a (in-band unchanged), with both arms' spreads twice
        E2's, or those of E2's M = 512 run (records/wp8/e2/e2_m512/report.json).
  r2a_correlated   as r2a (E2's spreads, in-band unchanged), with a seed's
        fine and in-band displacement errors correlated (rho), and its
        in-band displacement error and energy gap likewise (rho; rho^2
        between the fine error and the energy gap).
  r3_fixed_reference  the reference held at its observed per-seed values
        (the design: R3 compares with one recorded run), the record lying at
        its expectation or 5% or 10% above it (the arms' common true mean
        the record divided by 1, 1.05 or 1.10); the new arm drawn as in r3.
        With `tau_floor`: the part of tau the reference's own spread fixes
        (2 x its sample SD / sqrt(3) / its mean).

    python scripts/fine3d_sims.py --out records/wp9/fine3d/simulations.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TRIALS = 200_000
SEED = 20261015
E2 = "records/wp8/e2/e2_m1024/report.json"
E2_M512 = "records/wp8/e2/e2_m512/report.json"
P2B = "records/wp8/e2/baseline/report_phase2b.json"
GUARD_BAND, GUARD_K = 0.10, 2.0
R2B_FINE_REL_MAX = -0.5


def cv(x) -> float:
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / x.mean())


def draw(rng, mean, c, shape):
    """Log-normal seed values with mean `mean` and coefficient of variation c."""
    s2 = np.log1p(c * c)
    return mean * np.exp(rng.normal(-s2 / 2, np.sqrt(s2), size=shape))


def from_normal(z, mean, c):
    """Log-normal values from standard normals z (mean `mean`, CV c)."""
    s2 = np.log1p(c * c)
    return mean * np.exp(np.sqrt(s2) * z - s2 / 2)


def guard_vec(b, a):
    """Vectorised guard over trials (rows): (lower, worse, rel)."""
    mb, ma = b.mean(1), a.mean(1)
    se = np.sqrt(b.var(1, ddof=1) / b.shape[1] + a.var(1, ddof=1) / a.shape[1]) / mb
    tau = np.maximum(GUARD_BAND, GUARD_K * se)
    rel = ma / mb - 1
    return rel < -tau, rel > tau, rel


def _r2a_rule(fb, ib, eb, fs, is_, es) -> dict:
    lo_f, _, rel_f = guard_vec(fb, fs)
    lo_r, _, _ = guard_vec(fb / ib, fs / is_)
    _, wo_e, _ = guard_vec(eb, es)
    _, wo_d, _ = guard_vec(ib, is_)
    sup = lo_f & lo_r & ~wo_e & ~wo_d
    return {"p_supported": float(sup.mean()),
            "p_r2b_recommended": float((sup & (rel_f <= R2B_FINE_REL_MAX)).mean())}


def r2a_sim(rng, cvs: dict, d_fine: float, d_in: float, s_mult: float, n=TRIALS) -> dict:
    shape = (n, 3)
    fb = draw(rng, 1.0, cvs["fine"], shape)
    ib = draw(rng, 1.0, cvs["inband_disp"], shape)
    eb = draw(rng, 1.0, cvs["inband_egap"], shape)
    fs = draw(rng, 1.0 - d_fine, s_mult * cvs["fine"], shape)
    is_ = draw(rng, 1.0 + d_in, s_mult * cvs["inband_disp"], shape)
    es = draw(rng, 1.0 + d_in, s_mult * cvs["inband_egap"], shape)
    return _r2a_rule(fb, ib, eb, fs, is_, es)


def r2a_corr_sim(rng, cvs: dict, d_fine: float, rho: float, n=TRIALS) -> dict:
    """E2's spreads in both arms, in-band unchanged; within a seed the
    normals of (fine, in-band displacement, in-band energy gap) are
    correlated with an AR(1) matrix (rho, rho, rho^2)."""
    c = np.array([[1.0, rho, rho * rho], [rho, 1.0, rho], [rho * rho, rho, 1.0]])
    chol = np.linalg.cholesky(c)
    arms = []
    for mean_f in (1.0, 1.0 - d_fine):
        z = rng.standard_normal((n, 3, 3)) @ chol.T
        arms.append((from_normal(z[..., 0], mean_f, cvs["fine"]),
                     from_normal(z[..., 1], 1.0, cvs["inband_disp"]),
                     from_normal(z[..., 2], 1.0, cvs["inband_egap"])))
    (fb, ib, eb), (fs, is_, es) = arms
    return _r2a_rule(fb, ib, eb, fs, is_, es)


def r3_sim(rng, c_ref: float, c_new: float, d: float, n=TRIALS) -> float:
    b = draw(rng, 1.0, c_ref, (n, 3))
    a = draw(rng, 1.0 - d, c_new, (n, 3))
    lower, _, _ = guard_vec(b, a)
    return float(lower.mean())


def r3_fixed_sim(rng, ref, c_new: float, above: float, d: float, n=TRIALS) -> float:
    """The reference held at its observed values; the arms' common true mean
    is the record's mean over (1 + above); the new arm's true mean is that
    times (1 - d)."""
    ref = np.asarray(ref, float)
    b = np.broadcast_to(ref, (n, ref.size))
    a = draw(rng, ref.mean() / (1.0 + above) * (1.0 - d), c_new, (n, ref.size))
    lower, _, _ = guard_vec(b, a)
    return float(lower.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    e2 = json.loads((ROOT / E2).read_text())
    m512 = json.loads((ROOT / E2_M512).read_text())
    p2b = json.loads((ROOT / P2B).read_text())

    def spreads(rep):
        p3 = rep["results"]["p3_transfer"]["metrics"]["ar"]
        e8 = rep["results"]["e8"]["metrics"]["cells"]["ar"]["1024"]
        return {"fine": cv(p3["fine"]["disp_rel_l2"]["per_seed"]),
                "inband_disp": cv(p3["inband"]["disp_rel_l2"]["per_seed"]),
                "inband_egap": cv(e8["energy_gap_rel"]["per_seed"])}
    cvs, cvs512 = spreads(e2), spreads(m512)
    ft = p2b["results"]["p3_transfer"]["metrics"]["fewshot"]["16"]["finetune"]
    zs = p2b["results"]["p3_transfer"]["metrics"]["ar"]["fine"]
    r3cv = {"energy_gap_rel": cv(ft["energy_gap_rel"]["per_seed"]),
            "vm_rel_l2": cv(ft["vm_rel_l2"]["per_seed"]),
            "zero_shot_energy_gap_rel": cv(zs["energy_gap_rel"]["per_seed"])}
    rng = np.random.default_rng(SEED)
    r2a = {}
    for s_mult in (1.0, 1.5):
        for d_in, tag in ((0.0, "inband unchanged"), (-0.1, "inband 10% lower"),
                          (0.1, "inband 10% higher")):
            for d in (0.0, 0.2, 0.3, 0.5, 0.7):
                r2a[f"S spread x{s_mult}; {tag}; fine {-d:+.0%}"] = \
                    r2a_sim(rng, cvs, d, d_in, s_mult, a.trials)
    r3 = {}
    for metric in ("energy_gap_rel", "vm_rel_l2"):
        for c_new, tag in ((r3cv[metric], "x1"), (1.5 * r3cv[metric], "x1.5"), (0.5, "0.5")):
            for d in (0.0, 0.1, 0.2, 0.3, 0.5):
                r3[f"{metric}; new-arm cv {tag}; {-d:+.0%}"] = \
                    r3_sim(rng, r3cv[metric], c_new, d, a.trials)
    sens = {}
    for name, sp in (("E2 spreads x2", {k: 2 * v for k, v in cvs.items()}),
                     ("E2 M = 512 spreads", cvs512)):
        for d in (0.0, 0.3, 0.5, 0.7):
            sens[f"{name}; fine {-d:+.0%}"] = r2a_sim(rng, sp, d, 0.0, 1.0, a.trials)
    corr = {}
    for rho in (0.5, 0.8):
        for d in (0.0, 0.2, 0.3, 0.5, 0.7):
            corr[f"rho {rho}; fine {-d:+.0%}"] = r2a_corr_sim(rng, cvs, d, rho, a.trials)
    fixed, floors = {}, {}
    for metric in ("energy_gap_rel", "vm_rel_l2"):
        ref = [float(x) for x in ft[metric]["per_seed"]]
        floors[metric] = float(GUARD_K * np.std(ref, ddof=1) / np.sqrt(len(ref)) / np.mean(ref))
        for above in (0.0, 0.05, 0.10):
            for c_new, tag in ((r3cv[metric], "x1"), (1.5 * r3cv[metric], "x1.5"), (0.5, "0.5")):
                for d in (0.0, 0.2, 0.3, 0.5):
                    fixed[f"{metric}; record {above:+.0%}; new-arm cv {tag}; {-d:+.0%}"] = \
                        r3_fixed_sim(rng, ref, c_new, above, d, a.trials)
    res = {"what": "fine3d: properties of the R2a and R3 rules by simulation "
                   "(scripts/fine3d_sims.py); log-normal seed values, three seeds per arm",
           "trials": a.trials, "seed": SEED, "guard": {"band": GUARD_BAND, "k": GUARD_K},
           "r2b_fine_rel_max": R2B_FINE_REL_MAX,
           "r2a_cv": cvs, "r2a_cv_m512": cvs512, "r3_cv": r3cv, "r2a": r2a, "r3": r3,
           "r2a_sensitivity": sens, "r2a_correlated": corr,
           "r3_tau_floor": floors, "r3_fixed_reference": fixed}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
