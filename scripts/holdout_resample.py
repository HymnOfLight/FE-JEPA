#!/usr/bin/env python3
"""cmame-paper Stage 11 (PREREG_HOLDOUT Sec. 5, "What a verdict licenses"):
what the rules of Sec. 5 read on sets of the holdout sets' sizes drawn from
the validation instances.

The validation arrays of the two runs (the reports' per-instance arrays, and
the per-load gaps and energies of `records/cmame/timing/fields/energies_val.npz`
and `records/cmame/spectra/export/spectra_val.npz`) are resampled, instances
with replacement, the same instances for every row and seed, the seeds kept,
and every claim's rule and the reading H3 are applied to each resample
(`fejepa.analysis.adjudicate_holdout.claims_of`). Two schemes: `direct`
draws a set of the holdout set's size from the validation instances (the
holdout's own sampling, the validation instances taken as the
distribution); `predictive` first draws 256 instances from the validation
instances, then a set of the holdout set's size from those (the validation
set's own sampling added). The record gives, per scheme and claim, the
number of resamples that read REPLICATED, and the quantiles of the
quantities the rules compare. If the validation instances are
representative of their distribution, the holdout verdicts should read as
most resamples do; a verdict that few resamples show is not a matter of
which instances were drawn. Seed-to-seed and training variation do not
enter: the states are fixed.

It also calibrates V, whether the per-instance values are distributed alike
on the holdout set and on the validation instances
(`fejepa.analysis.adjudicate_holdout.representativeness`, with the
configuration's family-wise level), on the validation instances alone.
`halves`: each set's validation instances split at random into two disjoint
halves, one in the holdout set's place; `at_holdout_sizes`: a set of the
holdout set's size and one of the validation set's size drawn, with
replacement, from the validation instances. Each scheme without a change
(false alarms: the share of draws that read NOT REPRESENTATIVE, with the
quantities that were rejected) and with the label-free transformer's four
error metrics (every metric but the critical-region recall) multiplied by
a factor on the holdout side, in both sets, eight quantities at once
(power against that change). `single_quantity`: at the holdout sets' sizes,
each quantity alone made worse on the holdout side (an error metric
multiplied by a factor, the critical-region recall divided by it), the
share of draws in which V rejects it.

    python scripts/holdout_resample.py --out records/cmame/holdout/resample.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SHIFTS = (1.05, 1.10, 1.15, 1.20)
"""The factors of V's power readings (eight quantities at once)."""
FACTORS_ONE = (1.02, 1.05, 1.10, 1.20)
"""The factors of V's readings with one quantity made worse."""
INPUTS = {"h3d": ("records/wp8/e2/baseline/report_phase2b.json",
                  "records/cmame/timing/fields/energies_val.npz", "arms"),
          "h2d": ("records/cmame/cm2d/return/report.json",
                  "records/cmame/spectra/export/spectra_val.npz", "rows")}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validation_arrays(spec: dict) -> dict:
    from fejepa.analysis.adjudicate_holdout import arrays_from_report

    out = {}
    for name, (rp, ep, key) in INPUTS.items():
        sdef = spec["sets"][name]
        if rp != sdef["report"] or _sha(ROOT / rp) != sdef["report_sha256"]:
            raise SystemExit(f"{rp}: not the configuration's report for {name}")
        with np.load(ROOT / ep, allow_pickle=False) as z:
            export = {k: z[k] for k in z.files}
        out[name] = arrays_from_report(json.loads((ROOT / rp).read_text()), list(sdef["rows"]),
                                       export, key)
    return out


def _q(x) -> dict:
    x = np.asarray(x, float)
    return {"min": float(x.min()), "p01": float(np.quantile(x, 0.01)),
            "median": float(np.median(x)), "p99": float(np.quantile(x, 0.99)),
            "max": float(x.max())}


def _draw(rng, n_val: int, size: int, scheme: str) -> np.ndarray:
    if scheme == "direct":
        return rng.integers(0, n_val, size=size)
    base = rng.integers(0, n_val, size=n_val)          # the validation set's own sampling
    return base[rng.integers(0, n_val, size=size)]


def resample(spec: dict, n: int, seed: int) -> dict:
    A = validation_arrays(spec)
    sizes = {k: int(spec["sets"][k]["n_holdout"]) for k in A}
    from fejepa.analysis.adjudicate_holdout import claims_of

    on_validation = {c: v["verdict"] for c, v in claims_of(A["h3d"], A["h2d"],
                                                           spec["criteria"])[0].items()}
    schemes = {s: one_scheme(A, sizes, spec, n, seed, s) for s in ("direct", "predictive")}
    alpha = float(spec["criteria"]["representative_alpha"])
    return {"what": "PREREG_HOLDOUT: the rules of Sec. 5 on resamples of the validation "
                    "instances of the holdout sets' sizes, and V's calibration "
                    "(scripts/holdout_resample.py)",
            "inputs": {k: {"report": v[0], "report_sha256": _sha(ROOT / v[0]),
                           "export": v[1], "export_sha256": _sha(ROOT / v[1])}
                       for k, v in INPUTS.items()},
            "sizes": sizes, "resamples": n, "seed": seed, "on_validation": on_validation,
            "schemes": schemes,
            "V": {"alpha": alpha,
                  **{s: v_calibration(A, sizes, alpha, n, n // 5, seed, s)
                     for s in ("halves", "at_holdout_sizes")},
                  "single_quantity": v_single(A, sizes, alpha, n // 5, seed)}}


def one_scheme(A: dict, sizes: dict, spec: dict, n: int, seed: int, scheme: str) -> dict:
    from fejepa.analysis.adjudicate_holdout import CLAIMS, REPLICATED, claims_of, resampled

    rng = np.random.default_rng(seed)
    counts, failing = {c: 0 for c in CLAIMS}, {}
    h3, stats = {}, {k: [] for k in ("T2_disp_lower_share_min", "T2_vm_higher_share",
                                      "T3_worse_labels", "T3_worse_mgn", "H1_rel", "H1_tau",
                                      "H2a_rel", "H2b_rel", "H2c_rel", "H3_rel")}
    for _ in range(n):
        R = {k: resampled(A[k], _draw(rng, A[k].values.shape[2], sizes[k], scheme))
             for k in A}
        claims, readings = claims_of(R["h3d"], R["h2d"], spec["criteria"])
        for c, v in claims.items():
            counts[c] += v["verdict"] == REPLICATED
            for why in v["failing"]:
                failing[why] = failing.get(why, 0) + 1
        h3[readings["H3"]["reading"]] = h3.get(readings["H3"]["reading"], 0) + 1
        stats["T2_disp_lower_share_min"].append(min(claims["T2"]["disp_lower_share_per_seed"]))
        stats["T2_vm_higher_share"].append(claims["T2"]["vm_higher_share_pooled"])
        stats["T3_worse_labels"].append(claims["T3"]["per_row"]["labels"]["worse"])
        stats["T3_worse_mgn"].append(claims["T3"]["per_row"]["mgn"]["worse"])
        for c in ("H1", "H2a", "H2b"):
            stats[f"{c}_rel"].append(claims[c]["rel_change"])
        stats["H1_tau"].append(claims["H1"]["threshold"])
        stats["H2c_rel"].append(claims["H2c"]["rel_change"])
        stats["H3_rel"].append(readings["H3"]["rel_change"])
    return {"replicated": counts, "failing": dict(sorted(failing.items())), "H3": h3,
            "quantiles": {k: _q(v) for k, v in stats.items()}}


def scaled(A, row: str, factor: float):
    """The arrays with `row`'s four error metrics (not the critical-region
    recall) multiplied by `factor`."""
    from fejepa.analysis.adjudicate_holdout import Arrays
    from fejepa.analysis.holdout import HIGHER_BETTER, METRICS

    V = A.values.copy()
    for mi, m in enumerate(METRICS):
        if m not in HIGHER_BETTER:
            V[A.rows.index(row), :, :, mi] *= factor
    d = {"rows": np.array(A.rows), "seeds": np.array(A.seeds), "metrics": np.array(METRICS),
         "values": V, "gap_load": A.gap_load, "pi_load": A.pi_load}
    return Arrays(d, dict(A.used))


def worse(A, factor: float):
    """The arrays with every row's error metrics multiplied by `factor` and
    its critical-region recall divided by it: every quantity made worse by
    the factor."""
    from fejepa.analysis.adjudicate_holdout import Arrays
    from fejepa.analysis.holdout import HIGHER_BETTER, METRICS

    V = A.values.copy()
    for mi, m in enumerate(METRICS):
        V[:, :, :, mi] *= 1.0 / factor if m in HIGHER_BETTER else factor
    d = {"rows": np.array(A.rows), "seeds": np.array(A.seeds), "metrics": np.array(METRICS),
         "values": V, "gap_load": A.gap_load, "pi_load": A.pi_load}
    return Arrays(d, dict(A.used))


def _spread(x) -> dict:
    return {"min": float(np.min(x)), "median": float(np.median(x)), "max": float(np.max(x))}


def v_single(A: dict, sizes: dict, alpha: float, n: int, seed: int) -> dict:
    """V with one quantity alone made worse on the holdout side, at the
    holdout sets' sizes (draws with replacement from the validation
    instances, as `at_holdout_sizes`): per quantity and factor, the share of
    the draws in which V rejects that quantity. Per draw, `representativeness`
    gives every quantity's p-value without a change and with every quantity
    made worse (a quantity's p-value depends on its own values only), and
    `holm` decides the family with that one quantity's p-value changed, as V
    would decide it."""
    from fejepa.analysis.adjudicate_holdout import holm, representativeness, resampled

    rng = np.random.default_rng([seed, 3])
    keys = [f"{f:.2f}" for f in FACTORS_ONE]
    hits = {}
    for _ in range(n):
        sets = {}
        for k in sorted(A):
            nv = A[k].values.shape[2]
            sets[k] = (resampled(A[k], rng.integers(0, nv, size=sizes[k])),
                       resampled(A[k], rng.integers(0, nv, size=nv)))
        p0 = {q: x["p"] for q, x in representativeness(sets, alpha)["quantities"].items()}
        for f, key in zip(FACTORS_ONE, keys, strict=True):
            pf = {q: x["p"] for q, x in representativeness(
                {k: (worse(H, f), W) for k, (H, W) in sets.items()}, alpha)["quantities"].items()}
            for q in p0:
                hits.setdefault(q, dict.fromkeys(keys, 0))[key] += holm({**p0, q: pf[q]}, alpha)[q]
    detected = {q: {key: c / n for key, c in d.items()} for q, d in sorted(hits.items())}
    recall = [q for q in detected if q.endswith(":crit_recall")]
    errors = [q for q in detected if q not in recall]
    return {"sizes": {k: {"holdout": sizes[k], "validation": int(A[k].values.shape[2])}
                      for k in sorted(A)},
            "draws": n, "factors": keys, "detected": detected,
            "summary": {"error_metrics": {key: _spread([detected[q][key] for q in errors])
                                          for key in keys},
                        "crit_recall": {key: _spread([detected[q][key] for q in recall])
                                        for key in keys}}}


def v_calibration(A: dict, sizes: dict, alpha: float, n_null: int, n_shift: int, seed: int,
                  scheme: str) -> dict:
    """V on the validation instances alone (`halves` or `at_holdout_sizes`):
    without a change, the share of draws that read NOT REPRESENTATIVE and the
    quantities rejected; with the label-free rows' error metrics multiplied
    on the holdout side, per factor, the share that read NOT REPRESENTATIVE."""
    from fejepa.analysis.adjudicate_holdout import (NOT_REPRESENTATIVE, representativeness,
                                                    resampled)

    rng = np.random.default_rng([seed, 1 if scheme == "halves" else 2])

    def draw(factor: float) -> dict:
        sets = {}
        for k in sorted(A):
            n = A[k].values.shape[2]
            if scheme == "halves":
                perm = rng.permutation(n)
                ih, iv = perm[:n // 2], perm[n // 2:]
            else:
                ih, iv = rng.integers(0, n, size=sizes[k]), rng.integers(0, n, size=n)
            H = resampled(A[k], ih)
            sets[k] = (scaled(H, "ar", factor) if factor != 1.0 else H, resampled(A[k], iv))
        return representativeness(sets, alpha)

    alarms, rejected = 0, {}
    for _ in range(n_null):
        r = draw(1.0)
        alarms += r["verdict"] == NOT_REPRESENTATIVE
        for q in r["rejected"]:
            rejected[q] = rejected.get(q, 0) + 1
    power = {}
    for f in SHIFTS:
        power[f"{f:.2f}"] = sum(draw(f)["verdict"] == NOT_REPRESENTATIVE
                                for _ in range(n_shift)) / n_shift
    n_of = {k: A[k].values.shape[2] for k in sorted(A)}
    return {"sizes": {k: {"holdout": n // 2 if scheme == "halves" else sizes[k],
                          "validation": n - n // 2 if scheme == "halves" else n}
                      for k, n in n_of.items()},
            "draws": n_null, "false_alarm": alarms / n_null,
            "rejected": dict(sorted(rejected.items())),
            "draws_per_factor": n_shift, "power": power}


def main() -> None:
    from fejepa.analysis.holdout import load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "holdout_v1.json"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = resample(load_spec(a.config), a.n, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"on_validation": res["on_validation"],
                      **{s: {k: v[k] for k in ("replicated", "H3")}
                         for s, v in res["schemes"].items()},
                      "V": {**{s: {k: res["V"][s][k] for k in ("false_alarm", "power")}
                               for s in ("halves", "at_holdout_sizes")},
                            "single_quantity": res["V"]["single_quantity"]["summary"]}},
                     indent=1))


if __name__ == "__main__":
    main()
