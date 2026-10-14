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
    return {"what": "PREREG_HOLDOUT: the rules of Sec. 5 on resamples of the validation "
                    "instances of the holdout sets' sizes (scripts/holdout_resample.py)",
            "inputs": {k: {"report": v[0], "report_sha256": _sha(ROOT / v[0]),
                           "export": v[1], "export_sha256": _sha(ROOT / v[1])}
                       for k, v in INPUTS.items()},
            "sizes": sizes, "resamples": n, "seed": seed, "on_validation": on_validation,
            "schemes": schemes}


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
            if v["verdict"] != REPLICATED:
                for why in _failing(c, v):
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


def _failing(claim: str, v: dict) -> list:
    """The comparisons that a claim that did not replicate failed (the
    resamples in which each failed are counted)."""
    if claim in ("T1", "D2"):
        out = []
        for m, r in v["per_metric"].items():
            seeds = r.get("better_in_seed") or r.get("holds_in_seed")
            mean = r.get("better_in_mean", r.get("holds_in_mean"))
            out += [f"{claim} {m} seed {i}" for i, ok in enumerate(seeds) if not ok]
            out += [f"{claim} {m} seed mean"] if not mean else []
        return out
    if claim == "T2":
        return [f"T2 {k}" for k in ("disp_ok", "vm_ok") if not v.get(k)]
    if claim == "T3":
        return [f"T3 {r}" for r, w in v.get("per_row", {}).items()
                if not (w["worse"] >= 1 and w["sign_mismatch"] == 0)]
    return [claim]


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
                         for s, v in res["schemes"].items()}}, indent=1))


if __name__ == "__main__":
    main()
