#!/usr/bin/env python3
"""E1 adjudication (PREREG_E1 Sec. 5) -- executable, not hand-computed.

Inputs: the AR baseline run report and the AR+SIGReg run report (both with
E8 AR cells carrying per-seed values and P3 transfer ratios), plus the
latent-separation JSONs of every state of both arms (one per seed).

K1 (parity): at any seed, in-band disp or energy gap of the shaped arm worse
    than the AR arm by more than `band` (relative) => retired.
K2 (no effect): S(shaped) - S(AR) <= 0 at every seed => not supported.
GO: S improves at every seed AND the transfer ratio does not worsen beyond
    the band at any seed (ratios read from the reports' P3 blocks).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ar_per_seed(report: dict, metric: str) -> list:
    cells = report["results"]["e8"]["metrics"]["cells"]["ar"]
    cell = cells[max(cells, key=int)]
    return [float(v) for v in cell[metric]["per_seed"]]


def _ratio(report: dict):
    p3 = (report["results"].get("p3_transfer") or {}).get("metrics")
    if not p3:
        return None
    return float(p3["ar"]["fine_disp_mean"] / (p3["ar"]["inband_disp_mean"] + 1e-30))


def adjudicate(base: dict, shaped: dict, s_base: list, s_shaped: list, band: float) -> dict:
    out = {"band": band, "per_seed": []}
    k1 = False
    for i, (db, ds, eb, es) in enumerate(zip(_ar_per_seed(base, "disp_rel_l2"),
                                             _ar_per_seed(shaped, "disp_rel_l2"),
                                             _ar_per_seed(base, "energy_gap_rel"),
                                             _ar_per_seed(shaped, "energy_gap_rel"), strict=True)):
        worse_disp = ds / (db + 1e-30) - 1.0
        worse_egap = es / (eb + 1e-30) - 1.0
        k1 |= (worse_disp > band) or (worse_egap > band)
        out["per_seed"].append({"seed": i, "disp_rel_change": worse_disp,
                                "egap_rel_change": worse_egap})
    ds_ = [b - a for a, b in zip(s_base, s_shaped, strict=True)]
    k2 = all(d <= 0.0 for d in ds_)
    rb, rs = _ratio(base), _ratio(shaped)
    ratio_ok = (rb is None or rs is None) or (rs / (rb + 1e-30) - 1.0 <= band)
    go = (not k1) and all(d > 0.0 for d in ds_) and ratio_ok
    out.update({"S_base": s_base, "S_shaped": s_shaped, "S_delta": ds_,
                "transfer_ratio_base": rb, "transfer_ratio_shaped": rs,
                "K1_parity": k1, "K2_no_effect": k2, "GO": go,
                "verdict": "GO" if go else ("KILLED" if (k1 or k2) else "NO-GO")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-report", required=True)
    ap.add_argument("--shaped-report", required=True)
    ap.add_argument("--base-sep", nargs="+", required=True, help="latent_separation JSONs, AR arm")
    ap.add_argument("--shaped-sep", nargs="+", required=True, help="latent_separation JSONs, shaped arm")
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--out", default="runs/wp8/e1_verdict.json")
    a = ap.parse_args()
    base = json.loads(Path(a.base_report).read_text())
    shaped = json.loads(Path(a.shaped_report).read_text())
    s_b = [json.loads(Path(p).read_text())["S_silhouette"] for p in a.base_sep]
    s_s = [json.loads(Path(p).read_text())["S_silhouette"] for p in a.shaped_sep]
    res = adjudicate(base, shaped, s_b, s_s, a.band)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("K1_parity", "K2_no_effect", "GO", "verdict", "S_delta")}))


if __name__ == "__main__":
    main()
