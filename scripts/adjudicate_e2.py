#!/usr/bin/env python3
"""E2 adjudication (PREREG_E2 Sec. 4) -- executable, not hand-computed.

Inputs: the Phase-2 report (baseline: FE-JEPA AR cells and P3 fine zero-shot),
the E2 run report for one M (bottleneck AR cells and P3 fine zero-shot), and
the preconditions bench JSON carrying the `bottleneck<M>_fine` phase.

K1 (accuracy): in-band energy gap (AR cell, seed median) or fine zero-shot
    displacement error worse than the baseline by more than `band` => retired
    at this M.
K2 (speed): fine-scale step time not below `kill_s` (2.0 s) => no speed case.
GO: parity within the band AND fine step time below `go_s` (1.0 s).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _ar_median(report: dict, metric: str) -> float:
    cells = report["results"]["e8"]["metrics"]["cells"]["ar"]
    cell = cells[max(cells, key=int)]
    return float(statistics.median(cell[metric]["per_seed"]))


def _fine_disp(report: dict) -> float:
    return float(report["results"]["p3_transfer"]["metrics"]["ar"]["fine_disp_mean"])


def adjudicate(base: dict, e2: dict, bench: dict, m_tokens: int, band: float,
               kill_s: float, go_s: float) -> dict:
    eg_b, eg_e = _ar_median(base, "energy_gap_rel"), _ar_median(e2, "energy_gap_rel")
    fd_b, fd_e = _fine_disp(base), _fine_disp(e2)
    egap_change = eg_e / (eg_b + 1e-30) - 1.0
    fine_change = fd_e / (fd_b + 1e-30) - 1.0
    k1 = (egap_change > band) or (fine_change > band)
    phase = bench["phases"].get(f"bottleneck{m_tokens}_fine")
    step_s = float(phase["ms_per_step"]) / 1000.0 if phase else None
    k2 = step_s is None or step_s >= kill_s
    go = (not k1) and (step_s is not None) and step_s < go_s
    return {"M": m_tokens, "band": band, "egap_median_base": eg_b, "egap_median_e2": eg_e,
            "egap_rel_change": egap_change, "fine_disp_base": fd_b, "fine_disp_e2": fd_e,
            "fine_disp_rel_change": fine_change, "fine_step_s": step_s,
            "K1_accuracy": k1, "K2_speed": k2, "GO": go,
            "verdict": "GO" if go else ("KILLED" if (k1 or k2) else "NO-GO")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-report", required=True)
    ap.add_argument("--e2-report", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--kill-s", type=float, default=2.0)
    ap.add_argument("--go-s", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = adjudicate(json.loads(Path(a.base_report).read_text()),
                     json.loads(Path(a.e2_report).read_text()),
                     json.loads(Path(a.bench).read_text()), a.tokens, a.band, a.kill_s, a.go_s)
    out = Path(a.out or f"runs/wp8/e2_verdict_M{a.tokens}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("K1_accuracy", "K2_speed", "GO", "verdict", "fine_step_s")}))


if __name__ == "__main__":
    main()
