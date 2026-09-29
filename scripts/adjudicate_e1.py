#!/usr/bin/env python3
"""CLI: E1 adjudication (PREREG_E1 Sec. 5; see fejepa.analysis.adjudicate).

Stage 1.28: every separation file is tied to its arm and seed through the
state path it records -- the state must live in the arm's own run directory
(beside its report), be measured on the run's validation split, and the
seeds must be exactly the report's seeds; S values are paired by seed, never
by the order the shell happened to list the files in."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _separation_by_seed(paths: list, report_path: str, report: dict) -> list:
    run_dir = Path(report_path).resolve().parent
    rows = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        st = str(d.get("state") or "")
        m = re.search(r"_s(\d+)\.pt$", st)
        if not m:
            raise SystemExit(f"{p}: the recorded state path {st!r} names no seed")
        if Path(st).resolve().parent.parent != run_dir:
            raise SystemExit(f"{p}: state {st} does not belong to the run of {report_path}")
        if d.get("subset") != "val":
            raise SystemExit(f"{p}: measured on subset {d.get('subset')!r}; PREREG_E1 fixes 'val'")
        s = int(m.group(1))
        if s in rows:
            raise SystemExit(f"{p}: seed {s} given twice")
        rows[s] = d["S_silhouette"]
    want = [int(x) for x in (report.get("provenance") or {}).get("seeds", [])]
    if sorted(rows) != want:
        raise SystemExit(f"separation seeds {sorted(rows)} != the report's seeds {want}")
    return [rows[s] for s in want]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-report", required=True)
    ap.add_argument("--shaped-report", required=True)
    ap.add_argument("--base-sep", nargs="+", required=True)
    ap.add_argument("--shaped-sep", nargs="+", required=True)
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--out", default="runs/wp8/e1_verdict.json")
    a = ap.parse_args()

    from fejepa.analysis.adjudicate import adjudicate_e1
    from fejepa.analysis.common import inputs_provenance, write_json

    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    base, shaped = load(a.base_report), load(a.shaped_report)
    res = adjudicate_e1(base, shaped,
                        _separation_by_seed(a.base_sep, a.base_report, base),
                        _separation_by_seed(a.shaped_sep, a.shaped_report, shaped), a.band)
    res["inputs_sha256"] = inputs_provenance([a.base_report, a.shaped_report, *a.base_sep, *a.shaped_sep])
    write_json(a.out, res)
    print(json.dumps({k: res[k] for k in ("K1_parity", "K2_no_effect", "GO", "verdict", "S_delta")}))


if __name__ == "__main__":
    main()
