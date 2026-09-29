#!/usr/bin/env python3
"""CLI: E1 adjudication (PREREG_E1 Sec. 5; see fejepa.analysis.adjudicate).

Stage 1.28: every separation file is tied to its arm and seed through the
state path it records -- the state must live in the arm's own run directory
(beside its report), be measured on the run's validation split, and the
seeds must be exactly the report's seeds; S values are paired by seed, never
by the order the shell happened to list the files in.

Stage 1.31: each separation file must also be a valid reading (S_valid) on
the full validation split (n = the config's n_val), not a smoke run, measured
with the arm's own configuration (canonical config SHA-256 = the report's)
and on the exact state the report trained (state SHA-256 = the report's
d9_restart record for that seed) -- a state overwritten by a later rerun, or
a reading on fewer instances, is refused."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _separation_by_seed(paths: list, report_path: str, report: dict) -> list:
    run_dir = Path(report_path).resolve().parent
    n_val = int(((report.get("config") or {}).get("split") or {}).get("n_val", -1))
    cfg_sha = (report.get("provenance") or {}).get("config_sha256")
    states = (((report.get("results") or {}).get("e8") or {}).get("metrics") or {}) \
        .get("d9_restart", {}).get("ar_states", {})
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
        if d.get("smoke") or d.get("S_valid") is not True:
            raise SystemExit(f"{p}: not a valid reading (smoke={d.get('smoke')}, "
                             f"S_valid={d.get('S_valid')}: {d.get('S_invalid_reason')})")
        if int(d.get("n_instances", -1)) != n_val:
            raise SystemExit(f"{p}: measured on {d.get('n_instances')} instances; the run's "
                             f"validation split has {n_val}")
        if d.get("config_sha256") != cfg_sha:
            raise SystemExit(f"{p}: measured with configuration {str(d.get('config_sha256'))[:12]}, "
                             f"not the report's {str(cfg_sha)[:12]}")
        want_sha = (states.get(f"s{s}") or {}).get("sha256")
        if not want_sha or d.get("state_sha256") != want_sha:
            raise SystemExit(f"{p}: state SHA-256 {str(d.get('state_sha256'))[:12]} is not the "
                             f"state the report trained for seed {s} ({str(want_sha)[:12]})")
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
