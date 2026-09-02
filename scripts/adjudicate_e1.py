#!/usr/bin/env python3
"""CLI: E1 adjudication (PREREG_E1 Sec. 5; see fejepa.analysis.adjudicate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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
    from fejepa.analysis.common import write_json

    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    res = adjudicate_e1(load(a.base_report), load(a.shaped_report),
                        [load(p)["S_silhouette"] for p in a.base_sep],
                        [load(p)["S_silhouette"] for p in a.shaped_sep], a.band)
    write_json(a.out, res)
    print(json.dumps({k: res[k] for k in ("K1_parity", "K2_no_effect", "GO", "verdict", "S_delta")}))


if __name__ == "__main__":
    main()
