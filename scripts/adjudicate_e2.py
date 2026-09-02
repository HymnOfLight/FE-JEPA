#!/usr/bin/env python3
"""CLI: E2 adjudication (PREREG_E2 Sec. 4; see fejepa.analysis.adjudicate)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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

    from fejepa.analysis.adjudicate import adjudicate_e2
    from fejepa.analysis.common import write_json

    load = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    res = adjudicate_e2(load(a.base_report), load(a.e2_report), load(a.bench), a.tokens,
                        a.band, a.kill_s, a.go_s)
    write_json(a.out or f"runs/wp8/e2_verdict_M{a.tokens}.json", res)
    print(json.dumps({k: res[k] for k in ("K1_accuracy", "K2_speed", "GO", "verdict", "fine_step_s")}))


if __name__ == "__main__":
    main()
