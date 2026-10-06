#!/usr/bin/env python3
"""cmame-paper Stage 2: generate CM2D's configuration (PREREG_CM2D) from E1's
stamped 2D base configuration -- never edited by hand.

`configs/cm2d_v1.json` is E1's base configuration (corpus, split, model,
seeds 0-2, schedules, numeric policy, workers, guard on) with the supervised
grid switched on beside E1's label-free states:
  e8.ar_only false         the supervised grid runs (budgets 16, 64, 256, 1,024;
                           200 epochs; learning rate 1.5e-3: E1's, unchanged)
  e8.include_anchor false  no labels+anchor row
  e8.include_ar_ft false   no fine-tuning row (it would train the reused states)
  e8.include_knorm true    the stiffness-norm row: the labels-only network trained
                           on the relative stiffness-norm error instead of the
                           relative Euclidean displacement error
  e8.mgn_budgets [64,1024] the graph network at two budgets (include_mgn: E1's true)
  e8.reuse_from            E1's three base states (pool 1,024 x 200 epochs),
                           evaluated, not trained: refused unless each file's
                           SHA-256 is the one E1's report records and the
                           configuration equals E1's outside the supervised-grid
                           keys (`supervised_grid`)
  out, prereg_file, _comment   its own output directory and PREREG_CM2D.md.

    python scripts/make_cm2d_config.py            # writes configs/cm2d_v1.json
    python scripts/make_cm2d_config.py --check    # the committed file is the generator's
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "configs/e1_2d_base.json"
BASE_SHA256 = "4dfdea42e22c0cc45c113f11408723953e29e114ec8f94668b447c2a1740a52d"
"""The canonical config SHA-256 PREREG_E1 stamped for e1_2d_base (and E1's
report records)."""
E1_REPORT = "records/wp8/e1/e1_2d_base/report.json"
E1_STATES = "runs/e1_2d_base/e8_states"
NAME = "cm2d_v1"
MGN_BUDGETS = [64, 1024]
E8_KEYS = {"ar_only": False, "include_anchor": False, "include_ar_ft": False,
           "include_knorm": True, "mgn_budgets": MGN_BUDGETS}
"""The `experiments.e8` keys CM2D sets (besides `reuse_from`)."""
CHANGED = {("experiments", "e8", k) for k in (*E8_KEYS, "reuse_from")} | \
    {("out",), ("prereg_file",), ("_comment",)}
"""Every key path in which configs/cm2d_v1.json differs from E1's base."""


def cm2d_config(base: dict) -> dict:
    cfg = json.loads(json.dumps(base))
    e8 = cfg["experiments"]["e8"]
    e8.update(json.loads(json.dumps(E8_KEYS)))
    e8["reuse_from"] = {"report": E1_REPORT, "states_dir": E1_STATES, "supervised_grid": True}
    cfg["out"] = "runs/cm2d/report.json"
    cfg["prereg_file"] = "PREREG_CM2D.md"
    cfg["_comment"] = ("CM2D: E1's base states (evaluation only) beside the supervised grid "
                       "retrained with this code -- labels-only, labels with the stiffness-norm "
                       "loss, graph network at 64 and 1,024 labels; see PREREG_CM2D.md")
    return cfg


def load_base(path: str = BASE) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from fejepa.report import config_sha256

    base = json.loads((ROOT / path).read_text())
    got = config_sha256(base)
    if got != BASE_SHA256:
        raise SystemExit(f"{path}: config SHA-256 {got[:12]}... is not E1's stamped base "
                         f"({BASE_SHA256[:12]}...)")
    return base


def render(cfg: dict) -> str:
    return json.dumps(cfg, indent=1) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "configs"))
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless the file in --out-dir is the generator's")
    a = ap.parse_args()
    f = Path(a.out_dir) / f"{NAME}.json"
    text = render(cm2d_config(load_base()))
    if a.check:
        if not f.is_file() or f.read_text() != text:
            raise SystemExit(f"not the generator's output: {f}")
        return
    f.write_text(text)
    print(f"{f}  {hashlib.sha256(text.encode()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
