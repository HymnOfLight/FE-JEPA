#!/usr/bin/env python3
"""wp9 Stage 0b: generate the wp9 configurations (PREREG_W9) from E1's
stamped 2D base configuration -- never edited by hand.

Every arm is E1's base configuration (corpus, split, model, schedule, numeric
policy, workers) with:
  * the evaluation block: the in-band holdout IB (PREREG_W9 r3), the OOD-2D v1
    families F1-F5 and the remesh set R as evaluation-only holdouts
    (runs/w9/ood2d/<name>, verified by manifest at run start), and the
    amplitude readings on every evaluation;
  * its own output directory and the guard on PREREG_W9.md;
and, per arm:
  w9_c1_n1024   no training: E1's base states (seeds 0-2, SHA-verified against
                E1's report, `e8.reuse_from`) evaluated on the wp9 suite
  w9_c1_n4096   pool 4,096 x 50 epochs, seeds 0-2
  w9_c1_n25600  pool 25,600 x 8 epochs, seeds 0-2    } C1's largest pool: one of
  w9_c1_n12800  pool 12,800 x 16 epochs, seeds 0-2   } the two, by rule 2
  w9_b_n1024    pool 1,024 x 200 epochs, seeds 3-5: E1's base retrained with
                fresh seeds -- H2's reference, independent of the states rule 1
                read
  w9_s_n1024    as w9_b_n1024 with S: decode_scale "l1" times S_FACTOR, and
                features.load_density
(the fresh baseline runs in every session 2 since PREREG_W9 r3; S only if
session 1's rule 1 admits it). Every arm trains (or trained) 204,800 AR steps
per seed. Workers (rule 2) and activation
checkpointing (rule 3) are run-time settings, not configuration
(`run-config --workers`, `--activation-checkpointing`).

S_FACTOR = 1/64 sets the level of S's decoded output: the median of
max|F| / sum|F| over the first 256 instances of E1's training corpus is
0.0151 (10th-90th percentile 0.0096-0.0219; scripts/w9_scale_factor.py,
records/wp9/scale_factor.json), so at training mesh sizes S's network is asked
for outputs of about the level of E1's; on finer meshes E1's scale shrinks
with the element size and S's does not.

    python scripts/make_w9_configs.py            # writes configs/w9_*.json
    python scripts/make_w9_configs.py --check    # the committed files are the generator's
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
OOD_DIR = "runs/w9/ood2d"
HOLDOUTS = ("IB", "F1", "F2", "F3", "F4", "F5", "R")
STEPS_PER_SEED = 204800
S_FACTOR = 0.015625
FRESH_SEED_OFFSET = 3
ARMS = {
    "w9_c1_n1024": {"pool": 1024, "epochs": 200, "reuse": True, "s": False, "seed_offset": 0},
    "w9_c1_n4096": {"pool": 4096, "epochs": 50, "reuse": False, "s": False, "seed_offset": 0},
    "w9_c1_n25600": {"pool": 25600, "epochs": 8, "reuse": False, "s": False, "seed_offset": 0},
    "w9_c1_n12800": {"pool": 12800, "epochs": 16, "reuse": False, "s": False, "seed_offset": 0},
    "w9_b_n1024": {"pool": 1024, "epochs": 200, "reuse": False, "s": False,
                   "seed_offset": FRESH_SEED_OFFSET},
    "w9_s_n1024": {"pool": 1024, "epochs": 200, "reuse": False, "s": True,
                   "seed_offset": FRESH_SEED_OFFSET},
}


def evaluation_block() -> dict:
    return {"holdouts": {h: {"dir": f"{OOD_DIR}/{h}", "family": h} for h in HOLDOUTS},
            "amplitude": True}


def w9_config(base: dict, arm: str, spec: dict | None = None,
              steps: int = STEPS_PER_SEED) -> dict:
    """The wp9 configuration `arm` from E1's base (`spec`/`steps`: a miniature's
    pool sizes and step budget, for tests and rehearsals)."""
    spec = spec or ARMS[arm]
    if spec["pool"] * spec["epochs"] != steps:
        raise AssertionError(f"{arm}: {spec['pool']} x {spec['epochs']} != {steps}")
    cfg = json.loads(json.dumps(base))
    e8 = cfg["experiments"]["e8"]
    e8["pool_sizes"], e8["ar_epochs"] = [spec["pool"]], spec["epochs"]
    if spec.get("seed_offset", 0):
        e8["seed_offset"] = int(spec["seed_offset"])
    if spec["reuse"]:
        e8["reuse_from"] = {"report": E1_REPORT, "states_dir": E1_STATES}
    if spec["s"]:
        cfg["model"] = dict(cfg["model"], decode_scale="l1", decode_scale_factor=S_FACTOR,
                            features=dict(cfg["model"]["features"], load_density=True))
    cfg["evaluation"] = evaluation_block()
    cfg["out"] = f"runs/w9/{arm[3:]}/report.json"
    cfg["prereg_file"], cfg["prereg_guard"] = "PREREG_W9.md", True
    n = int(e8.get("seeds", 3))
    off = int(spec.get("seed_offset", 0))
    what = ("E1's base states, evaluation only" if spec["reuse"] else
            f"AR on pool[:{spec['pool']}] x {spec['epochs']} epochs, seeds {off}-{off + n - 1}"
            + (", with S (decode_scale l1 x 1/64, load densities)" if spec["s"] else ""))
    cfg["_comment"] = f"wp9 arm {arm[3:]}: {what}; see PREREG_W9.md"
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
                    help="exit non-zero unless the files in --out-dir are the generator's")
    a = ap.parse_args()
    base = load_base()
    out = Path(a.out_dir)
    bad = []
    for arm in ARMS:
        f = out / f"{arm}.json"
        text = render(w9_config(base, arm))
        if a.check:
            if not f.is_file() or f.read_text() != text:
                bad.append(str(f))
            continue
        f.write_text(text)
        print(f"{f}  {hashlib.sha256(text.encode()).hexdigest()[:12]}")
    if bad:
        raise SystemExit(f"not the generator's output: {bad}")


if __name__ == "__main__":
    main()
