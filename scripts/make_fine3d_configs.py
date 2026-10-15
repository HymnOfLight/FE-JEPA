#!/usr/bin/env python3
"""fine3d (PREREG_FINE3D Sec. 2-3): generate the four configurations of the
pre-registration -- never edited by hand.

R2a (S on the token bottleneck), from E2's stamped M = 1,024 configuration
(configs/e2_m1024.json; refused unless its canonical SHA-256 is PREREG_E2's
CONFIG_SHA256[e2_m1024]):
  r2a_b_m1024   E2's M = 1,024 arm trained again on fresh seeds 3-5
                (`e8.seed_offset = 3`): S's reference
  r2a_s_m1024   as r2a_b_m1024 with S: decode_scale "l1" times S3_FACTOR, and
                features.load_density
Both keep everything else of E2's configuration (corpus, split, the 1,024-
instance training prefix, 200 epochs at 1e-3, AR only, P3 zero-shot only,
TF32, one worker) and add only their own output directory, the guard on
PREREG_FINE3D.md and a comment.

R3 (fine-tuning at the fine mesh; both arms run by scripts/r3_finetune.py):
  r3_eft_b16    the Phase-2b AR states (seeds 0-2), each fine-tuned with the
                AR objective (label-free) on the first 16 instances of the fine
                few-shot prefix with the labelled few-shot fine-tuning's
                schedule (read from the Phase-2b report's configuration: 50
                epochs at lr 1.5e-3), evaluated on the fine evaluation set and
                the in-band validation split
  r3_lft_b16    the same with the displacement labels: P3's labelled few-shot
                fine-tuning at 16 run again in the same session, R3's reference

S3_FACTOR sets the level of S's decoded output in 3D: the power of two
nearest the median of max|F| / sum|F| over 256 fresh instances of the
training family (scripts/fine3d_scale_factor.py,
records/wp9/fine3d/scale_factor_3d.json), the rule that gave wp9's 1/64.

    python scripts/make_fine3d_configs.py            # writes the four configs
    python scripts/make_fine3d_configs.py --check    # the committed files are the generator's
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "configs/e2_m1024.json"
BASE_SHA256 = "526a5688eb49f3332e6f0ced926dcd21227f2bebe7e419f10e9a3aeae0b0b088"
"""PREREG_E2's stamped CONFIG_SHA256[e2_m1024] (canonical: fejepa.report.config_sha256)."""
PHASE2B_REPORT = "records/wp8/e2/baseline/report_phase2b.json"
PHASE2B_REPORT_SHA256 = "320b6db5060ecae9f4747327228c7705d30877bece67d09af2f68d6c466f2794"
"""The Phase-2b report, as PREREG_E2 Sec. 2 records it (the box's
runs/phase2/report_phase2b.json, copied byte for byte into records/wp8)."""
PHASE2B_STATES = "runs/phase2/e8_states"
PREREG_FILE = "PREREG_FINE3D.md"
OUT_ROOT = "runs/fine3d"
S3_FACTOR = 0.0009765625
"""2^-10: the power of two nearest the median 0.00113 of max|F| / sum|F| over
256 fresh training-family instances (records/wp9/fine3d/scale_factor_3d.json,
`sample.factor`; a test keeps the two equal)."""
FRESH_SEED_OFFSET = 3
R2A_ARMS = {"r2a_b_m1024": {"s": False}, "r2a_s_m1024": {"s": True}}
R3_ARMS = {"r3_eft_b16": {"loss": "ar"}, "r3_lft_b16": {"loss": "labels"}}
R3 = {"n_train": 16, "seeds": [0, 1, 2], "n_check": 8, "max_rel_dev": 1e-3}
"""R3's own settings: the training budget (the labelled fine-tuning's
smallest, P3 budget 16), the seeds (the Phase-2b states'), and the content
check (the first n_check fine evaluation and in-band validation instances
must reproduce the report's arrays to max_rel_dev: a wrong state or code path
shows deviations of order one; the transformer's states reproduced their
report's arrays exactly in wp8's post-hoc pass)."""
ARMS = (*R2A_ARMS, *R3_ARMS)


def r2a_config(base: dict, arm: str) -> dict:
    spec = R2A_ARMS[arm]
    cfg = json.loads(json.dumps(base))
    cfg["experiments"]["e8"]["seed_offset"] = FRESH_SEED_OFFSET
    if spec["s"]:
        cfg["model"] = dict(cfg["model"], decode_scale="l1", decode_scale_factor=S3_FACTOR,
                            features=dict(cfg["model"]["features"], load_density=True))
    cfg["out"] = f"{OUT_ROOT}/{arm}/report.json"
    cfg["prereg_file"], cfg["prereg_guard"] = PREREG_FILE, True
    n = int(cfg["experiments"]["e8"]["seeds"])
    what = (f"E2's M = 1,024 bottleneck, AR on pool[:1024] x 200 epochs, seeds "
            f"{FRESH_SEED_OFFSET}-{FRESH_SEED_OFFSET + n - 1}"
            + (f", with S (decode_scale l1 x {S3_FACTOR!r}, load densities)" if spec["s"]
               else ", fresh baseline"))
    cfg["_comment"] = f"fine3d R2a arm {arm}: {what}; see {PREREG_FILE}"
    return cfg


def r3_config(report_cfg: dict, arm: str = "r3_eft_b16") -> dict:
    """An R3 arm's configuration; the schedule is the labelled few-shot
    fine-tuning's as the Phase-2b report's configuration records it. The two
    arms differ in their loss, output directory and comment only."""
    p3 = report_cfg["experiments"]["p3_transfer"]
    if R3["n_train"] not in [int(b) for b in p3["fewshot_budgets"]]:
        raise SystemExit(f"the Phase-2b few-shot budgets {p3['fewshot_budgets']} do not hold "
                         f"{R3['n_train']}")
    loss = R3_ARMS[arm]["loss"]
    what = ("fine-tuned label-free (the AR objective)" if loss == "ar" else
            "fine-tuned with the displacement labels (P3's labelled few-shot fine-tuning "
            "run again in the same session: R3's reference)")
    return {
        "_comment": (f"fine3d R3: the Phase-2b AR states {what} on the first 16 instances of "
                     "the fine few-shot prefix with the labelled few-shot fine-tuning's "
                     f"schedule; see {PREREG_FILE}"),
        "source": {"report": PHASE2B_REPORT, "report_sha256": PHASE2B_REPORT_SHA256,
                   "states_dir": PHASE2B_STATES},
        "train": {"set": "fine_fewshot_prefix", "n": R3["n_train"], "loss": loss,
                  "epochs": int(p3["fewshot_epochs"]), "lr": float(p3["fewshot_lr"]),
                  "weight_decay": 1e-4, "clip": 1.0, "warmup_frac": 0.05},
        "seeds": list(R3["seeds"]),
        "evaluation": {"fine": int(report_cfg["data_transfer"]["split"]["n_eval"]),
                       "inband": int(report_cfg["split"]["n_val"]), "amplitude": True},
        "content_check": {"n_fine": R3["n_check"], "n_inband": R3["n_check"],
                          "max_rel_dev": R3["max_rel_dev"]},
        "out": f"{OUT_ROOT}/{arm}/report.json",
        "prereg_file": PREREG_FILE, "prereg_guard": True, "device": "auto"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(path: str = BASE) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from fejepa.report import config_sha256

    base = json.loads((ROOT / path).read_text())
    got = config_sha256(base)
    if got != BASE_SHA256:
        raise SystemExit(f"{path}: config SHA-256 {got[:12]}... is not PREREG_E2's stamped "
                         f"e2_m1024 ({BASE_SHA256[:12]}...)")
    return base


def load_phase2b(path: str = PHASE2B_REPORT) -> dict:
    p = ROOT / path
    got = _sha256(p)
    if got != PHASE2B_REPORT_SHA256:
        raise SystemExit(f"{path}: SHA-256 {got[:12]}... is not the Phase-2b report's "
                         f"({PHASE2B_REPORT_SHA256[:12]}...)")
    return json.loads(p.read_text())


def config_of(arm: str, base: dict | None = None, report: dict | None = None) -> dict:
    if arm in R2A_ARMS:
        return r2a_config(base if base is not None else load_base(), arm)
    if arm in R3_ARMS:
        return r3_config((report if report is not None else load_phase2b())["config"], arm)
    raise KeyError(arm)


def render(cfg: dict) -> str:
    return json.dumps(cfg, indent=1) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "configs"))
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero unless the files in --out-dir are the generator's")
    a = ap.parse_args()
    base, report = load_base(), load_phase2b()
    out = Path(a.out_dir)
    bad = []
    for arm in ARMS:
        f = out / f"{arm}.json"
        text = render(config_of(arm, base, report))
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
