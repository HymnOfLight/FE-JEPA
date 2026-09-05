#!/usr/bin/env python3
"""Generate the E-series configurations from the stamped Phase-1/Phase-2
configurations (wp8-lejepa, PREREG_E1 / PREREG_E2 mechanics).

Every E-series run is a `run-config` on a derived configuration:
  * E2 (M = 512 and 1024): the Phase-2 configuration with model.kind =
    bottleneck, model.n_tokens = M, E8 as AR pretraining only, P3 in
    zero-shot-only form, and the FE-JEPA-only probes (e6, wp6, e1) disabled.
  * E1 (2D, base and shaped): the Phase-1 configuration with E8 as AR
    pretraining only and everything else disabled; the shaped arm carries
    `pretrain.loss_spec` with lambda / head width to be FILLED at stamping
    (placeholders are null and the runner refuses to start on them).
Each configuration points at its own PREREG file with the guard enabled, so
the E-series inherits the stamping mechanics of Phase 2 unchanged.

    python scripts/make_e_series_configs.py [--out-dir configs]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

E_SERIES_OFF = ("e1", "e2", "e3", "e4", "e5", "e6", "e7", "wp2", "wp6")


def _disable_all_but_e8(exps: dict) -> dict:
    out = {}
    for k, v in exps.items():
        if k == "e8":
            out[k] = dict(v, enabled=True, ar_only=True)
        elif k == "p3_transfer":
            out[k] = dict(v, enabled=True, fewshot_budgets=[], naive_budget=0)
        else:
            out[k] = dict(v or {}, enabled=False)
    return out


def e2_config(phase2: dict, m_tokens: int) -> dict:
    cfg = json.loads(json.dumps(phase2))
    cfg["model"] = dict(cfg["model"], kind="bottleneck", n_tokens=int(m_tokens))
    cfg["experiments"] = _disable_all_but_e8(cfg["experiments"])
    cfg["out"] = f"runs/e2_m{m_tokens}/report.json"
    cfg["prereg_file"], cfg["prereg_guard"] = "PREREG_E2.md", True
    cfg["_comment"] = (f"E2 arm M={m_tokens}: bottleneck AR only; baseline = Phase-2 "
                       "report AR cells; see PREREG_E2.md")
    return cfg


def e1_config(phase1: dict, shaped: bool) -> dict:
    cfg = json.loads(json.dumps(phase1))
    cfg["experiments"] = _disable_all_but_e8(cfg["experiments"])
    cfg["experiments"].pop("p3_transfer", None)          # no transfer set in 2D
    cfg["out"] = f"runs/e1_2d_{'shaped' if shaped else 'base'}/report.json"
    cfg["prereg_file"], cfg["prereg_guard"] = "PREREG_E1.md", True
    if shaped:
        cfg.setdefault("pretrain", {})["loss_spec"] = {
            "reg_mode": "sigreg_ep_head", "lambda_reg": None,          # FILL at stamping
            "sigreg_n_proj": 256, "sigreg_head_width": None}          # FILL at stamping
    cfg["_comment"] = ("E1 2D " + ("shaped" if shaped else "base") +
                       " arm: AR pretraining only; see PREREG_E1.md")
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase2", default="configs/phase2_v1.json")
    ap.add_argument("--phase1", default="configs/phase1_rec8_v2.json")
    ap.add_argument("--out-dir", default="configs")
    a = ap.parse_args()
    p2 = json.loads(Path(a.phase2).read_text())
    p1 = json.loads(Path(a.phase1).read_text())
    out = Path(a.out_dir)
    written = []
    for m in (512, 1024):
        f = out / f"e2_m{m}.json"
        f.write_text(json.dumps(e2_config(p2, m), indent=1) + "\n"); written.append(str(f))
    for shaped in (False, True):
        f = out / f"e1_2d_{'shaped' if shaped else 'base'}.json"
        f.write_text(json.dumps(e1_config(p1, shaped), indent=1) + "\n"); written.append(str(f))
    print("\n".join(written))


if __name__ == "__main__":
    main()
