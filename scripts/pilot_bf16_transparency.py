#!/usr/bin/env python3
"""bf16 transparency pilot (PREREG_PHASE2 r10, L5 -- pre-declared, one-shot).

Question: does bf16 autocast (network compute only; the energy anchor
self-protects to fp32) change the decision-relevant metrics?

Protocol, fixed before execution:
  * own miniature corpus (gmsh3d, n = 128, seed 123, production lc band) with
    its OWN ledger -- zero contact with the frozen Phase-2 corpora;
  * arms per seed s in {0, 1}: AR pretrain (50 ep on 96 unlabelled) and
    supervised+balanced anchor (50 ep on 64 labelled); each at fp32 AND bf16
    with identical seeds; evaluation on the same 32 labelled instances;
  * acceptance (pre-declared): every run finite AND, for every (arm, seed),
    the relative deviation |bf16 - fp32| / fp32 of BOTH val disp_rel_l2 and
    val energy_gap_rel is <= 1e-2. Any violation => FAIL => precision stays
    fp32 in the stamped config.
Output: runs/phase2/pilot_bf16_report.json with the full table, the measured
per-step speedup (feeds the final wall-clock projection), and the verdict.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CRITERIA = {"max_rel_dev": 1e-2, "metrics": ["disp_rel_l2", "energy_gap_rel"],
            "seeds": [0, 1], "epochs": 50, "n_train_ar": 96, "n_train_sup": 64,
            "n_val": 32,
            "engagement": "encoder latents must be bf16 under autocast "
                          "(anti-no-op clause; the compile lesson)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/phase2/pilot_bf16_report.json")
    ap.add_argument("--data", default="runs/phase2/pilot_bf16_data")
    a = ap.parse_args()

    import torch

    from fejepa.data.archive import load_instance
    from fejepa.experiments.protocol import load_split, seeded_factory
    from fejepa.experiments.runner import _label_files
    from fejepa.fe.gmsh3d import generate_gmsh3d_dataset
    from fejepa.fe.solve import SolveLedger
    from fejepa.metrics import evaluate_model, torch_predictor
    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa
    from fejepa.train.losses import AR_CONFIG
    from fejepa.train.pretrain import PretrainConfig, pretrain
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from fejepa.runtime import setup_torch

    policy = setup_torch(dev, tf32=True)   # the run's baseline is TF32-fp32
    ddir = Path(a.data)
    ledger = SolveLedger()
    if not (ddir / "manifest.json").exists():
        generate_gmsh3d_dataset(ddir, 128, 123, labelled="none",
                                lc_range=(0.0579, 0.0906))
    sp = load_split(str(ddir), CRITERIA["n_val"], seed=1)
    val_files = sp.val_files
    train_files = sp.pool_files
    _label_files(val_files, ledger, "pilot-val")
    _label_files(train_files[:CRITERIA["n_train_sup"]], ledger, "pilot-sup")
    val = [load_instance(f) for f in val_files]
    ar_train = [load_instance(f) for f in train_files[:CRITERIA["n_train_ar"]]]
    sup_train = [load_instance(f) for f in
                 train_files[:CRITERIA["n_train_sup"]]]

    mcfg = {"dim": 256, "depth": 8, "heads": 8, "mask_frac": 0.2,
            "scale_decode": True,
            "features": {"load_summary": True, "geometry": True,
                         "spatial_dim": 3}}

    # engagement probe (anti-no-op): the encoder must actually compute in
    # bf16 under autocast; output-dtype is NOT valid evidence (the mask/scale
    # elementwise muls type-promote back to fp32 by design).
    probe_model = seeded_factory(
        lambda: build_fejepa(FEJEPAConfig.from_dict(mcfg)), 0)
    probe_model.to(dev)
    ppack = probe_model.prepare_instance(ar_train[0], dev)
    with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                        dtype=torch.bfloat16):
        z = probe_model.encode(ppack["feats"])
    engaged = bool(z.dtype == torch.bfloat16)

    rows, times = {}, {}
    for prec in ("fp32", "bf16"):
        for s in CRITERIA["seeds"]:
            # --- AR arm ---
            model = seeded_factory(
                lambda: build_fejepa(FEJEPAConfig.from_dict(mcfg)), s)
            t0 = time.perf_counter()
            pretrain(model, ar_train,
                     PretrainConfig(epochs=CRITERIA["epochs"], lr=1e-3,
                                    seed=s, device=dev, loss=AR_CONFIG,
                                    log_every=-1, precision=prec))
            times[("ar", s, prec)] = time.perf_counter() - t0
            rows[("ar", s, prec)] = evaluate_model(
                torch_predictor(model, dev), val)
            # --- supervised + balanced anchor arm ---
            model = seeded_factory(
                lambda: build_fejepa(FEJEPAConfig.from_dict(mcfg)), s)
            t0 = time.perf_counter()
            train_supervised(model, sup_train, val,
                             SupervisedConfig(epochs=CRITERIA["epochs"],
                                              lr=1.5e-3, seed=s, device=dev,
                                              anchor_mode="balanced",
                                              balance_ratio=1.0,
                                              log_every=-1, precision=prec))
            times[("sup", s, prec)] = time.perf_counter() - t0
            rows[("sup", s, prec)] = evaluate_model(
                torch_predictor(model, dev), val)

    import numpy as np

    table, devs, finite = {}, [], True
    for arm in ("ar", "sup"):
        for s in CRITERIA["seeds"]:
            e = {}
            for prec in ("fp32", "bf16"):
                r = rows[(arm, s, prec)]
                e[prec] = {m: float(r[m]) for m in CRITERIA["metrics"]}
                finite &= all(np.isfinite(v) for v in e[prec].values())
            e["rel_dev"] = {m: abs(e["bf16"][m] - e["fp32"][m])
                            / (abs(e["fp32"][m]) + 1e-30)
                            for m in CRITERIA["metrics"]}
            devs += list(e["rel_dev"].values())
            e["seconds"] = {p: round(times[(arm, s, p)], 1)
                            for p in ("fp32", "bf16")}
            table[f"{arm}_s{s}"] = e

    speedup = (sum(times[k] for k in times if k[2] == "fp32")
               / max(1e-9, sum(times[k] for k in times if k[2] == "bf16")))
    verdict = bool(finite and max(devs) <= CRITERIA["max_rel_dev"] and engaged)
    report = {"criteria": CRITERIA, "device": dev,
              "numeric_policy_baseline": policy,
              "engagement": {"encoder_bf16_under_autocast": engaged,
                             "note": "False on cuda => autocast no-op => FAIL"},
              "torch": torch.__version__, "table": table,
              "max_rel_dev_measured": float(max(devs)),
              "all_finite": bool(finite),
              "measured_speedup_fp32_over_bf16": round(float(speedup), 3),
              "pilot_ledger": ledger.as_dict(),
              "verdict": "PASS" if verdict else "FAIL",
              "consequence": ("stamped config sets runtime.precision = bf16"
                              if verdict else
                              ("precision stays fp32; eager schedule stands"
                               + ("" if engaged else
                                  " [failed the anti-no-op engagement clause]")))}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in
                      ("verdict", "max_rel_dev_measured",
                       "measured_speedup_fp32_over_bf16", "all_finite")},
                     indent=1))


if __name__ == "__main__":
    main()
