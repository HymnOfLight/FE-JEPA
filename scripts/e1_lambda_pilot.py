#!/usr/bin/env python3
"""E1 lambda-selection pilot (PREREG_E1 Sec. 3, pre-declared rule).

Seed 0, a short schedule, lambda in a fixed grid: train AR (lambda = 0) and
AR + SIGReg(head) at each lambda on the same instances; evaluate on the same
validation set; SELECT the largest lambda whose validation displacement error
is within `--tol` (default 5%) relative of the AR pilot. The selected lambda
is written into the E1 pre-registration at stamping and never changed.

Usage (box, 2D corpus): python scripts/e1_lambda_pilot.py --config <phase1 cfg> \
    --data <2D corpus dir> --n-train 512 --n-val 128 --epochs 20 --out runs/wp8/e1_pilot.json
Anywhere: --smoke.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_rec8_v2.json")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--head-width", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/wp8/e1_pilot.json")
    a = ap.parse_args()

    import torch

    from fejepa.analysis.common import build_model_from_config, write_json
    from fejepa.data.archive import load_instance
    from fejepa.experiments.protocol import load_split
    from fejepa.experiments.runner import _label_files
    from fejepa.fe.solve import SolveLedger
    from fejepa.metrics import evaluate_model, torch_predictor
    from fejepa.train.losses import AR_CONFIG, ar_sigreg_config
    from fejepa.train.pretrain import PretrainConfig, pretrain

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.smoke:
        import tempfile

        from fejepa.fe.synthetic import generate_synthetic_dataset

        ddir = generate_synthetic_dataset(Path(tempfile.mkdtemp()) / "pilot", n=8, seed=3)
        mcfg = {"dim": 16, "depth": 1, "heads": 2,
                "features": {"load_summary": True, "geometry": True}}
        a.n_train, a.n_val, a.epochs, lr = 4, 2, 1, 1e-3
    else:
        cfg = json.loads(Path(a.config).read_text())
        mcfg, ddir = cfg["model"], a.data
        lr = float(cfg.get("pretrain", {}).get("lr", 1e-3))
    sp = load_split(str(ddir), a.n_val, seed=1)
    ledger = SolveLedger()
    _label_files(sp.val_files, ledger, "pilot-val")            # val labels only
    val = [load_instance(f) for f in sp.val_files]
    train = [load_instance(f) for f in sp.pool_files[:a.n_train]]

    rows = {}
    for lam in [0.0] + list(a.lambdas):
        loss = AR_CONFIG if lam == 0.0 else ar_sigreg_config(lam, head=True, n_proj=256,
                                                              head_width=a.head_width)
        m = build_model_from_config(mcfg, mode="train")
        pretrain(m, train, PretrainConfig(epochs=a.epochs, lr=lr, seed=0, device=dev,
                                          loss=loss, log_every=-1,
                                          desc=f"E1 pilot lambda={lam}"))
        ev = evaluate_model(torch_predictor(m, dev), val)
        rows[str(lam)] = {"disp_rel_l2": float(ev["disp_rel_l2"]),
                          "energy_gap_rel": float(ev["energy_gap_rel"])}
        print(f"lambda={lam:<6}: disp {ev['disp_rel_l2']:.4f}  egap {ev['energy_gap_rel']:.4f}",
              flush=True)
    base = rows["0.0"]["disp_rel_l2"]
    admissible = [lam for lam in a.lambdas
                  if rows[str(lam)]["disp_rel_l2"] <= base * (1.0 + a.tol)]
    selected = max(admissible) if admissible else None
    res = {"rule": f"largest lambda with val disp <= AR * (1 + {a.tol})",
           "epochs": a.epochs, "n_train": a.n_train, "n_val": a.n_val, "seed": 0,
           "rows": rows, "admissible": admissible, "selected_lambda": selected,
           "pilot_ledger": ledger.as_dict(), "device": dev, "smoke": a.smoke}
    write_json(a.out, res)
    print(json.dumps({"selected_lambda": selected, "admissible": admissible}))


if __name__ == "__main__":
    main()
