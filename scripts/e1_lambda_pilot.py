#!/usr/bin/env python3
"""E1 lambda-selection pilot (PREREG_E1 Sec. 3, pre-declared rule).

Seed 0, a short schedule, lambda in a fixed grid: train AR (lambda = 0) and
AR + SIGReg(head) at each lambda on the same instances; evaluate on the same
pilot-validation set; SELECT the largest lambda whose validation displacement
error is within `--tol` (default 5%) relative of the AR pilot. The selected
lambda is written into the E1 pre-registration at stamping and never changed.

Data (Stage 1.28): the pilot uses the E1 runs' OWN split (the config's
`split` block) and never touches its validation set -- it trains on
pool[:n_train] and validates on pool[n_train : n_train + n_val], instances the
E1 arms see only label-free inside their 1024 training prefix. (Before 1.28
the pilot drew its own 128-instance split with the same seed: its validation
set was half of E1's and its training set held the other half -- lambda was
selected on E1's test data.) The pilot-validation instances must already carry
labels (they lie inside the Phase-1 labelled prefix); buying labels would
rewrite the corpus manifest, so it is refused unless --allow-labelling.

Head width (Stage 1.28): `--head-width auto` measures the intrinsic dimension
of the lambda = 0 pilot model's latents on the pilot-validation instances and
applies the pre-declared rule (round(2 x TwoNN ID) if ID < dim/4, else 0 =
full width) BEFORE the SIGReg candidates are trained, so lambda is selected
under the head the shaped arm will use. An integer fixes the width instead.

Usage (box, 2D corpus):
    python scripts/e1_lambda_pilot.py --config configs/phase1_rec8_v2.json \
        --data runs/data2d --n-train 512 --n-val 128 --epochs 20 --head-width auto \
        --out runs/wp8/e1_pilot.json
Anywhere: --smoke.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _list_sha(files) -> str:
    return hashlib.sha256("\n".join(Path(f).name for f in files).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1_rec8_v2.json")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--head-width", default="auto",
                    help="'auto' (intrinsic-dimension rule on the lambda=0 pilot model) "
                         "or an integer (0 = model dim)")
    ap.add_argument("--allow-labelling", action="store_true",
                    help="buy labels for unlabelled pilot-validation instances "
                         "(rewrites the corpus manifest; never on the Phase-1 corpus)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/wp8/e1_pilot.json")
    a = ap.parse_args()

    import torch

    from fejepa.analysis.common import build_model_from_config, write_json
    from fejepa.analysis.intrinsic_dim import measure_intrinsic_dimension
    from fejepa.data.archive import has_labels, load_instance, manifest_sha256
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

        ddir = generate_synthetic_dataset(Path(tempfile.mkdtemp()) / "pilot", n=10, seed=3)
        mcfg = {"dim": 16, "depth": 1, "heads": 2,
                "features": {"load_summary": True, "geometry": True}}
        split = {"n_val": 2, "seed": 1}
        a.n_train, a.n_val, a.epochs, lr = 4, 2, 1, 1e-3
        a.allow_labelling = True
        cfg_sha = None
    else:
        cfg = json.loads(Path(a.config).read_text())
        mcfg, ddir = cfg["model"], a.data or cfg["data"]["dir"]
        split = cfg["split"]
        lr = float(cfg.get("pretrain", {}).get("lr", 1e-3))
        cfg_sha = hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    sp = load_split(str(ddir), int(split["n_val"]), seed=int(split["seed"]))
    train_files = sp.pool_files[:a.n_train]
    pval_files = sp.pool_files[a.n_train:a.n_train + a.n_val]
    if len(pval_files) < a.n_val:
        raise SystemExit(f"pilot: pool too small for n_train + n_val = {a.n_train + a.n_val}")
    assert not set(map(str, pval_files)) & set(map(str, sp.val_files))   # never E1's test set
    missing = [f for f in pval_files if not has_labels(f)]
    if missing and not a.allow_labelling:
        raise SystemExit(f"pilot: {len(missing)} pilot-validation instances carry no labels; "
                         "they should lie inside the corpus's labelled prefix (reduce "
                         "--n-train + --n-val) -- or pass --allow-labelling on a scratch corpus")
    manifest_before = manifest_sha256(ddir)
    ledger = SolveLedger()
    _label_files(pval_files, ledger, "pilot-val")              # no-op when labelled
    val = [load_instance(f) for f in pval_files]
    train = [load_instance(f) for f in train_files]

    def train_eval(loss, desc):
        m = build_model_from_config(mcfg, mode="train")
        pretrain(m, train, PretrainConfig(epochs=a.epochs, lr=lr, seed=0, device=dev,
                                          loss=loss, log_every=-1, desc=desc))
        m.eval()
        ev = evaluate_model(torch_predictor(m, dev), val)
        return m, {"disp_rel_l2": float(ev["disp_rel_l2"]),
                   "energy_gap_rel": float(ev["energy_gap_rel"])}

    rows = {}
    ar_model, rows["0.0"] = train_eval(AR_CONFIG, "E1 pilot lambda=0")
    print(f"lambda=0     : disp {rows['0.0']['disp_rel_l2']:.4f}  "
          f"egap {rows['0.0']['energy_gap_rel']:.4f}", flush=True)
    id_reading = None
    if str(a.head_width) == "auto":
        id_reading = measure_intrinsic_dimension(ar_model, val)
        head_width = int(id_reading["suggested_head_width"])
        width_source = "auto: intrinsic-dimension rule on the lambda=0 pilot model (pilot-val)"
    else:
        head_width, width_source = int(a.head_width), "fixed by --head-width"
    del ar_model
    for lam in a.lambdas:
        _, rows[str(lam)] = train_eval(ar_sigreg_config(lam, head=True, n_proj=256,
                                                        head_width=head_width),
                                       f"E1 pilot lambda={lam}")
        print(f"lambda={lam:<6}: disp {rows[str(lam)]['disp_rel_l2']:.4f}  "
              f"egap {rows[str(lam)]['energy_gap_rel']:.4f}", flush=True)
    base = rows["0.0"]["disp_rel_l2"]
    admissible = [lam for lam in a.lambdas
                  if rows[str(lam)]["disp_rel_l2"] <= base * (1.0 + a.tol)]
    selected = max(admissible) if admissible else None
    res = {"rule": f"largest lambda with pilot-val disp <= AR * (1 + {a.tol})",
           "epochs": a.epochs, "n_train": a.n_train, "n_val": a.n_val, "seed": 0,
           "split": {"n_val": int(split["n_val"]), "seed": int(split["seed"])},
           "pilot_val": f"pool[{a.n_train}:{a.n_train + a.n_val}] of the E1 split "
                        "(disjoint from E1's validation set)",
           "train_files_sha256": _list_sha(train_files),
           "pilot_val_files_sha256": _list_sha(pval_files),
           "head_width": head_width, "head_width_source": width_source,
           "intrinsic_dimension": id_reading,
           "rows": rows, "admissible": admissible, "selected_lambda": selected,
           "pilot_ledger": ledger.as_dict(), "config_sha256": cfg_sha,
           "manifest_sha256_before": manifest_before,
           "manifest_sha256_after": manifest_sha256(ddir),
           "device": dev, "smoke": a.smoke}
    write_json(a.out, res)
    print(json.dumps({"selected_lambda": selected, "admissible": admissible,
                      "head_width": head_width}))


if __name__ == "__main__":
    main()
