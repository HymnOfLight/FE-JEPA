"""Training-regime comparison (RQ1 / RQ4 credit assignment).

Compares three ways to train the *same* surrogate backbone on the same set of
instances, to attribute credit between labels and the assembled-energy anchor:

1. ``labels``        -- supervised displacement rel-L2 loss only.
2. ``labels+anchor`` -- supervised loss plus the energy anchor (Lemma 1 term).
3. ``anchor_only``   -- label-free amortized Ritz: the energy anchor alone, using
   no solved fields at all.

Reports validation rel-L2 and relative energy gap for each, along with the
number of labelled solves each regime consumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fejepa.data.archive import load_problem
from fejepa.metrics import evaluate_instance
from fejepa.train.pretrain import PretrainConfig, amortized_ritz
from fejepa.train.supervised import SupervisedConfig, train_supervised


def compare_training_regimes(
    pool_files: list[Path],
    val_archs: list,
    n_train: int,
    sup_cfg: SupervisedConfig | None = None,
    pre_cfg: PretrainConfig | None = None,
    lambda_phys: float = 1.0,
    out_report: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    sup_cfg = sup_cfg or SupervisedConfig(epochs=60, lr=1.5e-3)
    pre_cfg = pre_cfg or PretrainConfig(epochs=sup_cfg.epochs, lr=1.5e-3, model=sup_cfg.model)
    pre_cfg.device = sup_cfg.device  # keep regimes on the same device
    device = sup_cfg.device
    train_archs = [load_problem(f) for f in pool_files[:n_train]]

    def _eval(model):
        vs = [evaluate_instance(model, a, device=device) for a in val_archs]
        agg = lambda k: float(np.mean([v[k] for v in vs if k in v]))  # noqa: E731
        return {
            "val_rel_l2": agg("rel_l2_disp"),
            "val_energy_gap_rel": agg("energy_gap_rel"),
            "val_rel_l2_vm": agg("rel_l2_vm"),
            "val_max_vm_rel_err": agg("max_vm_rel_err"),
            "val_crit_recall": agg("crit_recall"),
        }

    def _from_supervised(out):
        return {
            "val_rel_l2": out["val_rel_l2_disp"],
            "val_energy_gap_rel": out["val_energy_gap_rel"],
            "val_rel_l2_vm": out["val_rel_l2_vm"],
            "val_max_vm_rel_err": out["val_max_vm_rel_err"],
            "val_crit_recall": out["val_crit_recall"],
        }

    regimes: dict[str, dict] = {}

    if verbose:
        print("[regimes] training labels-only...")
    cfg0 = SupervisedConfig(**{**sup_cfg.__dict__, "lambda_phys": 0.0})
    regimes["labels"] = {**_from_supervised(train_supervised(train_archs, val_archs, cfg=cfg0)),
                         "labelled_solves": n_train}

    if verbose:
        print("[regimes] training labels+anchor...")
    cfg1 = SupervisedConfig(**{**sup_cfg.__dict__, "lambda_phys": lambda_phys})
    regimes["labels+anchor"] = {**_from_supervised(train_supervised(train_archs, val_archs, cfg=cfg1)),
                               "labelled_solves": n_train}

    if verbose:
        print("[regimes] training anchor-only (label-free)...")
    model2, _ = amortized_ritz(train_archs, cfg=pre_cfg)
    regimes["anchor_only"] = {**_eval(model2), "labelled_solves": 0}

    report = {"n_train": n_train, "regimes": regimes}
    if verbose:
        for name, m in regimes.items():
            print(
                f"  {name:14s} val_rel_l2={m['val_rel_l2']:.4f} "
                f"gap_rel={m['val_energy_gap_rel']:.4f} labels={m['labelled_solves']}"
            )
    if out_report is not None:
        Path(out_report).write_text(json.dumps(report, indent=2))
    return report
