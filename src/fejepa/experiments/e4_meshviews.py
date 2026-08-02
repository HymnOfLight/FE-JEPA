"""E4' -- cross-resolution transfer with the invariance term (plan Sec.6 E4').

Protocol: for each coarsening factor (plan: {1.8, 2.5}), pretrain label-free (AR) on
the coarse meshes with the invariance term off/on (fine twin as the view); evaluate
displacement on labelled val pairs at both resolutions. The transfer gap is
err_fine - err_coarse; the report includes the gap *relative to the absolute error*
(the audited honesty fix: 0.0041 vs 0.0119 was real but small against 0.22).

Kill K6: "gap reduction vanishes at the largest coarsening (< min_reduction, default
10%) => drop the augmentation claim."
"""

from __future__ import annotations

import numpy as np

from ..metrics import evaluate_model, torch_predictor
from ..train.losses import AR_CONFIG, with_inv
from ..train.pretrain import PretrainConfig, pretrain
from ..progress import Task
from .protocol import kill, result, seeded_factory

PLAN_REF = "plan v2.0 Sec.6 E4', kill K6"


def _pairs_split(pairs: list[dict], n_val: int, seed: int):
    perm = np.random.default_rng(seed).permutation(len(pairs))
    val = [pairs[i] for i in perm[:n_val]]
    train = [pairs[i] for i in perm[n_val:]]
    return train, val


def run_e4(model_factory, datasets: dict, load_pair, cfg: dict) -> dict:
    """datasets: {coarsen(float): data_dir}; load_pair(dir, rec) -> (fine, coarse)."""
    n_val = int(cfg.get("n_val", 128))
    n_train = int(cfg.get("n_train", 512))
    seed = int(cfg.get("seed", 0))
    device = cfg.get("device", "cpu")
    pcfg = dict(epochs=int(cfg.get("epochs", 100)), lr=float(cfg.get("lr", 1e-3)),
                seed=seed, device=device)

    from ..data.archive import load_manifest

    task = Task("E4'", total=2 * len(datasets))
    per_coarsen = []
    for coarsen in sorted(datasets):
        ddir = datasets[coarsen]
        pairs = load_manifest(ddir)["pairs"]
        train_recs, val_recs = _pairs_split(pairs, n_val, seed)
        train_pairs = [load_pair(ddir, r) for r in train_recs[:n_train]]
        val_pairs = [load_pair(ddir, r) for r in val_recs]
        val_fine = [p[0] for p in val_pairs]
        val_coarse = [p[1] for p in val_pairs]

        row = {"coarsen": float(coarsen), "n_train": len(train_pairs),
               "n_val": len(val_pairs)}
        for inv_on in (False, True):
            model = seeded_factory(model_factory, seed)
            pcfg2 = dict(pcfg, desc=f"E4' c{coarsen} inv={'on' if inv_on else 'off'}")
            pretrain(model, None, PretrainConfig(loss=with_inv(AR_CONFIG, inv_on),
                                                 **pcfg2), pairs=train_pairs)
            pred = torch_predictor(model, device)
            err_f = evaluate_model(pred, val_fine)["disp_rel_l2"]
            err_c = evaluate_model(pred, val_coarse)["disp_rel_l2"]
            tag = "inv_on" if inv_on else "inv_off"
            task.step(f"coarsen {coarsen} inv={'on' if inv_on else 'off'}")
            row[tag] = {"err_fine": err_f, "err_coarse": err_c,
                        "gap": err_f - err_c,
                        "gap_over_abs_err": (err_f - err_c) / (0.5 * (err_f + err_c) + 1e-30)}
        g_off, g_on = row["inv_off"]["gap"], row["inv_on"]["gap"]
        row["gap_reduction"] = (g_off - g_on) / (abs(g_off) + 1e-30)
        per_coarsen.append(row)

    task.done()
    min_red = float(cfg.get("min_reduction", 0.10))
    worst = per_coarsen[-1]                      # largest coarsening
    k6 = kill(f"K6: gap reduction < {min_red:.0%} at coarsen={worst['coarsen']}",
              triggered=bool(worst["gap_reduction"] < min_red),
              note=f"reduction at largest coarsening = {worst['gap_reduction']:.4f}")
    proto = {"coarsens": sorted(float(c) for c in datasets), "n_val": n_val,
             "n_train": n_train, "pretrain": pcfg, "label_free_training": True}
    return result("E4'", PLAN_REF, proto, {"per_coarsen": per_coarsen}, [k6])
