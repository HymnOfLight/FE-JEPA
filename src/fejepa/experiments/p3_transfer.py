"""P3 -- resolution transfer (PREREG_PHASE2 r8: Sec.3 P3, Sec.2(c), Sec.4 KP4).

Protocol, pinned by the pre-registration:
  * zero-shot: the P1 checkpoints (shared, never re-run -- e8 persists them)
    evaluated on the fine transfer set AND on the in-band validation set; the
    gate ratio is fine/in-band on seed-mean displacement (Sec.5 aggregation).
  * few-shot: fine-tune the seed-matched AR checkpoint at fine with the pinned
    schedule against scratch-at-fine under the same schedule; reported only,
    no gate or kill attaches.
  * naives at fine, strongest form: knn_field and the scale-aware polynomial
    built from the full in-band prefix (naive_budget) and transported / fitted
    cross-resolution (their predict is mesh-independent by construction).

The kills and the gate verdict are NOT computed here: gate_g2 is the single
source (this module produces its consumed contract block).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..baselines import KNNFieldBaseline, ScaleAwarePolyBaseline
from ..metrics import FIELD_KEYS, evaluate_model, torch_predictor
from .parallel import _build_model, cached_supervised_unit, map_units
from .protocol import divergence_flags, load_archs, mean_std, result, seeds_list

PLAN_REF = "PREREG_PHASE2 r8 Sec.3 P3 / Sec.2(c) / Sec.4 KP4"


def _load_state_model(kind: str, model_cfg: dict, seed: int, state_path,
                      device: str):
    import torch

    model = _build_model({"kind": kind, "model": model_cfg, "seed": seed})
    sd = torch.load(str(state_path), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def _mm(evals: list[dict], key: str = "disp_rel_l2") -> float:
    return float(np.mean([e[key] for e in evals]))


def _row(evals: list[dict]) -> dict:
    out = {k: mean_std([e[k] for e in evals]) for k in FIELD_KEYS}
    out["divergence_flags"] = divergence_flags(evals)
    out["per_seed_eval"] = evals
    return out


def run_p3(model_cfg: dict, pool_files, val_archs, fine_eval_archs,
           fine_eval_files, fine_prefix_files, cfg: dict) -> dict:
    fe_kind = str(model_cfg.get("kind", "fejepa"))   # wp8: config-driven architecture
    seeds = seeds_list(cfg.get("seeds", 3))
    device = cfg.get("device", "cpu")
    workers = int(cfg.get("workers", 1))
    tf32 = bool(cfg.get("tf32", True))
    state_dir = Path(cfg["state_dir"])
    p_max = int(cfg.get("pool_size", 1024))
    b_max = int(cfg.get("bmax", 1024))
    nb = int(cfg.get("naive_budget", 1024))
    fewshot_budgets = [int(b) for b in cfg.get("fewshot_budgets", [16, 64])]
    sup = dict(epochs=int(cfg.get("fewshot_epochs", 200)),
               lr=float(cfg.get("fewshot_lr", 1.5e-3)), device=device,
               anchor_mode="none")

    # ---- zero-shot rows from the shared P1 checkpoints ----------------------
    def zero_shot(kind: str, template: str):
        fine, inb = [], []
        for s in seeds:
            sp = state_dir / template.format(s=s)
            if not sp.exists():
                return None
            model = _load_state_model(kind, model_cfg, s, sp, device)
            pf = torch_predictor(model, device)
            fine.append(evaluate_model(pf, fine_eval_archs))
            inb.append(evaluate_model(pf, val_archs))
        return {"fine": _row(fine), "inband": _row(inb),
                "fine_disp_mean": _mm(fine), "inband_disp_mean": _mm(inb)}

    ar = zero_shot(fe_kind, f"ar_p{p_max}_s{{s}}.pt")
    if ar is None:
        raise FileNotFoundError(
            f"P3 requires the shared AR checkpoints ar_p{p_max}_s*.pt in "
            f"{state_dir} (run E8 first; identical configurations are trained "
            f"once and shared, never re-run)")
    reported = {"labels@max": zero_shot(fe_kind, f"labels_b{b_max}_s{{s}}.pt"),
                "mgn@max": zero_shot("mgn", f"mgn_b{b_max}_s{{s}}.pt")}

    ratio = ar["fine_disp_mean"] / (ar["inband_disp_mean"] + 1e-30)

    # ---- few-shot at fine: seed-matched AR init vs scratch (reported) -------
    fs_keys, fs_payloads = [], []
    for s in seeds:
        for b in fewshot_budgets:
            for arm, init in (("finetune", str(state_dir / f"ar_p{p_max}_s{s}.pt")),
                              ("scratch", None)):
                fs_keys.append((arm, b, s))
                fs_payloads.append({
                    "kind": fe_kind, "model": model_cfg, "seed": s,
                    "tf32": tf32,
                    "compile": cfg.get("compile", False),
                    "precision": cfg.get("precision", "fp32"),
                    "train_files": [str(f) for f in fine_prefix_files[:b]],
                    "val_files": [str(f) for f in fine_eval_files],
                    "sup": dict(sup, desc=f"P3 {arm} b{b} s{s}"),
                    "pretrained_path": init, "tag": f"P3 {arm} b{b} s{s}",
                    "cache_dir": str(state_dir / "unit_cache_p3"),
                    "reuse_existing": bool(cfg.get("reuse_states")),
                })
    fs_out = dict(zip(fs_keys, map_units(cached_supervised_unit, fs_payloads, workers,
                                         "P3 (few-shot at fine)"), strict=True))
    fewshot = {b: {arm: _row([fs_out[(arm, b, s)]["val"] for s in seeds])
                   for arm in ("finetune", "scratch")}
               for b in fewshot_budgets}

    # ---- naives at fine, strongest form (naive_budget in-band prefix) -------
    naive_at_fine, naive_rows = {}, {}
    if nb > 0:                          # wp8: naive_budget 0 = zero-shot-only run
        naive_pool = load_archs(pool_files[:nb])
        for name, cls in (("knn_field", KNNFieldBaseline),
                          ("scale_aware_poly", ScaleAwarePolyBaseline)):
            ev = evaluate_model(cls().fit(naive_pool).predict, fine_eval_archs)
            naive_rows[name] = _row([ev])
            naive_at_fine[name] = float(ev["disp_rel_l2"])

    proto = {"seeds": seeds, "n_fine_eval": len(fine_eval_archs),
             "n_inband_val": len(val_archs),
             "fewshot_budgets": fewshot_budgets, "naive_budget": nb,
             "shared_checkpoints": {"ar": f"ar_p{p_max}_s*.pt",
                                    "labels": f"labels_b{b_max}_s*.pt",
                                    "mgn": f"mgn_b{b_max}_s*.pt"},
             "ratio_def": "seed_mean_over_seed_mean",
             "fewshot_schedule": {"epochs": sup["epochs"], "lr": sup["lr"],
                                  "init": "seed-matched AR checkpoint"}}
    metrics = {"ar": {"fine_disp_mean": ar["fine_disp_mean"],
                      "inband_disp_mean": ar["inband_disp_mean"],
                      "fine": ar["fine"], "inband": ar["inband"]},
               "ratio_fine_over_inband": float(ratio),
               "zero_shot_reported": reported,
               "fewshot": fewshot,
               "naive_at_fine": naive_at_fine,
               "naive_rows": naive_rows,
               "thresholds_echo": {"gate_ratio": cfg.get("gate_ratio", 1.25),
                                   "kill_ratio": cfg.get("kill_ratio", 1.5)}}
    return result("P3", PLAN_REF, proto, metrics, kills=[])
