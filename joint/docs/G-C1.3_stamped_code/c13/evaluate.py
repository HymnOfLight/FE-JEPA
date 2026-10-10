"""Evaluate gate G-C1.3 as pre-specified: the metrics of every arm and seed on the held-out and tested instances,
the criteria on the label-free arm, the verdict. Writes <runs>/verdict.json.

A run's status comes from its files: a history.json whose status starts with "non-finite" means a final non-finite
stop, whatever else is there; otherwise predictions (preds.npz) mean finished, and their absence means missing. A non-finite label-free
seed enters the medians over seeds with |S_pred / S_ref - 1| = infinity (written as Infinity in verdict.json, which
Python's json reads) and intersection over union = 0. A missing label-free seed makes the verdict INCOMPLETE, and the
label-free metrics are then withheld from verdict.json. Supervised seeds do not enter the verdict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fejoint import hex8i as H
from fejoint.export import energy, stiffness, physical_displacement
from .data import load_export, load_label, manifest, check_files

TOL = 1e-3            # contact: u_x <= TOL * u_scale on the column face, prediction and reference alike


def contact_set(ex, u):
    col = 3 * np.nonzero(ex["node_flags"][:, 1])[0]
    return u[col] <= TOL * ex["u_scale"]


def metrics(ex, u, ref) -> dict:
    """Metrics of one predicted nodal vector u against the reference solution ref (dict with u, Pi, S_paper).
    The rescaled field alpha u, alpha = f^T u / u^T K u, is the energy's minimiser along the ray of u; it needs no
    reference and keeps the contact constraint. It is reported, not judged."""
    u = np.where(ex["dirichlet"], 0.0, np.asarray(u, dtype=np.float64))
    Pi = energy(ex, u)
    fu, uKu_p = float(ex["F"] @ u), float(u @ (ex["K"] @ u))
    alpha = fu / uKu_p if (fu > 0 and uKu_p > 0) else 0.0
    pc, tc = contact_set(ex, u), contact_set(ex, ref["u"])
    union = int((pc | tc).sum())
    e = u - ref["u"]
    uKu = float(ref["u"] @ (ex["K"] @ ref["u"]))
    g = ex["meta"]["geometry"]
    up, ur = physical_displacement(ex, u), physical_displacement(ex, ref["u"])
    vm_p = H.von_mises(H.gauss_strains(ex["nodes"], ex["hexes"], up, g["E"], g["nu"]), g["E"], g["nu"])
    vm_r = H.von_mises(H.gauss_strains(ex["nodes"], ex["hexes"], ur, g["E"], g["nu"]), g["E"], g["nu"])
    return {"S_err": stiffness(ex, u) / ref["S_paper"] - 1,
            "iou": 1.0 if union == 0 else float((pc & tc).sum() / union),
            "gap": (Pi - ref["Pi"]) / abs(ref["Pi"]),
            "err_K": float(np.sqrt(max(e @ (ex["K"] @ e), 0.0) / uKu)),
            "u_rel_l2": float(np.linalg.norm(up - ur) / np.linalg.norm(ur)),
            "vm_rel_l2": float(np.linalg.norm(vm_p - vm_r) / np.linalg.norm(vm_r)),
            "zero_field_pass": bool(Pi < energy(ex, np.zeros_like(u))),
            "min_nonneg_over_scale": float(u[ex["nonneg"]].min() / ex["u_scale"]),
            "alpha": alpha,
            "S_err_rescaled": stiffness(ex, alpha * u) / ref["S_paper"] - 1,
            "gap_rescaled": (energy(ex, alpha * u) - ref["Pi"]) / abs(ref["Pi"])}


def _summary(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"median": float(np.median(v)), "p10": float(np.quantile(v, 0.1)),
                  "p90": float(np.quantile(v, 0.9)), "min": float(v.min()), "max": float(v.max()),
                  "mean": float(v.mean())}
    for k in ("S_err", "S_err_rescaled"):
        v = [abs(r[k]) for r in rows]
        out["abs_" + k] = {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9))}
    return out


def run_status(run: Path) -> str:
    """non-finite (final), finished or missing."""
    hp = run / "history.json"
    if hp.exists() and str(json.loads(hp.read_text()).get("status", "")).startswith("non-finite"):
        return "non-finite"
    return "finished" if (run / "preds.npz").exists() else "missing"


def evaluate(cfg: dict, data_dir, runs_dir) -> dict:
    from .config import sha256

    man = manifest(data_dir)
    assert man["config_sha256"] == sha256(cfg), "the data were generated under another configuration"
    split_of = {r["id"]: r["split"] for r in man["instances"]}
    check_files(data_dir, [i for i, s in split_of.items() if s in cfg["evaluation"]["sets"]])
    cache = {}

    def ex_ref(iid):
        if iid not in cache:
            ex, _ = load_export(Path(data_dir) / "inst" / f"{iid}.npz")
            cache[iid] = (ex, load_label(data_dir, iid))
        return cache[iid]

    out = {"config_sha256": sha256(cfg), "arms": {}, "status": {}, "missing": [], "non_finite": []}
    for arm, a in cfg["arms"].items():
        out["arms"][arm] = {}
        for seed in a["seeds"]:
            name = f"{arm}_s{seed}"
            run = Path(runs_dir) / name
            st = out["status"][name] = run_status(run)
            if st != "finished":
                out["missing" if st == "missing" else "non_finite"].append(name)
                continue
            per = {"eval": {}, "tested": {}}
            with np.load(run / "preds.npz") as z:
                for iid in sorted(z.files):
                    ex, ref = ex_ref(iid)
                    per["eval" if split_of[iid] == "eval" else "tested"][iid] = metrics(ex, z[iid], ref)
            assert len(per["eval"]) == cfg["data"]["n_eval"], f"{name}: incomplete predictions"
            out["arms"][arm][str(seed)] = {"eval": _summary(list(per["eval"].values())), "tested": per["tested"],
                                           "eval_rows": per["eval"]}
    lf_seeds = cfg["arms"]["label_free"]["seeds"]
    if any(out["status"][f"label_free_s{s}"] == "missing" for s in lf_seeds):
        out["verdict"] = "INCOMPLETE"
        out["arms"]["label_free"] = {"withheld": "the verdict is INCOMPLETE"}
    else:
        s_med, i_med = [], []
        for s in lf_seeds:
            if out["status"][f"label_free_s{s}"] == "non-finite":
                s_med.append(float("inf")); i_med.append(0.0)
            else:
                ev = out["arms"]["label_free"][str(s)]["eval"]
                s_med.append(ev["abs_S_err"]["median"]); i_med.append(ev["iou"]["median"])
        out["per_seed_medians"] = {"abs_S_err": s_med, "iou": i_med}
        out["C1_value"] = float(np.median(s_med))
        out["C2_value"] = float(np.median(i_med))
        out["C1"] = bool(out["C1_value"] <= 0.05)
        out["C2"] = bool(out["C2_value"] >= 0.9)
        out["verdict"] = "PASS" if (out["C1"] and out["C2"]) else "FAIL"
    tmp = Path(runs_dir) / "verdict.tmp"
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(Path(runs_dir) / "verdict.json")
    return out
