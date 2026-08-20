"""The frozen metric hierarchy (plan Sec.4) with per-instance arrays (plan B6).

Primary (anchored / AR models): relative energy gap, von-Mises rel-L2, peak-vM relative
error, critical-region recall (top-10% elements), and -- in E7 -- CG iterations to
tolerance. Secondary: displacement rel-L2 (always reported, never the headline for
anchor-trained models), transfer gap, label-efficiency AUC, *standardized* effective
rank with the input-feature rank floor (plan B4), E6 Spearman rho.

All evaluators return means AND per-instance arrays so significance tests are computed
from stored data, never reconstructed (plan Sec.5 item 3).
"""

from __future__ import annotations

import numpy as np

from .anchor.energy import energy_gap, pi_star_abs
from .fe.stress import element_von_mises


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


# --------------------------------------------------------- per-instance suite --

def displacement_errors(U: np.ndarray, arch) -> np.ndarray:
    return np.array([relative_l2(U[j], arch.U_star[j]) for j in range(arch.n_loads)])


def energy_gap_rel(U: np.ndarray, arch) -> np.ndarray:
    return energy_gap(U, arch.U_star, arch.K, arch.F) / pi_star_abs(arch)


def vm_suite(U: np.ndarray, arch) -> dict:
    m = arch.meta["material"]
    if int(arch.nodes.shape[1]) == 3:                     # WP7 3D-P0.4
        from .fe.tet3d import tet_von_mises as _vm
    else:
        _vm = element_von_mises
    k = max(1, int(round(0.10 * arch.elements.shape[0])))
    rel, peak, recall = [], [], []
    for j in range(arch.n_loads):
        vm_p = _vm(arch.nodes, arch.elements, U[j], m)
        vm_t = _vm(arch.nodes, arch.elements, arch.U_star[j], m)
        rel.append(relative_l2(vm_p, vm_t))
        peak.append(abs(vm_p.max() - vm_t.max()) / (vm_t.max() + 1e-30))
        top_t = set(np.argsort(vm_t)[-k:].tolist())
        top_p = set(np.argsort(vm_p)[-k:].tolist())
        recall.append(len(top_t & top_p) / k)
    return {"vm_rel_l2": np.array(rel), "peak_vm_rel_err": np.array(peak),
            "crit_recall": np.array(recall)}


FIELD_KEYS = ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2",
              "peak_vm_rel_err", "crit_recall")


def evaluate_fields(U: np.ndarray, arch) -> dict:
    """Per-instance scalars (mean over the load battery) for the full suite."""
    vm = vm_suite(U, arch)
    return {
        "disp_rel_l2": float(displacement_errors(U, arch).mean()),
        "energy_gap_rel": float(energy_gap_rel(U, arch).mean()),
        "vm_rel_l2": float(vm["vm_rel_l2"].mean()),
        "peak_vm_rel_err": float(vm["peak_vm_rel_err"].mean()),
        "crit_recall": float(vm["crit_recall"].mean()),
    }


def evaluate_model(predict_fn, archs) -> dict:
    """{means..., 'per_instance': {key: [...]}} for labelled `archs`.

    ``predict_fn(arch) -> U (L, ndof) numpy`` (already masked on Dirichlet dofs).
    """
    per = {k: [] for k in FIELD_KEYS}
    for a in archs:
        vals = evaluate_fields(predict_fn(a), a)
        for k in FIELD_KEYS:
            per[k].append(vals[k])
    out = {k: float(np.mean(per[k])) for k in FIELD_KEYS}
    out["per_instance"] = {k: [float(x) for x in per[k]] for k in FIELD_KEYS}
    out["n_val"] = len(archs)
    return out


def torch_predictor(model, device):
    """Adapter: a trained model with prepare/forward_instance -> numpy predictor."""
    import torch

    def f(arch):
        pack = model.prepare_instance(arch, device)
        model.eval()
        with torch.no_grad():
            u = model.forward_instance(pack)
        model.train()
        return u.detach().cpu().numpy()

    return f


# ------------------------------------------------------- representation rank --

def effective_rank(z: np.ndarray, standardized: bool = False) -> float:
    """Participation-ratio effective rank of rows of z.

    ``standardized=True`` z-scores each dimension first -- the B4 repair: the raw PR
    rank is dominated by the largest-variance direction (audit V11: geometry
    descriptors read 1.37/4 raw but 3.81/4 standardized). E3' reports both.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    if standardized:
        s = z.std(axis=0)
        s[s < 1e-12] = 1.0
        z = z / s
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    ev = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    s_ = ev.sum()
    return 0.0 if s_ <= 1e-30 else float(s_ * s_ / (np.square(ev).sum() + 1e-30))


def label_efficiency_auc(budgets, errs) -> float:
    """Normalized trapezoid of error over log2(budget) -- lower is better."""
    b = np.log2(np.asarray(budgets, dtype=np.float64))
    e = np.asarray(errs, dtype=np.float64)
    if len(b) <= 1:
        return float(e[0])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(e, b) / (b[-1] - b[0]))
