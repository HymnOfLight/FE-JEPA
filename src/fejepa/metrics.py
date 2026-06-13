"""Evaluation metrics for FE-JEPA surrogates."""

from __future__ import annotations

import numpy as np
import torch

from fejepa.anchor.energy import energy_gap
from fejepa.data.archive import InstanceArchive
from fejepa.models.encoder import build_node_features
from fejepa.models.fejepa import FEJEPA


def effective_rank(z: np.ndarray) -> float:
    """Participation-ratio effective rank of an embedding matrix ``(n, d)``.

    Defined as ``(sum lambda_i)^2 / sum lambda_i^2`` over covariance eigenvalues
    -- a collapse diagnostic: a collapsed (low-dimensional) embedding has small
    effective rank, an isotropic one has effective rank close to ``d``.
    """

    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    s = ev.sum()
    if s <= 1e-30:
        return 0.0
    return float((s * s) / (np.square(ev).sum() + 1e-30))


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error ``||pred - ref|| / ||ref||`` (row-flattened)."""

    num = np.linalg.norm(pred.reshape(-1) - ref.reshape(-1))
    den = np.linalg.norm(ref.reshape(-1)) + 1e-30
    return float(num / den)


def von_mises_metrics(
    nodes: np.ndarray,
    elements: np.ndarray,
    u_pred: np.ndarray,
    u_star: np.ndarray,
    material,
    top_frac: float = 0.1,
) -> dict:
    """Per-load von Mises metrics: rel-L2, max-stress error, critical localization.

    * ``rel_l2_vm``     -- relative L2 of the element von Mises field,
    * ``max_vm_rel_err``-- |max σ̂_vM − max σ*_vM| / max σ*_vM (peak stress),
    * ``crit_recall``   -- recall of the top-``top_frac`` highest-stress elements
      (critical-region localization).
    """

    from fejepa.fe.stress import element_strain_stress

    _, _, vm_pred = element_strain_stress(nodes, elements, u_pred, material)
    _, _, vm_star = element_strain_stress(nodes, elements, u_star, material)

    rel = float(np.linalg.norm(vm_pred - vm_star) / (np.linalg.norm(vm_star) + 1e-30))
    max_err = float(abs(vm_pred.max() - vm_star.max()) / (abs(vm_star.max()) + 1e-30))

    k = max(1, int(round(top_frac * vm_star.size)))
    top_star = set(np.argsort(vm_star)[-k:].tolist())
    top_pred = set(np.argsort(vm_pred)[-k:].tolist())
    recall = len(top_star & top_pred) / k

    return {"rel_l2_vm": rel, "max_vm_rel_err": max_err, "crit_recall": recall}


@torch.no_grad()
def evaluate_instance(
    model: FEJEPA,
    arch: InstanceArchive,
    dtype: torch.dtype = torch.float32,
    device="cpu",
    stress: bool = True,
) -> dict:
    """Per-instance displacement rel-L2 and energy gap, averaged over loads.

    Requires a labelled instance (``arch.U_star is not None``) for rel-L2; the
    energy gap is always computable from ``K, F`` and the reference energy.
    """

    model.eval()
    free_mask = arch.free_mask
    disp_rows = []
    for j in range(arch.n_loads):
        feats = build_node_features(arch, j, dtype=dtype, device=device)
        latents, disp = model.encode_decode(feats)
        u = (disp.reshape(-1).cpu().numpy()) * free_mask
        disp_rows.append(u)
    U = np.stack(disp_rows, axis=0)

    out: dict = {}
    if arch.U_star is not None:
        rel = [relative_l2(U[j], arch.U_star[j]) for j in range(arch.n_loads)]
        out["rel_l2_disp"] = float(np.mean(rel))
        out["rel_l2_disp_per_load"] = rel
        gaps = energy_gap(U, arch.U_star, arch.K, arch.F)
        out["energy_gap"] = float(np.mean(gaps))
        # Normalised energy gap: gap / |Pi(U*)| keeps scale comparable across instances.
        pi_star = 0.5 * np.einsum(
            "bi,bi->b", arch.U_star, (arch.K @ arch.U_star.T).T
        ) - np.einsum("bi,bi->b", arch.F, arch.U_star)
        denom = np.abs(pi_star) + 1e-30
        out["energy_gap_rel"] = float(np.mean(gaps / denom))

        if stress:
            vm = [
                von_mises_metrics(
                    arch.nodes, arch.elements, U[j], arch.U_star[j], arch.material
                )
                for j in range(arch.n_loads)
            ]
            out["rel_l2_vm"] = float(np.mean([m["rel_l2_vm"] for m in vm]))
            out["max_vm_rel_err"] = float(np.mean([m["max_vm_rel_err"] for m in vm]))
            out["crit_recall"] = float(np.mean([m["crit_recall"] for m in vm]))
    return out
