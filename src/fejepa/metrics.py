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


@torch.no_grad()
def evaluate_instance(
    model: FEJEPA, arch: InstanceArchive, dtype: torch.dtype = torch.float32, device="cpu"
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
    return out
