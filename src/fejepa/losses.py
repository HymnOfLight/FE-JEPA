"""The combined FE-JEPA training objective (Eq. loss in the proposal).

.. math::

    L = L_pred + \\lambda_E L_phys + \\lambda_S \\mathrm{SIGReg}(z)
        + \\lambda_I L_inv .

* ``L_pred``  -- masked latent prediction (JEPA).
* ``L_phys``  -- assembled-energy anchor (Lemma 1), label-free.
* ``SIGReg``  -- isotropic-Gaussian collapse control.
* ``L_inv``   -- cross-mesh invariance (exact physical augmentation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fejepa.anchor.energy import EnergyAnchor
from fejepa.data.archive import InstanceArchive
from fejepa.models.encoder import build_node_features
from fejepa.models.fejepa import FEJEPA


@dataclass
class LossConfig:
    lambda_E: float = 1.0
    lambda_S: float = 0.05
    lambda_I: float = 0.1
    lambda_pred: float = 1.0
    mask_frac: float = 0.4
    sigreg_proj: int = 32
    energy_scale: float = 1.0
    use_pred: bool = True
    use_phys: bool = True
    use_sigreg: bool = True
    use_inv: bool = True


def _split_mask(n_nodes: int, frac: float, rng: np.random.Generator):
    n_target = max(1, int(round(frac * n_nodes)))
    perm = rng.permutation(n_nodes)
    target = np.sort(perm[:n_target])
    context = np.sort(perm[n_target:])
    if context.size == 0:  # keep at least one context node
        context = target[:1]
        target = target[1:]
    return torch.as_tensor(target), torch.as_tensor(context)


def physics_loss(
    model: FEJEPA, arch: InstanceArchive, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Decode every load case and return ``(mean energy, disp, latents)``.

    The decoded fields are stacked into a ``(n_loads, ndof)`` node-major tensor
    fed to the energy anchor.
    """

    free_mask = arch.free_mask
    anchor = EnergyAnchor(arch.K, arch.F, free_mask, dtype=dtype)
    disp_rows = []
    latents_all = []
    for j in range(arch.n_loads):
        feats = build_node_features(arch, j, dtype=dtype)
        latents, disp = model.encode_decode(feats)
        latents_all.append(latents)
        disp_rows.append(disp.reshape(-1))  # node-major (ux0,uy0,ux1,uy1,...)
    u = torch.stack(disp_rows, dim=0)  # (n_loads, ndof)
    energy = anchor(u, reduction="mean")
    return energy, u, latents_all


def compute_instance_loss(
    model: FEJEPA,
    arch: InstanceArchive,
    cfg: LossConfig | None = None,
    arch_coarse: InstanceArchive | None = None,
    rng: np.random.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict]:
    """Compute the combined loss and a dict of (detached) component values."""

    cfg = cfg or LossConfig()
    rng = rng or np.random.default_rng()
    device_zero = torch.zeros((), dtype=dtype)
    total = device_zero.clone()
    parts: dict[str, float] = {}

    latents_all: list[torch.Tensor] = []

    if cfg.use_phys:
        energy, _, latents_all = physics_loss(model, arch, dtype=dtype)
        l_phys = cfg.energy_scale * energy
        total = total + cfg.lambda_E * l_phys
        parts["phys"] = float(l_phys.detach())

    # Masked latent prediction on a representative load case.
    if cfg.use_pred:
        load_idx = int(rng.integers(arch.n_loads))
        feats = build_node_features(arch, load_idx, dtype=dtype)
        target_idx, context_idx = _split_mask(arch.n_nodes, cfg.mask_frac, rng)
        pred, tgt = model.masked_prediction(feats, target_idx, context_idx)
        l_pred = ((pred - tgt) ** 2).mean()
        total = total + cfg.lambda_pred * l_pred
        parts["pred"] = float(l_pred.detach())

    if cfg.use_sigreg:
        if not latents_all:
            feats = build_node_features(arch, 0, dtype=dtype)
            latents_all = [model.encode(feats)]
        z = torch.cat(latents_all, dim=0)
        from fejepa.models.sigreg import sigreg_loss

        l_sig = sigreg_loss(z, n_proj=cfg.sigreg_proj)
        total = total + cfg.lambda_S * l_sig
        parts["sigreg"] = float(l_sig.detach())

    if cfg.use_inv and arch_coarse is not None:
        feats_f = build_node_features(arch, 0, dtype=dtype)
        feats_c = build_node_features(arch_coarse, 0, dtype=dtype)
        e_f = model.project(model.encode(feats_f))
        e_c = model.project(model.encode(feats_c))
        l_inv = ((e_f - e_c) ** 2).mean()
        total = total + cfg.lambda_I * l_inv
        parts["inv"] = float(l_inv.detach())

    parts["total"] = float(total.detach())
    return total, parts
