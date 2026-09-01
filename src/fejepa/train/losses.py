"""Label-free training objectives: Amortized-Ritz (AR) and full JEPA.

Plan v2.0 mapping:
  - C1 / Sec.6 pipelines: AR_CONFIG is the anchor-only regime (zero labels; Lemma 1
    supplies the supervised energy-norm gradient). JEPA_CONFIG adds the redesigned
    masked-prediction task (region masking + cross-attention predictor, plan WP2) and a
    pooled-granularity anti-collapse regularizer.
  - E4': ``use_inv`` adds the cross-resolution invariance term when a coarse/fine twin
    is provided (projection-head MSE between pooled latents).
  - The physics term uses *raw* Pi_h: AR is label-free, so the |Pi(U*)| normalizer is
    unavailable by construction; per-instance scale imbalance is acknowledged and the
    supervised side handles balance via the gradient-balanced anchor instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..models import regularizers as R


@dataclass
class LossConfig:
    use_pred: bool = True
    use_phys: bool = True
    reg_mode: str = "vicreg_pooled"     # none | sigreg | sigreg_pooled | vicreg_pooled
    use_inv: bool = False
    lambda_pred: float = 1.0
    lambda_phys: float = 1.0
    lambda_reg: float = 0.1
    lambda_inv: float = 0.1
    # wp8-lejepa E1: validated Epps-Pulley SIGReg (train/sigreg.py) under NEW
    # mode names so the legacy 2D modes stay bitwise-unchanged:
    #   reg_mode = "sigreg_ep"       -> on the raw encoder tokens
    #   reg_mode = "sigreg_ep_head"  -> on a BatchNorm projector head attached
    #                                   to the model by pretrain()
    sigreg_n_proj: int = 256
    sigreg_head_width: int = 0          # 0 = model dim

    def to_dict(self) -> dict:
        return self.__dict__.copy()


AR_CONFIG = LossConfig(use_pred=False, use_phys=True, reg_mode="none", use_inv=False)


def ar_sigreg_config(lambda_reg: float, head: bool = True, n_proj: int = 256,
                     head_width: int = 0) -> LossConfig:
    """wp8-lejepa E1 arm: the AR objective plus lambda * SIGReg (Epps-Pulley)."""
    return replace(AR_CONFIG, reg_mode="sigreg_ep_head" if head else "sigreg_ep",
                   lambda_reg=float(lambda_reg), sigreg_n_proj=int(n_proj),
                   sigreg_head_width=int(head_width))


def build_sigreg_head(dim: int, width: int = 0):
    """Linear -> BatchNorm1d -> Linear projector (LeWorldModel's placement)."""
    import torch

    w = int(width) or int(dim)
    return torch.nn.Sequential(torch.nn.Linear(dim, w), torch.nn.BatchNorm1d(w),
                               torch.nn.Linear(w, w))
JEPA_CONFIG = LossConfig(use_pred=True, use_phys=True, reg_mode="vicreg_pooled",
                         use_inv=False)


def with_inv(cfg: LossConfig, on: bool) -> LossConfig:
    return replace(cfg, use_inv=on)


def compute_loss(model, pack, anchor, adj, buffer, rng: np.random.Generator,
                 cfg: LossConfig, twin_pack=None):
    """One-instance objective. Returns (total, parts dict of *detached tensors* --
    callers float() them only at log milestones to avoid per-step device syncs)."""
    import torch

    parts = {}
    total = None

    needs_pack = bool(getattr(model, "needs_pack", False))   # wp8 E2 bottleneck
    z = model.encode(pack["feats"], pack) if needs_pack else model.encode(pack["feats"])

    if cfg.use_phys:
        if needs_pack:
            u = model.decode(z, pack)                     # masked + scaled already
        else:
            u = model.decoder(z).reshape(z.shape[0], -1) * pack["free"]
        phys = anchor.energies(u).mean()
        parts["phys"] = phys.detach()        # tensors: no per-step device sync
        total = cfg.lambda_phys * phys

    if cfg.use_pred:
        j = int(rng.integers(0, pack["feats"].shape[0]))
        pred = model.masked_prediction(pack["feats"][j], adj, rng, z_full=z[j])
        parts["pred"] = pred.detach()
        total = cfg.lambda_pred * pred if total is None else total + cfg.lambda_pred * pred

    if cfg.reg_mode != "none":
        if cfg.reg_mode in ("sigreg_ep", "sigreg_ep_head"):
            from .sigreg import sigreg as _sigreg_ep

            zz = z.reshape(-1, z.shape[-1])
            if cfg.reg_mode == "sigreg_ep_head":
                zz = model.sigreg_head(zz)
            reg = _sigreg_ep(zz, n_proj=cfg.sigreg_n_proj)
        elif cfg.reg_mode == "sigreg":
            reg = R.sigreg_node(z)
        else:
            pooled = model.pooled(z)                      # (L, dim) rows as samples
            if cfg.reg_mode == "sigreg_pooled":
                reg = R.sigreg_pooled(pooled, buffer)
            elif cfg.reg_mode == "vicreg_pooled":
                reg = R.vicreg_pooled(pooled, buffer)
            else:
                raise ValueError(f"unknown reg_mode {cfg.reg_mode!r}")
            buffer.push(pooled)
        parts["reg"] = reg.detach()
        total = cfg.lambda_reg * reg if total is None else total + cfg.lambda_reg * reg

    if cfg.use_inv and twin_pack is not None:
        # E4': `pack` is the coarse training mesh, `twin_pack` its fine-resolution
        # twin (the invariance view); the MSE is symmetric in the two.
        zc = model.encode(twin_pack["feats"])
        p_f = model.proj(model.pooled(z).mean(dim=0))
        p_c = model.proj(model.pooled(zc).mean(dim=0))
        inv = torch.nn.functional.mse_loss(p_f, p_c)
        parts["inv"] = inv.detach()
        total = total + cfg.lambda_inv * inv if total is not None else cfg.lambda_inv * inv

    return total, parts
