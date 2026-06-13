"""Label-free FE-JEPA pretraining loop over an instance archive directory."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fejepa.data.archive import load_problem, read_manifest
from fejepa.losses import LossConfig, compute_instance_loss
from fejepa.models.fejepa import FEJEPA, FEJEPAConfig
from fejepa.train.schedule import make_scheduler


@dataclass
class PretrainConfig:
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 0
    model: FEJEPAConfig = field(default_factory=FEJEPAConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    grad_clip: float = 1.0
    log_every: int = 20
    max_instances: int | None = None
    schedule: str = "cosine"
    warmup_frac: float = 0.05
    device: str = "cpu"


def _instance_files(data_dir: Path, max_instances: int | None) -> list[Path]:
    manifest = read_manifest(data_dir)
    files = [data_dir / rec["file"] for rec in manifest["instances"]]
    if max_instances is not None:
        files = files[:max_instances]
    return files


def amortized_ritz(
    archs: list,
    cfg: PretrainConfig | None = None,
    model: FEJEPA | None = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> tuple[FEJEPA, list[dict]]:
    """Label-free amortized-Ritz training: minimise the assembled energy only.

    Trains the encoder+decoder to minimise the assembled discrete energy across
    an instance distribution using **no labels**.  By Lemma 1 the per-instance
    fixed point is the FE solution, so this amortises the Ritz minimisation over
    geometries/loads -- the proposal's core label-free mechanism.
    """

    cfg = cfg or PretrainConfig()
    loss_cfg = LossConfig(
        use_phys=True, use_pred=False, use_sigreg=False, use_inv=False, lambda_E=1.0
    )
    return pretrain_on_archs(archs, cfg=cfg, loss_cfg=loss_cfg, model=model, dtype=dtype, verbose=verbose)


def pretrain_on_archs(
    archs: list,
    cfg: PretrainConfig | None = None,
    loss_cfg: LossConfig | None = None,
    coarse_archs: list | None = None,
    model: FEJEPA | None = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> tuple[FEJEPA, list[dict]]:
    """Pretrain on an in-memory list of archives, returning ``(model, history)``.

    If ``coarse_archs`` is provided (a same-length list of coarser meshings of
    the same BVPs), the cross-mesh invariance term is exercised; otherwise it is
    disabled.  ``loss_cfg`` lets callers toggle individual terms (ablations).
    """

    cfg = cfg or PretrainConfig()
    loss_cfg = loss_cfg if loss_cfg is not None else cfg.loss
    loss_cfg.use_inv = loss_cfg.use_inv and coarse_archs is not None

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = cfg.device
    if model is None:
        model = FEJEPA(cfg.model).to(dtype).to(device)
    else:
        model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = make_scheduler(
        opt, cfg.epochs * max(1, len(archs)), cfg.schedule, cfg.warmup_frac
    )

    history: list[dict] = []
    step = 0
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(archs))
        for idx in order:
            arch = archs[idx]
            coarse = coarse_archs[idx] if coarse_archs is not None else None
            model.train()
            opt.zero_grad()
            total, parts = compute_instance_loss(
                model, arch, cfg=loss_cfg, arch_coarse=coarse, rng=rng,
                dtype=dtype, device=device,
            )
            total.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            parts["epoch"] = epoch
            parts["step"] = step
            history.append(parts)
            if verbose and cfg.log_every and step % cfg.log_every == 0:
                msg = "  ".join(
                    f"{k}={v:.3e}" for k, v in parts.items()
                    if k in {"total", "phys", "pred", "sigreg", "inv"}
                )
                print(f"[pretrain] epoch {epoch} step {step}  {msg}")
            step += 1
    return model, history


def pretrain(
    data_dir: str | Path,
    out_ckpt: str | Path | None = None,
    cfg: PretrainConfig | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Pretrain FE-JEPA on a directory of instance archives.

    The cross-mesh invariance term is disabled here (archives are single
    resolution); use multi-resolution archives or :func:`compute_instance_loss`
    directly to exercise it.
    """

    cfg = cfg or PretrainConfig()
    data_dir = Path(data_dir)
    files = _instance_files(data_dir, cfg.max_instances)
    if not files:
        raise ValueError(f"no instances found in {data_dir}")

    archs = [load_problem(f) for f in files]
    loss_cfg = cfg.loss
    loss_cfg.use_inv = False  # single-resolution archives

    t0 = time.time()
    model, history = pretrain_on_archs(
        archs, cfg=cfg, loss_cfg=loss_cfg, dtype=dtype, verbose=True
    )
    step = len(history)
    elapsed = time.time() - t0
    result = {"history": history, "steps": step, "elapsed_sec": elapsed}

    if out_ckpt is not None:
        out_ckpt = Path(out_ckpt)
        out_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_cfg": cfg.model.__dict__,
                "steps": step,
            },
            out_ckpt,
        )
        result["checkpoint"] = str(out_ckpt)

    return result
