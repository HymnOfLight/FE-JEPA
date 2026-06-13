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


def _instance_files(data_dir: Path, max_instances: int | None) -> list[Path]:
    manifest = read_manifest(data_dir)
    files = [data_dir / rec["file"] for rec in manifest["instances"]]
    if max_instances is not None:
        files = files[:max_instances]
    return files


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

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = FEJEPA(cfg.model).to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    loss_cfg = cfg.loss
    loss_cfg.use_inv = False  # single-resolution archives

    history: list[dict] = []
    step = 0
    t0 = time.time()
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(files))
        for idx in order:
            arch = load_problem(files[idx])
            model.train()
            opt.zero_grad()
            total, parts = compute_instance_loss(
                model, arch, cfg=loss_cfg, rng=rng, dtype=dtype
            )
            total.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            parts["epoch"] = epoch
            parts["step"] = step
            history.append(parts)
            if cfg.log_every and step % cfg.log_every == 0:
                msg = "  ".join(f"{k}={v:.3e}" for k, v in parts.items() if k in {"total", "phys", "pred", "sigreg"})
                print(f"[pretrain] epoch {epoch} step {step}  {msg}")
            step += 1

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
