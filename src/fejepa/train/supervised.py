"""Supervised training / fine-tuning harness (baselines and label efficiency).

Trains the FE-JEPA backbone (or a from-scratch copy) on a small number of
labelled solves with a relative-L2 displacement loss.  Supports:

* from-scratch supervised baseline (no pretraining), and
* fine-tuning from a pretrained FE-JEPA checkpoint,

which is exactly the comparison behind RQ2 (label efficiency) and the
falsification battery in the proposal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fejepa.data.archive import InstanceArchive, load_problem
from fejepa.models.encoder import build_node_features
from fejepa.models.fejepa import FEJEPA, FEJEPAConfig
from fejepa.metrics import evaluate_instance


@dataclass
class SupervisedConfig:
    epochs: int = 50
    lr: float = 3e-3
    weight_decay: float = 1e-5
    seed: int = 0
    model: FEJEPAConfig = field(default_factory=FEJEPAConfig)
    grad_clip: float = 1.0


def _disp_loss(model: FEJEPA, arch: InstanceArchive, dtype: torch.dtype) -> torch.Tensor:
    free = torch.as_tensor(arch.free_mask.astype(np.float64), dtype=dtype)
    losses = []
    for j in range(arch.n_loads):
        feats = build_node_features(arch, j, dtype=dtype)
        _, disp = model.encode_decode(feats)
        u = disp.reshape(-1) * free
        target = torch.as_tensor(arch.U_star[j], dtype=dtype)
        num = torch.linalg.vector_norm(u - target)
        den = torch.linalg.vector_norm(target) + 1e-12
        losses.append(num / den)
    return torch.stack(losses).mean()


def load_pretrained_into(model: FEJEPA, ckpt_path: str | Path) -> int:
    """Load matching parameters from a pretrained checkpoint.

    Only tensors whose names and shapes match the target model are copied, so a
    pretrained encoder can seed a fine-tuning run even if some heads differ.
    Returns the number of tensors loaded.
    """

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    own = model.state_dict()
    compatible = {
        k: v for k, v in state.items() if k in own and own[k].shape == v.shape
    }
    own.update(compatible)
    model.load_state_dict(own)
    return len(compatible)


def train_supervised(
    train_archs: list[InstanceArchive],
    val_archs: list[InstanceArchive],
    cfg: SupervisedConfig | None = None,
    init_ckpt: str | Path | None = None,
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
) -> dict:
    """Train a supervised surrogate and return validation metrics."""

    cfg = cfg or SupervisedConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = FEJEPA(cfg.model).to(dtype)
    if init_ckpt is not None:
        load_pretrained_into(model, init_ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train_archs))
        epoch_loss = 0.0
        for idx in order:
            opt.zero_grad()
            loss = _disp_loss(model, train_archs[idx], dtype)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_loss += float(loss.detach())
        epoch_loss /= len(train_archs)
        history.append(epoch_loss)
        if verbose and (epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1):
            print(f"  [sup] epoch {epoch} train_rel_l2={epoch_loss:.4f}")

    val = [evaluate_instance(model, a, dtype=dtype) for a in val_archs]
    rel = float(np.mean([v["rel_l2_disp"] for v in val]))
    gap = float(np.mean([v["energy_gap_rel"] for v in val]))
    return {
        "val_rel_l2_disp": rel,
        "val_energy_gap_rel": gap,
        "train_history": history,
        "n_train": len(train_archs),
        "model": model,
    }


def label_efficiency_sweep(
    data_dir: str | Path,
    budgets: list[int],
    n_val: int = 32,
    init_ckpt: str | Path | None = None,
    cfg: SupervisedConfig | None = None,
    seed: int = 0,
) -> list[dict]:
    """Train at several label budgets and report validation rel-L2 per budget."""

    from fejepa.data.archive import read_manifest

    data_dir = Path(data_dir)
    manifest = read_manifest(data_dir)
    files = [data_dir / r["file"] for r in manifest["instances"]]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    val_files = [files[i] for i in perm[:n_val]]
    pool = [files[i] for i in perm[n_val:]]
    val_archs = [load_problem(f) for f in val_files]

    results = []
    for b in budgets:
        if b > len(pool):
            break
        train_archs = [load_problem(f) for f in pool[:b]]
        out = train_supervised(train_archs, val_archs, cfg=cfg, init_ckpt=init_ckpt)
        results.append(
            {"budget": b, "val_rel_l2_disp": out["val_rel_l2_disp"], "val_energy_gap_rel": out["val_energy_gap_rel"]}
        )
    return results
