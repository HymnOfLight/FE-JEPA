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
from fejepa.device import resolve_device
from fejepa.train.schedule import make_scheduler


@dataclass
class SupervisedConfig:
    epochs: int = 50
    lr: float = 3e-3
    weight_decay: float = 1e-5
    seed: int = 0
    model: FEJEPAConfig = field(default_factory=FEJEPAConfig)
    grad_clip: float = 1.0
    schedule: str = "cosine"
    warmup_frac: float = 0.05
    device: str = "auto"
    # Physics-informed fine-tuning: add the assembled-energy anchor (Lemma 1),
    # normalised per instance by |Pi(U*)| so the term is scale-comparable. By
    # the gradient identity this is the supervised energy-norm gradient, so it
    # is expected to accelerate convergence. lambda_phys=0 disables it (E1).
    lambda_phys: float = 0.0
    # Gradient-balanced anchor: cap the physics-gradient norm to
    # ``phys_balance_ratio * ||label gradient||`` so a fixed lambda cannot
    # overwhelm the label signal when labels are plentiful.
    phys_grad_balance: bool = False
    phys_balance_ratio: float = 1.0


def _pi_star_norm(arch: InstanceArchive) -> float:
    """Mean over loads of ``|Pi_h(U*)|`` -- the per-instance energy scale."""

    U = arch.U_star
    pi = 0.5 * np.einsum("bi,bi->b", U, (arch.K @ U.T).T) - np.einsum(
        "bi,bi->b", arch.F, U
    )
    return float(np.mean(np.abs(pi)) + 1e-30)


def _instance_loss(
    model: FEJEPA,
    arch: InstanceArchive,
    dtype: torch.dtype,
    lambda_phys: float = 0.0,
    anchor: "EnergyAnchor | None" = None,
    pi_norm: float | None = None,
    device="cpu",
) -> tuple[torch.Tensor, float]:
    """Supervised displacement rel-L2 loss, optionally + energy anchor.

    Returns ``(loss, disp_rel_l2_value)``; a single forward per load feeds both
    the displacement term and (if enabled) the physics anchor.
    """

    free = torch.as_tensor(arch.free_mask.astype(np.float64), dtype=dtype, device=device)
    disp_terms = []
    u_rows = []
    for j in range(arch.n_loads):
        feats = build_node_features(arch, j, dtype=dtype, device=device)
        _, disp = model.encode_decode(feats)
        u = disp.reshape(-1) * free
        u_rows.append(u)
        target = torch.as_tensor(arch.U_star[j], dtype=dtype, device=device)
        num = torch.linalg.vector_norm(u - target)
        den = torch.linalg.vector_norm(target) + 1e-12
        disp_terms.append(num / den)
    disp_loss = torch.stack(disp_terms).mean()

    phys_loss = None
    if lambda_phys > 0.0 and anchor is not None:
        u_stack = torch.stack(u_rows, dim=0)
        phys_loss = anchor(u_stack, reduction="mean") / (pi_norm or 1.0)
    return disp_loss, phys_loss, float(disp_loss.detach())


def _global_grad_norm(grads) -> torch.Tensor:
    sq = [g.pow(2).sum() for g in grads if g is not None]
    if not sq:
        return torch.zeros(())
    return torch.sqrt(torch.stack(sq).sum())


def _backward_balanced(
    params,
    disp_loss: torch.Tensor,
    phys_loss: torch.Tensor,
    lambda_phys: float,
    ratio: float,
) -> float:
    """Set ``p.grad`` to the label gradient plus a magnitude-capped physics one.

    The physics-gradient norm is capped to ``ratio * ||label gradient||`` so the
    anchor complements the label signal without overwhelming it once labels are
    plentiful (motivated by the E1 result: fixed lambda helps at 16 labels but
    hurts at 64).  Returns the effective physics weight used.
    """

    gl = torch.autograd.grad(disp_loss, params, retain_graph=True, allow_unused=True)
    gp = torch.autograd.grad(phys_loss, params, retain_graph=True, allow_unused=True)
    nl = _global_grad_norm(gl)
    npx = _global_grad_norm(gp)
    w = min(lambda_phys, float(ratio * nl / (npx + 1e-12)))
    for p, a, b in zip(params, gl, gp):
        g = None
        if a is not None:
            g = a.clone()
        if b is not None:
            g = b.mul(w) if g is None else g.add_(b, alpha=w)
        p.grad = g
    return w


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
    device = resolve_device(cfg.device)
    model = FEJEPA(cfg.model).to(dtype).to(device)
    if init_ckpt is not None:
        load_pretrained_into(model, init_ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = make_scheduler(
        opt, cfg.epochs * max(1, len(train_archs)), cfg.schedule, cfg.warmup_frac
    )

    # Pre-build per-instance energy anchors and energy scales (physics fine-tune).
    anchors: list = [None] * len(train_archs)
    pi_norms: list = [None] * len(train_archs)
    if cfg.lambda_phys > 0.0:
        from fejepa.anchor.energy import EnergyAnchor

        for i, a in enumerate(train_archs):
            anchors[i] = EnergyAnchor(a.K, a.F, a.free_mask, dtype=dtype, device=device)
            pi_norms[i] = _pi_star_norm(a)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(train_archs))
        epoch_loss = 0.0
        params = list(model.parameters())
        for idx in order:
            opt.zero_grad()
            disp_loss, phys_loss, disp_val = _instance_loss(
                model, train_archs[idx], dtype,
                lambda_phys=cfg.lambda_phys, anchor=anchors[idx], pi_norm=pi_norms[idx],
                device=device,
            )
            if phys_loss is not None and cfg.phys_grad_balance:
                _backward_balanced(params, disp_loss, phys_loss, cfg.lambda_phys, cfg.phys_balance_ratio)
            else:
                total = disp_loss if phys_loss is None else disp_loss + cfg.lambda_phys * phys_loss
                total.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            epoch_loss += disp_val
        epoch_loss /= len(train_archs)
        history.append(epoch_loss)
        if verbose and (epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1):
            print(f"  [sup] epoch {epoch} train_rel_l2={epoch_loss:.4f}")

    val = [evaluate_instance(model, a, dtype=dtype, device=device) for a in val_archs]

    def _mean(key):
        vals = [v[key] for v in val if key in v]
        return float(np.mean(vals)) if vals else None

    return {
        "val_rel_l2_disp": _mean("rel_l2_disp"),
        "val_energy_gap_rel": _mean("energy_gap_rel"),
        "val_rel_l2_vm": _mean("rel_l2_vm"),
        "val_max_vm_rel_err": _mean("max_vm_rel_err"),
        "val_crit_recall": _mean("crit_recall"),
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
