"""Label-free pretraining loops (AR and JEPA).

Plan v2.0 mapping: C1 (Amortized-Ritz), C4/E2 (full-JEPA arm), E4' (invariance on
multires pairs). Batch-1 over instances with the whole load battery batched inside
the step; anchor/feature caches are the verified runtime assets (plan Sec.2.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..anchor.energy import AnchorCache
from ..models.fejepa import mesh_adjacency
from ..models.regularizers import PooledBuffer
from .losses import AR_CONFIG, LossConfig, compute_loss
from .schedule import make_scheduler


@dataclass
class PretrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    loss: LossConfig = field(default_factory=lambda: AR_CONFIG)
    log_every: int = 0            # 0 = auto (~10 milestones), -1 = silent, N = every N
    desc: str = ""                # progress tag, e.g. "E8 AR pool1024 s0"
    precision: str = "fp32"       # fp32 | bf16 (r10: network compute autocast;
                                  # the energy anchor self-protects to fp32)
    ckpt_path: str | None = None  # R9: epoch-boundary checkpoint file (always
                                  # written when set; consumed only if resume)
    ckpt_every_epochs: int = 1
    resume: bool = False          # R9: resume from ckpt_path if present
    stop_after_epoch: int | None = None   # test hook: return after this many
                                          # epochs (simulates an interruption)


def pretrain(model, archs, cfg: PretrainConfig, pairs=None) -> dict:
    """Train on `archs` (or on fine/coarse `pairs` for E4'); returns a small history.

    ``pairs``: list of (fine_arch, coarse_arch); training runs on the coarse mesh with
    the fine twin as the invariance view (plan E4' protocol).
    """
    import torch

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = model.to(cfg.device)
    model.train()

    items = pairs if pairs is not None else [(a, None) for a in archs or []]
    if not items:
        raise ValueError("pretrain: empty training set (check pool/pair splits)")
    anchors = AnchorCache(device=cfg.device)
    prepared = []
    for main, twin in items:
        train_arch = twin if twin is not None else main     # coarse for E4'
        pack = model.prepare_instance(train_arch, cfg.device)
        twin_pack = model.prepare_instance(main, cfg.device) if twin is not None else None
        # adjacency feeds region masking only; AR (phys-only) never touches it,
        # and building it for a 1024-arch pool costs Python seconds per unit
        adj = (mesh_adjacency(train_arch.elements, train_arch.n_nodes)
               if cfg.loss.use_pred else None)
        prepared.append((train_arch, pack, twin_pack, adj))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(prepared)
    sched = make_scheduler(opt, total_steps)
    every = total_steps + 1 if cfg.log_every < 0 else \
        (cfg.log_every or max(1, total_steps // 10))
    buffer = PooledBuffer()

    from contextlib import nullcontext
    dev_type = "cuda" if str(cfg.device).startswith("cuda") else "cpu"
    def _ac():
        if cfg.precision == "bf16":
            return torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
        return nullcontext()

    history = {"loss": []}
    step = 0
    start_epoch = 0
    if cfg.ckpt_path and cfg.resume and not cfg.loss.use_pred:
        from .checkpoint import load_epoch_checkpoint

        ck = load_epoch_checkpoint(cfg.ckpt_path, model=model, opt=opt,
                                   sched=sched, rng=rng, device=cfg.device)
        if ck is not None:
            start_epoch, step, extra = ck
            history = extra.get("history", history)
            print(f"[ckpt] resumed {cfg.desc or 'pretrain'} at epoch "
                  f"{start_epoch}/{cfg.epochs} (step {step})", flush=True)
    for _epoch in range(start_epoch, cfg.epochs):
        order = rng.permutation(len(prepared))
        for i in order:
            train_arch, pack, twin_pack, adj = prepared[i]
            with _ac():
                loss, parts = compute_loss(model, pack, anchors.get(train_arch),
                                           adj, buffer, rng, cfg.loss,
                                           twin_pack=twin_pack)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            sched.step()
            step += 1
            if step % every == 0 or step == total_steps:
                tag = f" {cfg.desc}" if cfg.desc else ""
                print(f"[pretrain{tag}] step {step}/{total_steps} "
                      f"({100.0 * step / total_steps:.0f}%) "
                      f"loss={float(loss):.4e}", flush=True)
        history["loss"].append(float(loss.detach()))
        if cfg.ckpt_path and (_epoch + 1) % max(1, cfg.ckpt_every_epochs) == 0:
            from .checkpoint import save_epoch_checkpoint

            save_epoch_checkpoint(cfg.ckpt_path, epochs_done=_epoch + 1,
                                  step=step, model=model, opt=opt, sched=sched,
                                  rng=rng, extra={"history": history})
        if cfg.stop_after_epoch is not None and _epoch + 1 >= cfg.stop_after_epoch:
            return history                     # simulated interruption (tests)
    return history


def amortized_ritz(model, archs, cfg: PretrainConfig) -> dict:
    """The AR regime (plan C1): anchor only; by Lemma 1 the per-instance fixed point is
    the FE solution -- zero labels consumed."""
    from dataclasses import replace

    return pretrain(model, archs, replace(cfg, loss=AR_CONFIG))
