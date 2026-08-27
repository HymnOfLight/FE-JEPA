"""Supervised training (pipeline P-A) and fine-tuning (the FT half of P-B).

Plan v2.0 mapping:
  - Sec.5 item 4 (lambda policy): the pre-registered anchored configuration is the
    *gradient-balanced anchor* (``anchor_mode='balanced'``, ratio 1.0) -- the physics
    gradient's norm is capped at ratio * ||label gradient|| per step, which is
    scale-invariant and removes the lambda x budget fragility diagnosed in E1 (B3).
    ``anchor_mode='fixed'`` (Pi_h / |Pi(U*)| with a fixed lambda) is reported alongside;
    a lambda grid is E1's secondary axis.
  - Metrics/eval: validation is evaluated with the frozen hierarchy via
    :func:`fejepa.metrics.evaluate_model` (per-instance arrays included, plan B6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..anchor.energy import AnchorCache, pi_star_abs
from ..metrics import evaluate_model, torch_predictor
from .schedule import make_scheduler


@dataclass
class SupervisedConfig:
    epochs: int = 200
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    anchor_mode: str = "none"        # none | fixed | balanced
    lambda_phys: float = 1.0         # for anchor_mode='fixed'
    balance_ratio: float = 1.0       # for anchor_mode='balanced'
    log_every: int = 0               # 0 = auto (~10 milestones), -1 = silent, N = every N
    desc: str = ""                   # progress tag, e.g. "E1' b64 balanced s1"
    precision: str = "fp32"          # fp32 | bf16 (r10; anchor self-protects)

    def protocol(self) -> dict:
        return {"epochs": self.epochs, "lr": self.lr, "seed": self.seed,
                "anchor_mode": self.anchor_mode, "lambda_phys": self.lambda_phys,
                "balance_ratio": self.balance_ratio}


def _disp_loss(u, u_star, free):
    """Mean per-load relative L2 (torch), matching the numpy metric."""

    diff = (u - u_star) * free
    num = diff.norm(dim=-1)
    den = (u_star * free).norm(dim=-1) + 1e-30
    return (num / den).mean()


def _balanced_grads(model, disp_loss, phys_loss, ratio: float) -> float:
    """Set p.grad = g_label + s * g_phys with s = min(1, ratio*||g_l||/||g_p||).
    Returns s as a 0-dim tensor (callers accumulate on-device; one sync at the end)."""
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    g_l = torch.autograd.grad(disp_loss, params, retain_graph=True, allow_unused=True)
    g_p = torch.autograd.grad(phys_loss, params, allow_unused=True)

    def sqnorm(gs):
        return sum((g * g).sum() for g in gs if g is not None)

    nl = torch.sqrt(sqnorm(g_l) + 1e-30)
    np_ = torch.sqrt(sqnorm(g_p) + 1e-30)
    s = torch.clamp(ratio * nl / np_, max=1.0)   # stays on-device: no per-step sync
    for p, gl, gp in zip(params, g_l, g_p, strict=True):
        if gl is None and gp is None:
            p.grad = None
        else:
            g = (gl if gl is not None else 0.0)
            if gp is not None:
                g = g + s * gp
            p.grad = g
    return s


def train_supervised(model, train_archs, val_archs, cfg: SupervisedConfig,
                     pretrained_state=None) -> dict:
    """Train on labelled `train_archs`; evaluate the frozen metric suite on `val_archs`.

    ``pretrained_state``: optional state_dict to seed from (P-B fine-tuning); the number
    of matched tensors is recorded in the result.
    """
    import torch

    from ..models.fejepa import load_pretrained_into

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = model.to(cfg.device)
    n_loaded = load_pretrained_into(model, pretrained_state) if pretrained_state else 0
    model.train()

    if not train_archs:
        raise ValueError("train_supervised: empty training set "
                         "(check budget/pool splits)")
    anchors = AnchorCache(device=cfg.device)
    prepared = []
    for a in train_archs:
        if a.U_star is None:
            raise ValueError(f"supervised training needs labels: {a.path}")
        pack = model.prepare_instance(a, cfg.device)
        pack["u_star"] = torch.as_tensor(a.U_star, dtype=pack["free"].dtype,
                                         device=cfg.device)
        pack["pi_norm"] = float(pi_star_abs(a).mean())
        prepared.append((a, pack))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(prepared))
    sched = make_scheduler(opt, total_steps)
    every = total_steps + 1 if cfg.log_every < 0 else \
        (cfg.log_every or max(1, total_steps // 10))

    balance_scale_sum, balance_n = None, 0
    step = 0
    for _epoch in range(cfg.epochs):
        for i in rng.permutation(len(prepared)):
            arch, pack = prepared[i]
            if cfg.precision == "bf16":
                import torch as _t
                _dt = "cuda" if str(cfg.device).startswith("cuda") else "cpu"
                with _t.autocast(device_type=_dt, dtype=_t.bfloat16):
                    u = model.forward_instance(pack)
                u = u.float()
            else:
                u = model.forward_instance(pack)
            disp = _disp_loss(u, pack["u_star"], pack["free"])

            opt.zero_grad(set_to_none=True)
            if cfg.anchor_mode == "none":
                disp.backward()
            elif cfg.anchor_mode == "fixed":
                phys = anchors.get(arch).energies(u).mean() / pack["pi_norm"]
                (disp + cfg.lambda_phys * phys).backward()
            elif cfg.anchor_mode == "balanced":
                phys = anchors.get(arch).energies(u).mean()
                s = _balanced_grads(model, disp, phys, cfg.balance_ratio)
                balance_scale_sum = s if balance_scale_sum is None \
                    else balance_scale_sum + s
                balance_n += 1
            else:
                raise ValueError(f"unknown anchor_mode {cfg.anchor_mode!r}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            sched.step()
            step += 1
            if step % every == 0 or step == total_steps:
                tag = f" {cfg.desc}" if cfg.desc else ""
                print(f"[sup:{cfg.anchor_mode}{tag}] step {step}/{total_steps} "
                      f"({100.0 * step / total_steps:.0f}%) disp={float(disp):.4f}",
                      flush=True)

    out = {"protocol": cfg.protocol(),
           "val": evaluate_model(torch_predictor(model, cfg.device), val_archs),
           "pretrained_tensors_loaded": n_loaded}
    if balance_n:
        out["balance_scale_mean"] = float(balance_scale_sum) / balance_n
    return out
