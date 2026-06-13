"""Learning-rate schedules shared by the training loops."""

from __future__ import annotations

import math

import torch


def make_scheduler(
    opt: torch.optim.Optimizer,
    total_steps: int,
    schedule: str = "cosine",
    warmup_frac: float = 0.05,
    min_lr_frac: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a per-step LambdaLR scheduler.

    ``cosine`` warms up linearly for ``warmup_frac`` of training then decays to
    ``min_lr_frac`` of the base LR; ``constant`` keeps the base LR throughout.
    """

    total_steps = max(1, total_steps)
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step: int) -> float:
        if schedule == "constant":
            return 1.0
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_frac + (1.0 - min_lr_frac) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
