"""Optimizer schedule: cosine decay with linear warmup (shared by all trainers)."""

from __future__ import annotations

import math


def make_scheduler(optimizer, total_steps: int, warmup_frac: float = 0.05):
    import torch

    warmup = max(1, int(total_steps * warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
