"""Latent predictor p_phi for masked latent prediction.

Predicts target-region latents from the pooled context latent and a per-target
descriptor ``m`` (here the target node's normalised position and load token).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    def __init__(self, dim: int = 128, cond_dim: int = 6, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, context: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        """Predict target latents.

        ``context``: ``(dim,)`` pooled context latent.
        ``descriptors``: ``(n_targets, cond_dim)`` per-target conditioning.
        Returns ``(n_targets, dim)``.
        """

        c = context.unsqueeze(0).expand(descriptors.shape[0], -1)
        h = torch.cat([c, descriptors], dim=-1)
        return self.net(h)
