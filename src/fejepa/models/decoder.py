"""Field decoder g_psi: per-node latent -> nodal displacement.

A lightweight per-node head conditioned on the pooled instance latent, so the
decoder is mesh-size-agnostic and emits an ``(n_nodes, 2)`` displacement field on
the instance's own mesh.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FieldDecoder(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 128, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, latents: torch.Tensor, pooled: torch.Tensor | None = None) -> torch.Tensor:
        """``latents``: ``(n_nodes, dim)`` -> displacement ``(n_nodes, 2)``.

        The pooled latent is broadcast and concatenated so each node head sees a
        global instance descriptor (load case, geometry).
        """

        if pooled is None:
            pooled = latents.mean(dim=0)
        g = pooled.unsqueeze(0).expand(latents.shape[0], -1)
        h = torch.cat([latents, g], dim=-1)
        return self.net(h)
