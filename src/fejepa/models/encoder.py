"""Graph-transformer encoder with physics tokens.

A Transolver-/graph-transformer-class backbone that maps tokenised instance data
(node coordinates, Dirichlet flags, and the per-node load token) to per-node
latents.  For the small Phase-0 meshes we use full self-attention over nodes;
token count therefore decouples from any fixed grid, and the encoder is
mesh-size-agnostic.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fejepa.data.archive import InstanceArchive

NODE_FEATURE_DIM = 6  # [x, y, dirichlet_x, dirichlet_y, fx, fy]


def build_node_features(
    arch: InstanceArchive,
    load_idx: int,
    dtype: torch.dtype = torch.float32,
    device="cpu",
) -> torch.Tensor:
    """Build the ``(n_nodes, NODE_FEATURE_DIM)`` node feature matrix.

    Coordinates are centred and scaled to unit RMS radius for conditioning
    stability; the load token is the per-node consistent force vector of the
    selected load case.
    """

    nodes = np.asarray(arch.nodes, dtype=np.float64)
    centre = nodes.mean(axis=0, keepdims=True)
    coords = nodes - centre
    scale = np.sqrt((coords**2).sum(axis=1).mean()) + 1e-8
    coords = coords / scale

    dmask = arch.dirichlet_mask.reshape(-1, 2).astype(np.float64)  # (n_nodes, 2)
    # Normalise the load token by a single per-instance scale (max over the whole
    # load battery) so that relative load magnitudes are preserved across cases.
    fscale = np.abs(arch.F).max() + 1e-12
    f = arch.F[load_idx].reshape(-1, 2) / fscale

    feats = np.concatenate([coords, dmask, f], axis=1)
    return torch.as_tensor(feats, dtype=dtype, device=device)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class GraphTransformerEncoder(nn.Module):
    """Encode node features into per-node latents."""

    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.embed = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [_TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """``feats``: ``(n_nodes, in_dim)`` -> latents ``(n_nodes, dim)``."""

        x = self.embed(feats).unsqueeze(0)  # (1, N, dim)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.squeeze(0)

    def pooled(self, latents: torch.Tensor) -> torch.Tensor:
        """Mean-pooled instance latent ``(dim,)``."""

        return latents.mean(dim=0)
