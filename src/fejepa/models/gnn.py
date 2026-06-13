"""MeshGraphNets-style GNN backbone (supervised baseline).

A compact encode-process-decode graph network operating on the mesh graph: node
and edge encoders, a stack of message-passing blocks with residual updates, and
a per-node displacement head.  Used as the supervised baseline called for in the
proposal's Phase-0 plan (alongside the Transolver-class transformer).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fejepa.data.archive import InstanceArchive


def build_edges(arch: InstanceArchive) -> torch.Tensor:
    """Undirected mesh edges as a ``(2, n_edges)`` index tensor (both directions)."""

    t = arch.elements  # (n_elems, 3)
    pairs = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]], axis=0)
    pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    pairs = np.unique(pairs, axis=0)
    return torch.as_tensor(pairs.T.copy(), dtype=torch.long)


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class MeshGNN(nn.Module):
    def __init__(self, in_dim: int = 6, dim: int = 96, depth: int = 4, out_dim: int = 2):
        super().__init__()
        self.node_enc = _mlp([in_dim, dim, dim])
        self.edge_enc = _mlp([3, dim, dim])  # rel-position (2) + length (1)
        self.node_upd = nn.ModuleList([_mlp([2 * dim, dim, dim]) for _ in range(depth)])
        self.edge_upd = nn.ModuleList([_mlp([3 * dim, dim, dim]) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = _mlp([dim, dim, out_dim])
        self.dim = dim

    def forward(
        self, feats: torch.Tensor, edge_index: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        rel = coords[dst] - coords[src]
        length = rel.norm(dim=-1, keepdim=True)
        e = self.edge_enc(torch.cat([rel, length], dim=-1))
        h = self.node_enc(feats)

        for e_upd, n_upd in zip(self.edge_upd, self.node_upd):
            e_in = torch.cat([e, h[src], h[dst]], dim=-1)
            e = e + e_upd(e_in)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, e)
            h = h + n_upd(torch.cat([h, agg], dim=-1))

        return self.head(self.norm(h))

    def encode_decode(self, feats, edge_index, coords):
        disp = self.forward(feats, edge_index, coords)
        return None, disp
