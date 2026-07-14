"""MeshGraphNets-style supervised baseline.

Plan v2.0 mapping: Sec.4 -- "Baselines every headline table must contain: ...,
trained MeshGraphNets" -- and WP4/E8 (the MGN column). Implements the same
``prepare_instance`` / ``forward_instance`` interface as FEJEPA so the supervised
trainer and E8 are model-agnostic.
"""

from __future__ import annotations

import numpy as np

from .features import FeatureSpec, build_features_battery


def _edges(elements: np.ndarray) -> np.ndarray:
    e = np.concatenate([elements[:, [0, 1]], elements[:, [1, 2]],
                        elements[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.concatenate([e, e[:, ::-1]], axis=0).T      # (2, 2E) both directions


def build_mesh_gnn(dim: int = 128, depth: int = 8,
                   features: FeatureSpec | None = None):
    import torch
    from torch import nn

    spec = features or FeatureSpec()

    class MLP(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(i, dim), nn.GELU(),
                                     nn.Linear(dim, o), nn.LayerNorm(o))

        def forward(self, x):
            return self.net(x)

    class MeshGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg_dict = {"dim": dim, "depth": depth, "features": spec.to_dict()}
            self.node_enc = MLP(spec.dim, dim)
            self.edge_enc = MLP(3, dim)
            self.node_upd = nn.ModuleList(MLP(2 * dim, dim) for _ in range(depth))
            self.edge_upd = nn.ModuleList(MLP(3 * dim, dim) for _ in range(depth))
            self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                      nn.Linear(dim, 2))

        def prepare_instance(self, arch, device):
            feats = torch.as_tensor(build_features_battery(arch, spec), device=device)
            edges = torch.as_tensor(_edges(arch.elements), device=device)
            rel = arch.nodes[edges[1].cpu().numpy()] - arch.nodes[edges[0].cpu().numpy()]
            efeat = np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True)],
                                   axis=1)
            efeat = torch.as_tensor(efeat, dtype=feats.dtype, device=device)
            free = torch.as_tensor(arch.free_mask, device=device).float()
            return {"feats": feats, "edges": edges, "efeat": efeat, "free": free,
                    "arch": arch}

        def _one(self, x, edges, efeat):
            src, dst = edges[0], edges[1]
            h = self.node_enc(x)
            e = self.edge_enc(efeat)
            for nu, eu in zip(self.node_upd, self.edge_upd, strict=True):
                e = e + eu(torch.cat([e, h[src], h[dst]], dim=-1))
                agg = torch.zeros_like(h).index_add_(0, dst, e)
                h = h + nu(torch.cat([h, agg], dim=-1))
            return self.head(h)

        def forward_instance(self, pack):
            us = [self._one(pack["feats"][j], pack["edges"], pack["efeat"])
                  for j in range(pack["feats"].shape[0])]
            u = torch.stack(us, dim=0)
            return u.reshape(u.shape[0], -1) * pack["free"]

    return MeshGNN()
