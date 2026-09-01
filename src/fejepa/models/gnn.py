"""MeshGraphNets-style supervised baseline.

Plan v2.0 mapping: Sec.4 -- "Baselines every headline table must contain: ...,
trained MeshGraphNets" -- and WP4/E8 (the MGN column). Implements the same
``prepare_instance`` / ``forward_instance`` interface as FEJEPA so the supervised
trainer and E8 are model-agnostic.
"""

from __future__ import annotations

import numpy as np

from .features import FeatureSpec, battery_fscale, build_features_battery
from .fejepa import element_edges


def _edges(elements: np.ndarray) -> np.ndarray:
    """(2, 2*|edges|) both directions -- generic over triangles and tetrahedra
    (WP7 3D-P0.1); bit-identical to the v2.1.5 construction on triangles."""
    return element_edges(elements)


def build_mesh_gnn(dim: int = 128, depth: int = 8,
                   features: FeatureSpec | None = None,
                   scale_decode: bool = True):
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint

    spec = features or FeatureSpec()

    def _mgn_layer(e, h, src, dst, nu, eu):
        """One message-passing layer: edge update, scatter-add, node update."""
        e = e + eu(torch.cat([e, h[src], h[dst]], dim=-1))
        agg = torch.zeros_like(h).index_add_(0, dst, e)
        h = h + nu(torch.cat([h, agg], dim=-1))
        return e, h

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
            self.cfg_dict = {"dim": dim, "depth": depth,
                             "scale_decode": scale_decode,
                             "features": spec.to_dict()}
            self.node_enc = MLP(spec.dim, dim)
            # rel-position (spatial_dim) + length (1); 2D value 3 unchanged
            self.edge_enc = MLP(int(spec.spatial_dim) + 1, dim)
            self.node_upd = nn.ModuleList(MLP(2 * dim, dim) for _ in range(depth))
            self.edge_upd = nn.ModuleList(MLP(3 * dim, dim) for _ in range(depth))
            self.head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                      nn.Linear(dim, int(spec.spatial_dim)))

        def prepare_instance(self, arch, device):
            feats = torch.as_tensor(build_features_battery(arch, spec), device=device)
            edges = torch.as_tensor(_edges(arch.elements), device=device)
            rel = arch.nodes[edges[1].cpu().numpy()] - arch.nodes[edges[0].cpu().numpy()]
            efeat = np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True)],
                                   axis=1)
            efeat = torch.as_tensor(efeat, dtype=feats.dtype, device=device)
            free = torch.as_tensor(arch.free_mask, device=device).float()
            fscale = torch.as_tensor(battery_fscale(arch.F), dtype=feats.dtype,
                                     device=device)
            return {"feats": feats, "edges": edges, "efeat": efeat, "free": free,
                    "fscale": fscale, "arch": arch}

        def _one(self, x, edges, efeat):
            src, dst = edges[0], edges[1]
            h = self.node_enc(x)
            e = self.edge_enc(efeat)
            # D9 (Phase-2): per-layer activation checkpointing in training mode.
            # Memory-only and exact (backward recomputes the identical ops on the
            # identical inputs); 3D tet meshes carry ~30 GiB of edge activations
            # across depth x load cases without it. use_checkpoint=False restores
            # the original path verbatim (tests assert bitwise equality on CPU).
            use_ckpt = (self.training and torch.is_grad_enabled()
                        and getattr(self, "use_checkpoint", True))
            for nu, eu in zip(self.node_upd, self.edge_upd, strict=True):
                if use_ckpt:
                    e, h = checkpoint(_mgn_layer, e, h, src, dst, nu, eu,
                                      use_reentrant=False)
                else:
                    e, h = _mgn_layer(e, h, src, dst, nu, eu)
            return self.head(h)

        def forward_instance(self, pack):
            us = [self._one(pack["feats"][j], pack["edges"], pack["efeat"])
                  for j in range(pack["feats"].shape[0])]
            u = torch.stack(us, dim=0)
            u = u.reshape(u.shape[0], -1) * pack["free"]
            if scale_decode:               # WP7 3D-P0.5: exact by linearity
                u = u * pack["fscale"]
            return u

    return MeshGNN()
