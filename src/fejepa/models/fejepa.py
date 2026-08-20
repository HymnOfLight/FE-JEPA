"""FE-JEPA model stack: encoder, field decoder, cross-attention predictor, region
masking, and the bundled module.

Plan v2.0 mapping:
  - Encoder: full self-attention over mesh nodes (verified asset; O(N^2) is adequate for
    Phase-1 plates; a linear-attention encoder is a WP4/Phase-2 decision and is
    intentionally absent -- see PLAN_MAP.md).
  - WP2 redesign, shipped here so E2 can render its verdict (plan Sec.5 item 6):
      * BFS-grown contiguous *region* masking replaces uniform node dropout
        (uniform dropout removed as superseded);
      * the predictor cross-attends from target descriptors to the *set* of context
        latents (replaces the single pooled context vector).
  - ``predictor_stop_grad``: target latents are detached (JEPA convention).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .features import FeatureSpec, battery_fscale, build_features_battery


# ------------------------------------------------------------ region masking --

def element_edges(elements: np.ndarray) -> np.ndarray:
    """(2, 2*|edges|) unique undirected element edges, both directions, for any
    fixed-arity simplex row layout: (E, 3) triangles and (E, 4) tetrahedra alike
    (WP7 3D-P0.1). For triangles the output is bit-identical to the v2.1.5
    triangle-only construction (sorted-unique pairs, then both directions)."""
    el = np.asarray(elements)
    k = int(el.shape[1])
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    e = np.concatenate([el[:, [a, b]] for a, b in pairs], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.concatenate([e, e[:, ::-1]], axis=0).T


def mesh_adjacency(elements: np.ndarray, n_nodes: int) -> list[np.ndarray]:
    """Undirected node adjacency from element edges (triangles AND tetrahedra).

    WP7 3D-P0.1: the v2.1.5 body unpacked rows as ``for a, b, c in elements`` and
    crashed on (E, 4) tets; rewritten as pairwise combinations per row. For
    triangles the per-row insertion sequence -- (a,b), (a,c), (b,c), each both
    ways -- reproduces the old ``update`` order exactly, so the returned
    neighbour arrays (and hence region-mask RNG consumption downstream) are
    bit-identical in 2D."""
    nbrs = [set() for _ in range(n_nodes)]
    k = int(np.asarray(elements).shape[1])
    for row in np.asarray(elements):
        for a in range(k):
            for b in range(a + 1, k):
                nbrs[row[a]].add(int(row[b]))
                nbrs[row[b]].add(int(row[a]))
    return [np.fromiter(s, dtype=np.int64) for s in nbrs]


def region_target_mask(adj: list[np.ndarray], frac: float,
                       rng: np.random.Generator) -> np.ndarray:
    """BFS-grown contiguous target regions covering ~frac of the nodes (plan WP2)."""
    n = len(adj)
    need = int(np.clip(round(frac * n), 1, n - 1))
    target = np.zeros(n, dtype=bool)
    while target.sum() < need:
        candidates = np.nonzero(~target)[0]
        frontier = deque([int(rng.choice(candidates))])
        while frontier and target.sum() < need:
            v = frontier.popleft()
            if target[v]:
                continue
            target[v] = True
            nxt = adj[v][~target[adj[v]]]
            rng.shuffle(nxt)
            frontier.extend(int(x) for x in nxt)
    return target


# ------------------------------------------------------------------ modules --

@dataclass
class FEJEPAConfig:
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mask_frac: float = 0.4
    predictor_stop_grad: bool = True
    scale_decode: bool = True   # WP7 3D-P0.5: multiply battery fscale back in decode
    features: FeatureSpec = field(default_factory=FeatureSpec)

    def to_dict(self) -> dict:
        return {"dim": self.dim, "depth": self.depth, "heads": self.heads,
                "mask_frac": self.mask_frac,
                "predictor_stop_grad": self.predictor_stop_grad,
                "scale_decode": self.scale_decode,
                "features": self.features.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "FEJEPAConfig":
        import dataclasses

        d = dict(d or {})
        feats = FeatureSpec.from_dict(d.pop("features", None))
        known = {f.name for f in dataclasses.fields(cls)} - {"features"}
        return cls(features=feats, **{k: v for k, v in d.items() if k in known})


def _nn():
    import torch
    from torch import nn

    return torch, nn


def build_encoder(in_dim: int, dim: int, depth: int, heads: int):
    torch, nn = _nn()

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.ln2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                     nn.Linear(4 * dim, dim))

        def forward(self, x):
            h = self.ln1(x)
            x = x + self.attn(h, h, h, need_weights=False)[0]
            return x + self.mlp(self.ln2(x))

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(in_dim, dim)
            self.blocks = nn.ModuleList(Block() for _ in range(depth))
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):                      # (..., N, in_dim) -> (..., N, dim)
            squeeze = x.dim() == 2
            if squeeze:
                x = x.unsqueeze(0)
            x = self.inp(x)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x.squeeze(0) if squeeze else x

    return Encoder()


def build_fejepa(cfg: FEJEPAConfig):
    torch, nn = _nn()
    in_dim = cfg.features.dim

    class FieldDecoder(nn.Module):
        """Per-node MLP on [node latent || pooled latent] -> spatial_dim displacement
        components (WP7 3D-P0: 2 in 2D, unchanged; 3 in 3D)."""

        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(2 * cfg.dim, 2 * cfg.dim), nn.GELU(),
                                     nn.Linear(2 * cfg.dim, cfg.dim), nn.GELU(),
                                     nn.Linear(cfg.dim, int(cfg.features.spatial_dim)))

        def forward(self, z):        # (..., N, dim) -> (..., N, spatial_dim)
            pooled = z.mean(dim=-2, keepdim=True).expand_as(z)
            return self.mlp(torch.cat([z, pooled], dim=-1))

    class CrossAttentionPredictor(nn.Module):
        """Targets attend over the *set* of context latents (plan WP2)."""

        def __init__(self):
            super().__init__()
            self.q = nn.Sequential(nn.Linear(in_dim, cfg.dim), nn.GELU(),
                                   nn.Linear(cfg.dim, cfg.dim))
            self.attn = nn.MultiheadAttention(cfg.dim, cfg.heads, batch_first=True)
            self.out = nn.Sequential(nn.LayerNorm(cfg.dim),
                                     nn.Linear(cfg.dim, cfg.dim), nn.GELU(),
                                     nn.Linear(cfg.dim, cfg.dim))

        def forward(self, target_desc, ctx_latents):    # (T,in_dim),(C,dim) -> (T,dim)
            q = self.q(target_desc).unsqueeze(0)
            kv = ctx_latents.unsqueeze(0)
            h = self.attn(q, kv, kv, need_weights=False)[0].squeeze(0)
            return h + self.out(h)

    class FEJEPA(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.encoder = build_encoder(in_dim, cfg.dim, cfg.depth, cfg.heads)
            self.decoder = FieldDecoder()
            self.predictor = CrossAttentionPredictor()
            self.proj = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(),
                                      nn.Linear(cfg.dim, cfg.dim))   # invariance head

        # ---- instance interface shared with the MGN baseline -----------------
        def prepare_instance(self, arch, device):
            feats = torch.as_tensor(build_features_battery(arch, cfg.features),
                                    device=device)
            free = torch.as_tensor(arch.free_mask, device=device).float()
            fscale = torch.as_tensor(battery_fscale(arch.F), dtype=feats.dtype,
                                     device=device)
            return {"feats": feats, "free": free, "fscale": fscale,
                    "arch": arch}

        def forward_instance(self, pack):
            """(L, ndof) masked displacement battery, differentiable."""
            z = self.encoder(pack["feats"])
            u = self.decoder(z)
            L = u.shape[0]
            u = u.reshape(L, -1) * pack["free"]
            if self.cfg.scale_decode:      # WP7 3D-P0.5: exact by linearity
                u = u * pack["fscale"]
            return u

        # ---- latent utilities -------------------------------------------------
        def encode(self, feats):
            return self.encoder(feats)

        @staticmethod
        def pooled(z):
            return z.mean(dim=-2)

        def masked_prediction(self, feats_one_load, adj, rng: np.random.Generator,
                              z_full=None):
            """JEPA loss for one load case with region masking + cross-attn prediction.

            ``z_full``: optional pre-computed latents for this load (the training step
            already encodes the whole battery for the physics term; reusing that row
            saves one encoder pass per step -- identical values and gradients, since
            targets are detached under ``predictor_stop_grad`` and the shared graph is
            mathematically the same function otherwise)."""
            if z_full is None:
                z_full = self.encoder(feats_one_load)                # (N, dim)
            tgt = region_target_mask(adj, cfg.mask_frac, rng)
            t_idx = torch.as_tensor(np.nonzero(tgt)[0], device=z_full.device)
            c_idx = torch.as_tensor(np.nonzero(~tgt)[0], device=z_full.device)
            z_t = z_full.index_select(0, t_idx)
            if cfg.predictor_stop_grad:
                z_t = z_t.detach()
            feats_ctx = feats_one_load.clone()
            feats_ctx[t_idx] = 0.0
            z_ctx = self.encoder(feats_ctx).index_select(0, c_idx)
            pred = self.predictor(feats_one_load.index_select(0, t_idx), z_ctx)
            return torch.nn.functional.mse_loss(pred, z_t)

    return FEJEPA()


def load_pretrained_into(model, state_dict) -> int:
    """Copy name+shape-matched tensors (pretrain -> fine-tune seeding); returns count."""
    own = model.state_dict()
    matched = {k: v for k, v in state_dict.items()
               if k in own and own[k].shape == v.shape}
    own.update(matched)
    model.load_state_dict(own)
    return len(matched)
