"""E2 prototype (wp8-lejepa): token-bottleneck FE surrogate.

Per-node transformers cost O(N^2) in attention; on 41k-node fine instances a
step takes ~12 s (Phase-2 bench). AeroJEPA-style bottleneck: aggregate the
mesh into M tokens seeded by farthest-point sampling, run attention over the
M tokens (O(M^2)), then decode every node from its token's latent, its own
embedding and its relative position -- O(N) decoding, resolution-independent
by construction. The output tail (mask by free dofs, scale by the battery
scale) is identical to FE-JEPA's, so the decoded u feeds the SAME exact
energy anchor: nothing about the label-free objective changes.

Interface: `needs_pack = True` -- encode/decode take the instance pack
(token assignment lives there); `compute_loss` and the instruments pass it
when the flag is set and leave the legacy call path untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import FeatureSpec, battery_fscale, build_features_battery


@dataclass
class BottleneckConfig:
    dim: int = 128
    depth: int = 4
    heads: int = 4
    n_tokens: int = 512
    scale_decode: bool = True
    features: FeatureSpec = field(default_factory=FeatureSpec)

    @classmethod
    def from_dict(cls, d: dict) -> "BottleneckConfig":
        d = dict(d)
        feats = d.pop("features", None)
        spec = FeatureSpec(**feats) if isinstance(feats, dict) else (feats or FeatureSpec())
        keep = {k: d[k] for k in ("dim", "depth", "heads", "n_tokens", "scale_decode")
                if k in d}
        return cls(features=spec, **keep)


def _canonical_pick(cands: np.ndarray, x: np.ndarray) -> int:
    """Among tied candidates choose by lexicographic coordinates, never by
    index -- keeps the result independent of the mesh's node numbering."""
    if cands.size == 1:
        return int(cands[0])
    order = np.lexsort(x[cands].T[::-1])          # sort by x, then y, then z
    return int(cands[order[0]])


def farthest_point_sampling(x: np.ndarray, m: int) -> np.ndarray:
    """Deterministic, numbering-independent FPS on coordinates x (N, sd):
    first seed = node nearest the centroid, then iteratively the node farthest
    from the chosen set; ties broken by coordinates; the seed set is returned
    in canonical (lexicographic coordinate) order so token indices do not
    depend on node numbering either."""
    n = x.shape[0]
    m = min(m, n)
    d0 = ((x - x.mean(0)) ** 2).sum(1)
    start = _canonical_pick(np.flatnonzero(d0 == d0.min()), x)
    seeds = np.empty(m, dtype=np.int64)
    seeds[0] = start
    dmin = ((x - x[start]) ** 2).sum(1)
    for i in range(1, m):
        nxt = _canonical_pick(np.flatnonzero(dmin == dmin.max()), x)
        seeds[i] = nxt
        dmin = np.minimum(dmin, ((x - x[nxt]) ** 2).sum(1))
    return seeds[np.lexsort(x[seeds].T[::-1])]


def nearest_seed(x: np.ndarray, seeds_xyz: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Index of the nearest seed for every node (Voronoi assignment), chunked.
    Seeds arrive in canonical coordinate order (see farthest_point_sampling),
    so argmin's lowest-index tie-break is itself numbering-independent."""
    out = np.empty(x.shape[0], dtype=np.int64)
    s2 = (seeds_xyz ** 2).sum(1)
    for a in range(0, x.shape[0], chunk):
        blk = x[a:a + chunk]
        d = (blk ** 2).sum(1)[:, None] + s2[None, :] - 2.0 * blk @ seeds_xyz.T
        out[a:a + chunk] = np.argmin(d, axis=1)
    return out


def build_bottleneck(cfg: BottleneckConfig):
    import torch
    from torch import nn

    spec = cfg.features
    in_dim = spec.dim
    sd = int(spec.spatial_dim)                 # coordinates and displacement components

    def mlp(i, h, o):
        return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))

    class Bottleneck(nn.Module):
        needs_pack = True

        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.node_embed = mlp(in_dim, cfg.dim, cfg.dim)
            self.seed_pos = mlp(sd, cfg.dim, cfg.dim)
            self.rel_pos = mlp(sd, cfg.dim, cfg.dim)
            layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, 4 * cfg.dim,
                                               dropout=0.0, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.tok_enc = nn.TransformerEncoder(layer, cfg.depth,
                                                 enable_nested_tensor=False)
            self.tok_norm = nn.LayerNorm(cfg.dim)
            self.dec = mlp(3 * cfg.dim, cfg.dim, sd)         # (..., N, 3*dim) -> (..., N, sd)
            self.out_dim = sd

        # ---- instance interface (same pack contract as FE-JEPA + token geometry) ----
        def prepare_instance(self, arch, device):
            feats = torch.as_tensor(build_features_battery(arch, spec), device=device)
            free = torch.as_tensor(arch.free_mask, device=device).float()
            fscale = torch.as_tensor(battery_fscale(arch.F), dtype=feats.dtype, device=device)
            xyz = np.asarray(arch.nodes, dtype=np.float64)
            if xyz.shape[1] != sd:
                raise ValueError(f"bottleneck: features.spatial_dim={sd} but the mesh has "
                                 f"{xyz.shape[1]} coordinates")
            lo, hi = xyz.min(0), xyz.max(0)
            xyz = (xyz - lo) / max(float((hi - lo).max()), 1e-12)   # unit bbox
            seeds = farthest_point_sampling(xyz, cfg.n_tokens)
            assign = nearest_seed(xyz, xyz[seeds])
            rel = xyz - xyz[seeds][assign]
            f32 = feats.dtype
            return {"feats": feats, "free": free, "fscale": fscale, "arch": arch,
                    "tok_idx": torch.as_tensor(assign, device=device),
                    "rel": torch.as_tensor(rel, dtype=f32, device=device),
                    "seed_xyz": torch.as_tensor(xyz[seeds], dtype=f32, device=device),
                    "n_tok": int(seeds.shape[0])}

        def encode(self, feats, pack):
            """(L, N, in_dim) -> token latents (L, M, dim)."""
            h = self.node_embed(feats)                              # (L, N, dim)
            L, N, D = h.shape
            M = pack["n_tok"]
            idx = pack["tok_idx"]
            pooled = h.new_zeros(L, M, D).index_add(1, idx, h)      # scatter-sum
            cnt = h.new_zeros(M).index_add(0, idx, h.new_ones(N)).clamp_min(1.0)
            pooled = pooled / cnt.view(1, M, 1)                     # scatter-mean
            tok = pooled + self.seed_pos(pack["seed_xyz"]).unsqueeze(0)
            return self.tok_norm(self.tok_enc(tok))                 # (L, M, dim)

        def decode(self, z, pack):
            """token latents (L, M, dim) -> (L, ndof) masked, scaled displacement."""
            h = self.node_embed(pack["feats"])                      # (L, N, dim)
            zt = z[:, pack["tok_idx"], :]                           # (L, N, dim)
            r = self.rel_pos(pack["rel"]).unsqueeze(0).expand_as(h)
            u = self.dec(torch.cat([h, zt, r], dim=-1))             # (L, N, sd)
            u = u.reshape(u.shape[0], -1) * pack["free"]
            if cfg.scale_decode:
                u = u * pack["fscale"]
            return u

        def forward_instance(self, pack):
            return self.decode(self.encode(pack["feats"], pack), pack)

        @staticmethod
        def pooled(z):
            return z.mean(dim=-2)

    return Bottleneck()
