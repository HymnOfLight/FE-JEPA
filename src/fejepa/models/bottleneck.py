"""E2 prototype (wp8-lejepa): token-bottleneck FE surrogate.

Per-node transformers cost O(N^2) in attention; on 41k-node fine instances a
step takes ~12 s (Phase-2 bench). AeroJEPA-style bottleneck: aggregate the
mesh into M tokens seeded by farthest-point sampling, run attention over the
M tokens (O(M^2)), then decode every node from a CONTINUOUS blend of its
k nearest tokens (their latents and relative-position embeddings), and its
own embedding -- O(N k) decoding, resolution-independent by construction.
The output tail (mask by free dofs, scale by the battery scale) is identical
to FE-JEPA's, so the decoded u feeds the SAME exact energy anchor: nothing
about the label-free objective changes.

Stage 1.32 (PI decision, 29 Sep 2026, before the E2 bench): the decoder read
each node's single nearest token (hard Voronoi assignment); on the bench
meshes 32-59% of the mesh edges joined nodes decoded from different tokens,
so the decoded field was conditioned discontinuously while the energy
penalises every mismatch. Nodes now blend their k = `decode_k` (default 4)
nearest tokens with Franke-Little weights (`blend_weights`), which vanish
exactly where a token leaves a node's k nearest: the conditioning is a
continuous function of position. k = 1 is the former hard decoder. The
encoder's pooling (scatter-mean over the nearest-seed cells) is unchanged.

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
    decode_k: int = 4                      # Stage 1.32: tokens blended per node (1 = hard)
    features: FeatureSpec = field(default_factory=FeatureSpec)

    @classmethod
    def from_dict(cls, d: dict) -> "BottleneckConfig":
        d = dict(d)
        feats = d.pop("features", None)
        spec = FeatureSpec(**feats) if isinstance(feats, dict) else (feats or FeatureSpec())
        keep = {k: d[k] for k in ("dim", "depth", "heads", "n_tokens", "scale_decode",
                                  "decode_k") if k in d}
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


def blend_weights(x: np.ndarray, seeds_xyz: np.ndarray, k: int, chunk: int = 8192) -> tuple:
    """Continuous token blending for the decoder (Stage 1.32).

    For every point x_i: its k nearest seeds by exact distance (ties broken by
    seed index; seeds arrive in canonical coordinate order, so the rule is
    independent of the mesh's node numbering) and the Franke-Little (modified
    Shepard) weights
        w_ij  proportional to  ((R_i - d_ij)_+ / (R_i d_ij))^2,
    with R_i the distance to the (k+1)-th nearest seed, normalised to sum 1.
    A seed's weight is exactly 0 where it leaves (or enters) the point's k
    nearest, and R_i is itself continuous, so the weights -- and every blend
    built from them -- are continuous functions of position; k = 1 gives
    weight 1 on the nearest seed (the hard assignment). With no (k+1)-th
    seed (M <= k) the weights are plain inverse-square Shepard weights over
    all seeds. A point on a seed gets weight ~1 on it (distances floored at
    1e-12); k+1 seeds exactly equidistant (measure zero) get equal weights.

    Returns (idx (N, kk) int64, w (N, kk) float64, rel (N, kk, sd) float64),
    kk = min(k, M), neighbours sorted nearest first; rel = x_i - seed_j."""
    n, m, sd = x.shape[0], seeds_xyz.shape[0], x.shape[1]
    kk = min(int(k), m)
    kq = min(int(k) + 1, m)
    s2 = (seeds_xyz ** 2).sum(1)
    idx = np.empty((n, kk), dtype=np.int64)
    w = np.empty((n, kk))
    rel = np.empty((n, kk, sd))
    for a in range(0, n, chunk):
        blk = x[a:a + chunk]
        b = blk.shape[0]
        if kq < m:                         # candidates: the kq smallest (Gram distances)
            d2 = (blk ** 2).sum(1)[:, None] + s2[None, :] - 2.0 * blk @ seeds_xyz.T
            cand = np.argpartition(d2, kq - 1, axis=1)[:, :kq]
        else:
            cand = np.broadcast_to(np.arange(m), (b, m)).copy()
        cand.sort(axis=1)                  # seed index ascending: the tie-break
        diff = blk[:, None, :] - seeds_xyz[cand]                   # exact offsets
        dist = np.sqrt((diff ** 2).sum(-1))
        o = np.argsort(dist, axis=1, kind="stable")
        cand = np.take_along_axis(cand, o, 1)
        dist = np.take_along_axis(dist, o, 1)
        diff = np.take_along_axis(diff, o[..., None], 1)
        dk = np.maximum(dist[:, :kk], 1e-12)
        if kq > kk:                        # a (k+1)-th seed exists: Franke-Little
            R = dist[:, kk:kk + 1]
            wh = (np.clip(R - dist[:, :kk], 0.0, None) / (R * dk)) ** 2
        else:                              # M <= k: all seeds, inverse-square Shepard
            wh = 1.0 / dk ** 2
        tot = wh.sum(1, keepdims=True)
        deg = ~(np.isfinite(tot[:, 0]) & (tot[:, 0] > 0.0))
        if deg.any():                      # k+1 equidistant seeds: equal weights
            wh[deg], tot[deg] = 1.0, float(kk)
        idx[a:a + b] = cand[:, :kk]
        w[a:a + b] = wh / tot
        rel[a:a + b] = diff[:, :kk]
    return idx, w, rel


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
            def enc_layer():
                return nn.TransformerEncoderLayer(cfg.dim, cfg.heads, 4 * cfg.dim,
                                                  dropout=0.0, batch_first=True,
                                                  norm_first=True, activation="gelu")

            self.tok_enc = nn.TransformerEncoder(enc_layer(), cfg.depth,
                                                 enable_nested_tensor=False)
            # Stage 1.31: nn.TransformerEncoder deep-copies ONE initialised
            # layer, so every layer would start from identical weights; the
            # FE-JEPA baseline initialises its blocks independently. Layers
            # 1..depth-1 get their own draws (layer 0 keeps the first one).
            for i in range(1, cfg.depth):
                self.tok_enc.layers[i] = enc_layer()
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
            # Stage 1.32: one neighbour search serves both sides -- the encoder
            # pools each node into its nearest seed's cell (column 0), the
            # decoder blends the k nearest with continuous weights
            nbr, wts, rel = blend_weights(xyz, xyz[seeds], cfg.decode_k)
            f32 = feats.dtype
            return {"feats": feats, "free": free, "fscale": fscale, "arch": arch,
                    "tok_idx": torch.as_tensor(np.ascontiguousarray(nbr[:, 0]), device=device),
                    "nbr_idx": torch.as_tensor(nbr, device=device),
                    "nbr_w": torch.as_tensor(wts, dtype=f32, device=device),
                    "nbr_rel": torch.as_tensor(rel, dtype=f32, device=device),
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
            """token latents (L, M, dim) -> (L, ndof) masked, scaled displacement.
            Each node reads the blend sum_j w_ij (z_tok(j), rel_pos(x_i - s_j))
            over its k nearest tokens (continuous in position; k = 1 is the
            former hard nearest-token decoder)."""
            h = self.node_embed(pack["feats"])                      # (L, N, dim)
            idx, w = pack["nbr_idx"], pack["nbr_w"]                 # (N, k), (N, k)
            zt = None
            for j in range(idx.shape[1]):                           # no (L, N, k, dim) gather
                term = z[:, idx[:, j], :] * w[:, j].view(1, -1, 1)
                zt = term if zt is None else zt + term              # (L, N, dim)
            r = (self.rel_pos(pack["nbr_rel"]) * w.unsqueeze(-1)).sum(1)   # (N, dim)
            r = r.unsqueeze(0).expand_as(h)
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
