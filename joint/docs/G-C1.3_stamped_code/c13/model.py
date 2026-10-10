"""The network of gate G-C1.3: the repository's encoder with a copy of its field decoder, node features from the
export, the output scale and the non-negative map on the contact unknowns."""
from __future__ import annotations

import numpy as np

from fejoint.family import RANGES


def features(ex: dict, q) -> np.ndarray:
    """(N, 23): centred and RMS-normalised coordinates (3), Dirichlet flags (3), the load over its largest component
    (3), the node flags of the export (6), the family parameters over their range maxima (8). No solve involved."""
    nodes = ex["nodes"]
    c = nodes - nodes.mean(axis=0, keepdims=True)
    c = c / (np.sqrt((c ** 2).sum(axis=1).mean()) + 1e-12)
    n = nodes.shape[0]
    dmask = ex["dirichlet"].reshape(n, 3).astype(float)
    f = ex["F"].reshape(n, 3) / (np.abs(ex["F"]).max() + 1e-12)
    flags = ex["node_flags"].astype(float)
    desc = np.array([getattr(q, k) / RANGES[k][1] for k in RANGES])
    return np.concatenate([c, dmask, f, flags, np.broadcast_to(desc, (n, desc.size))], axis=1)


def build(cfg: dict, in_dim: int):
    """(encoder, decoder): the repository's build_encoder and a copy of its FieldDecoder, last layer zeroed."""
    import torch
    from torch import nn
    from fejepa.models.fejepa import build_encoder

    m = cfg["model"]
    enc = build_encoder(in_dim, m["dim"], m["depth"], m["heads"])
    enc.use_checkpoint = bool(m["activation_checkpointing"])
    d = m["dim"]
    dec = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))
    with torch.no_grad():
        dec[-1].weight.zero_()
        dec[-1].bias.zero_()
    return enc, dec


def field(enc, dec, feats, nonneg, free, scale: float, T: float):
    """The predicted nodal vector (3N,): decoder output, softplus at temperature T on the non-negative unknowns,
    times the displacement scale, Dirichlet unknowns zeroed."""
    import torch
    from torch import nn

    z = enc(feats)                                           # (1, N, dim)
    pooled = z.mean(dim=-2, keepdim=True).expand_as(z)
    u = dec(torch.cat([z, pooled], dim=-1)).reshape(-1)
    u = torch.where(nonneg, T * nn.functional.softplus(u / T), u)
    return u * scale * free
