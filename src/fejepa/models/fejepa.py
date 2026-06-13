"""FE-JEPA model: encoder + predictor + decoder + projection head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fejepa.models.decoder import FieldDecoder
from fejepa.models.encoder import NODE_FEATURE_DIM, GraphTransformerEncoder
from fejepa.models.predictor import LatentPredictor


@dataclass
class FEJEPAConfig:
    dim: int = 128
    depth: int = 4
    heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    proj_dim: int = 64
    predictor_stop_grad: bool = True


class FEJEPA(nn.Module):
    """Bundles the encoder, latent predictor, field decoder and projection head.

    The conditioning descriptor for the predictor and the cross-mesh projection
    head re-use the first ``cond_dim`` node features (normalised position +
    Dirichlet flags + load token), so no extra bookkeeping is needed.
    """

    def __init__(self, cfg: FEJEPAConfig | None = None):
        super().__init__()
        self.cfg = cfg or FEJEPAConfig()
        c = self.cfg
        self.encoder = GraphTransformerEncoder(
            in_dim=NODE_FEATURE_DIM,
            dim=c.dim,
            depth=c.depth,
            heads=c.heads,
            mlp_ratio=c.mlp_ratio,
            dropout=c.dropout,
        )
        self.predictor = LatentPredictor(dim=c.dim, cond_dim=NODE_FEATURE_DIM)
        self.decoder = FieldDecoder(dim=c.dim)
        self.proj_head = nn.Sequential(
            nn.Linear(c.dim, c.dim), nn.GELU(), nn.Linear(c.dim, c.proj_dim)
        )

    # -- core passes ---------------------------------------------------------
    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.encoder(feats)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Return displacement field ``(n_nodes, 2)``."""

        return self.decoder(latents, latents.mean(dim=0))

    def encode_decode(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(feats)
        disp = self.decode(latents)
        return latents, disp

    # -- JEPA masked latent prediction --------------------------------------
    def masked_prediction(
        self,
        feats: torch.Tensor,
        target_idx: torch.Tensor,
        context_idx: torch.Tensor,
        mask_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(predicted_target_latents, target_latents)``.

        Context features keep visible nodes and replace masked (target) node
        features with a learned/zero mask token; the predictor reconstructs the
        masked-region latents from the pooled context and per-target descriptors.
        """

        target_latents = self.encode(feats)[target_idx]
        if self.cfg.predictor_stop_grad:
            target_latents = target_latents.detach()

        ctx_feats = feats.clone()
        if mask_token is None:
            ctx_feats[target_idx] = 0.0
        else:
            ctx_feats[target_idx] = mask_token
        ctx_latents = self.encode(ctx_feats)
        context_summary = ctx_latents[context_idx].mean(dim=0)

        descriptors = feats[target_idx]
        pred = self.predictor(context_summary, descriptors)
        return pred, target_latents

    def project(self, latents: torch.Tensor) -> torch.Tensor:
        """Projection-head embedding of the pooled latent (for invariance)."""

        return self.proj_head(latents.mean(dim=0))
