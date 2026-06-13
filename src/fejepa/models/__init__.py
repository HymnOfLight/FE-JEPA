"""Neural backbones for FE-JEPA: encoder, predictor, decoder, SIGReg."""

from fejepa.models.decoder import FieldDecoder
from fejepa.models.encoder import GraphTransformerEncoder, build_node_features
from fejepa.models.fejepa import FEJEPA, FEJEPAConfig
from fejepa.models.predictor import LatentPredictor
from fejepa.models.sigreg import sigreg_loss

__all__ = [
    "FieldDecoder",
    "GraphTransformerEncoder",
    "build_node_features",
    "FEJEPA",
    "FEJEPAConfig",
    "LatentPredictor",
    "sigreg_loss",
]
