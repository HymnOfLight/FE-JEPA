"""Training entry points: Gate G0 sanity check and the pretraining loop."""

from fejepa.train.g0 import GateG0Result, run_gate_g0
from fejepa.train.pretrain import PretrainConfig, pretrain

__all__ = ["GateG0Result", "run_gate_g0", "PretrainConfig", "pretrain"]
