"""Training entry points: Gate G0 sanity check and the pretraining loop."""

from fejepa.train.g0 import GateG0Result, run_gate_g0
from fejepa.train.pretrain import (
    PretrainConfig,
    amortized_ritz,
    pretrain,
    pretrain_on_archs,
)
from fejepa.train.supervised import (
    SupervisedConfig,
    label_efficiency_sweep,
    train_supervised,
)

__all__ = [
    "GateG0Result",
    "run_gate_g0",
    "PretrainConfig",
    "pretrain",
    "pretrain_on_archs",
    "amortized_ritz",
    "SupervisedConfig",
    "train_supervised",
    "label_efficiency_sweep",
]
