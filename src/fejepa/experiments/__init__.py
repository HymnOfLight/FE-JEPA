"""Phase-1 experiments: the pre-registered falsification battery and Gate G1."""

from fejepa.experiments.falsification import (
    BatteryConfig,
    ExperimentResult,
    collect_pooled_latents,
    e1_anchor_value,
    e3_collapse,
    e5_naive_sanity,
    gate_g1,
    load_split,
    run_battery,
)

__all__ = [
    "BatteryConfig",
    "ExperimentResult",
    "collect_pooled_latents",
    "e1_anchor_value",
    "e3_collapse",
    "e5_naive_sanity",
    "gate_g1",
    "load_split",
    "run_battery",
    "compare_training_regimes",
]

from fejepa.experiments.regimes import compare_training_regimes
