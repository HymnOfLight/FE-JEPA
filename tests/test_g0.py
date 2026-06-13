"""Gate G0: the neural solver must reach the FE solution in energy norm."""

import numpy as np

from fejepa.fe.generator import GeneratorConfig, sample_instance
from fejepa.train.g0 import run_gate_g0


def test_gate_g0_converges():
    # Tiny instance so the gate runs quickly while still exercising the full
    # differentiable-assembly path (encoder + decoder + energy anchor).
    cfg = GeneratorConfig(
        width_range=(1.4, 1.4),
        height_range=(0.9, 0.9),
        mesh_size_frac=(0.28, 0.28),
        max_holes=0,
    )
    prob = sample_instance(np.random.default_rng(2), cfg, labelled=True)
    arch_like = prob  # FEProblem exposes the same attributes the gate needs
    # FEProblem and InstanceArchive share the attribute surface used by G0.
    res = run_gate_g0(
        arch_like,
        steps=1500,
        lr=5e-3,
        dim=48,
        depth=3,
        rel_l2_threshold=0.2,
        energy_gap_rel_threshold=5e-2,
        seed=0,
    )
    assert res.passed, str(res)
