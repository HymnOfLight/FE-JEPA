"""Gate G0: differentiable-assembly / neural-solver sanity check.

The proposal's Gate G0: *training ``g_psi`` alone on single instances must
converge to the FE solution in energy norm.*  This re-derives a neural linear
solver, but it validates the entire differentiable-assembly path and Lemma 1
numerically.  We optimise the encoder+decoder of a fresh model on a single
instance using **only** the assembled-energy anchor (no labels) and check that
the decoded field reaches the FE solution in both energy gap and relative L2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fejepa.data.archive import InstanceArchive
from fejepa.device import resolve_device
from fejepa.losses import physics_loss
from fejepa.metrics import evaluate_instance
from fejepa.models.fejepa import FEJEPA, FEJEPAConfig


@dataclass
class GateG0Result:
    energy_gap: float
    energy_gap_rel: float
    rel_l2_disp: float
    passed: bool
    history: list[float]

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[Gate G0 {status}] rel_L2_disp={self.rel_l2_disp:.4e} "
            f"energy_gap={self.energy_gap:.4e} "
            f"energy_gap_rel={self.energy_gap_rel:.4e}"
        )


def run_gate_g0(
    arch: InstanceArchive,
    steps: int = 2500,
    lr: float = 5e-3,
    dim: int = 64,
    depth: int = 3,
    rel_l2_threshold: float = 0.1,
    energy_gap_rel_threshold: float = 1e-2,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    log_every: int = 0,
    device: str = "auto",
) -> GateG0Result:
    """Run Gate G0 on a single labelled instance."""

    if arch.U_star is None:
        raise ValueError("Gate G0 requires a labelled instance (U_star present)")

    device = resolve_device(device)
    torch.manual_seed(seed)
    cfg = FEJEPAConfig(dim=dim, depth=depth, heads=4)
    model = FEJEPA(cfg).to(dtype).to(device)
    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    history: list[float] = []
    for step in range(steps):
        model.train()
        opt.zero_grad()
        energy, _, _ = physics_loss(model, arch, dtype=dtype, device=device)
        energy.backward()
        opt.step()
        history.append(float(energy.detach()))
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  G0 step {step:5d}  energy={history[-1]:.6e}")

    metrics = evaluate_instance(model, arch, dtype=dtype, device=device)
    passed = (
        metrics["rel_l2_disp"] <= rel_l2_threshold
        and metrics["energy_gap_rel"] <= energy_gap_rel_threshold
    )
    return GateG0Result(
        energy_gap=metrics["energy_gap"],
        energy_gap_rel=metrics["energy_gap_rel"],
        rel_l2_disp=metrics["rel_l2_disp"],
        passed=passed,
        history=history,
    )
