"""Lemma-1 gradient identity at machine precision (plan Sec.1; audit V1 preserved)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.anchor.energy import EnergyAnchor, pi_h


def test_energy_matches_numpy(arch):
    anchor = EnergyAnchor(arch.K, arch.F, arch.dirichlet_mask, dtype=torch.float64)
    u = torch.as_tensor(arch.U_star, dtype=torch.float64)
    e_t = anchor.energies(u).numpy()
    e_np = pi_h(arch.U_star, arch.K, arch.F)
    assert np.allclose(e_t, e_np, rtol=1e-12, atol=1e-12)


def test_gradient_is_K_u_minus_F(arch, rng):
    anchor = EnergyAnchor(arch.K, arch.F, arch.dirichlet_mask, dtype=torch.float64)
    u_np = rng.standard_normal(arch.F.shape) * arch.free_mask
    u = torch.tensor(u_np, dtype=torch.float64, requires_grad=True)
    anchor.energies(u).sum().backward()
    expected = ((arch.K @ u_np.T).T - arch.F) * arch.free_mask
    assert np.allclose(u.grad.numpy(), expected, rtol=1e-12, atol=1e-12)


def test_gradient_zero_at_solution(arch):
    anchor = EnergyAnchor(arch.K, arch.F, arch.dirichlet_mask, dtype=torch.float64)
    u = torch.tensor(arch.U_star, dtype=torch.float64, requires_grad=True)
    anchor.energies(u).sum().backward()
    scale = float(np.abs(arch.F).max())
    assert float(u.grad.abs().max()) < 1e-8 * max(1.0, scale)
