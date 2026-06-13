import numpy as np
import torch

from fejepa.anchor.energy import EnergyAnchor, energy_gap, energy_norm_sq


def test_gradient_is_Ku_minus_F(small_problem):
    p = small_problem
    anchor = EnergyAnchor(p.K, p.F, p.free_mask, dtype=torch.float64)
    u = torch.randn(p.n_loads, p.n_dof, dtype=torch.float64, requires_grad=True)
    anchor(u, reduction="sum").backward()
    ufree = u.detach().numpy() * p.free_mask
    analytic = ((p.K @ ufree.T).T - p.F) * p.free_mask
    assert np.abs(u.grad.numpy() - analytic).max() < 1e-9


def test_lemma1_energy_gap_equals_half_energy_norm(small_problem):
    p = small_problem
    u = (np.random.default_rng(0).standard_normal(p.U_star.shape)) * p.free_mask
    gap = energy_gap(u, p.U_star, p.K, p.F)
    half = 0.5 * energy_norm_sq(u - p.U_star, p.K)
    assert np.allclose(gap, half, atol=1e-9)


def test_minimizing_energy_recovers_fe_solution(small_problem):
    p = small_problem
    anchor = EnergyAnchor(p.K, p.F, p.free_mask, dtype=torch.float64)
    u = torch.zeros(p.n_loads, p.n_dof, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([u], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = anchor(u, reduction="sum")
        loss.backward()
        return loss

    opt.step(closure)
    ufin = u.detach().numpy() * p.free_mask
    rel = np.linalg.norm(ufin - p.U_star) / np.linalg.norm(p.U_star)
    assert rel < 1e-3
