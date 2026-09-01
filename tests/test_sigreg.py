"""wp8-lejepa Stage 0: SIGReg behaves as a normality statistic and a usable
regulariser -- discriminates collapse and shape, agrees with its closed form,
and is minimised by gradient descent."""

import math

import pytest

torch = pytest.importorskip("torch")

from fejepa.train.sigreg import (epps_pulley_closed, epps_pulley_knots, sigreg,
                                 sigreg_monitor)


def test_knot_discretisation_matches_closed_form():
    g = torch.Generator().manual_seed(0)
    for sample in (torch.randn(400, generator=g), 3.0 * torch.rand(400, generator=g),
                   torch.randn(400, generator=g) * 0.3 + 1.0):
        ref = epps_pulley_closed(sample)
        approx = epps_pulley_knots(sample, n_knots=401, t_max=8.0)
        assert abs(float(approx) - float(ref)) <= 0.02 * abs(float(ref)) + 1e-3


def test_gaussian_low_collapsed_high_uniform_between():
    g = torch.Generator().manual_seed(1)
    gauss = torch.randn(2000, 64, generator=g)
    collapsed = torch.ones(2000, 64) * 0.7               # constant embedding
    unif = (torch.rand(2000, 64, generator=g) - 0.5) * math.sqrt(12)  # zero mean, unit var
    s_g, s_c, s_u = (sigreg_monitor(x, n_proj=256, seed=3) for x in (gauss, collapsed, unif))
    assert s_g < s_u < s_c                                # shape discrimination, not just moments


def test_gradient_descent_reduces_statistic():
    g = torch.Generator().manual_seed(2)
    x = torch.rand(1024, 16, generator=g) * 4.0 + 2.0     # non-Gaussian, off-centre
    proj = torch.nn.Linear(16, 16)
    opt = torch.optim.Adam(proj.parameters(), lr=5e-2)
    gen = torch.Generator().manual_seed(5)
    start = None
    for step in range(60):
        gen.manual_seed(5 + step)
        loss = sigreg(proj(x), n_proj=128, generator=gen)
        if start is None:
            start = float(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    gen.manual_seed(999)
    end = float(sigreg(proj(x).detach(), n_proj=128, generator=gen))
    assert end < 0.5 * start


def test_sigreg_is_differentiable_and_batched_shape_safe():
    z = torch.randn(300, 8, requires_grad=True)
    loss = sigreg(z, n_proj=32, generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    z3 = torch.randn(4, 75, 8)                             # (batch, nodes, d) flattens
    assert torch.isfinite(sigreg(z3, n_proj=16, generator=torch.Generator().manual_seed(1)))
