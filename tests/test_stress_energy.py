"""The audited energy identities (plan Sec.2.1), preserved as executable checks."""

import numpy as np

from fejepa.anchor.energy import energy_gap, pi_h, pi_star_abs
from fejepa.fe.stress import element_von_mises, strain_energy


def test_recovered_energy_equals_quadratic_form(arch):
    for j in range(arch.n_loads):
        u = arch.U_star[j]
        e_rec = strain_energy(arch.nodes, arch.elements, u, arch.meta["material"])
        e_quad = 0.5 * float(u @ (arch.K @ u))
        assert abs(e_rec - e_quad) < 1e-10 * max(1.0, abs(e_quad))


def test_gap_identity(arch, rng):
    v = arch.U_star + 0.1 * rng.standard_normal(arch.U_star.shape) * arch.free_mask
    gap = energy_gap(v, arch.U_star, arch.K, arch.F)
    d = (v - arch.U_star)
    half_k = 0.5 * np.einsum("ld,ld->l", d, (arch.K @ d.T).T)
    assert np.allclose(gap, half_k, rtol=1e-10, atol=1e-12)
    assert (gap >= -1e-12).all()


def test_pi_star_negative_and_normalizer(arch):
    assert (pi_h(arch.U_star, arch.K, arch.F) < 0).all()
    assert (pi_star_abs(arch) > 0).all()


def test_von_mises_shape_positive(arch):
    vm = element_von_mises(arch.nodes, arch.elements, arch.U_star[0],
                           arch.meta["material"])
    assert vm.shape == (arch.elements.shape[0],)
    assert (vm >= 0).all() and vm.max() > 0
