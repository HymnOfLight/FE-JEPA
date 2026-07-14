"""WP7 foundation: the node-major contract, one dimension up -- archive, exact
solver, anchor identities, warm start, polish, and the WP6 spectral checks all
run on 3D instances UNCHANGED."""

import numpy as np
import pytest

from fejepa.anchor.energy import energy_gap, pi_h
from fejepa.data.archive import load_instance, save_instance
from fejepa.fe.solve import solve_fe_displacement
from fejepa.fe.tet3d import (tet_instance, tet_strain_energy, tet_von_mises)
from fejepa.polish import polish_battery, polished
from fejepa.theory import check_chebyshev_polish, check_conditioning_lemma


@pytest.fixture(scope="module")
def arch3d():
    return tet_instance(np.random.default_rng(7), nx=3, ny=3, nz=2, labelled=True)


def test_contract_residual_and_symmetry(arch3d):
    asym = abs(arch3d.K - arch3d.K.T)
    assert asym.max() < 1e-12 * abs(arch3d.K).max()
    for j in range(arch3d.n_loads):
        r = arch3d.K @ arch3d.U_star[j] - arch3d.F[j]
        assert np.linalg.norm(r[arch3d.free_mask]) \
            < 1e-9 * max(1.0, np.linalg.norm(arch3d.F[j]))
    assert arch3d.ndof == 3 * arch3d.n_nodes
    assert arch3d.meta["extra"]["dim"] == 3


def test_energy_identity_one_dimension_up(arch3d):
    for j in range(arch3d.n_loads):
        u = arch3d.U_star[j]
        e_rec = tet_strain_energy(arch3d.nodes, arch3d.elements, u,
                                  arch3d.meta["material"])
        e_quad = 0.5 * float(u @ (arch3d.K @ u))
        assert abs(e_rec - e_quad) < 1e-10 * max(1.0, abs(e_quad))


def test_anchor_gap_identity_is_dimension_agnostic(arch3d, rng):
    v = arch3d.U_star + 0.1 * rng.standard_normal(arch3d.U_star.shape) \
        * arch3d.free_mask
    gap = energy_gap(v, arch3d.U_star, arch3d.K, arch3d.F)
    d = v - arch3d.U_star
    assert np.allclose(gap, 0.5 * np.einsum("ld,ld->l", d, (arch3d.K @ d.T).T),
                       rtol=1e-10, atol=1e-12)
    assert (pi_h(arch3d.U_star, arch3d.K, arch3d.F) < 0).all()


def test_archive_roundtrip_3d(tmp_path, arch3d):
    p = tmp_path / "a3d.npz"
    save_instance(arch3d, p)
    back = load_instance(p)
    assert back.nodes.shape[1] == 3 and back.elements.shape[1] == 4
    assert np.allclose(back.U_star, arch3d.U_star)
    assert (back.K != arch3d.K).nnz == 0


def test_solver_and_polish_unchanged_in_3d(arch3d):
    U, _ = solve_fe_displacement(arch3d.K, arch3d.F, arch3d.free_mask,
                                 method="cg", tol=1e-12)
    assert np.linalg.norm(U - arch3d.U_star) / np.linalg.norm(arch3d.U_star) < 1e-8
    naive = lambda a: np.zeros_like(a.F)              # noqa: E731
    U_ex = polished(naive, tol=1e-10)(arch3d)
    assert np.linalg.norm(U_ex - arch3d.U_star) / np.linalg.norm(arch3d.U_star) < 1e-7
    g0 = energy_gap(polish_battery(arch3d, naive(arch3d), k=0),
                    arch3d.U_star, arch3d.K, arch3d.F).mean()
    g5 = energy_gap(polish_battery(arch3d, naive(arch3d), k=5),
                    arch3d.U_star, arch3d.K, arch3d.F).mean()
    assert g5 < g0


def test_von_mises_3d_positive(arch3d):
    vm = tet_von_mises(arch3d.nodes, arch3d.elements, arch3d.U_star[0],
                       arch3d.meta["material"])
    assert vm.shape == (arch3d.elements.shape[0],) and vm.max() > 0


def test_wp6_theory_checks_hold_in_3d(arch3d):
    assert check_conditioning_lemma(arch3d, n_samples=4)["holds"]
    assert check_chebyshev_polish(arch3d, ks=(1, 4), n_inits=2)["holds"]
