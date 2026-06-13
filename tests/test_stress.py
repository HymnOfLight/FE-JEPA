import numpy as np

from fejepa.fe.stress import element_strain_stress, strain_energy
from fejepa.metrics import von_mises_metrics


def test_strain_energy_matches_uKu(small_problem):
    p = small_problem
    for j in range(p.n_loads):
        U = p.U_star[j]
        se = strain_energy(p.nodes, p.elements, U, p.material)
        uku = 0.5 * float(U @ (p.K @ U))
        assert abs(se - uku) <= 1e-6 * max(1.0, abs(uku))


def test_von_mises_nonnegative_and_shapes(small_problem):
    p = small_problem
    _, stress, vm = element_strain_stress(p.nodes, p.elements, p.U_star[0], p.material)
    assert stress.shape == (p.elements.shape[0], 3)
    assert vm.shape == (p.elements.shape[0],)
    assert np.all(vm >= 0.0)


def test_von_mises_metrics_zero_for_exact(small_problem):
    p = small_problem
    m = von_mises_metrics(p.nodes, p.elements, p.U_star[0], p.U_star[0], p.material)
    assert m["rel_l2_vm"] < 1e-9
    assert m["max_vm_rel_err"] < 1e-9
    assert m["crit_recall"] == 1.0


def test_von_mises_metrics_degrade_for_wrong(small_problem):
    p = small_problem
    bad = p.U_star[0] * 0.5  # under-predicted field
    m = von_mises_metrics(p.nodes, p.elements, bad, p.U_star[0], p.material)
    assert m["rel_l2_vm"] > 0.1  # halving displacement halves stress -> large rel error
