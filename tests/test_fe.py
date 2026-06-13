import numpy as np

from fejepa.fe.elasticity import lame_from_E_nu


def test_stiffness_symmetric_and_solves(small_problem):
    p = small_problem
    K = p.K
    assert abs(K - K.T).max() < 1e-12
    # reference solution satisfies K U = F on free dofs
    for j in range(p.n_loads):
        res = (K @ p.U_star[j] - p.F[j])[p.free_mask]
        assert np.abs(res).max() < 1e-8


def test_dirichlet_dofs_present(small_problem):
    # left-edge clamp must constrain some dofs and zero them in U*
    assert small_problem.dirichlet_mask.sum() > 0
    for j in range(small_problem.n_loads):
        assert np.allclose(small_problem.U_star[j][small_problem.dirichlet_mask], 0.0)


def test_reduced_stiffness_spd(small_problem):
    free = small_problem.free_mask
    Kff = small_problem.K[free][:, free].toarray()
    eigs = np.linalg.eigvalsh(Kff)
    assert eigs.min() > 0  # SPD after Dirichlet elimination (Lemma 1 hypothesis)


def test_plane_stress_strain_lame():
    ls = lame_from_E_nu(1.0, 0.3, "stress")
    lr = lame_from_E_nu(1.0, 0.3, "strain")
    assert ls[1] == lr[1]  # mu identical
    assert ls[0] != lr[0]  # lambda differs
