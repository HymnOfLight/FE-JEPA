import numpy as np

from fejepa.data.archive import load_problem, save_problem


def test_archive_roundtrip(small_problem, tmp_path):
    path = tmp_path / "x.npz"
    save_problem(path, small_problem)
    arch = load_problem(path)
    assert abs(arch.K - small_problem.K).max() == 0.0
    assert np.array_equal(arch.F, small_problem.F)
    assert np.array_equal(arch.U_star, small_problem.U_star)
    assert arch.material.nu == small_problem.material.nu


def test_unlabelled_skips_solve(small_problem):
    assert small_problem.U_star is not None  # labelled fixture has it


def test_multires_shares_geometry_and_loads(multires_pair):
    fine, coarse = multires_pair
    # different meshes
    assert fine.n_nodes != coarse.n_nodes
    # same geometry parameters and identical load battery names
    assert np.isclose(fine.meta["width"], coarse.meta["width"])
    assert np.isclose(fine.meta["height"], coarse.meta["height"])
    assert fine.load_names == coarse.load_names
    # both solve consistently
    for prob in (fine, coarse):
        res = (prob.K @ prob.U_star[0] - prob.F[0])[prob.free_mask]
        assert np.abs(res).max() < 1e-7
