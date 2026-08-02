"""Exact-solver guarantees. test_solver_reproduces_reference IS the old assembly-level
Gate G0, preserved as a test per PLAN_MAP (plan schedules no runtime G0)."""

import numpy as np

from fejepa.fe.solve import (cg_iterations_to_tol, cg_k_steps,
                             solve_fe_displacement)


def test_solver_reproduces_reference(arch):
    U, infos = solve_fe_displacement(arch.K, arch.F, arch.free_mask,
                                     method="cg", tol=1e-12)
    rel = np.linalg.norm(U - arch.U_star) / np.linalg.norm(arch.U_star)
    assert rel < 1e-8
    assert np.allclose(U[:, arch.dirichlet_mask], 0.0)


def test_residual_identity(arch):
    for j in range(arch.n_loads):
        r = arch.K @ arch.U_star[j] - arch.F[j]
        assert np.linalg.norm(r[arch.free_mask]) < 1e-9 * max(1.0, np.linalg.norm(arch.F[j]))


def test_k_steps_monotone_gap(arch):
    from fejepa.anchor.energy import energy_gap

    j = 0
    gaps = []
    for k in (0, 3, 10):
        u = cg_k_steps(arch.K, arch.F[j], arch.free_mask, None, k)
        gaps.append(float(energy_gap(u, arch.U_star[j], arch.K, arch.F[j])[0]))
    assert gaps[0] >= gaps[1] >= gaps[2] >= -1e-12


def test_solve_warm_start_matches_and_converges(arch):
    U_cold, _ = solve_fe_displacement(arch.K, arch.F, arch.free_mask,
                                      method="cg", tol=1e-12)
    U_warm, infos = solve_fe_displacement(arch.K, arch.F, arch.free_mask,
                                          method="cg", tol=1e-12, x0=arch.U_star)
    import numpy as np

    rel = np.linalg.norm(U_warm - arch.U_star) / np.linalg.norm(arch.U_star)
    assert rel < 1e-8
    assert all(i["cg_iters"] <= 2 for i in infos)      # exact start converges at once
    assert np.linalg.norm(U_cold - U_warm) / np.linalg.norm(U_warm) < 1e-6


def test_polish_battery_and_wrapper(arch, instances):
    from fejepa.anchor.energy import energy_gap
    from fejepa.baselines import ScaleAwarePolyBaseline
    from fejepa.fe.solve import SolveLedger
    from fejepa.polish import polish_battery, polished
    import numpy as np
    import pytest

    base = ScaleAwarePolyBaseline().fit(instances[:6]).predict
    U0 = base(arch)
    assert np.allclose(polish_battery(arch, U0, k=0),
                       np.asarray(U0, dtype=np.float64) * arch.free_mask)
    gaps = [float(energy_gap(polish_battery(arch, U0, k=k),
                             arch.U_star, arch.K, arch.F).mean())
            for k in (0, 3, 10)]
    assert gaps[0] >= gaps[1] >= gaps[2] >= -1e-12     # C3: gap contracts with k
    led = SolveLedger()
    U_exact = polished(base, tol=1e-10, ledger=led)(arch)
    rel = np.linalg.norm(U_exact - arch.U_star) / np.linalg.norm(arch.U_star)
    assert rel < 1e-7
    assert led.as_dict()["per_stage"]["polish-inference"] == arch.n_loads
    with pytest.raises(ValueError):
        polish_battery(arch, U0)                       # neither k nor tol
    with pytest.raises(ValueError):
        polish_battery(arch, U0, k=3, tol=1e-8)        # both


def test_iterations_to_tol_orders_inits(arch):
    j = 0
    it_zero, conv, _ = cg_iterations_to_tol(arch.K, arch.F[j], arch.free_mask,
                                            None, tol=1e-8)
    near = arch.U_star[j] + 1e-6 * np.linalg.norm(arch.U_star[j]) \
        * np.random.default_rng(0).standard_normal(arch.ndof) * arch.free_mask
    it_near, conv2, _ = cg_iterations_to_tol(arch.K, arch.F[j], arch.free_mask,
                                             near, tol=1e-8)
    assert conv and conv2
    assert it_near < it_zero
