"""Real-backend generation (gmsh + skfem); auto-skips without the gen extra."""

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("skfem")

from fejepa.fe.generator import build_instance, sample_params
from fejepa.fe.solve import solve_fe_displacement


def test_build_instance_and_residual():
    params = sample_params(np.random.default_rng(0))
    arch = build_instance(params)
    assert arch.n_nodes > 50 and arch.F.shape[0] == 4
    U, _ = solve_fe_displacement(arch.K, arch.F, arch.free_mask, method="direct")
    r = arch.K @ U[0] - arch.F[0]
    assert np.linalg.norm(r[arch.free_mask]) < 1e-8 * max(1.0, np.linalg.norm(arch.F[0]))
    assert (arch.K != arch.K.T).nnz == 0
