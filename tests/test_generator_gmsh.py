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
    # Symmetry at machine tolerance: bitwise K == K.T held on the July box
    # (earlier scikit-fem), but newer skfem assembly orders leave one-ulp
    # residues (observed 1.5e-16 relative with skfem 12.0.2). Pattern must
    # still match exactly; values to machine precision (wp7-3d portability).
    assert ((arch.K != 0).astype(int) - (arch.K.T != 0).astype(int)).nnz == 0
    asym = abs(arch.K - arch.K.T).max()
    assert asym <= 1e-12 * abs(arch.K).max(), asym
