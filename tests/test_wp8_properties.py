"""Mathematical properties of the branch's core pieces:
SIGReg is rotation-invariant in expectation (an isotropic sketch); the
bottleneck's output does not depend on the mesh's node numbering."""

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from fejepa.data.archive import load_instance
from fejepa.experiments.parallel import _build_model
from fejepa.train.sigreg import sigreg


def test_sigreg_is_rotation_invariant_in_expectation():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2000, 16, generator=g) * torch.linspace(0.3, 2.0, 16) + 0.5   # anisotropic, shifted
    Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g))                  # random orthogonal
    a = float(sigreg(z, n_proj=4096, generator=torch.Generator().manual_seed(1)))
    b = float(sigreg(z @ Q, n_proj=4096, generator=torch.Generator().manual_seed(2)))
    assert abs(a - b) <= 0.03 * max(abs(a), abs(b))            # same statistic, different directions


def _permute(arch, perm):
    """Node-permuted copy: nodes, elements, K (dof-major 2 per node), F, mask, U*."""
    sd = arch.nodes.shape[1]
    dof_perm = np.concatenate([[sd * p + k for k in range(sd)] for p in perm])
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    P = sp.csr_matrix((np.ones(len(dof_perm)), (np.arange(len(dof_perm)), dof_perm)),
                      shape=(len(dof_perm), len(dof_perm)))
    return dataclasses.replace(arch, nodes=arch.nodes[perm], elements=inv[arch.elements],
                               K=(P @ arch.K @ P.T).tocsr(), F=arch.F[:, dof_perm],
                               dirichlet_mask=arch.dirichlet_mask[dof_perm],
                               U_star=None if arch.U_star is None else arch.U_star[:, dof_perm])


def test_bottleneck_output_is_independent_of_node_numbering(tiny_corpus):
    sp_ = tiny_corpus(seed=53)
    arch = load_instance(sp_.pool_files[0])
    rng = np.random.default_rng(0)
    perm = rng.permutation(arch.nodes.shape[0])
    parch = _permute(arch, perm)
    m = _build_model({"kind": "bottleneck", "model": {"dim": 16, "depth": 1, "heads": 2, "n_tokens": 6,
                      "features": {"load_summary": True, "geometry": True}}, "seed": 0})
    m.eval()
    sd = arch.nodes.shape[1]
    with torch.no_grad():
        u = m.forward_instance(m.prepare_instance(arch, "cpu")).reshape(-1, arch.nodes.shape[0], sd)
        up = m.forward_instance(m.prepare_instance(parch, "cpu")).reshape(-1, arch.nodes.shape[0], sd)
    assert torch.allclose(up, u[:, perm, :], atol=1e-5, rtol=1e-4)
