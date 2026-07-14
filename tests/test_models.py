"""Model stack shapes and contracts (torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.models.features import FeatureSpec
from fejepa.models.fejepa import (FEJEPAConfig, build_fejepa, load_pretrained_into,
                                  mesh_adjacency)
from fejepa.models.gnn import build_mesh_gnn


def _cfg():
    return FEJEPAConfig(dim=16, depth=1, heads=2,
                        features=FeatureSpec(load_summary=True, geometry=True))


def test_forward_instance_shapes(arch):
    m = build_fejepa(_cfg())
    pack = m.prepare_instance(arch, "cpu")
    u = m.forward_instance(pack)
    assert u.shape == (arch.n_loads, arch.ndof)
    assert torch.allclose(u[:, torch.as_tensor(arch.dirichlet_mask)],
                          torch.zeros(1), atol=0)


def test_masked_prediction_scalar_and_grad(arch):
    m = build_fejepa(_cfg())
    pack = m.prepare_instance(arch, "cpu")
    adj = mesh_adjacency(arch.elements, arch.n_nodes)
    loss = m.masked_prediction(pack["feats"][0], adj, np.random.default_rng(0))
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.encoder.parameters())


def test_masked_prediction_reuse_is_identical(arch):
    m = build_fejepa(_cfg())
    pack = m.prepare_instance(arch, "cpu")
    adj = mesh_adjacency(arch.elements, arch.n_nodes)
    l1 = m.masked_prediction(pack["feats"][0], adj, np.random.default_rng(7))
    z0 = m.encode(pack["feats"])[0]
    l2 = m.masked_prediction(pack["feats"][0], adj, np.random.default_rng(7),
                             z_full=z0)
    assert torch.equal(l1, l2)


def test_load_pretrained_into_counts():
    a, b = build_fejepa(_cfg()), build_fejepa(_cfg())
    n = load_pretrained_into(b, a.state_dict())
    assert n == len(a.state_dict())
    sd = {k: v for k, v in a.state_dict().items() if "decoder" not in k}
    assert 0 < load_pretrained_into(build_fejepa(_cfg()), sd) < n


def test_seeded_factory_controls_initialization():
    from fejepa.experiments.protocol import seeded_factory

    a = seeded_factory(lambda: build_fejepa(_cfg()), 0)
    b = seeded_factory(lambda: build_fejepa(_cfg()), 0)
    c = seeded_factory(lambda: build_fejepa(_cfg()), 1)
    ka = next(iter(a.state_dict()))
    assert torch.equal(a.state_dict()[ka], b.state_dict()[ka])
    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k])
               for k in a.state_dict())


def test_mgn_shared_interface(arch):
    m = build_mesh_gnn(dim=16, depth=2, features=FeatureSpec(True, True))
    pack = m.prepare_instance(arch, "cpu")
    u = m.forward_instance(pack)
    assert u.shape == (arch.n_loads, arch.ndof) and torch.isfinite(u).all()
