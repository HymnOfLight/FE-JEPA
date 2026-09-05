"""WP7 3D-P0.5 -- scale-equivariant decode (memo 2026-08-17 sec 6.1; review
2026-08-18 secs 3.3/6).

The feature pipeline divides the load battery by ``fscale`` (correct: it makes
the encoder's job scale-free); v2.1.5 never multiplied it back, so predictions
could not track the instance's absolute load amplitude -- an input-side
information loss with an estimated irreducible ~0.11 relative-L2 displacement
floor. The repair multiplies the decoded field by the battery ``fscale`` inside
``forward_instance`` (both FEJEPA and the MGN baseline), behind the config flag
``scale_decode`` (default True on this branch), so the anchor loss, the
supervised loss and evaluation all see the physically scaled field. Exactness
comes from linearity: u(alpha F) = alpha u(F), and ``fscale`` is known at
assembly level.

Layout mirrors test_wp7_p0: numpy tests run everywhere; model equivariance
tests are torch-gated.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fejepa.fe.synthetic import synthetic_instance
from fejepa.fe.tet3d import tet_instance
from fejepa.models.features import (FeatureSpec, battery_fscale,
                                    build_features_battery)


def _scaled_clone(arch, alpha: float):
    b = copy.deepcopy(arch)
    b.F = arch.F * alpha
    return b


# --------------------------------------------------------------------------
# numpy layer
# --------------------------------------------------------------------------

def test_battery_fscale_single_source():
    rng = np.random.default_rng(0)
    for _ in range(5):
        F = rng.standard_normal((4, 40)) * rng.uniform(1e-3, 1e3)
        assert battery_fscale(F) == float(np.abs(F).max() + 1e-12)
    assert battery_fscale(np.zeros((4, 40))) == 1e-12


def test_features_invariant_under_load_scaling_2d():
    """Level-1 of the review, cemented as a golden: the consumed features are
    bit-identical under F -> alpha*F (by design -- the scale returns in decode)."""
    a = synthetic_instance(np.random.default_rng(3), labelled=False)
    b = _scaled_clone(a, 2.37)
    Fa = build_features_battery(a, FeatureSpec())
    Fb = build_features_battery(b, FeatureSpec())
    assert np.array_equal(Fa, Fb)


def test_features_invariant_under_load_scaling_3d():
    a = tet_instance(np.random.default_rng(7), labelled=False)
    b = _scaled_clone(a, 0.41)
    spec = FeatureSpec(load_summary=True, geometry=True, spatial_dim=3)
    assert np.array_equal(build_features_battery(a, spec),
                          build_features_battery(b, spec))


def test_config_scale_decode_roundtrip():
    from fejepa.models.fejepa import FEJEPAConfig

    assert FEJEPAConfig().scale_decode is True                 # branch default
    assert FEJEPAConfig.from_dict({}).scale_decode is True     # legacy configs
    assert FEJEPAConfig.from_dict({"scale_decode": False}).scale_decode is False
    d = FEJEPAConfig(scale_decode=False).to_dict()
    assert d["scale_decode"] is False
    assert FEJEPAConfig.from_dict(d).scale_decode is False


# --------------------------------------------------------------------------
# torch layer
# --------------------------------------------------------------------------

def _fejepa(spatial_dim: int, scale_decode: bool):
    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa

    cfg = FEJEPAConfig(dim=16, depth=1, heads=2,
                       features=FeatureSpec(load_summary=True, geometry=True,
                                            spatial_dim=spatial_dim),
                       scale_decode=scale_decode)
    return build_fejepa(cfg)


def _pair(make, alpha):
    a = make()
    return a, _scaled_clone(a, alpha)


@pytest.mark.parametrize("dim3", [False, True])
def test_fejepa_scale_equivariance(dim3):
    torch = pytest.importorskip("torch")
    alpha = 2.37
    make = ((lambda: tet_instance(np.random.default_rng(7), labelled=False))
            if dim3 else
            (lambda: synthetic_instance(np.random.default_rng(3), labelled=False)))
    a, b = _pair(make, alpha)

    model = _fejepa(3 if dim3 else 2, scale_decode=True)
    model.eval()
    with torch.no_grad():
        ua = model.forward_instance(model.prepare_instance(a, "cpu"))
        ub = model.forward_instance(model.prepare_instance(b, "cpu"))
    assert torch.allclose(ub, alpha * ua, rtol=1e-5, atol=1e-6)
    assert not torch.equal(ub, ua)              # no longer scale-blind

    blind = _fejepa(3 if dim3 else 2, scale_decode=False)
    blind.eval()
    with torch.no_grad():
        va = blind.forward_instance(blind.prepare_instance(a, "cpu"))
        vb = blind.forward_instance(blind.prepare_instance(b, "cpu"))
    assert torch.equal(va, vb)                  # control: v2.1.5 blindness


def test_pack_carries_battery_fscale():
    pytest.importorskip("torch")
    a = synthetic_instance(np.random.default_rng(3), labelled=False)
    pack = _fejepa(2, True).prepare_instance(a, "cpu")
    assert float(pack["fscale"]) == pytest.approx(battery_fscale(a.F), rel=1e-6)


def test_mgn_scale_equivariance():
    torch = pytest.importorskip("torch")
    from fejepa.models.gnn import build_mesh_gnn

    alpha = 0.53
    a, b = _pair(lambda: synthetic_instance(np.random.default_rng(5),
                                            labelled=False), alpha)
    spec = FeatureSpec(load_summary=True, geometry=True)

    m = build_mesh_gnn(dim=16, depth=1, features=spec, scale_decode=True)
    m.eval()
    with torch.no_grad():
        ua = m.forward_instance(m.prepare_instance(a, "cpu"))
        ub = m.forward_instance(m.prepare_instance(b, "cpu"))
    assert torch.allclose(ub, alpha * ua, rtol=1e-5, atol=1e-6)

    blind = build_mesh_gnn(dim=16, depth=1, features=spec, scale_decode=False)
    blind.eval()
    with torch.no_grad():
        va = blind.forward_instance(blind.prepare_instance(a, "cpu"))
        vb = blind.forward_instance(blind.prepare_instance(b, "cpu"))
    assert torch.equal(va, vb)
