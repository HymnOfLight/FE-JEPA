"""Frozen metric hierarchy (plan Sec.4), rank instrument (B4), features, region mask."""

import numpy as np

from fejepa.metrics import (FIELD_KEYS, effective_rank, evaluate_fields,
                            evaluate_model, label_efficiency_auc)
from fejepa.models.features import (FeatureSpec, build_features,
                                    geometry_descriptor, load_summary)
from fejepa.models.fejepa import mesh_adjacency, region_target_mask


def test_exact_prediction_scores_zero_error(arch):
    v = evaluate_fields(arch.U_star, arch)
    assert v["disp_rel_l2"] < 1e-12
    assert abs(v["energy_gap_rel"]) < 1e-10
    assert v["vm_rel_l2"] < 1e-10 and v["crit_recall"] == 1.0


def test_zero_prediction_scores_one(arch):
    v = evaluate_fields(np.zeros_like(arch.F), arch)
    assert abs(v["disp_rel_l2"] - 1.0) < 1e-12
    assert abs(v["energy_gap_rel"] - 1.0) < 1e-10   # gap(0) = |Pi*|


def test_evaluate_model_arrays(instances):
    out = evaluate_model(lambda a: np.zeros_like(a.F), instances)
    assert out["n_val"] == len(instances)
    for k in FIELD_KEYS:
        assert len(out["per_instance"][k]) == len(instances)


def test_effective_rank(rng):
    z = rng.standard_normal((500, 6))
    z[:, 0] *= 100.0                       # one dominant direction
    raw = effective_rank(z, standardized=False)
    std = effective_rank(z, standardized=True)
    assert raw < 1.5 and std > 5.0         # the B4 repair in one assertion


def test_auc_orders_curves():
    b = [16, 64, 256]
    assert label_efficiency_auc(b, [0.3, 0.2, 0.1]) > \
        label_efficiency_auc(b, [0.2, 0.1, 0.05])


def test_features(arch):
    spec = FeatureSpec(load_summary=True, geometry=True)
    x = build_features(arch, 0, spec)
    assert x.shape == (arch.n_nodes, spec.dim) and np.isfinite(x).all()
    g = geometry_descriptor(arch.meta)
    assert g.shape == (6,) and np.isfinite(g).all()
    s = load_summary(arch.F, 0)
    assert s.shape == (4,) and 0.0 <= s[3] <= 1.0
    assert FeatureSpec(False, False).dim == 6


def test_region_mask(arch, rng):
    adj = mesh_adjacency(arch.elements, arch.n_nodes)
    m = region_target_mask(adj, 0.4, rng)
    frac = m.mean()
    assert 0.3 <= frac <= 0.5 and 0 < m.sum() < arch.n_nodes
