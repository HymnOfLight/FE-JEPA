"""Repaired sanity baselines (plan B2/E5') -- pure numpy/scipy."""

import numpy as np

from fejepa.baselines import (GlobalPolyBaseline, KNNFieldBaseline,
                              ScaleAwarePolyBaseline, zero_predictor)
from fejepa.experiments.e5_sanity import run_e5
from fejepa.metrics import evaluate_model


def test_zero_is_the_floor(instances):
    out = evaluate_model(zero_predictor, instances)
    assert abs(out["disp_rel_l2"] - 1.0) < 1e-12


def test_poly_and_scale_aware_run(instances):
    fit, val = instances[:6], instances[6:]
    for B in (GlobalPolyBaseline, ScaleAwarePolyBaseline):
        err = evaluate_model(B().fit(fit).predict, val)["disp_rel_l2"]
        assert np.isfinite(err) and err > 0


def test_knn_self_retrieval_is_accurate(instances):
    knn = KNNFieldBaseline().fit(instances[:6])
    err = evaluate_model(knn.predict, instances[:1])["disp_rel_l2"]
    assert err < 0.05                       # query in the training set -> near-exact


def test_knn_triangulation_cache_reused(instances):
    knn = KNNFieldBaseline().fit(instances[:3])
    u1 = knn.predict(instances[0])
    assert len(knn._tri) == 1               # one neighbour -> one cached Delaunay
    tri = next(iter(knn._tri.values()))
    u2 = knn.predict(instances[0])
    assert next(iter(knn._tri.values())) is tri     # reused, not rebuilt
    import numpy as np

    assert np.array_equal(u1, u2)


def test_e8_naive_baseline_cells(instances):
    from fejepa.experiments.e8_regimes import naive_baseline_cells

    cells = naive_baseline_cells(instances[:6], instances[6:], budgets=[3, 6])
    assert set(cells) == {"zero", "scale_aware_poly", "knn_field"}
    zero3 = cells["zero"][3]
    assert abs(zero3["disp_rel_l2"]["mean"] - 1.0) < 1e-12
    assert zero3["disp_rel_l2"]["per_seed"] == [zero3["disp_rel_l2"]["mean"]]
    assert len(zero3["per_seed_eval"]) == 1            # deterministic, seedless
    assert cells["knn_field"][6]["disp_rel_l2"]["std"] == 0.0
    # cell shape is uniform with trained regimes (same _agg contract)
    from fejepa.metrics import FIELD_KEYS

    assert set(FIELD_KEYS) <= set(cells["scale_aware_poly"][3])


def test_e5_pass_and_fail_paths(instances):
    fit, val = instances[:6], instances[6:]     # disjoint: k-NN cannot self-retrieve
    n = len(val)
    good = {4: {"mean": 0.01, "per_instance": [0.01] * n},
            8: {"mean": 0.01, "per_instance": [0.01] * n}}
    res = run_e5(fit, val, {"budgets": [4, 8], "fit_budget": 6}, anchored=good)
    assert res["metrics"]["passed_all"] and not res["kills"][0]["triggered"]
    bad = {4: {"mean": 0.9, "per_instance": [0.9] * n},     # fails the 3x-zero margin
           8: {"mean": 0.9, "per_instance": [0.9] * n}}
    res = run_e5(fit, val, {"budgets": [4, 8], "fit_budget": 6}, anchored=bad)
    assert not res["metrics"]["passed_all"] and res["kills"][0]["triggered"]
