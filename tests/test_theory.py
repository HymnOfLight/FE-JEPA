"""WP6 falsification pass -- every check executes here (numpy/scipy)."""

import numpy as np
import pytest

from fejepa.theory import (check_chebyshev_polish, check_conditioning_lemma,
                           check_mode_contraction, lambda_extremes,
                           prop1_cross_geometry_counterexample,
                           prop1_within_geometry_premise, run_theory_checks)


def test_lambda_extremes_spd(arch):
    spec = lambda_extremes(arch.K, arch.free_mask)
    assert 0 < spec["lambda_min"] < spec["lambda_max"]
    assert spec["kappa"] == pytest.approx(spec["lambda_max"] / spec["lambda_min"])


def test_conditioning_lemma_holds_and_is_tight_somewhere(arch):
    out = check_conditioning_lemma(arch, n_samples=12,
                                   rng=np.random.default_rng(3))
    assert out["holds"] and out["max_ratio"] <= 1.0 + 1e-8
    assert out["median_tightness"] > 0.005       # non-vacuous (~sqrt(l_min/l_mean))


def test_mode_contraction_exact(arch):
    out = check_mode_contraction(arch)
    assert out["holds"] and out["max_err"] < 1e-9


def test_chebyshev_bound_holds(arch):
    out = check_chebyshev_polish(arch, ks=(1, 3, 5), n_inits=3,
                                 rng=np.random.default_rng(4))
    assert out["holds"]
    assert out["worst_measured_over_bound"] <= 1.0 + 1e-8


def test_prop1_premise_positive(instances):
    out = prop1_within_geometry_premise(instances[:4])
    assert out["min_separation"] > 0.0


def test_prop1_naive_cross_counterexample_found(instances):
    out = prop1_cross_geometry_counterexample(instances, load_idx=0)
    assert out["naive_extension_falsified"] is True
    w = out["witness"]
    assert w["d_cross"] < w["a_within_min"]


def test_run_theory_checks_aggregate_and_render(instances):
    res = run_theory_checks(instances, {"n_check": 4, "seed": 0})
    assert res["id"] == "WP6-theory"
    assert not res["kills"][0]["triggered"]
    m = res["metrics"]
    assert m["conditioning"]["holds"] and m["chebyshev_polish"]["holds"]

    from fejepa.results import render_results

    md = render_results({"results": {"wp6": res}, "provenance": {},
                         "config": {}, "runtime_policy": {}})
    assert "WP6" in md and "falsification" in md
    assert "conditioning max ratio" in md
