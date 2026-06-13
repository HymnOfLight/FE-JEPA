import numpy as np

from fejepa.baselines import evaluate_naive, fit_naive_polynomial
from fejepa.experiments.falsification import ExperimentResult, gate_g1
from fejepa.metrics import effective_rank


def _res(id, killed=False, **metrics):
    return ExperimentResult(id=id, description="", metrics=metrics, kill_condition="", killed=killed)


def test_gate_g1_go_when_all_conditions_met():
    results = {
        "E5": _res("E5", beats_naive_any_budget=True),
        "E1": _res("E1", improvement_at_decision_budget=0.18),
    }
    gate = gate_g1(results, pretrain_wins_at_le_256=True)
    assert gate["passed"] is True
    assert gate["cond_a_sanity"] and gate["cond_b_component_value"] and gate["cond_c_pretrain_wins"]


def test_gate_g1_nogo_when_sanity_fails():
    results = {
        "E5": _res("E5", beats_naive_any_budget=False),
        "E1": _res("E1", improvement_at_decision_budget=0.5),
    }
    gate = gate_g1(results, pretrain_wins_at_le_256=True)
    assert gate["passed"] is False
    assert gate["cond_a_sanity"] is False


def test_gate_g1_nogo_when_component_value_too_small():
    results = {
        "E5": _res("E5", beats_naive_any_budget=True),
        "E1": _res("E1", improvement_at_decision_budget=0.02),
    }
    gate = gate_g1(results, pretrain_wins_at_le_256=True)
    assert gate["passed"] is False
    assert gate["cond_b_component_value"] is False


def test_effective_rank_isotropic_vs_collapsed():
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((400, 16))
    collapsed = rng.standard_normal((400, 16)) @ np.diag([1.0] + [1e-4] * 15)
    assert effective_rank(iso) > 8.0
    assert effective_rank(collapsed) < 2.0


def test_naive_baseline_runs(small_archive):
    surrogate = fit_naive_polynomial([small_archive])
    pred = surrogate.predict(small_archive, 0)
    assert pred.shape == (small_archive.n_dof,)
    out = evaluate_naive(surrogate, [small_archive])
    assert out["val_rel_l2_disp"] >= 0.0
