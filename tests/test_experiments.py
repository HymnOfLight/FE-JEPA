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


def test_e1_lambda_grid_structure(tmp_path):
    # E1 with a lambda grid should record per-lambda errors and pick a best.
    from fejepa.data.archive import load_problem, read_manifest
    from fejepa.experiments.falsification import BatteryConfig, e1_anchor_value
    from fejepa.fe.generator import GeneratorConfig, generate_dataset
    from fejepa.models.fejepa import FEJEPAConfig
    from fejepa.train.supervised import SupervisedConfig

    gcfg = GeneratorConfig(width_range=(1.6, 1.7), height_range=(0.9, 1.0),
                           mesh_size_frac=(0.22, 0.26), max_holes=0)
    d = generate_dataset(tmp_path / "ds", n_instances=3, seed=0, cfg=gcfg, verbose=False)
    files = [d / r["file"] for r in read_manifest(d)["instances"]]
    pool, val = files[:2], [load_problem(files[2])]

    cfg = BatteryConfig(
        budgets=[2], n_val=1, decision_budget=2, lambda_grid=[0.5, 1.0], n_seeds=1,
        device="cpu",
        sup=SupervisedConfig(epochs=1, model=FEJEPAConfig(dim=16, depth=1), device="cpu"),
    )
    res = e1_anchor_value(pool, val, cfg)
    row = res.metrics["per_budget"][0]
    assert set(row["lambda_grid"].keys()) == {"0.5", "1.0"}
    assert row["best_lambda"] in {"0.5", "1.0"}
    assert "improvement_at_decision_budget" in res.metrics


def test_grad_balanced_anchor_runs(small_archive):
    from fejepa.models.fejepa import FEJEPAConfig
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    cfg = SupervisedConfig(
        epochs=2, model=FEJEPAConfig(dim=16, depth=1), device="cpu",
        lambda_phys=1.0, phys_grad_balance=True, phys_balance_ratio=0.5,
    )
    out = train_supervised([small_archive], [small_archive], cfg=cfg)
    assert out["val_rel_l2_disp"] is not None
    assert out["val_rel_l2_disp"] >= 0.0
