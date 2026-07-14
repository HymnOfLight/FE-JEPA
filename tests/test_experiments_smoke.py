"""Tiny end-to-end experiment + runner smokes (torch; synthetic backend).

test_e1_parallel_matches_serial is the reliability contract for unit parallelism:
with single-threaded torch, workers=2 must reproduce workers=1 bit-for-bit.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from fejepa.experiments.e1_anchor import run_e1
from fejepa.experiments.e8_regimes import run_e8
from fejepa.experiments.protocol import load_split
from fejepa.experiments.runner import _label_files, run_config
from fejepa.fe.solve import SolveLedger
from fejepa.fe.synthetic import generate_synthetic_dataset
from fejepa.models.features import FeatureSpec
from fejepa.models.fejepa import FEJEPAConfig, build_fejepa

MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 16, "mgn_depth": 1,
         "features": {"load_summary": True, "geometry": True}}


def _factory(features=None):
    spec = features or FeatureSpec(True, True)
    return build_fejepa(FEJEPAConfig(dim=16, depth=1, heads=2, features=spec))


@pytest.fixture()
def split(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "ds", n=10, seed=0)
    sp = load_split(d, n_val=3, seed=1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "lv")
    _label_files(sp.pool_files[:4], led, "lp")
    return sp


def test_e1_smoke(split):
    res = run_e1(MODEL, split.pool_files, split.val_files,
                 {"budgets": [2, 3], "seeds": 1, "epochs": 1, "grid": [1.0],
                  "decision_budget": 3, "device": "cpu"})
    entry = res["metrics"]["per_budget"][0]
    assert {"none", "balanced", "fixed", "grid_1.0"} <= set(entry["arms"])
    assert entry["grid_best"]["selection_bias"] is True
    assert res["kills"][0]["condition"].startswith("K1")
    assert res["metrics"]["retired_criterion_report"]["fixed_disp_improvement_at_decision"]


def test_e1_parallel_matches_serial(split, monkeypatch):
    monkeypatch.setenv("FEJEPA_WORKER_THREADS", "1")
    torch.set_num_threads(1)
    cfg = {"budgets": [2, 3], "seeds": 2, "epochs": 1, "grid": [],
           "decision_budget": 3, "device": "cpu"}
    r1 = run_e1(MODEL, split.pool_files, split.val_files, dict(cfg, workers=1))
    r2 = run_e1(MODEL, split.pool_files, split.val_files, dict(cfg, workers=2))
    m1, m2 = r1["metrics"]["per_budget"], r2["metrics"]["per_budget"]
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)


def test_e8_smoke(split, tmp_path):
    res = run_e8(MODEL, split.pool_files, split.val_files,
                 {"budgets": [2, 3], "pool_sizes": [3], "seeds": 1,
                  "sup_epochs": 1, "ar_epochs": 1, "device": "cpu",
                  "include_mgn": True, "state_dir": str(tmp_path / "st")})
    cells = res["metrics"]["cells"]
    assert set(cells) == {"labels", "labels_anchor", "ar_ft", "mgn", "ar",
                          "zero", "scale_aware_poly", "knn_field"}
    assert abs(cells["zero"][2]["disp_rel_l2"]["mean"] - 1.0) < 1e-12
    assert 3 in cells["ar"]
    assert len(cells["labels"][2]["per_seed_eval"]) == 1
    assert (tmp_path / "st" / "ar_p3_s0.pt").exists()
    assert len(res["kills"]) == 2


def test_wp2_smoke(instances):
    from fejepa.experiments.wp2_masking import run_wp2

    res = run_wp2(MODEL, instances,
                  {"ratios": [0.2, 0.4], "steps": 8, "n_train": 4, "n_holdout": 2,
                   "device": "cpu"})
    rows = res["metrics"]["per_ratio"]
    assert [r["ratio"] for r in rows] == [0.2, 0.4]
    assert all(r["holdout_pred_mse"] >= 0 for r in rows)
    assert res["metrics"]["recommended_ratio"] in (0.2, 0.4)
    assert res["kills"] == []                     # design dial, not a falsifier
    import pytest

    with pytest.raises(ValueError):
        run_wp2(MODEL, instances[:3],
                {"n_train": 4, "n_holdout": 2, "device": "cpu"})


def test_e3_enforces_step_floor(instances):
    from fejepa.experiments.e3_collapse import run_e3

    with pytest.raises(ValueError):
        run_e3(_factory, instances, {"steps": 100})


def test_runner_smoke(tmp_path):
    cfg = {
        "data": {"dir": str(tmp_path / "ds"), "n": 10, "seed": 0,
                 "backend": "synthetic", "labelled_policy": "economy"},
        "split": {"n_val": 3, "seed": 1},
        "model": MODEL,
        "sup": {"epochs": 1}, "pretrain": {"epochs": 1},
        "workers": 1, "tf32": True, "label_workers": 1,
        "experiments": {
            "e1": {"enabled": True, "budgets": [2, 3], "seeds": 1, "epochs": 1,
                   "grid": [], "decision_budget": 3},
            "e5": {"enabled": True, "budgets": [2, 3], "fit_budget": 3},
            "e8": {"enabled": True, "budgets": [2, 3], "pool_sizes": [3],
                   "seeds": 1, "sup_epochs": 1, "ar_epochs": 1},
        },
        "gate": {"decision_budget": 3},
        "out": str(tmp_path / "report.json"),
    }
    cp = tmp_path / "cfg.json"
    cp.write_text(json.dumps(cfg))
    payload = run_config(cp)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["provenance"]["config_sha256"]
    assert report["runtime_policy"]["tf32"] is True
    assert report["gate_g1_prime"]["conditions"].keys() == {"a", "b", "c"}
    led = report["solve_ledger"]["per_stage"]
    assert led.get("labelling-val") == 3 * 4          # n_val * n_loads
    assert led.get("labelling-pool-prefix") == 3 * 4  # max budget prefix
    assert payload["planned_steps"]["total"] > 0
