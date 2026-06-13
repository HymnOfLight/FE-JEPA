import json

import torch

from fejepa.anchor.energy import EnergyAnchor, csr_to_torch_sparse
from fejepa.experiments.runner import run_config


def test_csr_to_torch_sparse_matvec(small_problem):
    K = small_problem.K
    Ksp = csr_to_torch_sparse(K, dtype=torch.float64)
    u = torch.randn(small_problem.n_dof, dtype=torch.float64)
    Ku = torch.sparse.mm(Ksp, u.unsqueeze(1)).squeeze(1).numpy()
    ref = K @ u.numpy()
    assert abs(Ku - ref).max() < 1e-9


def test_energy_anchor_cpu_device(small_problem):
    anchor = EnergyAnchor(
        small_problem.K, small_problem.F, small_problem.free_mask,
        dtype=torch.float32, device="cpu",
    )
    u = torch.zeros(small_problem.n_loads, small_problem.n_dof, requires_grad=True)
    anchor(u, reduction="sum").backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_run_config_pipeline(tmp_path):
    cfg = {
        "device": "cpu",
        "model": {"dim": 32, "depth": 2, "heads": 4},
        "dataset": {"out": str(tmp_path / "ds"), "n": 6, "seed": 0, "max_holes": 0,
                     "labelled": True},
        "regimes": {"enabled": True, "n_train": 3, "n_val": 2, "epochs": 2, "lr": 0.002,
                     "out": str(tmp_path / "regimes.json")},
        "battery": {"enabled": True, "experiments": ["E5"], "budgets": [2, 3],
                     "n_val": 2, "epochs": 2, "decision_budget": 3,
                     "out": str(tmp_path / "battery.json")},
        "label_efficiency": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    run_config(cfg_path)

    regimes = json.loads((tmp_path / "regimes.json").read_text())
    assert {"labels", "labels+anchor", "anchor_only"} <= set(regimes["regimes"])
    battery = json.loads((tmp_path / "battery.json").read_text())
    assert "gate_g1" in battery and "E5" in battery["results"]
