"""P3 resolution-transfer smokes + the r8 Sec.5 divergence-rule units.

The end-to-end smoke drives the REAL sharing path: a tiny E8 persists AR /
labels / mgn states, then run_p3 consumes them zero-shot on a second synthetic
corpus, fine-tunes few-shot against scratch, and builds the strongest-form
naive rows -- asserting the exact contract block gate_g2 consumes.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.experiments.e8_regimes import run_e8
from fejepa.experiments.gate_g2 import gate_g2
from fejepa.experiments.p3_transfer import run_p3
from fejepa.experiments.protocol import divergence_flags, load_split
from fejepa.experiments.runner import _label_files
from fejepa.fe.solve import SolveLedger
from fejepa.fe.synthetic import generate_synthetic_dataset

MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 16, "mgn_depth": 1,
         "features": {"load_summary": True, "geometry": True}}


def test_divergence_flags_rule():
    evals = [{"disp_rel_l2": 0.2}, {"disp_rel_l2": float("inf")},
             {"disp_rel_l2": 11.0}, {"disp_rel_l2": float("nan")}]
    assert divergence_flags(evals) == [False, True, True, True]
    # means always include flagged runs -- the rule constrains reporting only
    assert divergence_flags([0.5, 9.9, 10.1]) == [False, False, True]


def test_gate_g2_reads_e6_mean_key():
    r = gate_g2(None, None, None, {"metrics": {"rho_within_mean": 0.25}}, None)
    assert r["kills"]["KP6"] is True


@pytest.fixture()
def shared_states(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "inband", n=12, seed=0)
    sp = load_split(d, n_val=3, seed=1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "lv")
    _label_files(sp.pool_files[:4], led, "lp")
    state_dir = tmp_path / "states"
    e8 = run_e8(MODEL, sp.pool_files, sp.val_files,
                {"budgets": [2, 4], "pool_sizes": [4], "seeds": 1,
                 "ar_epochs": 1, "sup_epochs": 1, "include_mgn": True,
                 "include_ar_ft": False,   # r8 D-1: no ar_ft arm in Phase 2
                 "state_dir": str(state_dir), "device": "cpu", "workers": 1})
    return sp, state_dir, e8, led


def test_p3_end_to_end_shared_checkpoints(tmp_path, shared_states):
    sp, state_dir, e8, led = shared_states
    # the sharing contract: e8 persisted labels/mgn states at b_max
    assert (state_dir / "ar_p4_s0.pt").exists()
    assert (state_dir / "labels_b4_s0.pt").exists()
    assert (state_dir / "mgn_b4_s0.pt").exists()
    # r8 D-1: the ar_ft arm is excluded from the Phase-2 battery
    assert "ar_ft" not in e8["metrics"]["cells"]

    fd = generate_synthetic_dataset(tmp_path / "fine", n=6, seed=7)
    fsp = load_split(fd, n_val=0, seed=1)
    ffiles = list(fsp.pool_files)
    fine_eval, fine_prefix = ffiles[:3], ffiles[3:5]
    _label_files(fine_eval, led, "lfv")
    _label_files(fine_prefix, led, "lfp")
    from fejepa.data.archive import load_instance
    fine_eval_archs = [load_instance(f) for f in fine_eval]
    val_archs = [load_instance(f) for f in sp.val_files]

    p3 = run_p3(MODEL, sp.pool_files, val_archs, fine_eval_archs,
                fine_eval, fine_prefix,
                {"seeds": 1, "device": "cpu", "workers": 1,
                 "state_dir": str(state_dir), "pool_size": 4, "bmax": 4,
                 "naive_budget": 4, "fewshot_budgets": [2],
                 "fewshot_epochs": 1, "fewshot_lr": 1.5e-3})

    m = p3["metrics"]
    assert np.isfinite(m["ar"]["fine_disp_mean"])
    assert np.isfinite(m["ar"]["inband_disp_mean"])
    assert np.isfinite(m["ratio_fine_over_inband"])
    assert set(m["naive_at_fine"]) == {"knn_field", "scale_aware_poly"}
    assert all(np.isfinite(v) for v in m["naive_at_fine"].values())
    assert m["zero_shot_reported"]["labels@max"] is not None
    assert m["zero_shot_reported"]["mgn@max"] is not None
    fs = m["fewshot"][2]
    assert set(fs) == {"finetune", "scratch"}
    assert isinstance(fs["finetune"]["divergence_flags"], list)

    # the block feeds gate_g2 directly (contract closure)
    g = gate_g2(e8, None, p3, None, None,
                gate_cfg={"transfer_win": 1e9}, kill_cfg={})
    assert g["transfer_zone"] in ("win", "weakened", "retired")
    assert isinstance(g["conditions"]["c"], bool)


def test_e8_cells_carry_divergence_flags(shared_states):
    _, _, e8, _ = shared_states
    cell = e8["metrics"]["cells"]["labels"][4]
    assert cell["divergence_flags"] == [False]


def test_lambda_policy_helper():
    """r8 Sec.6: fixed lambda=1 strictly below 64; balanced at and above."""
    from fejepa.experiments.e8_regimes import POLICY_BALANCED_FROM, _anchor_kw

    assert POLICY_BALANCED_FROM == 64
    assert _anchor_kw(16) == {"anchor_mode": "fixed", "lambda_phys": 1.0}
    assert _anchor_kw(63) == {"anchor_mode": "fixed", "lambda_phys": 1.0}
    assert _anchor_kw(64) == {"anchor_mode": "balanced", "balance_ratio": 1.0}
    assert _anchor_kw(1024) == {"anchor_mode": "balanced", "balance_ratio": 1.0}


def test_e8_mgn_budget_subset(tmp_path):
    """r10 L4: the comparator trains only on its budget subset."""
    d = generate_synthetic_dataset(tmp_path / "sub", n=10, seed=3)
    sp = load_split(d, n_val=2, seed=1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "v")
    _label_files(sp.pool_files[:4], led, "p")
    e8 = run_e8(MODEL, sp.pool_files, sp.val_files,
                {"budgets": [2, 4], "pool_sizes": [4], "seeds": 1,
                 "ar_epochs": 1, "sup_epochs": 1, "include_mgn": True,
                 "include_ar_ft": False, "mgn_budgets": [4],
                 "state_dir": str(tmp_path / "st"), "device": "cpu",
                 "workers": 1})
    mg = e8["metrics"]["cells"]["mgn"]
    assert set(int(k) for k in mg) == {4}
    assert set(int(k) for k in e8["metrics"]["cells"]["labels"]) == {2, 4}


def test_bf16_precision_smoke(tmp_path):
    """r10 L5 wiring: both trainers run under bf16 autocast on CPU; the
    anchor self-protects to fp32; metrics finite."""
    import torch

    from fejepa.data.archive import load_instance
    from fejepa.train.losses import AR_CONFIG
    from fejepa.train.pretrain import PretrainConfig, pretrain
    from fejepa.train.supervised import SupervisedConfig, train_supervised
    from fejepa.metrics import evaluate_model, torch_predictor
    from fejepa.experiments.parallel import _build_model

    d = generate_synthetic_dataset(tmp_path / "bf", n=6, seed=5)
    sp = load_split(d, n_val=2, seed=1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "v")
    _label_files(sp.pool_files[:2], led, "t")
    val = [load_instance(f) for f in sp.val_files]
    tr = [load_instance(f) for f in sp.pool_files[:2]]

    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    h = pretrain(m, tr, PretrainConfig(epochs=1, lr=1e-3, seed=0,
                                       device="cpu", loss=AR_CONFIG,
                                       log_every=-1, precision="bf16"))
    assert np.isfinite(h["loss"][-1])
    ev = evaluate_model(torch_predictor(m, "cpu"), val)
    assert np.isfinite(ev["disp_rel_l2"])

    m2 = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    train_supervised(m2, tr, val,
                     SupervisedConfig(epochs=1, lr=1.5e-3, seed=0,
                                      device="cpu", anchor_mode="balanced",
                                      balance_ratio=1.0, log_every=-1,
                                      precision="bf16"))
    ev2 = evaluate_model(torch_predictor(m2, "cpu"), val)
    assert np.isfinite(ev2["energy_gap_rel"])
