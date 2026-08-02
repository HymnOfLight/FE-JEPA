"""Split determinism (audit V4) and Gate G1' logic on crafted results (plan Sec.5)."""


from fejepa.experiments.gate import g1_prime
from fejepa.experiments.protocol import load_split, mean_std, t_stat
from fejepa.fe.synthetic import generate_synthetic_dataset


def test_split_deterministic(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "ds", n=10, seed=0)
    s1 = load_split(d, n_val=3, seed=1)
    s2 = load_split(d, n_val=3, seed=1)
    assert [f.name for f in s1.val_files] == [f.name for f in s2.val_files]
    s3 = load_split(d, n_val=3, seed=2)
    assert [f.name for f in s3.val_files] != [f.name for f in s1.val_files]


def _cell(disp, egap, vm):
    z = {"mean": 0.0, "std": 0.0, "per_seed": [0.0]}
    return {"disp_rel_l2": {"mean": disp, "std": 0.01, "per_seed": [disp]},
            "energy_gap_rel": {"mean": egap, "std": 0.01, "per_seed": [egap]},
            "vm_rel_l2": {"mean": vm, "std": 0.01, "per_seed": [vm]},
            "peak_vm_rel_err": z, "crit_recall": z}


def _e8(labels, anchored, ar_ft):
    return {"metrics": {"cells": {
        "labels": {64: _cell(*labels)},
        "labels_anchor": {64: _cell(*anchored)},
        "ar_ft": {64: _cell(*ar_ft)}}}}


def _e5(passed):
    return {"metrics": {"passed_all": passed}}


def test_gate_pass():
    g = g1_prime(_e5(True),
                 _e8(labels=(0.24, 3.0, 0.66),
                     anchored=(0.24, 0.5, 0.30),     # egap -83%, vM -55%
                     ar_ft=(0.22, 2.0, 0.6)),        # egap +33%, disp +8%
                 decision_budget=64)
    assert g["conditions"] == {"a": True, "b": True, "c": True} and g["passed"]


def test_gate_fails_closed_when_unmeasured():
    g = g1_prime(_e5(True), None)
    assert not g["passed"] and "unmeasured" in g["reasons"]["b_physics"]


def test_gate_b_needs_both_reductions():
    g = g1_prime(_e5(True),
                 _e8(labels=(0.24, 3.0, 0.66),
                     anchored=(0.24, 0.5, 0.60),     # vM only -9% -> (b) fails
                     ar_ft=(0.20, 2.0, 0.6)))
    assert g["conditions"]["b"] is False and not g["passed"]


def test_stats_helpers():
    a, b = mean_std([1.0, 1.1, 0.9]), mean_std([0.5, 0.6, 0.4])
    assert a["mean"] > b["mean"] and len(a["per_seed"]) == 3
    assert t_stat(a, b, 3) > 0
