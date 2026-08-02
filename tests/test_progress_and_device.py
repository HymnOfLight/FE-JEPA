"""Progress reporting, device plumbing, and the config-crash regression (numpy-only)."""


import numpy as np

from fejepa.experiments.runner import _label_need, _pool_need
from fejepa.models.fejepa import FEJEPAConfig
from fejepa.progress import Task, _hms, stage


def test_task_progress_output(capsys):
    t = Task("E8", total=4)
    t.step("labels b16 s0")
    t.step()
    t.done()
    out = capsys.readouterr().out
    assert "[E8] starting: 4 units" in out
    assert "[E8] 1/4 (25%)" in out and "eta" in out and "labels b16 s0" in out
    assert "[E8] done in" in out


def test_hms_and_stage(capsys):
    assert _hms(3725) == "01:02:05"
    stage("labelling (WP5 economy)")
    assert "=== labelling (WP5 economy) ===" in capsys.readouterr().out


def test_config_from_dict_ignores_unknown_keys():
    # regression: configs carry mgn_dim/mgn_depth in the model block; this crashed
    cfg = FEJEPAConfig.from_dict({"dim": 64, "depth": 2, "heads": 2,
                                  "mgn_dim": 128, "mgn_depth": 8,
                                  "features": {"load_summary": True,
                                               "geometry": False}})
    assert cfg.dim == 64 and cfg.features.geometry is False


def test_label_and_pool_need_mirror_experiment_defaults():
    exps = {"e5": {"enabled": True, "budgets": [4, 8]},          # fit defaults to 8
            "e7": {"enabled": True},                              # fit defaults to 256
            "e6": {"enabled": True}}                              # pool defaults to 256
    assert _label_need(exps) == 256
    assert _pool_need(exps) == 256
    exps2 = {"e8": {"enabled": True, "budgets": [16, 64], "pool_sizes": [128]}}
    assert _label_need(exps2) == 64 and _pool_need(exps2) == 128


def test_device_field_reaches_trainer_configs():
    from fejepa.train.pretrain import PretrainConfig
    from fejepa.train.supervised import SupervisedConfig

    assert SupervisedConfig(device="cuda").device == "cuda"
    assert PretrainConfig(device="cuda").device == "cuda"
    assert SupervisedConfig().desc == "" and PretrainConfig().desc == ""


def test_setup_torch_and_worker_runtime_are_safe_without_torch():
    from fejepa.experiments.parallel import _apply_runtime
    from fejepa.runtime import setup_torch

    policy = setup_torch("cpu", tf32=True)
    assert policy["tf32"] is True and policy["device"] == "cpu"
    # numpy-only environment: must not raise (workers on the box re-apply TF32)
    _apply_runtime({"sup": {"device": "cpu"}, "tf32": True})


def test_cli_end_to_end_numpy(tmp_path, capsys):
    """generate -> info -> label -> prereg(check) -> results --figures, all
    through cli.main: exercises argument plumbing, not just library calls."""
    import json

    from fejepa.cli import main
    from fejepa.data.archive import load_instance
    from fejepa.experiments.protocol import load_split

    ds = tmp_path / "ds"
    main(["generate", str(ds), "--n", "8", "--backend", "synthetic",
          "--seed", "0"])
    assert "generated 8" in capsys.readouterr().out
    assert not load_instance(load_split(ds, 2, seed=1).val_files[0]).labelled

    main(["info", str(ds)])
    info_out = capsys.readouterr().out
    assert '"n_instances": 8' in info_out and '"n_labelled": 0' in info_out

    main(["label", str(ds), "--n-val", "2", "--pool-prefix", "3",
          "--workers", "1"])
    capsys.readouterr()
    split = load_split(ds, 2, seed=1)
    assert all(load_instance(f).labelled for f in split.val_files)
    assert all(load_instance(f).labelled for f in split.pool_files[:3])
    assert not load_instance(split.pool_files[3]).labelled   # economy respected

    cfgp = tmp_path / "cfg.json"
    cfgp.write_text(json.dumps({"a": 1}))
    main(["prereg", str(cfgp), "--prereg-file", str(tmp_path / "P.md")])
    out = capsys.readouterr().out
    assert "config hash:" in out and "missing" in out

    e8 = {"protocol": {"budgets": [16]},
          "metrics": {"cells": {"labels": {16: {
              "disp_rel_l2": {"mean": .3, "std": 0, "per_seed": [.3]},
              "energy_gap_rel": {"mean": .5, "std": 0, "per_seed": [.5]},
              "vm_rel_l2": {"mean": .3, "std": 0, "per_seed": [.3]},
              "peak_vm_rel_err": {"mean": 0, "std": 0, "per_seed": [0]},
              "crit_recall": {"mean": 0, "std": 0, "per_seed": [0]},
              "per_seed_eval": [{}]}}},
              "label_efficiency_auc_disp": {}}, "kills": []}
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps({"results": {"e8": e8}, "provenance": {},
                              "config": {}, "runtime_policy": {}}))
    main(["results", str(rp), "--figures"])
    assert (tmp_path / "RESULTS.md").exists()
    assert (tmp_path / "figure1_energy_gap.png").stat().st_size > 3000


def test_cli_theory_refuses_unlabelled_val(tmp_path):
    import pytest

    from fejepa.cli import main
    from fejepa.fe.synthetic import generate_synthetic_dataset

    generate_synthetic_dataset(tmp_path / "ds", n=6, seed=0)
    with pytest.raises(SystemExit, match="unlabelled"):
        main(["theory", "--data", str(tmp_path / "ds"), "--n-val", "2"])


def test_prereg_stamp_verify_roundtrip(tmp_path):
    import pytest

    from fejepa.report import (PREREG_PLACEHOLDER, read_prereg_hash, stamp_prereg,
                               verify_prereg)

    pf = tmp_path / "PREREG.md"
    pf.write_text(f"frozen gate ...\n\n    CONFIG_SHA256 = {PREREG_PLACEHOLDER}\n")
    cfg = {"a": 1, "experiments": {}}
    with pytest.raises(ValueError, match="unstamped"):
        verify_prereg(cfg, pf)
    h = stamp_prereg(pf, cfg)
    assert read_prereg_hash(pf) == h
    assert verify_prereg(cfg, pf) == h
    with pytest.raises(ValueError, match="mismatch"):
        verify_prereg({"a": 2, "experiments": {}}, pf)
    with pytest.raises(ValueError, match="not found"):
        verify_prereg(cfg, tmp_path / "missing.md")


def test_data_economy_summary_math():
    from fejepa.experiments.runner import data_economy_summary
    from fejepa.fe.solve import SolveLedger

    led = SolveLedger()
    led.add("labelling-val", n=12)
    led.add("labelling-pool-prefix", n=32)
    d = data_economy_summary(led, n_val=3, labelled_prefix=8,
                             unlabeled_pool_depth=64, n_loads=4)
    assert d["labelled_instances"] == 11
    assert d["reference_solves_total"] == 44 == 11 * d["solves_per_labelled_instance"]
    assert d["unlabeled_over_labelled_pool"] == 8.0
    assert d["ledger"]["per_stage"]["labelling-val"] == 12


def test_runner_numpy_smoke_all_disabled_with_prereg_guard(tmp_path):
    """End-to-end run_config without torch: guard verifies, dataset generates,
    val gets labelled, gate fails closed on unmeasured, report lands with the
    data-economy table. Every stage here is numpy-only by construction."""
    import json

    from fejepa.experiments.runner import run_config
    from fejepa.report import stamp_prereg

    pf = tmp_path / "PREREG.md"
    pf.write_text("CONFIG_SHA256 = <fill before tagging>\n")
    cfg = {
        "data": {"dir": str(tmp_path / "ds"), "n": 6, "seed": 0,
                 "backend": "synthetic", "labelled_policy": "economy"},
        "split": {"n_val": 2, "seed": 1},
        "model": {"dim": 8, "depth": 1, "heads": 2},
        "experiments": {},
        "prereg_guard": True, "prereg_file": str(pf),
        "out": str(tmp_path / "report.json"),
        "label_workers": 1,
    }
    stamp_prereg(pf, cfg)
    cp = tmp_path / "cfg.json"
    cp.write_text(json.dumps(cfg))
    payload = run_config(cp)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["prereg"]["config_sha256"] == payload["prereg"]["config_sha256"]
    assert report["gate_g1_prime"]["passed"] is False          # fails closed
    assert report["data_economy"]["labelled_val"] == 2
    assert report["solve_ledger"]["per_stage"]["labelling-val"] == 2 * 4
    md = (tmp_path / "RESULTS.md").read_text()
    assert "**Verdict: NO-GO**" in md and "not run" in md
    assert "labelled instances: 2" in md
    g = report["gate_g1_prime"]
    assert set(g["reasons"]) == {"a_sanity", "b_physics", "c_transfer"}
    assert "unmeasured" in g["reasons"]["a_sanity"]      # fails-closed wording

    import pytest

    cfg["model"]["dim"] = 16                                    # post-tag edit
    cp.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="mismatch"):
        run_config(cp)


def test_run_config_signature_has_overrides():
    import inspect

    from fejepa.experiments.runner import run_config

    params = inspect.signature(run_config).parameters
    assert "device_override" in params and "workers_override" in params


def test_map_units_empty_payloads():
    from fejepa.experiments.parallel import map_units

    assert map_units(lambda p: p, [], workers=4, label="T") == []


def test_map_units_inline_preserves_order_and_verbosity(capsys):
    from fejepa.experiments.parallel import map_units

    payloads = [{"x": i, "tag": f"u{i}"} for i in range(3)]
    out = map_units(lambda p: p["x"] * 10, payloads, workers=1, label="T")
    assert out == [0, 10, 20]
    assert all("quiet" not in p for p in payloads)   # inline keeps milestones on
    assert "[T] 3/3 (100%)" in capsys.readouterr().out


def test_parallel_labelling_matches_serial(tmp_path):

    from fejepa.data.archive import load_instance
    from fejepa.experiments.runner import _label_files
    from fejepa.fe.solve import SolveLedger
    from fejepa.fe.synthetic import generate_synthetic_dataset

    generate_synthetic_dataset(tmp_path / "a", n=6, seed=0)
    generate_synthetic_dataset(tmp_path / "b", n=6, seed=0)
    files1 = sorted((tmp_path / "a").glob("instance_*.npz"))
    files2 = sorted((tmp_path / "b").glob("instance_*.npz"))
    l1, l2 = SolveLedger(), SolveLedger()
    _label_files(files1, l1, "lab", workers=1)
    _label_files(files2, l2, "lab", workers=2)
    assert l1.as_dict()["per_stage"] == l2.as_dict()["per_stage"]
    for f1, f2 in zip(files1, files2, strict=True):

        a, b = load_instance(f1), load_instance(f2)
        assert a.labelled and b.labelled
        assert np.allclose(a.U_star, b.U_star, rtol=0, atol=0)
