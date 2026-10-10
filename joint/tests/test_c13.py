"""Checks of the G-C1.3 run code (c13). Training tests need torch and the FE-JEPA repository: set FEJEPA_REPO to
its checkout, whose files c13/config.py pins; otherwise they are skipped."""
import copy
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from c13.config import config, sha256, SEEDS_SEEN
from c13 import data
from fejoint.export import solve_exported

REPO = os.environ.get("FEJEPA_REPO")
TEST_SEEDS = (1, 3)        # seeds seen before the stamp: the tests never draw, export or solve a member of the gate


def small_cfg(n_train=2, n_eval=2, epochs=2):
    cfg = copy.deepcopy(config())
    d = cfg["data"]
    d["n_train"], d["n_eval"] = n_train, n_eval
    d["train_seed"], d["eval_seed"] = TEST_SEEDS
    d["seeds_seen_before_stamp"] = [s for s in d["seeds_seen_before_stamp"] if s not in TEST_SEEDS]
    cfg["model"].update(dim=16, depth=1, heads=2)
    for a in cfg["arms"].values():
        a["epochs"] = epochs
        a["labels"] = min(a["labels"], n_train)
    return cfg


def test_config_hash_is_stable_and_seeds_are_fresh():
    a, b = config(), config()
    assert sha256(a) == sha256(b)
    assert a["data"]["train_seed"] not in SEEDS_SEEN and a["data"]["eval_seed"] not in SEEDS_SEEN
    assert set(TEST_SEEDS) <= set(SEEDS_SEEN) and a["data"]["seeds_seen_before_stamp"] == SEEDS_SEEN
    c = copy.deepcopy(a); c["map"]["T"] = 2e-3
    assert sha256(c) != sha256(a)
    with pytest.raises(AssertionError):                 # a seed seen before the stamp is refused
        c = copy.deepcopy(a); c["data"]["train_seed"] = 101
        data.members(c)


def test_members_are_deterministic():
    cfg = small_cfg()
    m1, m2 = data.members(cfg), data.members(cfg)
    assert [x[0] for x in m1] == [x[0] for x in m2] and [x[2] for x in m1] == [x[2] for x in m2]
    assert sum(1 for x in m1 if x[1] == "tested") == 4


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    cfg = small_cfg()
    d = tmp_path_factory.mktemp("c13data")
    data.generate(cfg, d)
    man = data.manifest(d)
    data.label(d, [r["id"] for r in man["instances"]])
    return cfg, d


def test_generate_and_load_round_trip(dataset):
    cfg, d = dataset
    man = data.manifest(d)
    assert man["config_sha256"] == sha256(cfg) and man["n"] == 2 + 2 + 4
    r = man["instances"][0]
    ex, q = data.load_export(d / r["file"])
    direct = data.export_member(cfg, q)
    s1, s2 = solve_exported(ex), solve_exported(direct)
    assert abs(s1["S_paper"] / s2["S_paper"] - 1) < 1e-12
    lab = data.load_label(d, r["id"])
    assert abs(lab["S_paper"] / s2["S_paper"] - 1) < 1e-12
    ledger = json.loads((d / "labels" / "ledger.json").read_text())
    assert len(ledger) == man["n"] and all(e["kkt"]["min_gap"] > -1e-12 for e in ledger)
    data.check_files(d, [r["id"] for r in man["instances"]])
    r = data.check_labels(cfg, d)
    assert r["ok"] and r["labels"] == man["n"], r


def test_files_of_another_configuration_or_a_changed_label_are_refused(dataset, tmp_path):
    cfg, d = dataset
    c = tmp_path / "copy"
    shutil.copytree(d, c)
    other = copy.deepcopy(cfg); other["data"]["P_full_N"] = 20000.0
    with pytest.raises(RuntimeError):
        data.generate(other, c)
    iid = data.manifest(c)["instances"][0]["id"]
    lab = data.load_label(c, iid)
    np.savez_compressed(c / "labels" / f"{iid}.npz", u=2 * lab["u"], Pi=lab["Pi"], S_paper=lab["S_paper"])
    with pytest.raises(RuntimeError):
        data.check_files(c, [iid])
    shutil.copy(d / "labels" / f"{iid}.npz", c / "labels" / f"{iid}.npz")
    led = json.loads((c / "labels" / "ledger.json").read_text())
    led[0]["kkt"]["max_residual_unconstrained"] = 1e-3
    (c / "labels" / "ledger.json").write_text(json.dumps(led))
    assert not data.check_labels(cfg, c)["ok"]


def test_evaluation_of_exact_predictions(dataset, tmp_path):
    """Predictions equal to the reference: zero errors, full contact overlap, PASS."""
    cfg, d = dataset
    from c13.evaluate import evaluate
    man = data.manifest(d)
    ids = [r["id"] for r in man["instances"] if r["split"] in ("eval", "tested")]
    for arm, a in cfg["arms"].items():
        for s in a["seeds"]:
            run = tmp_path / f"{arm}_s{s}"; run.mkdir()
            np.savez_compressed(run / "preds.npz", **{i: data.load_label(d, i)["u"] for i in ids})
    v = evaluate(cfg, d, tmp_path)
    assert v["verdict"] == "PASS" and v["C1_value"] < 1e-9 and v["C2_value"] == 1.0
    ev = v["arms"]["label_free"]["0"]["eval"]
    assert ev["gap"]["max"] < 1e-9 and ev["vm_rel_l2"]["max"] < 1e-9 and ev["zero_field_pass"]["min"] == 1.0
    assert abs(ev["alpha"]["min"] - 1) < 1e-9 and abs(ev["alpha"]["max"] - 1) < 1e-9     # Clapeyron: f.u* = u*.K.u*
    assert ev["abs_S_err_rescaled"]["median"] < 1e-9


def test_non_finite_and_missing_label_free_seeds(dataset, tmp_path):
    """A non-finite label-free seed enters the medians as |S err| = inf and IoU = 0; a missing one gives INCOMPLETE."""
    cfg, d = dataset
    from c13.evaluate import evaluate
    man = data.manifest(d)
    ids = [r["id"] for r in man["instances"] if r["split"] in ("eval", "tested")]
    for arm, a in cfg["arms"].items():
        for s in a["seeds"]:
            run = tmp_path / f"{arm}_s{s}"; run.mkdir()
            if (arm, s) in (("label_free", 1), ("label_free", 2), ("supervised_16", 0)):
                (run / "history.json").write_text(json.dumps({"status": "non-finite loss at step 7"}))
            else:
                np.savez_compressed(run / "preds.npz", **{i: data.load_label(d, i)["u"] for i in ids})
    v = evaluate(cfg, d, tmp_path)
    assert v["verdict"] == "FAIL" and v["C1_value"] == float("inf") and v["C2_value"] == 0.0
    assert set(v["non_finite"]) == {"label_free_s1", "label_free_s2", "supervised_16_s0"} and not v["missing"]
    np.savez_compressed(tmp_path / "label_free_s1" / "preds.npz", **{i: data.load_label(d, i)["u"] for i in ids})
    v = evaluate(cfg, d, tmp_path)                        # predictions do not undo a non-finite stop
    assert v["status"]["label_free_s1"] == "non-finite" and v["verdict"] == "FAIL"
    (tmp_path / "label_free_s1" / "history.json").unlink()
    v = evaluate(cfg, d, tmp_path)                        # one non-finite seed of three: the median still passes
    assert v["verdict"] == "PASS" and v["per_seed_medians"]["iou"] == [1.0, 1.0, 0.0]
    (tmp_path / "label_free_s2" / "history.json").unlink()
    v = evaluate(cfg, d, tmp_path)
    assert v["verdict"] == "INCOMPLETE" and v["missing"] == ["label_free_s2"] and "C1_value" not in v
    assert v["arms"]["label_free"] == {"withheld": "the verdict is INCOMPLETE"}


def test_evaluation_of_the_zero_field_fails(dataset, tmp_path):
    cfg, d = dataset
    from c13.evaluate import evaluate
    man = data.manifest(d)
    ids = [r["id"] for r in man["instances"] if r["split"] in ("eval", "tested")]
    for arm, a in cfg["arms"].items():
        for s in a["seeds"]:
            run = tmp_path / f"{arm}_s{s}"; run.mkdir()
            np.savez_compressed(run / "preds.npz", **{i: np.zeros_like(data.load_label(d, i)["u"]) for i in ids})
    v = evaluate(cfg, d, tmp_path)
    assert v["verdict"] == "FAIL"
    ev = v["arms"]["label_free"]["0"]["eval"]
    # with the beam segment condensed, the zero nodal field leaves the end section free: Pi(0) = const < 0
    assert 0.9 < ev["gap"]["median"] < 1.0 and ev["zero_field_pass"]["max"] == 0.0


@pytest.mark.skipif(REPO is None, reason="FEJEPA_REPO not set")
@pytest.mark.parametrize("arm", ["label_free", "supervised_16"])
def test_train_resume_and_predict(dataset, tmp_path, arm, monkeypatch):
    """A tiny model trains, an interrupted run resumes (checkpoint loaded on the CPU) to the same end state as an
    uninterrupted one, and a finished run returns without loading its members."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(REPO) / "src"))
    import torch
    import c13.train as ct
    from c13.train import train
    cfg, d = dataset
    a_full, a_int = tmp_path / "full", tmp_path / "interrupted"
    h1 = train(cfg, arm, 0, d, a_full, device="cpu", ckpt_every_epochs=1)
    assert h1["status"] == "finished" and np.isfinite(h1["loss"]).all()
    assert h1["config_sha256"] == sha256(cfg) and h1["matmul_precision"] == "highest"
    train(cfg, arm, 0, d, a_int, device="cpu", ckpt_every_epochs=1, stop_after_epoch=1)
    assert not (a_int / f"{arm}_s0" / "preds.npz").exists()
    where, real = [], torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: where.append(k.get("map_location")) or real(*a, **k))
    h2 = train(cfg, arm, 0, d, a_int, device="cpu", ckpt_every_epochs=1)
    monkeypatch.undo()
    assert where == ["cpu"]
    assert h2["resumed_from_epoch"] == 1 and h2["resumes"] == [1] and h2["status"] == "finished"
    monkeypatch.setattr(ct, "_prepare", lambda *a, **k: pytest.fail("a finished run loaded its members"))
    assert train(cfg, arm, 0, d, a_full, device="cpu")["status"] == "finished"
    monkeypatch.undo()
    s1 = torch.load(a_full / f"{arm}_s0" / "state.pt", weights_only=False)
    s2 = torch.load(a_int / f"{arm}_s0" / "state.pt", weights_only=False)
    for k in s1["enc"]:
        assert torch.equal(s1["enc"][k], s2["enc"][k]), k
    with np.load(a_full / f"{arm}_s0" / "preds.npz") as z1, np.load(a_int / f"{arm}_s0" / "preds.npz") as z2:
        assert set(z1.files) == set(z2.files) and all(z1[k].dtype == np.float32 for k in z1.files)
        for k in z1.files:
            assert np.array_equal(z1[k], z2[k]) and z1[k].size > 0


@pytest.mark.skipif(REPO is None, reason="FEJEPA_REPO not set")
def test_a_non_finite_stop_is_final(dataset, tmp_path, monkeypatch):
    """A run that meets a non-finite loss records it, writes no predictions, and is never retried."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(Path(REPO) / "src"))
    import c13.train as ct
    cfg, d = dataset
    real = ct.field
    monkeypatch.setattr(ct, "field", lambda *a, **k: real(*a, **k) * float("nan"))
    h1 = ct.train(cfg, "label_free", 1, d, tmp_path, device="cpu", ckpt_every_epochs=1)
    monkeypatch.undo()
    run = tmp_path / "label_free_s1"
    assert h1["status"] == "non-finite loss at step 0" and not (run / "preds.npz").exists()
    before = (run / "history.json").read_text()
    h2 = ct.train(cfg, "label_free", 1, d, tmp_path, device="cpu", ckpt_every_epochs=1)
    assert h2["status"] == h1["status"] and not (run / "preds.npz").exists()
    assert (run / "history.json").read_text() == before
    import torch                                          # non-finite predictions after a finite training: final too
    monkeypatch.setattr(ct, "field", lambda *a, **k: real(*a, **k) * (1.0 if torch.is_grad_enabled() else float("nan")))
    h3 = ct.train(cfg, "label_free", 2, d, tmp_path, device="cpu", ckpt_every_epochs=1)
    monkeypatch.undo()
    run2 = tmp_path / "label_free_s2"
    assert h3["status"] == "non-finite predictions" and not (run2 / "preds.npz").exists()
    assert ct.train(cfg, "label_free", 2, d, tmp_path, device="cpu")["status"] == "non-finite predictions"
    assert not (run2 / "preds.npz").exists() and (run2 / "attempts.log").read_text().count("start") == 2
    (tmp_path / "y" / "label_free_s0").mkdir(parents=True)     # a run folder of another configuration is refused
    (tmp_path / "y" / "label_free_s0" / "history.json").write_text(json.dumps({"config_sha256": "0" * 64}))
    with pytest.raises(RuntimeError):
        ct.train(cfg, "label_free", 0, d, tmp_path / "y", device="cpu")


@pytest.mark.skipif(REPO is None, reason="FEJEPA_REPO not set")
def test_repository_files_are_checked(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(REPO) / "src"))
    from c13.train import verify_repository
    cfg = config()
    assert Path(verify_repository(cfg)) == Path(REPO).resolve()
    bad = copy.deepcopy(cfg); bad["repository_code"]["src/fejepa/train/schedule.py"] = "0" * 64
    with pytest.raises(RuntimeError):
        verify_repository(bad)
