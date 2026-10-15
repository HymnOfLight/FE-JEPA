"""fine3d (PREREG_FINE3D Sec. 3, R3): scripts/r3_energy_finetune.py on a CPU
miniature of the Phase-2b run (E8 AR, P3 zero-shot and labelled few-shot at
budgets of 2 and 3). The script verifies its inputs, reproduces the report's
arrays before training (both halves of the check), trains label-free on the
few-shot prefix in the order P3's labelled fine-tuning trained, records what
the adjudicator reads, survives an interruption, does not run a finished
seed again, moves an unusable result aside, and refuses what it should."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fejepa.data.archive import instance_files, load_instance, save_instance
from fejepa.experiments.runner import run_config
from fejepa.fe.synthetic import generate_synthetic_dataset
from fejepa.metrics import FIELD_KEYS
from fejepa.report import _git_describe, config_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}
TRAIN = {"set": "fine_fewshot_prefix", "n": 2, "loss": "ar", "epochs": 2, "lr": 1.5e-3,
         "weight_decay": 1e-4, "clip": 1.0, "warmup_frac": 0.05}


def _script():
    spec = importlib.util.spec_from_file_location("r3_energy_finetune",
                                                  ROOT / "scripts" / "r3_energy_finetune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R3 = _script()


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _name(pack):
    return Path(str(pack["arch"].path)).name


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    """A miniature 'Phase-2b' report with its states and corpora; the order in
    which each of P3's labelled few-shot units visited its training instances
    is recorded (p3_orders.json, by unit)."""
    from fejepa.train import supervised as sv

    tmp = tmp_path_factory.mktemp("r3")
    data = generate_synthetic_dataset(tmp / "corpus", n=12, seed=3, labelled="all")
    fine = generate_synthetic_dataset(tmp / "fine", n=6, seed=4, labelled="all")
    cfg = {"data": {"dir": str(data), "n": 12, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "asis"},
           "data_transfer": {"dir": str(fine), "n": 6, "seed": 4, "backend": "synthetic",
                             "labelled_policy": "asis",
                             "split": {"n_eval": 3, "n_fewshot_prefix": 3}},
           "split": {"n_val": 4, "seed": 1}, "model": MODEL,
           "sup": {"epochs": 1, "lr": 1e-3}, "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {"e8": {"enabled": True, "budgets": [2], "pool_sizes": [4],
                                  "seeds": 2, "ar_epochs": 2, "sup_epochs": 1,
                                  "include_mgn": False, "ar_only": True},
                           "p3_transfer": {"enabled": True, "fewshot_budgets": [2, 3],
                                           "fewshot_epochs": 2, "fewshot_lr": 1.5e-3,
                                           "naive_budget": 0}},
           "gate_g2": {"sanity_x": 3.0, "naive_set": [], "parity_band": 0.1,
                       "egap_adv_min": 0.4, "transfer_win": 1.25, "decision_budget": 2,
                       "logic": "a AND (b OR c)"},
           "device": "cpu", "workers": 1, "tf32": False, "prereg_guard": False,
           "out": str(tmp / "p2b" / "report.json")}
    (tmp / "p2b.json").write_text(json.dumps(cfg))
    orders, real = {}, sv.train_supervised

    def spy(model, train_archs, val_archs, scfg, pretrained_state=None):
        seen = orders.setdefault(scfg.desc or "", [])
        fwd = model.forward_instance

        def rec(pack, *a, **k):
            if torch.is_grad_enabled():                    # training, not evaluation
                seen.append(_name(pack))
            return fwd(pack, *a, **k)
        model.forward_instance = rec
        return real(model, train_archs, val_archs, scfg, pretrained_state=pretrained_state)
    sv.train_supervised = spy                              # P3's units import it per call
    try:
        run_config(tmp / "p2b.json")
    finally:
        sv.train_supervised = real
    (tmp / "p3_orders.json").write_text(json.dumps(orders))
    return tmp


def _r3cfg(mini, out, **over):
    rp = mini / "p2b" / "report.json"
    c = {"source": {"report": str(rp), "report_sha256": _sha(rp),
                    "states_dir": str(mini / "p2b" / "e8_states")},
         "train": dict(TRAIN),
         "seeds": [0, 1], "evaluation": {"fine": 3, "inband": 4, "amplitude": True},
         "content_check": {"n_fine": 2, "n_inband": 2, "max_rel_dev": 1e-6},
         "out": str(out / "report.json"), "prereg_file": "PREREG_FINE3D.md",
         "prereg_guard": False, "device": "cpu"}
    for k, v in over.items():
        c[k] = v
    p = out.parent / f"{out.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(c))
    return p


@pytest.fixture(scope="module")
def done(mini):
    p = _r3cfg(mini, mini / "runs" / "A")
    return R3.run(p), p


def test_r3_runs_reproduces_and_evaluates(done, mini):
    rep, p = done
    cfg = json.loads(p.read_text())
    assert sorted(rep["per_seed"]) == ["s0", "s1"]
    names = [f.name for f in instance_files(mini / "fine")]
    for s in (0, 1):
        r = rep["per_seed"][f"s{s}"]
        assert r["content_check"]["passed"] and r["content_check"]["max_rel_dev"] == \
            {"fine": 0.0, "inband": 0.0}                       # CPU: bit for bit
        assert r["seed"] == s and r["git"] == _git_describe()
        assert r["config_sha256"] == config_sha256(cfg) == rep["provenance"]["config_sha256"]
        t = r["train"]
        assert t["files"] == names[3:5] and t["steps"] == 4 and t["seed"] == s
        assert t["epochs"] == t["epochs_run"] == len(t["loss_by_epoch"]) == 2
        assert all(math.isfinite(x) for x in t["loss_by_epoch"])
        orders = R3.epoch_orders(s, 2, 2)
        assert t["order_sha256"] == hashlib.sha256(json.dumps(orders).encode()).hexdigest()
        assert t["first_epoch_order"] == orders[0] and t["resumed_from_epoch"] is None
        assert r["source_state"]["sha256"] == _sha(mini / "p2b" / "e8_states" / f"ar_p4_s{s}.pt")
        assert r["state"]["sha256"] == _sha(r["state"]["path"])
        for name, n in (("fine", 3), ("inband", 4)):
            assert r[name]["n_val"] == n and len(r[name]["per_instance"]["vm_rel_l2"]) == n
            assert len(r[name]["amplitude"]["per_instance"]["c_star"]) == n
        assert set(r["seconds"]) == {"load_and_check", "train", "evaluate"}
        assert not r.get("from_disk")
        ck = Path(r["state"]["path"]).with_suffix(".ckpt")
        assert not ck.exists() and not ck.with_name(ck.name + ".json").exists()
    p2b = json.loads((mini / "p2b" / "report.json").read_text())
    prov = rep["provenance"]
    assert prov["seeds"] == [0, 1] and rep["prereg"] is None        # the guard is off here
    assert prov["source_report_sha256"] == _sha(mini / "p2b" / "report.json")
    assert prov["source_config_sha256"] == p2b["provenance"]["config_sha256"]
    assert {d["manifest_sha256"] for d in prov["datasets"]} == \
        {d["manifest_sha256"] for d in p2b["provenance"]["datasets"]}
    assert set(rep["runtime_policy"]) >= {"torch", "tf32"}
    # training moved the states
    assert rep["per_seed"]["s0"]["state"]["sha256"] != rep["per_seed"]["s0"]["source_state"][
        "sha256"]


def test_r3_trains_without_labels(mini, tmp_path):
    """The training instances' labels deleted on disk (a copy of the fine
    corpus, manifest unchanged): R3 trains and gives the same state."""
    import shutil

    fine2 = tmp_path / "fine"
    shutil.copytree(mini / "fine", fine2)
    files = instance_files(fine2)
    for f in files[3:5]:
        a = load_instance(f)
        a.U_star = None
        save_instance(a, f)
        assert not load_instance(f).labelled
    p = _r3cfg(mini, tmp_path / "B", seeds=[0])
    rep = R3.run(p, fine_dir=str(fine2))
    first = json.loads((mini / "runs" / "A" / "seed_0.json").read_text())
    assert rep["per_seed"]["s0"]["state"]["sha256"] == first["state"]["sha256"]


def test_r3_trains_in_the_order_p3s_labelled_fine_tuning_trained(mini, tmp_path, monkeypatch):
    """Through the script, at a budget of 3: each seed visits the instances in
    exactly the order P3's labelled unit of that seed and budget visited them
    (recorded while the miniature Phase-2b run trained), which is the order
    `epoch_orders` gives and the report hashes."""
    from fejepa.train import pretrain as pt

    real, seen = pt.compute_loss, []

    def spy(model, pack, *a, **k):
        seen.append(_name(pack))
        return real(model, pack, *a, **k)
    monkeypatch.setattr(pt, "compute_loss", spy)
    rep = R3.run(_r3cfg(mini, tmp_path / "O", train=dict(TRAIN, n=3)))
    p3 = json.loads((mini / "p3_orders.json").read_text())
    names = [f.name for f in instance_files(mini / "fine")][3:6]
    assert len(seen) == 2 * 3 * 2
    for i, s in enumerate((0, 1)):
        mine = seen[i * 6:(i + 1) * 6]
        assert mine == p3[f"P3 finetune b3 s{s}"] == p3[f"P3 scratch b3 s{s}"]
        orders = R3.epoch_orders(s, 3, 2)
        assert mine == [names[j] for o in orders for j in o]
        t = rep["per_seed"][f"s{s}"]["train"]
        assert t["files"] == names and t["steps"] == 6
        assert t["order_sha256"] == hashlib.sha256(json.dumps(orders).encode()).hexdigest()
    assert seen[:6] != seen[6:]                           # the seeds' orders differ


def test_both_trainers_draw_the_same_order(mini):
    """pretrain and train_supervised alike: one generator from the seed, one
    permutation per epoch, nothing else drawn on the AR path."""
    from fejepa.experiments.parallel import _build_model
    from fejepa.train import losses
    from fejepa.train import pretrain as pt
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    files = instance_files(mini / "fine")[3:6]
    archs = [load_instance(f) for f in files]
    for seed in (0, 1, 5):
        seen_sup, seen_ar = [], []
        m = _build_model({"kind": "fejepa", "model": MODEL, "seed": seed})
        real_fwd = m.forward_instance

        def fwd(pack, _real=real_fwd, _seen=seen_sup):
            if torch.is_grad_enabled():
                _seen.append(_name(pack))
            return _real(pack)
        m.forward_instance = fwd
        train_supervised(m, archs, archs[:1], SupervisedConfig(epochs=3, seed=seed,
                                                               log_every=-1))
        m2 = _build_model({"kind": "fejepa", "model": MODEL, "seed": seed})
        real_loss = losses.compute_loss

        def spy(model, pack, *a, _seen=seen_ar, **k):
            _seen.append(_name(pack))
            return real_loss(model, pack, *a, **k)
        pt.compute_loss = spy
        try:
            pt.pretrain(m2, archs, pt.PretrainConfig(epochs=3, seed=seed, log_every=-1))
        finally:
            pt.compute_loss = real_loss
        names = [f.name for f in files]
        assert seen_sup == seen_ar == [names[j] for o in R3.epoch_orders(seed, 3, 3) for j in o]


def test_r3_resumes_an_interrupted_seed(mini, tmp_path, monkeypatch):
    from fejepa.train import pretrain as pt

    real, calls = pt.compute_loss, {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:                          # epoch 2 of seed 0 (2 steps per epoch)
            raise RuntimeError("simulated interruption")
        return real(*a, **k)
    monkeypatch.setattr(pt, "compute_loss", boom)
    p = _r3cfg(mini, tmp_path / "C")
    with pytest.raises(RuntimeError, match="simulated"):
        R3.run(p)
    assert (tmp_path / "C" / "eft_b2_s0.ckpt").exists()
    assert not (tmp_path / "C" / "seed_0.json").exists()
    monkeypatch.setattr(pt, "compute_loss", real)
    rep = R3.run(p)
    first = json.loads((mini / "runs" / "A" / "seed_0.json").read_text())
    t = rep["per_seed"]["s0"]["train"]
    assert t["resumed_from_epoch"] == 1 and t["epochs_run"] == 2 and t["steps"] == 4
    assert t["loss_by_epoch"] == first["train"]["loss_by_epoch"]
    assert rep["per_seed"]["s0"]["state"]["sha256"] == first["state"]["sha256"]
    assert rep["per_seed"]["s1"]["train"]["resumed_from_epoch"] is None


def test_r3_does_not_run_a_finished_seed_again(done, capsys):
    rep, p = done
    again = R3.run(p)
    assert "result on disk" in capsys.readouterr().out
    assert all(again["per_seed"][k] == dict(rep["per_seed"][k], from_disk=True)
               for k in ("s0", "s1"))
    s1 = Path(rep["per_seed"]["s1"]["state"]["path"])
    s1.unlink()
    third = R3.run(p)
    assert "seed 1: seed_1.json unusable" in capsys.readouterr().out
    assert len(list(s1.parent.glob("seed_1.json.unusable.*"))) == 1
    assert third["per_seed"]["s1"]["state"]["sha256"] == rep["per_seed"]["s1"]["state"]["sha256"]
    assert not third["per_seed"]["s1"].get("from_disk") and third["per_seed"]["s0"]["from_disk"]


def _patch_adjudicator(monkeypatch, mini, label):
    """adjudicate_fine3d's R3 constants set to the miniature's (budget 2, 4
    steps, two seeds, 3 + 4 evaluation instances; the run's directory)."""
    from fejepa.analysis import adjudicate_fine3d as A

    names = [f.name for f in instance_files(mini / "fine")]
    for k, v in {"R3_SEEDS": [0, 1], "R3_TRAIN_N": 2, "R3_STEPS": 4,
                 "R3_TRAIN_FILES": names[3:5], "N_FINE": 3, "N_INBAND": 4, "R3_LABEL": label,
                 "PHASE2B_REPORT_SHA256": _sha(mini / "p2b" / "report.json")}.items():
        monkeypatch.setattr(A, k, v)
    return A


def _stamped(rep):
    """The report as a guard-verified run would carry it (the guard is off here)."""
    return dict(rep, prereg={"file": "PREREG_FINE3D.md",
                             "config_sha256": rep["provenance"]["config_sha256"]})


@pytest.mark.parametrize("what", ["control", "unreadable", "another seed", "check failed",
                                  "state changed", "another commit", "another configuration"])
def test_r3_reuses_only_a_usable_result(mini, tmp_path, what):
    """The configuration's own result is reused (control); one field changed
    makes it unusable: moved aside, the seed run again."""
    out = tmp_path / "U"
    cfgp = _r3cfg(mini, out, seeds=[0])
    R3.run(cfgp)
    rf = out / "seed_0.json"
    first = json.loads(rf.read_text())
    changed = out / "eft_b2_s0_changed.pt"            # a state no longer the recorded one
    changed.write_bytes(Path(first["state"]["path"]).read_bytes() + b"x")
    text = {"control": rf.read_text(),
            "unreadable": rf.read_text()[:100],
            "another seed": json.dumps(dict(first, seed=1)),
            "check failed": json.dumps(dict(first, content_check=dict(first["content_check"],
                                                                    passed=False))),
            "state changed": json.dumps(dict(first, state=dict(first["state"],
                                                               path=str(changed)))),
            "another commit": json.dumps(dict(first, git="prereg-fine3d-1-gabcdef")),
            "another configuration": json.dumps(dict(first, config_sha256="0" * 64))}[what]
    rf.write_text(text)
    rep = R3.run(cfgp)
    aside = list(out.glob("seed_0.json.unusable.*"))
    if what == "control":
        assert rep["per_seed"]["s0"]["from_disk"] is True and not aside
        return
    assert len(aside) == 1 and aside[0].read_text() == text
    assert not rep["per_seed"]["s0"].get("from_disk")
    assert rep["per_seed"]["s0"]["state"]["sha256"] == first["state"]["sha256"]
    assert json.loads(rf.read_text())["seed"] == 0


def test_r3_resumes_only_its_own_checkpoint(mini, tmp_path, monkeypatch):
    """An interrupted seed's checkpoint written by another code (its sidecar
    says so) is moved aside and the seed trains from the start."""
    from fejepa.train import pretrain as pt

    real, calls = pt.compute_loss, {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated interruption")
        return real(*a, **k)
    monkeypatch.setattr(pt, "compute_loss", boom)
    out = tmp_path / "K"
    cfgp = _r3cfg(mini, out, seeds=[0])
    with pytest.raises(RuntimeError, match="simulated"):
        R3.run(cfgp)
    side = out / "eft_b2_s0.ckpt.json"
    assert (out / "eft_b2_s0.ckpt").exists() and json.loads(side.read_text())["git"]
    side.write_text(json.dumps(dict(json.loads(side.read_text()), git="prereg-fine3d-1-gabcdef")))
    monkeypatch.setattr(pt, "compute_loss", real)
    rep = R3.run(cfgp)
    first = json.loads((mini / "runs" / "A" / "seed_0.json").read_text())
    assert len(list(out.glob("eft_b2_s0.ckpt.unusable.*"))) == 1
    assert rep["per_seed"]["s0"]["train"]["resumed_from_epoch"] is None
    assert rep["per_seed"]["s0"]["state"]["sha256"] == first["state"]["sha256"]
    assert not (out / "eft_b2_s0.ckpt").exists() and not side.exists()


def test_r3_counts_epochs_and_steps_from_the_trainers_history(mini, tmp_path, monkeypatch):
    """A trainer that stops after its first epoch: the result says one epoch
    and its steps, not the configuration's, and the adjudicator refuses it."""
    from fejepa.train import pretrain as pt

    real = pt.pretrain

    def short(model, archs, cfg, *a, **k):
        cfg.stop_after_epoch = 1
        return real(model, archs, cfg, *a, **k)
    monkeypatch.setattr(pt, "pretrain", short)
    rep = R3.run(_r3cfg(mini, tmp_path / "H"))
    for s in ("s0", "s1"):
        t = rep["per_seed"][s]["train"]
        assert t["epochs"] == 2 and t["epochs_run"] == 1 and t["steps"] == 2
        assert len(t["loss_by_epoch"]) == 1
    A = _patch_adjudicator(monkeypatch, mini, "H")
    p2b = json.loads((mini / "p2b" / "report.json").read_text())
    with pytest.raises(ValueError, match="trained on"):
        A.check_r3(_stamped(rep), p2b, {"H": rep["provenance"]["config_sha256"]},
                   _git_describe())


@pytest.mark.parametrize("which,key", [("fine", "disp_rel_l2"), ("fine", "energy_gap_rel"),
                                       ("inband", "disp_rel_l2"), ("inband", "energy_gap_rel")])
def test_r3_content_check_reads_both_sets_and_both_metrics(mini, tmp_path, monkeypatch, which,
                                                            key):
    real = R3._ref_arrays

    def shifted(report, w, i):
        out = dict(real(report, w, i))
        if w == which:
            out[key] = [v * 1.01 for v in out[key]]
        return out
    monkeypatch.setattr(R3, "_ref_arrays", shifted)
    with pytest.raises(SystemExit, match="do not reproduce"):
        R3.run(_r3cfg(mini, tmp_path / "X"))
    assert not list((tmp_path / "X").glob("*.pt"))        # stopped before training


def test_r3_refusals(mini, tmp_path):
    rp = mini / "p2b" / "report.json"
    with pytest.raises(SystemExit, match="not the source report"):
        R3.run(_r3cfg(mini, tmp_path / "R1", source={"report": str(rp), "report_sha256": "0" * 64,
                                                      "states_dir": str(mini / "p2b" / "e8_states")}))
    with pytest.raises(SystemExit, match="warmup_frac"):
        R3.run(_r3cfg(mini, tmp_path / "R2", train=dict(TRAIN, warmup_frac=0.1)))
    with pytest.raises(SystemExit, match="loss"):
        R3.run(_r3cfg(mini, tmp_path / "R3", train=dict(TRAIN, loss="labels")))
    with pytest.raises(SystemExit, match="loss"):
        R3.run(_r3cfg(mini, tmp_path / "R3b", train=dict(TRAIN, set="inband")))
    with pytest.raises(SystemExit, match="trained"):
        R3.run(_r3cfg(mini, tmp_path / "R4", seeds=[0, 7]))
    with pytest.raises(SystemExit, match="R3 needs"):
        R3.run(_r3cfg(mini, tmp_path / "R4b", train=dict(TRAIN, n=4)))
    with pytest.raises(SystemExit, match="smaller"):
        R3.run(_r3cfg(mini, tmp_path / "R4c", evaluation={"fine": 3, "inband": 5}))
    pf = tmp_path / "PREREG_X.md"
    pf.write_text("CONFIG_SHA256[R5] = <fill before tagging>\n")
    with pytest.raises(ValueError, match="unstamped"):
        R3.run(_r3cfg(mini, tmp_path / "R5", prereg_guard=True, prereg_file=str(pf)))
    assert not any((tmp_path / d).exists() for d in ("R1", "R2", "R3", "R4", "R5"))


def test_r3_dry_run_trains_nothing(mini, tmp_path):
    out = R3.run(_r3cfg(mini, tmp_path / "D"), dry_run=True)
    assert out["dry_run"] and out["states_verified"] == 2 and out["train"] == \
        [f.name for f in instance_files(mini / "fine")[3:5]]
    assert not (tmp_path / "D").exists()


def test_r3_report_is_what_the_adjudicator_reads(done, mini, monkeypatch):
    """The miniature's report through adjudicate_fine3d's R3 reader, with the
    constants set to the miniature's (budget 2, 4 steps, two seeds, 3 + 4
    evaluation instances): every field the reader needs is there and no
    reading is left unevaluated."""
    rep, _ = done
    A = _patch_adjudicator(monkeypatch, mini, "A")
    p2b = json.loads((mini / "p2b" / "report.json").read_text())
    stamped = _stamped(rep)
    out = A.adjudicate_r3(stamped, p2b, {"A": rep["provenance"]["config_sha256"]},
                          expected_git=_git_describe())
    assert set(out["verdicts"]) == {"R3-E", "R3-V"}
    assert all(not v["verdict"].startswith("NOT EVALUATED") for v in out["verdicts"].values())
    rd = out["readings"]
    assert "not_evaluated" not in rd["amplitude"]
    for part in ("vs_labelled_ft", "vs_zero_shot", "inband_vs_before"):
        assert set(rd[part]) == set(FIELD_KEYS)
        assert all("not_evaluated" not in v["guard"] for v in rd[part].values())
    assert out["deviations"] == [] and out["notes"] == []
    assert out["checks"]["seeds"]["s0"]["state_sha256"] == rep["per_seed"]["s0"]["state"]["sha256"]
    with pytest.raises(ValueError, match="stamped"):
        A.adjudicate_r3(rep, p2b, None, expected_git=None)          # the guard was off
