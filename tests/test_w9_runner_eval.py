"""wp9 Stage 0b: the runner's evaluation additions -- evaluation-only holdout
sets with amplitude readings, an evaluation-only E8 that reuses another run's
verified states, and the run-time checkpointing override. A configuration
without them runs exactly as before; with them, training is unchanged."""

import hashlib
import json
import shutil

import numpy as np
import pytest

pytest.importorskip("torch")

from fejepa.data.archive import instance_files, load_instance, save_instance, write_manifest
from fejepa.experiments.runner import run_config
from fejepa.fe.synthetic import generate_synthetic_dataset

MODEL = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}


def _family(out, src_dir, family, n=None):
    """An OOD-2D-style family directory (labelled instances, per-file SHA-256)."""
    out.mkdir(parents=True)
    recs = []
    for i, f in enumerate(instance_files(src_dir)[:n]):
        a = load_instance(f)
        name = f"instance_{i:05d}.npz"
        save_instance(a, out / name)
        recs.append({"file": name, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
                     "n_nodes": a.n_nodes, "labelled": True})
    write_manifest(out, recs, {"family": family, "seed": 1})
    return out


def _cfg(tmp, name, data, evaluation=None, **e8):
    cfg = {"data": {"dir": str(data), "n": 12, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "asis"},
           "split": {"n_val": 4, "seed": 1}, "model": MODEL,
           "sup": {"epochs": 1, "lr": 1e-3}, "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {"e8": {"enabled": True, "budgets": [2], "pool_sizes": [4],
                                  "seeds": 2, "ar_epochs": 2, "sup_epochs": 1,
                                  "include_mgn": False, "ar_only": True, **e8}},
           "device": "cpu", "workers": 1, "tf32": False, "prereg_guard": False,
           "out": str(tmp / name / "report.json")}
    if evaluation is not None:
        cfg["evaluation"] = evaluation
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w9run")
    data = generate_synthetic_dataset(tmp / "corpus", n=12, seed=3, labelled="all")
    fa = generate_synthetic_dataset(tmp / "fa_src", n=3, seed=8, labelled="all", nx=12, ny=8)
    fb = generate_synthetic_dataset(tmp / "fb_src", n=2, seed=9, labelled="all", nx=5, ny=4)
    ev = {"holdouts": {"FA": {"dir": str(_family(tmp / "FA", fa, "F5")), "family": "F5"},
                       "FB": {"dir": str(_family(tmp / "FB", fb, "F2")), "family": "F2"}},
          "amplitude": True}
    rep = {}
    rep["A"] = run_config(_cfg(tmp, "A", data))                         # wp8 form
    rep["B"] = run_config(_cfg(tmp, "B", data, ev))                     # + evaluation
    rep["C"] = run_config(_cfg(tmp, "C", data, ev, reuse_from={
        "report": str(tmp / "A" / "report.json"), "states_dir": str(tmp / "A" / "e8_states")}))
    rep["D"] = run_config(_cfg(tmp, "D", data), activation_checkpointing=False)
    return tmp, data, ev, rep


def _states(tmp, name):
    return {p.name: p.read_bytes() for p in sorted((tmp / name / "e8_states").glob("ar_*.pt"))}


def test_default_configuration_gains_nothing(world):
    tmp, _, _, rep = world
    a = rep["A"]
    assert "evaluation" not in a and "reuse_from" not in a and "runtime_overrides" not in a
    e8 = a["results"]["e8"]
    assert "holdouts" not in e8["metrics"] and "holdouts" not in e8["protocol"]
    assert "amplitude" not in e8["protocol"] and "reused_from" not in e8["metrics"]["d9_restart"]
    ev = _cell(a, "cells", "ar", "4")["per_seed_eval"][0]
    assert set(ev) == {"disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err",
                       "crit_recall", "per_instance", "n_val"}


def _cell(rep, *path):
    m = rep["results"]["e8"]["metrics"]
    for k in path:
        m = m[k] if k in m else m[int(k)]
    return m


def test_evaluation_block_leaves_training_and_validation_unchanged(world):
    tmp, _, _, rep = world
    assert _states(tmp, "A") == _states(tmp, "B")                     # the same training
    for s in range(2):
        a = _cell(rep["A"], "cells", "ar", "4")["per_seed_eval"][s]
        b = _cell(rep["B"], "cells", "ar", "4")["per_seed_eval"][s]
        assert a["per_instance"] == b["per_instance"]                 # the same metrics
        assert set(b["amplitude"]["per_instance"]) == {"c_star", "c_battery", "disp_c",
                                                       "egap_c", "u_norm_K", "ustar_norm_K"}
        assert len(b["amplitude"]["per_instance"]["c_star"]) == 4


def test_holdouts_are_evaluated_and_recorded(world):
    tmp, _, ev, rep = world
    b = rep["B"]
    assert b["results"]["e8"]["protocol"]["holdouts"] == ["FA", "FB"]
    for name, n in (("FA", 3), ("FB", 2)):
        cell = _cell(b, "holdouts", name, "4")
        assert len(cell["per_seed_eval"]) == 2
        for e in cell["per_seed_eval"]:
            assert e["n_val"] == n and len(e["per_instance"]["energy_gap_rel"]) == n
            amp = e["amplitude"]
            assert len(amp["per_instance"]["c_star"]) == n
            # c* never raises the energy gap
            assert all(c <= g + 1e-12 for c, g in zip(amp["per_instance"]["egap_c"],
                                                       e["per_instance"]["energy_gap_rel"]))
        prov = b["evaluation"]["holdouts"][name]
        assert prov["family"] == ev["holdouts"][name]["family"] and prov["n_instances"] == n
        assert len(prov["manifest_sha256"]) == 64
    assert b["evaluation"]["amplitude"] is True


def test_reuse_from_evaluates_the_source_states_without_training(world):
    tmp, _, _, rep = world
    c = rep["C"]
    e8 = c["results"]["e8"]
    assert e8["protocol"]["eval_only"] is True and c["planned_steps"]["e8"] == 0
    d9 = e8["metrics"]["d9_restart"]
    src = rep["A"]["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    assert {k: v["sha256"] for k, v in d9["ar_states"].items()} == \
        {k: v["sha256"] for k, v in src.items()}
    assert all(v["reused"] for v in d9["ar_states"].values())
    assert d9["reused_from"]["states_sha256"] == {k: v["sha256"] for k, v in src.items()}
    assert c["reuse_from"] == d9["reused_from"]
    assert not list((tmp / "C").glob("e8_states/*.pt"))              # nothing trained or saved
    for s in range(2):
        assert _cell(c, "cells", "ar", "4")["per_seed_eval"][s]["per_instance"] == \
            _cell(rep["B"], "cells", "ar", "4")["per_seed_eval"][s]["per_instance"]
        assert _cell(c, "holdouts", "FA", "4")["per_seed_eval"][s] == \
            _cell(rep["B"], "holdouts", "FA", "4")["per_seed_eval"][s]


def test_reuse_from_refuses_other_states_and_other_configurations(world, tmp_path):
    tmp, data, ev, _ = world
    st = tmp_path / "states"
    shutil.copytree(tmp / "A" / "e8_states", st)
    f = st / "ar_p4_s1.pt"
    f.write_bytes(f.read_bytes() + b"\0")
    reuse = {"report": str(tmp / "A" / "report.json"), "states_dir": str(st)}
    with pytest.raises(ValueError, match="is not the state"):
        run_config(_cfg(tmp_path, "X", data, ev, reuse_from=reuse))
    reuse = {"report": str(tmp / "A" / "report.json"), "states_dir": str(tmp / "A" / "e8_states")}
    with pytest.raises(ValueError, match="differs from"):
        run_config(_cfg(tmp_path, "Y", data, ev, reuse_from=reuse, ar_epochs=3))


def test_checkpointing_override_is_recorded_and_exact(world):
    tmp, _, _, rep = world
    assert rep["D"]["runtime_overrides"] == {"activation_checkpointing": False}
    assert rep["D"]["config"]["model"] == MODEL                       # the config as stamped
    assert _states(tmp, "A") == _states(tmp, "D")


def test_holdouts_are_verified_before_training(world, tmp_path):
    tmp, data, ev, _ = world
    bad = json.loads(json.dumps(ev))
    bad["holdouts"]["FA"]["family"] = "F4"
    with pytest.raises(ValueError, match="holds family 'F5'"):
        run_config(_cfg(tmp_path, "P", data, bad))
    fam = tmp_path / "FA"
    shutil.copytree(ev["holdouts"]["FA"]["dir"], fam)
    (fam / "instance_00001.npz").write_bytes(b"x")
    bad = json.loads(json.dumps(ev))
    bad["holdouts"]["FA"]["dir"] = str(fam)
    with pytest.raises(ValueError, match="differ from their manifest"):
        run_config(_cfg(tmp_path, "Q", data, bad))
    assert not (tmp_path / "Q").exists()                             # nothing ran
    summary = run_config(_cfg(tmp_path, "R", data, bad), dry_run=True)
    assert "differ from their manifest" in summary["w9"]["error"]
    ok = run_config(_cfg(tmp_path, "S", data, ev), dry_run=True,
                    activation_checkpointing=False)
    assert ok["w9"] == {"holdouts": {"FA": 3, "FB": 2}} and ok["activation_checkpointing"] is False


def test_s_arm_trains_and_evaluates_through_the_runner(tmp_path, world):
    _, data, ev, _ = world
    p = _cfg(tmp_path, "S", data, ev)
    cfg = json.loads(p.read_text())
    cfg["model"] = {**MODEL, "decode_scale": "l1",
                    "features": {**MODEL["features"], "load_density": True}}
    p.write_text(json.dumps(cfg))
    r = run_config(p)
    cell = r["results"]["e8"]["metrics"]["holdouts"]["FA"]
    cell = cell["4"] if "4" in cell else cell[4]
    assert np.isfinite(cell["energy_gap_rel"]["mean"])


# ------------------------------------------------- Stage 0b review additions --

def test_w9_evaluator_is_the_frozen_evaluator_plus_amplitude(world):
    """evaluate_model_w9 returns evaluate_model's dict (means, arrays, count)
    from the same predictions, plus the amplitude block."""
    import torch

    from fejepa.data.archive import LazyArchives
    from fejepa.experiments.parallel import _build_model
    from fejepa.experiments.w9_eval import evaluate_model_w9
    from fejepa.metrics import evaluate_model, torch_predictor

    tmp, data, ev, _ = world
    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    m.load_state_dict(torch.load(str(tmp / "B" / "e8_states" / "ar_p4_s0.pt"), weights_only=True))
    files = instance_files(ev["holdouts"]["FA"]["dir"])
    pred = torch_predictor(m, "cpu")
    ref = evaluate_model(pred, LazyArchives(files))
    plain = evaluate_model_w9(pred, LazyArchives(files), amplitude=False)
    amp = evaluate_model_w9(pred, LazyArchives(files), amplitude=True)
    assert plain == ref and {k: v for k, v in amp.items() if k != "amplitude"} == ref


def test_amplitude_readings_are_the_predictions_own(world):
    """The report's per-instance amplitude values equal analysis.w9's reading
    of the reloaded state's prediction (not of the label)."""
    import torch

    from fejepa.analysis.w9 import amplitude_row
    from fejepa.experiments.parallel import _build_model

    tmp, _, ev, rep = world
    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    m.load_state_dict(torch.load(str(tmp / "B" / "e8_states" / "ar_p4_s1.pt"), weights_only=True))
    m.eval()
    a = load_instance(instance_files(ev["holdouts"]["FA"]["dir"])[1])
    with torch.no_grad():
        U = m.forward_instance(m.prepare_instance(a, "cpu")).numpy().astype(np.float64)
    row = amplitude_row(a, U)
    got = _cell(rep["B"], "holdouts", "FA", "4")["per_seed_eval"][1]["amplitude"]["per_instance"]
    assert np.allclose(got["c_star"][1], row["c_star"]) and np.isclose(got["egap_c"][1], row["egap_c"])
    assert not np.allclose(row["c_star"], 1.0)                       # a prediction, not U*


def test_amplitude_summary_by_hand():
    from fejepa.experiments.w9_eval import amplitude_summary

    per = {"c_star": [[1.0, 2.0, np.nan], [10.0]], "c_battery": [1.0, np.nan],
           "disp_c": [0.1, 0.3], "egap_c": [0.2, 0.4]}
    s = amplitude_summary(per)
    assert s["c_star_median"] == 2.0 and np.isclose(s["frac_c_star_gt_1"], 2 / 3)  # not the mean
    assert np.isclose(s["disp_c"], 0.2) and np.isclose(s["egap_c"], 0.3)
    assert s["c_battery_median"] == 1.0


def test_unlabelled_holdouts_and_bad_reuse_sources_are_refused(world, tmp_path):
    from fejepa.experiments.w9_eval import resolve_holdouts, verify_reuse

    tmp, data, ev, _ = world
    src = generate_synthetic_dataset(tmp_path / "u_src", n=2, seed=4, labelled="none")
    out = tmp_path / "U"
    out.mkdir()
    recs = []
    for i, f in enumerate(instance_files(src)):
        save_instance(load_instance(f), out / f"i{i}.npz")
        recs.append({"file": f"i{i}.npz", "labelled": False,
                     "sha256": hashlib.sha256((out / f"i{i}.npz").read_bytes()).hexdigest()})
    write_manifest(out, recs, {"family": "F1"})
    with pytest.raises(ValueError, match="unlabelled"):
        resolve_holdouts({"holdouts": {"U": {"dir": str(out), "family": "F1"}}})
    cfg = json.loads((tmp / "C.json").read_text())
    reuse = dict(cfg["experiments"]["e8"]["reuse_from"])
    st = tmp_path / "st"
    shutil.copytree(reuse["states_dir"], st)
    (st / "ar_p4_s1.pt").unlink()
    with pytest.raises(ValueError, match="not found"):
        verify_reuse(cfg, {**reuse, "states_dir": str(st)})
    two = json.loads((tmp / "A" / "report.json").read_text())
    two["results"]["e8"]["protocol"]["pool_sizes"] = [4, 8]
    (tmp_path / "two.json").write_text(json.dumps(two))
    with pytest.raises(ValueError, match="one pool size"):
        verify_reuse(cfg, {**reuse, "report": str(tmp_path / "two.json")})


def test_configurations_the_wp9_additions_do_not_support_are_refused(world, tmp_path):
    tmp, data, ev, _ = world
    with pytest.raises(SystemExit, match="runner-provided"):
        run_config(_cfg(tmp_path, "K1", data, ev, holdouts={"x": []}))
    with pytest.raises(SystemExit, match="ar_only"):
        run_config(_cfg(tmp_path, "K2", data, ev, ar_only=False, reuse_from={
            "report": str(tmp / "A" / "report.json"), "states_dir": str(tmp / "A" / "e8_states")}))
    p = _cfg(tmp_path, "K3", data, ev)
    c = json.loads(p.read_text())
    c["model"] = {**MODEL, "kind": "bottleneck", "n_tokens": 4, "decode_scale": "l1"}
    p.write_text(json.dumps(c))
    with pytest.raises(SystemExit, match="fejepa' only"):
        run_config(p)


def test_seed_offset_trains_independent_replicates(world, tmp_path):
    tmp, data, _, rep = world
    r = run_config(_cfg(tmp_path, "O", data, seed_offset=3))
    assert r["provenance"]["seeds"] == [3, 4]
    d9 = r["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    assert sorted(d9) == ["s3", "s4"]
    assert sorted(p.name for p in (tmp_path / "O" / "e8_states").glob("ar_*.pt")) == \
        ["ar_p4_s3.pt", "ar_p4_s4.pt"]
    a = rep["A"]["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    assert {v["sha256"] for v in d9.values()}.isdisjoint({v["sha256"] for v in a.values()})
