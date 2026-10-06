"""cmame-paper Stage 2: CM2D's adjudication (PREREG_CM2D Sec. 4-6) and its
pre-run checks, on a miniature laid out like the box -- an E1-like label-free
run and a CM2D-like run that evaluates its states beside the supervised grid --
and, for the command-line path, on a made-up return built around E1's
committed report."""

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.experiments.runner import run_config
from fejepa.fe.synthetic import generate_synthetic_dataset
from fejepa.report import config_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 8, "mgn_depth": 1,
         "features": {"load_summary": True, "geometry": True}}
GRID = {"ar_only": False, "include_anchor": False, "include_ar_ft": False,
        "include_knorm": True, "mgn_budgets": [4]}
STATUS = "run.log start 2026-10-08T20:00:00Z\nrun.log exit=0\n"
E1_RECORD = ROOT / "records" / "wp8" / "e1" / "e1_2d_base" / "report.json"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


adj = _script("adjudicate_cm2d")
pre = _script("cm2d_precheck")
st = _script("stamp_prereg_cm2d")
mk = _script("make_cm2d_config")


def _cfg(tmp, name, data, out=None, **e8):
    cfg = {"data": {"dir": str(data), "n": 12, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "asis"},
           "split": {"n_val": 4, "seed": 1}, "model": MODEL,
           "sup": {"epochs": 1, "lr": 1e-3}, "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {"e8": {"enabled": True, "budgets": [2, 4], "pool_sizes": [4],
                                  "seeds": 2, "ar_epochs": 2, "sup_epochs": 1,
                                  "include_mgn": True, "ar_only": True, **e8}},
           "device": "cpu", "workers": 1, "tf32": False, "prereg_guard": False,
           "out": str(out or tmp / name / "report.json")}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cm2d_adj")
    data = generate_synthetic_dataset(tmp / "corpus", n=12, seed=3, labelled="all")
    run_config(_cfg(tmp, "E1", data))
    reuse = {"report": str(tmp / "E1" / "report.json"),
             "states_dir": str(tmp / "E1" / "e8_states"), "supervised_grid": True}
    cpath = _cfg(tmp, "C", data, reuse_from=reuse, **GRID)
    run_config(cpath)
    fresh = _cfg(tmp, "F", data, out=tmp / "F" / "report.json", reuse_from=reuse, **GRID)
    e1 = json.loads((tmp / "E1" / "report.json").read_text())
    rep = json.loads((tmp / "C" / "report.json").read_text())
    # as the stamped run on the tag would record them
    rep["prereg"] = {"file": "PREREG_CM2D.md", "config_sha256": rep["provenance"]["config_sha256"]}
    rep["provenance"]["git"] = "prereg-cm2d"
    e1_sha = hashlib.sha256((tmp / "E1" / "report.json").read_bytes()).hexdigest()
    return {"tmp": tmp, "data": data, "e1": e1, "rep": rep, "e1_sha": e1_sha,
            "cfg": json.loads(cpath.read_text()), "cpath": cpath, "fresh": fresh,
            "e1_path": tmp / "E1" / "report.json",
            "entries": {"cm2d_v1": rep["provenance"]["config_sha256"]}}


def _adj(w, rep=None, **kw):
    args = dict(rep=rep if rep is not None else w["rep"], e1=w["e1"], expected_cfg=w["cfg"],
                prereg_entries=w["entries"], expected_git="prereg-cm2d",
                e1_report_sha256=w["e1_sha"], status_text=STATUS, run_logs=1)
    args.update(kw)
    return adj.adjudicate_cm2d(**args)


def _cells(rep):
    return rep["results"]["e8"]["metrics"]["cells"]


def _set(rep, row, b, metric, vals):
    for e, v in zip(_cells(rep)[row][str(b)]["per_seed_eval"], vals, strict=True):
        e[metric] = v


# ------------------------------------------------------------ verdicts --
def test_the_miniature_is_adjudicated(world):
    res = _adj(world)
    assert res["reuse"]["ok"] and res["reuse"]["reproduction_max_rel_dev"] == 0.0
    assert res["reuse"]["other_metrics_max_rel_dev"] == 0.0
    for h, metric in (("H1", "energy_gap_rel"), ("H2a", "energy_gap_rel"),
                      ("H2b", "vm_rel_l2")):
        r = res[h]
        assert r["verdict"] in ("SUPPORTED", "NOT SUPPORTED") and r["metric"] == metric
        assert set(r) >= {"rel_change", "se_rel", "threshold", "base_per_seed",
                          "new_per_seed", "robustness"}
        assert r["threshold"] >= 0.10 and len(r["new_per_seed"]) == 2
        assert r["robustness"]["medians_show_the_direction"] in (True, False)
    assert res["H3"]["reading"] in ("the stiffness-norm transformer lower beyond the guard",
                                    "the label-free transformer lower beyond the guard",
                                    "no difference shown (within the guard)")
    sw = res["H3"]["roles_exchanged"]                      # beside H3, no criterion
    assert sw["base_per_seed"] == res["H3"]["new_per_seed"]
    assert math.isclose(sw["rel_change"], res["H3"]["base_mean"] / res["H3"]["new_mean"] - 1)
    sec = res["secondary"]
    assert set(sec["comparisons"]) == set(adj.METRICS)
    assert set(sec["comparisons"]["energy_gap_rel"]) == {"2", "4"}
    assert set(sec["comparisons"]["energy_gap_rel"]["4"]) == {
        "labels_vs_label_free", "labels_knorm_vs_label_free", "mgn_vs_label_free",
        "labels_knorm_vs_labels", "mgn_vs_labels"}
    assert "mgn_vs_labels" not in sec["comparisons"]["energy_gap_rel"]["2"]
    n = sum(len(v) for b in sec["comparisons"].values() for v in b.values())
    assert n == len(adj.METRICS) * (3 * 2 + 2 * 1)            # 16 per metric on the box
    for pair in sec["pairing"].values():
        for m in pair.values():
            assert m["pairs"] == 2 * 4 and 0 <= m["first_lower"] + m["ties"] <= m["pairs"]
    assert set(sec["worse_than_zero"]) == {"ar", "labels", "labels_knorm", "mgn"}
    assert sec["worse_than_zero"]["labels"]["2"]["of"] == 8
    gn = sec["graph_network"]
    assert gn["budget"] == 4 and set(gn["energy_gap_seed_means"]) == {"ar", "labels", "mgn"}
    assert gn["mgn_gap_above_label_free"] == \
        (gn["energy_gap_seed_means"]["mgn"] > gn["energy_gap_seed_means"]["ar"])
    assert set(sec["label_efficiency"]) == {"labels", "labels_knorm"}
    assert sec["july_labels_only"] is None
    assert set(sec["e8_builtin"]["divergence_flags"]) == {"ar", "labels", "labels_knorm", "mgn"}
    # the only deviation of the miniature: its single worker
    assert res["deviations"] == ["workers 1 (PREREG_CM2D Sec. 6: 3)"]
    json.dumps(adj.jsonable(res), allow_nan=False)          # strict JSON


def test_the_verdicts_follow_the_guard(world):
    rep = copy.deepcopy(world["rep"])
    _set(rep, "ar", 4, "energy_gap_rel", [0.10, 0.11])
    _set(rep, "labels", 4, "energy_gap_rel", [0.40, 0.42])
    _set(rep, "labels_knorm", 4, "energy_gap_rel", [0.20, 0.21])
    _set(rep, "labels", 4, "vm_rel_l2", [0.30, 0.31])
    _set(rep, "labels_knorm", 4, "vm_rel_l2", [0.29, 0.30])
    res = _adj(world, rep)
    assert res["H1"]["verdict"] == "SUPPORTED" and res["H2a"]["verdict"] == "SUPPORTED"
    assert math.isclose(res["H1"]["rel_change"], 0.105 / 0.41 - 1)
    assert res["H2b"]["verdict"] == "NOT SUPPORTED" and "note" not in res["H2b"]
    assert res["H3"]["reading"] == "the label-free transformer lower beyond the guard"
    _set(rep, "labels_knorm", 4, "vm_rel_l2", [0.20, 0.21])
    assert _adj(world, rep)["H2b"]["verdict"] == "SUPPORTED"
    _set(rep, "labels_knorm", 4, "energy_gap_rel", [0.06, 0.07])
    assert _adj(world, rep)["H3"]["reading"] == \
        "the stiffness-norm transformer lower beyond the guard"
    _set(rep, "labels_knorm", 4, "energy_gap_rel", [0.104, 0.106])
    assert _adj(world, rep)["H3"]["reading"] == "no difference shown (within the guard)"
    # a reduction inside the 10% floor is not shown; a higher value is "worse"
    _set(rep, "ar", 4, "energy_gap_rel", [0.38, 0.39])
    r = _adj(world, rep)["H1"]
    assert r["verdict"] == "NOT SUPPORTED" and "note" not in r
    _set(rep, "ar", 4, "energy_gap_rel", [0.60, 0.61])
    r = _adj(world, rep)["H1"]
    assert r["verdict"] == "NOT SUPPORTED" and r["note"] == "worse beyond the guard"
    # a large finite blow-up of one reference seed: NOT SUPPORTED, not NOT EVALUATED
    _set(rep, "ar", 4, "energy_gap_rel", [0.01, 0.01])
    _set(rep, "labels", 4, "energy_gap_rel", [0.40, 40.0])
    assert _adj(world, rep)["H1"]["verdict"] == "NOT SUPPORTED"


def test_divergence_and_a_non_finite_reference(world):
    rep = copy.deepcopy(world["rep"])
    _set(rep, "labels_knorm", 4, "energy_gap_rel", [0.2, float("nan")])
    res = _adj(world, rep)
    assert res["H2a"]["verdict"] == "NOT SUPPORTED (diverged)"
    assert res["H2b"]["verdict"] in ("SUPPORTED", "NOT SUPPORTED")
    assert res["H3"]["reading"] == "the stiffness-norm transformer diverged"
    _set(rep, "labels", 4, "energy_gap_rel", [float("inf"), 0.4])
    res = _adj(world, rep)                       # the reference first, even if A diverged
    assert res["H1"]["verdict"] == "NOT EVALUATED" and res["H2a"]["verdict"] == "NOT EVALUATED"
    json.dumps(adj.jsonable(res), allow_nan=False)


def test_a_label_free_row_that_fails_its_reuse_checks_voids_h1_and_h3_only(world):
    def check(rep, what):
        res = _adj(world, rep)
        assert not res["reuse"]["ok"] and what in " ".join(res["reuse"]["reasons"])
        assert res["H1"]["verdict"] == "NOT EVALUATED" and res["H3"]["reading"] == "NOT EVALUATED"
        assert res["H2a"]["verdict"] in ("SUPPORTED", "NOT SUPPORTED")
        assert res["H2b"]["verdict"] in ("SUPPORTED", "NOT SUPPORTED")

    def mutate(f):
        rep = copy.deepcopy(world["rep"])
        f(rep)
        return rep

    ar = lambda r, s: _cells(r)["ar"]["4"]["per_seed_eval"][s]["per_instance"]  # noqa: E731
    d9 = lambda r: r["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]    # noqa: E731
    check(mutate(lambda r: ar(r, 1)["energy_gap_rel"].__setitem__(
        2, ar(r, 1)["energy_gap_rel"][2] * (1 + 2e-4))), "does not reproduce")
    check(mutate(lambda r: ar(r, 0)["disp_rel_l2"].__setitem__(0, float("nan"))),
          "does not reproduce")
    check(mutate(lambda r: d9(r)["s1"].__setitem__("sha256", "0" * 64)),
          "did not evaluate E1's states")
    check(mutate(lambda r: d9(r)["s0"].__setitem__("reused", False)),
          "did not evaluate E1's states")
    check(mutate(lambda r: r["results"]["e8"]["protocol"].pop("eval_only")), "evaluation only")
    check(mutate(lambda r: r["reuse_from"].__setitem__("report_sha256", "1" * 64)),
          "another report")
    check(mutate(lambda r: r["reuse_from"].pop("supervised_grid")), "supervised grid")
    # the other three metrics are reported, not required
    rep = mutate(lambda r: ar(r, 0)["crit_recall"].__setitem__(0, ar(r, 0)["crit_recall"][0]
                                                               + 0.25))
    res = _adj(world, rep)
    assert res["reuse"]["ok"] and res["reuse"]["other_metrics_max_rel_dev"] > 1e-3


def test_refusals(world):
    def refused(match, rep=None, **kw):
        with pytest.raises(ValueError, match=match):
            _adj(world, rep, **kw)

    refused("ran on", expected_git="prereg-other")
    refused("CONFIG_SHA256", prereg_entries={"cm2d_v1": "0" * 64})
    other = copy.deepcopy(world["cfg"])
    other["experiments"]["e8"]["sup_epochs"] = 2
    refused("not the generator's", expected_cfg=other)
    for path, value, match in ((("prereg",), None, "stamped"),
                               (("prereg", "file"), "PREREG_W9.md", "PREREG_CM2D.md"),
                               (("provenance", "seeds"), [0, 2], "seeds")):
        rep = copy.deepcopy(world["rep"])
        d = rep
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        refused(match, rep)
    rep = copy.deepcopy(world["rep"])
    rep["provenance"]["datasets"][0]["manifest_sha256"] = "2" * 64
    refused("another corpus", rep)
    rep = copy.deepcopy(world["rep"])
    del _cells(rep)["mgn"]["4"]
    refused("mgn row has no cell at 4", rep)
    rep = copy.deepcopy(world["rep"])
    _cells(rep)["labels_knorm"]["2"]["per_seed_eval"].pop()
    refused("holds 1 evaluations, not 2", rep)
    rep = copy.deepcopy(world["rep"])
    ev = _cells(rep)["labels"]["4"]["per_seed_eval"][0]
    ev["per_instance"]["vm_rel_l2"] = ev["per_instance"]["vm_rel_l2"][:-1]
    refused("lacks values", rep)
    rep = copy.deepcopy(world["rep"])
    del _cells(rep)["labels_knorm"]["4"]["per_seed_eval"][1]["vm_rel_l2"]
    refused("lacks a seed value", rep)


def test_deviations_are_recorded(world):
    rep = copy.deepcopy(world["rep"])
    rep["d9_reuse_states"] = True
    rep["results"]["e8"]["metrics"]["d9_restart"]["sup_units_from_cache"] = ["labels 2 0"]
    rep["results"]["e8"]["metrics"]["d9_restart"]["units_resumed_from_epoch"] = {"labels 4 1": 3}
    rep["runtime_overrides"] = {"activation_checkpointing": False}
    rep["solve_ledger"] = {"total": 8}
    rep["provenance"]["versions"]["torch"] = "0.0"
    rep["runtime_policy"]["gpu"] = "another GPU"
    status = "run.log start a\nrun.log exit=1\nrun.log start b\nrun.log exit=0\n"
    dev = " | ".join(_adj(world, rep, status_text=status, run_logs=2)["deviations"])
    for s in ("restarted", "unit cache", "resumed", "checkpointing off", "ledger reads 8",
              "torch '0.0'", "GPU 'another GPU'", "2 started, exit codes [1, 0]",
              "2 run logs"):
        assert s in dev, s
    # an attempt killed before its exit line, and an empty status file
    dev = " | ".join(_adj(world, status_text="run.log start a\nrun.log start b\n"
                                             "run.log exit=0\n")["deviations"])
    assert "2 started, exit codes [0]" in dev
    assert "0 started, exit codes []" in " | ".join(_adj(world, status_text="")["deviations"])
    assert adj.status_attempts("x\nrun.log exit=0\nrun.log exit=-9\n") == \
        {"starts": 0, "exits": [0, -9]}


def test_secondary_readings_never_block_a_verdict(world):
    res = _adj(world, july_rep={"results": {}})
    assert "not_evaluated" in res["secondary"]["july_labels_only"]
    assert res["H1"]["verdict"] in ("SUPPORTED", "NOT SUPPORTED")
    rep = copy.deepcopy(world["rep"])
    _set(rep, "labels", 2, "crit_recall", [0.5, 0.52])
    _set(rep, "labels_knorm", 2, "crit_recall", [0.8, 0.81])
    c = _adj(world, rep)["secondary"]["comparisons"]["crit_recall"]["2"]["labels_knorm_vs_labels"]
    assert c["better_direction"] == "higher" and c["worse"] is True
    assert c["new_better_beyond_guard"] is True and c["new_worse_beyond_guard"] is False


# ---------------------------------------------------- the command line --
def _made_up_return(tmp, prereg_path, git="prereg-cm2d-test-only", mutate=None):
    """A CM2D return around E1's committed report: E1's own label-free cell
    (so it reproduces) beside made-up supervised and naive cells."""
    e1 = json.loads(E1_RECORD.read_text())
    cfg = mk.cm2d_config(e1["config"])
    h = config_sha256(cfg)
    rng = np.random.default_rng(0)

    def ev(level):
        per = {m: list(level * rng.uniform(0.5, 1.5, 256)) for m in adj.METRICS}
        return {**{m: float(np.mean(v)) for m, v in per.items()}, "per_instance": per,
                "n_val": 256}

    cells = {"ar": {"1024": e1["results"]["e8"]["metrics"]["cells"]["ar"]["1024"]}}
    for r, lev in (("labels", 0.3), ("labels_knorm", 0.1)):
        cells[r] = {str(b): {"per_seed_eval": [ev(lev) for _ in range(3)]}
                    for b in (16, 64, 256, 1024)}
    cells["mgn"] = {str(b): {"per_seed_eval": [ev(0.5) for _ in range(3)]} for b in (64, 1024)}
    for r in ("zero", "scale_aware_poly", "knn_field"):
        cells[r] = {str(b): {"per_seed_eval": [ev(1.0)]} for b in (16, 64, 256, 1024)}
    d9 = copy.deepcopy(e1["results"]["e8"]["metrics"]["d9_restart"])
    for v in d9["ar_states"].values():
        v["reused"] = True
    d9.update(sup_units_from_cache=[], units_resumed_from_epoch={})
    rep = {"config": cfg, "prereg": {"file": "PREREG_CM2D.md", "config_sha256": h},
           "provenance": {**e1["provenance"], "git": git, "config_sha256": h},
           "runtime_policy": e1["runtime_policy"], "solve_ledger": {"total": 0},
           "d9_reuse_states": False, "gate_g1_prime": {"passed": False},
           "reuse_from": {"report_sha256": hashlib.sha256(E1_RECORD.read_bytes()).hexdigest(),
                          "supervised_grid": True},
           "results": {"e8": {"protocol": {"workers": 3, "eval_only": True},
                              "metrics": {"cells": cells, "d9_restart": d9}, "kills": []}}}
    if mutate:
        mutate(rep)
    ret = tmp / "ret"
    ret.mkdir(exist_ok=True)
    (ret / "report.json").write_text(json.dumps(rep))
    sha = hashlib.sha256((ret / "report.json").read_bytes()).hexdigest()
    (ret / "status.txt").write_text(STATUS)
    (ret / "run.log").write_text("log\n")
    (ret / "provenance.txt").write_text(f"HEAD {'a' * 40}\ntree {'b' * 40}\n--- report ---\n"
                                        f"{sha}  runs/cm2d/report.json\n")
    return ret


def _stamped_prereg(tmp):
    p = tmp / "PREREG_CM2D.md"
    p.write_text((ROOT / "PREREG_CM2D.md").read_text(encoding="utf-8"), encoding="utf-8")
    if "<fill before tagging>" in p.read_text(encoding="utf-8"):
        st.stamp(p, ROOT / "configs" / "cm2d_v1.json", "8 October 2026")
    return p


def _main(tmp, ret, prereg, *extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "adjudicate_cm2d.py"),
                           "--return", str(ret), "--prereg", str(prereg), "--out",
                           str(tmp / "verdict.json"), "--expected-git", "prereg-cm2d-test-only",
                           "--code-ref", "prereg-cm2d-test-only", *extra],
                          capture_output=True, text=True, cwd=ROOT, timeout=600)


def test_main_adjudicates_a_return(tmp_path):
    prereg = _stamped_prereg(tmp_path)
    ret = _made_up_return(tmp_path, prereg)
    res = _main(tmp_path, ret, prereg)
    assert res.returncode == 0, res.stderr[-3000:]
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert v["reuse"]["ok"] and v["reuse"]["reproduction_max_rel_dev"] == 0.0
    assert v["H1"]["verdict"] == "SUPPORTED"                  # E1's 0.038 against 0.3
    assert v["H2a"]["verdict"] == "SUPPORTED" and v["H2b"]["verdict"] == "SUPPORTED"
    assert set(v["inputs"]) >= {"report", "status.txt", "provenance.txt", "run_logs",
                                "recorded_commit", "e1_report", "prereg", "july_report"}
    assert set(v["secondary"]["july_labels_only"]) == {"16", "64", "256", "1024"}
    dev = " | ".join(v["deviations"])
    assert "not compared with prereg-cm2d-test-only (no such tag" in dev
    assert "adjudicating code not compared" in dev
    assert "workers" not in dev and "attempts" not in dev
    # states reused through another report than E1's records copy: H1 and H3 void
    ret = _made_up_return(tmp_path, prereg, mutate=lambda rep: rep["reuse_from"].__setitem__(
        "report_sha256", "d" * 64))
    res = _main(tmp_path, ret, prereg)
    assert res.returncode == 0, res.stderr[-3000:]
    v = json.loads((tmp_path / "verdict.json").read_text())
    assert "another report" in " ".join(v["reuse"]["reasons"])
    assert v["H1"]["verdict"] == "NOT EVALUATED" and v["H3"]["reading"] == "NOT EVALUATED"
    assert v["H2a"]["verdict"] == "SUPPORTED"


def test_main_refusals(tmp_path):
    prereg = _stamped_prereg(tmp_path)

    def fails(match, mutate=None, prereg_path=prereg, *extra):
        ret = _made_up_return(tmp_path, prereg, mutate=mutate)
        res = _main(tmp_path, ret, prereg_path, *extra)
        assert res.returncode != 0 and match in res.stderr, res.stderr[-2000:]

    def wrong_hash(rep):
        rep["prereg"]["config_sha256"] = rep["provenance"]["config_sha256"] = "c" * 64

    fails("CONFIG_SHA256", wrong_hash)
    fails("ran on", lambda rep: rep["provenance"].__setitem__("git", "prereg-cm2d-dirty"))
    draft = tmp_path / "draft.md"
    draft.write_text("CONFIG_SHA256[cm2d_v1] = <fill before tagging>\n")
    fails("not stamped", None, draft)
    # the report the box hashed, and E1's committed report
    ret = _made_up_return(tmp_path, prereg)
    (ret / "report.json").write_text((ret / "report.json").read_text() + " ")
    res = _main(tmp_path, ret, prereg)
    assert res.returncode != 0 and "not the report the box hashed" in res.stderr
    e1 = json.loads(E1_RECORD.read_text())
    e1["provenance"]["git"] = "prereg-e1-1-gabcdef"
    bad = tmp_path / "e1.json"
    bad.write_text(json.dumps(e1))
    ret = _made_up_return(tmp_path, prereg)
    res = _main(tmp_path, ret, prereg, "--e1-report", str(bad))
    assert res.returncode != 0 and "not 'prereg-e1'" in res.stderr


def test_the_recorded_commit_is_the_tags(monkeypatch):
    prov = f"HEAD {'a' * 40}\ntree {'b' * 40}\n"
    monkeypatch.setattr(adj, "_git", lambda *a: None)
    assert "no such tag" in adj.check_commit(prov)
    answers = {"prereg-cm2d^{commit}": "a" * 40, "prereg-cm2d^{tree}": "b" * 40}
    monkeypatch.setattr(adj, "_git", lambda *a: answers[a[-1]])
    assert adj.check_commit(prov) is None
    answers["prereg-cm2d^{tree}"] = "c" * 40
    with pytest.raises(SystemExit, match="records HEAD"):
        adj.check_commit(prov)
    assert adj.provenance_commit("tree x\n") == {"HEAD": None, "tree": None}


def test_main_refuses_a_report_that_is_not_e1s(world, tmp_path):
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "adjudicate_cm2d.py"),
                          "--return", str(tmp_path), "--e1-report", str(world["e1_path"]),
                          "--out", str(tmp_path / "v.json")],
                         capture_output=True, text=True, cwd=ROOT)
    assert res.returncode != 0 and "not the stamped e1_2d_base run" in res.stderr


def test_the_return_and_prereg_inputs(world, tmp_path):
    rep_path = tmp_path / "report.json"
    rep_path.write_text(json.dumps(world["rep"]))
    sha = hashlib.sha256(rep_path.read_bytes()).hexdigest()
    (tmp_path / "status.txt").write_text(STATUS)
    (tmp_path / "provenance.txt").write_text(f"--- report ---\n{sha}  runs/cm2d/report.json\n")
    (tmp_path / "run.log").write_text("x")
    (tmp_path / "run.log.20261008-120000").write_text("y")
    rep, got, status, prov, logs = adj.load_return(tmp_path)
    assert got == sha and rep == world["rep"] and status == STATUS
    assert logs == ["run.log", "run.log.20261008-120000"]
    (tmp_path / "provenance.txt").write_text(f"{'3' * 64}  runs/cm2d/report.json\n")
    with pytest.raises(SystemExit, match="not the report the box hashed"):
        adj.load_return(tmp_path)
    (tmp_path / "status.txt").unlink()
    with pytest.raises(SystemExit, match="lacks"):
        adj.load_return(tmp_path)
    p = tmp_path / "PREREG_CM2D.md"
    p.write_text((ROOT / "PREREG_CM2D.md").read_text(encoding="utf-8"), encoding="utf-8")
    if "<fill before tagging>" in p.read_text(encoding="utf-8"):
        with pytest.raises(SystemExit, match="not stamped"):
            adj.prereg_hash(p)
    p = _stamped_prereg(tmp_path)
    cfg = json.loads((ROOT / "configs" / "cm2d_v1.json").read_text())
    assert adj.prereg_hash(p)["cm2d_v1"] == config_sha256(cfg)
    p.write_text(p.read_text(encoding="utf-8").replace("tau", "t"), encoding="utf-8")
    with pytest.raises(SystemExit, match="own SHA-256"):
        adj.prereg_hash(p)


# --------------------------------------------------------------- precheck --
def _pre(world, cfg=None, e1=None, **kw):
    return pre.precheck(cfg or world["fresh"], e1 or world["e1_path"], git=False, stack=False,
                        generator=False, **kw)


def test_the_precheck_passes_a_fresh_miniature_before_the_stamp(world):
    res = _pre(world, pre_stamp=True)
    assert res["go"], res
    c = res["checks"]
    assert c["stamp"]["ok"] is None and "8 of 8 labelled" in c["labels"]["detail"]
    assert c["reproduction"]["ok"] and "deviation 0" in c["reproduction"]["detail"]
    assert c["smoke"]["ok"] and c["fresh"]["ok"]
    # after the stamp the guard must verify: a run with the guard off is refused
    res = _pre(world)
    assert not res["go"] and res["stop"] == ["stamp"]


def test_the_precheck_refuses_an_earlier_attempt_and_a_finished_run(world):
    res = _pre(world, cfg=world["cpath"], pre_stamp=True)
    assert "fresh" in res["stop"] and "B3" in res["checks"]["fresh"]["detail"]
    res = _pre(world, cfg=world["cpath"], pre_stamp=True, restart=True)
    assert "fresh" in res["stop"] and "do not restart" in res["checks"]["fresh"]["detail"]
    out = Path(json.loads(world["fresh"].read_text())["out"]).parent
    out.mkdir(parents=True, exist_ok=True)
    for trace in ("e8_states", "run.log", "status.txt"):          # an attempt's traces
        (out / trace).mkdir() if trace == "e8_states" else (out / trace).write_text("x")
        try:
            assert _pre(world, pre_stamp=True, restart=True)["go"]   # a restart is fine
            res = _pre(world, pre_stamp=True)
            assert "fresh" in res["stop"] and trace in res["checks"]["fresh"]["detail"]
        finally:
            (out / trace).rmdir() if trace == "e8_states" else (out / trace).unlink()
    assert _pre(world, pre_stamp=True)["go"]


def test_the_precheck_stops_on_another_corpus_missing_labels_or_states(world, tmp_path):
    data = generate_synthetic_dataset(tmp_path / "corpus", n=12, seed=3, labelled="none")
    cfg = json.loads(world["fresh"].read_text())
    cfg["data"]["dir"] = str(data)
    p = tmp_path / "C.json"
    p.write_text(json.dumps(cfg))
    res = _pre(world, cfg=p, pre_stamp=True)
    assert not res["go"] and {"corpus", "labels", "reproduction", "smoke"} <= set(res["stop"])
    assert "0 of 8 labelled" in res["checks"]["labels"]["detail"]
    cfg = json.loads(world["fresh"].read_text())
    cfg["experiments"]["e8"]["reuse_from"]["states_dir"] = str(tmp_path / "nowhere")
    p.write_text(json.dumps(cfg))
    assert {"e1_states", "reproduction"} <= set(_pre(world, cfg=p, pre_stamp=True)["stop"])


def test_the_precheck_stops_when_e1s_values_are_not_reproduced(world, tmp_path):
    e1 = copy.deepcopy(world["e1"])
    ev = e1["results"]["e8"]["metrics"]["cells"]["ar"]["4"]["per_seed_eval"][1]
    ev["per_instance"]["disp_rel_l2"][3] *= 1 + 5e-4
    bad = tmp_path / "E1.json"
    bad.write_text(json.dumps(e1))
    res = _pre(world, e1=bad, pre_stamp=True)
    assert res["stop"] == ["reproduction"], res["stop"]


def test_the_precheck_git_checks(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                        *a], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "f.txt").write_text("1")
    git("add", "f.txt")
    git("commit", "-q", "-m", "one")
    c = pre.git_checks(False, root=tmp_path)
    assert c["tag"]["ok"] is False and c["report_git"]["ok"] is False and c["clean"]["ok"]
    assert pre.git_checks(True, root=tmp_path)["tag"]["ok"] is None
    git("tag", "-a", "prereg-cm2d", "-m", "stamp")
    c = pre.git_checks(False, root=tmp_path)
    assert c["tag"]["ok"] and c["report_git"]["ok"] and c["clean"]["ok"]
    (tmp_path / "f.txt").write_text("2")
    c = pre.git_checks(False, root=tmp_path)
    assert c["clean"]["ok"] is False and c["report_git"]["ok"] is False      # "-dirty"
    git("commit", "-q", "-am", "two")
    c = pre.git_checks(False, root=tmp_path)
    assert c["tag"]["ok"] is False and c["tag"]["detail"].startswith("prereg-cm2d-1-g")


def test_the_precheck_stack_checks(monkeypatch, tmp_path):
    e1 = tmp_path / "e1.json"
    e1.write_text(json.dumps({"provenance": {"versions": {"torch": torch.__version__}}}))

    here = str(ROOT / "src" / "fejepa" / "__init__.py")

    def fake(gpu, pgrep, imp=(0, here, "")):
        def run(args, cwd=None, timeout=60, env=None):
            if args[0] == "nvidia-smi":
                return gpu
            if args[0] == "pgrep":
                assert args[2] == pre.OTHER_RUNS
                return pgrep
            assert args[1:3] == ["-c", "import fejepa; print(fejepa.__file__)"]
            assert env is not None and "PYTHONPATH" not in env
            return imp
        return run

    monkeypatch.setattr(pre, "_run", fake((0, "12, NVIDIA GeForce RTX 5090", ""), (1, "", "")))
    c = pre.stack_checks(e1)
    assert c["gpu_idle"]["ok"] and c["no_other_run"]["ok"] and c["imports"]["ok"]
    monkeypatch.setattr(pre, "_run", fake((0, "12, X", ""), (1, "", ""),
                                          (0, "/usr/lib/python3/site-packages/fejepa/__init__.py",
                                           "")))
    assert pre.stack_checks(e1)["imports"]["ok"] is False      # another copy of the code
    monkeypatch.setattr(pre, "_run", fake((0, "12, X", ""), (0, "4321 python scripts/"
                                                             "export_fields.py --out x", "")))
    assert pre.stack_checks(e1)["no_other_run"]["ok"] is False  # a Sec. A script still running
    monkeypatch.setattr(pre, "_run", fake((0, "4000, NVIDIA GeForce RTX 5090", ""),
                                          (0, "123 python -m fejepa.cli run-config x", "")))
    c = pre.stack_checks(e1)
    assert c["gpu_idle"]["ok"] is False and c["no_other_run"]["ok"] is False
    monkeypatch.setattr(pre, "_run", fake((None, "", "FileNotFoundError"), (None, "", "no pgrep"),
                                          (1, "", "ModuleNotFoundError")))
    c = pre.stack_checks(e1)                                   # all fail closed
    assert c["gpu_idle"]["ok"] is False and c["no_other_run"]["ok"] is False
    assert c["imports"]["ok"] is False
    assert c["torch"]["ok"] is torch.cuda.is_available()
    assert "need 7" in pre.stack_checks(e1)["disk"]["detail"]
    assert "need 5" in pre.stack_checks(e1, pre.FREE_GB_RESTART)["disk"]["detail"]


def test_the_precheck_leaves_the_gpu_alone_when_it_is_busy(world, monkeypatch):
    def run(args, cwd=None, timeout=60, env=None):
        if args[0] == "nvidia-smi":
            return 0, "9000, NVIDIA GeForce RTX 5090", ""
        if args[0] == "pgrep":
            return 1, "", ""
        return 0, str(ROOT / "src" / "fejepa" / "__init__.py"), ""

    monkeypatch.setattr(pre, "_run", run)
    called = []
    monkeypatch.setattr(pre, "reproduction_check", lambda *a: called.append("r"))
    monkeypatch.setattr(pre, "smoke_check", lambda *a: called.append("s"))
    res = pre.precheck(world["fresh"], world["e1_path"], pre_stamp=True, git=False, stack=True,
                       generator=False)
    assert not called and not res["go"]
    for k in ("reproduction", "smoke"):
        assert res["checks"][k] == {"ok": False, "detail": "not run (the GPU is busy)"}


def test_a_failing_reproduction_is_a_fail_line_not_a_crash(world, monkeypatch):
    def boom(*a):
        raise TypeError("an unreadable instance")

    monkeypatch.setattr(pre, "reproduction_check", boom)
    res = _pre(world, pre_stamp=True)
    assert res["stop"] == ["reproduction"]
    assert "TypeError: an unreadable instance" in res["checks"]["reproduction"]["detail"]
