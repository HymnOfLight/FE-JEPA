"""fine3d (PREREG_FINE3D Sec. 6-8): the adjudication on synthetic returns built
from the records -- E2's M = 1,024 report as both R2a arms (S's cells scaled),
the Phase-2b labelled fine-tuning at 16 as both R3 arms' networks (their cells
scaled). The verdicts follow Sec. 6 condition by condition, R3's read the
labelled arm of the session (the record is a reading), the R2b rule is the
one fixed there, every refusal of Sec. 8 fires, deviations are recorded, and a
refused part leaves the other's verdicts standing."""

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fejepa.analysis import adjudicate_fine3d as A
from fejepa.report import config_sha256

ROOT = Path(__file__).resolve().parents[1]
E2 = ROOT / "records" / "wp8" / "e2" / "e2_m1024" / "report.json"
P2B = ROOT / "records" / "wp8" / "e2" / "baseline" / "report_phase2b.json"
S_FACTOR = 2.0 ** -10


def _load(p):
    return json.loads(Path(p).read_text())


@pytest.fixture(scope="module")
def recs():
    return _load(E2), _load(P2B)


def _cfg(name):
    return _load(ROOT / "configs" / f"{name}.json")


def _scale_cell(cell, metric, k):
    """Scale one metric of a cell: per-instance values, per-seed means and the
    cell's mean/std/per_seed summary, consistently."""
    for e in cell["per_seed_eval"]:
        e["per_instance"][metric] = [v * k for v in e["per_instance"][metric]]
        e[metric] = float(np.mean(e["per_instance"][metric]))
    if metric in cell:
        ps = [e[metric] for e in cell["per_seed_eval"]]
        cell[metric] = {"mean": float(np.mean(ps)), "std": float(np.std(ps)), "per_seed": ps}


def _restamp(r):
    sha = config_sha256(r["config"])
    r["provenance"]["config_sha256"] = sha
    r["prereg"] = {"file": "PREREG_FINE3D.md", "config_sha256": sha}
    return r


def r2a(e2, name, fine=1.0, egap=1.0, disp=1.0, inband_p3=None, git="prereg-fine3d",
        seeds=(3, 4, 5)):
    """`disp` scales E8's in-band displacement and, unless `inband_p3` is
    given, P3's in-band pass alike (the same states on the same split)."""
    r = copy.deepcopy(e2)
    r["config"] = _cfg(name)
    _restamp(r)
    r["provenance"].update(git=git, seeds=list(seeds))
    r["results"]["p3_transfer"]["protocol"]["seeds"] = list(seeds)
    p3 = r["results"]["p3_transfer"]["metrics"]["ar"]
    _scale_cell(p3["fine"], "disp_rel_l2", fine)
    _scale_cell(p3["inband"], "disp_rel_l2", disp if inband_p3 is None else inband_p3)
    c = r["results"]["e8"]["metrics"]["cells"]["ar"]["1024"]
    _scale_cell(c, "energy_gap_rel", egap)
    _scale_cell(c, "disp_rel_l2", disp)
    return r


def _amp(fine_cstar, inband_cstar=(1.0, 1.0, 1.0), complete=True, report_sha=None):
    def summ(c):
        return {"c_star_median": c, "c_battery_median": c, "amp_ratio_median": c,
                "disp": 0.5, "disp_c": 0.1, "egap": 0.3, "egap_c": 0.05, "complete": complete}
    rem = {"c_battery_median_by_lc": {"0.0906": 1.0, "0.0374": 1.5},
           "u_norm_K_ratio_median_by_lc": {"0.0906": 1.0, "0.0374": 0.5},
           "fscale_ratio_median_by_lc": {"0.0906": 1.0, "0.0374": 0.19}, "n_geometries": 16}
    return {"report_sha256": report_sha,
            "seeds": {f"s{s}": {"inband": {"summary": summ(ci), "reproduction": {}},
                                "fine": {"summary": summ(c), "reproduction": {}},
                                "remesh": rem}
                      for s, c, ci in zip((3, 4, 5), fine_cstar, inband_cstar)}}


def _order(s):
    """A seed's instance orders (the same in both arms)."""
    return hashlib.sha256(f"the orders of seed {s}".encode()).hexdigest()


def r3(p2b, egap=1.0, vm=1.0, git="prereg-fine3d", arm="energy"):
    """An R3 arm's report: the recorded labelled fine-tuning at 16 as its fine
    cells (energy gap and von Mises error scaled), E8's AR cell as its in-band
    ones."""
    cfg = _cfg(A.R3_LABELS[arm])
    sha = config_sha256(cfg)
    ft = copy.deepcopy(p2b["results"]["p3_transfer"]["metrics"]["fewshot"]["16"]["finetune"])
    _scale_cell(ft, "energy_gap_rel", egap)
    _scale_cell(ft, "vm_rel_l2", vm)
    inb = p2b["results"]["e8"]["metrics"]["cells"]["ar"]["1024"]
    rec = p2b["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    per = {}
    for s in (0, 1, 2):
        def ev(e):
            e = copy.deepcopy(e)
            n = len(e["per_instance"]["disp_rel_l2"])
            e["amplitude"] = {"per_instance": {"c_star": [[1.0] * 4] * n},
                              "summary": {"c_star_median": 1.0, "c_battery_median": 1.0,
                                          "disp_c": 0.04, "egap_c": 0.03,
                                          "frac_c_star_gt_1": 0.5}}
            return e
        per[f"s{s}"] = {"seed": s, "git": git, "config_sha256": sha,
                        "machine": {"start": {"host": "box", "gpu_driver": "575.64"},
                                    "end": {"host": "box", "gpu_driver": "575.64"}},
                        "source_state": {"sha256": rec[f"s{s}"]["sha256"]},
                        "content_check": {"passed": True, "max_rel_dev": {"fine": 0.0,
                                                                         "inband": 0.0}},
                        "train": {"loss": A.R3_LOSS[arm], "trainer": A.R3_TRAINER[arm],
                                  "files": list(A.R3_TRAIN_FILES), "seed": s, "epochs": 50,
                                  "epochs_run": 50, "steps": 800, "loss_by_epoch": [-1.0] * 50,
                                  "order_sha256": _order(s), "resumed_from_epoch": None},
                        "state": {"sha256": "f" * 64},
                        "fine": ev(ft["per_seed_eval"][s]), "inband": ev(inb["per_seed_eval"][s]),
                        "seconds": {}}
    return {"config": cfg, "prereg": {"file": "PREREG_FINE3D.md", "config_sha256": sha},
            "provenance": {"config_sha256": sha, "git": git, "seeds": [0, 1, 2],
                           "source_report_sha256": A.PHASE2B_REPORT_SHA256,
                           "datasets": [{"dir": d["dir"], "manifest_sha256": d["manifest_sha256"]}
                                        for d in p2b["provenance"]["datasets"]]},
            "runtime_policy": dict(p2b["runtime_policy"]), "per_seed": per}


def r3pair(p2b, egap=1.0, vm=1.0, lab_egap=1.0, lab_vm=1.0, git="prereg-fine3d"):
    """R3's two arms: the energy arm's fine energy gap and von Mises error
    scaled by egap and vm, the labelled arm's by lab_egap and lab_vm."""
    return {"energy": r3(p2b, egap, vm, git, "energy"),
            "labelled": r3(p2b, lab_egap, lab_vm, git, "labelled")}


def _adj3(pair, p2b, entries=None, **kw):
    return A.adjudicate_r3(pair["energy"], pair["labelled"], p2b,
                           entries if entries is not None else _entries(), **kw)


def _entries():
    return {n: config_sha256(_cfg(n))
            for n in ("r2a_b_m1024", "r2a_s_m1024", "r3_eft_b16", "r3_lft_b16")}


def _adj(b, s, recs, **kw):
    return A.adjudicate_r2a(b, s, recs[0], _entries(), s_factor=S_FACTOR, **kw)


# ------------------------------------------------------------------- R2a --

def test_r2a_supported_and_r2b_recommended(recs):
    e2, _ = recs
    out = _adj(r2a(e2, "r2a_b_m1024"), r2a(e2, "r2a_s_m1024", fine=0.4), recs)
    assert out["verdict"] == "SUPPORTED" and out["conditions"]["failing"] == []
    c = out["conditions"]
    assert math.isclose(c["fine_displacement"]["rel_change"], -0.6, rel_tol=1e-9)
    assert math.isclose(c["fine_over_inband_ratio"]["rel_change"], -0.6, rel_tol=1e-9)
    assert out["r2b"]["recommended"] is True
    assert out["uninformative_note"] is None and out["deviations"] == []
    assert out["fresh_baseline_vs_e2"]["flags"] == []          # the same cells as E2's
    assert out["mechanism"]["b"]["not_evaluated"]
    for k in ("fine_displacement", "fine_over_inband_ratio"):
        r = out["robustness"][k]
        assert r["medians"]["lower"] and r["welch_95"]["high"] < 0
        assert r["instance_resampling_95"]["high"] < 0


def test_the_r2b_line_is_forty_percent_of_the_fine_error(recs):
    e2, _ = recs
    assert A.R2B_FINE_REL_MAX == -0.4
    b = r2a(e2, "r2a_b_m1024")
    yes = _adj(b, r2a(e2, "r2a_s_m1024", fine=0.59), recs)
    no = _adj(b, r2a(e2, "r2a_s_m1024", fine=0.61), recs)
    edge = _adj(b, r2a(e2, "r2a_s_m1024", fine=0.6), recs)
    assert yes["verdict"] == no["verdict"] == edge["verdict"] == "SUPPORTED"
    assert yes["r2b"]["recommended"] is True and no["r2b"]["recommended"] is False
    assert "below the R2b line" in no["r2b"]["why"] and "-40%" in no["r2b"]["why"]
    rel = edge["conditions"]["fine_displacement"]["rel_change"]
    assert edge["r2b"]["recommended"] is (rel <= -0.4)          # the line itself: rel <= -40%
    # the rule reads the fine error, not the ratio: in-band errors also lower
    both = _adj(b, r2a(e2, "r2a_s_m1024", fine=0.55, disp=0.85, inband_p3=0.85), recs)
    assert both["verdict"] == "SUPPORTED"
    assert both["conditions"]["fine_over_inband_ratio"]["rel_change"] > -0.4
    assert both["r2b"]["recommended"] is True


def test_r2a_not_supported_names_the_failing_conditions(recs):
    e2, _ = recs
    b = r2a(e2, "r2a_b_m1024")
    same = _adj(b, r2a(e2, "r2a_s_m1024"), recs)
    assert same["verdict"].startswith("NOT SUPPORTED") and len(same["conditions"]["failing"]) == 2
    assert same["r2b"]["recommended"] is False
    # both levels fall alike: (i) holds, (ii) the growth does not shrink
    both = _adj(b, r2a(e2, "r2a_s_m1024", fine=0.5, disp=0.5), recs)
    assert [f[:4] for f in both["conditions"]["failing"]] == ["(ii)"]
    # K1: the in-band energy gap worse beyond the guard
    k1 = _adj(b, r2a(e2, "r2a_s_m1024", fine=0.4, egap=1.6), recs)
    assert [f[:5] for f in k1["conditions"]["failing"]] == ["(iii)"]
    assert "energy gap" in k1["conditions"]["failing"][0]
    # K1: E8's in-band displacement worse, P3's in-band pass unchanged
    kd = _adj(b, r2a(e2, "r2a_s_m1024", fine=0.4, disp=1.6, inband_p3=1.0), recs)
    assert [f[:5] for f in kd["conditions"]["failing"]] == ["(iii)"]
    assert "displacement error" in kd["conditions"]["failing"][0]
    assert "energy gap" not in kd["conditions"]["failing"][0]


def test_condition_ii_reads_p3s_in_band_pass(recs):
    e2, _ = recs
    out = _adj(r2a(e2, "r2a_b_m1024"), r2a(e2, "r2a_s_m1024", fine=0.4, inband_p3=0.5), recs)
    assert math.isclose(out["conditions"]["fine_over_inband_ratio"]["rel_change"], -0.2,
                        rel_tol=1e-9)                     # E8's cell would give -0.6


def test_r2a_divergence_and_void(recs):
    e2, _ = recs
    s = r2a(e2, "r2a_s_m1024", fine=0.4)
    s["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["per_seed_eval"][1]["disp_rel_l2"] = \
        float("nan")
    out = _adj(r2a(e2, "r2a_b_m1024"), s, recs)
    assert "diverged" in out["verdict"]
    b = r2a(e2, "r2a_b_m1024")
    b["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["per_seed_eval"][0]["disp_rel_l2"] = \
        float("nan")
    b["solve_ledger"] = {"total": 2}
    res = A.adjudicate_fine3d({"b": b, "s": r2a(e2, "r2a_s_m1024")}, None, recs[1], e2,
                              _entries(), s_factor=S_FACTOR)
    assert res["R2a"]["verdict"].startswith("NOT EVALUATED: a per-seed value of the fresh "
                                            "baseline is not finite: the comparison is void")
    # the void keeps the rest: deviations, readings and no recommendation
    assert any("solve ledger 2" in d for d in res["R2a"]["deviations"])
    assert res["R2a"]["r2b"] == {"recommended": False, "why": "R2a not evaluated"}
    assert "fine_displacement" in res["R2a"]["robustness"] and res["R2a"]["checks"]["b"]


def test_r2a_uninformative_note_and_mechanism(recs):
    e2, _ = recs
    b, s = r2a(e2, "r2a_b_m1024"), r2a(e2, "r2a_s_m1024")
    out = _adj(b, s, recs, amp_b=_amp((1.05, 1.0, 1.08)), amp_s=_amp((1.0, 1.0, 1.0)))
    assert "uninformative" in out["uninformative_note"] and "c*" in out["uninformative_note"]
    mech = out["mechanism"]["b"]
    assert math.isclose(mech["fine"]["c_star_median"]["seed_mean"], (1.05 + 1.0 + 1.08) / 3)
    assert mech["remesh"]["c_battery_median_by_lc"]["0.0374"] == 1.5
    # the note reads the fine set's c*, not the in-band one
    deficit = _adj(b, s, recs, amp_b=_amp((1.7, 2.0, 1.5), inband_cstar=(1.0, 1.0, 1.0)))
    assert deficit["uninformative_note"] is None
    inv = _adj(b, s, recs, amp_b=_amp((1.05, 1.0, 1.08), inband_cstar=(1.5, 1.5, 1.5)))
    assert "c*" in inv["uninformative_note"]
    # an amplitude record of another report is not read
    amp = _amp((1.7, 2.0, 1.5), report_sha="0" * 64)
    out = _adj(b, s, recs, amp_b=amp, b_sha256="1" * 64)
    assert "another report" in out["mechanism"]["b"]["not_evaluated"]
    assert out["uninformative_cstar"].startswith("not evaluated")
    # nor an incomplete one, with or without the report's SHA-256
    for sha in (None, "2" * 64):
        out = _adj(b, s, recs, amp_b=_amp((1.7, 2.0, 1.5), complete=False, report_sha=sha),
                   b_sha256=sha)
        assert "incomplete" in out["mechanism"]["b"]["not_evaluated"]
    # a baseline without the growth
    flat = r2a(e2, "r2a_b_m1024", fine=0.2)
    out = _adj(flat, r2a(e2, "r2a_s_m1024", fine=0.2), recs)
    assert "ratio of seed means" in out["uninformative_note"]
    assert out["fresh_baseline_vs_e2"]["flags"]                # lower than E2's beyond the guard


def test_r2a_deviations(recs):
    e2, _ = recs
    b, s = r2a(e2, "r2a_b_m1024"), r2a(e2, "r2a_s_m1024", fine=0.4)
    b["solve_ledger"] = {"total": 4}
    s["results"]["e8"]["metrics"]["d9_restart"]["units_resumed_from_epoch"] = {"ar s3": 12}
    s["d9_reuse_states"] = True
    s["runtime_policy"] = dict(s["runtime_policy"], gpu="another GPU")
    dev = _adj(b, s, recs)["deviations"]
    assert any("solve ledger 4" in d for d in dev)
    assert any("R2a s: restarted" in d and "ar s3" in d for d in dev)
    assert any("different gpu" in d for d in dev)


R2A_REFUSALS = {"unstamped": "not a stamped", "git": "ran on", "seeds": "seeds are",
                "p3 seeds": "seeds are", "p3 in band": "P3 in-band cell",
                "extra key": "beyond PREREG_FINE3D", "s keys in b": "baseline carries S",
                "not s": "is not S", "no load density": "is not S",
                "entries": "verified hash", "corpus": "another corpus",
                "cells": "per-instance values", "e8 cell": "E8 AR cell",
                "label": "not r2a_s_m1024", "factor": "is not S"}


@pytest.mark.parametrize("what", sorted(R2A_REFUSALS))
def test_r2a_refusals(recs, what):
    e2, _ = recs
    b, s = r2a(e2, "r2a_b_m1024"), r2a(e2, "r2a_s_m1024", fine=0.4)
    entries = _entries()
    if what == "unstamped":
        s["prereg"] = None
    elif what == "git":
        s = r2a(e2, "r2a_s_m1024", fine=0.4, git="prereg-fine3d-1-gabc")
    elif what == "seeds":
        s = r2a(e2, "r2a_s_m1024", fine=0.4, seeds=(0, 1, 2))
    elif what == "extra key":
        s["config"]["experiments"]["e8"]["ar_epochs"] = 100
        _restamp(s)
    elif what == "s keys in b":
        b = r2a(e2, "r2a_s_m1024")
        b["config"]["out"] = "runs/fine3d/r2a_b_m1024/report.json"
        _restamp(b)
    elif what == "not s":
        s = r2a(e2, "r2a_b_m1024", fine=0.4)
        s["config"]["out"] = "runs/fine3d/r2a_s_m1024/report.json"
        _restamp(s)
    elif what == "no load density":
        s["config"]["model"]["features"].pop("load_density")
        _restamp(s)
        entries = dict(entries, r2a_s_m1024=s["prereg"]["config_sha256"])
    elif what == "entries":
        entries = dict(entries, r2a_s_m1024="0" * 64)
    elif what == "corpus":
        s["provenance"]["datasets"][1] = dict(s["provenance"]["datasets"][1],
                                              manifest_sha256="0" * 64)
    elif what == "cells":
        s["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["per_seed_eval"][2][
            "per_instance"]["vm_rel_l2"] = [0.1] * 255
    elif what == "p3 seeds":
        s["results"]["p3_transfer"]["protocol"]["seeds"] = [0, 1, 2]
    elif what == "p3 in band":
        s["results"]["p3_transfer"]["metrics"]["ar"]["inband"]["per_seed_eval"][1][
            "per_instance"]["energy_gap_rel"] = [0.1] * 200
    elif what == "e8 cell":
        s["results"]["e8"]["metrics"]["cells"]["ar"]["1024"]["per_seed_eval"][0][
            "per_instance"]["crit_recall"] = [0.5] * 100
    elif what == "label":
        s["config"]["out"] = "runs/fine3d/r2a_x/report.json"
        _restamp(s)
    factor = 2.0 ** -9 if what == "factor" else S_FACTOR
    with pytest.raises(ValueError, match=R2A_REFUSALS[what]):
        A.adjudicate_r2a(b, s, e2, entries, s_factor=factor)
    res = A.adjudicate_fine3d({"b": b, "s": s}, r3pair(recs[1], egap=0.5), recs[1], e2, entries,
                              s_factor=factor)
    assert res["R2a"]["verdict"].startswith("NOT EVALUATED")
    assert res["R3"]["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"    # R3 stands


def test_any_error_in_a_part_leaves_the_other_standing(recs):
    e2, p2b = recs
    s = r2a(e2, "r2a_s_m1024", fine=0.4)
    del s["provenance"]["seeds"]                               # a malformed report: TypeError
    res = A.adjudicate_fine3d({"b": r2a(e2, "r2a_b_m1024"), "s": s}, r3pair(p2b, egap=0.5), p2b,
                              e2, _entries(), s_factor=S_FACTOR)
    assert res["R2a"]["verdict"].startswith("NOT EVALUATED: refused: TypeError")
    assert res["R3"]["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"


# -------------------------------------------------------------------- R3 --

def test_r3_verdicts_and_readings(recs):
    _, p2b = recs
    out = _adj3(r3pair(p2b, egap=0.5), p2b)
    v = out["verdicts"]
    assert v["R3-E"]["verdict"] == "SUPPORTED"
    assert math.isclose(v["R3-E"]["guard"]["rel_change"], -0.5, rel_tol=1e-9)
    assert v["R3-V"]["verdict"] == "NOT SUPPORTED"
    rd = out["readings"]
    assert rd["vs_labelled"]["disp_rel_l2"]["new_is"] == "within the guard"
    assert all(math.isclose(x, -0.5, rel_tol=1e-9) for x in
               rd["vs_labelled"]["energy_gap_rel"]["paired_per_seed_rel"])
    assert rd["inband_vs_labelled"]["energy_gap_rel"]["new_is"] == "within the guard"
    assert rd["inband_vs_before"]["energy"]["energy_gap_rel"]["new_is"] == "within the guard"
    assert rd["vs_zero_shot"]["energy"]["energy_gap_rel"]["new_is"] == "better"
    assert rd["vs_zero_shot"]["labelled"]["disp_rel_l2"]["new_is"] == "better"
    assert math.isclose(rd["labelled_vs_record"]["energy_gap_rel"]["guard"]["rel_change"], 0.0,
                        abs_tol=1e-12)
    assert set(rd["fine_over_inband_disp_ratio_per_seed"]) == {"energy", "labelled"}
    assert rd["robustness"]["R3-E"]["instance_resampling_95"]["high"] < 0
    assert out["deviations"] == [] and out["notes"] == [] and rd["record_flags"] == []
    worse = _adj3(r3pair(p2b, vm=2.0), p2b)
    assert worse["verdicts"]["R3-V"]["verdict"] == "NOT SUPPORTED: worse beyond the guard"
    # recall: higher is better
    assert A._direction({"lower": True, "worse": False}, "crit_recall") == "worse"


def test_r3_reads_the_labelled_arm_not_the_record(recs):
    """The labelled arm at twice the record's energy gap, the energy arm at the
    record's: R3-E reads the labelled arm (rel -50%); the record is a reading,
    and the labelled arm's difference from it a flag. Conversely, an energy arm
    below the record but level with the labelled arm is not supported."""
    _, p2b = recs
    out = _adj3(r3pair(p2b, lab_egap=2.0), p2b)
    assert out["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"
    assert math.isclose(out["verdicts"]["R3-E"]["guard"]["rel_change"], -0.5, rel_tol=1e-9)
    rd = out["readings"]
    assert math.isclose(rd["energy_vs_record"]["energy_gap_rel"]["guard"]["rel_change"], 0.0,
                        abs_tol=1e-12)
    assert rd["labelled_vs_record"]["energy_gap_rel"]["new_is"] == "worse"
    assert len(rd["record_flags"]) == 1
    assert rd["record_flags"][0].startswith("energy_gap_rel: the labelled arm is higher than the "
                                            "recorded labelled fine-tuning beyond the guard")
    assert out["notes"] == rd["record_flags"] and out["deviations"] == []
    level = _adj3(r3pair(p2b, egap=0.5, lab_egap=0.5), p2b)
    assert level["verdicts"]["R3-E"]["verdict"] == "NOT SUPPORTED"
    assert level["readings"]["energy_vs_record"]["energy_gap_rel"]["new_is"] == "better"


def test_r3_divergence_and_void(recs):
    """A non-finite per-seed value of the energy arm fails the verdict it
    enters (diverged); one of the labelled arm voids the verdict whose metric
    it is, and the other verdict, the readings and the deviations stand."""
    _, p2b = recs
    pair = r3pair(p2b, egap=0.5, vm=0.5)
    pair["energy"]["per_seed"]["s1"]["fine"]["vm_rel_l2"] = float("nan")
    out = _adj3(pair, p2b)
    assert out["verdicts"]["R3-V"]["verdict"] == "NOT SUPPORTED (diverged)"
    assert out["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"
    for metric, gone, stands in (("energy_gap_rel", "R3-E", "R3-V"),
                                 ("vm_rel_l2", "R3-V", "R3-E")):
        pair = r3pair(p2b, egap=0.5, vm=0.5)
        pair["labelled"]["per_seed"]["s0"]["fine"][metric] = float("inf")
        pair["energy"]["per_seed"]["s2"]["train"]["resumed_from_epoch"] = 9
        out = _adj3(pair, p2b)
        assert out["verdicts"][gone]["verdict"].startswith(
            f"NOT EVALUATED: the labelled arm's per-seed {metric} ")
        assert out["verdicts"][gone]["verdict"].endswith("is not finite: the comparison is void")
        assert out["verdicts"][stands]["verdict"] == "SUPPORTED"
        assert "R3 energy seed s2: resumed from epoch 9" in out["deviations"]
        assert "not_evaluated" in out["readings"]["vs_labelled"][metric]["guard"]
        assert "not_evaluated" not in out["readings"]["vs_labelled"]["disp_rel_l2"]["guard"]
        paired = out["readings"]["vs_labelled"][metric]["paired_per_seed_rel"]
        assert paired[0] is None and all(isinstance(x, float) for x in paired[1:])
    # the labelled arm's displacement error is no verdict's metric: both stand
    pair = r3pair(p2b, egap=0.5, vm=0.5)
    pair["labelled"]["per_seed"]["s0"]["fine"]["disp_rel_l2"] = float("nan")
    assert all(v["verdict"] == "SUPPORTED" for v in _adj3(pair, p2b)["verdicts"].values())


def test_r3_machines(recs):
    """Each seed's host and GPU driver: one machine for both arms, or a
    deviation naming them."""
    _, p2b = recs
    pair = r3pair(p2b, egap=0.5)
    assert not any("machine" in d for d in _adj3(pair, p2b)["deviations"])
    pair["labelled"]["per_seed"]["s2"]["machine"]["end"] = {"host": "other",
                                                             "gpu_driver": "575.64"}
    out = _adj3(pair, p2b)
    dev = [d for d in out["deviations"] if "machine" in d]
    assert len(dev) == 1 and dev[0].startswith("R3: the seeds of the two arms ran on more than "
                                               "one machine") and '"other"' in dev[0]
    assert [m["host"] for m in out["checks"]["labelled"]["seeds"]["s2"]["machines"]] == \
        ["box", "other"]
    # a seed that started on another driver
    pair = r3pair(p2b, egap=0.5)
    pair["energy"]["per_seed"]["s0"]["machine"]["start"] = {"host": "box", "gpu_driver": None}
    assert any("more than one machine" in d for d in _adj3(pair, p2b)["deviations"])
    assert out["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"          # a deviation, not a refusal


def test_r3_amplitude_readings_are_secondary(recs):
    _, p2b = recs
    pair = r3pair(p2b, egap=0.5)
    del pair["energy"]["per_seed"]["s1"]["fine"]["amplitude"]
    out = _adj3(pair, p2b)
    assert out["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"
    assert "not_evaluated" in out["readings"]["amplitude"]["energy"]
    assert "not_evaluated" not in out["readings"]["amplitude"]["labelled"]


def test_r3_deviations_and_notes(recs):
    _, p2b = recs
    pair = r3pair(p2b, egap=0.5)
    e, lab = pair["energy"], pair["labelled"]
    e["per_seed"]["s0"]["train"]["resumed_from_epoch"] = 7
    lab["per_seed"]["s2"]["from_disk"] = True
    e["runtime_policy"] = dict(e["runtime_policy"], torch="2.13.0", gpu="another GPU")
    lab["runtime_policy"] = dict(lab["runtime_policy"], torch="2.13.0")
    out = _adj3(pair, p2b)
    dev = out["deviations"]
    assert "R3 energy seed s0: resumed from epoch 7" in dev
    assert any("R3 labelled seed s2" in d and "earlier attempt" in d for d in dev)
    assert [d for d in dev if d.startswith("R3: the two arms ran with different")] == \
        [f"R3: the two arms ran with different gpu ('another GPU' / "
         f"{p2b['runtime_policy']['gpu']!r})"]
    assert len(out["notes"]) == 1 and "torch" in out["notes"][0]
    assert "than the recorded labelled fine-tuning" in out["notes"][0]


R3_REFUSALS = {"unstamped": "not a stamped", "entries": "verified hash", "git": "ran on",
               "seed git": "not the report's", "source": "another source report",
               "state": "Phase-2b state", "check": "content check", "files": "trained on",
               "steps": "trained on", "epochs run": "trained on", "loss": "trained on",
               "train seed": "trained with the seed", "seeds": "seeds are",
               "corpus": "another corpus", "evaluation": "per-instance values",
               "fine evaluation": "per-instance values",
               "label": "another configuration", "schedule": "schedule",
               "objective": "trained with loss", "trainer": "trained with loss",
               "config loss": "another configuration"}


@pytest.mark.parametrize("arm", ["energy", "labelled"])
@pytest.mark.parametrize("what", sorted(R3_REFUSALS))
def test_r3_refusals(recs, what, arm):
    e2, p2b = recs
    pair = r3pair(p2b, egap=0.5)
    rep = pair[arm]
    label = A.R3_LABELS[arm]
    entries = _entries()
    s0 = rep["per_seed"]["s0"]
    if what == "unstamped":
        rep["prereg"] = {}
    elif what == "entries":
        entries = dict(entries, **{label: "0" * 64})
    elif what == "git":
        rep["provenance"]["git"] = "prereg-w9"
    elif what == "seed git":
        s0["git"] = "prereg-fine3d-2-gabc"
    elif what == "source":
        rep["provenance"]["source_report_sha256"] = "0" * 64
    elif what == "state":
        s0["source_state"]["sha256"] = "0" * 64
    elif what == "check":
        s0["content_check"]["passed"] = False
    elif what == "files":
        s0["train"]["files"] = list(A.R3_TRAIN_FILES[1:]) + ["instance_00272.npz"]
    elif what == "steps":
        s0["train"]["steps"] = 400
    elif what == "epochs run":
        s0["train"]["epochs_run"] = 25
    elif what == "loss":
        s0["train"]["loss_by_epoch"] = [-1.0] * 25
    elif what == "train seed":
        rep["per_seed"]["s1"]["train"]["seed"] = 0
    elif what == "seeds":
        rep["provenance"]["seeds"] = [0, 1]
    elif what == "corpus":
        rep["provenance"]["datasets"][0]["manifest_sha256"] = "0" * 64
    elif what == "evaluation":
        s0["inband"]["per_instance"]["vm_rel_l2"] = [0.1] * 10
    elif what == "fine evaluation":
        rep["per_seed"]["s2"]["fine"]["per_instance"]["crit_recall"] = [1.0] * 255
    elif what == "label":
        rep["config"] = dict(rep["config"], out="runs/fine3d/other/report.json")
    elif what == "objective":
        rep["per_seed"]["s1"]["train"]["loss"] = "labels" if arm == "energy" else "ar"
    elif what == "trainer":
        rep["per_seed"]["s2"]["train"]["trainer"] = \
            "train_supervised" if arm == "energy" else "pretrain"
    elif what in ("schedule", "config loss"):
        rep["config"] = copy.deepcopy(rep["config"])
        if what == "schedule":
            rep["config"]["train"]["epochs"] = 60
        else:
            rep["config"]["train"]["loss"] = "labels" if arm == "energy" else "ar"
        sha = config_sha256(rep["config"])
        rep["prereg"]["config_sha256"] = rep["provenance"]["config_sha256"] = sha
        for v in rep["per_seed"].values():
            v["config_sha256"] = sha
        entries = dict(entries, **{label: sha})
    with pytest.raises(ValueError, match=R3_REFUSALS[what]):
        _adj3(pair, p2b, entries)
    res = A.adjudicate_fine3d({"b": r2a(e2, "r2a_b_m1024"), "s": r2a(e2, "r2a_s_m1024", fine=0.4)},
                              pair, p2b, e2, entries, s_factor=S_FACTOR)
    assert all(v["verdict"].startswith("NOT EVALUATED") for v in res["R3"]["verdicts"].values())
    assert f"R3 {arm}" in res["R3"]["verdicts"]["R3-E"]["verdict"]   # the arm is named
    assert res["R2a"]["verdict"] == "SUPPORTED"                     # R2a stands


@pytest.mark.parametrize("what", ["git", "order", "swapped"])
def test_r3_pair_refusals(recs, what):
    """The two arms must be one session: the same code, the same instance
    orders, each in its own place."""
    e2, p2b = recs
    pair = r3pair(p2b, egap=0.5)
    lab = pair["labelled"]
    if what == "git":
        lab["provenance"]["git"] = "prereg-fine3d-1-gabc"
        for v in lab["per_seed"].values():
            v["git"] = "prereg-fine3d-1-gabc"
        with pytest.raises(ValueError, match="different code"):
            _adj3(pair, p2b, expected_git=None)
    elif what == "order":
        lab["per_seed"]["s1"]["train"]["order_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="recorded different epoch orders for seed 1"):
            _adj3(pair, p2b)
    else:
        pair = {"energy": pair["labelled"], "labelled": pair["energy"]}
        with pytest.raises(ValueError, match="verified hash"):
            _adj3(pair, p2b)
        with pytest.raises(ValueError, match="another configuration"):
            A.adjudicate_r3(pair["energy"], pair["labelled"], p2b, None)
    res = A.adjudicate_fine3d({"b": r2a(e2, "r2a_b_m1024"), "s": r2a(e2, "r2a_s_m1024", fine=0.4)},
                              pair, p2b, e2, _entries(), s_factor=S_FACTOR)
    assert all(v["verdict"].startswith("NOT EVALUATED") for v in res["R3"]["verdicts"].values())
    assert res["R2a"]["verdict"] == "SUPPORTED"


def test_r3_corpus_is_compared_by_manifest_not_by_directory_name(recs):
    _, p2b = recs
    pair = r3pair(p2b, egap=0.5)
    for rep in pair.values():
        for d in rep["provenance"]["datasets"]:
            d["dir"] = "/elsewhere/copy_of_" + Path(d["dir"]).name
    assert _adj3(pair, p2b)["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"


def test_missing_parts_are_not_evaluated(recs):
    e2, p2b = recs
    res = A.adjudicate_fine3d(None, None, p2b, e2, _entries(),
                              missing={"R3": "R3's report is not in the return (x)"})
    assert res["R2a"]["verdict"].startswith("NOT EVALUATED")
    assert all(v["verdict"] == "NOT EVALUATED: R3's report is not in the return (x)"
               for v in res["R3"]["verdicts"].values())


def test_constants_are_the_records(recs):
    e2, p2b = recs
    spec = importlib.util.spec_from_file_location("make_fine3d_configs",
                                                  ROOT / "scripts" / "make_fine3d_configs.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    assert set(A.R3_LABELS.values()) == set(mk.R3_ARMS)
    for arm, label in A.R3_LABELS.items():
        assert _cfg(label)["train"]["loss"] == mk.R3_ARMS[label]["loss"] == A.R3_LOSS[arm]
    assert A.R3_TRAINER == {"energy": "pretrain", "labelled": "train_supervised"}
    assert hashlib.sha256(P2B.read_bytes()).hexdigest() == A.PHASE2B_REPORT_SHA256
    assert e2["provenance"]["config_sha256"] == A.E2_M1024_CONFIG_SHA256
    assert A.R3_TRAIN_FILES == [f"instance_{i:05d}.npz" for i in range(256, 272)]
    p3 = p2b["config"]["experiments"]["p3_transfer"]
    assert int(p3["fewshot_epochs"]) * A.R3_TRAIN_N == A.R3_STEPS == 800
    assert A.R3_TRAIN_N in p3["fewshot_budgets"]


# ------------------------------------------------------------------ script --

def _draft(text):
    """The file with its CONFIG_SHA256 lines and its footer open (as on the
    stamped commit the box tests, so on the draft)."""
    text = re.sub(r"^(CONFIG_SHA256\[[^\]]+\] = )[0-9a-f]{64}$", r"\g<1><fill before tagging>",
                  text, flags=re.M)
    return re.sub(r"^PREREG_FINE3D_SHA256 = [0-9a-f]{64}$",
                  "PREREG_FINE3D_SHA256 = <record after commit>", text, flags=re.M)


def _script(tmp_path, *args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "adjudicate_fine3d.py"), *args],
                          capture_output=True, text=True, cwd=str(tmp_path))


def _stamped_copy(tmp_path):
    """A copy of the pre-registration with the synthetic configurations' lines
    filled and its footer open; and the draft."""
    text = _draft((ROOT / "PREREG_FINE3D.md").read_text())
    stamped = text
    for n, h in _entries().items():
        stamped = stamped.replace(f"CONFIG_SHA256[{n}] = <fill before tagging>",
                                  f"CONFIG_SHA256[{n}] = {h}")
    pf = tmp_path / "PREREG_FINE3D.md"
    pf.write_text(stamped)
    return pf, text


STEPS = ["r2a_b_m1024", "r2a_s_m1024", "amp_r2a_b_m1024", "amp_r2a_s_m1024", "r3_eft_b16",
         "r3_lft_b16"]


def _status(steps, extra=""):
    return "".join(f"{s}.log start 2026-10-18 10:00:00\n{s}.log exit=0 2026-10-18 11:00:00\n"
                   for s in steps) + extra


def test_script_end_to_end(recs, tmp_path):
    """scripts/adjudicate_fine3d.py on files, against a stamped copy of the
    pre-registration (the synthetic configurations' hashes)."""
    e2, p2b = recs
    pf, text = _stamped_copy(tmp_path)
    pair = r3pair(p2b, egap=0.5)
    pair["energy"]["per_seed"]["s0"]["train"]["resumed_from_epoch"] = 7
    for rep in pair.values():
        rep["runtime_policy"] = dict(rep["runtime_policy"], torch="2.13.0")
    files = {"b": r2a(e2, "r2a_b_m1024"), "s": r2a(e2, "r2a_s_m1024", fine=0.4),
             "r3": pair["energy"], "r3l": pair["labelled"]}
    for k, v in files.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v))
    b_sha = hashlib.sha256((tmp_path / "b.json").read_bytes()).hexdigest()
    (tmp_path / "amp_b.json").write_text(json.dumps(_amp((1.7, 2.0, 1.5), report_sha=b_sha)))
    (tmp_path / "amp_s.json").write_text(json.dumps(_amp((1.0, 1.0, 1.0), report_sha="9" * 64)))
    (tmp_path / "status.txt").write_text(
        _status(STEPS, "r3_eft_b16.log start 2026-10-19 09:00:00\n"))
    out = tmp_path / "verdict.json"
    common = ("--r2a-b", str(tmp_path / "b.json"), "--r2a-s", str(tmp_path / "s.json"),
              "--amp-b", str(tmp_path / "amp_b.json"), "--amp-s", str(tmp_path / "amp_s.json"),
              "--r3", str(tmp_path / "r3.json"), "--r3-labelled", str(tmp_path / "r3l.json"),
              "--prereg", str(pf), "--out", str(out))
    r = _script(tmp_path, *common, "--status", str(tmp_path / "status.txt"))
    assert r.returncode == 0, r.stderr[-2000:]
    v = json.loads(out.read_text())
    assert v["R2a"]["verdict"] == "SUPPORTED" and v["R2a"]["r2b"]["recommended"]
    assert v["R3"]["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"
    assert "fine" in v["R2a"]["mechanism"]["b"]
    assert "another report" in v["R2a"]["mechanism"]["s"]["not_evaluated"]
    assert v["attempts"]["r3_eft_b16"] == 2
    dev = v["deviations"]
    assert "r3_eft_b16: 2 attempts" in dev and "R3 energy seed s0: resumed from epoch 7" in dev
    assert not any(d.startswith("R3: the two arms") for d in dev)
    assert f"{pf}: its last line does not verify (PREREG_FINE3D_SHA256)" in dev
    # without the tag here (the sandbox), or with it (the stamped commit): either way recorded
    assert ("the adjudicating code and the pre-registration read were not compared with "
            "prereg-fine3d (the tag cannot be read here)" in dev
            or f"{pf}: the pre-registration read is not prereg-fine3d's PREREG_FINE3D.md" in dev)
    assert set(v["session_and_code_deviations"]) < set(dev)
    summary = json.loads(r.stdout)
    assert summary["deviations"] == dev
    assert len(v["notes"]) == 1 and "torch" in v["notes"][0] and summary["notes"] == v["notes"]
    assert "NaN" not in out.read_text() and "Infinity" not in out.read_text()
    assert str(pf) in v["inputs"] and len(v["inputs"]) == 10
    assert str(tmp_path / "r3l.json") in v["inputs"]
    adj = v["adjudicator"]
    assert set(adj["files_sha256"]) >= {"src/fejepa/analysis/adjudicate_fine3d.py",
                                        "scripts/adjudicate_fine3d.py", "src/fejepa/report.py",
                                        "scripts/stamp_prereg_fine3d.py",
                                        "configs/r2a_s_m1024.json"}
    assert adj["prereg_self_hash_ok"] is False and not adj["prereg_is_repository_file"]
    # a step missing from the status file, or the status file missing
    (tmp_path / "status2.txt").write_text(_status([s for s in STEPS if s != "amp_r2a_s_m1024"]))
    _script(tmp_path, *common, "--status", str(tmp_path / "status2.txt"))
    v = json.loads(out.read_text())
    assert "amp_r2a_s_m1024: not started" in v["deviations"]
    assert not any(d.endswith("attempts") for d in v["deviations"])
    _script(tmp_path, *common, "--status", str(tmp_path / "gone.txt"))
    v = json.loads(out.read_text())
    assert v["attempts"] is None and not any("not started" in d for d in v["deviations"])
    assert f"the session's status file is not in the return ({tmp_path / 'gone.txt'})" in \
        v["deviations"]


def test_script_reads_what_it_can(recs, tmp_path):
    """A report not in the return or not readable leaves its part NOT
    EVALUATED, an unreadable amplitude record the mechanism readings; the rest
    is adjudicated. The reports must carry the tag; the records and the stamp
    are checked."""
    e2, p2b = recs
    pf, text = _stamped_copy(tmp_path)
    pair = r3pair(p2b, egap=0.5)
    files = {"b": r2a(e2, "r2a_b_m1024"), "s": r2a(e2, "r2a_s_m1024", fine=0.4),
             "r3": pair["energy"], "r3l": pair["labelled"]}
    for k, v in files.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v))
    out = tmp_path / "verdict.json"
    both = ("--r3", str(tmp_path / "r3.json"), "--r3-labelled", str(tmp_path / "r3l.json"))
    r = _script(tmp_path, "--r2a-b", str(tmp_path / "b.json"), "--r2a-s", str(tmp_path / "gone.json"),
                *both, "--prereg", str(pf), "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    v = json.loads(out.read_text())
    assert v["R2a"]["verdict"] == ("NOT EVALUATED: R2a needs both reports: "
                                   f"{tmp_path / 'gone.json'} is not in the return")
    assert v["R3"]["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"
    assert "the session's status file was not given (--status)" in v["deviations"]
    assert v["attempts"] is None
    # R3 needs both arms: one not given, or not in the return
    r = _script(tmp_path, "--r3", str(tmp_path / "r3.json"), "--prereg", str(pf), "--out", str(out))
    v = json.loads(out.read_text())
    assert all(x["verdict"] == "NOT EVALUATED: R3 needs both arms' reports: --r3-labelled was "
               "not given" for x in v["R3"]["verdicts"].values())
    assert v["R2a"]["verdict"] == ("NOT EVALUATED: R2a needs both reports: --r2a-b was not "
                                   "given, --r2a-s was not given")
    r = _script(tmp_path, "--r3", str(tmp_path / "gone.json"), "--r3-labelled",
                str(tmp_path / "r3l.json"), "--prereg", str(pf), "--out", str(out))
    v = json.loads(out.read_text())
    assert all(x["verdict"].endswith(f"{tmp_path / 'gone.json'} is not in the return")
               for x in v["R3"]["verdicts"].values())
    # truncated files
    (tmp_path / "s_cut.json").write_text((tmp_path / "s.json").read_text()[:5000])
    (tmp_path / "r3l_cut.json").write_text((tmp_path / "r3l.json").read_text()[:5000])
    (tmp_path / "amp_cut.json").write_text(json.dumps(_amp((1.7, 2.0, 1.5)))[:300])
    r = _script(tmp_path, "--r2a-b", str(tmp_path / "b.json"), "--r2a-s", str(tmp_path / "s_cut.json"),
                *both, "--prereg", str(pf), "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    v = json.loads(out.read_text())
    assert v["R2a"]["verdict"].startswith("NOT EVALUATED: ") and "is not readable" in \
        v["R2a"]["verdict"]
    assert v["R3"]["verdicts"]["R3-E"]["verdict"] == "SUPPORTED"
    r = _script(tmp_path, "--r2a-b", str(tmp_path / "b.json"), "--r2a-s", str(tmp_path / "s.json"),
                "--amp-b", str(tmp_path / "amp_cut.json"), "--r3", str(tmp_path / "r3.json"),
                "--r3-labelled", str(tmp_path / "r3l_cut.json"), "--prereg", str(pf),
                "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]
    v = json.loads(out.read_text())
    assert v["R2a"]["verdict"] == "SUPPORTED"
    assert "unreadable" in v["R2a"]["mechanism"]["b"]["not_evaluated"]
    assert all("is not readable" in x["verdict"] and "r3l_cut.json" in x["verdict"]
               for x in v["R3"]["verdicts"].values())
    # the reports must carry the tag: another describe string is refused by default
    for k in ("b", "s"):
        rep = r2a(e2, f"r2a_{k}_m1024", fine=0.4 if k == "s" else 1.0, git="prereg-fine3d-1-gabc")
        (tmp_path / f"{k}2.json").write_text(json.dumps(rep))
    r = _script(tmp_path, "--r2a-b", str(tmp_path / "b2.json"), "--r2a-s", str(tmp_path / "s2.json"),
                "--prereg", str(pf), "--out", str(out))
    assert "ran on" in json.loads(out.read_text())["R2a"]["verdict"]
    # the records are checked
    bad = tmp_path / "e2_copy.json"
    bad.write_text(E2.read_text() + " ")
    r = _script(tmp_path, "--r3", str(tmp_path / "r3.json"), "--prereg", str(pf), "--e2", str(bad))
    assert r.returncode != 0 and "not E2's" in (r.stderr + r.stdout)
    # the draft (unstamped) is refused
    (tmp_path / "draft.md").write_text(text)
    r = _script(tmp_path, "--r3", str(tmp_path / "r3.json"), "--prereg", str(tmp_path / "draft.md"))
    assert r.returncode != 0 and "not stamped" in (r.stderr + r.stdout)


def test_the_verdict_file_is_strict_json(recs, tmp_path):
    """Non-finite values (a diverged energy arm, a void verdict) are written as
    null: the verdict file parses as strict JSON, and the verdicts say why."""
    e2, p2b = recs
    pf, _ = _stamped_copy(tmp_path)
    pair = r3pair(p2b, egap=0.5)
    pair["energy"]["per_seed"]["s1"]["fine"]["vm_rel_l2"] = float("nan")
    pair["labelled"]["per_seed"]["s0"]["fine"]["energy_gap_rel"] = float("inf")
    for k, rep in (("r3", pair["energy"]), ("r3l", pair["labelled"])):
        (tmp_path / f"{k}.json").write_text(json.dumps(rep))
    out = tmp_path / "verdict.json"
    r = _script(tmp_path, "--r3", str(tmp_path / "r3.json"), "--r3-labelled",
                str(tmp_path / "r3l.json"), "--prereg", str(pf), "--out", str(out))
    assert r.returncode == 0, r.stderr[-2000:]

    def no_constants(c):
        raise ValueError(f"non-standard JSON constant {c}")
    v = json.loads(out.read_text(), parse_constant=no_constants)
    json.loads(r.stdout, parse_constant=no_constants)
    assert v["R3"]["verdicts"]["R3-V"]["verdict"] == "NOT SUPPORTED (diverged)"
    assert v["R3"]["verdicts"]["R3-E"]["verdict"].startswith("NOT EVALUATED: the labelled arm's")
    assert v["R3"]["verdicts"]["R3-V"]["guard"]["new_mean"] is None


def test_the_adjudicator_knows_the_sessions_steps():
    """The steps whose attempts the adjudicator counts are the session's, in
    its order."""
    S = _adj_script()
    sess = (ROOT / "scripts" / "fine3d_session.sh").read_text()
    calls = re.findall(r"^(r2a|amp|r3) (\S+)", sess, re.M)
    assert tuple(n if f != "amp" else f"amp_{n}" for f, n in calls) == S.STEPS == tuple(STEPS)


def _adj_script():
    spec = importlib.util.spec_from_file_location("adjudicate_fine3d_script",
                                                  ROOT / "scripts" / "adjudicate_fine3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_adjudicator_compares_its_code_and_the_prereg_with_the_tag(tmp_path):
    """In a repository holding the adjudicating files: without the tag nothing
    is compared, and that is a deviation; with it, a file edited after the tag
    and a pre-registration other than the tag's are deviations."""
    S = _adj_script()
    repo = tmp_path / "repo"
    for f in (*S.ADJ_FILES, "PREREG_FINE3D.md"):
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / f, repo / f)
    pre = repo / "PREREG_FINE3D.md"
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(git + cmd, check=True, capture_output=True, env=env)
    rec = S.adjudicator_record(pre, "prereg-fine3d", root=repo)
    assert rec["tag_available"] is False and rec["differs_from_tag"] is None
    assert rec["prereg_is_tags"] is None and rec["prereg_is_repository_file"]
    assert any("(the tag cannot be read here)" in d for d in S.code_deviations(rec, pre))
    subprocess.run(git + ["tag", "-a", "prereg-fine3d", "-m", "t"], check=True, capture_output=True,
                   env=env)
    rec = S.adjudicator_record(pre, "prereg-fine3d", root=repo)
    assert rec["tag_available"] and rec["differs_from_tag"] == [] and rec["prereg_is_tags"]
    assert not any("differ" in d or "is not prereg" in d for d in S.code_deviations(rec, pre))
    f = repo / "src" / "fejepa" / "analysis" / "adjudicate_fine3d.py"
    edited = f.read_text().replace("R2B_FINE_REL_MAX = -0.4", "R2B_FINE_REL_MAX = -0.3")
    assert edited != f.read_text()
    f.write_text(edited)
    other = tmp_path / "PREREG_other.md"
    text = pre.read_text().replace("rel <= -40% in condition (i)", "rel <= -30% in condition (i)")
    assert text != pre.read_text()
    other.write_text(text)
    rec = S.adjudicator_record(other, "prereg-fine3d", root=repo)
    assert rec["differs_from_tag"] == ["src/fejepa/analysis/adjudicate_fine3d.py"]
    assert rec["prereg_is_tags"] is False and not rec["prereg_is_repository_file"]
    dev = S.code_deviations(rec, other)
    assert "the adjudicating files differ from prereg-fine3d: " \
           "['src/fejepa/analysis/adjudicate_fine3d.py']" in dev
    assert f"{other}: the pre-registration read is not prereg-fine3d's PREREG_FINE3D.md" in dev
