"""wp9 Stage 0b: session 2 end to end on a miniature laid out like the box --
an E1-like base run (stamped, its report copied to records/ as on the box),
the evaluation sets under runs/w9/ood2d/, the six wp9 arms generated from the
base by the real generator and stamped against a PREREG_W9.md, session 1's
readings written so that the frozen rules decide what each test needs, the
real decision script run on them; then the session-2 plan (gate, selection,
pre-flight dry runs, command script) and the adjudication (verdicts, the
integrity checks, refusals, deviations)."""

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from fejepa.analysis.adjudicate_w9 import _spec, adjudicate_w9, guard
from fejepa.data.archive import manifest_sha256, save_instance, write_manifest
from fejepa.experiments.runner import run_config
from fejepa.fe.synthetic import generate_synthetic_dataset, synthetic_instance
from fejepa.report import read_prereg_entries, stamp_prereg

ROOT = Path(__file__).resolve().parents[1]
STEPS = 16
MINI = {"w9_c1_n1024": dict(pool=4, epochs=4, reuse=True, s=False, seed_offset=0),
        "w9_c1_n4096": dict(pool=8, epochs=2, reuse=False, s=False, seed_offset=0),
        "w9_c1_n25600": dict(pool=16, epochs=1, reuse=False, s=False, seed_offset=0),
        "w9_c1_n12800": dict(pool=16, epochs=1, reuse=False, s=False, seed_offset=0),
        "w9_b_n1024": dict(pool=4, epochs=4, reuse=False, s=False, seed_offset=3),
        "w9_s_n1024": dict(pool=4, epochs=4, reuse=False, s=True, seed_offset=3)}
SPECS = {k: _spec(v["pool"], v["epochs"], v["reuse"], v["seed_offset"], v["s"])
         for k, v in MINI.items()}
FAMS = ("IB", "F1", "F2", "F3", "F4", "F5", "R")
SETS7 = ("val", *FAMS)                  # the name predates IB: val, IB, F1-F5, R


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _save_family(out: Path, archs: list, family: str, fields=None) -> None:
    out.mkdir(parents=True)
    recs = []
    for i, a in enumerate(archs):
        name = f"instance_{i:05d}.npz"
        save_instance(a, out / name)
        recs.append({"file": name, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
                     "n_nodes": a.n_nodes, "labelled": True, **((fields or [{}] * len(archs))[i])})
    write_manifest(out, recs, {"family": family, "seed": 1})


def _labelled(seed, nx=8, ny=6):
    return synthetic_instance(np.random.default_rng(seed), nx=nx, ny=ny, labelled=True)


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w9box")
    old = os.getcwd()
    os.chdir(tmp)
    try:
        generate_synthetic_dataset(tmp / "runs" / "data2d", n=40, seed=3, labelled="all")
        for k, fam in enumerate(FAMS[1:-1]):                        # F1-F5
            _save_family(tmp / "runs/w9/ood2d" / fam,
                         [_labelled(100 + 10 * k + i, nx=6 + 3 * (fam == "F5")) for i in range(3)], fam)
        # IB larger than the validation split (4), so that weighting by instance
        # count differs from averaging the two sets' means
        _save_family(tmp / "runs/w9/ood2d" / "IB", [_labelled(200 + i) for i in range(12)], "IB")
        r_archs, r_fields = [], []
        for g in range(2):
            for (nx, ny), h in (((6, 4), 0.2), ((12, 8), 0.1)):   # nominal mesh sizes
                r_archs.append(_labelled(500 + g, nx=nx, ny=ny))
                r_fields.append({"geometry": g, "target_h": h})
        _save_family(tmp / "runs/w9/ood2d/R", r_archs, "R", r_fields)
        base = json.loads((ROOT / "configs" / "e1_2d_base.json").read_text())
        base["data"].update(n=40, backend="synthetic")
        base["split"] = {"n_val": 4, "seed": 1}
        base["model"].update(dim=16, depth=1, heads=2)
        base["experiments"]["e8"].update(pool_sizes=[4], ar_epochs=4, seeds=3)
        base.update(device="cpu", workers=1, tf32=False)
        (tmp / "configs").mkdir()
        (tmp / "configs" / "e1_2d_base.json").write_text(json.dumps(base, indent=1))
        (tmp / "PREREG_E1.md").write_text("CONFIG_SHA256[e1_2d_base] = <fill before tagging>\n")
        stamp_prereg(tmp / "PREREG_E1.md", base, label="e1_2d_base")
        run_config("configs/e1_2d_base.json")
        (tmp / "records/wp8/e1/e1_2d_base").mkdir(parents=True)
        shutil.copyfile(tmp / "runs/e1_2d_base/report.json",
                        tmp / "records/wp8/e1/e1_2d_base/report.json")
        mk = _module("make_w9_configs")
        (tmp / "PREREG_W9.md").write_text("".join(f"CONFIG_SHA256[{a}] = <fill before tagging>\n"
                                                  for a in MINI))
        for arm, spec in MINI.items():
            (tmp / "configs" / f"{arm}.json").write_text(
                mk.render(mk.w9_config(base, arm, spec, steps=STEPS)))
            stamp_prereg(tmp / "PREREG_W9.md",
                         json.loads((tmp / "configs" / f"{arm}.json").read_text()), label=arm)
        reports = {}
        for role, arm in (("n1024", "w9_c1_n1024"), ("nmax", "w9_c1_n25600"), ("b1024", "w9_b_n1024"),
                          ("s", "w9_s_n1024"), ("n4096", "w9_c1_n4096")):
            reports[role] = json.loads(json.dumps(run_config(f"configs/{arm}.json"), default=str))
    finally:
        os.chdir(old)
    return tmp, base, reports


def _ood_record(tmp):
    """Session 1's record; sizes and seeds as it records PREREG_W9's sets (the
    plan and the adjudication check these fields; the miniature's files are
    smaller)."""
    from fejepa.fe.ood2d import DEFAULT_SEEDS, SET_SIZES

    return {"families": {f: {"manifest_sha256": manifest_sha256(tmp / "runs/w9/ood2d" / f),
                             "status": "generated", "n_instances": SET_SIZES[f],
                             "seed": DEFAULT_SEEDS[f]} for f in FAMS}, "failed": []}


def _describe() -> str:
    from fejepa.report import _git_describe

    return _git_describe()


def _session1(where: Path, tmp: Path, s_enters=True, pool="25600p", ckpt_off=False,
              repro=1e-9, timing=True) -> Path:
    """Session 1's readings, written so that the FROZEN rules decide as asked,
    and the real decision script run on them (its SHA-256 and the inputs'
    are recorded as on the box). pool: "25600p" | "25600s" | "12800p" | "none";
    timing=False: the timing step failed (no profile file; rule 3 undecided)."""
    s1 = where / "s1"
    s1.mkdir(parents=True, exist_ok=True)
    rec = _ood_record(tmp)
    (s1 / "ood2d.json").write_text(json.dumps(rec))
    (s1 / "c0_amp2d.json").write_text(json.dumps({
        "F5": {"manifest_sha256": rec["families"]["F5"]["manifest_sha256"]},
        "R": {"manifest_sha256": rec["families"]["R"]["manifest_sha256"]},
        "summary": {"ratio_disp_F5_over_inband": 2.0 if s_enters else 1.1,
                    "c_star_median_F5": 1.3}}))
    if timing:
        (s1 / "profile_2d_w9.json").write_text(json.dumps({"variants": {
            "worker3": {"ar": {"ms_per_step_per_unit": 52.0, "valid": True}},
            "worker3_no_ckpt": {"ar": {"ms_per_step_per_unit": 30.0 if ckpt_off else 50.0,
                                       "valid": True}}}}))
    rss = {"25600p": (1e9, 2e9, 4e9, 8e9), "25600s": (2e9, 6e9, 15e9, 30e9),
           "12800p": (2e9, 6e9, 20e9, 90e9), "none": (2e9, 6e9, 100e9, 200e9)}[pool]
    (s1 / "c0_memory.json").write_text(json.dumps({
        "limits": {"usable": 100e9, "usable_source": "cgroup_limit"}, "workers": 3,
        "marks": [{"n": n, "rss": r} for n, r in zip((1024, 4096, 12800, 25600), rss)]}))
    (s1 / "c0_trainval.json").write_text(json.dumps({"summary": {"egap_train_over_val": 0.95}}))
    (s1 / "c0_val.json").write_text(json.dumps({"summary": {
        "cg_estimates": {"10": {"spearman_seed_mean": 0.95}},
        "reproduction": {"disp": {"s0": 0.0, "s1": 0.0}, "egap": {"s0": 0.0, "s1": repro}}}}))
    steps = ("ood2d.log", "c0_val.log", "c0_trainval.log", "c0_amp2d.log", "c0_memory.log",
             "profile_w9.log", "decisions.log")
    (s1 / "status.txt").write_text("".join(
        f"{s} exit={1 if (s == 'profile_w9.log' and not timing) else 0}\n" for s in steps))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "w9_session1_decisions.py"),
                        "--dir", str(s1), "--out", str(s1 / "decisions.json")],
                       capture_output=True, text=True, cwd=str(tmp))
    assert (s1 / "decisions.json").is_file(), r.stdout + r.stderr
    return s1


def _s1_args(s1: Path) -> dict:
    from fejepa.analysis.adjudicate_w9 import file_sha256

    inputs, sha = _module("adjudicate_w9").session1_readings(s1, file_sha256)
    rules = _module("w9_session1_decisions")
    return dict(session1_inputs=inputs, session1_sha256=sha, decide=rules.decide,
                rules_sha256=hashlib.sha256(
                    (ROOT / "scripts" / "w9_session1_decisions.py").read_bytes()).hexdigest())


def _adj(tmp, reports, s1, **kw):
    e1p = tmp / "records/wp8/e1/e1_2d_base/report.json"
    dec = json.loads((s1 / "decisions.json").read_text())
    args = dict(r_manifest=json.loads((tmp / "runs/w9/ood2d/R/manifest.json").read_text()),
                specs=SPECS, prereg_entries=dict(read_prereg_entries(tmp / "PREREG_W9.md")),
                e1_report_sha256=hashlib.sha256(e1p.read_bytes()).hexdigest(),
                expected_git=None, **_s1_args(s1))
    args.update(kw)
    ood = args.pop("ood_record", None) or json.loads((s1 / "ood2d.json").read_text())
    dec = args.pop("decisions", None) or dec
    return adjudicate_w9(reports, json.loads(e1p.read_text()), dec, ood, **args)


def _copy(reports):
    return json.loads(json.dumps(reports))


def _set(rep: dict, set_name: str, metric: str, values) -> None:
    m = rep["results"]["e8"]["metrics"]
    cells = m["cells"]["ar"] if set_name == "val" else m["holdouts"][set_name]
    for e, v in zip(cells[max(cells, key=int)]["per_seed_eval"], values, strict=True):
        e[metric] = v


# ------------------------------------------------------------- the guard --

def test_guard_semantics():
    g = guard([1.0, 1.0, 1.0], [0.8, 0.8, 0.8])
    assert g["lower"] and not g["worse"] and g["threshold"] == 0.10
    assert np.isclose(g["rel_change"], -0.2)
    g = guard([1.0, 1.1, 0.9], [0.95, 0.85, 1.05])        # noise: tau > 10 %
    assert g["threshold"] > 0.10 and not g["lower"] and not g["worse"]
    assert guard([1.0, 1.0, 1.0], [0.9, 0.9, 0.9])["lower"] is False      # exactly -10 %: not beyond
    g = guard([1.0, 1.0, 1.0], [1.0, float("nan"), 1.0])
    assert g["diverged"] and g["worse"] and not g["lower"]
    with pytest.raises(ValueError, match="non-finite"):
        guard([1.0, float("inf"), 1.0], [1.0, 1.0, 1.0])


# ------------------------------------------------------------ adjudication --

def test_adjudication_reads_every_arm(box, tmp_path):
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)
    res = _adj(tmp, reports, s1)
    assert res["notes"] == ["decisions recomputed from the session-1 readings: identical"]
    assert res["refused_reports"] == {}
    assert res["H1"]["verdict"].startswith(("SUPPORTED", "NOT SUPPORTED"))
    assert res["H1"]["n_max_configuration"] == "w9_c1_n25600"
    assert res["H2"]["verdict"].startswith(("SUPPORTED", "NOT SUPPORTED"))
    assert set(res["H2"]) >= {"F5_displacement", "F5_over_inband_ratio", "K1_inband_energy_gap",
                              "K1_inband_displacement"}
    assert set(res["tables"]) == {"n1024", "nmax", "b1024", "s", "n4096"}
    assert res["arms"]["b1024"]["identity"]["seed_offset"] == 3
    assert sorted(res["arms"]["s"]["states"]) == ["s3", "s4", "s5"]
    assert res["arms"]["n1024"]["reproduction_max_rel_dev"] == 0.0
    for tab in res["tables"].values():
        assert len(tab["F5"]["tail_energy_gap_rel"]["per_seed"]) == 3
    rm = res["remesh"]["n1024"]
    assert len(rm) == 2 and rm["0.2"]["u_norm_ratio_median"] == [1.0, 1.0, 1.0]
    assert rm["0.2"]["ustar_norm_ratio_median"] == [1.0, 1.0, 1.0]
    assert set(res["replication_b1024_vs_n1024"]) == {*SETS7, "inband"}   # r3: every set
    assert set(res["nmax_vs_b1024_exploratory"]) == {*SETS7, "inband"}
    assert set(res["H1"]["robustness"]) == {"medians", "welch_95", "instance_resampling_95"}
    assert "not_evaluated" not in res["H1"]["robustness"]["instance_resampling_95"]
    t = res["tables"]["nmax"]
    for k in range(3):                    # H1's per-seed value: val (4) and IB (12) by count
        want = (4 * t["val"]["energy_gap_rel"]["per_seed"][k]
                + 12 * t["IB"]["energy_gap_rel"]["per_seed"][k]) / 16
        assert t["inband"]["energy_gap_rel"]["per_seed"][k] == pytest.approx(want)
    assert res["H1"]["new_per_seed"] == pytest.approx(t["inband"]["energy_gap_rel"]["per_seed"])
    assert set(res["H2"]["robustness"]) == {"F5_displacement", "F5_over_inband_ratio",
                                            "K1_inband_energy_gap", "K1_inband_displacement"}
    for k, r in res["H2"]["robustness"].items():
        assert "not_evaluated" not in r["instance_resampling_95"], k
        assert r["instance_resampling_95"]["low"] <= r["instance_resampling_95"]["high"]
    t = res["tables"]["s"]
    assert t["ratio_F5_over_val_disp_of_seed_means"] == pytest.approx(
        t["F5"]["disp_rel_l2"]["seed_mean"] / t["val"]["disp_rel_l2"]["seed_mean"])
    assert set(res["remesh_growth"]["s"]) == {"0.2"} and len(res["remesh_growth"]["s"]["0.2"]) == 3
    assert set(res["remesh_growth_s_vs_b1024_exploratory"]) == {"0.2"}
    assert set(res["comparisons_s_vs_b1024_exploratory"]) == {*SETS7, "inband"}
    assert any("no session-2 plan" in d for d in res["deviations"])
    assert not any("checkpointing" in d or "workers" in d for d in res["deviations"])


def _inband(rep: dict, metric: str, values) -> None:
    """H1's in-band set (r3): the validation split and IB, both set alike."""
    for s in ("val", "IB"):
        _set(rep, s, metric, values)


def test_h1_verdicts_by_constructed_values(box, tmp_path):
    """H1 reads the in-band energy gap -- the validation split and IB
    together (r3) -- the 1,024 arm as the reference."""
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)

    def h1(nmax_egap, nmax_disp=None):
        rep = _copy(reports)
        _inband(rep["n1024"], "energy_gap_rel", (0.040, 0.041, 0.039))
        _inband(rep["nmax"], "energy_gap_rel", nmax_egap)
        if nmax_disp:
            _inband(rep["nmax"], "disp_rel_l2", nmax_disp)
        return _adj(tmp, rep, s1)["H1"]

    r = h1((0.030, 0.031, 0.029))
    assert r["verdict"] == "SUPPORTED" and np.isclose(r["rel_change"], -0.25)
    assert np.allclose(r["base_per_seed"], [0.040, 0.041, 0.039])
    assert np.allclose(r["new_per_seed"], [0.030, 0.031, 0.029]) and r["inband_sets"] == ["val", "IB"]
    rep = _copy(reports)                         # the two sets weigh by their instance counts
    _inband(rep["n1024"], "energy_gap_rel", (0.040, 0.040, 0.040))
    _set(rep["nmax"], "val", "energy_gap_rel", (0.020, 0.020, 0.020))         # 4 instances
    _set(rep["nmax"], "IB", "energy_gap_rel", (0.035, 0.035, 0.035))          # 12 instances
    r = _adj(tmp, rep, s1)["H1"]
    assert np.isclose(r["rel_change"], -0.21875) and r["verdict"] == "SUPPORTED"
    _set(rep["nmax"], "IB", "energy_gap_rel", (0.042, 0.042, 0.042))
    r = _adj(tmp, rep, s1)["H1"]                 # the two sets' means averaged would give -22.5 %
    assert np.isclose(r["rel_change"], -0.0875) and r["verdict"] == "NOT SUPPORTED"
    assert h1((0.040, 0.0405, 0.0395))["verdict"] == "NOT SUPPORTED"
    assert h1((0.050, 0.051, 0.049))["verdict"] == "NOT SUPPORTED (worse beyond the guard)"
    assert h1((0.030, float("nan"), 0.030))["verdict"] == "NOT SUPPORTED (diverged)"
    assert h1((0.040, 0.041, 0.039), nmax_disp=(0.01, 0.01, 0.01))["verdict"] == "NOT SUPPORTED"
    rep = _copy(reports)                         # r3: the fresh baseline checks the reuse
    _inband(rep["n1024"], "energy_gap_rel", (0.040, 0.041, 0.039))
    _inband(rep["b1024"], "energy_gap_rel", (0.040, 0.0405, 0.0395))
    assert "fresh_baseline_flag" not in _adj(tmp, rep, s1)["H1"]
    _inband(rep["b1024"], "energy_gap_rel", (0.052, 0.053, 0.051))
    r = _adj(tmp, rep, s1)["H1"]
    assert r["fresh_baseline_flag"].startswith("the fresh baseline's in-band energy gap is worse "
                                               "than E1's states beyond the guard")
    assert r["verdict"] in ("SUPPORTED", "NOT SUPPORTED", "NOT SUPPORTED (worse beyond the guard)")
    _set(rep["b1024"], "IB", "energy_gap_rel", (0.04, float("nan"), 0.04))    # never blocks H1
    for s_enters in (True, False):
        res = _adj(tmp, rep, _session1(tmp_path / f"nan{s_enters}", tmp, s_enters=s_enters))
        assert "non-finite (diverged)" in res["H1"]["fresh_baseline_flag"]
        assert res["H1"]["verdict"].startswith(("SUPPORTED", "NOT SUPPORTED"))


def test_h2_conditions_and_a_void_reference(box, tmp_path):
    """H2 = (i) F5 lower AND (ii) the F5 / in-band ratio lower AND (iii) K1 on
    both in-band metrics; a lower F5 error with the ratio unchanged is not
    support; a non-finite fresh baseline voids H2 only; a diverged S fails the
    condition."""
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)

    def arms(s_f5, s_val_disp, s_val_egap=(0.05, 0.05, 0.05), b_f5=(0.40, 0.41, 0.39)):
        rep = _copy(reports)
        _set(rep["b1024"], "F5", "disp_rel_l2", b_f5)
        _set(rep["b1024"], "val", "disp_rel_l2", (0.20, 0.20, 0.20))
        _set(rep["b1024"], "val", "energy_gap_rel", (0.05, 0.05, 0.05))
        _set(rep["s"], "F5", "disp_rel_l2", s_f5)
        _set(rep["s"], "val", "disp_rel_l2", s_val_disp)
        _set(rep["s"], "val", "energy_gap_rel", s_val_egap)
        return rep

    res = _adj(tmp, arms((0.20, 0.21, 0.19), (0.20, 0.20, 0.20)), s1)
    assert res["H2"]["verdict"] == "SUPPORTED" and "uninformative" not in res["H2"]
    res = _adj(tmp, arms((0.24, 0.25, 0.26), (0.20, 0.20, 0.20), b_f5=(0.25, 0.25, 0.25)), s1)
    assert res["H2"]["verdict"].startswith("NOT SUPPORTED: (i)")        # r3: no deficit to remove
    assert "F5 / in-band ratio 1.25 < 1.5" in res["H2"]["uninformative"]
    res = _adj(tmp, arms((0.32, 0.33, 0.31), (0.16, 0.16, 0.16)), s1)    # both 20 % lower
    assert res["H2"]["verdict"].startswith("NOT SUPPORTED: (ii)")       # the growth is unchanged
    res = _adj(tmp, arms((0.20, 0.21, 0.19), (0.20, 0.20, 0.20), (0.07, 0.07, 0.07)), s1)
    assert res["H2"]["verdict"] == ("NOT SUPPORTED: (iii) K1: in-band energy gap worse "
                                    "beyond the guard")
    res = _adj(tmp, arms((0.26, 0.27, 0.25), (0.26, 0.26, 0.26)), s1)
    assert res["H2"]["verdict"] == ("NOT SUPPORTED: (iii) K1: in-band displacement error worse "
                                    "beyond the guard")
    res = _adj(tmp, arms((0.20, float("nan"), 0.19), (0.20, 0.20, 0.20)), s1)
    assert "(i) F5 displacement error not lower beyond the guard -- diverged" in res["H2"]["verdict"]
    res = _adj(tmp, arms((0.20, 0.21, 0.19), (0.20, 0.20, 0.20), b_f5=(0.4, float("nan"), 0.4)),
               s1)
    assert res["H2"]["verdict"].startswith("NOT EVALUATED (a non-finite value of the fresh")
    assert any(d.startswith("b1024: a non-finite value") for d in res["deviations"])
    assert res["H1"]["verdict"] and "S_mechanism" in res


def test_reports_that_fail_their_checks_are_refused_alone(box, tmp_path):
    """A report that is not what PREREG_W9 stamped for its role is left out
    and recorded; only the hypothesis that needs it is not evaluated."""
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)
    e1p = tmp / "records/wp8/e1/e1_2d_base/report.json"

    def refused(edit=None, roles=None, **kw):
        rep = _copy(reports)
        if roles:
            rep = {r: rep[src] for r, src in roles.items()}
        if edit:
            edit(rep)
        return _adj(tmp, rep, s1, **kw)

    def setp(path, value):
        def f(rep):
            d = rep
            for k in path[:-1]:
                d = d[k]
            d[path[-1]] = value
        return f

    cases = [
        ("nmax", setp(("nmax", "prereg"), None), "not a stamped"),
        ("nmax", None, "CONFIG_SHA256\\[w9_c1_n25600\\]"),
        ("n4096", setp(("n4096", "config", "pretrain", "lr"), 5e-4), "beyond PREREG_W9"),
        ("s", lambda r: r["s"]["config"]["model"].pop("decode_scale_factor"), "stamps"),
        ("nmax", lambda r: r["nmax"]["provenance"]["datasets"][0].update(manifest_sha256="0" * 64),
         "another corpus"),
        ("b1024", setp(("b1024", "provenance", "seeds"), [0, 1, 2]), "seeds"),
        ("n4096", setp(("n4096", "evaluation", "amplitude"), False), "evaluation is not"),
        ("s", setp(("s", "results", "e8", "protocol", "eval_only"), True), "did not train"),
        ("n1024", lambda r: r["n1024"]["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
         ["s1"].update(sha256="f" * 64), "did not evaluate E1's"),
        ("nmax", lambda r: r["nmax"]["results"]["e8"]["metrics"]["cells"]["ar"]["16"]
         ["per_seed_eval"].pop(), "one evaluation per seed"),
        ("n4096", lambda r: r["n4096"]["evaluation"]["holdouts"]["F3"].update(
            manifest_sha256="0" * 64), "holdout manifests"),
        ("nmax", lambda r: r["nmax"]["results"]["e8"]["metrics"]["holdouts"]["IB"]["16"]
         ["per_seed_eval"][1]["per_instance"]["energy_gap_rel"].pop(),
         "IB cell does not hold per-instance values for its 12 instances"),
        ("n4096", lambda r: r["n4096"]["results"]["e8"]["metrics"]["cells"]["ar"]["8"]
         ["per_seed_eval"][0].pop("per_instance"), "val cell does not hold per-instance"),
    ]
    for role, edit, match in cases:
        kw = {}
        if match.startswith("CONFIG"):
            entries = dict(read_prereg_entries(tmp / "PREREG_W9.md"))
            entries["w9_c1_n25600"] = "a" * 64
            kw["prereg_entries"] = entries
        res = refused(edit, **kw)
        assert set(res["refused_reports"]) == {role}, (match, res["refused_reports"])
        assert re.search(match, res["refused_reports"][role]), (match, res["refused_reports"])
        assert any(d.startswith(f"{role}: report refused") for d in res["deviations"])
        h1_needs, h2_needs = role in ("n1024", "nmax"), role in ("b1024", "s")
        assert res["H1"]["verdict"].startswith("NOT EVALUATED") == h1_needs, match
        assert res["H2"]["verdict"].startswith("NOT EVALUATED") == h2_needs, match
    for mutate, match in (
            (lambda pi: pi.__setitem__(0, pi[0] * 1.01), "reproduce E1's"),
            (lambda pi: pi.__setitem__(0, float("nan")), "reproduce E1's")):
        rep = _copy(reports)
        mutate(rep["n1024"]["results"]["e8"]["metrics"]["cells"]["ar"]["4"]["per_seed_eval"][0]
               ["per_instance"]["energy_gap_rel"])
        res = _adj(tmp, rep, s1)
        assert match in res["refused_reports"]["n1024"]
    res = _adj(tmp, reports, s1, e1_report_sha256="a" * 64)
    assert "through another report" in res["refused_reports"]["n1024"]
    res = refused(roles={"n1024": "n1024", "n4096": "nmax"})            # a report in another role
    assert "not one of ('w9_c1_n4096',)" in res["refused_reports"]["n4096"]
    describe = reports["nmax"]["provenance"]["git"]
    dec = json.loads((s1 / "decisions.json").read_text())
    dec["git"] = describe
    res = refused(setp(("n4096", "provenance", "git"), describe + "-x"), decisions=dec,
                  expected_git=describe)
    assert set(res["refused_reports"]) == {"n4096"} and "ran on" in res["refused_reports"]["n4096"]
    res = _adj(tmp, reports, _session1(tmp_path / "a", tmp, pool="12800p"))
    assert "rule 2 selected w9_c1_n12800" in res["refused_reports"]["nmax"]
    assert res["H1"]["verdict"] == "NOT EVALUATED (nmax: refused)"
    res = _adj(tmp, reports, _session1(tmp_path / "b", tmp, s_enters=False))
    assert {"s"} == set(res["refused_reports"])         # r3: the fresh baseline runs regardless
    assert "did not admit S" in res["refused_reports"]["s"]
    assert res["H2"]["verdict"].startswith("NOT RUN") and e1p.is_file()
    assert "replication_b1024_vs_n1024" in res and "nmax_vs_b1024_exploratory" in res
    rep_ = _copy(reports)
    del rep_["b1024"], rep_["s"]
    res = _adj(tmp, rep_, _session1(tmp_path / "c", tmp, s_enters=False))
    assert "b1024: no report (planned arm not run or not returned)" in res["deviations"]


def test_session_level_faults_refuse_the_adjudication(box, tmp_path):
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)
    ood = json.loads((s1 / "ood2d.json").read_text())
    ood["families"]["F3"]["manifest_sha256"] = None
    with pytest.raises(ValueError, match="record is incomplete"):
        _adj(tmp, reports, s1, ood_record=ood)
    for fam, key, value in (("IB", "n_instances", 512), ("F2", "seed", 1)):     # not Sec. 3's
        ood = json.loads((s1 / "ood2d.json").read_text())
        ood["families"][fam][key] = value
        with pytest.raises(ValueError, match=f"sets are not PREREG_W9's .*'{fam}'"):
            _adj(tmp, reports, s1, ood_record=ood)
    with pytest.raises(ValueError, match="R manifest given"):
        _adj(tmp, reports, s1, r_manifest_sha256="b" * 64)
    dec = json.loads((s1 / "decisions.json").read_text())
    dec["_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="planned from another decision file"):
        _adj(tmp, reports, s1, decisions=dec, plan={"decisions_sha256": "d" * 64, "arms": []})
    status = "c1_n4096.log start 2026-10-03 01:00:00\nc1_n4096.log exit=0 2026-10-03 04:00:00\n"
    with pytest.raises(ValueError, match="shows \\['c1_n4096'\\] finished"):
        _adj(tmp, {k: v for k, v in reports.items() if k != "n4096"}, s1, status_text=status)
    status = "b_n1024: report exists, skipped\n"
    with pytest.raises(ValueError, match="shows \\['b_n1024'\\] finished"):
        _adj(tmp, {k: v for k, v in reports.items() if k != "b1024"}, s1, status_text=status)
    status = "c1_n4096.log start 2026-10-03 01:00:00\nc1_n4096.log exit=1 2026-10-03 01:10:00\n"
    res = _adj(tmp, {k: v for k, v in reports.items() if k != "n4096"}, s1, status_text=status)
    assert any(d.startswith("n4096: no report") for d in res["deviations"])


def test_adjudication_checks_the_decisions_integrity(box, tmp_path):
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)
    dec = json.loads((s1 / "decisions.json").read_text())
    dec["decisions"]["S_enters"]["value"] = False                  # edited by hand
    with pytest.raises(ValueError, match="rule S_enters recomputed"):
        _adj(tmp, {k: v for k, v in reports.items() if k not in ("s", "b1024")}, s1,
             decisions=dec)
    with pytest.raises(ValueError, match="frozen rules"):
        _adj(tmp, reports, s1, rules_sha256="0" * 64)
    args = _s1_args(s1)
    args["session1_sha256"]["c0_memory.json"] = "f" * 64
    with pytest.raises(ValueError, match="recorded inputs"):
        _adj(tmp, reports, s1, session1_sha256=args["session1_sha256"])
    dec = json.loads((s1 / "decisions.json").read_text())
    dec["git"] = "1b32a34-dirty"                                    # not the stamped commit
    with pytest.raises(ValueError, match="decisions ran on '1b32a34-dirty'"):
        _adj(tmp, reports, s1, decisions=dec, expected_git="prereg-w9")


def test_adjudication_records_deviations_and_tolerates_secondary_gaps(box, tmp_path):
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp, pool="25600s", ckpt_off=True)
    rep = _copy(reports)
    rep["nmax"]["results"]["e8"]["protocol"]["workers"] = 3        # as if run in parallel
    rep["n4096"]["provenance"]["versions"]["torch"] = "9.9.9"
    rep["b1024"]["runtime_policy"]["gpu"] = "another GPU"
    rep["s"]["solve_ledger"] = {"total": 12}
    rep["s"]["d9_reuse_states"] = True
    rep["s"]["results"]["e8"]["metrics"]["d9_restart"]["units_resumed_from_epoch"] = {"ar s3": 7}
    cell = rep["n1024"]["results"]["e8"]["metrics"]["holdouts"]["F3"]["4"]["per_seed_eval"][0]
    cell["amplitude"]["summary"]["egap_c"] = float("nan")         # a secondary reading
    plan = {"arms": [{"config": "configs/w9_c1_n25600.json",
                      "flags": ["--workers", "1", "--activation-checkpointing", "off"]}]}
    status = ("c1_n25600.log start 2026-10-03 01:00:00\nc1_n25600.log exit=1 2026-10-03 02:00:00\n"
              "c1_n25600.log start 2026-10-03 02:01:00\nc1_n25600.log exit=0 2026-10-03 05:00:00\n")
    res = _adj(tmp, rep, s1, plan=plan, status_text=status)
    devs = " | ".join(res["deviations"])
    assert "nmax: 3 workers, the plan chose 1" in devs
    assert "nmax: activation checkpointing on, the plan chose off" in devs
    assert "n4096: software or hardware differs" in devs and "b1024: software or hardware" in devs
    assert "s: solve ledger 12" in devs and "s: run in restart mode" in devs
    assert "s: units resumed from epoch checkpoints {'ar s3': 7}" in devs
    assert "c1_n25600.log: 2 attempts" in devs
    assert "n1024" not in devs.split("software")[0]                 # evaluation only
    cmp_ = res["comparisons_vs_n1024_exploratory"]["nmax"]["F3"]["egap_c"]
    assert "not_evaluated" in cmp_ and res["H1"]["verdict"]
    res = _adj(tmp, reports, _session1(tmp_path / "t", tmp, timing=False))
    assert "rule 3 undecided: checkpointing kept on (memory and time only)" in res["deviations"]


def test_a_missing_primary_arm_voids_only_its_own_hypothesis(box, tmp_path):
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)
    res = _adj(tmp, {k: v for k, v in reports.items() if k != "nmax"}, s1)
    assert res["H1"]["verdict"] == "NOT EVALUATED (nmax: no report)"
    assert res["H2"]["verdict"].startswith(("SUPPORTED", "NOT SUPPORTED"))
    assert any(d.startswith("nmax: no report") for d in res["deviations"])
    res = _adj(tmp, {k: v for k, v in reports.items() if k != "n1024"}, s1)
    assert res["H1"]["verdict"].startswith("NOT EVALUATED")
    assert res["comparisons_vs_n1024_exploratory"] == {"not_evaluated": "no valid n1024 report"}
    assert set(res["S_mechanism"]) == {"b1024", "s"}
    res = _adj(tmp, {k: v for k, v in reports.items() if k != "s"}, s1)
    assert res["H2"]["verdict"] == "NOT EVALUATED (s: no report)" and res["H1"]["verdict"]



# --------------------------------------------------------- paper material --

def test_paper_material_from_the_verdict(box, tmp_path):
    """scripts/make_w9_paper_material.py reads the adjudication's verdict only:
    deterministic, every cell from the verdict, LaTeX text checked; without S
    no H2 tables."""
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)
    mk = _module("make_w9_paper_material")
    res = json.loads(json.dumps(_adj(tmp, reports, s1), default=str))
    files = mk.build(res, "0" * 64)
    assert set(files) == {"table_w9_h1.tex", "table_w9_c1.tex", "table_w9_c1_disp.tex",
                          "table_w9_h2.tex", "table_w9_s_sets.tex", "table_w9_remesh.tex",
                          "tables.md", "sources.json"}
    assert mk.build(json.loads(json.dumps(res)), "0" * 64) == files        # deterministic
    md = files["tables.md"]
    for role in ("n1024", "nmax"):                   # H1's quantity: val and IB together (r3)
        per = res["tables"][role]["inband"]["energy_gap_rel"]["per_seed"]
        assert " / ".join(mk._g(x) for x in per) in md
    assert f"Verdict: {res['H1']['verdict']}." in md and f"Verdict: {res['H2']['verdict']}." in md
    assert "N_max against 1,024, beside the verdict (no criterion): per-seed medians" in md
    assert "(i) F5 displacement, beside the verdict (no criterion)" in md
    assert "| AR, fresh seeds | 4 | 4 | 3-5 |" in md          # r3: H1's table (miniature pool)
    assert "| 4 | 0-2 |" in md and "| 4 | 3-5 |" in md          # C1: both 1,024-like arms
    assert "AR, fresh seeds (4): displacement error at the finest mesh over that at h 0.2: " in md
    assert "growth over h 0.2 (exploratory)" in md
    assert "displacement error at the finest mesh" in files["table_w9_remesh.tex"]
    assert json.loads(files["sources.json"])["verdict_sha256"] == "0" * 64
    for name, text in files.items():
        if name.endswith(".tex"):
            assert "_W9" not in text.replace(r"\_W9", "") and "N_max" not in text, name
    rep = _copy(reports)                             # a diverged N_max seed prints as such
    _inband(rep["nmax"], "energy_gap_rel", (0.03, float("nan"), 0.03))
    res = json.loads(json.dumps(_adj(tmp, rep, s1), default=str))
    assert res["H1"]["verdict"] == "NOT SUPPORTED (diverged)"
    files = mk.build(res, "2" * 64)
    assert "0.0300 / diverged / 0.0300 | diverged |" in files["tables.md"]
    rep = _copy(reports)
    del rep["s"]
    res = json.loads(json.dumps(_adj(tmp, rep, s1), default=str))
    assert res["H2"]["verdict"].startswith("NOT EVALUATED")
    files = mk.build(res, "1" * 64)
    assert "table_w9_h2.tex" not in files and "table_w9_s_sets.tex" not in files
    assert f"H2: {res['H2']['verdict']}." in files["tables.md"]


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no pdflatex")
def test_paper_material_compiles(box, tmp_path):
    tmp, _, reports = box
    s1 = _session1(tmp_path, tmp)
    files = _module("make_w9_paper_material").build(
        json.loads(json.dumps(_adj(tmp, reports, s1), default=str)), "0" * 64)
    tex = sorted(n for n in files if n.endswith(".tex"))
    for n in tex:
        (tmp_path / n).write_text(files[n], encoding="utf-8")
    (tmp_path / "doc.tex").write_text(
        "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n\\begin{document}\n"
        + "".join(f"\\input{{{n}}}\n\\clearpage\n" for n in tex) + "\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"],
                       cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-3000:]


def _return2(tmp, where, arms=("c1_n1024", "c1_n25600", "b_n1024", "s_n1024", "c1_n4096"),
             s1=None):
    """A session-2 return as RUNBOOK 2d writes it (plan, status, provenance
    with the reports' SHA-256, one directory per arm)."""
    s2 = where / "s2"
    lines = []
    for d in arms:
        (s2 / d).mkdir(parents=True)
        shutil.copyfile(tmp / "runs/w9" / d / "report.json", s2 / d / "report.json")
        lines.append(f"{hashlib.sha256((s2 / d / 'report.json').read_bytes()).hexdigest()}  "
                     f"runs/w9/{d}/report.json")
    dec_sha = hashlib.sha256((s1 / "decisions.json").read_bytes()).hexdigest() if s1 else None
    (s2 / "plan.json").write_text(json.dumps({"decisions_sha256": dec_sha, "arms": []}))
    (s2 / "status.txt").write_text("".join(f"{d}.log start 2026-10-03 01:00:00\n"
                                           f"{d}.log exit=0 2026-10-03 02:00:00\n" for d in arms))
    (s2 / "provenance.txt").write_text("HEAD x\n--- states ---\n" + "\n".join(lines) + "\n")
    return s2


def test_adjudication_cli_from_the_two_returns(box, tmp_path, monkeypatch, capsys):
    """scripts/adjudicate_w9.py on returns laid out as RUNBOOK_W9 Sec. 1f and
    2d write them (E1's stamped hash and the arms' identities set to the
    miniature's)."""
    tmp, _, reports = box
    from fejepa.analysis import adjudicate_w9 as adj
    from fejepa.report import config_sha256

    e1p = tmp / "records/wp8/e1/e1_2d_base/report.json"
    cli = _module("adjudicate_w9")
    monkeypatch.setattr(cli, "E1_BASE_CONFIG_SHA256",
                        config_sha256(json.loads(e1p.read_text())["config"]))
    monkeypatch.setattr(adj, "DEFAULT_SPECS", SPECS)
    describe = reports["nmax"]["provenance"]["git"]

    def run(s1, s2, out, prereg=tmp / "PREREG_W9.md"):
        monkeypatch.setattr(sys, "argv", [
            "adjudicate_w9.py", "--session1", str(s1), "--session2", str(s2), "--e1-report",
            str(e1p), "--prereg", str(prereg), "--expected-git", describe, "--out", str(out)])
        cli.main()
        return json.loads(Path(out).read_text())

    def s1_return(where, **kw):
        s1 = _session1(where, tmp, **kw)
        (s1 / "ood2d" / "R").mkdir(parents=True)
        shutil.copyfile(tmp / "runs/w9/ood2d/R/manifest.json", s1 / "ood2d/R/manifest.json")
        return s1

    s1 = s1_return(tmp_path)
    res = run(s1, _return2(tmp, tmp_path, s1=s1), tmp_path / "v.json")
    assert res["H1"]["verdict"] and res["H2"]["verdict"] and res["refused_reports"] == {}
    assert set(res["inputs"]) >= {"n1024", "nmax", "b1024", "s", "n4096", "decisions",
                                  "session1_readings", "ood2d_record", "r_manifest", "plan.json",
                                  "status.txt", "provenance.txt", "e1_report", "prereg"}
    assert set(res["adjudicator"]["sha256"]) >= {"src/fejepa/analysis/adjudicate_w9.py",
                                                 "scripts/w9_session1_decisions.py"}
    assert '"H1"' in capsys.readouterr().out
    s1t = s1_return(tmp_path / "t", timing=False)                   # rule 3 undecided
    assert "profile_2d_w9.json" not in json.loads((s1t / "decisions.json").read_text())[
        "inputs_sha256"]
    res = run(s1t, _return2(tmp, tmp_path / "t", s1=s1t), tmp_path / "t.json")
    assert "rule 3 undecided: checkpointing kept on (memory and time only)" in res["deviations"]
    s2 = _return2(tmp, tmp_path / "x", s1=s1)
    shutil.rmtree(s2 / "c1_n4096")
    with pytest.raises(SystemExit, match="lists c1_n4096's report, the return lacks it"):
        run(s1, s2, tmp_path / "x.json")
    s2 = _return2(tmp, tmp_path / "y", s1=s1)
    (s2 / "b_n1024" / "report.json").write_text((s2 / "b_n1024" / "report.json").read_text() + " ")
    with pytest.raises(SystemExit, match="not the report the box hashed"):
        run(s1, s2, tmp_path / "y.json")
    s2 = _return2(tmp, tmp_path / "z", s1=s1)
    shutil.rmtree(s2 / "c1_n4096")
    (s2 / "provenance.txt").write_text("HEAD x\n")
    with pytest.raises(SystemExit, match="finished, but the return holds no report"):
        run(s1, s2, tmp_path / "z.json")
    s2 = _return2(tmp, tmp_path / "w", s1=s1)
    (s2 / "plan.json").unlink()
    with pytest.raises(SystemExit, match="lacks \\['plan.json'\\]"):
        run(s1, s2, tmp_path / "w.json")
    (tmp_path / "open.md").write_text("CONFIG_SHA256[w9_c1_n1024] = <fill before tagging>\n")
    with pytest.raises(SystemExit, match="not stamped"):
        run(s1, _return2(tmp, tmp_path / "o", s1=s1), tmp_path / "o.json",
            prereg=tmp_path / "open.md")


# ------------------------------------------------------------ the plan ----

def _plan(tmp, s1, out, *extra):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "w9_session2_plan.py"),
                           "--session1", str(s1), "--out-dir", str(out), "--repo", str(tmp),
                           "--allow-cpu", "--expected-git", _describe(), *extra],
                          capture_output=True, text=True, cwd=str(tmp))


def test_plan_selects_orders_and_preflights(box, tmp_path):
    tmp, _, _ = box
    mk = _module("w9_session2_plan")
    s1 = _session1(tmp_path, tmp, pool="25600s", ckpt_off=True)
    problems, dec, facts = mk.gate(s1, tmp, gpu_check=False, expected_git=_describe())
    assert problems == [] and facts["reproduction_max_rel_dev"] == 1e-9
    arms, notes = mk.select(dec, facts["memory"])
    assert [a[1] for a in arms] == ["w9_c1_n1024", "w9_c1_n25600", "w9_b_n1024", "w9_s_n1024",
                                    "w9_c1_n4096"]
    assert arms[0][2] == [] and arms[1][2] == ["--workers", "1", "--activation-checkpointing", "off"]
    assert arms[2][2] == ["--workers", "3", "--activation-checkpointing", "off"] and notes == []
    prev = os.getcwd()
    os.chdir(tmp)                       # the configurations' paths are the box's, relative
    try:
        assert mk.preflight(arms, tmp / "configs") == []        # stamped, holdouts, E1 states
    finally:
        os.chdir(prev)
    out = tmp_path / "s2"
    r = _plan(tmp, s1, out)
    assert r.returncode == 0 and "GO" in r.stdout, r.stdout + r.stderr
    assert "S admitted: yes | N_max 25,600 with 1 worker(s) at a time | checkpointing off" in r.stdout
    sh = (out / "commands.sh").read_text()
    assert f"cd {tmp}" in sh and "flock -n 9" in sh and "torch.cuda" not in sh   # --allow-cpu
    assert "STOP before $name: the code is" in sh and _describe() in sh
    order = [sh.index(f"arm {a} ") for a in ("c1_n1024", "c1_n25600", "b_n1024", "s_n1024",
                                             "c1_n4096")]
    assert order == sorted(order)
    assert subprocess.run(["bash", "-n", str(out / "commands.sh")]).returncode == 0
    plan = json.loads((out / "plan.json").read_text())
    assert plan["decisions_sha256"] == hashlib.sha256((s1 / "decisions.json").read_bytes()).hexdigest()
    s1b = _session1(tmp_path / "b", tmp, s_enters=False, pool="12800p")
    arms, _ = mk.select(mk.gate(s1b, tmp, gpu_check=False)[1])
    assert [a[1] for a in arms] == ["w9_c1_n1024", "w9_c1_n12800", "w9_b_n1024", "w9_c1_n4096"]
    assert arms[1][2] == ["--workers", "3", "--activation-checkpointing", "on"]


def test_plan_keeps_checkpointing_where_memory_would_not_fit(box, tmp_path):
    tmp, _, _ = box
    mk = _module("w9_session2_plan")
    s1 = _session1(tmp_path, tmp, pool="25600s", ckpt_off=True)
    dec = json.loads((s1 / "decisions.json").read_text())
    mem = json.loads((s1 / "c0_memory.json").read_text())
    mem["marks"][1]["rss"] = 40e9                                   # 4,096 x 3 > 80 % of 100 GB
    arms, notes = mk.select(dec, mem)
    flags = {a[1]: a[2] for a in arms}
    assert flags["w9_c1_n4096"][-1] == "on" and flags["w9_c1_n25600"][-1] == "off"
    assert any("w9_c1_n4096: checkpointing kept on" in n for n in notes)
    s1t = _session1(tmp_path / "t", tmp, timing=False)              # rule 3 undecided: on
    problems, dec, facts = mk.gate(s1t, tmp, gpu_check=False)
    arms, notes = mk.select(dec, facts["memory"])
    assert problems == [] and all(a[2][-1] == "on" for a in arms[1:])
    assert notes == ["rule 3 undecided: checkpointing on (memory and time only)"]


def test_plan_gate_stops(box, tmp_path):
    tmp, _, _ = box
    mk = _module("w9_session2_plan")
    s1 = _session1(tmp_path, tmp)
    (s1 / "status.txt").write_text("ood2d.log exit=0\nc0_val.log exit=1\n")
    problems = mk.gate(s1, tmp, gpu_check=False)[0]
    assert any("c0_amp2d.log: exit missing" in p for p in problems)
    assert any("c0_val.log: exit 1" in p for p in problems)
    assert any("no pool fits" in p for p in
               mk.gate(_session1(tmp_path / "x", tmp, pool="none"), tmp, gpu_check=False)[0])
    assert any("not reproduced" in p for p in
               mk.gate(_session1(tmp_path / "y", tmp, repro=3e-3), tmp, gpu_check=False)[0])
    s1 = _session1(tmp_path / "n", tmp)
    v = json.loads((s1 / "c0_val.json").read_text())
    v["summary"]["reproduction"]["egap"] = {"s0": float("nan"), "s1": 0.0}
    (s1 / "c0_val.json").write_text(json.dumps(v))
    assert any("not reproduced" in p for p in mk.gate(s1, tmp, gpu_check=False)[0])
    s1 = _session1(tmp_path / "z", tmp)
    (s1 / "c0_memory.json").write_text((s1 / "c0_memory.json").read_text() + " ")
    assert any("another c0_memory.json" in p for p in mk.gate(s1, tmp, gpu_check=False)[0])
    s1 = _session1(tmp_path / "r", tmp)
    d = json.loads((s1 / "decisions.json").read_text())
    d["rules_sha256"] = "0" * 64
    (s1 / "decisions.json").write_text(json.dumps(d))
    assert any("frozen rules" in p for p in mk.gate(s1, tmp, gpu_check=False)[0])
    s1 = _session1(tmp_path / "e", tmp)
    d = json.loads((s1 / "decisions.json").read_text())
    d["decisions"]["S_enters"]["value"] = False                    # edited by hand
    (s1 / "decisions.json").write_text(json.dumps(d))
    assert any("rule S_enters recomputed" in p for p in mk.gate(s1, tmp, gpu_check=False)[0])
    s1 = _session1(tmp_path / "g", tmp)
    problems = mk.gate(s1, tmp, gpu_check=False, expected_git="prereg-w9-not-this-commit")[0]
    assert any("decisions.json was written on" in p for p in problems)
    assert any("this checkout is" in p for p in problems)
    s1 = _session1(tmp_path / "a", tmp)
    a = json.loads((s1 / "c0_amp2d.json").read_text())
    a["F5"]["manifest_sha256"] = "0" * 64
    (s1 / "c0_amp2d.json").write_text(json.dumps(a))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "w9_session1_decisions.py"),
                    "--dir", str(s1), "--out", str(s1 / "decisions.json")],
                   capture_output=True, text=True, cwd=str(tmp))
    assert any("amplitude reading used another F5" in p for p in mk.gate(s1, tmp, gpu_check=False)[0])
    s1 = _session1(tmp_path / "w", tmp)
    rec = json.loads((s1 / "ood2d.json").read_text())
    rec["families"]["F2"]["status"] = "failed"
    rec["families"]["F4"]["manifest_sha256"] = "0" * 64
    rec["families"]["IB"]["n_instances"] = 512
    (s1 / "ood2d.json").write_text(json.dumps(rec))
    problems = mk.gate(s1, tmp, gpu_check=False)[0]
    assert any("incomplete: ['F2']" in p for p in problems)
    assert any("ood2d/F4: the manifest on disk" in p for p in problems)
    assert "IB: 512 instances from seed 91007; PREREG_W9 Sec. 3 fixes 2048 from seed 91007" \
        in problems
    out = tmp_path / "s2w"
    out.mkdir()
    (out / "commands.sh").write_text("stale")
    r = _plan(tmp, s1, out)
    assert r.returncode != 0 and "STOP" in r.stdout
    assert not (out / "commands.sh").exists()                      # no stale script survives


def test_command_script_restarts_and_keeps_every_attempt(box, tmp_path):
    """The generated script on the miniature: first run, a complete re-run
    (all skipped), a truncated report moved aside and the arm restarted."""
    tmp, _, _ = box
    work = tmp_path / "box"
    shutil.copytree(tmp, work, ignore=shutil.ignore_patterns("w9", "b_n1024", "s_n1024",
                                                             "c1_n4096", "c1_n12800"))
    for d in ("c1_n1024", "c1_n25600"):
        shutil.rmtree(work / "runs/w9" / d, ignore_errors=True)
    shutil.copytree(tmp / "runs/w9/ood2d", work / "runs/w9/ood2d")
    mk = _module("w9_session2_plan")
    s1 = _session1(tmp_path, work, s_enters=False, pool="25600p")
    arms = [a for a in mk.select(json.loads((s1 / "decisions.json").read_text()))[0][:2]]
    out = work / "runs/w9/session2"
    (out).mkdir(parents=True)
    (out / "commands.sh").write_text(mk.script(arms, "x" * 64, str(out), str(work),
                                               gpu_check=False))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    run = lambda: subprocess.run(["bash", str(out / "commands.sh")], capture_output=True,  # noqa: E731
                                 text=True, env=env)
    r1 = run()
    st = (out / "status.txt").read_text()
    assert st.count("exit=0") == 2, r1.stdout[-3000:] + r1.stderr[-3000:]
    r2 = run()
    assert (out / "status.txt").read_text().count("report exists, skipped") == 2, r2.stdout[-2000:]
    rep = work / "runs/w9/c1_n25600/report.json"
    rep.write_text(rep.read_text()[:500])                           # truncated
    run()
    st = (out / "status.txt").read_text()
    assert "c1_n25600: unreadable report moved aside" in st
    assert "c1_n25600: restarting from its states" in st
    assert json.loads(rep.read_text())["d9_reuse_states"] is True
    assert list(out.glob("c1_n25600.log.*"))                         # the earlier attempt's log
    assert list((work / "runs/w9/c1_n25600").glob("report.json.unreadable.*"))


def test_command_script_stops_on_another_commit(box, tmp_path):
    tmp, _, _ = box
    mk = _module("w9_session2_plan")
    out = tmp_path / "s2"
    out.mkdir()
    (out / "commands.sh").write_text(mk.script(
        [("c1_n1024", "w9_c1_n1024", [])], "x" * 64, str(out), str(tmp), gpu_check=False,
        expected_git="prereg-w9-not-this"))
    r = subprocess.run(["bash", str(out / "commands.sh")], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
    st = (out / "status.txt").read_text()
    assert "STOP before c1_n1024: the code is" in st and "start" not in st, r.stdout + r.stderr
