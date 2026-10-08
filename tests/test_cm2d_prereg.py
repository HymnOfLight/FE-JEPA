"""cmame-paper Stage 2: CM2D's configuration is the generator's and differs
from E1's stamped base only in the keys PREREG_CM2D names; the stamp script
accepts only the approved draft and verifies what it writes; the committed
PREREG_CM2D.md is either that draft or a verified stamp of the committed
configuration; the simulation record is the script's output, and the
operating characteristics the pre-registration quotes are the record's."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "PREREG_CM2D.md"
CONFIG = ROOT / "configs" / "cm2d_v1.json"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mk = _script("make_cm2d_config")
st = _script("stamp_prereg_cm2d")


def _paths(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.add(path + (k,))
            else:
                out |= _paths(a[k], b[k], path + (k,))
        return out
    return set() if a == b else {path}


def test_the_configuration_is_the_generators_and_differs_only_where_named():
    base = mk.load_base()
    text = mk.render(mk.cm2d_config(base))
    assert CONFIG.read_text() == text
    cfg = json.loads(text)
    assert _paths(cfg, base) == mk.CHANGED
    e8 = cfg["experiments"]["e8"]
    assert (e8["ar_only"], e8["include_anchor"], e8["include_ar_ft"], e8["include_knorm"]) == \
        (False, False, False, True)
    assert e8["mgn_budgets"] == [64, 1024] and e8["include_mgn"] is True
    assert e8["reuse_from"] == {"report": "records/wp8/e1/e1_2d_base/report.json",
                                "states_dir": "runs/e1_2d_base/e8_states",
                                "supervised_grid": True}
    assert cfg["prereg_guard"] is True and cfg["prereg_file"] == "PREREG_CM2D.md"
    assert cfg["out"] == "runs/cm2d/report.json"
    # E1's grid otherwise: budgets, epochs, seeds, pool, workers, corpus, split
    assert e8["budgets"] == [16, 64, 256, 1024] and e8["sup_epochs"] == 200
    assert e8["seeds"] == 3 and e8["pool_sizes"] == [1024] and cfg["workers"] == 3
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_cm2d_config.py"),
                          "--check"], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr


def test_the_configuration_passes_the_reuse_comparison_with_e1():
    sys.path.insert(0, str(ROOT / "src"))
    from fejepa.experiments.cost import count_steps
    from fejepa.experiments.w9_eval import _strip

    cfg = json.loads(CONFIG.read_text())
    e1 = json.loads((ROOT / "records/wp8/e1/e1_2d_base/report.json").read_text())
    assert _strip(cfg, True) == _strip(e1["config"], True)
    assert _strip(cfg) != _strip(e1["config"])          # only as a supervised grid
    assert count_steps(cfg)["e8"] == 3 * 200 * (2 * (16 + 64 + 256 + 1024) + 64 + 1024) \
        == 2_284_800


def _draft_text() -> str:
    """PREREG_CM2D.md as its approved draft (a stamped file un-stamped)."""
    text = PREREG.read_text(encoding="utf-8")
    m = re.search(r"\*\*Status:\*\* r3, stamped (.+?) before the run:", text)
    if m:
        text = text.replace(st.stamped_status(m.group(1)), st.DRAFT_STATUS)
    text = re.sub(r"^CONFIG_SHA256\[cm2d_v1\] = [0-9a-f]{64}$",
                  "CONFIG_SHA256[cm2d_v1] = <fill before tagging>", text, flags=re.M)
    return re.sub(r"^PREREG_CM2D_SHA256 = [0-9a-f]{64}$", st.FOOTER_OPEN, text, flags=re.M)


def test_the_committed_file_is_the_draft_or_a_verified_stamp():
    sys.path.insert(0, str(ROOT / "src"))
    from fejepa.report import config_sha256, read_prereg_entries

    text = PREREG.read_text(encoding="utf-8")
    entries = read_prereg_entries(PREREG)
    assert [lab for lab, _ in entries] == ["cm2d_v1"]
    if entries[0][1] == "<fill before tagging>":
        assert text.count(st.DRAFT_STATUS) == 1 and text.rstrip("\n").endswith(st.FOOTER_OPEN)
    else:
        assert entries[0][1] == config_sha256(json.loads(CONFIG.read_text()))
        assert st.self_hash_ok(text) and st.DRAFT_STATUS not in text
        draft = _draft_text()                     # un-stamped: the draft the stamp accepted
        assert draft != text and draft.count(st.DRAFT_STATUS) == 1
        assert draft.rstrip("\n").endswith(st.FOOTER_OPEN) and "<fill before tagging>" in draft


def test_the_stamp_fills_the_draft_and_verifies(tmp_path):
    sys.path.insert(0, str(ROOT / "src"))
    from fejepa.report import config_sha256, read_prereg_entries, verify_prereg

    p = tmp_path / "PREREG_CM2D.md"
    p.write_text(_draft_text(), encoding="utf-8")
    res = st.stamp(p, CONFIG, "8 October 2026")
    cfg = json.loads(CONFIG.read_text())
    assert res["config_sha256"]["cm2d_v1"] == config_sha256(cfg)
    text = p.read_text(encoding="utf-8")
    assert dict(read_prereg_entries(p))["cm2d_v1"] == config_sha256(cfg)
    assert st.self_hash_ok(text) and st.stamped_status("8 October 2026") in text
    assert verify_prereg(cfg, p, label="cm2d_v1") == config_sha256(cfg)
    # everything but the three filled lines is the draft's
    a, b = _draft_text().splitlines(), text.splitlines()
    assert len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 3
    with pytest.raises(SystemExit, match="already stamped"):
        st.stamp(p, CONFIG, "8 October 2026")


def test_the_stamp_refuses(tmp_path):
    p = tmp_path / "PREREG_CM2D.md"
    for bad_date in ("08 October 2026", "October 8, 2026", "8 Oct 2026"):
        p.write_text(_draft_text(), encoding="utf-8")
        with pytest.raises(SystemExit, match="--date"):
            st.stamp(p, CONFIG, bad_date)
    r2 = _draft_text().replace("r3 DRAFT (8 October 2026)", "r2 DRAFT (7 October 2026)")
    p.write_text(r2, encoding="utf-8")                      # the superseded draft
    with pytest.raises(SystemExit, match="status line"):
        st.stamp(p, CONFIG, "8 October 2026")
    p.write_text(_draft_text() + "\nmore\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="last line"):
        st.stamp(p, CONFIG, "8 October 2026")
    cfg = tmp_path / "cm2d_v1.json"
    other = json.loads(CONFIG.read_text())
    other["experiments"]["e8"]["sup_epochs"] = 100
    cfg.write_text(mk.render(other))
    p.write_text(_draft_text(), encoding="utf-8")
    with pytest.raises(SystemExit, match="generator"):
        st.stamp(p, cfg, "8 October 2026")
    assert p.read_text(encoding="utf-8") == _draft_text()     # nothing written on refusal
    p.write_text(_draft_text().replace("CONFIG_SHA256[cm2d_v1]", "CONFIG_SHA256[cm2d_v2]"),
                 encoding="utf-8")
    with pytest.raises(SystemExit, match="CONFIG_SHA256 lines"):
        st.stamp(p, CONFIG, "8 October 2026")
    # a changed file fails the self-hash
    q = tmp_path / "q.md"
    q.write_text(_draft_text(), encoding="utf-8")
    st.stamp(q, CONFIG, "8 October 2026")
    assert not st.self_hash_ok(q.read_text(encoding="utf-8").replace("tau", "t"))


def test_the_simulation_record_is_the_scripts_output():
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "cm2d_sims.py"), "--check",
                          str(ROOT / "records" / "cmame" / "cm2d_sims.json")],
                         capture_output=True, text=True, cwd=ROOT, timeout=300)
    assert res.returncode == 0, res.stderr[-2000:]


def test_the_quoted_operating_characteristics_are_the_records():
    cells = json.loads((ROOT / "records/cmame/cm2d_sims.json").read_text())["cells"]
    text = " ".join(PREREG.read_text(encoding="utf-8").split())

    def pct(x, digits):
        return f"{100 * x:.{digits}f}%"

    eq = [f"c_ref={c:.2f},c_new={c:.2f}" for c in (0.05, 0.10, 0.20, 0.30)]
    lower0 = " / ".join(pct(cells[k]["r=1"]["lower"], 1) for k in eq)
    assert f"in {lower0} of trials at c = 5% / 10% / 20% / 30%" in text
    worse0 = " / ".join(pct(cells[k]["r=1"]["worse"], 1) for k in eq)
    assert f'and "worse beyond the guard" in {worse0};' in text
    a, b = cells["c_ref=0.05,c_new=0.20"]["r=1"], cells["c_ref=0.20,c_new=0.05"]["r=1"]
    lo, hi = sorted([a["lower"], b["worse"]]), sorted([a["worse"], b["lower"]])
    assert (f"the noisier arm reads lower beyond the guard in {100 * lo[0]:.1f}-"
            f"{100 * lo[1]:.1f}% of trials and higher in {100 * hi[0]:.1f}-"
            f"{100 * hi[1]:.1f}%") in text
    assert all(round(100 * (x["lower"] + x["worse"])) == 15 for x in (a, b))
    assert "(about 15% with spreads of 5% and 20%)" in text
    for r, digits in (("0.8", (1, 0, 0, 0)), ("0.7", (0, 0, 0, 0)), ("0.5", (0, 0, 0, 0))):
        quoted = " / ".join(pct(cells[k][f"r={r}"]["lower"], d) for k, d in zip(eq, digits))
        assert f"{r} in {quoted}" in text or f"{r} is detected in {quoted}" in text, r
    two = " / ".join(pct(cells[k]["r=1"]["lower"] + cells[k]["r=1"]["worse"], 1) for k in eq)
    assert f"occurs in {two} of trials" in text
    rec = json.loads((ROOT / "records/cmame/cm2d_sims.json").read_text())
    jr = rec["july_reference"]
    ks = [f"c_ref=0.09,c_new={c:.2f}" for c in (0.05, 0.09, 0.20, 0.30, 0.50)]
    assert ("reads lower beyond the guard in "
            + " / ".join(pct(jr[k]["r=1"]["lower"], 1) for k in ks)
            + " of trials at new-arm spreads of 5% / 9% / 20% / 30% / 50%") in text
    assert ("a true ratio of 0.8 is detected in "
            + " / ".join(pct(jr[k]["r=0.8"]["lower"], 0) for k in ks)) in text
    vm = rec["july_reference_vm"]                    # r3: H2b's reference spread, 7.1%
    kv = [f"c_ref=0.071,c_new={c:.2f}" for c in (0.05, 0.09, 0.20, 0.30, 0.50)]
    assert ("with the reference at its spread in the von Mises error (7.1%, H2b's reference), "
            "at the same new-arm spreads, in "
            + " / ".join(pct(vm[k]["r=1"]["lower"], 1) for k in kv) + " and "
            + " / ".join(pct(vm[k]["r=0.8"]["lower"], 0) for k in kv)) in text
    fs = [x[k]["r=1"]["lower"] for x, kk in ((jr, ks), (vm, kv)) for k in kk[2:]]
    assert (round(100 * min(fs)), round(100 * max(fs))) == (8, 16)
    assert "at new-arm spreads of 20-50% H2a's and H2b's false-support rate is 8-16%" in text
    bl = rec["blowups"]["cells"]
    lo, hi = sorted([bl["ref"]["lower"], bl["new"]["worse"]])
    assert f"the other arm reads lower beyond the guard in {100 * lo:.0f}-{100 * hi:.0f}%" in text
    assert all(round(100 * bl["both"][w]) == 5 for w in ("lower", "worse"))
    assert "(in about 5% when both arms have them)" in text
    m = rec["blowups"]["model"]
    assert (m["low"], m["high"]) == (20.0, 40.0) and "at 20-40 times an arm's level" in text
    assert f"by {(m['low'] - 1) / 256 * 100:.0f}-{(m['high'] - 1) / 256 * 100:.0f}%" in text
    hrs = rec["schedule"]["hours_by_mgn_step_factor"]
    assert f"gives {hrs['1']:.1f} h if the graph network's step is as long" in text
    assert (f"{hrs['0.5']:.1f} / {hrs['1.5']:.1f} / {hrs['2']:.1f} / {hrs['2.5']:.1f} / "
            f"{hrs['3']:.1f} h at a half / one and a half / two / two and a half / three times "
            "as long") in text
    assert "the instance's balance must cover about 30 h" in text and hrs["3"] < 30 - 0.5


def test_the_runs_on_the_box_before_the_stamp_are_disclosed_as_recorded():
    """r3: Sec. 4 quotes the readiness check B0 and Sec. A of 8 October 2026 from
    their records, and Sec. 7 the two hosts from theirs."""
    text = " ".join(PREREG.read_text(encoding="utf-8").split())
    log = (ROOT / "records/cmame/cm2d_ready.log").read_text()
    assert "smoke: relative energy gap 0.995 after one epoch on cuda" in log
    assert "reproduction: largest relative deviation 0 over 3 states on cuda" in log
    assert ("the readiness check RUNBOOK_CMAME B0 on the box (`records/cmame/cm2d_ready.log`) "
            "trained the run's transformer with L_K for one epoch of two steps on two pool "
            "instances (learning rate 1e-4; relative energy gap 0.995 on the first of them; "
            "nothing kept)") in text
    assert "reproducing E1's values (largest relative deviation 0)" in text
    prc = _script("cm2d_precheck")
    import inspect
    smoke = inspect.getsource(prc.smoke_check)
    assert "split.pool_files[:2]" in smoke and "archs[:1]" in smoke
    assert "epochs=1, lr=1e-4, loss=\"knorm\"" in smoke and '"model": cfg["model"]' in smoke
    timing = json.loads((ROOT / "records/cmame/timing/timing_2d.json").read_text())
    assert timing["seed"] == 0 and timing["sets"]["val"]["n"] == 32
    assert ("RUNBOOK_CMAME Sec. A timed E1's label-free state of seed 0 on 32 validation "
            "instances (`records/cmame/timing/`)") in text
    prof = json.loads((ROOT / "records/wp8/posthoc/profile_2d_head.json").read_text())
    assert "Platinum 8470Q" in prof["cpu"] and prof["cpus"]["cgroup_cpu_max"] == 25.0
    machine = (ROOT / "records/cmame/timing/machine.txt").read_text()
    assert "Xeon(R) Gold 6459C" in machine and "nproc 16" in machine
    assert "1600000 100000" in machine                      # cgroup cpu.max: 16 CPUs
    assert prof["driver"] in machine and prof["torch"] == timing["machine"]["torch"]
    assert ("That profile ran on the host the instance had then (a Xeon Platinum 8470Q with 25 "
            "CPUs); on 8 October 2026 the instance ran on another host (a Xeon Gold 6459C with "
            "16 CPUs) with the same GPU model, driver and torch") in text


def test_the_prereg_names_the_adjudicators_constants():
    adj = _script("adjudicate_cm2d")
    text = " ".join(PREREG.read_text(encoding="utf-8").split())
    assert adj.REPRO_MAX == 1e-4 and "at most 1e-4" in text
    assert adj.REPRO_METRICS == ("energy_gap_rel", "disp_rel_l2")
    assert "values of the relative energy gap and the displacement error" in text
    assert adj.TAG == "prereg-cm2d" and adj.LABEL == "cm2d_v1"
    prc = _script("cm2d_precheck")
    assert prc.TAG == adj.TAG and (prc.FREE_GB, prc.FREE_GB_RESTART) == (7.0, 5.0)
    assert "7 GB free before the run (5 GB before a restart)" in text
    assert "(no report, `e8_states/`, run log or status file)" in text
    assert prc.ATTEMPT_TRACES == ("report.json", "e8_states", "run.log", "status.txt")


FUNCCHECK = ROOT / "records" / "cmame" / "cm2d_funccheck.json"
STAGE2_SRC = "2da09d0593669f50149f4fb546c01861ac0d5149"
"""The `src` tree of cmame-paper Stage 2 (records/cmame/README.md)."""


def test_the_functional_check_is_disclosed_as_recorded():
    """Sec. 4 quotes the functional check run before the stamp from its record,
    the script's output on the Stage 2 code."""
    import hashlib

    rec = json.loads(FUNCCHECK.read_text())
    script = ROOT / "scripts" / "cm2d_funccheck.py"
    assert rec["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert rec["src_tree"] == STAGE2_SRC
    s, c = rec["settings"], rec["corpus"]
    assert (s["model"]["dim"], s["model"]["depth"], s["epochs"], s["seeds"]) == (32, 2, 25, [0, 1])
    assert (c["n_train"], c["n_val"], c["seed"], s["device"], s["threads"]) == \
        (48, 24, 777, "cpu", 1)
    # float32 arithmetic of L_K against Lemma 1 in float64: round-off, amplified by the
    # stiffness spread of these meshes (1.1e-4 at most, median 7e-6)
    lem = rec["knorm_loss_against_lemma_1"]
    assert lem["steps"] == 200 and lem["max_rel_dev"] < 1e-3 and lem["median_rel_dev"] < 1e-5

    def pair(kind, metric):
        return " / ".join(f"{rec['runs'][kind][str(k)]['val'][metric]:#.3g}" for k in s["seeds"])

    text = " ".join(PREREG.read_text(encoding="utf-8").split())
    assert "(`scripts/cm2d_funccheck.py`, `records/cmame/cm2d_funccheck.json`)" in text
    assert ("trained a transformer of width 32 and depth 2 for 25 epochs on 48 instances "
            "drawn by the 2D generator with its own seed (not E1's corpus)") in text
    assert "seeds 0 and 1, and evaluated it on 24 others" in text
    assert (f"relative energy gap {pair('knorm', 'energy_gap_rel')} with L_K against "
            f"{pair('disp', 'energy_gap_rel')} with L_D, von Mises error "
            f"{pair('knorm', 'vm_rel_l2')} against {pair('disp', 'vm_rel_l2')} (the label-free "
            f"objective: {pair('ar', 'energy_gap_rel')} and {pair('ar', 'vm_rel_l2')})") in text
    for k in s["seeds"]:                       # both reductions, on every seed
        lk, ld = rec["runs"]["knorm"][str(k)]["val"], rec["runs"]["disp"][str(k)]["val"]
        assert lk["energy_gap_rel"] < ld["energy_gap_rel"] and lk["vm_rel_l2"] < ld["vm_rel_l2"]
    assert "Both reductions that H2a and H2b test were thus seen before the stamp" in text
    assert "The effects of L_K have never been measured" not in text
    assert "No supervised network has been trained with the stiffness norm" not in text
    assert "(since then, only in the small functional check of Sec. 4)" in text
    assert "its adjudication read H2a and H2b as SUPPORTED, H1 as NOT SUPPORTED" in text


def test_the_functional_check_script_runs(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("gmsh")
    out = tmp_path / "fc.json"
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "cm2d_funccheck.py"), "--out",
                          str(out), "--n-train", "2", "--n-val", "1", "--epochs", "1",
                          "--seeds", "0"], capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    rec = json.loads(out.read_text())
    assert set(rec["runs"]) == {"disp", "knorm", "ar"} and rec["corpus"]["n"] == 3
    assert rec["knorm_loss_against_lemma_1"]["steps"] == 2
    assert all(rec["runs"][k]["0"]["grad_norm_before_clip"]["steps"] == 2 for k in rec["runs"])
