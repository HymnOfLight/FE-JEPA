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
    m = re.search(r"\*\*Status:\*\* r1, stamped (.+?) before the run:", text)
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
    p.write_text(_draft_text().replace("r1 DRAFT", "r0 DRAFT"), encoding="utf-8")
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
    assert (f"{hrs['0.5']:.1f} / {hrs['1.5']:.1f} / {hrs['2']:.1f} h at a half / one and a half "
            "/ twice as long") in text


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
