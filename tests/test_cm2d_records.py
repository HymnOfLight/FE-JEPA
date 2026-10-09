"""cmame-paper Stage 6: the return of PREREG_CM2D's run (RUNBOOK_CMAME Sec. B,
8-9 October 2026, copied byte for byte into records/cmame/cm2d/return/) and
its verdict (records/cmame/cm2d/verdict.json) are complete and tied to what
they claim: one attempt on the stamped commit and tree, exit 0; the report is
the stamped configuration's, on E1's corpus and seeds; the verdict was made
from these files; and its verdicts and the secondary counts follow from the
report's per-instance arrays by PREREG_CM2D Sec. 4's rule, recomputed here."""

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CM = ROOT / "records" / "cmame"
RET = CM / "cm2d" / "return"
VERDICT = CM / "cm2d" / "verdict.json"
PREREG = ROOT / "PREREG_CM2D.md"
E1 = ROOT / "records" / "wp8" / "e1" / "e1_2d_base" / "report.json"
JULY = ROOT / "records" / "phase1" / "report_rec8_v2.json"
HEAD = "094c8040c828706996173afb5f7a60bd2f5337fd"        # tag prereg-cm2d (Stage 5)
TREE = "cd3ec0c5baada05cf86eb6d12d64153d28930e10"
FILES = ("RESULTS.md", "figure1_energy_gap.png", "precheck.json.used",
         "precheck.log.20261009-142644", "provenance.txt", "pytest.log", "report.json",
         "run.log", "status.txt")
METRICS = ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err", "crit_recall")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json(p) -> dict:
    return json.loads(Path(p).read_text())


def _section(text: str, title: str) -> str:
    m = re.search(r"^## " + re.escape(title) + r".*?(?=^## |\Z)", text, re.M | re.S)
    assert m, title
    return m.group(0)


def _cells(rep: dict) -> dict:
    return rep["results"]["e8"]["metrics"]["cells"]


def _arrays(rep: dict, row: str, budget: str, metric: str) -> np.ndarray:
    """(seeds, 256) per-instance values of one cell."""
    return np.array([s["per_instance"][metric] for s in _cells(rep)[row][budget]["per_seed_eval"]],
                    dtype=float)


def test_the_files_are_the_ones_the_readme_lists():
    sec = _section((CM / "README.md").read_text(encoding="utf-8"), "CM2D: the run of")
    listed = dict(re.findall(r"^\| `return/([^`]+)` \| `([0-9a-f]{64})` \|$", sec, re.M))
    present = sorted(p.name for p in RET.iterdir() if p.is_file())
    assert present == sorted(FILES) == sorted(listed)
    for name in present:
        assert _sha(RET / name) == listed[name], name
    m = re.search(r"`cm2d/verdict.json` \(SHA-256\s+`([0-9a-f]{64})`\)", sec)
    assert m and _sha(VERDICT) == m.group(1)


def test_one_attempt_on_the_stamped_commit():
    status = [x for x in (RET / "status.txt").read_text().splitlines() if x.strip()]
    assert len(status) == 2 and re.fullmatch(r"run\.log start 2026-10-08T\d\d:\d\d:\d\dZ", status[0])
    assert status[1] == "run.log exit=0"
    prov = (RET / "provenance.txt").read_text().splitlines()
    assert prov[:4] == [f"HEAD {HEAD}", f"tree {TREE}", "describe prereg-cm2d",
                        "describe-all prereg-cm2d"]                      # not -dirty
    text = "\n".join(prov)
    assert f"{_sha(RET / 'report.json')}  runs/cm2d/report.json" in text
    assert "torch 2.12.1+cu130 True" in text and "NVIDIA GeForce RTX 5090" in text
    assert re.search(r"^oom_kill 0$", text, re.M) and re.search(r"^oom 0$", text, re.M)
    summary = r"^514 passed, 2 deselected(, \d+ warnings?)? in [0-9.]+s \(\d+:\d\d:\d\d\)$"
    assert re.search(summary, (RET / "pytest.log").read_text(), re.M)
    go = _json(RET / "precheck.json.used")
    assert go["go"] is True and go["stop"] == [] and go["pre_stamp"] is False
    assert go["restart"] is False
    assert all(c["ok"] is True for k, c in go["checks"].items() if k != "head")
    assert go["checks"]["head"]["detail"] == f"HEAD {HEAD} tree {TREE}"
    assert go["checks"]["tag"]["detail"] == "prereg-cm2d"
    assert go["checks"]["stamp"]["detail"] == "verified"
    log = [x for x in (RET / "precheck.log.20261009-142644").read_text().splitlines() if x.strip()]
    assert log[-1] == "GO" and f"[precheck] info head: HEAD {HEAD} tree {TREE}" in log
    assert [x for x in log if x.startswith("[precheck] ")] == [
        f"[precheck] {'info' if k == 'head' else 'ok  '} {k}: {c['detail']}"
        for k, c in go["checks"].items()]                      # the log of this GO


def test_the_run_log_is_one_complete_run():
    log = (RET / "run.log").read_text()
    assert log.count("=== fejepa v2 run: configs/cm2d_v1.json | device=cuda | workers=3 |") == 1
    assert log.startswith("[prereg] verified against PREREG_CM2D.md: bf1f1143b397")
    assert "[plan] steps by experiment: {'e8': 2284800, 'total': 2284800}" in log
    assert "[labelling] asis: verified labels on val 256 + pool prefix 1024" in log
    assert "[E8 (AR pretrain)] starting: 3 units" in log
    assert re.search(r"^\[E8 \(supervised grid\)\] 30/30 \(100%\)", log, re.M)
    assert "[E8 (supervised grid)] done in" in log
    assert "'total': 0" in log.splitlines()[-1]                         # the solve ledger
    for bad in ("Traceback", "a worker process died", "out of memory", "[d9]"):
        assert bad not in log, bad


def test_the_report_is_the_stamped_configurations():
    sys.path.insert(0, str(ROOT / "src"))
    from fejepa.report import config_sha256, read_prereg_entries

    rep = _json(RET / "report.json")
    stamped = dict(read_prereg_entries(PREREG))["cm2d_v1"]
    assert rep["provenance"]["config_sha256"] == stamped == config_sha256(rep["config"])
    assert rep["config"] == _json(ROOT / "configs" / "cm2d_v1.json")
    assert rep["provenance"]["git"] == "prereg-cm2d" and rep["provenance"]["seeds"] == [0, 1, 2]
    e1 = _json(E1)
    assert rep["provenance"]["datasets"][0]["manifest_sha256"] == \
        e1["provenance"]["datasets"][0]["manifest_sha256"]
    assert rep["solve_ledger"]["total"] == 0 and rep["d9_reuse_states"] is False
    d9 = rep["results"]["e8"]["metrics"]["d9_restart"]
    assert d9["reuse_states"] is False                          # no restart
    assert d9["sup_units_from_cache"] == [] and d9["units_resumed_from_epoch"] == {}
    # E1's three states, evaluated only, each the one E1's report records
    e1_states = e1["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    for s in ("s0", "s1", "s2"):
        assert d9["ar_states"][s]["reused"] is True
        assert d9["ar_states"][s]["sha256"] == e1_states[s]["sha256"], s
    assert d9["reused_from"]["report_sha256"] == _sha(E1)
    assert d9["reused_from"]["supervised_grid"] is True


def test_the_verdict_was_made_from_these_files():
    v = _json(VERDICT)
    inp = v["inputs"]
    assert inp["report"] == {"path": "records/cmame/cm2d/return/report.json",
                             "sha256": _sha(RET / "report.json")}
    for name in ("status.txt", "provenance.txt"):
        assert inp[name] == {"path": f"records/cmame/cm2d/return/{name}",
                             "sha256": _sha(RET / name)}
    assert inp["run_logs"] == {"run.log": _sha(RET / "run.log")}
    assert inp["recorded_commit"] == {"HEAD": HEAD, "tree": TREE}
    assert inp["e1_report"]["sha256"] == _sha(E1)
    assert inp["prereg"]["sha256"] == _sha(PREREG)
    assert inp["july_report"]["sha256"] == _sha(JULY)
    assert v["adjudicator"]["against_stamp"] == {"ref": "prereg-cm2d", "compared": True,
                                                 "differs": []}
    assert v["deviations"] == [] and v["budget"] == 1024
    assert v["run"]["git"] == "prereg-cm2d" and v["run"]["workers"] == 3


def _guard(new: np.ndarray, base: np.ndarray) -> dict:
    """PREREG_CM2D Sec. 4, written out again: sample standard deviations, three seeds."""
    rel = new.mean() / base.mean() - 1
    se = math.sqrt(new.var(ddof=1) / 3 + base.var(ddof=1) / 3) / base.mean()
    tau = max(0.10, 2 * se)
    return {"rel": rel, "se": se, "tau": tau, "lower": rel < -tau, "worse": rel > tau}


def test_the_verdicts_follow_from_the_report():
    rep, v = _json(RET / "report.json"), _json(VERDICT)
    seeds = {r: {m: _arrays(rep, r, "1024", m).mean(axis=1) for m in METRICS}
             for r in ("ar", "labels", "labels_knorm")}
    plan = {"H1": ("ar", "labels", "energy_gap_rel"),
            "H2a": ("labels_knorm", "labels", "energy_gap_rel"),
            "H2b": ("labels_knorm", "labels", "vm_rel_l2"),
            "H3": ("labels_knorm", "ar", "energy_gap_rel")}
    for h, (a, b, m) in plan.items():
        g, rec = _guard(seeds[a][m], seeds[b][m]), v[h]
        assert rec["metric"] == m, h
        assert np.allclose(rec["new_per_seed"], seeds[a][m], rtol=1e-12, atol=0), h
        assert np.allclose(rec["base_per_seed"], seeds[b][m], rtol=1e-12, atol=0), h
        for k, kk in (("rel_change", "rel"), ("se_rel", "se"), ("threshold", "tau")):
            assert math.isclose(rec[k], g[kk], rel_tol=1e-12), (h, k)
        assert (rec["lower"], rec["worse"]) == (g["lower"], g["worse"]), h
    for h in ("H1", "H2a", "H2b"):
        assert v[h]["verdict"] == ("SUPPORTED" if v[h]["lower"] else "NOT SUPPORTED"), h
    assert (v["H1"]["verdict"], v["H2a"]["verdict"], v["H2b"]["verdict"]) == ("SUPPORTED",) * 3
    assert not v["H3"]["lower"] and not v["H3"]["worse"]
    assert v["H3"]["reading"] == "no difference shown (within the guard)"
    # the label-free row: E1's states reproduce E1's per-instance values exactly
    e1 = _json(E1)
    dev = max(float(np.max(np.abs(_arrays(rep, "ar", "1024", m) - _arrays(e1, "ar", "1024", m))
                           / np.abs(_arrays(e1, "ar", "1024", m))))
              for m in ("energy_gap_rel", "disp_rel_l2"))
    assert dev <= 1e-4 and dev == v["reuse"]["reproduction_max_rel_dev"] == 0.0
    assert v["reuse"]["ok"] is True


def test_the_secondary_counts_follow_from_the_report():
    rep, sec = _json(RET / "report.json"), _json(VERDICT)["secondary"]
    for row, budgets in sec["worse_than_zero"].items():
        for b, c in budgets.items():
            g = _arrays(rep, row, b, "energy_gap_rel")
            assert (c["count"], c["of"]) == (int((g > 1).sum()), g.size), (row, b)
    names = {"ar": "ar", "labels": "labels", "labels_knorm": "labels_knorm", "mgn": "mgn"}
    for pair, by_metric in sec["pairing"].items():
        first, second = pair.split("_vs_")
        for m, c in by_metric.items():
            x, y = _arrays(rep, names[first], "1024", m), _arrays(rep, names[second], "1024", m)
            better = (x > y) if m == "crit_recall" else (x < y)
            assert (c["first_lower"], c["ties"], c["pairs"]) == \
                (int(better.sum()), int((x == y).sum()), x.size), (pair, m)
    gn = sec["graph_network"]
    means = {r: float(_arrays(rep, r, "1024", "disp_rel_l2").mean()) for r in names}
    assert gn["mgn_lowest_displacement"] is True and min(means, key=means.get) == "mgn"
    gaps = {r: float(_arrays(rep, r, "1024", "energy_gap_rel").mean())
            for r in ("ar", "labels", "mgn")}
    assert gn["mgn_gap_above_labels"] is True and gaps["mgn"] > gaps["labels"]
    assert gn["mgn_gap_above_label_free"] is True and gaps["mgn"] > gaps["ar"]


def test_the_comparisons_and_label_efficiency_follow_from_the_report():
    """The 80 secondary comparisons and the label efficiency (PREREG_CM2D Sec. 5),
    recomputed here from the per-instance arrays: the label-free row (E1's states,
    pool 1,024) is the reference at every budget."""
    rep, sec = _json(RET / "report.json"), _json(VERDICT)["secondary"]
    lf = {m: _arrays(rep, "ar", "1024", m).mean(axis=1) for m in METRICS}
    rows = {"labels": "labels", "labels_knorm": "labels_knorm", "mgn": "mgn"}
    n = 0
    for m, by_budget in sec["comparisons"].items():
        for b, comps in by_budget.items():
            for name, c in comps.items():
                new_row, base_row = name.split("_vs_")
                new = _arrays(rep, rows[new_row], b, m).mean(axis=1)
                base = lf[m] if base_row == "label_free" else \
                    _arrays(rep, rows[base_row], b, m).mean(axis=1)
                g = _guard(new, base)
                assert np.allclose(c["new_per_seed"], new, rtol=1e-12, atol=0), (m, b, name)
                assert np.allclose(c["base_per_seed"], base, rtol=1e-12, atol=0), (m, b, name)
                assert math.isclose(c["rel_change"], g["rel"], rel_tol=1e-9), (m, b, name)
                assert math.isclose(c["threshold"], g["tau"], rel_tol=1e-9), (m, b, name)
                higher = m == "crit_recall"
                assert c["better_direction"] == ("higher" if higher else "lower")
                assert c["new_better_beyond_guard"] == (g["worse"] if higher else g["lower"])
                assert c["new_worse_beyond_guard"] == (g["lower"] if higher else g["worse"])
                n += 1
    assert n == 80
    budgets = ("16", "64", "256", "1024")
    for row, eff in sec["label_efficiency"].items():
        gaps = {b: _arrays(rep, row, b, "energy_gap_rel").mean() for b in budgets}
        for b in budgets:
            assert math.isclose(eff["label_free_advantage_in_energy_gap"][b],
                                1 - lf["energy_gap_rel"].mean() / gaps[b], rel_tol=1e-9)
        for m in ("energy_gap_rel", "disp_rel_l2"):
            below = [int(b) for b in budgets
                     if _arrays(rep, row, b, m).mean() < lf[m].mean()]
            assert eff["first_budget_below_label_free"][m] == (min(below) if below else None)


def test_the_adjudicator_reproduces_the_verdict_in_process():
    """Everything in verdict.json that does not depend on git -- the hypotheses with
    their medians, Welch and resampling intervals, the reuse checks, the secondary
    readings (July's row and E8's built-ins included), the deviations the report,
    status file and logs show, and the run block -- is what scripts/adjudicate_cm2d.py
    computes from the committed files."""
    adj = _script("adjudicate_cm2d")
    e1 = adj.load_json(E1)
    adj.check_e1(e1)
    rep, rep_sha, status, prov, logs = adj.load_return(RET)
    assert rep_sha == _sha(RET / "report.json") and logs == ["run.log"]
    expected = _script("make_cm2d_config").cm2d_config(e1["config"])
    res = adj.adjudicate_cm2d(rep, e1, expected, adj.prereg_hash(PREREG), "prereg-cm2d",
                              _sha(E1), status, adj.load_json(JULY), len(logs))
    got = json.loads(json.dumps(adj.jsonable(res), allow_nan=False))
    v = _json(VERDICT)
    assert set(got) == set(v) - {"inputs", "adjudicator"}
    for k in got:
        assert got[k] == v[k], k
