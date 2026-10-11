"""wp9 Stage 1 (11 October 2026): the box returns of session 1, session 2 and
RUNBOOK_W9 Sec. 4 as committed under records/wp9/, the adjudication's verdict
(records/wp9/w9_verdict.json), the paper material generated from it, and the
facts that DEVIATIONS_W9.md, records/wp9/README.md and the Stage 1 notes
quote from the returns.

H1 and H2 are recomputed here from the arm reports, from PREREG_W9 Sec. 5-6
and without the adjudicator's code; the adjudication itself is re-run on the
committed returns and compared with the committed verdict. The returns'
tarballs are not committed: the README is the record of their SHA-256."""

import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "records" / "wp9"
S1, S2, THREADS = REC / "session1", REC / "session2", REC / "threads"
VERDICT = REC / "w9_verdict.json"
LEDGER = ROOT / "DEVIATIONS_W9.md"
README = REC / "README.md"
NOTES = ROOT / "BRANCH_NOTES_wp9-pool.md"
ARMS = ("c1_n1024", "c1_n25600", "b_n1024", "s_n1024", "c1_n4096")
STAMPED_COMMIT = "92dfc9381555ea40bc3f83fc015d51963febfdf2"
SIZES = {"val": 256, "IB": 2048, "F1": 256, "F2": 256, "F3": 256, "F4": 256, "F5": 256,
         "R": 80}
SEEDS = {"IB": 91007, "F1": 91001, "F2": 91002, "F3": 91003, "F4": 91004, "F5": 91005,
         "R": 91006}
ADJUDICATING = ("src/fejepa/analysis/adjudicate_w9.py", "src/fejepa/analysis/adjudicate.py",
                "src/fejepa/report.py", "scripts/adjudicate_w9.py",
                "scripts/w9_session1_decisions.py")


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _report(arm):
    return _load(S2 / arm / "report.json")


def _evals(report, set_name):
    """The per-seed evaluations of an arm report on one set (the validation
    split is the run's own cell, the other sets its holdouts)."""
    m = report["results"]["e8"]["metrics"]
    cell = m["cells"]["ar"] if set_name == "val" else m["holdouts"][set_name]
    (pool,) = cell
    return cell[pool]["per_seed_eval"]


def _seed_means(report, set_name, metric):
    return [statistics.fmean(e["per_instance"][metric]) for e in _evals(report, set_name)]


def _cstar_medians(report, set_name):
    """Per seed, the median of c* over every load case of the set."""
    return [statistics.median([c for inst in e["amplitude"]["per_instance"]["c_star"]
                               for c in inst]) for e in _evals(report, set_name)]


def _level_cos(report, set_name):
    """Per seed, the medians over the set's load cases of the prediction's
    energy norm over the solution's and of the cosine between them in the
    energy inner product (c* = (|u*| / |u|) cos, so cos = c* |u| / |u*|)."""
    levels, cosines = [], []
    for e in _evals(report, set_name):
        amp = e["amplitude"]["per_instance"]
        lv, cs = [], []
        for c_i, u_i, s_i in zip(amp["c_star"], amp["u_norm_K"], amp["ustar_norm_K"]):
            for c, u, s in zip(c_i, u_i, s_i):
                lv.append(u / s)
                cs.append(c * u / s)
        levels.append(statistics.median(lv))
        cosines.append(statistics.median(cs))
    return levels, cosines


def _three(xs, fmt=".2f") -> str:
    return "{}, {} and {}".format(*(format(x, fmt) for x in xs))


def _guard(new, base):
    """PREREG_W9 Sec. 6: rel, SE_rel with sample SDs over three seeds each, tau."""
    b = statistics.fmean(base)
    rel = statistics.fmean(new) / b - 1
    se = math.sqrt(statistics.stdev(new) ** 2 / 3 + statistics.stdev(base) ** 2 / 3) / b
    tau = max(0.10, 2 * se)
    return {"rel": rel, "se": se, "tau": tau, "lower": rel < -tau, "worse": rel > tau}


def _pct(x: float) -> str:
    """A relative change as the documents quote it: -2.5%, +1.1%."""
    return f"{x * 100:+.1f}%"


def _units(text: str) -> list:
    """A Markdown file cut into paragraphs and list items, each with its line
    breaks and indentation folded into single spaces."""
    return [" ".join(u.split()) for u in re.split(r"\n(?=\s*- )|\n\s*\n", text) if u.strip()]


def _unit_with(text: str, needle: str) -> str:
    found = [u for u in _units(text) if needle in u]
    assert len(found) == 1, (needle, len(found))
    return found[0]


# ------------------------------------------------------------ the returns --

def test_the_session_1_return_is_whole():
    status = (S1 / "status.txt").read_text().splitlines()
    assert [ln.split()[0] for ln in status] == [
        "ood2d.log", "c0_val.log", "c0_trainval.log", "c0_amp2d.log", "c0_memory.log",
        "profile_w9.log", "decisions.log"]
    assert all(ln.split()[1] == "exit=0" for ln in status)
    prov = (S1 / "provenance.txt").read_text()
    assert f"HEAD {STAMPED_COMMIT}" in prov and "describe prereg-w9" in prov
    rec = _load(S1 / "ood2d.json")
    assert rec["failed"] == []
    for fam, seed in SEEDS.items():
        f = rec["families"][fam]
        assert (f["status"], f["seed"], f["n_instances"]) == ("generated", seed, SIZES[fam])
        assert f["manifest_sha256"] == _sha(S1 / "ood2d" / fam / "manifest.json")


def test_the_session_2_return_is_whole_and_hashed():
    plan = _load(S2 / "plan.json")
    assert [a["arm"] for a in plan["arms"]] == list(ARMS)
    assert plan["gate_problems"] == [] and plan["facts"]["git"] == "prereg-w9"
    # the plan was made from session 1's decision file, which the return carries unchanged
    assert (S2 / "decisions.json").read_bytes() == (S1 / "decisions.json").read_bytes()
    assert plan["decisions_sha256"] == _sha(S1 / "decisions.json")
    # the manifests the plan checked on disk are session 1's
    assert sorted(plan["facts"]["ood_manifests"]) == sorted(SEEDS)
    for fam, sha in plan["facts"]["ood_manifests"].items():
        assert sha == _sha(S1 / "ood2d" / fam / "manifest.json")
    # one attempt per arm, in the plan's order, each exit 0; no earlier attempt's log
    status = [ln.split() for ln in (S2 / "status.txt").read_text().splitlines()]
    assert [(s[0], s[1]) for s in status] == [
        (f"{a}.log", w) for a in ARMS for w in ("start", "exit=0")]
    assert not list(S2.glob("*.log.*"))
    prov = (S2 / "provenance.txt").read_text()
    assert f"HEAD {STAMPED_COMMIT}" in prov and "describe prereg-w9" in prov
    listed = dict((m.group(2), m.group(1)) for m in re.finditer(
        r"^([0-9a-f]{64})\s+runs/w9/([^/\s]+)/report\.json$", prov, re.M))
    assert sorted(listed) == sorted(ARMS)
    for arm, sha in listed.items():
        assert _sha(S2 / arm / "report.json") == sha, arm
        rep = _report(arm)
        assert rep["provenance"]["git"] == "prereg-w9"
        assert rep["solve_ledger"]["total"] == 0
    assert re.search(r"^oom_kill 0$", prov, re.M)


def test_section_4_return():
    status = (THREADS / "status.txt").read_text().splitlines()
    assert [ln.split()[0] for ln in status] == [
        "timing_2d_env.log", "timing_2d_one.log", "timing_3d_env.log", "timing_3d_one.log",
        "timing_3d_env2.log", "blas_env.log", "blas_one.log"]
    assert all(ln.split()[1] == "exit=0" for ln in status)
    for name, n in (("blas_env.log", 25), ("blas_one.log", 1)):
        b = _load(THREADS / name)
        assert len(b["pools"]) == 2 and {p["kind"] for p in b["pools"]} == {"openblas"}
        assert all(p["num_threads"] == n for p in b["pools"]) and b["torch_threads"] == n
    for name in ("timing_2d_env", "timing_2d_one", "timing_3d_env", "timing_3d_one",
                 "timing_3d_env2"):
        t = _load(THREADS / f"{name}.json")
        assert t["git"] == "prereg-w9" and not t["skipped_families"]
        for s in t["results"].values():
            sm = s["summary"]
            assert sm["direct_n_solved"] == sm["cg_n_solved"] == sm["n_instances"]
            assert sm["direct_label_max_rel_dev_max"] < 1e-12
            assert sm["cg_label_max_rel_dev_max"] < 1e-8
        chk = t["check_against_report"]
        assert chk["checked"] and max(chk["disp_rel_l2_max_rel_dev"],
                                      chk["energy_gap_rel_max_rel_dev"]) < 1e-12


# ------------------------------------------------------------ the verdict --

def test_the_verdict_reproduces_from_the_committed_returns(tmp_path):
    out = tmp_path / "verdict.json"
    res = subprocess.run(
        [sys.executable, "scripts/adjudicate_w9.py", "--session1", "records/wp9/session1",
         "--session2", "records/wp9/session2", "--e1-report",
         "records/wp8/e1/e1_2d_base/report.json", "--prereg", "PREREG_W9.md", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stderr[-3000:]
    fresh, committed = _load(out), _load(VERDICT)
    # the committed verdict was written by the stamped adjudicating code, in a clean
    # checkout of the stamp
    assert committed["adjudicator"]["git"] == "prereg-w9"
    assert committed["adjudicator"]["against_stamp"] == {
        "ref": "prereg-w9", "compared": True, "differs": []}
    tag = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet",
                          "prereg-w9^{commit}"], capture_output=True, text=True)
    if tag.returncode == 0:
        assert tag.stdout.strip() == STAMPED_COMMIT
        for path in ADJUDICATING:
            blob = subprocess.run(["git", "-C", str(ROOT), "show", f"prereg-w9:{path}"],
                                  capture_output=True, check=True).stdout
            assert hashlib.sha256(blob).hexdigest() == committed["adjudicator"]["sha256"][path]
    # the adjudicator's notes about its own code (this checkout's) are not the verdict's
    fresh["deviations"] = [d for d in fresh["deviations"]
                           if not d.startswith("adjudicating code")]
    for k in set(fresh) | set(committed):
        if k != "adjudicator":
            assert fresh.get(k) == committed.get(k), k


def test_h1_and_h2_recomputed_from_the_reports():
    v = _load(VERDICT)
    n1024, nmax, b, s = (_report(a) for a in ("c1_n1024", "c1_n25600", "b_n1024", "s_n1024"))
    for rep in (n1024, nmax, b, s, _report("c1_n4096")):
        for set_name, n in SIZES.items():
            for metric in ("energy_gap_rel", "disp_rel_l2"):
                ev = _evals(rep, set_name)
                assert len(ev) == 3 and all(len(e["per_instance"][metric]) == n for e in ev)
                assert all(math.isfinite(x) for e in ev for x in e["per_instance"][metric])

    def inband(rep):        # Sec. 5: the validation split and IB together, per seed
        out = []
        for a, c in zip(_evals(rep, "val"), _evals(rep, "IB")):
            g = a["per_instance"]["energy_gap_rel"] + c["per_instance"]["energy_gap_rel"]
            out.append(statistics.fmean(g))
        return out

    h1 = _guard(inband(nmax), inband(n1024))
    assert not h1["lower"] and not h1["worse"] and h1["tau"] == 0.10
    assert v["H1"]["verdict"] == "NOT SUPPORTED" and v["H1"]["n_max"] == 25600
    assert h1["rel"] == pytest.approx(v["H1"]["rel_change"], abs=1e-12)
    assert h1["se"] == pytest.approx(v["H1"]["se_rel"], abs=1e-12)

    def ratio(rep):         # per seed, F5 over the validation split, then their mean
        return [f / w for f, w in zip(_seed_means(rep, "F5", "disp_rel_l2"),
                                      _seed_means(rep, "val", "disp_rel_l2"))]

    conds = {"F5_displacement": _guard(_seed_means(s, "F5", "disp_rel_l2"),
                                       _seed_means(b, "F5", "disp_rel_l2")),
             "F5_over_inband_ratio": _guard(ratio(s), ratio(b)),
             "K1_inband_energy_gap": _guard(_seed_means(s, "val", "energy_gap_rel"),
                                            _seed_means(b, "val", "energy_gap_rel")),
             "K1_inband_displacement": _guard(_seed_means(s, "val", "disp_rel_l2"),
                                              _seed_means(b, "val", "disp_rel_l2"))}
    for key, g in conds.items():
        assert g["rel"] == pytest.approx(v["H2"][key]["rel_change"], abs=1e-12), key
        assert g["tau"] == pytest.approx(v["H2"][key]["threshold"], abs=1e-12), key
        assert (g["lower"], g["worse"]) == (v["H2"][key]["lower"], v["H2"][key]["worse"]), key
    assert conds["F5_displacement"]["lower"] and conds["F5_over_inband_ratio"]["lower"]
    assert not conds["K1_inband_energy_gap"]["worse"]
    assert not conds["K1_inband_displacement"]["worse"]
    assert v["H2"]["verdict"] == "SUPPORTED"
    assert v["refused_reports"] == {} and v["deviations"] == []
    # the fresh baseline shows the deficit rule 1 admitted S on (Sec. 7), so a
    # verdict on S is informative about its mechanism; recomputed from the arrays
    f5, val = _seed_means(b, "F5", "disp_rel_l2"), _seed_means(b, "val", "disp_rel_l2")
    assert statistics.fmean(f5) / statistics.fmean(val) >= 1.5
    cb = _cstar_medians(b, "F5")
    assert statistics.fmean(cb) >= 1.10
    assert cb == pytest.approx(v["S_mechanism"]["b1024"]["c_star_median_F5"]["per_seed"],
                               abs=1e-12)
    assert _cstar_medians(s, "F5") == pytest.approx(
        v["S_mechanism"]["s"]["c_star_median_F5"]["per_seed"], abs=1e-12)


def test_e1_states_give_the_same_values_on_both_hosts():
    """Session 1 (its host, driver 595.71.05) and session 2's 1,024 arm (another
    host, driver 580.95.05) evaluated E1's states on the same instances
    (DEVIATIONS_W9.md W1): the energy gap before and after c*, and c*,
    identical; the displacement error before and after c* within a relative
    1e-15, and identical on the validation split."""
    c0 = _load(S1 / "c0_amp2d.json")
    rep = _report("c1_n1024")
    assert _load(VERDICT)["arms"]["n1024"]["reproduction_max_rel_dev"] == 0.0
    sources = {"val": lambda s: c0["sets"]["inband"][s]["rows"],
               "F5": lambda s: c0["sets"]["F5"][s]["rows"],
               "R": lambda s: c0["remesh"][s]["rows"]}
    worst = 0.0
    for set_name, rows_of in sources.items():
        for i, seed in enumerate(("s0", "s1", "s2")):
            rows, ev = rows_of(seed), _evals(rep, set_name)[i]
            pi, amp = ev["per_instance"], ev["amplitude"]["per_instance"]
            for a, y in (("egap", pi["energy_gap_rel"]), ("egap_c", amp["egap_c"]),
                         ("disp", pi["disp_rel_l2"]), ("disp_c", amp["disp_c"])):
                x = [r[a] for r in rows]
                assert len(x) == len(y) == SIZES[set_name]
                if a.startswith("egap") or set_name == "val":
                    assert x == y, (set_name, seed, a)
                else:
                    worst = max(worst, max(abs(p - q) / abs(q) for p, q in zip(x, y)))
            x = [c for r in rows for c in r["c_star"]]
            y = [c for inst in amp["c_star"] for c in inst]
            assert len(x) == len(y) == 4 * SIZES[set_name] and x == y
    assert 0 < worst <= 1e-15
    # the plan's summary line quotes session 1's own reproduction reading
    rep1 = _load(S1 / "c0_val.json")["summary"]["reproduction"]
    assert set(rep1["disp"].values()) == {0.0}
    quoted = max(rep1["egap"].values())
    assert f"E1 reproduction {quoted!r}" in (S2 / "session2_plan.log").read_text()
    inference = _unit_with(LEDGER.read_text(encoding="utf-8"), "Inference. The 1,024 arm")
    for text in ("reproduced E1's per-instance validation arrays exactly (largest relative "
                 "deviation 0;", f"The plan's summary line quotes {quoted!r} instead",
                 "displacement errors were exact", "are identical, and the displacement error, "
                 "before and after scaling, agrees to within a relative 1e-15 (identical on "
                 "the validation split)"):
        assert text in inference, text


# ------------------------------------------------- what the documents quote --

def test_the_ledger_quotes_the_records():
    ledger = LEDGER.read_text(encoding="utf-8")
    e1_prov = (ROOT / "records" / "wp8" / "e1" / "provenance.txt").read_text()
    s1_prov = (S1 / "provenance.txt").read_text()
    s1_prof = _load(S1 / "profile_2d_w9.json")
    s2_prov = (S2 / "provenance.txt").read_text()
    machine = (THREADS / "machine.txt").read_text()

    def host(text):
        return re.search(r"^hostname (\S+)$", text, re.M).group(1)

    hosts = {"e1": host(e1_prov), "s1": host(s1_prov), "s2": host(s2_prov)}
    assert len(set(hosts.values())) == 3 and "hostname" not in machine
    # each host sits with its own driver and CPUs, as the records give them
    assert ", 595.71.05, " in e1_prov
    e1_prof = _load(ROOT / "records" / "wp8" / "posthoc" / "profile_2d_head.json")
    assert (e1_prof["driver"], e1_prof["cpus"]["usable"]) == ("595.71.05", 25)
    assert "Xeon(R) Platinum 8470Q" in e1_prof["cpu"]
    e1 = _unit_with(ledger, f"`{hosts['e1']}`")
    assert all(x in e1 for x in ("595.71.05", "Xeon Platinum 8470Q", "25 CPUs"))
    assert "580.95.05" not in e1
    assert s1_prof["driver"] == "595.71.05" and "Xeon(R) Gold 6459C" in s1_prof["cpu"]
    assert s1_prof["threads_default"] == 16
    s1 = _unit_with(ledger, f"`{hosts['s1']}`")
    assert all(x in s1 for x in ("Xeon Gold 6459C", "16 CPUs", "595.71.05", "is inferred"))
    assert "580.95.05" not in s1 and "8470Q" not in s1
    assert ", 580.95.05, " in s2_prov and re.search(r"^nproc 25$", s2_prov, re.M)
    s2 = _unit_with(ledger, f"`{hosts['s2']}`")
    assert all(x in s2 for x in ("25 CPUs", "580.95.05", "18:50", "03:34"))
    assert "595.71.05" not in s2
    st = (S2 / "status.txt").read_text().splitlines()
    first, last = st[0].split(), st[-1].split()
    assert (first[2], first[3][:5], last[2], last[3][:5]) == (
        "2026-10-10", "18:50", "2026-10-11", "03:34")
    assert ", 580.95.05, " in machine and re.search(r"^nproc 25$", machine, re.M)
    assert "Xeon(R) Platinum 8470Q" in machine and "\n98784247808\n" in machine
    sec4 = _unit_with(ledger, "Sec. 4, run about seven hours later")
    assert "Xeon Platinum 8470Q" in sec4 and "98,784,247,808 bytes" in sec4
    assert "is inferred from it" in sec4
    # session 1's provenance was written without a GPU (W2)
    assert re.search(r"^nproc 1$", s1_prov, re.M) and "\n2147483648\n50000 100000\n" in s1_prov
    assert "RTX 5090" not in s1_prov
    w2 = _unit_with(ledger, "`cpu.max` 50000 100000")
    assert "1 CPU" in w2 and "2,147,483,648" in w2
    # the percentages, each with the comparison and the set it belongs to
    v = _load(VERDICT)
    rep = v["replication_b1024_vs_n1024"]
    training = _unit_with(ledger, "Training. The fresh baseline")
    inband = [rep["val"]["energy_gap_rel"], rep["IB"]["energy_gap_rel"],
              rep["val"]["disp_rel_l2"], rep["IB"]["disp_rel_l2"]]
    assert not any(g["lower"] or g["worse"] for g in inband)
    assert ("relative energy gap {} on the validation split and {} on IB and the "
            "displacement error {} and {} (tau 10%)".format(
                *(_pct(g["rel_change"]) for g in inband))) in training
    n = sum(1 for d in rep.values() if isinstance(d, dict)
            for g in d.values() if isinstance(g, dict) and "rel_change" in g)
    crossing = {(s, m): g for s, d in rep.items() if isinstance(d, dict)
                for m, g in d.items() if isinstance(g, dict) and "rel_change" in g
                and (g["lower"] or g["worse"])}
    assert sorted(crossing) == [("F3", "egap_c"), ("F5", "disp_rel_l2")]
    f5, f3 = crossing[("F5", "disp_rel_l2")], crossing[("F3", "egap_c")]
    assert f5["worse"] and f3["lower"]
    assert (f"Of the {n} comparisons on all sets, two cross the guard: the displacement "
            f"error on F5, {_pct(f5['rel_change'])} (worse), and the c*-scaled energy gap "
            f"on F3, {_pct(f3['rel_change'])} (lower)") in training
    bearing = _unit_with(ledger, "**Bearing on the verdicts.**")
    nm = [v["nmax_vs_b1024_exploratory"][s]["energy_gap_rel"] for s in ("val", "IB")]
    assert not any(g["lower"] or g["worse"] for g in nm)
    assert ("the 25,600 arm differs by {} on the validation split and {} on IB in relative "
            "energy gap".format(*(_pct(g["rel_change"]) for g in nm))) in bearing


def test_the_readme_quotes_the_records():
    readme = README.read_text(encoding="utf-8")
    v = _load(VERDICT)
    verdict = _unit_with(readme, "The decisions recomputed from session 1's readings")
    assert f"H1 is {v['H1']['verdict']} and H2 {v['H2']['verdict']};" in verdict
    # the sandbox's sets against the box's
    sb = _load(REC / "ood2d_sandbox.json")["families"]
    box = _load(S1 / "ood2d.json")["families"]
    same = [f for f in SEEDS if sb[f]["manifest_sha256"] == box[f]["manifest_sha256"]]
    assert all(sb[f]["n_nodes_min_median_max"] == box[f]["n_nodes_min_median_max"]
               for f in SEEDS)
    assert len(same) == 1
    sandbox = _unit_with(readme, "manifest is the box's byte for byte")
    assert (f"The sandbox's {same[0]} manifest is the box's byte for byte; the other six "
            "differ, while every set's smallest, median and largest node counts are the "
            "same") in sandbox

    def med(name, set_name, key):
        return _load(THREADS / f"{name}.json")["results"][set_name]["summary"][key]["median"]

    env, one, env2 = "timing_3d_env", "timing_3d_one", "timing_3d_env2"
    # the direct solve: one thread as fast as the environment's 25 (within 3%)
    for set_name in ("val", "fine"):
        d_env, d_one = med(env, set_name, "direct_s"), med(one, set_name, "direct_s")
        assert abs(d_one / d_env - 1) < 0.03
        # conjugate gradients: faster with one thread; the repeat agrees with the first
        c_env, c_one, c_env2 = (med(n, set_name, "cg_s") for n in (env, one, env2))
        assert c_one < c_env and abs(c_env2 / c_env - 1) < 0.03
    sec4 = _unit_with(readme, "the direct solve (SuperLU)")
    lo, hi = sorted((med(env, "fine", "direct_s"), med(one, "fine", "direct_s")))
    for text in (
            "took the same time with one thread as with 25: medians "
            f"{med(env, 'val', 'direct_s'):.2f} s on the validation instances and "
            f"{lo:.0f}-{hi:.0f} s on the fine ones",
            f"Conjugate gradients was faster with one thread: {med(one, 'val', 'cg_s'):.2f} s "
            f"against {med(env, 'val', 'cg_s'):.2f} s, and {med(one, 'fine', 'cg_s'):.1f} s "
            f"against {med(env, 'fine', 'cg_s'):.1f} s; the repeat in the environment, after "
            f"the one-thread run, gave {med(env2, 'val', 'cg_s'):.2f} s and "
            f"{med(env2, 'fine', 'cg_s'):.1f} s"):
        assert text in sec4, text
    for sha in ("58148fb7ac3a2055a5529202804dd93115101e5e940aa64b149f66f032991d0e",
                "a47cb50ad9c2e21080e978fecc4249b2986ad61c52adfb6b5503c56d629f47dd",
                "cef9d17f23069ed5baa508559cd4dcaba54ab0609f07fe0dde9d0778d565ea81"):
        assert f"`{sha}`" in readme


def test_the_stage_1_notes_quote_the_verdict_and_the_reports():
    notes = NOTES.read_text(encoding="utf-8")
    stage = " ".join(notes[notes.index("## Stage 1 (11 Oct 2026)"):].split())
    v = _load(VERDICT)
    nm, cmp_s, h1, h2 = (v["nmax_vs_b1024_exploratory"], v["comparisons_s_vs_b1024_exploratory"],
                         v["H1"], v["H2"])
    reports = {a: _report(a) for a in ARMS}
    b, s = reports["b_n1024"], reports["s_n1024"]
    assert f"**H1 {h1['verdict']}.**" in stage and f"**H2 {h2['verdict']}.**" in stage
    assert f"against E1's states: {_pct(h1['rel_change'])}, tau 10%" in stage
    assert ("the 25,600 arm reads {} (validation split) and {} (IB)".format(
        _pct(nm["val"]["energy_gap_rel"]["rel_change"]),
        _pct(nm["IB"]["energy_gap_rel"]["rel_change"]))) in stage
    r = h2["F5_over_inband_ratio"]
    assert ("(i) F5 displacement error {}; (ii) the F5 / in-band ratio {} ({:.2f} to {:.2f}); "
            "(iii) in band, energy gap {} and displacement error {}, neither worse".format(
                _pct(h2["F5_displacement"]["rel_change"]), _pct(r["rel_change"]),
                r["base_mean"], r["new_mean"], _pct(h2["K1_inband_energy_gap"]["rel_change"]),
                _pct(h2["K1_inband_displacement"]["rel_change"]))) in stage
    cs_f5, cb_f5 = _cstar_medians(s, "F5"), _cstar_medians(b, "F5")
    assert (f"S's median c* on F5 was {statistics.fmean(cs_f5):.3f} against the baseline's "
            f"{statistics.fmean(cb_f5):.2f}") in stage
    growth = v["remesh_growth"]
    sm = [statistics.fmean(growth["s"][h]) for h in growth["s"]]
    bm = [statistics.fmean(growth["b1024"][h]) for h in growth["b1024"]]
    assert (f"by a factor of {min(sm):.2f}-{max(sm):.2f} (seed means, one per coarser size) "
            f"against the baseline's {min(bm):.2f}-{max(bm):.2f}") in stage
    # F2: instances worse than the zero field, per seed
    counts = {a: [sum(1 for x in e["per_instance"]["energy_gap_rel"] if x > 1)
                  for e in _evals(rep, "F2")] for a, rep in reports.items()}
    assert all(max(c) > 256 / 4 for c in counts.values())
    assert max(counts["s_n1024"]) == max(max(c) for c in counts.values())
    assert f"S has the most ({_three(counts['s_n1024'], 'd')} by seed)" in stage
    # F2: S's raw means against the baseline's, and S's seed 3
    g_eg, g_dp = cmp_s["F2"]["energy_gap_rel"], cmp_s["F2"]["disp_rel_l2"]
    assert not (g_eg["worse"] or g_dp["worse"])
    s_disp = _seed_means(s, "F2", "disp_rel_l2")
    assert s_disp[0] == max(s_disp)
    assert ("on F2 are {} and {} against the baseline, within the guard only because its tau "
            "is {:.0f}% and {:.0f}%; S's displacement error there is {:.2f} in seed 3 against "
            "{:.2f} and {:.2f} in the others".format(
                _pct(g_eg["rel_change"]), _pct(g_dp["rel_change"]), g_eg["threshold"] * 100,
                g_dp["threshold"] * 100, *s_disp)) in stage
    # F2: level and shape
    (ls, cs), (lb, cb) = _level_cos(s, "F2"), _level_cos(b, "F2")
    assert (f"S's predicted energy norm on F2 is {_three(ls)} times the solution's by seed "
            f"(medians over the load cases), against the baseline's {_three(lb)}, and its "
            f"cosine {_three(cs)} against {_three(cb)}") in stage
    assert min(ls) > max(lb) and max(cs) < min(cb)
    c_s, c_b = _cstar_medians(s, "F2"), _cstar_medians(b, "F2")
    assert (f"S's median c* there is {min(c_s):.3f}-{max(c_s):.3f} by seed against the "
            f"baseline's {min(c_b):.3f}-{max(c_b):.3f}") in stage
    (l5s, _), (l5b, _) = _level_cos(s, "F5"), _level_cos(b, "F5")
    assert (f"lifts the baseline's {min(l5b):.2f}-{max(l5b):.2f} of the solution's energy norm "
            f"to {min(l5s):.2f}-{max(l5s):.2f}") in stage
    worse = sorted((st_, m) for st_, d in cmp_s.items() if isinstance(d, dict)
                   for m, g in d.items() if isinstance(g, dict) and g.get("worse"))
    assert worse == [("F2", "disp_c"), ("F2", "egap_c")]
    assert nm["F2"]["egap_c"]["lower"] and nm["F2"]["disp_c"]["lower"]
    assert ("(energy gap {}, displacement error {}), the only crossings against S, while the "
            "25,600 arm's are lower than the baseline's ({}, {})".format(
                _pct(cmp_s["F2"]["egap_c"]["rel_change"]), _pct(cmp_s["F2"]["disp_c"]["rel_change"]),
                _pct(nm["F2"]["egap_c"]["rel_change"]),
                _pct(nm["F2"]["disp_c"]["rel_change"]))) in stage
    # outside F2: the fresh baseline's failures all in its third seed (seed 5)
    outside = ("val", "IB", "F1", "F3", "F4", "F5", "R")
    fails = {a: [[sum(1 for x in e["per_instance"]["energy_gap_rel"] if x > 1)
                  for e in _evals(rep, st_)] for st_ in outside] for a, rep in reports.items()}
    assert sum(sum(c[:2]) for c in fails["b_n1024"]) == 0
    tops = ", ".join(f"{max(_evals(b, st_)[2]['per_instance']['energy_gap_rel']):,.0f} on {st_}"
                     for st_ in ("F1", "F3", "F4"))
    top5 = max(_evals(b, "F5")[2]["per_instance"]["energy_gap_rel"])
    assert (f"is in its third seed, seed 5 ({sum(c[2] for c in fails['b_n1024'])}; the largest "
            f"relative energy gaps {tops} and {top5:,.0f} on F5)") in stage
    changes = [g[st_]["energy_gap_rel"]["rel_change"] for g in (cmp_s, nm)
               for st_ in ("F1", "F3", "F4", "F5")]
    assert (f"They make the {round(max(changes) * 100)}% to {round(min(changes) * 100)}% "
            "changes of S's and the 25,600 arm's mean energy gaps on F1, F3, F4 and F5") in stage
    e1 = fails["c1_n1024"]
    assert (f"E1's states have {sum(map(sum, e1))} such instance-seed values outside F2 "
            f"({sum(e1[outside.index('F3')])} on F3); S and the 4,096 and 25,600 arms have "
            "none") in stage
    assert all(sum(map(sum, fails[a])) == 0 for a in ("s_n1024", "c1_n25600", "c1_n4096"))


def test_the_paper_material_regenerates_byte_for_byte():
    mk = _module("make_w9_paper_material")
    raw = VERDICT.read_bytes()
    files = mk.build(json.loads(raw), hashlib.sha256(raw).hexdigest())
    out = ROOT / "paper" / "wp9"
    for name, text in files.items():
        assert (out / name).read_bytes() == text.encode("utf-8"), name
    # no table of an earlier generator left behind
    assert {p.name for p in out.glob("table_w9_*.tex")} == {
        n for n in files if n.startswith("table_w9_")}
    md = files["tables.md"]
    assert "Verdict: NOT SUPPORTED." in md and "Verdict: SUPPORTED." in md
    assert "DEVIATIONS_W9.md" in md
    assert "tail" not in files["table_w9_c1_disp.tex"]
    assert "the tail is the per-seed 90th percentile" in files["table_w9_c1.tex"]
    amp = files["table_w9_s_amp.tex"]
    f2 = [ln for ln in amp.splitlines() if ln.startswith("F2 slender")]
    assert len(f2) == 1 and f2[0].count("(worse)") == 2
    assert "a small $c^*$ can also mean a poor shape" in amp
