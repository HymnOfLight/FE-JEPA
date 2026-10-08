"""cmame-paper Stage 4b: the return of RUNBOOK_CMAME Sec. A (inference timing
and field export, run on the box on 8 October 2026 and copied byte for byte
into records/cmame/timing/) and the B0 readiness log (records/cmame/
cm2d_ready.log) are complete and tied to what they read: every step exited 0
on the named commit; the reports are the committed records and the states
the runs' own; the solvers agree with the stored labels and the timed model
with its report, on every timed instance; the field export is complete and
within the runbook's ranges; its counts agree with the manuscript's numbers,
which come from other records; and its per-load arrays, recomputed here,
bear out the exactness lemma, the zero-field test and ranking, the
energy-optimal amplitude and the stress bound on every load case, the
report's per-instance arrays, the counts and the figure selections."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "records"
CM = REC / "cmame"
TM = CM / "timing"
HEAD = "61f018f59243d5268b908fbabdc96ae2ecd94152"      # Stage 3, Sec. A's and B0's commit
TREE = "ec2ecbf2057f28a97a94b1fe867ad2c9eb89a4ca"
RUNS = {"timing_2d.json": "wp8/e1/e1_2d_base/report.json",
        "timing_3d.json": "wp8/e2/baseline/report_phase2b.json"}
PHASE2B = "wp8/e2/baseline/report_phase2b.json"
SUP = REC / "wp9" / "phase2_supervised_states.json"
ARMS = ["ar", "labels", "mgn"]
PER_INSTANCE = ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err", "crit_recall")
FIGURES = {"fig5": ("val", 107, "instance_00485.npz"),
           "fig6": ("val", 154, "instance_00130.npz"),
           "fig7": ("fine", 77, "instance_00077.npz")}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(p) -> dict:
    return json.loads(Path(p).read_text())


def _number(name: str) -> int:
    tex = (ROOT / "paper" / "cmame" / "generated" / "numbers.tex").read_text()
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{([0-9,]+)\}", tex)
    assert m, name
    return int(m.group(1).replace(",", ""))


def _section(text: str, title: str) -> str:
    m = re.search(r"^## " + re.escape(title) + r".*?(?=^## |\Z)", text, re.M | re.S)
    assert m, title
    return m.group(0)


def _per_instance(rep: dict, arm: str, budget, seed: int) -> dict:
    cell = rep["results"]["e8"]["metrics"]["cells"][arm][str(budget)]
    return cell["per_seed_eval"][seed]["per_instance"]


def _fine(rep: dict, seed: int) -> dict:
    return rep["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["per_seed_eval"][seed][
        "per_instance"]


def _median_rank(values) -> int:
    v = np.asarray(values, float)
    return int(np.argsort(v, kind="stable")[(len(v) - 1) // 2])


def _max_rel_dev(got, want) -> float:
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert got.shape == want.shape and (want != 0).all()
    return float(np.max(np.abs(got - want) / np.abs(want)))


def test_the_files_are_the_ones_the_readme_lists():
    readme = (CM / "README.md").read_text(encoding="utf-8")
    listed = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$",
                             _section(readme, "Sec. A return"), re.M))
    present = sorted(p.relative_to(TM).as_posix() for p in TM.rglob("*") if p.is_file())
    assert present == sorted(listed)
    assert len(present) == 14
    for name in present:
        assert _sha(TM / name) == listed[name], name
    ready = re.search(r"`cm2d_ready\.log` \(SHA-256 `([0-9a-f]{64})`\)",
                      _section(readme, "B0 readiness"))
    assert ready and _sha(CM / "cm2d_ready.log") == ready.group(1)


def test_every_step_exited_zero_on_the_named_commit():
    lines = [x for x in (TM / "status.txt").read_text().splitlines() if x.strip()]
    assert lines == ["timing_2d.log exit=0", "timing_3d.log exit=0", "fields_3d.log exit=0"]
    machine = (TM / "machine.txt").read_text().splitlines()
    assert machine[0] == f"HEAD {HEAD}" and machine[1] == f"tree {TREE}"
    assert machine[2].startswith("NVIDIA GeForce RTX 5090")    # git status printed nothing
    for name in ("timing_2d.json", "timing_3d.json", "fields/fields.json"):
        assert _json(TM / name)["git"].endswith("-g" + HEAD[:7]), name   # no "-dirty"
    summary = r"^505 passed, 2 deselected(, \d+ warnings?)? in [0-9.]+s \(\d+:\d\d:\d\d\)$"
    assert re.search(summary, (TM / "pytest.log").read_text(), re.M)
    pre = [x for x in (TM / "precheck.txt").read_text().splitlines() if x.strip()]
    assert len(pre) == 2                                       # date and GPU line: no other run
    assert re.fullmatch(r"\w{3} Oct +8 [0-9:]{8} \w+ 2026", pre[0])
    util, mem = re.fullmatch(r"(\d+) %, (\d+) MiB", pre[1]).groups()
    assert int(util) == 0 and int(mem) < 1000


def test_the_inputs_are_the_committed_records_and_the_runs_own_states():
    # the scripts take the configuration from the report, so the report's hash covers it
    for name, rel in RUNS.items():
        d, rep = _json(TM / name), _json(REC / rel)
        assert d["report"] == f"records/{rel}" and d["report_sha256"] == _sha(REC / rel)
        states = rep["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
        assert d["seed"] == 0 and d["state"].endswith("/ar_p1024_s0.pt")
        assert d["state_sha256"] == states["s0"]["sha256"]
    f = _json(TM / "fields" / "fields.json")
    rep = _json(REC / PHASE2B)
    assert f["report_sha256"] == _sha(REC / PHASE2B)
    assert f["sup_hashes_sha256"] == _sha(SUP)
    ar = rep["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    sup = _json(SUP)["sha256"]
    assert sorted(f["states"]) == sorted(f"{a}_s{s}" for a in ARMS for s in (0, 1, 2))
    for key, st in f["states"].items():
        arm, seed = key.rsplit("_s", 1)
        assert st["used"] is True and st["sha256_ok"] is True, key
        want = ar[f"s{seed}"]["sha256"] if arm == "ar" else sup[st["file"]]
        assert st["sha256"] == want, key


def test_the_solvers_agree_with_the_labels_and_the_model_with_its_report():
    for name, rel in RUNS.items():
        d, rep = _json(TM / name), _json(REC / rel)
        assert d["settings"]["cg_tol"] == 1e-10
        assert d["settings"]["solvers"] == ["direct", "cg", "cg_warm", "cg_match",
                                            "cg_match_cstar"]
        sets = ("val",) if name == "timing_2d.json" else ("val", "fine")
        assert sorted(d["results"]) == sorted(sets)
        for s in sets:
            sm, rows = d["results"][s]["summary"], d["results"][s]["per_instance"]
            n = len(rows)
            assert n == sm["n_instances"] == d["sets"][s]["n"] and sm["complete"] is True
            for kind in d["settings"]["solvers"]:                      # every instance solved
                assert sm[f"{kind}_n_solved"] == n, (name, s, kind)
            for kind in ("cg", "cg_warm"):                  # no CG fell back to the direct solve
                assert all(r[f"{kind}_fallback"] is False for r in rows), (name, s, kind)
            assert max(r["direct_label_max_rel_dev"] for r in rows) <= 1e-12
            assert max(r["cg_label_max_rel_dev"] for r in rows) <= 1e-8
            assert max(r["cg_warm_label_max_rel_dev"] for r in rows) <= 1e-8
            for kind in ("cg_match", "cg_match_cstar"):     # every matching CG met its target
                assert sum(r[f"{kind}_unreached"] for r in rows) == 0, (name, s, kind)
                assert sum(r[f"{kind}_iters_mismatch"] for r in rows) == 0, (name, s, kind)
                assert {e for r in rows for e in r[f"{kind}_trace_end"]} <= {"target", "zero"}
            # the timed model reproduces its report's per-instance arrays on every timed
            # instance (the script itself checks the first eight)
            ref = _per_instance(rep, "ar", 1024, 0) if s == "val" else _fine(rep, 0)
            for k in PER_INSTANCE:
                got = [r[k] for r in rows]
                assert _max_rel_dev(got, ref[k][:n]) <= 1e-12, (name, s, k)


def test_the_field_export_is_complete_and_within_the_runbooks_ranges():
    f = _json(TM / "fields" / "fields.json")
    assert f["left_out"] == [] and f["content_mismatch"] == []
    assert all(c["ok"] for c in f["content_checks"].values())
    for arm in ARMS:
        g = f["diagnostics"][arm]
        assert g["prop1_bound_ratio_max"] <= 1
        assert g["stress_identity_max_rel_dev"] < 1e-6 and g["gap_identity_max_rel_dev"] < 1e-6
        assert g["u_dirichlet_max"] == 0
    for fig, (kind, index, file) in FIGURES.items():
        sel = f["figures"][fig]
        assert (sel["set"], sel["index"], sel["file"]) == (kind, index, file)
        with np.load(TM / "fields" / f"{fig}.npz", allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
        assert (meta["set"], meta["index"], meta["file"], meta["seed"]) == (kind, index, file, 0)
    assert FIGURES["fig5"][1] == _number("numWorstIndex")


def test_the_counts_agree_with_the_manuscripts_numbers():
    c = _json(TM / "fields" / "fields.json")["counts"]
    pairs, loads = _number("numThreePairs"), _number("numThreeFreeLoads")
    for arm, macro in (("ar", "numThreeFreeWorse"), ("labels", "numThreeLabWorse"),
                       ("mgn", "numThreeMgnWorse")):
        a = c[arm]
        assert a["instances"] == pairs and a["load_cases"] == loads
        assert a["instances_mean_rel_gap_above_1"] == _number(macro), arm
        assert a["load_cases_rel_gap_above_1_after_cstar"] == 0, arm
        assert a["load_cases_cstar_increased_gap"] == 0, arm
        assert a["median_rel_gap_after_cstar"] <= a["median_rel_gap"], arm
    assert c["ar"]["load_cases_rel_gap_above_1"] == _number("numThreeFreeLoadWorse")


def test_the_per_load_arrays_bear_out_the_theory_the_report_and_the_selections():
    f = _json(TM / "fields" / "fields.json")
    rep = _json(REC / PHASE2B)
    with np.load(TM / "fields" / "energies_val.npz", allow_pickle=False) as z:
        arms, seeds = [str(a) for a in z["arms"]], [int(s) for s in z["seeds"]]
        files = [str(x) for x in z["files"]]
        pi, rel, rel_c, eK, vm = (z[k] for k in ("pi", "rel", "rel_c", "eK", "vm_vol"))
        e2, vm_elem = z["e2"], z["vm_elem"]
        pi_star, uK, u2, gamma = z["pi_star"], z["uK_star"], z["u2_star"], z["gamma_star"]
    n = len(files)
    assert arms == ARMS and seeds == [0, 1, 2]
    assert files == f["val"]["files"] and n == f["val"]["n"]
    for x in (pi, rel, rel_c, eK, vm, e2, vm_elem):
        assert x.shape == (3, 3, n, 4) and np.isfinite(x).all()
    assert (pi_star < 0).all() and (uK > 0).all() and (u2 > 0).all()
    g = eK / uK                                  # squared relative energy-norm error
    # Lemma "Exactness": the relative energy gap is g (the stored gap is computed from the
    # energies, g from the error itself)
    assert np.allclose(rel, g, rtol=1e-6, atol=0)
    # Corollary "Zero-field test and ranking": Pi_h(u) > 0 exactly when u is further from
    # U* in the energy norm than the zero field, and two predictions for the same load case
    # are ranked by their energies as by their energy-norm errors
    assert ((pi > 0) == (eK > uK)).all()
    for a in range(3):
        for b in range(a + 1, 3):
            assert ((pi[a] < pi[b]) == (eK[a] < eK[b])).all(), (arms[a], arms[b])
    # Proposition "Energy-optimal amplitude": c* u is worse than neither u nor the zero field
    assert (rel_c <= rel).all() and (rel_c <= 1).all()
    # Proposition "Energy gap and stress error": vm_vol^2 <= (1 + gamma*) g
    assert (vm ** 2 <= (1 + gamma) * g * (1 + 1e-9)).all()
    for ai, arm in enumerate(arms):
        c = f["counts"][arm]
        assert int((pi[ai] > 0).sum()) == c["load_cases_pi_positive"], arm
        assert int((rel[ai] > 1).sum()) == c["load_cases_rel_gap_above_1"], arm
        assert int((rel[ai].mean(axis=-1) > 1).sum()) == c["instances_mean_rel_gap_above_1"], arm
        # the report's per-instance arrays, the means over the load cases (the export's
        # content check, recomputed and widened here; the graph network's CUDA scatter
        # reductions are not bitwise reproducible)
        budget = f["pool"] if arm == "ar" else f["budget"]
        tol = f["content_tol"] if arm == "mgn" else 1e-12
        for s in range(3):
            want = _per_instance(rep, arm, budget, s)
            for k, got in (("energy_gap_rel", rel[ai, s]), ("vm_rel_l2", vm_elem[ai, s]),
                           ("disp_rel_l2", np.sqrt(e2[ai, s] / u2))):
                assert _max_rel_dev(got.mean(axis=-1), want[k]) <= tol, (arm, s, k)
    # the figure rules of fields.json, applied to the report's arrays
    lab = [_per_instance(rep, "labels", f["budget"], s)["energy_gap_rel"] for s in range(3)]
    fig5 = f["selections"]["fig5"]
    assert [int(np.argmax(v)) for v in lab] == fig5["per_seed_choice"]
    assert int(np.argmax(lab[0])) == fig5["index"] == FIGURES["fig5"][1]
    ar0 = _per_instance(rep, "ar", f["pool"], 0)["energy_gap_rel"]
    assert _median_rank(ar0) == f["selections"]["fig6"]["index"] == FIGURES["fig6"][1]
    assert _median_rank(_fine(rep, 0)["disp_rel_l2"]) == f["selections"]["fig7"]["index"] \
        == FIGURES["fig7"][1]
    # the validation figures carry their instance's per-load arrays, seed 0
    for fig in ("fig5", "fig6"):
        i = FIGURES[fig][1]
        with np.load(TM / "fields" / f"{fig}.npz", allow_pickle=False) as z:
            assert (z["pi_star"] == pi_star[i]).all(), fig
            for ai, arm in enumerate(arms):
                assert (z[f"pi_{arm}"] == pi[ai, 0, i]).all(), (fig, arm)
                assert (z[f"rel_{arm}"] == rel[ai, 0, i]).all(), (fig, arm)


def test_the_readiness_check_said_go_on_the_same_commit():
    lines = [x for x in (CM / "cm2d_ready.log").read_text().splitlines() if x.strip()]
    assert lines[-1] == "GO"
    checks = [x for x in lines if x.startswith("[precheck] ")]
    status = {re.match(r"\[precheck\] (\w+)\s+(\w+):", x).group(2):
              re.match(r"\[precheck\] (\w+)", x).group(1) for x in checks}
    assert {k for k, v in status.items() if v == "info"} == {"tag", "report_git", "head", "stamp"}
    assert all(v in ("ok", "info") for v in status.values())
    assert {"gpu_idle", "no_other_run", "torch", "disk", "imports", "clean", "config", "fresh",
            "e1_states", "plan", "corpus", "labels", "reproduction", "smoke"} <= set(status)
    assert f"[precheck] info head: HEAD {HEAD} tree {TREE}" in lines
    # "stamp" is "info" before the stamp whatever its state: the line says which
    assert any(x.startswith("[precheck] info stamp: ") and "PREREG_CM2D.md is unstamped" in x
               for x in lines)
    assert any("reproduction: largest relative deviation 0 over 3 states on cuda" in x
               for x in lines)
    smoke = [re.search(r"smoke: relative energy gap (\S+) after one epoch on cuda$", x)
             for x in lines if x.startswith("[precheck] ok   smoke: ")]
    assert len(smoke) == 1 and smoke[0] and np.isfinite(float(smoke[0].group(1)))
