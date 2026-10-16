"""cmame-paper Stage 11: configs/holdout_v1.json and PREREG_HOLDOUT.md.

Every value of the configuration is the record it names: the reports'
SHA-256, the corpora's manifest SHA-256 and the splits the reports record,
each state's SHA-256 from the record of the run that trained it; each
holdout set is the tail of its corpus's pool after the deepest prefix any
recorded run or tool used (records/cmame/holdout/pool_depths.json,
re-derived from the branches and tags it records when they are present and
checked for consistency otherwise), without the instances that another
recorded set drew again, that a recorded tool rebuilt, or whose own file is
held outside the box (records/cmame/holdout/overlap.json, re-derived here,
with its bounds; the regression tool's miniatures within them; the sandbox
scan's tool), and its instance names follow from the split; the validation
names that the earlier exports recorded follow the same rule. The
thresholds and tolerances are the values the pre-registration states.
The pre-registration is the r6 draft the stamp script accepts, or a stamp of
it that verifies; the stamp script stamps a copy of the draft so that the
box script's and the adjudication's checks pass."""

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fejepa.analysis.holdout import (ROWS, expected_names, holdout_names, load_spec,
                                     spec_sha256)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "holdout_v1.json"
PREREG = ROOT / "PREREG_HOLDOUT.md"
DEPTHS = ROOT / "records" / "cmame" / "holdout" / "pool_depths.json"
OVERLAP = ROOT / "records" / "cmame" / "holdout" / "overlap.json"


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def spec():
    return load_spec(CONFIG)


def test_reports_corpora_and_splits_are_the_records(spec):
    for name, s in spec["sets"].items():
        rep = json.loads((ROOT / s["report"]).read_text())
        assert _sha(ROOT / s["report"]) == s["report_sha256"], name
        assert rep["config"]["data"]["dir"] == s["corpus"]
        assert rep["config"]["split"] == s["split"]
        ds = rep["provenance"]["datasets"][0]
        assert ds["dir"] == s["corpus"] and ds["manifest_sha256"] == s["manifest_sha256"]
        assert ds["n_instances"] == s["n_corpus"]
        assert list(s["rows"]) == list(ROWS[name])
        cells = rep["results"]["e8"]["metrics"]["cells"]
        for row, r in s["rows"].items():
            assert len(cells[row][r["cell"]]["per_seed_eval"]) == len(spec["seeds"])
            assert r["kind"] == ("mgn" if row == "mgn" else "fejepa")


def test_state_hashes_are_those_of_the_runs_that_trained_them(spec):
    h3, h2 = spec["sets"]["h3d"], spec["sets"]["h2d"]
    rep3 = json.loads((ROOT / h3["report"]).read_text())
    ar3 = rep3["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    assert h3["rows"]["ar"]["sha256"] == {k[1:]: v["sha256"] for k, v in ar3.items()}
    sup = json.loads((ROOT / "records/wp9/phase2_supervised_states.json").read_text())
    assert sup["states_dir"] == "runs/phase2/e8_states"
    for row in ("labels", "mgn"):
        assert h3["rows"][row]["sha256_source"] == "records/wp9/phase2_supervised_states.json"
        for s, want in h3["rows"][row]["sha256"].items():
            assert sup["sha256"][f"{row}_b1024_s{s}.pt"] == want
            assert h3["rows"][row]["state"].format(seed=s) == \
                f"runs/phase2/e8_states/{row}_b1024_s{s}.pt"
    assert h3["rows"]["ar"]["state"] == "runs/phase2/e8_states/ar_p1024_s{seed}.pt"
    prov = {}
    for line in (ROOT / "records/cmame/cm2d/return/provenance.txt").read_text().splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line.strip())
        if m:
            prov[m.group(2)] = m.group(1)
    assert prov["runs/cm2d/report.json"] == h2["report_sha256"]
    for row, r in h2["rows"].items():
        for s, want in r["sha256"].items():
            assert prov[r["state"].format(seed=s)] == want, (row, s)
    # E1's states, as E1's own report and CM2D's report record them
    e1 = json.loads((ROOT / "records/wp8/e1/e1_2d_base/report.json").read_text())
    rep2 = json.loads((ROOT / h2["report"]).read_text())
    for rep in (e1, rep2):
        ar2 = rep["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
        assert h2["rows"]["ar"]["sha256"] == {k[1:]: v["sha256"] for k, v in ar2.items()}
    assert h2["rows"]["ar"]["sha256_source"].startswith("records/wp8/e1/e1_2d_base/report.json")
    assert rep2["config"]["experiments"]["e8"]["reuse_from"]["states_dir"] == \
        "runs/e1_2d_base/e8_states"


def _deepest(rec: dict) -> dict:
    out = {}
    for r in rec["reports"]:
        assert r["recorded_depth"] == r["config_depth"], r["path"]   # by construction
        out[r["corpus"]] = max(out.get(r["corpus"], 0), r["recorded_depth"])
    return out


def test_holdout_starts_after_the_deepest_recorded_pool(spec):
    rec = json.loads(DEPTHS.read_text())
    assert len(rec["scanned"]) == 6 and len(rec["reports"]) == 15
    assert rec["history"]["reports"] == 15 and rec["history"]["same_as_heads"]
    assert _deepest(rec) == rec["deepest"]
    for name, s in spec["sets"].items():
        assert rec["deepest"][s["corpus"]] == s["used_prefix"], name
        reps = [r for r in rec["reports"] if r["corpus"] == s["corpus"]]
        assert reps and all(r["split"] == s["split"] for r in reps)
        others = [o["depth"] for o in rec["other_reads"] if o["corpus"] == s["corpus"]]
        assert all(d <= s["used_prefix"] for d in others), name
    assert max(o["depth"] for o in rec["other_reads"]) == spec["sets"]["h2d"]["used_prefix"]
    # the records this repository holds agree with the listing
    for r in rec["reports"]:
        if "origin/cmame-paper" in r["branches"]:
            assert _sha(ROOT / r["path"]) == r["sha256"], r["path"]
    for o in rec["other_reads"]:
        if o["branch"] == "origin/cmame-paper":
            assert _sha(ROOT / o["path"]) == o["sha256"], o["path"]


def test_pool_depths_rederived_from_the_recorded_commits():
    rec = json.loads(DEPTHS.read_text())
    commits = [s["commit"] for s in rec["scanned"]]
    have = [subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{c}^{{commit}}"],
                           capture_output=True).returncode == 0 for c in commits]
    if not all(have):
        # a single-branch checkout (the box's): the record is checked for consistency
        # above; what this checkout holds of it is checked here
        assert any(have), "none of the scanned commits is in this checkout"
        return
    mod = _script("holdout_pool_depths")
    got = mod.report_depths(commits)
    assert got["deepest"] == rec["deepest"]
    key = lambda rows: sorted((r["sha256"], r["corpus"], r["recorded_depth"],  # noqa: E731
                               r["config_depth"]) for r in rows)
    assert key(got["reports"]) == key(rec["reports"])
    assert mod.other_reads({s["branch"]: s["commit"] for s in rec["scanned"]}) == \
        rec["other_reads"]
    # the scan of every JSON file reachable from the recorded branches and tags
    refs = rec["history"]["refs"]
    assert {r["object"] for r in refs} >= set(commits)
    if all(subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", r["object"]],
                          capture_output=True).returncode == 0 for r in refs):
        hist = mod.history_reports([r["object"] for r in refs])
        assert hist["json_objects"] == rec["history"]["json_objects"]
        assert len(hist["reports"]) == rec["history"]["reports"] == 15
        assert set(hist["reports"]) == {r["sha256"] for r in rec["reports"]}


def test_exclusions_are_the_overlap_record_and_the_record_rederives(spec):
    rec = json.loads(OVERLAP.read_text())
    s3 = spec["sets"]["h3d"]
    tail3 = expected_names(s3["n_corpus"], s3["split"]["n_val"], s3["split"]["seed"],
                           s3["used_prefix"])
    # the regression's miniature (in-band 0 to 79) and the fine set's 106 and 294
    mini = {f"instance_{i:05d}.npz" for i in range(80)}
    want3 = sorted((set(tail3) & mini) | {"instance_00106.npz", "instance_00294.npz"})
    assert s3["excluded"] == rec["h3d_drawn_again"] == want3 and len(want3) == 34
    assert "instance_00028.npz" in want3 and len(set(tail3) & mini) == 32
    assert spec["sets"]["h2d"]["excluded"] == rec["h2d_excluded"] == \
        sorted(set(rec["rebuilt_2d"]["in_h2d"]) | set(rec["copies_2d"]["in_h2d"]))
    assert len(rec["rebuilt_2d"]["in_h2d"]) == 32 and len(rec["h2d_excluded"]) == 33
    three = ["instance_00006.npz", "instance_00009.npz", "instance_00010.npz"]
    assert rec["rebuilt_2d"]["remeshed_in_h2d"] == three
    assert rec["copies_2d"]["in_h2d"] == three + ["instance_29992.npz"]
    # the 2D regression miniature is inside the rebuilt range
    assert set(rec["regress_2d"]["in_h2d"]) <= set(rec["rebuilt_2d"]["in_h2d"])
    assert rec["regress_2d"]["instances"] == [0, 79]
    assert all(c["ok"] for c in rec["checks"]) and len(rec["checks"]) == 3
    pairs = {(c["a"], c["i"], c["b"], c["j"]) for c in rec["coincidences_3d"]}
    corpus, fine = "runs/data3d_phase2", "runs/data3d_phase2_fine"
    mdata, mfine = "regression miniature (data)", "regression miniature (data_transfer)"
    assert pairs == ({(corpus, i, fine, j) for i, j in
                      ((28, 30), (106, 112), (198, 212), (210, 224), (218, 232), (294, 316))}
                     | {(corpus, i, mdata, i) for i in range(80)}
                     | {(fine, j, mfine, j) for j in range(7)}
                     | {(fine, 30, mdata, 28)})
    tail_mini = [c for c in rec["coincidences_3d"] if c["a"] == corpus and c["b"] == mdata
                 and c["a_role"] == "tail"]
    assert sorted(f"instance_{c['i']:05d}.npz" for c in tail_mini) == sorted(set(tail3) & mini)
    assert sum(c["b_role"] == "miniature pool, labelled and trained on" for c in tail_mini) == 24
    # the pairs that share the box and Poisson ratio only stay in
    partial = {(c["a"], c["i"]) for c in rec["same_box_and_nu_only_3d"]}
    assert (corpus, 269) in partial and "instance_00269.npz" not in s3["excluded"]
    assert rec["same_geometry_other_loads_3d"] == []
    # the sandbox as scanned, and the bounds within which every shared tail instance is out
    assert rec["bounds"] == {"3d_band": 80, "3d_one_size": 320, "2d": 256,
                             "within_bounds_left_out": True}
    assert [rec["sandbox"][k]["longest"] for k in ("3d_band", "3d_one_size", "2d")] == \
        [80, 20, 80]
    assert all(rec["sandbox"][k]["longest"] <= rec["bounds"][k]
               for k in ("3d_band", "3d_one_size", "2d"))
    tail2 = expected_names(spec["sets"]["h2d"]["n_corpus"], 256, 1,
                           spec["sets"]["h2d"]["used_prefix"])
    assert {n for n in tail2 if int(n[9:14]) < 256} <= set(spec["sets"]["h2d"]["excluded"])
    assert {n for n in tail3 if int(n[9:14]) < 80} <= set(s3["excluded"])
    got = _script("holdout_overlap").overlap(spec)
    assert {k: v for k, v in got.items() if k != "seconds"} == \
        {k: v for k, v in rec.items() if k != "seconds"}


def test_the_regression_miniatures_are_the_records_and_within_the_bounds(tmp_path):
    """The cross-branch regression keeps drawing its miniatures from the
    corpora's seed whenever it runs: they are the sets the overlap record
    replays, within its bounds."""
    reg = _script("regress_against_branch")
    rec = json.loads(OVERLAP.read_text())
    sets = {s["name"]: s for s in rec["sets_3d"]}
    c3 = json.loads((ROOT / "configs" / "phase2b_v1.json").read_text())
    m3 = reg.miniature(c3, tmp_path / "p2b")
    data, fine = sets["regression miniature (data)"], sets["regression miniature (data_transfer)"]
    assert m3["data"]["seed"] == data["seed"] == 0 == c3["data"]["seed"]
    assert m3["data"]["n"] == data["n"] == 80 <= rec["bounds"]["3d_band"]
    assert m3["data"]["lc_range"] == data["mesh_size"] == [0.30, 0.36]
    assert m3["data_transfer"]["seed"] == fine["seed"] == 0
    assert m3["data_transfer"]["n"] == fine["n"] == 7 <= rec["bounds"]["3d_one_size"]
    assert "lc_range" not in m3["data_transfer"]
    c2 = json.loads((ROOT / "configs" / "e1_2d_base.json").read_text())
    m2 = reg.miniature(c2, tmp_path / "e1")
    assert m2["data"]["seed"] == c2["data"]["seed"] == rec["rebuilt_2d"]["corpus_seed"] == 0
    assert m2["data"]["n"] == rec["regress_2d"]["n"] == 80 <= rec["bounds"]["2d"]
    assert m2["split"] == rec["regress_2d"]["split"] == {"n_val": 3, "seed": 1}


def test_the_sandbox_scan_reads_the_generators_sets_against_the_bounds(tmp_path):
    """scripts/holdout_scan_sets.py: the corpora's generators' sets of seed 0,
    per kind, against the overlap record's bounds; other seeds and other
    generators left out; a copy of a corpus manifest listed apart (refused
    with instance files beside it); stray instance files listed. The test's
    manifests are removed afterwards, so that a later scan of the sandbox
    does not read them."""
    try:
        _scan_cases(tmp_path)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def _scan_cases(tmp_path):
    def manifest(d, backend, seed, n, **kw):
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps(
            {"backend": backend, "seed": seed, **kw,
             "instances": [{"file": f"instance_{i:05d}.npz"} for i in range(n)]}))

    manifest(tmp_path / "a", "gmsh3d", 0, 80, lc_range=[0.30, 0.36])
    manifest(tmp_path / "b", "gmsh3d", 0, 20, lc=0.24)
    manifest(tmp_path / "c", "gmsh", 0, 256)
    manifest(tmp_path / "d", "gmsh", 7, 5000)                  # another seed
    manifest(tmp_path / "e", "synthetic", 0, 1300)             # another generator
    manifest(tmp_path / "f", "gmsh", 0, 30000)                 # a copy of the corpus manifest
    manifest(tmp_path / "g", "gmsh3d", 0, 2000, lc_range=[0.0579, 0.0906])
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "instance_00006.npz").write_bytes(b"x")
    out = tmp_path / "scan.json"

    def run():
        r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/holdout_scan_sets.py"),
                            str(tmp_path), "--out", str(out)], capture_output=True, text=True)
        return r, json.loads(out.read_text())

    r, o = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert o["bounds"] == {"3d_band": 80, "3d_one_size": 320, "2d": 256}
    assert {k: (v["sets"], v["longest"], v["beyond"]) for k, v in o["kinds"].items()} == \
        {"3d_band": (1, 80, []), "3d_one_size": (1, 20, []), "2d": (1, 256, [])}
    assert sorted(c["n"] for c in o["corpus_manifest_copies"]) == [2000, 30000]
    assert [(s["first"], s["all_empty"]) for s in o["stray_instance_files"]] == [(6, False)]
    manifest(tmp_path / "h", "gmsh", 0, 257)                   # beyond the 2D bound
    r, o = run()
    assert r.returncode == 1 and not o["within_bounds"]
    assert [b["n"] for b in o["kinds"]["2d"]["beyond"]] == [257]
    (tmp_path / "h" / "manifest.json").unlink()
    (tmp_path / "f" / "instance_00000.npz").write_bytes(b"x")  # the corpus copied with its files
    r, o = run()
    assert r.returncode == 1 and not o["within_bounds"]


def test_holdout_names_follow_the_split(spec):
    for name, s in spec["sets"].items():
        sp = s["split"]
        tail = expected_names(s["n_corpus"], sp["n_val"], sp["seed"], s["used_prefix"])
        names = holdout_names(s)
        assert names == [n for n in tail if n not in set(s["excluded"])]
        assert len(names) == s["n_holdout"] == len(set(names))
        assert set(s["excluded"]) <= set(tail)
        perm = np.random.default_rng(sp["seed"]).permutation(s["n_corpus"])
        val = {f"instance_{i:05d}.npz" for i in perm[:sp["n_val"]]}
        used = {f"instance_{i:05d}.npz" for i in perm[sp["n_val"]:sp["n_val"] + s["used_prefix"]]}
        assert not (set(tail) & (val | used))
        assert len(val | used | set(tail)) == s["n_corpus"]
    assert spec["sets"]["h3d"]["n_holdout"] == 686 and spec["sets"]["h2d"]["n_holdout"] == 4111
    # the validation names the earlier exports recorded follow the same rule
    f3 = json.loads((ROOT / "records/cmame/timing/fields/fields.json").read_text())
    f2 = json.loads((ROOT / "records/cmame/spectra/export/spectra.json").read_text())
    for files, n in ((f3["val"]["files"], 2000), (f2["val"]["files"], 30000)):
        perm = np.random.default_rng(1).permutation(n)
        assert files == [f"instance_{i:05d}.npz" for i in perm[:256]]


def test_criteria_are_the_prereg_values(spec):
    """The thresholds that decide (PREREG_HOLDOUT Sec. 5) and the content
    tolerance (Sec. 3), as the configuration fixes them."""
    assert spec["criteria"] == {"guard_band": 0.1, "guard_k": 2.0, "most": 0.5,
                                "almost_all": 0.95, "halved": -0.5,
                                "representative_alpha": 0.05}
    assert spec["content_tol"] == 1e-3 and spec["seeds"] == [0, 1, 2]
    assert spec["labels"] == {"method": "direct", "stage": "holdout-labels",
                              "residual_max": 1e-8}
    text = PREREG.read_text(encoding="utf-8")
    for words in ("on more than 50% of the instances in each seed",
                  "higher on at least 95% of the instance-seed pairs",
                  "tau = max(10%, 2 x SE_rel)", "rel < -50%, the point value",
                  "which must be at most 1e-8",
                  "of the relative energy gap are at most 1e-3",
                  "Holm's step-down procedure at a family-wise level of 5% "
                  "(`representative_alpha` in the configuration"):
        assert words in text, words


def _draft_text(stamp) -> str:
    """PREREG_HOLDOUT.md as its approved draft (a stamped file un-stamped)."""
    text = PREREG.read_text(encoding="utf-8")
    m = re.search(r"\*\*Status:\*\* r6, stamped (.+?) before the run:", text)
    if m:
        text = text.replace(stamp.stamped_status(m.group(1)), stamp.DRAFT_STATUS)
    text = re.sub(r"^CONFIG_SHA256\[holdout_v1\] = [0-9a-f]{64}$",
                  "CONFIG_SHA256[holdout_v1] = <fill before tagging>", text, flags=re.M)
    return re.sub(r"^PREREG_HOLDOUT_SHA256 = [0-9a-f]{64}$", stamp.FOOTER_OPEN, text,
                  flags=re.M)


def test_the_text_names_the_draft_the_stamp_script_accepts():
    """r6 (an independent review, D6): Sec. 7 names the draft the stamp
    script accepts (it still named r5 after the script had moved to r6)."""
    stamp = _script("stamp_prereg_holdout")
    rev = re.search(r"\*\*Status:\*\* (r\d+) DRAFT", stamp.DRAFT_STATUS).group(1)
    text = _draft_text(stamp)
    assert re.findall(r"this (r\d+) draft's", text) == [rev]
    assert f"the approved {rev} draft's" in (ROOT / "scripts" / "stamp_prereg_holdout.py").read_text()


def test_prereg_is_the_draft_or_a_verified_stamp(spec, tmp_path):
    from fejepa.report import read_prereg_entries, verify_prereg

    stamp = _script("stamp_prereg_holdout")
    text = PREREG.read_text(encoding="utf-8")
    entries = read_prereg_entries(PREREG)
    assert [lab for lab, _ in entries] == ["holdout_v1"]
    assert spec["prereg_file"] == "PREREG_HOLDOUT.md"
    if entries[0][1] == "<fill before tagging>":
        assert text.count(stamp.DRAFT_STATUS) == 1
        assert text.rstrip("\n").endswith(stamp.FOOTER_OPEN)
    else:
        assert entries[0][1] == spec_sha256(spec)
        assert stamp.self_hash_ok(text) and stamp.DRAFT_STATUS not in text
        assert verify_prereg(spec, PREREG, label="holdout_v1") == spec_sha256(spec)
    draft = _draft_text(stamp)
    for k in ("T1", "T2", "T3", "H1", "H2a", "H2b", "H2c", "D2", "H3"):
        assert f"**{k}.**" in draft or f"**{k}** (a reading" in draft, k
    assert "**V** (not a claim of the manuscript)" in draft
    # the stamp script stamps a copy of the draft; the original is never written
    before = _sha(PREREG)
    p = tmp_path / "PREREG_HOLDOUT.md"
    p.write_text(draft, encoding="utf-8")
    with pytest.raises(SystemExit):
        stamp.stamp(p, CONFIG, "18th October 2026")                # the date's form
    assert p.read_text(encoding="utf-8") == draft
    res = stamp.stamp(p, CONFIG, "18 October 2026")
    assert res["config_sha256"]["holdout_v1"] == spec_sha256(spec)
    out = p.read_text(encoding="utf-8")
    assert stamp.self_hash_ok(out) and stamp.stamped_status("18 October 2026") in out
    assert verify_prereg(spec, p, label="holdout_v1") == spec_sha256(spec)
    assert out != draft
    with pytest.raises(SystemExit):
        stamp.stamp(p, CONFIG, "18 October 2026")                  # already stamped
    assert _sha(PREREG) == before


def test_config_refuses_a_changed_structure(tmp_path, spec):
    def refused(edit) -> None:
        bad = json.loads(CONFIG.read_text())
        edit(bad)
        p = tmp_path / "c.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_spec(p)

    refused(lambda d: d["sets"]["h3d"].update(used_prefix=1000))
    refused(lambda d: d["labels"].update(method="cg"))
    refused(lambda d: d["sets"]["h2d"]["rows"].pop("labels_knorm"))
    refused(lambda d: d["sets"]["h3d"].update(n_holdout=720))           # exclusions not counted
    refused(lambda d: d["criteria"].pop("representative_alpha"))        # a threshold missing
    refused(lambda d: d["criteria"].update(guard_k2=2.0))               # one not read
    first_val = f"instance_{np.random.default_rng(1).permutation(2000)[0]:05d}.npz"

    def outside(d):                                                     # a validation name
        s = d["sets"]["h3d"]
        s["excluded"] = sorted(s["excluded"] + [first_val])
        s["n_holdout"] -= 1

    refused(outside)
    refused(lambda d: d["sets"]["h2d"]["excluded"].reverse())           # not sorted
    refused(lambda d: d["sets"]["h3d"].pop("excluded"))
