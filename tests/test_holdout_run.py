"""cmame-paper Stage 11 (PREREG_HOLDOUT Sec. 2-4, 7): the box script
(scripts/holdout_run.py) and the adjudication (scripts/adjudicate_holdout.py)
end to end, on miniature runs shaped like the manuscript's: a 3D corpus of
structured tetrahedra and a 2D corpus of gmsh plates, each with its seeded
split, labelled validation instances and unlabelled pool; random-weight
transformer and graph-network states for three seeds; reports with every
row's per-instance validation arrays; a configuration naming them, with one
excluded instance, and a pre-registration stamped for it.

The labels are the exact direct solves and the corpus files keep their
bytes; a second labelling keeps every label and the manifest as it was
(its summary line says so), a manifest record that differs from what the
labelling finds is written again, and a truncated labels file is solved
again; every state passes its content check with the evaluation's own
numbers; the arrays and summaries agree, and the per-load gaps and energies
give the zero-field identity, on the holdout and the validation instances;
an unstamped pre-registration, a changed state and a changed report are
refused before anything is written; a state that does not reproduce its
report ends the evaluation with exit status 5 and leaves the claims that
read it NOT EVALUATED; the adjudication reads the return, records as
deviations a set that did not return, a labels manifest other than the one
its evaluation read, a resumed labelling, labelled instances, attempts and
hosts, and refuses a pre-registration stamped for another configuration, a
return other than the one listed or of another commit or tree, and each
altered input."""

import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}
RUN = ROOT / "scripts" / "holdout_run.py"
ADJ = ROOT / "scripts" / "adjudicate_holdout.py"
METRICS = ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err", "crit_recall")
SEEDS = (0, 1, 2)
N_VAL, USED = 3, 2
MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 8, "mgn_depth": 1,
         "features": {"load_summary": True, "geometry": True}}
ROWS = {"h3d": ("ar", "labels", "mgn"), "h2d": ("ar", "labels", "labels_knorm", "mgn")}
STAGES = ("label_h3d.log", "eval_h3d.log", "label_h2d.log", "eval_h2d.log")
DRAFT = """# PRE-REGISTRATION HOLDOUT -- test copy

**Branch:** test. {status}

CONFIG_SHA256[holdout_v1] = <fill before tagging>
PREREG_HOLDOUT_SHA256 = <record after commit>
"""


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _label_val(corpus: Path, n: int) -> None:
    """Label the split's validation instances, as the runs' corpora carry them."""
    from fejepa.data.archive import add_labels, load_instance, mark_labelled
    from fejepa.experiments.protocol import load_split
    from fejepa.fe.solve import solve_fe_displacement

    sp = load_split(str(corpus), N_VAL, seed=1)
    for f in sp.val_files:
        a = load_instance(f)
        U, _ = solve_fe_displacement(a.K, a.F, np.asarray(a.free_mask, bool), method="direct")
        add_labels(f, U)
    mark_labelled(corpus, {Path(f).name for f in sp.val_files})


def _corpus_3d(out: Path, n: int) -> Path:
    from fejepa.fe.tet3d import generate_tet3d_dataset

    generate_tet3d_dataset(out, n=n, seed=5)
    _label_val(out, n)
    return out


def _corpus_2d(out: Path, n: int) -> Path:
    from fejepa.data.archive import save_instance, write_manifest
    from fejepa.fe.elasticity import LOAD_NAMES
    from fejepa.fe.generator import build_instance, sample_params

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    recs = []
    for i in range(n):
        params = sample_params(rng)
        params["target_h"] = 0.2
        arch = build_instance(params)
        fn = f"instance_{i:05d}.npz"
        save_instance(arch, out / fn)
        recs.append({"file": fn, "n_nodes": arch.n_nodes,
                     "n_holes": arch.meta["extra"]["n_holes"], "labelled": False})
    write_manifest(out, recs, {"backend": "gmsh", "seed": 4, "labelled_policy": "none",
                               "load_names": LOAD_NAMES})
    _label_val(out, n)
    return out


def _save(kind: str, model: dict, seed: int, path: Path) -> str:
    import torch

    from fejepa.experiments.parallel import _build_model

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_build_model({"kind": kind, "model": model, "seed": seed}).state_dict(), path)
    return _sha(path)


def _val_arrays(report: dict, kind: str, model: dict, state: Path, seed: int) -> dict:
    from fejepa.analysis.common import build_model_from_config
    from fejepa.analysis.posthoc import run_files
    from fejepa.data.archive import load_instance
    from fejepa.metrics import evaluate_model, torch_predictor

    m = build_model_from_config(dict(model, kind=kind), state_path=str(state), seed=seed)
    ev = evaluate_model(torch_predictor(m, "cpu"),
                        [load_instance(f) for f in run_files(report, "val")])
    return {"per_instance": ev["per_instance"]}


def _run_set(tmp: Path, name: str, corpus: Path, model: dict, n_excluded: int) -> dict:
    """A run of `name`'s shape: states, a report with validation arrays, and
    the configuration's set block (the first `n_excluded` tail names left out)."""
    from fejepa.analysis.holdout import expected_names
    from fejepa.data.archive import load_manifest, manifest_sha256
    from fejepa.report import config_sha256

    n = len(load_manifest(corpus)["instances"])
    cfg = {"model": model, "tf32": False, "split": {"n_val": N_VAL, "seed": 1},
           "data": {"dir": str(corpus)}, "experiments": {"e8": {"pool_sizes": [4]}}}
    report = {"config": cfg,
              "provenance": {"config_sha256": config_sha256(cfg), "seeds": list(SEEDS),
                             "datasets": [{"dir": str(corpus),
                                           "manifest_sha256": manifest_sha256(corpus),
                                           "n_instances": n}]},
              "results": {"e8": {"metrics": {"cells": {}}}}}
    rows = {}
    for k, row in enumerate(ROWS[name]):
        kind = "mgn" if row == "mgn" else "fejepa"
        pat = str(tmp / "states" / name / f"{row}_s{{seed}}.pt")
        shas, evals = {}, []
        for s in SEEDS:
            p = Path(pat.format(seed=s))
            shas[str(s)] = _save(kind, model, 10 * k + s, p)
            evals.append(_val_arrays(report, kind, model, p, s))
        report["results"]["e8"]["metrics"]["cells"][row] = {"4": {"per_seed_eval": evals}}
        rows[row] = {"state": pat, "cell": "4", "kind": kind, "sha256": shas,
                     "sha256_source": "test"}
    rp = tmp / f"report_{name}.json"
    rp.write_text(json.dumps(report))
    excluded = sorted(expected_names(n, N_VAL, 1, USED)[:n_excluded])
    return {"corpus": str(corpus), "manifest_sha256": manifest_sha256(corpus), "n_corpus": n,
            "split": {"n_val": N_VAL, "seed": 1}, "used_prefix": USED, "excluded": excluded,
            "n_holdout": n - N_VAL - USED - n_excluded, "report": str(rp),
            "report_sha256": _sha(rp), "rows": rows}


def _spec_and_prereg(tmp: Path, sets: dict, out=None) -> tuple:
    stamp = _script("stamp_prereg_holdout")
    prereg = tmp / "PREREG_HOLDOUT.md"
    prereg.write_text(DRAFT.format(status=stamp.DRAFT_STATUS), encoding="utf-8")
    spec = {"_comment": "test", "prereg_file": str(prereg), "seeds": list(SEEDS),
            "out": str(out or tmp / "runs" / "holdout"),
            "labels": {"method": "direct", "stage": "holdout-labels", "residual_max": 1e-8},
            "content_tol": 1e-3, "sets": sets,
            "criteria": {"guard_band": 0.10, "guard_k": 2.0, "most": 0.5,
                         "almost_all": 0.95, "halved": -0.5}}
    cp = tmp / "holdout_v1.json"
    cp.write_text(json.dumps(spec, indent=1))
    return cp, prereg, stamp


def _run(*args, script=RUN, cwd=None):
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True,
                          text=True, env=ENV, cwd=cwd)


def _corpus_bytes(corpus: Path) -> dict:
    return {p.name: _sha(p) for p in sorted(corpus.iterdir())}


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    pytest.importorskip("gmsh")
    pytest.importorskip("skfem")
    tmp = tmp_path_factory.mktemp("holdout")
    c3 = _corpus_3d(tmp / "data3d", n=8)
    c2 = _corpus_2d(tmp / "data2d", n=10)
    sets = {"h3d": _run_set(tmp, "h3d", c3, dict(MODEL, features=dict(MODEL["features"],
                                                                        spatial_dim=3)), 1),
            "h2d": _run_set(tmp, "h2d", c2, MODEL, 0)}
    cp, prereg, stamp = _spec_and_prereg(tmp, sets)
    before = {k: _corpus_bytes(Path(s["corpus"])) for k, s in sets.items()}
    # refused before the stamp
    r = _run("--config", cp, "--set", "h3d", "--stage", "label", "--device", "cpu")
    assert r.returncode != 0 and "unstamped" in (r.stderr + r.stdout)
    assert not (Path(json.loads(cp.read_text())["out"]) / "h3d").exists()
    stamp.stamp(prereg, cp, "18 October 2026")
    logs = {}
    for name in ("h3d", "h2d"):
        for st in ("label", "eval"):
            r = _run("--config", cp, "--set", name, "--stage", st, "--device", "cpu")
            assert r.returncode == 0, r.stderr[-3000:]
            logs[f"{st}_{name}.log"] = r.stdout + r.stderr
    return {"tmp": tmp, "spec_path": cp, "spec": json.loads(cp.read_text()), "prereg": prereg,
            "before": before, "logs": logs}


def test_labels_are_exact_and_the_corpus_untouched(box):
    from fejepa.analysis.holdout import expected_names, holdout_names
    from fejepa.data.archive import load_instance
    from fejepa.fe.solve import solve_fe_displacement

    spec = box["spec"]
    assert len(spec["sets"]["h3d"]["excluded"]) == 1
    for name, s in spec["sets"].items():
        assert _corpus_bytes(Path(s["corpus"])) == box["before"][name]
        out = Path(spec["out"]) / name
        man = json.loads((out / "manifest.json").read_text())
        names = holdout_names(s)
        tail = expected_names(s["n_corpus"], N_VAL, 1, USED)
        assert names == [n for n in tail if n not in s["excluded"]]
        assert [r["file"] for r in man["instances"]] == names and man["n"] == len(names)
        assert man["excluded"] == s["excluded"] and man["ledger"]["total"] == 4 * len(names)
        assert man["reused"] == 0
        assert sorted(p.name for p in (out / "labels").iterdir()) == sorted(names)
        for r in man["instances"]:
            assert r["residual"] < 1e-10 and not r["source_labelled"]
            assert r["corpus_labelled_flag"] is False
            a = load_instance(Path(s["corpus"]) / r["file"])
            U, _ = solve_fe_displacement(a.K, a.F, np.asarray(a.free_mask, bool),
                                         method="direct")
            with np.load(out / "labels" / r["file"]) as z:
                assert np.array_equal(z["U_star"], U)
                assert str(z["source_sha256"]) == r["source_sha256"]


def _restamped(box, tmp: Path, edit=None) -> tuple:
    """The box's configuration with its output under `tmp` (and `edit`
    applied), its own pre-registration stamped; the h3d and h2d label
    directories copied there."""
    spec = json.loads(box["spec_path"].read_text())
    s = dict(spec, out=str(tmp / "again"))
    for name in ("h3d", "h2d"):
        shutil.copytree(Path(spec["out"]) / name / "labels", tmp / "again" / name / "labels")
        shutil.copy(Path(spec["out"]) / name / "manifest.json",
                    tmp / "again" / name / "manifest.json")
    if edit:
        edit(s)
    stamp = _script("stamp_prereg_holdout")
    pr = tmp / "PREREG_HOLDOUT.md"
    pr.write_text(DRAFT.format(status=stamp.DRAFT_STATUS), encoding="utf-8")
    s["prereg_file"] = str(pr)
    cp = tmp / "holdout_v1.json"
    cp.write_text(json.dumps(s))
    stamp.stamp(pr, cp, "18 October 2026")
    return cp, s


def _summary(stdout: str) -> dict:
    """The label stage's summary line."""
    return json.loads([x for x in stdout.splitlines() if x.startswith('{"set"')][-1])


def test_relabelling_keeps_every_label_and_the_manifest(box, tmp_path):
    cp, s = _restamped(box, tmp_path)
    man_path = tmp_path / "again" / "h3d" / "manifest.json"
    keep = man_path.read_bytes()
    n = json.loads(keep)["n"]
    r = _run("--config", cp, "--set", "h3d", "--stage", "label", "--device", "cpu")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "labels unchanged" in r.stdout and man_path.read_bytes() == keep
    sm = _summary(r.stdout)
    assert (sm["n"], sm["reused"], sm["solves"], sm["kept"]) == (n, n, 0, True)
    # a record that is not what the labelling finds: the manifest is written again
    _edit_json(man_path, lambda d: d["instances"][1].update(residual=0.5))
    r = _run("--config", cp, "--set", "h3d", "--stage", "label", "--device", "cpu")
    assert r.returncode == 0, r.stderr[-2000:]
    sm = _summary(r.stdout)
    assert (sm["reused"], sm["solves"], sm["kept"]) == (n, 0, False)
    assert "labels unchanged" not in r.stdout
    again = json.loads(man_path.read_text())
    assert again["instances"] == json.loads(keep)["instances"] and again["reused"] == n
    # a truncated labels file (a machine stop mid-write) is solved again
    man = json.loads(keep)
    first = tmp_path / "again" / "h3d" / "labels" / man["instances"][0]["file"]
    want = first.read_bytes()
    first.write_bytes(b"")
    r = _run("--config", cp, "--set", "h3d", "--stage", "label", "--device", "cpu")
    assert r.returncode == 0, r.stderr[-2000:]
    sm = _summary(r.stdout)
    assert (sm["reused"], sm["solves"], sm["kept"]) == (n - 1, 4, False)
    man2 = json.loads(man_path.read_text())
    assert man2["reused"] == man2["n"] - 1 and man2["ledger"]["total"] == 4
    strip = lambda recs: [{k: v for k, v in r.items() if k != "labels_sha256"}  # noqa: E731
                          for r in recs]
    assert strip(man2["instances"]) == strip(man["instances"])      # the zip's time differs
    assert man2["instances"][1:] == man["instances"][1:]
    with np.load(first) as z, np.load(io.BytesIO(want)) as w:
        assert np.array_equal(z["U_star"], w["U_star"])
    assert not list((tmp_path / "again" / "h3d" / "labels").glob("*.tmp"))
    # a labels file that records other bytes of its corpus file is solved again
    from fejepa.analysis.holdout import _read_labels, _save_labels

    second = tmp_path / "again" / "h3d" / "labels" / man["instances"][1]["file"]
    U, src, _ = _read_labels(second)
    _save_labels(second, U, src, "0" * 64)
    r = _run("--config", cp, "--set", "h3d", "--stage", "label", "--device", "cpu")
    assert r.returncode == 0, r.stderr[-2000:]
    sm = _summary(r.stdout)
    assert (sm["reused"], sm["solves"], sm["kept"]) == (n - 1, 4, False)
    U2, _, sha2 = _read_labels(second)
    assert sha2 == man["instances"][1]["source_sha256"] and np.array_equal(U2, U)


def test_eval_checks_states_and_writes_consistent_arrays(box):
    from fejepa.analysis.holdout import holdout_names, worse_than_zero
    from fejepa.data.archive import load_instance
    from fejepa.metrics import evaluate_fields

    spec = box["spec"]
    for name, s in spec["sets"].items():
        out = Path(spec["out"]) / name
        ev = json.loads((out / "eval.json").read_text())
        assert ev["content_mismatch"] == [] and ev["val"]["n"] == N_VAL
        assert ev["excluded"] == s["excluded"] and ev["holdout"]["n"] == s["n_holdout"]
        for k, c in ev["content_checks"].items():
            assert c["ok"] and c["finite"]
            for m in METRICS:
                assert c[f"{m}_max_rel_dev"] < 1e-6, (name, k, m)
        with np.load(out / "eval_arrays.npz") as z:
            Z = {k: z[k] for k in z.files}
        assert list(Z["rows"]) == list(ROWS[name]) and list(Z["seeds"]) == list(SEEDS)
        files = [str(f) for f in Z["files"]]
        assert files == holdout_names(s)
        R, N = len(ROWS[name]), s["n_holdout"]
        assert Z["values"].shape == (R, 3, N, 5) and Z["gap_load"].shape == (R, 3, N, 4)
        assert Z["val_values"].shape == (R, 3, N_VAL, 5)
        assert Z["val_gap_load"].shape == Z["val_pi_load"].shape == (R, 3, N_VAL, 4)
        assert Z["pi_star"].shape == (N, 4) and Z["val_pi_star"].shape == (N_VAL, 4)
        assert ev["arrays_sha256"] == _sha(out / "eval_arrays.npz")
        for ri, row in enumerate(ROWS[name]):
            for si, seed in enumerate(SEEDS):
                assert np.allclose(Z["values"][ri, si].mean(axis=0),
                                   [ev["summary"][f"{row}_s{seed}"][m] for m in METRICS])
                w = worse_than_zero(Z["gap_load"][ri, si], Z["pi_load"][ri, si])
                assert w == ev["worse_than_zero"][f"{row}_s{seed}"]
                assert w["sign_mismatch"] == 0
        # Pi_h(u) - Pi_h(U*) = g |Pi_h(U*)| per load case (Lemma 1), on both sets
        for p in ("", "val_"):
            G, P, PS = Z[p + "gap_load"], Z[p + "pi_load"], Z[p + "pi_star"]
            assert np.allclose(P - PS[None, None], G * np.abs(PS)[None, None], rtol=1e-9,
                               atol=1e-12)
        # one instance recomputed with the stored labels
        from fejepa.analysis.holdout import predict, row_model

        report = json.loads(Path(s["report"]).read_text())
        a = load_instance(Path(s["corpus"]) / files[0])
        with np.load(out / "labels" / files[0]) as z:
            a.U_star = z["U_star"]
        m = row_model(report, s, "ar", s["rows"]["ar"]["state"].format(seed=0), 0, "cpu")
        vals = evaluate_fields(predict(m, a, "cpu"), a)
        assert np.allclose([vals[k] for k in METRICS], Z["values"][0, 0, 0], rtol=1e-9)
        assert not list(out.glob("*.tmp"))


def test_eval_refuses_a_changed_state_or_report_before_writing(box):
    spec = json.loads(box["spec_path"].read_text())
    evs = {n: (Path(spec["out"]) / n / "eval.json") for n in ("h3d", "h2d")}
    before = {n: (p.read_bytes(), p.stat().st_mtime_ns) for n, p in evs.items()}
    p = Path(spec["sets"]["h2d"]["rows"]["labels"]["state"].format(seed=1))
    keep = p.read_bytes()
    try:
        p.write_bytes(keep + b"\0")
        r = _run("--config", box["spec_path"], "--set", "h2d", "--stage", "eval", "--device",
                 "cpu")
        assert r.returncode == 3 and "states refused" in r.stderr
    finally:
        p.write_bytes(keep)
    rp = Path(spec["sets"]["h3d"]["report"])
    keep = rp.read_bytes()
    try:
        rp.write_bytes(keep + b" ")
        r = _run("--config", box["spec_path"], "--set", "h3d", "--stage", "eval", "--device",
                 "cpu")
        assert r.returncode == 3 and "not the report" in r.stderr
    finally:
        rp.write_bytes(keep)
    assert {n: (p.read_bytes(), p.stat().st_mtime_ns) for n, p in evs.items()} == before


# ------------------------------------------------------------ adjudication --

def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def _return(box, tmp: Path, out=None, sets=("h3d", "h2d"), logs=None, status=None) -> Path:
    """A return as RUNBOOK_CMAME E2 writes it."""
    out = Path(out or box["spec"]["out"])
    ret = tmp / "return"
    for name in ("h3d", "h2d"):
        (ret / name).mkdir(parents=True)
        if name in sets:
            for f in ("eval.json", "eval_arrays.npz", "manifest.json"):
                shutil.copy(out / name / f, ret / name / f)
    for k, text in (logs or box["logs"]).items():
        (ret / k).write_text(text)
    (ret / "status.txt").write_text(status or "".join(
        f"{k} start 2026-10-18T10:0{i}:00Z box1\n{k} exit=0\n" for i, k in enumerate(STAGES)))
    (ret / "precheck.txt").write_text("Sun Oct 18 10:00:00 UTC 2026\nbox1\n0 %, 2 MiB\n")
    (ret / "machine.txt").write_text("HEAD x\nhostname box1\n")
    _provenance(ret)
    return ret


def _provenance(ret: Path, head=None, tree=None) -> None:
    lines = [f"HEAD {head or _git('rev-parse', 'HEAD')}",
             f"tree {tree or _git('rev-parse', 'HEAD^{tree}')}",
             "describe x", "describe-all x", "hostname box1", "--- files ---"]
    lines += [f"{_sha(p)}  ./{p.relative_to(ret).as_posix()}" for p in sorted(ret.rglob("*"))
              if p.is_file() and p.name != "provenance.txt"]
    (ret / "provenance.txt").write_text("\n".join(lines) + "\n")


def _adjudicate(box, ret: Path, out: Path, spec_path=None, prereg=None):
    first = next(ret / n / "eval.json" for n in ("h3d", "h2d")
                 if (ret / n / "eval.json").is_file())
    ev = json.loads(first.read_text())
    return _run("--return", ret, "--config", spec_path or box["spec_path"], "--prereg",
                prereg or box["prereg"], "--expected-git", ev["git"], "--code-ref", "HEAD",
                "--out", out, script=ADJ)


def test_adjudication_reads_the_return(box, tmp_path):
    ret = _return(box, tmp_path)
    out = tmp_path / "verdict.json"
    r = _adjudicate(box, ret, out)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads(out.read_text())
    assert list(res["claims"]) == ["T1", "T2", "T3", "H1", "H2a", "H2b", "H2c", "D2"]
    assert sum(res["tally"].values()) == 8 and res["tally"]["NOT EVALUATED"] == 0
    assert all(v["verdict"] in ("REPLICATED", "NOT REPLICATED") for v in res["claims"].values())
    assert res["readings"]["H3"]["reading"] != "NOT EVALUATED"
    assert all(c["ok"] for cs in res["content_checks"].values() for c in cs.values())
    assert res["inputs"]["h3d"]["eval.json"]["sha256"] == _sha(ret / "h3d" / "eval.json")
    assert {"status.txt", "precheck.txt", "machine.txt", "h3d/eval.json"} <= \
        set(res["inputs"]["listed"])
    for name in ("h3d", "h2d"):
        sec = res["secondary"][name]
        assert set(sec) == {"holdout", "validation", "ratio_holdout_over_validation"}
        assert set(sec["holdout"]) == set(sec["validation"])
        assert set(sec["holdout"]["worse_than_zero"]["ar"]) == {"s0", "s1", "s2", "all_seeds"}
        with np.load(ret / name / "eval_arrays.npz") as z:      # each side its own arrays
            for side, key in (("holdout", "values"), ("validation", "val_values")):
                for ri, row in enumerate(ROWS[name]):
                    got = sec[side]["per_seed_means"][row]["disp_rel_l2"]
                    assert np.allclose(got, z[key][ri, :, :, 0].mean(axis=1), rtol=1e-12)
            h, v = z["values"][0, :, :, 0].mean(), z["val_values"][0, :, :, 0].mean()
        assert sec["ratio_holdout_over_validation"]["ar"]["disp_rel_l2"] == pytest.approx(h / v)
    assert set(res["validation_reference"]["claims"]) == set(res["claims"])
    assert any("torch" in d for d in res["deviations"])        # CPU, not the original stack
    assert not any("attempt" in d or "host" in d for d in res["deviations"])
    assert json.dumps(res, allow_nan=False)


def _altered(box, tmp: Path, edit) -> subprocess.CompletedProcess:
    ret = _return(box, tmp)
    edit(ret)
    _provenance(ret)
    return _adjudicate(box, ret, tmp / "verdict.json")


def _edit_json(path: Path, fn) -> None:
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def _edit_npz(path: Path, fn) -> None:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    fn(d)
    np.savez_compressed(path, **d)


def _fix_arrays_sha(ret: Path, name: str) -> None:
    _edit_json(ret / name / "eval.json",
               lambda e: e.update(arrays_sha256=_sha(ret / name / "eval_arrays.npz")))


def _fix_manifest_sha(ret: Path, name: str) -> None:
    _edit_json(ret / name / "eval.json",
               lambda e: e.update(labels_manifest_sha256=_sha(ret / name / "manifest.json")))


REFUSAL = {
    "limit": "n_holdout_limit is 2, expected None",
    "val_limit": "n_val_limit is 2, expected None",
    "seeds": "eval.json: seeds is [2, 1, 0], expected [0, 1, 2]",
    "metrics": "eval.json: metrics is ['crit_recall',",
    "npz_seeds": "rows, seeds or metrics differ",
    "npz_metrics": "rows, seeds or metrics differ",
    "counts": "instance counts differ from the arrays'",
    "content_tol": "eval.json: content_tol is 0.5, expected 0.001",
    "git": "git is 'something-else'",
    "config": "config_sha256 is",
    "report": "report_sha256 is",
    "excluded": "excluded is",
    "state": "state labels_s1 is not the configuration's",
    "summary": "summary ar_s0 is not its arrays' mean",
    "worse": "worse_than_zero ar_s1 is not its arrays'",
    "content": "content check mgn_s2 reads False, its arrays read True",
    "order": "not the holdout instances in pool order",
    "rows": "rows, seeds or metrics differ",
    "val_names": "not the run's validation split",
    "loads": "gap_load has shape",
    "pi_star": "pi_star has shape",
    "arrays_hash": "arrays_sha256 is",
    "residual": "a label above the residual bound",
    "names": "not the split's holdout set in pool order",
    "manifest_key": "used_prefix is",
    "unlisted": "not the file the provenance file lists",
}


@pytest.mark.parametrize("what", sorted(REFUSAL))
def test_adjudication_refuses_an_altered_return(box, tmp_path, what):
    def edit(ret: Path):
        ev, man, arr = ret / "h2d" / "eval.json", ret / "h2d" / "manifest.json", \
            ret / "h2d" / "eval_arrays.npz"
        if what == "limit":
            _edit_json(ev, lambda d: d.update(n_holdout_limit=2))
        elif what == "val_limit":
            _edit_json(ev, lambda d: d.update(n_val_limit=2))
        elif what == "seeds":
            _edit_json(ev, lambda d: d.update(seeds=d["seeds"][::-1]))
        elif what == "metrics":
            _edit_json(ev, lambda d: d.update(metrics=d["metrics"][::-1]))
        elif what == "counts":
            _edit_json(ev, lambda d: d["holdout"].update(n=d["holdout"]["n"] + 1))
        elif what == "content_tol":
            _edit_json(ev, lambda d: d.update(content_tol=0.5))
        elif what == "git":
            _edit_json(ev, lambda d: d.update(git="something-else"))
        elif what == "config":
            _edit_json(ev, lambda d: d.update(config_sha256="0" * 64))
        elif what == "report":
            _edit_json(ev, lambda d: d.update(report_sha256="0" * 64))
        elif what == "excluded":
            _edit_json(ev, lambda d: d.update(excluded=["instance_00000.npz"]))
        elif what == "state":
            _edit_json(ev, lambda d: d["states"]["labels_s1"].update(sha256="0" * 64))
        elif what == "summary":
            _edit_json(ev, lambda d: d["summary"]["ar_s0"].update(disp_rel_l2=0.5))
        elif what == "worse":
            _edit_json(ev, lambda d: d["worse_than_zero"]["ar_s1"].update(worse=7))
        elif what == "content":
            _edit_json(ev, lambda d: d["content_checks"]["mgn_s2"].update(ok=False))
        elif what in ("order", "rows", "npz_seeds", "npz_metrics", "val_names", "loads",
                      "pi_star"):
            def f(d):
                if what == "order":
                    d["files"] = d["files"][::-1]
                elif what == "rows":
                    d["rows"] = d["rows"][::-1]
                elif what == "npz_seeds":
                    d["seeds"] = d["seeds"][::-1]
                elif what == "npz_metrics":
                    d["metrics"] = d["metrics"][::-1]
                elif what == "val_names":
                    d["val_files"] = d["val_files"][::-1]
                elif what == "loads":
                    d["gap_load"] = d["gap_load"][..., :3]
                    d["pi_load"] = d["pi_load"][..., :3]
                else:
                    d["pi_star"] = d["pi_star"][:, :3]
            _edit_npz(arr, f)
            _fix_arrays_sha(ret, "h2d")
        elif what == "arrays_hash":
            _edit_npz(arr, lambda d: d.update(values=d["values"] * 2.0))   # other values
        elif what == "residual":
            _edit_json(man, lambda d: d["instances"][0].update(residual=1e-3))
            _fix_manifest_sha(ret, "h2d")
        elif what == "names":
            _edit_json(man, lambda d: d["instances"].reverse())
            _fix_manifest_sha(ret, "h2d")
        elif what == "manifest_key":
            _edit_json(man, lambda d: d.update(used_prefix=1))
            _fix_manifest_sha(ret, "h2d")

    if what == "unlisted":
        ret = _return(box, tmp_path)
        _edit_json(ret / "h3d" / "eval.json", lambda d: d.update(note="x"))
        r = _adjudicate(box, ret, tmp_path / "verdict.json")
    else:
        r = _altered(box, tmp_path, edit)
    assert r.returncode != 0, what
    assert REFUSAL[what] in r.stderr, (what, r.stderr[-500:])
    assert not (tmp_path / "verdict.json").exists()


@pytest.mark.parametrize("what", ["status_edited", "extra_file", "listed_missing",
                                  "status_unlisted", "commit", "tree"])
def test_adjudication_refuses_a_return_that_is_not_the_one_listed(box, tmp_path, what):
    ret = _return(box, tmp_path)
    want = {"status_edited": "not the file the provenance file lists",
            "extra_file": "holds files provenance.txt does not list",
            "listed_missing": "lists files the return lacks",
            "status_unlisted": "does not list status.txt",
            "commit": "the return records HEAD",
            "tree": f"and tree {'0' * 40}, HEAD is"}[what]
    if what == "status_edited":
        with open(ret / "status.txt", "a") as fh:
            fh.write("eval_h2d.log exit=0\n")
    elif what == "extra_file":
        (ret / "notes.txt").write_text("x")
    elif what == "listed_missing":
        (ret / "precheck.txt").unlink()
    elif what == "status_unlisted":
        text = (ret / "provenance.txt").read_text().splitlines()
        (ret / "provenance.txt").write_text(
            "\n".join(x for x in text if not x.endswith("./status.txt")) + "\n")
    elif what == "tree":
        _provenance(ret, tree="0" * 40)
    else:
        _provenance(ret, head="0" * 40)
    r = _adjudicate(box, ret, tmp_path / "verdict.json")
    assert r.returncode != 0 and want in r.stderr, (what, r.stderr[-500:])
    assert not (tmp_path / "verdict.json").exists()


def test_adjudication_refuses_a_report_that_is_not_the_configurations(box, tmp_path):
    """The report a configuration names, changed in the adjudicating checkout
    after the run: refused."""
    ret = _return(box, tmp_path)
    rp = Path(box["spec"]["sets"]["h2d"]["report"])
    keep = rp.read_bytes()
    try:
        rp.write_bytes(keep + b" ")
        r = _adjudicate(box, ret, tmp_path / "v.json")
    finally:
        rp.write_bytes(keep)
    assert r.returncode != 0 and "not the report the configuration names" in r.stderr
    assert not (tmp_path / "v.json").exists()


def test_adjudication_refuses_an_unstamped_prereg(box, tmp_path):
    ret = _return(box, tmp_path)
    pr = tmp_path / "PREREG_HOLDOUT.md"
    pr.write_text(box["prereg"].read_text().replace("PREREG_HOLDOUT_SHA256 = ",
                                                    "PREREG_HOLDOUT_SHA256 = 0"))
    r = _adjudicate(box, ret, tmp_path / "v.json", prereg=pr)
    assert r.returncode != 0 and "refused:" in r.stderr and "SHA-256" in r.stderr
    assert not (tmp_path / "v.json").exists()


def test_adjudication_refuses_a_prereg_stamped_for_another_configuration(box, tmp_path):
    """The CONFIG_SHA256 line of another configuration, with the last line
    recomputed so that the file's own hash verifies."""
    import re

    stamp = _script("stamp_prereg_holdout")
    ret = _return(box, tmp_path)
    text = box["prereg"].read_text(encoding="utf-8")
    other = hashlib.sha256(b"another configuration").hexdigest()
    text = re.sub(r"^CONFIG_SHA256\[holdout_v1\] = [0-9a-f]{64}$",
                  f"CONFIG_SHA256[holdout_v1] = {other}", text, flags=re.M)
    blank = re.sub(r"^PREREG_HOLDOUT_SHA256 = .+$", stamp.FOOTER_OPEN, text, flags=re.M)
    own = hashlib.sha256(blank.encode("utf-8")).hexdigest()
    pr = tmp_path / "PREREG_HOLDOUT.md"
    pr.write_text(blank.replace(stamp.FOOTER_OPEN, f"PREREG_HOLDOUT_SHA256 = {own}"),
                  encoding="utf-8")
    assert stamp.self_hash_ok(pr.read_text(encoding="utf-8"))
    r = _adjudicate(box, ret, tmp_path / "v.json", prereg=pr)
    assert r.returncode != 0 and "refused:" in r.stderr
    assert "prereg_guard: config hash mismatch" in r.stderr
    assert not (tmp_path / "v.json").exists()


RELABELLED = "".join(f"{k} start 2026-10-18T10:0{i}:00Z box1\n{k} exit=0\n"
                     for i, k in enumerate(STAGES)) + \
    "label_h2d.log start 2026-10-18T10:09:00Z box1\nlabel_h2d.log exit=0\n"
"""A status file with the h2d labelling run again after its evaluation."""
RECOVERED = "".join(f"{k} start 2026-10-18T10:0{i}:00Z box1\n{k} exit=0\n"
                    for i, k in enumerate(STAGES[:2])) + (
    "label_h2d.log start 2026-10-18T10:02:00Z box1\n"                 # cut off (Ctrl-C)
    "eval_h2d.log start 2026-10-18T10:03:00Z box1\neval_h2d.log exit=1\n"
    "label_h2d.log start 2026-10-18T11:00:00Z box1\nlabel_h2d.log exit=0\n"
    "eval_h2d.log start 2026-10-18T11:01:00Z box1\neval_h2d.log exit=0\n")
"""E1' after a Ctrl-C in the h2d labelling: the labelling and the evaluation
run again in order; no labelling follows the evaluation that wrote the
returned file."""


@pytest.mark.parametrize("how", ["relabelled", "not_relabelled", "recovered",
                                 "and_another_config"])
def test_a_labels_manifest_other_than_the_evaluations(box, tmp_path, how):
    """A returned labels manifest that is not the one the evaluation read:
    after a labelling that status.txt records after the evaluation, the set
    is not adjudicable (its claims NOT EVALUATED), every other check of it
    still applying; without such a labelling, the return is refused."""
    ret = _return(box, tmp_path, status={"not_relabelled": None,
                                         "recovered": RECOVERED}.get(how, RELABELLED))
    _edit_json(ret / "h2d" / "manifest.json", lambda d: d.update(seconds=d["seconds"] + 1))
    if how == "and_another_config":
        _edit_json(ret / "h2d" / "eval.json", lambda d: d.update(config_sha256="0" * 64))
    _provenance(ret)
    out = tmp_path / "verdict.json"
    r = _adjudicate(box, ret, out)
    if how in ("not_relabelled", "recovered"):
        assert r.returncode != 0 and "status.txt records no labelling after the evaluation" \
            in r.stderr, r.stderr[-1000:]
        assert not out.exists()
        return
    if how == "and_another_config":
        assert r.returncode != 0 and "config_sha256 is" in r.stderr, r.stderr[-1000:]
        assert not out.exists()
        return
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    for c in ("H1", "H2a", "H2b", "H2c", "D2"):
        assert res["claims"][c]["verdict"] == "NOT EVALUATED"
        assert res["claims"][c]["why"].startswith("h2d: not adjudicable: its evaluation records")
    assert res["readings"]["H3"]["why"].startswith("h2d: not adjudicable")
    assert all(res["claims"][c]["verdict"] != "NOT EVALUATED" for c in ("T1", "T2", "T3"))
    assert any(d.startswith("h2d: not adjudicable: its evaluation records the labels manifest")
               and d.endswith("(a labelling after the evaluation)") for d in res["deviations"])
    assert "label_h2d.log: 2 attempts (exits [0, 0])" in res["deviations"]
    assert "h2d" not in res["secondary"] and "h2d" not in res["content_checks"]
    assert res["inputs"]["h2d"]["manifest.json"]["sha256"] == _sha(ret / "h2d" / "manifest.json")


def test_labelled_after_evaluation():
    mod = _script("adjudicate_holdout")
    assert mod.labelled_after_evaluation(RELABELLED, "h2d")
    assert not mod.labelled_after_evaluation(RELABELLED, "h3d")
    ordered = "".join(f"{k} start 2026-10-18T10:00:00Z h\n{k} exit=0\n" for k in STAGES)
    assert not mod.labelled_after_evaluation(ordered, "h2d")
    assert not mod.labelled_after_evaluation("label_h2d.log start x h\n", "h2d")   # no eval
    assert mod.labelled_after_evaluation(
        "eval_h3d.log start x h\neval_h3d.log exit=1\nlabel_h3d.log start y h\n", "h3d")
    assert not mod.labelled_after_evaluation(RECOVERED, "h2d")     # before the evaluation kept
    later = RECOVERED + "label_h2d.log start 2026-10-18T12:00:00Z box1\nlabel_h2d.log exit=0\n"
    assert mod.labelled_after_evaluation(later, "h2d")
    # a later evaluation refused (exit 3) leaves the earlier one's file returned
    refused = RELABELLED + "eval_h2d.log start 2026-10-18T10:10:00Z box1\neval_h2d.log exit=3\n"
    assert mod.labelled_after_evaluation(refused, "h2d")


@pytest.mark.parametrize("how", ["no_eval", "incomplete"])
def test_a_set_that_did_not_return_is_a_deviation(box, tmp_path, how):
    if how == "no_eval":
        ret = _return(box, tmp_path, sets=("h3d",))
        shutil.copy(Path(box["spec"]["out"]) / "h2d" / "manifest.json", ret / "h2d")
        status = "".join(f"{k} start 2026-10-18T10:00:00Z box1\n"
                         f"{k} exit={3 if k == 'eval_h2d.log' else 0}\n" for k in STAGES)
        (ret / "status.txt").write_text(status)
        _provenance(ret)
    else:
        ret = _return(box, tmp_path)
        _edit_json(ret / "h2d" / "eval.json", lambda d: d.pop("arrays_sha256"))
        _provenance(ret)
    out = tmp_path / "verdict.json"
    r = _adjudicate(box, ret, out)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert all(res["claims"][c]["verdict"] == "NOT EVALUATED"
               for c in ("H1", "H2a", "H2b", "H2c", "D2"))
    assert res["readings"]["H3"]["reading"] == "NOT EVALUATED"
    assert all(res["claims"][c]["verdict"] != "NOT EVALUATED" for c in ("T1", "T2", "T3"))
    assert any(d.startswith("h2d: not returned") for d in res["deviations"])
    assert "h2d" not in res["secondary"]
    if how == "no_eval":
        assert "eval_h2d.log: last exit 3" in res["deviations"]


def test_labelled_and_resumed_instances_and_attempts_are_deviations(box, tmp_path):
    ret = _return(box, tmp_path, status="".join(
        [f"{k} start 2026-10-18T10:00:00Z box1\n{k} exit=0\n" for k in STAGES[:3]]
        + ["eval_h2d.log start 2026-10-18T11:00:00Z box1\n",
           "eval_h2d.log start 2026-10-18T12:00:00Z box2\neval_h2d.log exit=0\n"]))
    def edit(d):
        d["instances"][0].update(corpus_labelled_flag=True)
        d["instances"][1].update(source_labelled=True)
        d.update(reused=1)

    _edit_json(ret / "h3d" / "manifest.json", edit)
    _fix_manifest_sha(ret, "h3d")
    _provenance(ret)
    out = tmp_path / "verdict.json"
    r = _adjudicate(box, ret, out)
    assert r.returncode == 0, r.stderr[-2000:]
    dev = json.loads(out.read_text())["deviations"]
    assert any("2 holdout instance(s) recorded as labelled" in d for d in dev)
    assert "h3d: labelling resumed (1 labels kept from an earlier attempt)" in dev
    assert "eval_h2d.log: 2 attempts (exits [0])" in dev
    assert "eval_h2d.log: 1 attempt(s) without an exit line" in dev
    assert "stages started on more than one host: ['box1', 'box2']" in dev


def test_a_state_that_does_not_reproduce_its_report(box, tmp_path):
    """Exit status 5 on the box; the claims that read the state NOT EVALUATED."""
    def edit(s):
        rp = Path(s["sets"]["h2d"]["report"])
        rep = json.loads(rp.read_text())
        pi = rep["results"]["e8"]["metrics"]["cells"]["labels"]["4"]["per_seed_eval"][1][
            "per_instance"]
        pi["disp_rel_l2"] = [2 * x for x in pi["disp_rel_l2"]]
        new = tmp_path / "report_h2d_changed.json"
        new.write_text(json.dumps(rep))
        s["sets"]["h2d"].update(report=str(new), report_sha256=_sha(new))

    cp, s = _restamped(box, tmp_path, edit)
    r = _run("--config", cp, "--set", "h2d", "--stage", "eval", "--device", "cpu")
    assert r.returncode == 5, r.stderr[-2000:]
    ev = json.loads((tmp_path / "again" / "h2d" / "eval.json").read_text())
    assert ev["content_mismatch"] == ["labels_s1"] and "arrays_sha256" in ev
    logs = {k: "" for k in STAGES}
    status = "".join(f"{k} start 2026-10-18T10:00:00Z box1\n"
                     f"{k} exit={5 if k == 'eval_h2d.log' else 0}\n" for k in STAGES)
    ret = _return(box, tmp_path / "r", out=tmp_path / "again", sets=("h2d",), logs=logs,
                  status=status)
    out = tmp_path / "verdict.json"
    r = _adjudicate(box, ret, out, spec_path=cp, prereg=Path(s["prereg_file"]))
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert all(res["claims"][c]["verdict"] == "NOT EVALUATED"
               for c in ("T1", "T2", "T3", "H1", "H2a", "H2b", "H2c", "D2"))
    assert res["readings"]["H3"]["reading"] != "NOT EVALUATED"       # reads ar and L_K only
    assert "h2d: content mismatch ['labels_s1']" in res["deviations"]
    assert "eval_h2d.log: last exit 5" in res["deviations"]
    assert "h3d: no return" not in res["deviations"]                  # its directory is there
    assert any(d.startswith("h3d: not returned") for d in res["deviations"])


@pytest.mark.parametrize("what", ["corpus", "labels"])
def test_eval_checks_every_file_against_the_labels_manifest(box, tmp_path, what):
    """A corpus file or a labels file whose bytes are not those the labels
    manifest records ends the evaluation before its arrays are written."""
    cp, s = _restamped(box, tmp_path)
    mp = tmp_path / "again" / "h3d" / "manifest.json"
    key = "source_sha256" if what == "corpus" else "labels_sha256"
    _edit_json(mp, lambda d: d["instances"][1].update({key: "0" * 64}))
    r = _run("--config", cp, "--set", "h3d", "--stage", "eval", "--device", "cpu")
    want = ("its bytes differ from those the label was solved from" if what == "corpus"
            else "its bytes differ from the labels manifest's record")
    assert r.returncode != 0 and want in r.stderr, r.stderr[-1000:]
    assert not (tmp_path / "again" / "h3d" / "eval_arrays.npz").exists()
    ev = json.loads((tmp_path / "again" / "h3d" / "eval.json").read_text())
    assert "arrays_sha256" not in ev


def test_non_finite_values_give_strict_json(box, tmp_path):
    """A return whose L_K values hold a NaN: adjudicated, the claims that
    read it divergent, the verdict strict JSON."""
    from fejepa.analysis.holdout import METRICS as M

    ret = _return(box, tmp_path)
    arr = ret / "h2d" / "eval_arrays.npz"
    ri = ROWS["h2d"].index("labels_knorm")

    def f(d):
        v = d["values"].copy()
        v[ri, 1, 0, M.index("vm_rel_l2")] = np.nan
        d["values"] = v

    _edit_npz(arr, f)
    with np.load(arr) as z:
        means = z["values"][ri, 1].mean(axis=0)
    _edit_json(ret / "h2d" / "eval.json", lambda d: d["summary"]["labels_knorm_s1"].update(
        {m: float(x) for m, x in zip(M, means)}))
    _fix_arrays_sha(ret, "h2d")
    _provenance(ret)
    out = tmp_path / "verdict.json"
    r = _adjudicate(box, ret, out)
    assert r.returncode == 0, r.stderr[-2000:]

    def strict(c):
        raise ValueError(f"non-strict JSON: {c}")

    res = json.loads(out.read_text(), parse_constant=strict)
    for c in ("H2b", "H2c"):
        assert res["claims"][c]["verdict"] == "NOT REPLICATED" and res["claims"][c]["diverged"]
    assert res["claims"]["H2a"]["verdict"] != "NOT EVALUATED"          # reads the gap only


def test_adjudicating_code_that_differs_is_a_deviation(box, tmp_path):
    """Adjudicated against a commit whose adjudicating files differ (the
    root commit, which has none of them): recorded as a deviation."""
    if any(subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", p],
                          capture_output=True).returncode != 0
           for p in ("scripts/adjudicate_holdout.py",
                     "src/fejepa/analysis/adjudicate_holdout.py")):
        pytest.skip("the adjudicating code is not committed in this checkout")
    root = _git("rev-list", "--max-parents=0", "HEAD").splitlines()[0]
    ret = _return(box, tmp_path)
    _provenance(ret, head=root, tree=_git("rev-parse", f"{root}^{{tree}}"))
    ev = json.loads((ret / "h3d" / "eval.json").read_text())
    out = tmp_path / "verdict.json"
    r = _run("--return", ret, "--config", box["spec_path"], "--prereg", box["prereg"],
             "--expected-git", ev["git"], "--code-ref", root, "--out", out, script=ADJ)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert any(d.startswith(f"adjudicating code differs from {root} in ")
               and "scripts/adjudicate_holdout.py" in d for d in res["deviations"])
    assert res["adjudicator"]["against_stamp"]["compared"]


def test_status_deviations():
    mod = _script("adjudicate_holdout")
    ok = "".join(f"{k} exit=0\n" for k in mod.STAGES)                 # without start lines
    assert mod.status_deviations(ok) == []
    twice = ok + "eval_h2d.log exit=0\n"
    assert any("2 attempts" in d for d in mod.status_deviations(twice))
    bad = ok.replace("eval_h3d.log exit=0", "eval_h3d.log exit=5")
    assert any("last exit 5" in d for d in mod.status_deviations(bad))
    assert any("no exit line" in d for d in mod.status_deviations("label_h3d.log exit=0\n"))
    started = "".join(f"{k} start 2026-10-18T10:00:00Z h\n{k} exit=0\n" for k in mod.STAGES)
    assert mod.status_deviations(started) == []
    cut = started.replace("eval_h2d.log exit=0\n", "")
    assert mod.status_deviations(cut) == ["eval_h2d.log: no exit line in status.txt "
                                          "(1 start line(s))"]
    assert mod.status_deviations(started + "other.log exit=0\n") == \
        ["other.log: a stage outside the runbook"]
    # read by attempt: a stage killed (exit 137) and then cut off ends cut off
    killed = started.replace("eval_h2d.log exit=0\n", "eval_h2d.log exit=137\n"
                             "eval_h2d.log start 2026-10-18T11:00:00Z h\n")
    assert mod.status_deviations(killed) == [
        "eval_h2d.log: 2 attempts (exits [137])", "eval_h2d.log: 1 attempt(s) without an exit line",
        "eval_h2d.log: last attempt cut off (no exit line)"]
    again = killed + "eval_h2d.log exit=0\n"
    assert mod.status_deviations(again) == ["eval_h2d.log: 2 attempts (exits [137, 0])"]


def test_check_commit():
    mod = _script("adjudicate_holdout")
    head, tree = _git("rev-parse", "HEAD"), _git("rev-parse", "HEAD^{tree}")
    assert mod.check_commit(f"HEAD {head}\ntree {tree}\n", "HEAD") is None
    with pytest.raises(ValueError, match="the return records HEAD"):
        mod.check_commit(f"HEAD {'0' * 40}\ntree {tree}\n", "HEAD")
    assert "no such tag" in mod.check_commit(f"HEAD {head}\ntree {tree}\n",
                                             "no-such-tag-holdout")
