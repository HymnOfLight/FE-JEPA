"""wp9 Stage 0e: the field export (scripts/export_fields.py) runs end to end on
a small run shaped like Phase-2b's (label-free, labels-only and graph-network
states; in-band validation split and fine set), selects the figure instances
by its fixed rules from the report's arrays, records per-load energies whose
identities hold (Pi > 0 exactly when the relative gap exceeds 1; the c*
rescaling never raises the gap and never leaves it above the zero field's),
checks every model by hash and by content, and leaves out a supervised state
that fails both while exporting the rest."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}
SCRIPT = ROOT / "scripts" / "export_fields.py"
MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 8, "mgn_depth": 1,
         "features": {"load_summary": True, "geometry": True, "spatial_dim": 3}}


def _save(model_kind: str, seed: int, path: Path, legacy: bool = False) -> str:
    import torch

    from fejepa.experiments.parallel import _build_model

    m = _build_model({"kind": model_kind, "model": MODEL, "seed": seed})
    if legacy:
        torch.save(m.state_dict(), path, _use_new_zipfile_serialization=False)
    else:
        torch.save(m.state_dict(), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arrays(report: dict, kind: str, state: Path, seed: int, which: str) -> dict:
    from fejepa.analysis.common import build_model_from_config
    from fejepa.analysis.posthoc import run_files
    from fejepa.data.archive import load_instance
    from fejepa.metrics import evaluate_model, torch_predictor

    m = build_model_from_config(dict(MODEL, kind=kind), state_path=str(state), seed=seed)
    ev = evaluate_model(torch_predictor(m, "cpu"),
                        [load_instance(f) for f in run_files(report, which)])
    return {"per_instance": ev["per_instance"]}


def _run_3d(tmp_path: Path):
    """A Phase-2b-shaped run: in-band and fine sets, the three models' states
    for two seeds, a report with their per-instance arrays and the AR hashes,
    and the supervised hash table."""
    from fejepa.data.archive import manifest_sha256
    from fejepa.fe.tet3d import generate_tet3d_dataset
    from fejepa.report import config_sha256

    inb = generate_tet3d_dataset(tmp_path / "inband", n=9, seed=1, labelled="all")
    fine = generate_tet3d_dataset(tmp_path / "fine", n=4, seed=2, labelled="all", nx=6, ny=4, nz=4)
    cfg = {"model": MODEL, "tf32": False, "split": {"n_val": 5, "seed": 1},
           "data": {"dir": str(inb)}, "data_transfer": {"dir": str(fine), "split": {"n_eval": 3}},
           "experiments": {"e8": {"pool_sizes": [4], "budgets": [2, 4]}}}
    sdir = tmp_path / "states"
    sdir.mkdir()
    ar_rec, sup = {}, {}
    for s in (0, 1):
        ar_rec[f"s{s}"] = {"sha256": _save("fejepa", s, sdir / f"ar_p4_s{s}.pt"), "reused": False}
        sup[f"labels_b4_s{s}.pt"] = _save("fejepa", 10 + s, sdir / f"labels_b4_s{s}.pt")
        sup[f"mgn_b4_s{s}.pt"] = _save("mgn", 20 + s, sdir / f"mgn_b4_s{s}.pt")
    report = {"config": cfg,
              "provenance": {"config_sha256": config_sha256(cfg), "seeds": [0, 1],
                             "datasets": [{"dir": str(inb), "manifest_sha256": manifest_sha256(inb)},
                                          {"dir": str(fine), "manifest_sha256": manifest_sha256(fine)}]},
              "results": {"e8": {"metrics": {"d9_restart": {"ar_states": ar_rec}}}}}
    cells = {"ar": {"4": {"per_seed_eval": []}}, "labels": {"4": {"per_seed_eval": []}},
             "mgn": {"4": {"per_seed_eval": []}}}
    fine_eval = []
    for s in (0, 1):
        cells["ar"]["4"]["per_seed_eval"].append(
            _arrays(report, "fejepa", sdir / f"ar_p4_s{s}.pt", s, "val"))
        cells["labels"]["4"]["per_seed_eval"].append(
            _arrays(report, "fejepa", sdir / f"labels_b4_s{s}.pt", s, "val"))
        cells["mgn"]["4"]["per_seed_eval"].append(
            _arrays(report, "mgn", sdir / f"mgn_b4_s{s}.pt", s, "val"))
        fine_eval.append(_arrays(report, "fejepa", sdir / f"ar_p4_s{s}.pt", s, "fine"))
    report["results"]["e8"]["metrics"]["cells"] = cells
    report["results"]["p3_transfer"] = {"metrics": {"ar": {"fine": {"per_seed_eval": fine_eval}}}}
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(report))
    hp = tmp_path / "sup.json"
    hp.write_text(json.dumps({"sha256": sup}))
    return rp, sdir, hp


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args), "--device", "cpu"],
                          capture_output=True, text=True, env=ENV)


def test_export_end_to_end(tmp_path):
    from fejepa.analysis.posthoc import run_files
    from fejepa.data.archive import load_instance
    from fejepa.fe.tet3d import tet_von_mises

    rp, sdir, hp = _run_3d(tmp_path)
    out = tmp_path / "fields"
    r = _run("--report", rp, "--states-dir", sdir, "--sup-hashes", hp, "--out", out)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((out / "fields.json").read_text())
    report = json.loads(rp.read_text())
    assert res["budget"] == 4 and res["pool"] == 4 and res["seeds"] == [0, 1]
    assert res["left_out"] == [] and res["content_mismatch"] == [] and res["val"]["n"] == 5
    assert res["torch"]["torch"] and res["torch"]["device"] == "cpu"
    for k, rec in res["states"].items():
        assert rec["sha256_ok"] and rec["used"], k
        assert rec["content"]["ok"] and rec["content"]["n"] == 5
        assert rec["content"]["disp_rel_l2_max_rel_dev"] < 1e-6, k   # the evaluation's numbers
        assert rec["content"]["energy_gap_rel_max_rel_dev"] < 1e-6, k
    # the rules, from the report's arrays
    cells = report["results"]["e8"]["metrics"]["cells"]
    lab0 = np.asarray(cells["labels"]["4"]["per_seed_eval"][0]["per_instance"]["energy_gap_rel"])
    ar0 = np.asarray(cells["ar"]["4"]["per_seed_eval"][0]["per_instance"]["energy_gap_rel"])
    fine0 = np.asarray(report["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["per_seed_eval"][0]
                       ["per_instance"]["disp_rel_l2"])
    figs = res["figures"]
    assert figs["fig5"]["index"] == int(np.argmax(lab0))
    assert len(figs["fig5"]["per_seed_choice"]) == 2
    assert figs["fig6"]["index"] == int(np.argsort(ar0, kind="stable")[2])      # (5 - 1) // 2
    assert figs["fig7"]["index"] == int(np.argsort(fine0, kind="stable")[1])    # (3 - 1) // 2
    val, fine = run_files(report, "val"), run_files(report, "fine")
    assert figs["fig5"]["file"] == Path(val[figs["fig5"]["index"]]).name
    assert figs["fig7"]["file"] == Path(fine[figs["fig7"]["index"]]).name
    # per-load energies and their identities
    E = np.load(out / "energies_val.npz")
    assert list(E["arms"]) == ["ar", "labels", "mgn"] and list(E["seeds"]) == [0, 1]
    L = E["pi_star"].shape[1]
    assert E["rel"].shape == (3, 2, 5, L) and not np.isnan(E["rel"]).any()
    assert np.allclose(E["rel"], (E["pi"] - E["pi_star"]) / np.abs(E["pi_star"]))
    assert np.array_equal(E["pi"] > 0, E["rel"] > 1)
    assert np.all(E["rel_c"] <= np.minimum(E["rel"], 1.0) * (1 + 1e-9) + 1e-12)
    assert res["counts"]["ar"]["load_cases"] == 2 * 5 * L
    assert res["counts"]["labels"]["load_cases_pi_positive"] == int(np.sum(E["pi"][1] > 0))
    for arm in ("ar", "labels", "mgn"):
        assert res["counts"][arm]["load_cases_rel_gap_above_1_after_cstar"] == 0
        assert res["counts"][arm]["load_cases_cstar_increased_gap"] == 0
    # the per-instance means agree with the report's arrays
    assert np.allclose(E["rel"][0, 0].mean(axis=-1), ar0)
    # a figure file: mesh, reference, predictions, stresses, energies
    F5 = np.load(out / "fig5.npz")
    a = load_instance(val[figs["fig5"]["index"]])
    assert np.array_equal(F5["tets"], a.elements) and np.allclose(F5["U_star"], a.U_star)
    for arm in ("ar", "labels", "mgn"):
        assert F5[f"U_{arm}"].shape == a.U_star.shape
        assert F5[f"vm_{arm}"].shape == (a.n_loads, a.elements.shape[0])
        fr = np.asarray(a.free_mask, dtype=bool)
        U = F5[f"U_{arm}"]
        kff = a.K.tocsr()[fr][:, fr]
        c = np.einsum("ld,ld->l", a.F[:, fr], U[:, fr]) / np.einsum("ld,ld->l", U[:, fr],
                                                                     (kff @ U[:, fr].T).T)
        assert np.allclose(F5[f"c_{arm}"], c)
    m = a.meta["material"]
    assert np.allclose(F5["vm_ref"][0], tet_von_mises(a.nodes, a.elements, a.U_star[0], m),
                       rtol=1e-5)
    meta = json.loads(str(F5["meta"]))
    assert meta["figure"] == "fig5" and meta["seed"] == 0 and meta["set"] == "val"
    assert json.loads(str(np.load(out / "fig7.npz")["meta"]))["set"] == "fine"
    # the validation figures reuse the energy pass's predictions: identical energies
    i5 = figs["fig5"]["index"]
    assert figs["fig5"]["from_energy_pass"] and not figs["fig7"]["from_energy_pass"]
    for ai, arm in enumerate(("ar", "labels", "mgn")):
        assert np.array_equal(F5[f"pi_{arm}"], E["pi"][ai, 0, i5])
    assert np.array_equal(F5["pi_star"], E["pi_star"][i5])


def test_supervised_states_by_hash_and_by_content(tmp_path):
    rp, sdir, hp = _run_3d(tmp_path)
    import torch

    # same weights, other bytes: the hash fails, the content passes -> used
    p0 = sdir / "labels_b4_s0.pt"
    torch.save(torch.load(p0, weights_only=True), p0, _use_new_zipfile_serialization=False)
    # other weights: both fail -> left out, the rest exported, exit 4
    _save("mgn", 99, sdir / "mgn_b4_s1.pt")
    out = tmp_path / "fields"
    r = _run("--report", rp, "--states-dir", sdir, "--sup-hashes", hp, "--out", out)
    assert r.returncode == 4, r.stderr[-3000:]
    res = json.loads((out / "fields.json").read_text())
    s0 = res["states"]["labels_s0"]
    assert not s0["sha256_ok"] and s0["content"]["ok"] and s0["used"]
    s1 = res["states"]["mgn_s1"]
    assert not s1["sha256_ok"] and not s1["content"]["ok"] and not s1["used"]
    assert res["left_out"] == ["mgn_s1"]
    E = np.load(out / "energies_val.npz")
    assert np.isnan(E["rel"][2, 1]).all() and not np.isnan(E["rel"][2, 0]).any()
    assert res["counts"]["mgn"]["instances"] == 5                 # seed 0 only
    assert set(res["figures"]) == {"fig5", "fig6", "fig7"}        # seed 0 models: all used
    assert "mgn" in res["figures"]["fig5"]["models"]


def test_content_mismatch_on_a_hash_verified_state(tmp_path):
    """A state used on its hash whose recomputed arrays differ from the
    report's is recorded and ends the run with exit 5 (after the outputs)."""
    rp, sdir, hp = _run_3d(tmp_path)
    rep = json.loads(rp.read_text())
    cell = rep["results"]["e8"]["metrics"]["cells"]["labels"]["4"]
    per = cell["per_seed_eval"][1]["per_instance"]
    per["disp_rel_l2"] = [1.1 * x for x in per["disp_rel_l2"]]
    per["energy_gap_rel"] = [1.1 * x for x in per["energy_gap_rel"]]
    rp.write_text(json.dumps(rep))
    out = tmp_path / "fields"
    r = _run("--report", rp, "--states-dir", sdir, "--sup-hashes", hp, "--out", out)
    assert r.returncode == 5, r.stderr[-3000:]
    res = json.loads((out / "fields.json").read_text())
    assert res["content_mismatch"] == ["labels_s1"] and res["left_out"] == []
    rec = res["states"]["labels_s1"]
    assert rec["sha256_ok"] and rec["used"] and not rec["content"]["ok"]
    assert abs(rec["content"]["disp_rel_l2_median_rel_dev"] - 0.1 / 1.1) < 1e-6
    assert (out / "fig7.npz").exists() and "content_median_rel_dev" in r.stdout


def test_figure_inputs_are_checked_before_the_long_pass(tmp_path):
    from fejepa.analysis.posthoc import run_files

    rp, sdir, hp = _run_3d(tmp_path)
    report = json.loads(rp.read_text())
    p3 = report["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["per_seed_eval"][0]
    fine0 = np.asarray(p3["per_instance"]["disp_rel_l2"])
    f7 = Path(run_files(report, "fine")[int(np.argsort(fine0, kind="stable")[1])])
    f7.rename(f7.with_suffix(".moved"))
    out = tmp_path / "fields"
    r = _run("--report", rp, "--states-dir", sdir, "--sup-hashes", hp, "--out", out)
    assert r.returncode != 0 and not (out / "energies_val.npz").exists()
    assert "[fields] val" not in r.stdout
    r = _run("--report", rp, "--states-dir", sdir, "--sup-hashes", hp, "--n-val", 0, "--out", out)
    assert r.returncode != 0 and "at least one validation instance" in r.stderr


def test_rules_and_hash_record_on_the_real_report():
    """The rules applied to Phase-2b's report give the instances the paper
    names, and the hash record covers exactly the states the script reads."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ef", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = json.loads((ROOT / "records/wp8/e2/baseline/report_phase2b.json").read_text())
    sel = mod.select(report)
    assert sel["fig5"]["index"] == 107 and sel["fig5"]["per_seed_choice"] == [107, 107, 107]
    assert sel["fig6"]["index"] == 154 and sel["fig7"]["index"] == 77
    assert mod.budgets(report) == ("1024", "1024")
    b, _pool = mod.budgets(report)
    rec = json.loads(mod.DEFAULT_SUP_HASHES.read_text())
    seeds = report["provenance"]["seeds"]
    want = {f"{arm}_b{b}_s{s}.pt" for arm in ("labels", "mgn") for s in seeds}
    assert set(rec["sha256"]) == want
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in rec["sha256"].values())


def test_refusals(tmp_path):
    rp, sdir, hp = _run_3d(tmp_path)
    out = tmp_path / "fields"
    (sdir / "ar_p4_s1.pt").write_bytes((sdir / "ar_p4_s1.pt").read_bytes() + b"x")
    r = _run("--report", rp, "--states-dir", sdir, "--sup-hashes", hp, "--out", out)
    assert r.returncode != 0 and "is not the state the report trained" in r.stderr
    rp2, sdir2, hp2 = _run_3d(tmp_path / "b")
    (sdir2 / "labels_b4_s0.pt").unlink()
    r = _run("--report", rp2, "--states-dir", sdir2, "--sup-hashes", hp2, "--out", out)
    assert r.returncode != 0 and "missing" in r.stderr
    rep = json.loads(rp2.read_text())
    del rep["results"]["p3_transfer"]
    rp2.write_text(json.dumps(rep))
    _save("fejepa", 10, sdir2 / "labels_b4_s0.pt")
    r = _run("--report", rp2, "--states-dir", sdir2, "--sup-hashes", hp2, "--out", out)
    assert r.returncode != 0 and "no P3 fine-set arrays" in r.stderr
