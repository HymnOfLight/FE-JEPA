"""wp9 Stages 0d and 0e: the cost-table timing
(scripts/time_inference_vs_solve.py) runs end to end on small runs shaped like
the real ones (3D: validation split and fine set; 2D: validation split and an
OOD-2D family), times what it says, checks the timed model against the
report's own arrays, respects the solve budget, and refuses a family other
than the one named; the accuracy-matched CG (Stage 0e) stops at the first
step as accurate as the surrogate."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}
SCRIPT = ROOT / "scripts" / "time_inference_vs_solve.py"


def _states(tmp_path, model, seeds=(0, 1)):
    import torch

    from fejepa.experiments.parallel import _build_model

    sdir = tmp_path / "states"
    sdir.mkdir()
    rec = {}
    for s in seeds:
        m = _build_model({"kind": "fejepa", "model": model, "seed": s})
        p = sdir / f"ar_p4_s{s}.pt"
        torch.save(m.state_dict(), p)
        rec[f"s{s}"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "reused": False}
    return sdir, rec


def _run_3d(tmp_path):
    """A labelled in-band set, a labelled fine set, untrained states and a
    report that records them (the shape of Phase-2b's)."""
    from fejepa.data.archive import manifest_sha256
    from fejepa.fe.tet3d import generate_tet3d_dataset
    from fejepa.report import config_sha256

    inb = generate_tet3d_dataset(tmp_path / "inband", n=7, seed=1, labelled="all")
    fine = generate_tet3d_dataset(tmp_path / "fine", n=3, seed=2, labelled="all", nx=6, ny=4, nz=4)
    model = {"dim": 16, "depth": 1, "heads": 2,
             "features": {"load_summary": True, "geometry": True, "spatial_dim": 3}}
    cfg = {"model": model, "tf32": False, "split": {"n_val": 3, "seed": 1},
           "data": {"dir": str(inb)}, "data_transfer": {"dir": str(fine), "split": {"n_eval": 2}},
           "experiments": {"e8": {"pool_sizes": [4]}}}
    sdir, rec = _states(tmp_path, model)
    report = {"config": cfg,
              "provenance": {"config_sha256": config_sha256(cfg), "seeds": [0, 1],
                             "datasets": [{"dir": str(inb), "manifest_sha256": manifest_sha256(inb)},
                                          {"dir": str(fine), "manifest_sha256": manifest_sha256(fine)}]},
              "results": {"e8": {"metrics": {"d9_restart": {"ar_states": rec}}}}}
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(report))
    return rp, sdir


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args), "--device", "cpu"],
                          capture_output=True, text=True, env=ENV)


def test_times_both_sides_on_every_set(tmp_path):
    rp, sdir = _run_3d(tmp_path)
    out = tmp_path / "t.json"
    r = _run("--report", rp, "--states-dir", sdir, "--n-val", 3, "--n-fine", 2, "--repeats", 2,
             "--warmup", 1, "--check", 2, "--out", out)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.count("[timing] val ") == 3 and r.stdout.count("[timing] fine ") == 2
    res = json.loads(out.read_text())
    assert set(res["results"]) == {"val", "fine"} and res["seed"] == 0
    assert res["sets"]["val"]["n"] == 3 and res["sets"]["fine"]["n"] == 2
    assert res["settings"]["solvers"] == ["direct", "cg", "cg_warm", "cg_match", "cg_match_cstar"]
    assert res["machine"]["device"] == "cpu" and res["machine"]["torch"]
    assert res["state_sha256"] == hashlib.sha256((sdir / "ar_p4_s0.pt").read_bytes()).hexdigest()
    for name, n in (("val", 3), ("fine", 2)):
        s = res["results"][name]["summary"]
        assert s["n_instances"] == n and s["complete"] is True
        for k in ("direct", "cg", "cg_warm"):
            assert s[f"{k}_n_solved"] == n and s[f"{k}_label_max_rel_dev_max"] < 1e-6
        assert s["direct_label_max_rel_dev_max"] < 1e-10           # the labels' own solver
        assert s["cg_fallbacks"] == 0 and s["cg_iters_per_load"]["min"] > 0
        assert s["disp_rel_l2"]["n"] == n and s["energy_gap_rel"]["n"] == n
        for k in ("cg_match", "cg_match_cstar"):
            assert s[f"{k}_n_solved"] == n and s[f"{k}_unreached"] == 0
            assert s[f"{k}_iters_mismatch"] == 0
            assert s[f"{k}_label_max_rel_dev_max"] is None              # no full solution
        assert s["energy_gap_rel_cstar"]["n"] == n and s["cstar_s"]["n"] == n
        for row in res["results"][name]["per_instance"]:
            assert len(row["fwd_s"]) == 2 and row["fwd_first_s"] == row["fwd_s"][0]
            assert row["fwd_warm_s"] == row["fwd_s"][1]
            assert np.isclose(row["surrogate_cold_s"],
                              row["prep_s"] + row["fwd_first_s"] + row["d2h_s"])
            assert np.isclose(row["surrogate_warm_s"],
                              row["prep_s"] + row["fwd_warm_s"] + row["d2h_s"])
            for k in ("direct", "cg", "cg_warm"):
                assert row[f"{k}_s"] > 0 and len(row[f"{k}_runs"]) == 3   # fast: repeated
                assert row[f"{k}_s"] == sorted(row[f"{k}_runs"])[1]
                assert np.isclose(row[f"{k}_over_surrogate_cold"],
                                  row[f"{k}_s"] / row["surrogate_cold_s"])
            assert len(row["cg_iters"]) == row["n_loads"] == len(row["cg_warm_iters"])
            assert "cg_warm_iteration_saving" in row
            for k in ("cg_match", "cg_match_cstar"):
                assert len(row[f"{k}_iters"]) == row["n_loads"] == len(row[f"{k}_targets"])
                assert all(0 <= m <= c for m, c in zip(row[f"{k}_iters"], row["cg_iters"]))
                assert row[f"{k}_s"] > 0 and len(row[f"{k}_runs"]) == 3
                assert row[f"{k}_timed_iters"] == row[f"{k}_iters"]
                assert row[f"{k}_iters_mismatch"] == 0
                assert set(row[f"{k}_trace_end"]) <= {"zero", "target"}
                assert all((e == "zero") == (m == 0) for e, m in
                           zip(row[f"{k}_trace_end"], row[f"{k}_iters"]))
                assert len(row[f"{k}_disp_at_match"]) == row["n_loads"]
            # the rescaled prediction is never less accurate, so it needs as many steps or more
            assert all(c <= r * (1 + 1e-9) + 1e-12 for c, r in
                       zip(row["cg_match_cstar_targets"], row["cg_match_targets"]))
            assert all(c >= r for c, r in zip(row["cg_match_cstar_iters"], row["cg_match_iters"]))
            assert all(t <= 1 + 1e-9 for t in row["cg_match_cstar_targets"])
            assert row["energy_gap_rel_cstar"] <= row["energy_gap_rel"] * (1 + 1e-9) + 1e-12
            assert np.isclose(row["cg_match_cstar_over_surrogate_cold"],
                              row["cg_match_cstar_s"] / (row["surrogate_cold_s"] + row["cstar_s"]))
            assert 0 < row["n_free_dof"] < 3 * row["n_nodes"]
    fine_dof = res["results"]["fine"]["summary"]["n_free_dof"]["median"]
    assert fine_dof > res["results"]["val"]["summary"]["n_free_dof"]["median"]
    assert res["check_against_report"]["checked"] is False      # no arrays in this report


def test_the_timed_model_is_checked_against_the_reports_arrays(tmp_path):
    from fejepa.analysis.posthoc import run_files, run_model
    from fejepa.data.archive import load_instance
    from fejepa.metrics import evaluate_model, torch_predictor

    rp, sdir = _run_3d(tmp_path)
    report = json.loads(rp.read_text())
    per_seed = []
    for s in (0, 1):                     # the arrays as the evaluation computes them
        m = run_model(report, sdir / f"ar_p4_s{s}.pt", s, "cpu")
        ev = evaluate_model(torch_predictor(m, "cpu"),
                            [load_instance(f) for f in run_files(report, "val")])
        per_seed.append({"per_instance": ev["per_instance"]})
    report["results"]["e8"]["metrics"]["cells"] = {"ar": {"4": {"per_seed_eval": per_seed}}}
    rp.write_text(json.dumps(report))
    for seed in (0, 1):
        out = tmp_path / f"c{seed}.json"
        r = _run("--report", rp, "--states-dir", sdir, "--seed", seed, "--n-val", 3,
                 "--n-fine", 0, "--repeats", 1, "--warmup", 0, "--check", 3,
                 "--solvers", "direct", "--out", out)
        assert r.returncode == 0, r.stderr[-2000:]
        chk = json.loads(out.read_text())["check_against_report"]
        assert chk["checked"] and chk["n"] == 3
        assert chk["disp_rel_l2_max_rel_dev"] < 1e-6 and chk["energy_gap_rel_max_rel_dev"] < 1e-6
    per_seed[0]["per_instance"]["disp_rel_l2"][1] *= 1.5          # another model's arrays
    rp.write_text(json.dumps(report))
    out = tmp_path / "c.json"
    r = _run("--report", rp, "--states-dir", sdir, "--n-val", 3, "--n-fine", 0, "--warmup", 0,
             "--check", 3, "--solvers", "direct", "--out", out)
    assert r.returncode == 0
    assert json.loads(out.read_text())["check_against_report"]["disp_rel_l2_max_rel_dev"] > 0.3


def _family(out: Path, family: str, n: int, seed: int) -> Path:
    from fejepa.data.archive import save_instance, write_manifest
    from fejepa.fe.synthetic import synthetic_instance

    out.mkdir(parents=True)
    recs = []
    for i in range(n):
        a = synthetic_instance(np.random.default_rng(seed + i), nx=9, ny=6, labelled=True)
        name = f"instance_{i:05d}.npz"
        save_instance(a, out / name)
        recs.append({"file": name, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
                     "n_nodes": a.n_nodes, "labelled": True})
    write_manifest(out, recs, {"family": family, "seed": seed})
    return out


def _run_2d(tmp_path):
    from fejepa.data.archive import manifest_sha256
    from fejepa.fe.synthetic import generate_synthetic_dataset
    from fejepa.report import config_sha256

    d = generate_synthetic_dataset(tmp_path / "d2", n=10, seed=3, labelled="all")
    model = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True}}
    cfg = {"model": model, "tf32": False, "split": {"n_val": 4, "seed": 1},
           "data": {"dir": str(d)}, "experiments": {"e8": {"pool_sizes": [4]}}}
    sdir, rec = _states(tmp_path, model, seeds=(0,))
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps({"config": cfg, "provenance": {
        "config_sha256": config_sha256(cfg), "seeds": [0],
        "datasets": [{"dir": str(d), "manifest_sha256": manifest_sha256(d)}]},
        "results": {"e8": {"metrics": {"d9_restart": {"ar_states": rec}}}}}))
    return rp, sdir


def test_2d_family_budget_and_refusals(tmp_path):
    from fejepa.data.archive import manifest_sha256

    rp, sdir = _run_2d(tmp_path)
    f5 = _family(tmp_path / "F5", "F5", 3, 50)
    out = tmp_path / "t2.json"
    r = _run("--report", rp, "--states-dir", sdir, "--family", f"F5={f5}", "--n-family", 2,
             "--n-val", 3, "--solve-budget-s", 0, "--warmup", 0, "--out", out)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert set(res["results"]) == {"val", "F5"}                   # no fine set in a 2D run
    assert res["sets"]["F5"]["manifest_sha256"] == manifest_sha256(f5)
    for name, n in (("val", 3), ("F5", 2)):
        s = res["results"][name]["summary"]
        assert s["n_instances"] == n and s["surrogate_cold_s"]["n"] == n
        for k in ("direct", "cg", "cg_warm", "cg_match", "cg_match_cstar"):
            assert s[f"{k}_n_solved"] == 0 and s[f"{k}_s"] is None
        for row in res["results"][name]["per_instance"]:
            assert row["direct_s"] is None and row["direct_skipped"] == "budget"
            assert row["disp_rel_l2"] >= 0                        # accuracy without solves
    r = _run("--report", rp, "--states-dir", sdir, "--family", f"F4={f5}", "--n-val", 2,
             "--warmup", 0, "--solvers", "direct", "--out", out)
    assert r.returncode == 3 and "holds family 'F5', not 'F4'" in r.stdout
    res = json.loads(out.read_text())                             # the rest still timed
    assert set(res["results"]) == {"val"} and "F4" in res["skipped_families"]
    r = _run("--report", rp, "--states-dir", sdir, "--n-val", 2, "--warmup", 0, "--solvers", "cg",
             "--cg-maxiter", 1, "--out", out)                     # CG cut short: recorded
    assert r.returncode == 0, r.stderr[-2000:]
    s = json.loads(out.read_text())["results"]["val"]["summary"]
    assert s["cg_fallbacks"] == 2 and s["cg_label_max_rel_dev_max"] < 1e-10
    r = _run("--report", rp, "--states-dir", sdir, "--n-val", 0, "--warmup", 0, "--out", out)
    assert r.returncode == 0
    assert json.loads(out.read_text())["results"]["val"] == {"skipped": "no instances"}
    rp3, _ = _run_3d(tmp_path / "three")
    r = _run("--report", rp3, "--states-dir", tmp_path / "three" / "states",
             "--family", f"F5={f5}", "--out", out)
    assert r.returncode != 0 and "differ in spatial dimension" in r.stderr
    (sdir / "ar_p4_s0.pt").write_bytes((sdir / "ar_p4_s0.pt").read_bytes() + b"x")
    r = _run("--report", rp, "--states-dir", sdir, "--out", out)
    assert r.returncode != 0 and "is not the state the report trained" in r.stderr


def test_summary_statistics():
    spec = importlib.util.spec_from_file_location("tivs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [{"n_nodes": 10 * i, "n_free_dof": 18 * i, "prep_s": float(i), "direct_s": None}
            for i in range(1, 6)]
    s = mod.summarise(rows, ("direct",))
    assert s["n_instances"] == 5 and s["direct_n_solved"] == 0 and s["direct_s"] is None
    assert s["prep_s"]["median"] == 3.0 and s["prep_s"]["min"] == 1.0 and s["prep_s"]["n"] == 5
    assert s["n_free_dof"] == {"median": 54.0, "min": 18, "max": 90}
    assert s["direct_label_max_rel_dev_max"] is None
    assert mod.summarise([])["n_free_dof"] is None


def test_accuracy_matched_cg_stops_at_the_first_step_as_accurate():
    """Stage 0e: the trace is CG's own energy error, the match is the first
    step at or below the target, one trace serves both kinds and stops at
    the smaller target, and the timed CG of that many steps reaches it (the
    same CG, deterministic)."""
    import scipy.sparse as sp

    from fejepa.analysis.posthoc import amplitude_factor, apply_amplitude
    from fejepa.fe.solve import cg_k_steps
    from fejepa.fe.synthetic import synthetic_instance

    spec = importlib.util.spec_from_file_location("tivs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = synthetic_instance(np.random.default_rng(7), nx=10, ny=7, labelled=True)
    free = np.asarray(a.free_mask, dtype=bool)
    kff = sp.csr_matrix(a.K)[free][:, free]
    traces = []
    for j in range(a.n_loads):
        tr, dtr, end = mod.energy_trace(kff, a.F[j][free], a.U_star[j][free], 1e-10, 20000)
        assert end == "converged" and tr.size == dtr.size > 12
        assert np.all(np.diff(tr) <= 1e-12) and tr[-1] < 1e-12
        traces.append(tr)
        part, _d, end = mod.energy_trace(kff, a.F[j][free], a.U_star[j][free], 1e-10, 20000,
                                         stop_at=tr[5])
        assert end == "target" and part.size == 6 and np.allclose(part, tr[:6])
        _t, _d, end = mod.energy_trace(kff, a.F[j][free], a.U_star[j][free], 1e-10, 3)
        assert end == "maxiter"
    raw = [3 + 2 * j for j in range(a.n_loads)]
    fine = [k + 4 for k in raw]
    for tr, k in zip(traces, raw):
        assert mod.match_steps(tr, tr[k - 1]) == k                 # strictly decreasing
        assert mod.match_steps(tr, 0.5 * (tr[k - 1] + tr[k - 2])) == k
        assert mod.match_steps(tr, 1.0) == 0 and mod.match_steps(tr, 7.0) == 0
        assert mod.match_steps(tr, float("nan")) == 0
        assert mod.match_steps(tr, 0.5 * tr[-1]) is None
    targets = {"cg_match": [tr[k - 1] for tr, k in zip(traces, raw)],
               "cg_match_cstar": [tr[k - 1] for tr, k in zip(traces, fine)]}
    found, secs = mod.match_steps_for(a, targets, 1e-10, 20000)
    assert secs > 0
    assert found["cg_match"]["steps"] == raw and found["cg_match_cstar"]["steps"] == fine
    assert found["cg_match"]["unreached"] == found["cg_match_cstar"]["unreached"] == 0
    assert set(found["cg_match"]["end"]) == {"target"}
    assert all(0 < d < 1 for d in found["cg_match"]["disp"])
    for k, steps in (("cg_match", raw), ("cg_match_cstar", fine)):
        out = mod.match_time(k, a, steps, 1e-10)
        assert out[f"{k}_iters"] == out[f"{k}_timed_iters"] == steps
        assert out[f"{k}_iters_mismatch"] == 0 and out[f"{k}_s"] > 0
    for j, k in enumerate(raw):                                   # the timed CG's iterate
        u = cg_k_steps(a.K, a.F[j], a.free_mask, None, k)
        e = (u - a.U_star[j])[free]
        rel = float(e @ (kff @ e)) / float(a.U_star[j][free] @ (kff @ a.U_star[j][free]))
        assert rel <= traces[j][k - 1] * (1 + 1e-6)
    found, _ = mod.match_steps_for(a, {"cg_match": [0.5 * tr[-1] for tr in traces]}, 1e-10, 20000)
    assert found["cg_match"]["unreached"] == a.n_loads             # recorded, full CG counted
    assert found["cg_match"]["steps"] == [len(tr) for tr in traces]
    assert set(found["cg_match"]["end"]) == {"converged"}
    found, _ = mod.match_steps_for(a, {"cg_match": [2.0] * a.n_loads}, 1e-10, 20000)
    assert found["cg_match"]["steps"] == [0] * a.n_loads
    assert set(found["cg_match"]["end"]) == {"zero"} and found["cg_match"]["disp"][0] == 1.0
    # c* with the full K equals the free-dof value (the prediction is zero on Dirichlet dofs)
    pred = a.U_star * np.array([[0.7], [1.3], [0.9], [1.1]])[:a.n_loads]
    pred[:, ~free] = 0.0
    secs, c, pc = mod.cstar_time(pred, a)
    assert secs > 0
    assert np.allclose(c, amplitude_factor(pred, a.K, a.F, free))
    assert np.allclose(pc, apply_amplitude(pred, c))


def test_matching_kinds_need_labels():
    """An unlabelled instance: the matching kinds are skipped as such, the
    full solves still run."""
    from fejepa.experiments.parallel import _build_model
    from fejepa.fe.synthetic import synthetic_instance

    spec = importlib.util.spec_from_file_location("tivs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = synthetic_instance(np.random.default_rng(3), nx=8, ny=6, labelled=False)
    m = _build_model({"kind": "fejepa", "model": {"dim": 16, "depth": 1, "heads": 2,
                                                  "features": {"load_summary": True}}, "seed": 0})
    m.eval()
    row, _pred = mod.time_instance(m, a, "cpu", 1, list(mod.SOLVERS), lambda k: True, 1e-10, 20000)
    for k in ("cg_match", "cg_match_cstar"):
        assert row[f"{k}_s"] is None and row[f"{k}_skipped"] == "no labels"
    assert row["cg_s"] > 0 and row["direct_s"] > 0 and "cstar_s" not in row
