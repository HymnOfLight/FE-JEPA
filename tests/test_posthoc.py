"""wp8-lejepa Stage 1.36: the post-hoc readings (no training) are exact on
constructed cases, refuse states and corpora other than the report's, and
their scripts run end to end on small synthetic runs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}


def _tet_arch(seed=0, nx=4, ny=3, nz=3):
    from fejepa.fe.tet3d import tet_instance

    return tet_instance(np.random.default_rng(seed), nx=nx, ny=ny, nz=nz, labelled=True)


def test_amplitude_factor_is_the_energy_optimal_scale():
    from fejepa.anchor.energy import energy_gap
    from fejepa.analysis.posthoc import amplitude_factor, apply_amplitude, free_mask

    a = _tet_arch()
    free = free_mask(a)
    c = amplitude_factor(0.6 * a.U_star, a.K, a.F, free)
    assert np.allclose(c, 1 / 0.6, rtol=1e-10)                 # pure amplitude error: exact
    rng = np.random.default_rng(1)
    U = (0.7 * a.U_star + 0.3 * np.abs(a.U_star).max() * rng.standard_normal(a.U_star.shape)) \
        * free
    c = amplitude_factor(U, a.K, a.F, free)
    Uc = apply_amplitude(U, c)
    g0, g1 = energy_gap(U, a.U_star, a.K, a.F), energy_gap(Uc, a.U_star, a.K, a.F)
    assert np.all(g1 <= g0 + 1e-15)                             # never worse in energy
    for j in range(a.n_loads):                                  # stationarity of Pi(c u)
        u = U[j]
        assert np.isclose(a.F[j] @ u, c[j] * (u @ (a.K @ u)), rtol=1e-10)
        for dc in (-1e-3, 1e-3):                                # and a minimum
            gj = energy_gap((c[j] + dc) * u[None], a.U_star[j][None], a.K, a.F[j][None])[0]
            assert gj >= g1[j]
    zero = amplitude_factor(np.zeros_like(U), a.K, a.F, free)
    assert np.isnan(zero).all() and np.array_equal(apply_amplitude(U, zero), U)


def test_element_energy_is_an_exact_decomposition():
    from fejepa.analysis.posthoc import element_energy

    a = _tet_arch(2)
    e = np.random.default_rng(3).standard_normal(3 * a.n_nodes) * a.free_mask
    ee = element_energy(a.nodes, a.elements, e, a.meta["material"])
    assert np.all(ee >= -1e-14)
    assert np.isclose(ee.sum(), 0.5 * e @ (a.K @ e), rtol=1e-10)


def test_geometry_helpers():
    from fejepa.analysis.posthoc import cavity_distance, low_order_fraction, region_share

    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    d = cavity_distance(pts, [(0.0, 0.0, 0.0, 0.5)])
    assert np.allclose(d, [0.5, 1.5, 0.0])
    assert np.isinf(cavity_distance(pts, [])).all()
    x = np.random.default_rng(4).uniform(size=(400, 3))
    quad = np.stack([x[:, 0] ** 2 - x[:, 1] * x[:, 2], x[:, 2], 1 + x[:, 0]], axis=1)
    assert low_order_fraction(x, quad) > 1 - 1e-10
    noise = np.random.default_rng(5).standard_normal((400, 3))
    assert low_order_fraction(x, noise) < 0.1
    w = np.array([1.0, 3.0, 0.0])
    assert region_share(w, np.array([True, False, True])) == 0.25


def test_seed_margin_on_a_bottleneck_pack():
    torch = pytest.importorskip("torch")
    from fejepa.analysis.posthoc import seed_margin
    from fejepa.experiments.parallel import _build_model

    a = _tet_arch(6)
    m = _build_model({"kind": "bottleneck", "seed": 0, "model": {
        "dim": 16, "depth": 1, "heads": 2, "n_tokens": 8, "decode_k": 3,
        "features": {"load_summary": True, "geometry": True, "spatial_dim": 3}}})
    pack = m.prepare_instance(a, "cpu")
    mg = seed_margin(pack["nbr_rel"].numpy())
    assert mg.shape == (a.n_nodes,) and np.all(mg >= 0) and np.all(mg <= 1)
    assert int(np.isclose(mg, 1.0).sum()) == 8                 # the 8 seed nodes themselves
    assert isinstance(pack["nbr_rel"], torch.Tensor)


# ---- the scripts, end to end on a small synthetic run -----------------------------

def _fake_run(tmp_path, kind="fejepa"):
    """A labelled in-band set, a labelled 'fine' set, untrained states and a
    report that records them -- the shape of a real run's report."""
    import torch

    from fejepa.data.archive import manifest_sha256
    from fejepa.experiments.parallel import _build_model
    from fejepa.fe.tet3d import generate_tet3d_dataset
    from fejepa.report import config_sha256

    inb = generate_tet3d_dataset(tmp_path / "inband", n=7, seed=1, labelled="all")
    fine = generate_tet3d_dataset(tmp_path / "fine", n=3, seed=2, labelled="all", nx=6, ny=4, nz=4)
    model = {"dim": 16, "depth": 1, "heads": 2,
             "features": {"load_summary": True, "geometry": True, "spatial_dim": 3}}
    if kind == "bottleneck":
        model.update({"kind": "bottleneck", "n_tokens": 8, "decode_k": 3})
    cfg = {"model": model, "tf32": False, "split": {"n_val": 3, "seed": 1},
           "data": {"dir": str(inb)},
           "data_transfer": {"dir": str(fine), "split": {"n_eval": 2}},
           "experiments": {"e8": {"pool_sizes": [4]}}}
    sdir = tmp_path / "states"
    sdir.mkdir()
    states = {}
    for s in (0, 1):
        m = _build_model({"kind": kind, "model": model, "seed": s})
        p = sdir / f"ar_p4_s{s}.pt"
        torch.save(m.state_dict(), p)
        states[f"s{s}"] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "reused": False}
    report = {"config": cfg,
              "provenance": {"config_sha256": config_sha256(cfg), "seeds": [0, 1],
                             "datasets": [{"dir": str(inb), "manifest_sha256": manifest_sha256(inb)},
                                          {"dir": str(fine), "manifest_sha256": manifest_sha256(fine)}]},
              "results": {"e8": {"metrics": {"d9_restart": {"ar_states": states}}}}}
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(report))
    return rp, sdir


@pytest.mark.parametrize("kind", ["fejepa", "bottleneck"])
def test_amplitude_and_anatomy_scripts_run_and_never_worsen_the_energy(tmp_path, kind):
    pytest.importorskip("torch")
    rp, sdir = _fake_run(tmp_path, kind)
    out = tmp_path / "amp.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_amplitude.py"),
                        "--report", str(rp), "--states-dir", str(sdir), "--device", "cpu",
                        "--out", str(out)], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    assert set(res["seeds"]) == {"s0", "s1"} and set(res["seed_means"]) == {"inband", "fine"}
    for s in res["seeds"].values():
        assert s["inband"]["summary"]["n_instances"] == 3 and s["fine"]["summary"]["n_instances"] == 2
        for row in s["inband"]["per_instance"] + s["fine"]["per_instance"]:
            assert row["egap_c"] <= row["egap"] + 1e-12
    out2 = tmp_path / "anat.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_error_anatomy.py"),
                        "--report", str(rp), "--states-dir", str(sdir), "--device", "cpu",
                        "--n", "3", "--out", str(out2)], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    summ = json.loads(out2.read_text())["seeds"]["s0"]["inband"]["summary"]
    assert ("token_boundary" in summ) == (kind == "bottleneck")
    assert 0.0 <= summ["support"]["err_share_median"] <= 1.0


def test_scripts_refuse_a_state_or_corpus_other_than_the_reports(tmp_path):
    pytest.importorskip("torch")
    rp, sdir = _fake_run(tmp_path)
    p = sdir / "ar_p4_s1.pt"
    p.write_bytes(p.read_bytes() + b"x")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_amplitude.py"),
                        "--report", str(rp), "--states-dir", str(sdir), "--device", "cpu",
                        "--out", str(tmp_path / "x.json")], capture_output=True, text=True, env=ENV)
    assert r.returncode != 0 and "is not the state the report trained" in r.stderr
    rp2, sdir2 = _fake_run(tmp_path / "b")
    man = Path(json.loads(rp2.read_text())["config"]["data"]["dir"]) / "manifest.json"
    man.write_text(man.read_text() + " ")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_amplitude.py"),
                        "--report", str(rp2), "--states-dir", str(sdir2), "--device", "cpu",
                        "--out", str(tmp_path / "y.json")], capture_output=True, text=True, env=ENV)
    assert r.returncode != 0 and "is not the report's" in r.stderr


def test_random_init_probe_script_runs(tmp_path):
    pytest.importorskip("torch")
    from fejepa.data.archive import manifest_sha256
    from fejepa.fe.synthetic import generate_synthetic_dataset
    from fejepa.report import config_sha256

    d = generate_synthetic_dataset(tmp_path / "d2", n=12, seed=3, labelled="none")
    cfg = {"model": {"dim": 16, "depth": 1, "heads": 2,
                     "features": {"load_summary": True, "geometry": True}},
           "tf32": False, "split": {"n_val": 8, "seed": 1}, "data": {"dir": str(d)}}
    rep = {"config": cfg, "provenance": {"config_sha256": config_sha256(cfg),
                                         "datasets": [{"dir": str(d),
                                                       "manifest_sha256": manifest_sha256(d)}]}}
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(rep))
    out = tmp_path / "probe.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_probe_random_init.py"),
                        "--report", str(rp), "--seeds", "0", "--device", "cpu", "--out", str(out)],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    rd = json.loads(out.read_text())["readings"]
    assert set(rd) == {"geometry_input_true_s0", "geometry_input_false_s0"}
    assert json.loads(out.read_text())["n_instances"] == 8


def test_profile_script_runs_every_variant_on_cpu(tmp_path):
    pytest.importorskip("torch")
    from fejepa.fe.synthetic import generate_synthetic_dataset

    d = generate_synthetic_dataset(tmp_path / "d2", n=6, seed=4, labelled="all")
    cfg = {"model": {"dim": 16, "depth": 1, "heads": 2,
                     "features": {"load_summary": True, "geometry": True}},
           "tf32": False, "split": {"n_val": 2, "seed": 1}}
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(cfg))
    out = tmp_path / "prof.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_profile_2d.py"),
                        "--config", str(cp), "--data", str(d), "--n-inst", "2", "--e1", "1",
                        "--e2", "2", "--pairs", "1", "--device", "cpu", "--no-profile",
                        "--out", str(out)], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    v = json.loads(out.read_text())["variants"]
    assert set(v) == {"default", "threads_w3", "no_ckpt", "resident", "raw_sigreg"}
    for row in v.values():
        assert isinstance(row["ar"]["ms_per_step"], float) and "sup" in row
    assert isinstance(v["default"]["sup"]["ms_per_step"], float)
    assert isinstance(v["resident"]["sup"], str)                # anchors unused by the sup loss
    env = json.loads(out.read_text())
    assert env["cpus"]["usable"] >= 1 and env["fejepa_file"].startswith(str(ROOT / "src"))


def _script_module(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_amplitude_script_reads_exactly_the_runs_evaluation_sets(tmp_path):
    pytest.importorskip("torch")
    from fejepa.data.archive import instance_files
    from fejepa.experiments.protocol import load_split

    rp, sdir = _fake_run(tmp_path)
    cfg = json.loads(rp.read_text())["config"]
    out = tmp_path / "amp.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_amplitude.py"),
                        "--report", str(rp), "--states-dir", str(sdir), "--device", "cpu",
                        "--out", str(out)], capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(out.read_text())
    val = [p.name for p in load_split(cfg["data"]["dir"], 3, seed=1).val_files]
    fine = [p.name for p in instance_files(Path(cfg["data_transfer"]["dir"]))[:2]]
    got_val = [row["file"] for row in res["seeds"]["s0"]["inband"]["per_instance"]]
    got_fine = [row["file"] for row in res["seeds"]["s0"]["fine"]["per_instance"]]
    assert got_val == val and got_fine == fine
    row = res["seeds"]["s0"]["inband"]["per_instance"][0]
    assert {"c_battery", "u_norm_K", "ustar_norm_K", "cos_K", "fscale", "lc"} <= set(row)
    assert res["seeds"]["s0"]["inband"]["reproduction"] == {"checked": False}


def test_reproduction_compares_with_the_reports_per_instance_arrays():
    mod = _script_module("posthoc_amplitude")
    rows = [{"disp": 0.1, "egap": 0.01}, {"disp": 0.2, "egap": 0.04}]
    cell = {"per_seed_eval": [{"per_instance": {"disp_rel_l2": [0.1, 0.2, 9.0],
                                                "energy_gap_rel": [0.01, 0.0404, 9.0]}}]}
    rep = mod.reproduction(rows, cell, 0)
    assert rep["checked"] and rep["disp_max_rel_dev"] == 0.0
    assert rep["egap_max_rel_dev"] == pytest.approx(0.04 / 0.0404 - 1, abs=1e-12) or \
        rep["egap_max_rel_dev"] == pytest.approx(abs(0.04 - 0.0404) / 0.0404)


def test_anatomy_regions_on_a_gmsh_instance_with_cavities_and_gravity():
    pytest.importorskip("gmsh")
    from fejepa.fe.gmsh3d import gmsh3d_instance

    mod = _script_module("posthoc_error_anatomy")
    seen = {}
    for seed in range(40):                       # one instance with cavities, one without
        rng = np.random.default_rng(seed)
        a = gmsh3d_instance(rng, lc=0.16, labelled=True)
        key = a.meta["extra"]["n_holes"] > 0
        if key not in seen:
            seen[key] = a
        if len(seen) == 2:
            break
    rng = np.random.default_rng(1)
    for has_cav, a in seen.items():
        U = a.U_star * (1 + 0.2 * rng.standard_normal(a.U_star.shape)) * a.free_mask
        res = mod.anatomy(a, U, None)
        assert ("err_cavity" in res) == has_cav
        assert res["vol_load"] < 0.5                 # gravity (every node loaded) excluded
        assert 0.0 <= res["err_support"] <= 1.0 and "err_token_boundary" not in res
        assert 0.0 < res["ref_low_order"] <= 1.0


def test_remesh_meshes_each_geometry_once_for_every_seed(monkeypatch):
    pytest.importorskip("gmsh")
    torch = pytest.importorskip("torch")
    import fejepa.fe.gmsh3d as g3
    from fejepa.experiments.parallel import _build_model

    mod = _script_module("posthoc_amplitude")
    monkeypatch.setattr(mod, "REMESH_LC", (0.30, 0.22))
    calls, real = [], g3.gmsh3d_instance

    def counting(rng, lc=0.30, **kw):
        a = real(rng, lc=lc, **kw)
        calls.append((lc, a.meta["extra"]["holes"], a.meta["extra"]["width"]))
        return a

    monkeypatch.setattr(g3, "gmsh3d_instance", counting)
    mcfg = {"dim": 16, "depth": 1, "heads": 2,
            "features": {"load_summary": True, "geometry": True, "spatial_dim": 3}}
    models = {s: _build_model({"kind": "fejepa", "model": mcfg, "seed": s}).eval()
              for s in (0, 1)}
    with torch.no_grad():
        out = list(mod.remesh_rows(models, 2, "cpu"))
    assert len(calls) == 2 * 2                                  # geometries x lc, not x seeds
    for g in range(2):                                          # same geometry at every lc
        assert calls[2 * g][1:] == calls[2 * g + 1][1:]
    assert len(out) == 2 and set(out[0]) == {0, 1}
    for g_rows in out:
        for s, row in g_rows.items():
            by = row["by_lc"]
            assert [r["lc"] for r in by] == [0.30, 0.22]
            assert by[1]["n_nodes"] > by[0]["n_nodes"]
            assert by[1]["fscale"] < by[0]["fscale"]            # the load scale shrinks
        assert g_rows[0]["by_lc"][0]["fscale"] == g_rows[1]["by_lc"][0]["fscale"]
