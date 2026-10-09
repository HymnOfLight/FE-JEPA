"""cmame-paper Stage 7: the spectral export of CM2D's errors
(scripts/cm2d_spectra.py) runs end to end on a small run shaped like CM2D's
(E1's label-free states in a directory of their own, the labels-only,
stiffness-norm and graph-network states of the run, a report with their
per-instance arrays, a provenance file with the states' and the report's
SHA-256), refuses a state or a report that its provenance file does not
record, checks every model by content against the report's arrays (on the
median deviation), and records per-load error norms, modal spectra in two
binnings, von Mises errors and energies whose identities hold: the spectra
sum to the Euclidean and stiffness norms, ||e||_K^2 = 2 (Pi_h(u) - Pi_h(U*))
(Lemma 1) and equals the plane-stress stress-energy integral (Proposition
"Energy gap and stress error"), whose bound holds on every prediction, with
gamma* and the area-weighted von Mises error checked against independent
computations; the per-load arrays average to the report's per-instance
values; the summaries read the arrays and order constructed smooth and rough
errors the right way round."""

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
SCRIPT = ROOT / "scripts" / "cm2d_spectra.py"
MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 8, "mgn_depth": 1,
         "features": {"load_summary": True, "geometry": True}}
ROWS = ("ar", "labels", "labels_knorm", "mgn")
SEEDS = (0, 1)
N_VAL = 6                       # even: the figure rule's rank (n - 1) // 2 differs from n // 2
SPECTRA = ("S2_rank", "SK_rank", "S2_log", "SK_log")
IEEE = ("e2_ieee", "eK_ieee", "d2_tf32", "dK_tf32")
IEEE_SPECTRA = ("S2_log_ieee", "SK_log_ieee", "SK_log_tf32")
SUMMARY_KEYS = {"content_median_rel_dev", "content_mismatch", "rayleigh_ratio_median", "pairs",
                "prop1_bound_ratio_max", "figures"}


def _mod():
    spec = importlib.util.spec_from_file_location("cm2d_spectra", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _save(kind: str, seed: int, path: Path) -> str:
    import torch

    from fejepa.experiments.parallel import _build_model

    torch.save(_build_model({"kind": kind, "model": MODEL, "seed": seed}).state_dict(), path)
    return _sha(path)


def _arrays(report: dict, kind: str, state: Path, seed: int) -> dict:
    from fejepa.analysis.common import build_model_from_config
    from fejepa.analysis.posthoc import run_files
    from fejepa.data.archive import load_instance
    from fejepa.metrics import evaluate_model, torch_predictor

    m = build_model_from_config(dict(MODEL, kind=kind), state_path=str(state), seed=seed)
    ev = evaluate_model(torch_predictor(m, "cpu"),
                        [load_instance(f) for f in run_files(report, "val")])
    return {"per_instance": ev["per_instance"]}


def _write_provenance(path: Path, report: Path, states: list) -> None:
    lines = ["HEAD x", "--- report ---", f"{_sha(report)}  {report}", "--- states ---"]
    lines += [f"{_sha(p)}  {p}" for p in states]
    path.write_text("\n".join(lines) + "\n")


def _gmsh_corpus(out: Path, n: int, seed: int) -> Path:
    """A labelled corpus of coarse gmsh plates, the corpus's own kind of mesh
    (unstructured triangles of unequal areas, holes, scikit-fem assembly), in
    the generator's manifest format."""
    from fejepa.data.archive import save_instance, write_manifest
    from fejepa.fe.elasticity import LOAD_NAMES
    from fejepa.fe.generator import build_instance, sample_params
    from fejepa.fe.solve import solve_fe_displacement

    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        params = sample_params(rng)
        params["target_h"] = 0.15
        arch = build_instance(params)
        arch.U_star, _ = solve_fe_displacement(arch.K, arch.F, np.asarray(arch.free_mask, bool),
                                               method="direct")
        fn = f"instance_{i:05d}.npz"
        save_instance(arch, out / fn)
        records.append({"file": fn, "n_nodes": arch.n_nodes,
                        "n_holes": arch.meta["extra"]["n_holes"], "labelled": True})
    write_manifest(out, records, {"backend": "gmsh", "seed": seed, "labelled_policy": "all",
                                  "load_names": LOAD_NAMES})
    return out


def _run_2d(tmp_path: Path):
    """A CM2D-shaped run: a labelled 2D corpus, E1's label-free states in a
    directory of their own, the run's supervised states, a report with every
    row's per-instance arrays and the label-free hashes, and a provenance file."""
    from fejepa.data.archive import manifest_sha256
    from fejepa.report import config_sha256

    data = _gmsh_corpus(tmp_path / "data2d", n=12, seed=3)
    cfg = {"model": MODEL, "tf32": False, "split": {"n_val": N_VAL, "seed": 1},
           "data": {"dir": str(data)},
           "experiments": {"e8": {"pool_sizes": [4], "budgets": [2, 4], "mgn_budgets": [4]}}}
    e1, sdir = tmp_path / "e1_states", tmp_path / "states"
    e1.mkdir()
    sdir.mkdir()
    ar_rec, paths = {}, {}
    for s in SEEDS:
        paths[("ar", s)] = e1 / f"ar_p4_s{s}.pt"
        ar_rec[f"s{s}"] = {"sha256": _save("fejepa", s, paths[("ar", s)]), "reused": True}
        for k, (row, kind) in enumerate((("labels", "fejepa"), ("labels_knorm", "fejepa"),
                                         ("mgn", "mgn"))):
            paths[(row, s)] = sdir / f"{row}_b4_s{s}.pt"
            _save(kind, 10 * (k + 1) + s, paths[(row, s)])
    report = {"config": cfg,
              "provenance": {"config_sha256": config_sha256(cfg), "seeds": list(SEEDS),
                             "datasets": [{"dir": str(data),
                                           "manifest_sha256": manifest_sha256(data)}]},
              "reuse_from": {"states_dir": str(e1)},
              "results": {"e8": {"metrics": {"d9_restart": {"ar_states": ar_rec}}}}}
    cells = {"ar": {"4": {"per_seed_eval": []}},
             **{r: {"2": {"per_seed_eval": []}, "4": {"per_seed_eval": []}}
                for r in ("labels", "labels_knorm")},
             "mgn": {"4": {"per_seed_eval": []}}}
    for row in ROWS:
        cell = cells[row]["4"]
        for s in SEEDS:
            cell["per_seed_eval"].append(
                _arrays(report, "mgn" if row == "mgn" else "fejepa", paths[(row, s)], s))
    report["results"]["e8"]["metrics"]["cells"] = cells
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(report))
    prov = tmp_path / "provenance.txt"
    _write_provenance(prov, rp, [paths[k] for k in sorted(paths)])
    return rp, prov, sdir, paths


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args), "--device", "cpu"],
                          capture_output=True, text=True, env=ENV)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    pytest.importorskip("gmsh")
    pytest.importorskip("skfem")
    tmp = tmp_path_factory.mktemp("cm2d_spectra")
    rp, prov, sdir, paths = _run_2d(tmp)
    out = tmp / "spectra"
    r = _run("--report", rp, "--provenance", prov, "--states-dir", sdir, "--out", out)
    assert r.returncode == 0, r.stderr[-3000:]
    return {"tmp": tmp, "report": rp, "prov": prov, "sdir": sdir, "paths": paths, "out": out,
            "stdout": r.stdout}


def test_export_end_to_end(exported):
    from fejepa.analysis.posthoc import run_files
    from fejepa.data.archive import load_instance

    mod = _mod()
    out, report = exported["out"], json.loads(exported["report"].read_text())
    res = json.loads((out / "spectra.json").read_text())
    assert res["budget"] == 4 and res["pool"] == 4 and res["seeds"] == list(SEEDS)
    assert res["rows"] == list(ROWS) and res["rank_bins"] == 40 and res["val"]["n"] == N_VAL
    assert res["log_edges"] == [float(x) for x in mod.LOG_EDGES] and res["decades"] == [1, 2, 3, 4]
    assert res["content_mismatch"] == [] and res["torch"]["device"] == "cpu"
    assert json.dumps(res, allow_nan=False)                  # strict JSON: no NaN
    assert res["report_sha256"] == _sha(exported["report"])
    assert res["provenance_sha256"] == _sha(exported["prov"])
    for (row, s), p in exported["paths"].items():
        rec = res["states"][f"{row}_s{s}"]
        assert rec["sha256"] == _sha(p) and rec["sha256_ok"] and rec["file"] == p.name
        assert rec["checked_against"] == (["provenance", "d9_restart"] if row == "ar"
                                          else ["provenance"])
        c = rec["content"]
        assert c["ok"] and c["n"] == N_VAL
        for m in ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err",
                  "crit_recall"):
            assert c[f"{m}_max_rel_dev"] < 1e-6, (row, s, m)      # the evaluation's numbers
    # the rule, from the report's arrays
    cells = report["results"]["e8"]["metrics"]["cells"]
    lab0 = np.asarray(cells["labels"]["4"]["per_seed_eval"][0]["per_instance"]["energy_gap_rel"])
    fig = res["figures"]["fig2d"]
    assert fig["index"] == int(np.argsort(lab0, kind="stable")[(N_VAL - 1) // 2])
    val = run_files(report, "val")
    assert fig["file"] == Path(val[fig["index"]]).name
    # what the script prints: the progress, the figure line and, last, the summary line
    lines = exported["stdout"].splitlines()
    assert f"[spectra] fig2d: val #{fig['index']} {fig['file']} ({fig['n_nodes']} nodes)" in lines
    assert any(x.startswith(f"[spectra] val {N_VAL}/{N_VAL} | ") for x in lines)
    summ = json.loads(lines[-1])
    assert set(summ) == SUMMARY_KEYS and summ["content_mismatch"] == []
    assert summ["figures"] == {"fig2d": [fig["file"], fig["index"]]}
    # the arrays
    S = np.load(out / "spectra_val.npz")
    assert list(S["rows"]) == list(ROWS) and list(S["seeds"]) == list(SEEDS)
    assert int(S["rank_bins"]) == 40 and np.array_equal(S["log_edges"], mod.LOG_EDGES)
    assert list(S["files"]) == [Path(f).name for f in val]
    L = S["pi_star"].shape[1]
    shape = (len(ROWS), len(SEEDS), N_VAL, L)
    for k in ("pi", "rel", "c", "rel_c", "e2", "eK", "vm_elem", "vm_area", "stress_energy",
              "u_dirichlet_max"):
        assert S[k].shape == shape and np.isfinite(S[k]).all(), k
    for k, nb in (("S2_rank", 40), ("SK_rank", 40), ("S2_log", 70), ("SK_log", 70)):
        assert S[k].shape == shape + (nb,) and S[f"{k}_star"].shape == (N_VAL, L, nb), k
    for k in ("u2_star", "uK_star", "gamma_star", "vm2_area_star", "p2_area_star",
              "rq_star_over_lam_min", "ustar_dirichlet_max"):
        assert S[k].shape == (N_VAL, L) and np.isfinite(S[k]).all(), k
    for k in IEEE:
        assert S[k].shape == shape and np.isfinite(S[k]).all(), k
    for k in IEEE_SPECTRA:
        assert S[k].shape == shape + (70,) and np.isfinite(S[k]).all(), k
    # the run's policy here is TF32 off, so the second pass repeats the first
    assert res["precision"]["tf32_policy"] is False and "setup_torch(tf32=False)" in \
        res["precision"]["ieee_pass"]
    assert res["torch_after"]["tf32_matmul"] == res["torch"]["tf32_matmul"]
    assert np.allclose(S["e2_ieee"], S["e2"], rtol=1e-9) and np.allclose(S["eK_ieee"], S["eK"],
                                                                         rtol=1e-9)
    assert (S["d2_tf32"] <= 1e-12 * S["e2"]).all() and (S["dK_tf32"] <= 1e-12 * S["eK"]).all()
    assert np.allclose(S["S2_log_ieee"].sum(-1), S["e2_ieee"], rtol=1e-9)
    assert np.allclose(S["SK_log_ieee"].sum(-1), S["eK_ieee"], rtol=1e-9)
    assert np.allclose(S["SK_log_tf32"].sum(-1), S["dK_tf32"], rtol=1e-6, atol=1e-30)
    # per-load arrays average to the report's per-instance values, row by row
    for ri, row in enumerate(ROWS):
        for si in range(len(SEEDS)):
            pi = cells[row]["4"]["per_seed_eval"][si]["per_instance"]
            assert np.allclose(S["rel"][ri, si].mean(-1), pi["energy_gap_rel"], rtol=1e-9)
            assert np.allclose(np.sqrt(S["e2"][ri, si] / S["u2_star"]).mean(-1),
                               pi["disp_rel_l2"], rtol=1e-9)
            assert np.allclose(S["vm_elem"][ri, si].mean(-1), pi["vm_rel_l2"], rtol=1e-9)
    # the eigenvalues and the material; element areas that differ, so that the
    # element-count and area-weighted von Mises errors differ
    from fejepa.fe.stress import _geometry

    ratios = []
    for i, f in enumerate(val):
        a = load_instance(f)
        area = _geometry(np.asarray(a.nodes), np.asarray(a.elements))[0]
        ratios.append(area.max() / area.min())
        fr = np.asarray(a.free_mask, bool)
        lam = np.linalg.eigvalsh(a.K.tocsr()[fr][:, fr].toarray())
        assert S["n_free"][i] == fr.sum()
        assert np.isclose(S["lam_min"][i], lam[0]) and np.isclose(S["lam_max"][i], lam[-1])
        edges = np.floor(np.arange(41) * lam.size / 40).astype(int)
        assert np.allclose(S["lam_rank_start"][i], lam[edges[:-1]])
        nu = a.meta["material"]["nu"]
        assert np.isclose(S["G"][i], 1 / (2 * (1 + nu)))
        assert np.isclose(S["Bk"][i], 1 / (3 * (1 - 2 * nu)))
        assert np.allclose(S["rq_star_over_lam_min"][i],
                           S["uK_star"][i] / S["u2_star"][i] / lam[0])
    assert max(ratios) > 1.5                                # robust to the gmsh version
    assert not np.allclose(S["vm_elem"], S["vm_area"], rtol=1e-3)
    _check_identities(S)
    _check_summaries(S, res)
    _check_figure(out, res, S, val)


def _check_identities(S):
    """Lemma 1, Proposition "Energy gap and stress error" and the spectral sums."""
    assert (S["u_dirichlet_max"] == 0).all() and (S["ustar_dirichlet_max"] == 0).all()
    for k in ("S2_rank", "S2_log"):
        assert np.allclose(S[k].sum(-1), S["e2"], rtol=1e-9, atol=0), k
        assert np.allclose(S[f"{k}_star"].sum(-1), S["u2_star"], rtol=1e-9, atol=0), k
    for k in ("SK_rank", "SK_log"):
        assert np.allclose(S[k].sum(-1), S["eK"], rtol=1e-9, atol=0), k
        assert np.allclose(S[f"{k}_star"].sum(-1), S["uK_star"], rtol=1e-9, atol=0), k
    for k in SPECTRA:
        assert (S[k] >= 0).all() and (S[f"{k}_star"] >= 0).all(), k
    assert np.allclose(S["eK"], 2 * (S["pi"] - S["pi_star"]), rtol=1e-8, atol=0)
    assert np.allclose(S["uK_star"], -2 * S["pi_star"], rtol=1e-10, atol=0)
    assert np.allclose(S["rel"], S["eK"] / S["uK_star"], rtol=1e-8, atol=0)
    assert np.allclose(S["stress_energy"], S["eK"], rtol=1e-8, atol=0)
    # gamma* from the stress integrals of U*, which add up to ||U*||_K^2
    G, B = S["G"][:, None], S["Bk"][:, None]
    assert np.allclose(S["gamma_star"], 3 * G * S["p2_area_star"] / (B * S["vm2_area_star"]),
                       rtol=1e-12)
    assert np.allclose(S["vm2_area_star"] / (3 * G) + S["p2_area_star"] / B, S["uK_star"],
                       rtol=1e-9)
    assert (S["gamma_star"] > 0).all()
    g = S["eK"] / S["uK_star"]
    assert (S["vm_area"] ** 2 <= (1 + S["gamma_star"]) * g * (1 + 1e-9)).all()
    assert np.array_equal(S["pi"] > 0, S["rel"] > 1)                     # the zero-field test
    assert np.all(S["rel_c"] <= np.minimum(S["rel"], 1.0) * (1 + 1e-9) + 1e-12)


def _tail(X: np.ndarray, j: int) -> np.ndarray:
    """Share of each spectrum in its bins j and above, flattened."""
    return (X[..., j:].sum(-1) / X.sum(-1)).ravel()


def gm_seed(x: np.ndarray) -> np.ndarray:
    """(n, L) geometric mean over the seed axis."""
    return np.exp(np.log(x).mean(axis=0))


def _tail_curve(X: np.ndarray) -> np.ndarray:
    """Per bin j, the median over the spectra of their share in bins j and above."""
    return np.array([np.median(_tail(X, j)) for j in range(X.shape[-1])])


def _check_summaries(S, res):
    """The summaries are the arrays' readings."""
    mod = _mod()
    rq = (S["eK"] / S["e2"]) / (S["uK_star"] / S["u2_star"])[None, None]
    g, d2 = S["eK"] / S["uK_star"], S["e2"] / S["u2_star"]
    assert np.allclose(g, rq * d2, rtol=1e-12)              # per load case: g = rq d^2
    bound = S["vm_area"] ** 2 / ((1 + S["gamma_star"]) * g)
    diag = res["diagnostics"]
    for ri, row in enumerate(ROWS):
        d = diag["rows"][row]
        assert d["n"] == rq[ri].size
        assert np.isclose(d["rayleigh_ratio_median"], np.median(rq[ri]), rtol=1e-12)
        assert np.allclose(d["rayleigh_ratio_p10_p90"],
                           [np.percentile(rq[ri], 10), np.percentile(rq[ri], 90)], rtol=1e-12)
        assert np.allclose(d["rayleigh_ratio_median_by_seed"], np.median(rq[ri], axis=(1, 2)))
        for k, X in (("e2", S["S2_log"][ri]), ("eK", S["SK_log"][ri])):
            for dec in (1, 2, 3, 4):
                j = mod.edge_bin(dec)
                assert mod.LOG_EDGES[j - 1] == dec
                assert np.isclose(d[f"share_{k}_above_median"][f"1e{dec}"],
                                  np.median(_tail(X, j)), rtol=1e-12)
        for kind, nb in (("log", 70), ("rank", 40)):
            for k, X in (("e2", S[f"S2_{kind}"][ri]), ("eK", S[f"SK_{kind}"][ri])):
                curve = np.asarray(d[f"tail_{kind}_{k}_median"])
                assert curve.shape == (nb,) and np.isclose(curve[0], 1.0)
                assert (np.diff(curve) <= 1e-12).all()                      # non-increasing
                assert np.allclose(curve, _tail_curve(X), rtol=1e-12, atol=0)
        assert np.isclose(d["prop1_bound_ratio_max"], bound[ri].max(), rtol=1e-12)
        assert np.isclose(d["prop1_bound_ratio_median"], np.median(bound[ri]), rtol=1e-12)
        assert d["prop1_bound_ratio_max"] <= 1 + 1e-9
        assert np.isclose(d["vm_area_median"], np.median(S["vm_area"][ri]), rtol=1e-12)
        assert np.isclose(d["vm_elem_median"], np.median(S["vm_elem"][ri]), rtol=1e-12)
        assert np.isclose(d["rel_gap_median"], np.median(S["rel"][ri]), rtol=1e-12)
        assert np.isclose(d["rel_gap_after_cstar_median"], np.median(S["rel_c"][ri]))
        for k in ("stress_identity_max_rel_dev", "gap_identity_max_rel_dev"):
            assert d[k] < 1e-8, (row, k)
        for k, tot in (("eK", "eK"), ("e2", "e2")):
            for b in ("rank", "log"):
                X = S[f"S{k[1]}_{b}"][ri]
                want = np.max(np.abs(X.sum(-1) - S[tot][ri]) / np.abs(S[tot][ri]))
                assert d[f"spectral_{k}_max_rel_dev"][b] == pytest.approx(want, rel=1e-9, abs=0)
                assert want < 1e-8, (row, k, b)
        assert d["u_dirichlet_max"] == 0
    rq_i = (S["eK_ieee"] / S["e2_ieee"]) / (S["uK_star"] / S["u2_star"])[None, None]
    with np.errstate(all="ignore"):
        noise = S["dK_tf32"] / S["eK"]
    for ri, row in enumerate(ROWS):
        d = diag["rows"][row]
        assert np.allclose(d["rayleigh_ratio_median_by_load"],
                           np.median(rq[ri].reshape(-1, rq.shape[-1]), axis=0), rtol=1e-12)
        di = d["ieee"]
        assert np.isclose(di["rayleigh_ratio_median"], np.median(rq_i[ri]), rtol=1e-12)
        assert np.allclose(di["rayleigh_ratio_p10_p90"],
                           [np.percentile(rq_i[ri], 10), np.percentile(rq_i[ri], 90)], rtol=1e-12)
        assert np.isclose(di["rel_gap_median"], np.median(S["eK_ieee"][ri] / S["uK_star"]),
                          rtol=1e-12)
        for k, X in (("e2", S["S2_log_ieee"][ri]), ("eK", S["SK_log_ieee"][ri])):
            assert np.isclose(di[f"share_{k}_above_median"]["1e1"],
                              np.median(_tail(X, mod.edge_bin(1))), rtol=1e-12)
            assert np.allclose(di[f"tail_log_{k}_median"], _tail_curve(X), rtol=1e-12, atol=0)
        dt = d["tf32"]
        assert np.isclose(dt["dK_over_eK_median"], np.median(noise[ri]), rtol=1e-9, atol=1e-30)
        assert set(dt["dK_over_eK_above_median"]) == {"1e2", "1e3", "1e4"}
        for k in (2, 3, 4):
            j = mod.edge_bin(k)
            with np.errstate(all="ignore"):
                band = S["SK_log_tf32"][ri][..., j:].sum(-1) / S["SK_log"][ri][..., j:].sum(-1)
            assert dt["dK_over_eK_above_n"][f"1e{k}"] == int(np.isfinite(band).sum())
    gm = lambda r: np.exp(np.mean(np.log(r)))                              # noqa: E731
    for a, b in mod.PAIRS:
        p = diag["pairs"][f"{a}_vs_{b}"]
        ia, ib = ROWS.index(a), ROWS.index(b)
        x, y = rq[ia], rq[ib]
        assert p["n"] == x.size
        assert np.isclose(p["share_first_above"], np.mean(x > y))
        assert np.isclose(p["median_ratio"], np.median(x / y), rtol=1e-12)
        assert np.isclose(p["ratio_of_means_gap"], g[ia].mean() / g[ib].mean(), rtol=1e-12)
        assert np.isclose(p["ratio_of_means_vm_elem"],
                          S["vm_elem"][ia].mean() / S["vm_elem"][ib].mean(), rtol=1e-12)
        assert np.isclose(p["geomean_ratio_gap"], gm(g[ia] / g[ib]), rtol=1e-12)
        assert np.isclose(p["geomean_ratio_rayleigh"], gm(x / y), rtol=1e-12)
        assert np.isclose(p["geomean_ratio_disp_sq"], gm(d2[ia] / d2[ib]), rtol=1e-12)
        assert np.isclose(p["geomean_ratio_gap"],
                          p["geomean_ratio_rayleigh"] * p["geomean_ratio_disp_sq"], rtol=1e-9)
        assert np.isclose(p["geomean_ratio_vm_elem_sq"],
                          gm((S["vm_elem"][ia] / S["vm_elem"][ib]) ** 2), rtol=1e-12)
        assert np.isclose(p["geomean_ratio_vm_area_sq"],
                          gm((S["vm_area"][ia] / S["vm_area"][ib]) ** 2), rtol=1e-12)
        assert np.isclose(p["share_first_gap_lower"], np.mean(g[ia] < g[ib]))
        assert np.isclose(p["share_first_disp_lower"], np.mean(d2[ia] < d2[ib]))
        assert np.isclose(p["share_first_disp_lower_gap_higher"],
                          np.mean((d2[ia] < d2[ib]) & (g[ia] > g[ib])))
        assert np.isclose(p["share_first_disp_higher_gap_lower"],
                          np.mean((d2[ia] > d2[ib]) & (g[ia] < g[ib])))
        assert [s["seed_index"] for s in p["by_seed"]] == list(range(len(SEEDS)))
        xs, ys = gm_seed(x), gm_seed(y)
        assert np.isclose(p["share_first_above_seed_geomean"], np.mean(xs > ys))
        pi_ = p["ieee"]
        xi, yi = rq_i[ia], rq_i[ib]
        assert np.isclose(pi_["share_first_above"], np.mean(xi > yi))
        assert np.isclose(pi_["geomean_ratio_rayleigh"], gm(xi / yi), rtol=1e-12)
        assert np.isclose(pi_["geomean_ratio_gap"], pi_["geomean_ratio_rayleigh"]
                          * pi_["geomean_ratio_disp_sq"], rtol=1e-9)
        for si, s in enumerate(p["by_seed"]):
            assert np.isclose(s["geomean_ratio_gap"], gm(g[ia][si] / g[ib][si]), rtol=1e-12)
            assert np.isclose(s["geomean_ratio_rayleigh"], gm(x[si] / y[si]), rtol=1e-12)
            assert np.isclose(s["geomean_ratio_disp_sq"], gm(d2[ia][si] / d2[ib][si]),
                              rtol=1e-12)
    ref = diag["reference"]
    for k in ("energy_identity_max_rel_dev", "gamma_identity_max_rel_dev"):
        assert ref[k] < 1e-9, k
    for k, tot in (("uK", "uK_star"), ("u2", "u2_star")):
        for b in ("rank", "log"):
            X = S[f"S{k[1]}_{b}_star"]
            want = np.max(np.abs(X.sum(-1) - S[tot]) / np.abs(S[tot]))
            assert ref[f"spectral_{k}_max_rel_dev"][b] == pytest.approx(want, rel=1e-9, abs=0)
            assert want < 1e-9, (k, b)
    assert ref["ustar_dirichlet_max"] == 0
    q = S["rq_star_over_lam_min"]
    assert np.isclose(ref["rayleigh_over_lam_min_median"], np.median(q))
    assert np.allclose(ref["rayleigh_over_lam_min_min_max"], [q.min(), q.max()], rtol=1e-12)
    assert q.min() >= 1 - 1e-9                                  # RQ* is at least lambda_min
    assert np.isclose(ref["rayleigh_median"], np.median(S["uK_star"] / S["u2_star"]))
    assert np.isclose(ref["gamma_star_median"], np.median(S["gamma_star"]))
    for k, X in (("u2", S["S2_log_star"]), ("uK", S["SK_log_star"])):
        for dec in (1, 2, 3, 4):
            assert np.isclose(ref[f"share_{k}_above_median"][f"1e{dec}"],
                              np.median(_tail(X, mod.edge_bin(dec))), rtol=1e-12)
    for kind in ("log", "rank"):
        for k, X in (("u2", S[f"S2_{kind}_star"]), ("uK", S[f"SK_{kind}_star"])):
            assert np.allclose(ref[f"tail_{kind}_{k}_median"], _tail_curve(X), rtol=1e-12,
                               atol=0)
    assert res["eigen"]["n_free_min_median_max"][0] == int(S["n_free"].min())
    for ri, row in enumerate(ROWS):
        c = res["counts"][row]
        assert c["load_cases"] == S["pi"][ri].size
        assert c["instances"] == len(SEEDS) * N_VAL
        assert c["load_cases_pi_positive"] == int((S["pi"][ri] > 0).sum())
        assert c["load_cases_rel_gap_above_1"] == int((S["rel"][ri] > 1).sum())
        assert c["instances_mean_rel_gap_above_1"] == int((S["rel"][ri].mean(-1) > 1).sum())
        assert c["load_cases_rel_gap_above_1_after_cstar"] == 0
        assert c["load_cases_cstar_increased_gap"] == 0


def _check_figure(out, res, S, val):
    from fejepa.data.archive import load_instance
    from fejepa.fe.stress import element_von_mises

    F = np.load(out / "fig2d.npz")
    i = res["figures"]["fig2d"]["index"]
    a = load_instance(val[i])
    meta = json.loads(str(F["meta"]))
    assert meta["figure"] == "fig2d" and meta["index"] == i and meta["seed"] == 0
    assert np.array_equal(F["elements"], a.elements) and np.allclose(F["U_star"], a.U_star)
    n_free = int(np.asarray(a.free_mask, bool).sum())
    assert F["lam"].shape == (n_free,) and F["C2_star"].shape == (a.n_loads, n_free)
    assert np.allclose(F["C2_star"].sum(-1), S["u2_star"][i])
    assert np.allclose((F["C2_star"] * F["lam"]).sum(-1), S["uK_star"][i])
    for k in ("pi_star", "u2_star", "uK_star", "gamma_star"):
        assert np.array_equal(F[k], S[k][i]), k
    assert np.allclose(F["rq_star"], S["uK_star"][i] / S["u2_star"][i])
    m = a.meta["material"]
    assert np.allclose(F["vm_ref"][0], element_von_mises(a.nodes, a.elements, a.U_star[0], m),
                       rtol=1e-5)
    # the log bins of the stored spectra: lambda_m / RQ* of U*, 1/8 decade, at or above an edge
    idx = np.searchsorted(F["log_edges"], np.log10(F["lam"][None] / F["rq_star"][:, None]),
                          side="right")
    rebin = lambda C2: np.stack([np.bincount(idx[j], weights=C2[j], minlength=70)  # noqa: E731
                                 for j in range(C2.shape[0])])
    assert np.allclose(rebin(F["C2_star"]), S["S2_log_star"][i], rtol=1e-10, atol=0)
    for ri, row in enumerate(ROWS):
        assert np.allclose(rebin(F[f"C2_{row}"]), S["S2_log"][ri, 0, i], rtol=1e-10, atol=0)
        assert np.allclose(rebin(F[f"C2_{row}"] * F["lam"]), S["SK_log"][ri, 0, i], rtol=1e-10,
                           atol=0)
    for ri, row in enumerate(ROWS):
        U = F[f"U_{row}"]
        assert U.shape == a.U_star.shape and F[f"U_ieee_{row}"].shape == U.shape
        assert np.array_equal(F[f"pi_{row}"], S["pi"][ri, 0, i])           # the energy pass
        assert np.allclose(F[f"C2_{row}"].sum(-1), S["e2"][ri, 0, i])
        assert np.allclose((F[f"C2_{row}"] * F["lam"]).sum(-1), S["eK"][ri, 0, i])
        assert np.allclose(F[f"vm_{row}"][1], element_von_mises(a.nodes, a.elements, U[1], m),
                           rtol=1e-5)


def test_binning(instances):
    """The rank bins are floor(b n / 40); an eigenmode's spectra are one (and
    its eigenvalue) in its own bin of each binning, and zero elsewhere; the
    log bin of a mode is that of lambda_m / RQ* on the 1/8-decade edges."""
    mod = _mod()
    arch = instances[2]
    eig = mod.eigenbasis(arch)
    n = eig["lam"].size
    assert np.array_equal(eig["edges"], np.floor(np.arange(41) * n / 40).astype(int))
    assert np.allclose(eig["W"].T @ eig["W"], np.eye(n), atol=1e-10)
    assert mod.LOG_BINS == 70 and mod.LOG_EDGES[0] == -2 and mod.LOG_EDGES[-1] == 6.5
    assert np.allclose(np.diff(mod.LOG_EDGES), 0.125)
    free = eig["free"]
    rq = np.array([3.0 * eig["lam"][0], 50.0 * eig["lam"][0]])           # two "load cases"
    idx = mod.log_index(eig["lam"], rq)
    for j in range(2):
        r = np.log10(eig["lam"] / rq[j])
        for m in range(n):
            b = idx[j, m]
            assert b == 0 or mod.LOG_EDGES[b - 1] <= r[m]
            assert b == 69 or r[m] < mod.LOG_EDGES[b]
    for m in (0, 1, n // 2, n - 1):
        x = np.zeros((2, free.size))
        x[:, free] = eig["W"][:, m]
        sp = mod.binned(eig, x, idx)
        b = int(np.searchsorted(eig["edges"], m, side="right") - 1)
        want = np.zeros(40)
        want[b] = 1.0
        assert np.allclose(sp["S2_rank"], want, atol=1e-12)
        assert np.isclose(sp["SK_rank"][0, b], eig["lam"][m], rtol=1e-10)
        for j in range(2):
            wl = np.zeros(70)
            wl[idx[j, m]] = 1.0
            assert np.allclose(sp["S2_log"][j], wl, atol=1e-12)
            assert np.isclose(sp["SK_log"][j, idx[j, m]], eig["lam"][m], rtol=1e-10)
    with pytest.raises(SystemExit):
        mod.log_index(eig["lam"], np.array([1.0, 0.0]))
    with pytest.raises(SystemExit):                    # a non-positive eigenvalue
        mod.log_index(np.array([-1e-20, 1.0]), np.array([1.0]))
    # the pairing-free share reads geometric means over the seeds
    x = np.array([[[1.0]], [[100.0]]])                  # seeds x instances x loads
    assert np.isclose(mod._seed_geomean(x)[0, 0], 10.0)
    # a mode exactly at 10^k RQ* belongs to the bins at and above 10^k; below
    # 10^-2 RQ* the first bin, at or above 10^6.5 RQ* the last
    on_edges = mod.log_index(np.array([1e-3, 1.0, 10.0, 100.0, 1e7]), np.array([1.0]))[0]
    assert list(on_edges) == [0, mod.edge_bin(0), mod.edge_bin(1), mod.edge_bin(2), 69]
    assert [mod.edge_bin(k) for k in (0, 1, 2)] == [17, 25, 33]


def _plane_stress_checks(arch, rng):
    """tri_stress against the evaluation's stresses; Proposition 1's identity
    on random fields; gamma*, the area-weighted von Mises error and the bound
    against independent computations, near and far from U*."""
    from fejepa.fe.stress import _geometry, element_stresses, element_von_mises

    mod = _mod()
    op = mod.tri_ops(arch)
    assert op is not None
    m = arch.meta["material"]
    free = np.asarray(arch.free_mask, bool)
    u = rng.normal(size=(3, free.size)) * free
    sig = mod.tri_stress(op, u)
    for j in range(3):
        assert np.allclose(sig[j], element_stresses(arch.nodes, arch.elements, u[j], m),
                           rtol=1e-12)
    vm, p = mod.vm_p(sig)
    energy = (vm ** 2 / (3 * op["G"]) + p ** 2 / op["Bk"]) @ op["area"]
    assert np.allclose(energy, mod.quad(arch.K, u), rtol=1e-9)       # Proposition 1's identity
    eig = mod.eigenbasis(arch)
    ref = mod.reference(arch, op, eig)
    area = _geometry(np.asarray(arch.nodes), np.asarray(arch.elements))[0]
    E, nu = m["E"], m["nu"]
    G, B = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    for j in range(arch.n_loads):
        vm_t = element_von_mises(arch.nodes, arch.elements, arch.U_star[j], m)
        s = element_stresses(arch.nodes, arch.elements, arch.U_star[j], m)
        p_t = (s[:, 0] + s[:, 1]) / 3
        gamma = 3 * G * (area @ p_t ** 2) / (B * (area @ vm_t ** 2))
        assert np.isclose(ref["gamma_star"][j], gamma, rtol=1e-12)
        assert np.isclose((1 + gamma) * (area @ vm_t ** 2) / (3 * G), ref["uK_star"][j],
                          rtol=1e-9)
    for scale in (1e-3, 0.3, 3.0):
        noise = rng.normal(size=arch.U_star.shape) * free
        U = arch.U_star + scale * np.abs(arch.U_star).max() * noise
        d = mod.diagnostics(U, arch, op, ref, eig)
        for j in range(arch.n_loads):
            vm_t = element_von_mises(arch.nodes, arch.elements, arch.U_star[j], m)
            vm_u = element_von_mises(arch.nodes, arch.elements, U[j], m)
            assert np.isclose(d["vm_area"][j],
                              np.sqrt((area @ (vm_u - vm_t) ** 2) / (area @ vm_t ** 2)),
                              rtol=1e-12)
        g = d["eK"] / ref["uK_star"]
        assert (d["vm_area"] ** 2 <= (1 + ref["gamma_star"]) * g * (1 + 1e-9)).all()
        assert np.allclose(d["stress_energy"], d["eK"], rtol=1e-9)
    return area


def test_plane_stress_identity_and_bound_on_a_structured_mesh(instances):
    _plane_stress_checks(instances[0], np.random.default_rng(5))


def test_plane_stress_identity_and_bound_on_a_gmsh_mesh():
    """The corpus's own kind of mesh: unstructured triangles of unequal areas,
    a hole, scikit-fem assembly."""
    pytest.importorskip("gmsh")
    pytest.importorskip("skfem")
    from fejepa.fe.generator import build_instance, sample_params
    from fejepa.fe.solve import solve_fe_displacement

    rng = np.random.default_rng(11)
    params = sample_params(rng)
    params["target_h"] = 0.15
    params["holes"] = [[0.5 * params["width"], 0.5 * params["height"],
                        0.1 * min(params["width"], params["height"])]]
    arch = build_instance(params)
    arch.U_star, _ = solve_fe_displacement(arch.K, arch.F, np.asarray(arch.free_mask, bool),
                                           method="direct")
    area = _plane_stress_checks(arch, rng)
    assert area.max() > 1.5 * area.min()                    # the area weights matter here


def test_constrained_dofs_count_in_the_euclidean_norm(instances):
    """A prediction that is not zero on a constrained dof: ||e||_2^2 is the
    norm over every dof (the evaluation's displacement error), the largest
    constrained value is recorded, and the free-dof spectra then fall short
    of it, which the summary's spectral deviation would report."""
    mod = _mod()
    arch = instances[1]
    eig = mod.eigenbasis(arch)
    op = mod.tri_ops(arch)
    ref = mod.reference(arch, op, eig)
    fixed = ~eig["free"]
    U = arch.U_star * 1.01
    U[:, np.flatnonzero(fixed)[0]] = 0.02
    d = mod.diagnostics(U, arch, op, ref, eig)
    E = U - arch.U_star
    assert np.allclose(d["e2"], (E ** 2).sum(-1)) and (d["u_dirichlet_max"] == 0.02).all()
    assert (d["S2_rank"].sum(-1) < d["e2"] * (1 - 1e-6)).all()
    assert np.allclose(d["S2_log"].sum(-1), (E[:, eig["free"]] ** 2).sum(-1))


def test_summaries_order_smooth_and_rough_errors(instances):
    """Constructed errors through diagnostics and summary: an error made of
    the stiffest modes has the larger normalised Rayleigh quotient, shares
    above the thresholds at least as large, and the larger energy gap than
    one of equal Euclidean norm made of the softest modes; doubling the
    smooth error's size gives the opposite rankings of Remark "Opposite
    rankings"."""
    mod = _mod()
    arch = instances[3]
    eig = mod.eigenbasis(arch)
    op = mod.tri_ops(arch)
    ref = mod.reference(arch, op, eig)
    n, free, L = eig["lam"].size, eig["free"], arch.n_loads
    rng = np.random.default_rng(2)

    def error(modes, size):
        e = np.zeros((L, free.size))
        e[:, free] = (eig["W"][:, list(modes)] @ rng.normal(size=(len(modes), L))).T
        return size * e / np.linalg.norm(e, axis=1, keepdims=True) * np.linalg.norm(
            arch.U_star, axis=1, keepdims=True)

    low, high = error(range(0, 4), 0.1), error(range(n - 8, n), 0.1)
    preds = {"ar": arch.U_star + low, "labels": arch.U_star + high,
             "labels_knorm": arch.U_star + 2 * low, "mgn": arch.U_star + 0.5 * high}
    dg = {row: mod.diagnostics(U, arch, op, ref, eig) for row, U in preds.items()}
    # an "IEEE" prediction that differs from the TF32 one by a rough
    # perturbation of 1e-3 of the solution's size (the rounding u - u_ieee)
    rough = error(range(n - 8, n), 1e-3)
    di = {row: mod.ieee_diagnostics(U, U - rough, arch, ref, eig) for row, U in preds.items()}
    D = {k: np.stack([dg[row][k] for row in ROWS])[:, None, None]
         for k in (*mod.DIAG, *mod.SPECTRA)}
    D.update({k: np.stack([di[row][k] for row in ROWS])[:, None, None]
              for k in (*mod.IEEE, *mod.IEEE_SPECTRA)})
    R = {k: np.asarray(ref[k])[None] for k in (*mod.REF, *mod.SPECTRA)}
    R.update(G=np.array([op["G"]]), Bk=np.array([op["Bk"]]))
    s = mod.summary(D, R, np.ones((len(ROWS), 1), dtype=bool))
    rows, pairs = s["rows"], s["pairs"]
    assert rows["labels"]["rayleigh_ratio_median"] > 10 * rows["ar"]["rayleigh_ratio_median"]
    assert np.isclose(rows["labels_knorm"]["rayleigh_ratio_median"],
                      rows["ar"]["rayleigh_ratio_median"])           # scale-free
    p = pairs["labels_vs_ar"]
    assert p["share_first_above"] == 1 and p["share_first_gap_lower"] == 0
    assert np.isclose(p["geomean_ratio_disp_sq"], 1) and p["geomean_ratio_rayleigh"] > 10
    assert np.isclose(p["geomean_ratio_gap"], p["geomean_ratio_rayleigh"], rtol=1e-9)
    q = pairs["labels_vs_labels_knorm"]                      # smaller in displacement, rougher
    assert np.isclose(q["geomean_ratio_disp_sq"], 0.25)
    assert q["share_first_disp_lower"] == 1 and q["share_first_disp_lower_gap_higher"] == 1
    assert q["share_first_disp_higher_gap_lower"] == 0
    assert np.isclose(pairs["mgn_vs_labels"]["geomean_ratio_gap"], 0.25)
    assert pairs["mgn_vs_labels"]["ratio_of_means_gap"] == pytest.approx(0.25)
    for k in ("share_e2_above_median", "share_eK_above_median"):
        for dec in ("1e1", "1e2"):
            assert rows["labels"][k][dec] >= rows["ar"][k][dec]
    t_hi = np.asarray(rows["labels"]["tail_log_e2_median"])
    t_lo = np.asarray(rows["ar"]["tail_log_e2_median"])
    assert (t_hi >= t_lo - 1e-12).all() and (t_hi > t_lo).any()
    # the IEEE error here carries the rough perturbation: rougher than the
    # TF32 error of the smooth row; the rounding's share of the error's
    # stiffness norm is read off
    assert rows["ar"]["ieee"]["rayleigh_ratio_median"] > rows["ar"]["rayleigh_ratio_median"]
    for row in ROWS:
        want = np.median(di[row]["dK_tf32"] / dg[row]["eK"])
        assert np.isclose(rows[row]["tf32"]["dK_over_eK_median"], want, rtol=1e-12)
        assert np.allclose(di[row]["SK_log_tf32"].sum(-1), di[row]["dK_tf32"], rtol=1e-9)
        assert np.allclose(di[row]["d2_tf32"], (rough ** 2).sum(-1), rtol=1e-12)
        assert np.allclose(di[row]["e2_ieee"],
                           ((preds[row] - rough - arch.U_star) ** 2).sum(-1), rtol=1e-12)
    assert rows["ar"]["tf32"]["dK_over_eK_median"] > rows["labels"]["tf32"]["dK_over_eK_median"]
    assert pairs["labels_vs_ar"]["ieee"]["share_first_above"] == 1
    for row in ROWS:                       # the rounding's share of each band of the error
        for k in (2, 3, 4):
            j = mod.edge_bin(k)
            with np.errstate(all="ignore"):
                band = (di[row]["SK_log_tf32"][..., j:].sum(-1)
                        / dg[row]["SK_log"][..., j:].sum(-1))
            band = band[np.isfinite(band)]
            got = rows[row]["tf32"]["dK_over_eK_above_median"][f"1e{k}"]
            assert (got is None and band.size == 0) or np.isclose(got, np.median(band),
                                                                  rtol=1e-12)
    rq_i = {row: (di[row]["eK_ieee"] / di[row]["e2_ieee"]) / (ref["uK_star"] / ref["u2_star"])
            for row in ROWS}
    for a, b in mod.PAIRS:                 # the pairs' IEEE readings read the IEEE arrays
        want = np.exp(np.mean(np.log(rq_i[a] / rq_i[b])))
        assert np.isclose(pairs[f"{a}_vs_{b}"]["ieee"]["geomean_ratio_rayleigh"], want,
                          rtol=1e-12)
    for row in ROWS:                       # None, not NaN, where no energy lies above 10^k
        for v in rows[row]["tf32"]["dK_over_eK_above_median"].values():
            assert v is None or np.isfinite(v)
    assert json.dumps(s, allow_nan=False)                    # strict JSON: no NaN
    # the box-side checks of gamma* and of Proposition 1's bound respond to a wrong gamma*
    assert s["reference"]["gamma_identity_max_rel_dev"] < 1e-9
    bad = mod.summary(D, dict(R, gamma_star=2 * R["gamma_star"]),
                      np.ones((len(ROWS), 1), dtype=bool))
    assert bad["reference"]["gamma_identity_max_rel_dev"] > 1e-3
    assert bad["rows"]["labels"]["prop1_bound_ratio_max"] < rows["labels"]["prop1_bound_ratio_max"]
    # and the spectral checks report each binning on its own
    off = mod.summary(dict(D, SK_log=1.01 * D["SK_log"], S2_rank=1.04 * D["S2_rank"]),
                      dict(R, S2_rank=1.02 * R["S2_rank"], SK_log=1.03 * R["SK_log"]),
                      np.ones((len(ROWS), 1), dtype=bool))
    for row in ROWS:
        dev = off["rows"][row]["spectral_eK_max_rel_dev"]
        assert np.isclose(dev["log"], 0.01) and dev["rank"] < 1e-9
        dev = off["rows"][row]["spectral_e2_max_rel_dev"]
        assert np.isclose(dev["rank"], 0.04) and dev["log"] < 1e-9
    dev = off["reference"]["spectral_u2_max_rel_dev"]
    assert np.isclose(dev["rank"], 0.02) and dev["log"] < 1e-9
    dev = off["reference"]["spectral_uK_max_rel_dev"]
    assert np.isclose(dev["log"], 0.03) and dev["rank"] < 1e-9


def test_a_state_or_report_the_provenance_does_not_record_is_refused(exported, tmp_path):
    rp, sdir, paths = exported["report"], exported["sdir"], exported["paths"]
    # a supervised state whose bytes differ from the recorded ones
    bad = tmp_path / "states"
    bad.mkdir()
    for p in sdir.iterdir():
        (bad / p.name).write_bytes(p.read_bytes())
    (bad / "labels_knorm_b4_s1.pt").write_bytes(
        (sdir / "labels_knorm_b4_s0.pt").read_bytes())
    out = tmp_path / "o1"
    r = _run("--report", rp, "--provenance", exported["prov"], "--states-dir", bad, "--out", out)
    assert r.returncode != 0 and "labels_knorm_b4_s1.pt: SHA-256" in r.stderr
    assert not out.exists()                                    # refused before anything ran
    # a report that is not the recorded one
    rp2 = tmp_path / "report.json"
    rp2.write_text(rp.read_text() + "\n")
    out = tmp_path / "o2"
    r = _run("--report", rp2, "--provenance", exported["prov"], "--states-dir", sdir,
             "--out", out)
    assert r.returncode != 0 and "is not the report the provenance file records" in r.stderr
    assert not out.exists()
    # a label-free state that is not the one the report's d9_restart record names
    e1 = tmp_path / "e1"
    e1.mkdir()
    for s in SEEDS:
        (e1 / f"ar_p4_s{s}.pt").write_bytes(paths[("ar", 1 - s)].read_bytes())
    out = tmp_path / "o3"
    r = _run("--report", rp, "--provenance", exported["prov"], "--states-dir", sdir,
             "--ar-states-dir", e1, "--out", out)
    assert r.returncode != 0 and "is not the state the report trained" in r.stderr
    assert not out.exists()
    # a label-free state the d9_restart record names but the provenance file does not
    prov = tmp_path / "provenance.txt"
    prov.write_text(exported["prov"].read_text().replace(_sha(paths[("ar", 1)]), "0" * 64))
    out = tmp_path / "o4"
    r = _run("--report", rp, "--provenance", prov, "--states-dir", sdir, "--out", out)
    assert r.returncode != 0 and "ar_p4_s1.pt: SHA-256" in r.stderr
    assert not out.exists()


def test_a_model_that_does_not_reproduce_the_report_ends_with_status_5(exported, tmp_path):
    """Everything is still written; one outlying instance does not fail a
    model (the median is gated); the figure instance, here beyond --n-val, is
    evaluated on its own and agrees with the full export."""
    report = json.loads(exported["report"].read_text())
    cells = report["results"]["e8"]["metrics"]["cells"]
    lab = cells["labels"]["4"]["per_seed_eval"]
    lab[1]["per_instance"]["energy_gap_rel"] = [
        1.1 * x for x in lab[1]["per_instance"]["energy_gap_rel"]]
    lab[0]["per_instance"]["energy_gap_rel"] = [0.1, 0.2, 0.6, 0.3, 0.5, 0.4]  # the rule: #3
    kn = cells["labels_knorm"]["4"]["per_seed_eval"][0]["per_instance"]
    kn["disp_rel_l2"][0] *= 1.1                                # one outlier: still used
    rp = tmp_path / "report.json"
    rp.write_text(json.dumps(report))
    prov = tmp_path / "provenance.txt"
    _write_provenance(prov, rp, [exported["paths"][k] for k in sorted(exported["paths"])])
    out = tmp_path / "spectra"
    r = _run("--report", rp, "--provenance", prov, "--states-dir", exported["sdir"],
             "--n-val", 3, "--out", out)
    assert r.returncode == 5, r.stderr[-2000:]
    res = json.loads((out / "spectra.json").read_text())
    assert res["content_mismatch"] == ["labels_s0", "labels_s1"] and res["val"]["n"] == 3
    assert not res["content_checks"]["labels_s1"]["ok"]
    assert np.isclose(res["content_checks"]["labels_s1"]["energy_gap_rel_median_rel_dev"],
                      0.1 / 1.1)
    k0 = res["content_checks"]["labels_knorm_s0"]
    assert k0["ok"] and np.isclose(k0["disp_rel_l2_max_rel_dev"], 0.1 / 1.1)
    assert k0["disp_rel_l2_median_rel_dev"] < 1e-6
    assert all(res["content_checks"][f"{r_}_s{s}"]["ok"] for r_ in ("ar", "labels_knorm", "mgn")
               for s in SEEDS)
    assert res["figures"]["fig2d"]["index"] == 3
    S = np.load(out / "spectra_val.npz")
    assert S["pi"].shape[2] == 3
    full = np.load(exported["out"] / "spectra_val.npz")
    assert np.allclose(S["eK"], full["eK"][:, :, :3], rtol=1e-12)
    F = np.load(out / "fig2d.npz")
    for ri, row in enumerate(ROWS):
        assert np.allclose(F[f"pi_{row}"], full["pi"][ri, 0, 3], rtol=1e-12)
        assert np.allclose(F[f"C2_{row}"].sum(-1), full["e2"][ri, 0, 3], rtol=1e-9)
    assert np.allclose(F["C2_star"].sum(-1), full["u2_star"][3], rtol=1e-9)
    assert np.allclose(F["gamma_star"], full["gamma_star"][3], rtol=1e-12)


def test_the_ieee_switch_restores_the_policy():
    """The second pass turns TF32 off with the codebase's own switch and puts
    the run's policy back afterwards, also when the pass fails."""
    import torch

    from fejepa.runtime import setup_torch

    mod = _mod()
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    sdp = lambda: (torch.backends.cuda.flash_sdp_enabled(),                # noqa: E731
                   torch.backends.cuda.mem_efficient_sdp_enabled(),
                   torch.backends.cuda.math_sdp_enabled())
    before = sdp()
    try:
        setup_torch("cpu", tf32=True)
        with mod.ieee_fp32("cpu", True):
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert torch.get_float32_matmul_precision() == "highest"
            assert sdp() == (False, False, True)          # the attention by the math backend
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "high"
        assert sdp() == before
        with pytest.raises(RuntimeError):
            with mod.ieee_fp32("cpu", True):
                raise RuntimeError("a failing pass")
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.get_float32_matmul_precision() == "high" and sdp() == before
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]
        torch.set_float32_matmul_precision(saved[2])


def test_the_second_pass_is_recorded_as_the_rounding(exported, tmp_path, monkeypatch):
    """In process, with every prediction perturbed only inside the second
    pass's switch (where the attention is restricted to the math backend): the
    export records the perturbation as u - u_ieee, the IEEE prediction's error
    beside the run's, the content check reads the run's own predictions, and
    the policy and the attention backends are back afterwards."""
    import torch

    mod = _mod()
    real = mod.predict
    rough = {}

    def perturbed(model, arch, dev):
        U = real(model, arch, dev)
        if torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled():
            d = 1e-2 * np.abs(arch.U_star).max() * np.sin(np.arange(U.shape[1]) ** 2)[None, :] \
                * np.asarray(arch.free_mask, float)              # the same for every model
            rough[arch.path.name] = d
            return U - d
        return U

    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    sdp = (torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled(),
           torch.backends.cuda.math_sdp_enabled())
    out = tmp_path / "spectra"
    monkeypatch.setattr(mod, "predict", perturbed)
    monkeypatch.setattr(sys, "argv", ["cm2d_spectra.py", "--report", str(exported["report"]),
                                      "--provenance", str(exported["prov"]), "--states-dir",
                                      str(exported["sdir"]), "--n-val", "2", "--device", "cpu",
                                      "--out", str(out)])
    try:
        mod.main()                                     # no mismatch: returns, no SystemExit
        assert (torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled()) == sdp
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]
        torch.set_float32_matmul_precision(saved[2])
    res = json.loads((out / "spectra.json").read_text())
    assert res["content_mismatch"] == []
    S = np.load(out / "spectra_val.npz")
    for i, f in enumerate(S["files"]):
        d = rough[str(f)]
        assert np.allclose(S["d2_tf32"][:, :, i], (d ** 2).sum(-1), rtol=1e-6)
    assert np.allclose(S["SK_log_tf32"].sum(-1), S["dK_tf32"], rtol=1e-9)
    assert not np.allclose(S["e2_ieee"], S["e2"], rtol=1e-6)
    full = np.load(exported["out"] / "spectra_val.npz")
    assert np.allclose(S["e2"], full["e2"][:, :, :2], rtol=1e-12)    # the run's predictions
    F = np.load(out / "fig2d.npz")                     # here beyond --n-val: its own two passes
    assert res["figures"]["fig2d"]["index"] >= 2
    d = rough[res["figures"]["fig2d"]["file"]]
    for row in ROWS:
        assert np.allclose(F[f"U_ieee_{row}"], F[f"U_{row}"] - d, rtol=0, atol=1e-12)
