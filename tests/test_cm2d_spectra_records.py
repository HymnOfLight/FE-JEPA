"""cmame-paper Stage 8: the return of RUNBOOK_CMAME Sec. D (the error spectra
of CM2D's models, run on the box on 9 October 2026 and copied byte for byte
into records/cmame/spectra/) is complete and tied to what it read and to
CM2D's verdict: one export on Stage 7's commit, exit 0; its inputs are CM2D's
committed report and provenance file and the twelve states that file lists;
every model reproduced the report's per-instance arrays; the identities the
export checks hold on every array; the summaries, the counts and the
eigenvalue ranges follow from the per-load arrays, recomputed with the
script's functions and, for the readings the manuscript may quote, written
out again; the export's means are the verdict's (H1, H2a, H2b and H3, seed by
seed); and the figure file is the instance the rule selects, its stiffness
matrix reassembled here from its mesh."""

import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CM = ROOT / "records" / "cmame"
SP = CM / "spectra"
EXP = SP / "export"
RET = CM / "cm2d" / "return"
VERDICT = CM / "cm2d" / "verdict.json"
PREREG = ROOT / "PREREG_CM2D.md"
HEAD = "8b5d4436ee4a91836b627ef852cd2f5e98d88b0f"        # Stage 7
TREE = "e288619b0338a6fa497dfabba6c953e8b3a4e553"
FILES = ("export/fig2d.npz", "export/spectra.json", "export/spectra_val.npz", "machine.txt",
         "pytest.log", "spectra.log", "status.txt")
ROWS = ("ar", "labels", "labels_knorm", "mgn")
SEEDS = (0, 1, 2)
N_VAL, LOADS = 256, 4
FIG = (221, "instance_27356.npz")
METRICS = ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err", "crit_recall")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(p) -> dict:
    return json.loads(Path(p).read_text())


def _mod():
    spec = importlib.util.spec_from_file_location("cm2d_spectra",
                                                  ROOT / "scripts" / "cm2d_spectra.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _section(text: str, title: str) -> str:
    m = re.search(r"^## " + re.escape(title) + r".*?(?=^## |\Z)", text, re.M | re.S)
    assert m, title
    return m.group(0)


def _npz() -> dict:
    with np.load(EXP / "spectra_val.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _per_instance(rep: dict, row: str, seed_index: int) -> dict:
    budget = "1024"
    cell = rep["results"]["e8"]["metrics"]["cells"][row][budget]
    return cell["per_seed_eval"][seed_index]["per_instance"]


def _rel_dev(got, want) -> np.ndarray:
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert got.shape == want.shape and (want != 0).all()
    return np.abs(got - want) / np.abs(want)


def test_the_files_are_the_ones_the_readme_lists():
    sec = _section((CM / "README.md").read_text(encoding="utf-8"), "CM2D's error spectra")
    listed = dict(re.findall(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", sec, re.M))
    present = sorted(p.relative_to(SP).as_posix() for p in SP.rglob("*") if p.is_file())
    assert present == sorted(FILES) == sorted(listed)
    for name in present:
        assert _sha(SP / name) == listed[name], name


def test_one_export_on_stage_7s_commit():
    assert [x for x in (SP / "status.txt").read_text().splitlines() if x.strip()] == \
        ["spectra.log exit=0"]
    machine = (SP / "machine.txt").read_text().splitlines()
    assert machine[0] == f"HEAD {HEAD}" and machine[1] == f"tree {TREE}"
    # git status printed nothing; the GPU, its driver and the 16 CPUs the README quotes
    assert machine[2] == "NVIDIA GeForce RTX 5090, 595.71.05, 32607 MiB"
    assert "nproc 16" in machine and "1600000 100000" in machine
    text = (SP / "pytest.log").read_text()
    summary = r"^535 passed, 2 deselected(, \d+ warnings?)? in [0-9.]+s \(\d+:\d\d:\d\d\)$"
    assert re.search(summary, text, re.M)
    progress = "".join(re.sub(r"\[\s*\d+%\]", "", x).strip()
                       for x in text.split("=" * 10, 1)[0].splitlines())
    assert progress == "." * 535                                  # every test passed
    res = _json(EXP / "spectra.json")
    assert res["what"] == ("cmame-paper Stage 7: spectral content of the errors of CM2D's "
                           "models (post hoc; reported only)")
    assert res["git"] == "prereg-cm2d-2-g" + HEAD[:7]              # describe, not -dirty
    torch = {"torch": "2.12.1+cu130", "cuda": "13.0", "device": "cuda", "tf32_matmul": True,
             "tf32_cudnn": True, "gpu": "NVIDIA GeForce RTX 5090"}
    assert res["torch"] == torch and res["torch_after"] == torch    # the policy after the IEEE pass
    assert res["precision"] == {
        "tf32_policy": True,
        "ieee_pass": "every model predicts every instance a second time under "
                     "fejepa.runtime.setup_torch(tf32=False), the attention by the math SDPA "
                     "backend; the run's policy and the default backends are restored after "
                     "each instance"}
    # the timings the README quotes, and their order
    t_inf = sum(res["inference_seconds"].values())
    assert res["eigen"]["seconds"] + res["inference_seconds_ieee"] + t_inf <= res["val_seconds"]
    assert res["val_seconds"] <= res["seconds"]
    assert (round(res["seconds"]), round(res["eigen"]["seconds"]),
            round(res["inference_seconds_ieee"])) == (70, 31, 14)
    # the log: one progress line per 16 instances (times not decreasing), the figure
    # line, the summary line, which is what the script prints from the JSON it wrote
    log = (SP / "spectra.log").read_text().splitlines()
    assert len(log) == N_VAL // 16 + 2
    times = []
    for k, line in enumerate(log[:-2]):
        m = re.fullmatch(rf"\[spectra\] val {16 * (k + 1)}/{N_VAL} \| (\d+) s", line)
        assert m, line
        times.append(int(m.group(1)))
    assert times == sorted(times) and times[-1] <= res["val_seconds"] + 0.5
    assert log[-2] == f"[spectra] fig2d: val #{FIG[0]} {FIG[1]} (547 nodes)"
    diag = res["diagnostics"]
    printed = {"content_median_rel_dev": {k: [c["disp_rel_l2_median_rel_dev"],
                                              c["energy_gap_rel_median_rel_dev"]]
                                          for k, c in res["content_checks"].items()},
               "content_mismatch": res["content_mismatch"],
               "rayleigh_ratio_median": {r: v["rayleigh_ratio_median"]
                                         for r, v in diag["rows"].items()},
               "pairs": {k: v["share_first_above"] for k, v in diag["pairs"].items()},
               "prop1_bound_ratio_max": {r: v["prop1_bound_ratio_max"]
                                         for r, v in diag["rows"].items()},
               "figures": {k: [v["file"], v["index"]] for k, v in res["figures"].items()}}
    assert json.loads(log[-1]) == printed


def test_the_inputs_are_cm2ds_committed_return():
    sys.path.insert(0, str(ROOT / "src"))
    from fejepa.report import read_prereg_entries

    mod, res = _mod(), _json(EXP / "spectra.json")
    rep = _json(RET / "report.json")
    assert res["report"] == "records/cmame/cm2d/return/report.json"
    assert res["report_sha256"] == _sha(RET / "report.json")
    assert res["provenance"] == "records/cmame/cm2d/return/provenance.txt"
    assert res["provenance_sha256"] == _sha(RET / "provenance.txt")
    stamped = dict(read_prereg_entries(PREREG))["cm2d_v1"]
    assert res["config_sha256"] == stamped == rep["provenance"]["config_sha256"]
    assert (res["budget"], res["pool"], res["seeds"]) == (1024, 1024, list(SEEDS))
    assert res["rows"] == list(ROWS) == list(mod.ROWS)
    assert res["rank_bins"] == mod.RANK_BINS and res["decades"] == list(mod.DECADES)
    assert res["log_edges"] == [float(x) for x in mod.LOG_EDGES]
    assert res["pairs"] == [list(p) for p in mod.PAIRS] and res["rules"] == mod.RULES
    assert res["content_tol"] == mod.CONTENT_TOL
    # the twelve states: each the one CM2D's provenance file lists; the label-free
    # ones also the ones the report's d9_restart record (E1's states) names
    prov = (RET / "provenance.txt").read_text()
    d9 = rep["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    assert sorted(res["states"]) == sorted(f"{r}_s{s}" for r in ROWS for s in SEEDS)
    for key, st in res["states"].items():
        row, s = key.rsplit("_s", 1)
        if row == "ar":
            assert (st["dir"], st["file"]) == ("runs/e1_2d_base/e8_states", f"ar_p1024_s{s}.pt")
            assert st["checked_against"] == ["provenance", "d9_restart"]
            assert st["sha256"] == d9[f"s{s}"]["sha256"], key
        else:
            assert (st["dir"], st["file"]) == ("runs/cm2d/e8_states", f"{row}_b1024_s{s}.pt")
            assert st["checked_against"] == ["provenance"]
        assert st["sha256_ok"] is True, key
        assert re.search(rf"^{st['sha256']}  {st['dir']}/{st['file']}$", prov, re.M), key
    # the run's own validation split (E1's corpus of 30,000, split seed 1)
    split, n = rep["config"]["split"], int(rep["config"]["data"]["n"])
    perm = np.random.default_rng(int(split["seed"])).permutation(n)
    want = [f"instance_{perm[k]:05d}.npz" for k in range(int(split["n_val"]))]
    assert res["val"] == {"n": N_VAL, "files": want}
    assert [str(f) for f in _npz()["files"]] == want


def test_every_model_reproduced_the_reports_arrays():
    """The export's content check, and the same comparison recomputed here from the
    per-load arrays: their means over the load cases are the report's per-instance
    values (the transformers' to round-off; the graph network's CUDA scatter
    reductions are not bitwise reproducible)."""
    res, rep, z = _json(EXP / "spectra.json"), _json(RET / "report.json"), _npz()
    assert res["content_mismatch"] == []
    for key, c in res["content_checks"].items():
        assert c["ok"] is True and c["n"] == N_VAL, key
        assert res["states"][key]["content"] == c, key
        devs = [c[f"{m}_{q}_rel_dev"] for m in METRICS for q in ("median", "max")]
        if key.startswith("mgn_"):
            assert max(c[f"{m}_median_rel_dev"] for m in ("disp_rel_l2", "energy_gap_rel")) \
                <= res["content_tol"], key
            assert max(c[f"{m}_median_rel_dev"] for m in METRICS) < 1e-3, key
        else:
            assert devs == [0.0] * len(devs), key
    u2 = z["u2_star"]
    for ri, row in enumerate(ROWS):
        for si in range(len(SEEDS)):
            want = _per_instance(rep, row, si)
            c = res["content_checks"][f"{row}_s{SEEDS[si]}"]
            for k, got in (("energy_gap_rel", z["rel"][ri, si]), ("vm_rel_l2", z["vm_elem"][ri, si]),
                           ("disp_rel_l2", np.sqrt(z["e2"][ri, si] / u2))):
                dev = _rel_dev(got.mean(axis=-1), want[k])
                if row == "mgn":
                    # the recorded deviations are these, recomputed (median and largest)
                    for q, x in (("median", np.median(dev)), ("max", dev.max())):
                        assert math.isclose(x, c[f"{k}_{q}_rel_dev"], rel_tol=1e-9), (si, k, q)
                    assert np.median(dev) <= res["content_tol"], (si, k)
                else:
                    assert dev.max() <= 1e-12, (row, si, k)


def test_the_identities_hold_on_every_array():
    z = _npz()
    shape = (len(ROWS), len(SEEDS), N_VAL, LOADS)
    for k in ("pi", "rel", "c", "rel_c", "e2", "eK", "vm_elem", "vm_area", "stress_energy",
              "u_dirichlet_max", "e2_ieee", "eK_ieee", "d2_tf32", "dK_tf32"):
        assert z[k].shape == shape and np.isfinite(z[k]).all(), k
    for k, nb in (("S2_rank", 40), ("SK_rank", 40), ("S2_log", 70), ("SK_log", 70),
                  ("S2_log_ieee", 70), ("SK_log_ieee", 70), ("SK_log_tf32", 70)):
        assert z[k].shape == (*shape, nb) and (z[k] >= 0).all(), k
    pi_s, u2, uK = z["pi_star"], z["u2_star"], z["uK_star"]
    assert (pi_s < 0).all() and (u2 > 0).all() and (uK > 0).all()
    e2, eK = z["e2"], z["eK"]
    g = eK / uK                                  # squared relative energy-norm error
    close = lambda a, b, rtol: np.allclose(a, b, rtol=rtol, atol=0)        # noqa: E731
    # Lemma "Exactness": the gap from the energies is the error's stiffness norm
    assert close(eK, 2 * (z["pi"] - pi_s), 1e-8) and close(z["rel"], g, 1e-8)
    # Corollary "Zero-field test and ranking", on every load case and pair of rows
    assert ((z["pi"] > 0) == (eK > uK)).all()
    for a in range(len(ROWS)):
        for b in range(a + 1, len(ROWS)):
            assert ((z["pi"][a] < z["pi"][b]) == (eK[a] < eK[b])).all(), (ROWS[a], ROWS[b])
    # Proposition "Energy-optimal amplitude": c* u is worse than neither u nor the zero
    # field, and its gap is the one c* gives (Pi_h(c u) = -c^2 Pi_h(u) / (1 - 2c) when
    # c = F^T u / u^T K u)
    c, rel, rel_c = z["c"], z["rel"], z["rel_c"]
    assert (rel_c <= rel * (1 + 1e-9) + 1e-12).all() and (rel_c <= 1 + 1e-9).all()
    assert close(rel_c, (-c ** 2 * z["pi"] / (1 - 2 * c) - pi_s) / np.abs(pi_s), 1e-7)
    # Proposition "Energy gap and stress error", plane stress: the identity and the bound;
    # gamma* from the stored integrals and moduli, Young's modulus 1 (the generator's)
    assert close(z["stress_energy"], eK, 1e-9)
    assert (z["vm_area"] ** 2 <= (1 + z["gamma_star"]) * g * (1 + 1e-9)).all()
    G, Bk = z["G"][:, None], z["Bk"][:, None]
    assert close(z["gamma_star"], 3 * G * z["p2_area_star"] / (Bk * z["vm2_area_star"]), 1e-12)
    assert close(9 * z["Bk"] * z["G"] / (3 * z["Bk"] + z["G"]), np.ones(N_VAL), 1e-12)
    vm_id = (1 + z["gamma_star"]) * z["vm2_area_star"] / (3 * G)
    assert close(vm_id, uK, 1e-9) and close(-2 * pi_s, uK, 1e-9)
    # the spectra sum to the norms: the errors' and the solution's in both binnings, the
    # IEEE predictions' errors' and the rounding's in the log binning
    for S, x in (("S2_rank", e2), ("S2_log", e2), ("SK_rank", eK), ("SK_log", eK),
                 ("S2_log_ieee", z["e2_ieee"]), ("SK_log_ieee", z["eK_ieee"]),
                 ("SK_log_tf32", z["dK_tf32"]), ("S2_rank_star", u2), ("S2_log_star", u2),
                 ("SK_rank_star", uK), ("SK_log_star", uK)):
        assert close(z[S].sum(axis=-1), x, 1e-9), S
    # every Rayleigh quotient (the errors', the IEEE errors', the rounding's and the
    # solution's) lies between the smallest and the largest eigenvalue
    lo, hi = z["lam_min"][:, None], z["lam_max"][:, None]
    for q in (eK / e2, z["eK_ieee"] / z["e2_ieee"], z["dK_tf32"] / z["d2_tf32"], uK / u2):
        assert (q >= lo * (1 - 1e-9)).all() and (q <= hi * (1 + 1e-9)).all()
    assert close(z["rq_star_over_lam_min"], (uK / u2) / z["lam_min"][:, None], 1e-12)
    assert (z["u_dirichlet_max"] == 0).all() and (z["ustar_dirichlet_max"] == 0).all()
    assert (z["lam_min"] > 0).all() and (z["n_free"] > 0).all()


def _log_bin_range(z: dict) -> tuple:
    """(n, L, 70) the eigenvalues a mode in each log bin can have: the bin's
    edges times RQ*, within the instance's smallest and largest eigenvalue."""
    rq = (z["uK_star"] / z["u2_star"])[..., None]
    lo_e = np.concatenate([[-np.inf], z["log_edges"]])
    hi_e = np.concatenate([z["log_edges"], [np.inf]])
    lo = np.maximum(10.0 ** lo_e * rq, z["lam_min"][:, None, None])
    hi = np.minimum(10.0 ** hi_e * rq, z["lam_max"][:, None, None])
    return lo, hi


def test_the_spectra_fit_the_eigenvalues_and_the_rounding_is_tf32s():
    """In every occupied bin the stiffness-weighted mass over the Euclidean mass is
    an eigenvalue average, so it lies in that bin's eigenvalue range (a shift of
    mass between bins, or a spectrum not the norm's, breaks it); the outermost log
    bins are empty (the edges cover every mesh's modes). The rounding u - u_ieee
    is bracketed the same way, and on every prediction it is of TF32's size: the
    second pass was not the first one again."""
    z = _npz()
    tol = 1e-9
    lo, hi = _log_bin_range(z)
    for a, b, wl, wh in (("S2_log", "SK_log", lo, hi), ("S2_log_ieee", "SK_log_ieee", lo, hi),
                         ("S2_log_star", "SK_log_star", lo, hi)):
        S2, SK = z[a], z[b]
        assert (S2[..., 0] == 0).all() and (S2[..., -1] == 0).all(), a
        assert (SK[..., 0] == 0).all() and (SK[..., -1] == 0).all(), b
        assert ((S2 > 0) == (SK > 0)).all(), a
        with np.errstate(all="ignore"):
            q = SK / S2
        assert ((S2 == 0) | ((q >= wl * (1 - tol)) & (q <= wh * (1 + tol)))).all(), a
    # a rank bin holds the modes from its first eigenvalue to the next bin's first
    start = z["lam_rank_start"][:, None, :]
    end = np.concatenate([z["lam_rank_start"][:, 1:], z["lam_max"][:, None]], axis=1)[:, None, :]
    for a, b in (("S2_rank", "SK_rank"), ("S2_rank_star", "SK_rank_star")):
        q = z[b] / z[a]
        assert ((q >= start * (1 - tol)) & (q <= end * (1 + tol))).all(), a
    T = z["SK_log_tf32"]
    assert (T[..., 0] == 0).all() and (T[..., -1] == 0).all()
    d2 = z["d2_tf32"]
    assert (d2 >= (T / hi).sum(axis=-1) * (1 - tol)).all()
    assert (d2 <= (T / lo).sum(axis=-1) * (1 + tol)).all()
    # TF32 rounds products to about 5e-4; the smallest relative rounding of a whole
    # prediction recorded is 1.5e-4 (a reordering of IEEE sums would be about 1e-7)
    assert (np.sqrt(d2 / z["u2_star"]) > 1e-5).all()


def test_the_summaries_follow_from_the_arrays():
    """spectra.json's counts, summaries and eigenvalue ranges are what the
    script's own functions compute from the committed arrays (floats to 1e-12:
    the box's numpy may round a logarithm or an exponential differently in the
    last place)."""
    mod, res, z = _mod(), _json(EXP / "spectra.json"), _npz()
    D = {k: z[k] for k in (*mod.DIAG, *mod.SPECTRA, *mod.IEEE, *mod.IEEE_SPECTRA)}
    R = {k: z[k] for k in mod.REF}
    R.update({k: z[f"{k}_star"] for k in mod.SPECTRA})
    R.update({k: z[k] for k in ("n_free", "lam_min", "lam_max", "lam_rank_start", "G", "Bk")})
    assert [str(r) for r in z["rows"]] == list(ROWS) and list(z["seeds"]) == list(SEEDS)
    assert int(z["rank_bins"]) == mod.RANK_BINS and (z["log_edges"] == mod.LOG_EDGES).all()
    used = np.ones((len(ROWS), len(SEEDS)), dtype=bool)
    with np.errstate(all="ignore"):
        got = json.loads(json.dumps({"diagnostics": mod.summary(D, R, used),
                                     "counts": mod.counts(D, used)}, allow_nan=False))

    def same(a, b, path):
        if isinstance(a, dict):
            assert isinstance(b, dict) and set(a) == set(b), path
            for k in a:
                same(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]")
        elif isinstance(a, float) or isinstance(b, float):
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=0), (path, a, b)
        else:
            assert a == b, (path, a, b)

    same(got["diagnostics"], res["diagnostics"], "diagnostics")
    same(got["counts"], res["counts"], "counts")
    e = res["eigen"]
    assert e["n_free_min_median_max"] == [int(z["n_free"].min()), float(np.median(z["n_free"])),
                                          int(z["n_free"].max())]
    assert e["lam_min_min"] == float(z["lam_min"].min())
    sk = np.sqrt(z["lam_max"] / z["lam_min"])
    assert e["sqrt_kappa_min_max"] == [float(sk.min()), float(sk.max())]
    # the rank bins' first eigenvalues ascend from the smallest (the figure
    # instance's are checked against its eigenvalues below)
    assert (z["lam_rank_start"][:, 0] == z["lam_min"]).all()
    assert (np.diff(z["lam_rank_start"], axis=1) >= 0).all()


def test_the_readings_written_out_again():
    """The readings the manuscript may quote, from the definitions rather than the
    script: per row the median normalised Rayleigh quotient, the shares of the
    error above 10^k RQ*, the IEEE quotient and the rounding's share, overall and
    above 10^k RQ*; of the solution, its shares and its quotient over the smallest
    eigenvalue; per pair the shares, the ratios of means and the geometric-mean
    ratios, overall, per seed and for the IEEE predictions. (Per load case
    g = rho d^2, so the recorded gap ratio is the product of the other two.)"""
    res, z = _json(EXP / "spectra.json"), _npz()
    e2, eK, u2, uK = z["e2"], z["eK"], z["u2_star"], z["uK_star"]
    rho = (eK / e2) / (uK / u2)
    g, d2 = eK / uK, e2 / u2
    rho_i = (z["eK_ieee"] / z["e2_ieee"]) / (uK / u2)
    g_i, d2_i = z["eK_ieee"] / uK, z["e2_ieee"] / u2
    gm = lambda r: math.exp(float(np.mean(np.log(r))))                   # noqa: E731
    near = lambda a, b: math.isclose(a, b, rel_tol=1e-12)                 # noqa: E731
    first = lambda k: int(np.flatnonzero(np.isclose(z["log_edges"], k))[0]) + 1   # noqa: E731
    share = lambda S, k: S[..., first(k):].sum(axis=-1) / S.sum(axis=-1)          # noqa: E731
    for ri, row in enumerate(ROWS):
        d = res["diagnostics"]["rows"][row]
        assert d["n"] == len(SEEDS) * N_VAL * LOADS
        assert near(d["rayleigh_ratio_median"], float(np.median(rho[ri])))
        assert near(d["ieee"]["rayleigh_ratio_median"], float(np.median(rho_i[ri])))
        for k in (1, 2, 3, 4):
            for name, S in (("e2", z["S2_log"][ri]), ("eK", z["SK_log"][ri])):
                assert near(d[f"share_{name}_above_median"][f"1e{k}"],
                            float(np.median(share(S, k))))
        assert near(d["tf32"]["dK_over_eK_median"], float(np.median(z["dK_tf32"][ri] / eK[ri])))
        for k in (2, 3, 4):                     # the rounding's share in the stiff modes
            j = first(k)
            den = z["SK_log"][ri][..., j:].sum(axis=-1)
            num = z["SK_log_tf32"][ri][..., j:].sum(axis=-1)
            assert d["tf32"]["dK_over_eK_above_n"][f"1e{k}"] == int((den > 0).sum())
            assert near(d["tf32"]["dK_over_eK_above_median"][f"1e{k}"],
                        float(np.median(num[den > 0] / den[den > 0])))
        assert near(d["rel_gap_median"], float(np.median(z["rel"][ri])))
    ref = res["diagnostics"]["reference"]
    for k in (1, 2, 3, 4):
        assert near(ref["share_u2_above_median"][f"1e{k}"],
                    float(np.median(share(z["S2_log_star"], k))))
        assert near(ref["share_uK_above_median"][f"1e{k}"],
                    float(np.median(share(z["SK_log_star"], k))))
    assert near(ref["rayleigh_over_lam_min_median"],
                float(np.median((uK / u2) / z["lam_min"][:, None])))
    vm = z["vm_elem"]
    for a, b in (("labels", "labels_knorm"), ("labels", "ar"), ("labels_knorm", "ar"),
                 ("mgn", "labels"), ("mgn", "ar")):
        ia, ib = ROWS.index(a), ROWS.index(b)
        p = res["diagnostics"]["pairs"][f"{a}_vs_{b}"]
        assert p["n"] == len(SEEDS) * N_VAL * LOADS
        assert near(p["share_first_above"], float(np.mean(rho[ia] > rho[ib])))
        seed_gm = lambda x: np.exp(np.log(x).mean(axis=0))                 # noqa: E731
        assert near(p["share_first_above_seed_geomean"],
                    float(np.mean(seed_gm(rho[ia]) > seed_gm(rho[ib]))))
        assert near(p["median_ratio"], float(np.median(rho[ia] / rho[ib])))
        assert near(p["geomean_ratio_gap"], gm(g[ia] / g[ib]))
        assert near(p["geomean_ratio_rayleigh"], gm(rho[ia] / rho[ib]))
        assert near(p["geomean_ratio_disp_sq"], gm(d2[ia] / d2[ib]))
        assert math.isclose(p["geomean_ratio_gap"],                      # as quoted
                            p["geomean_ratio_rayleigh"] * p["geomean_ratio_disp_sq"],
                            rel_tol=1e-12)
        assert near(p["ratio_of_means_gap"], float(g[ia].mean() / g[ib].mean()))
        assert near(p["ratio_of_means_vm_elem"], float(vm[ia].mean() / vm[ib].mean()))
        assert near(p["share_first_disp_lower"], float(np.mean(d2[ia] < d2[ib])))
        assert near(p["share_first_disp_lower_gap_higher"],
                    float(np.mean((d2[ia] < d2[ib]) & (g[ia] > g[ib]))))
        for si, s in enumerate(p["by_seed"]):
            assert s["seed_index"] == si and s["n"] == N_VAL * LOADS
            assert near(s["geomean_ratio_gap"], gm(g[ia, si] / g[ib, si]))
            assert near(s["geomean_ratio_rayleigh"], gm(rho[ia, si] / rho[ib, si]))
            assert near(s["geomean_ratio_disp_sq"], gm(d2[ia, si] / d2[ib, si]))
        q = p["ieee"]
        assert near(q["share_first_above"], float(np.mean(rho_i[ia] > rho_i[ib])))
        assert near(q["geomean_ratio_gap"], gm(g_i[ia] / g_i[ib]))
        assert near(q["geomean_ratio_rayleigh"], gm(rho_i[ia] / rho_i[ib]))
        assert near(q["geomean_ratio_disp_sq"], gm(d2_i[ia] / d2_i[ib]))


def test_the_export_reproduces_the_verdict():
    """End to end (RUNBOOK_CMAME D3): the export's seed means of the relative
    energy gap (here the squared energy-norm error, computed from the error) and
    of the von Mises error are the verdict's, seed by seed, for every
    hypothesis; and the pair readings' ratios of means are the verdict's
    base_mean / new_mean (4.596 for H2a, 2.215 for H2b)."""
    res, v, z = _json(EXP / "spectra.json"), _json(VERDICT), _npz()
    g = z["eK"] / z["uK_star"]
    seed_mean = lambda x, row: x[ROWS.index(row)].mean(axis=(1, 2))      # noqa: E731
    plan = {"H1": ("ar", "labels", "energy_gap_rel"),
            "H2a": ("labels_knorm", "labels", "energy_gap_rel"),
            "H2b": ("labels_knorm", "labels", "vm_rel_l2"),
            "H3": ("labels_knorm", "ar", "energy_gap_rel")}
    for h, (new, base, metric) in plan.items():
        assert v[h]["metric"] == metric, h
        x = g if metric == "energy_gap_rel" else z["vm_elem"]
        for side, row in (("new", new), ("base", base)):
            assert np.allclose(seed_mean(x, row), v[h][f"{side}_per_seed"], rtol=1e-12, atol=0), \
                (h, side)
    pairs = res["diagnostics"]["pairs"]
    for h, key, ratio in (("H2a", "labels_vs_labels_knorm", "ratio_of_means_gap"),
                          ("H2b", "labels_vs_labels_knorm", "ratio_of_means_vm_elem"),
                          ("H1", "labels_vs_ar", "ratio_of_means_gap")):
        assert math.isclose(pairs[key][ratio], v[h]["base_mean"] / v[h]["new_mean"],
                            rel_tol=1e-12), h
    assert math.isclose(pairs["labels_knorm_vs_ar"]["ratio_of_means_gap"],
                        v["H3"]["new_mean"] / v["H3"]["base_mean"], rel_tol=1e-12)
    assert round(pairs["labels_vs_labels_knorm"]["ratio_of_means_gap"], 3) == 4.596
    assert round(pairs["labels_vs_labels_knorm"]["ratio_of_means_vm_elem"], 3) == 2.215


def test_the_figure_file_is_the_rules_instance():
    res, rep, z = _json(EXP / "spectra.json"), _json(RET / "report.json"), _npz()
    lab0 = np.asarray(_per_instance(rep, "labels", 0)["energy_gap_rel"], float)
    i = int(np.argsort(lab0, kind="stable")[(lab0.size - 1) // 2])
    assert (i, str(z["files"][i])) == FIG
    assert res["selections"]["fig2d"] == {"set": "val", "index": FIG[0],
                                          "value": float(lab0[i]), "file": FIG[1]}
    assert res["figures"]["fig2d"] == dict(res["selections"]["fig2d"], n_nodes=547,
                                           n_elements=992, n_free=1064)
    with np.load(EXP / "fig2d.npz", allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        assert (meta["figure"], meta["set"], meta["index"], meta["file"], meta["seed"]) == \
            ("fig2d", "val", FIG[0], FIG[1], 0)
        assert meta["rule"] == res["rules"]["fig2d"]
        assert (f["log_edges"] == z["log_edges"]).all()
        assert int(z["n_free"][i]) == f["lam"].size == 1064
        assert f["lam"][0] == z["lam_min"][i] and f["lam"][-1] == z["lam_max"][i]
        edges = np.floor(np.arange(40) * 1064 / 40).astype(np.int64)
        assert (z["lam_rank_start"][i] == f["lam"][edges]).all()
        for k in ("pi_star", "u2_star", "uK_star", "gamma_star"):
            assert (f[k] == z[k][i]).all(), k
        assert (f["rq_star"] == z["uK_star"][i] / z["u2_star"][i]).all()
        for ri, row in enumerate(ROWS):
            for k in ("pi", "rel", "c", "rel_c"):
                assert (f[f"{k}_{row}"] == z[k][ri, 0, i]).all(), (row, k)


def test_the_figure_instance_reassembled_from_its_mesh():
    """The figure file's stiffness matrix, rebuilt here from its mesh and material
    by the generator's own assembly: its free block has the stored eigenvalues,
    U* solves it with the stored loads, and the stored per-mode coefficients,
    norms, energies, gaps, c*, spectra and von Mises stresses of every row's
    seed-0 error, IEEE prediction and rounding are those of the stored fields."""
    pytest.importorskip("skfem")
    sys.path.insert(0, str(ROOT / "src"))
    from fejepa.fe.elasticity import assemble_plate
    from fejepa.fe.stress import _geometry, element_von_mises

    mod, z = _mod(), _npz()
    i = FIG[0]
    with np.load(EXP / "fig2d.npz", allow_pickle=False) as f:
        F = {k: f[k] for k in f.files}
    material = json.loads(str(F["meta"]))["material"]
    nodes, tris = F["nodes"], F["elements"]
    K, _, mask = assemble_plate(nodes, tris, material, float(nodes[:, 0].max()),
                                float(nodes[:, 1].max()), np.ones(4))
    assert (mask == F["dirichlet_mask"]).all()
    free = ~mask
    Kf = K[free][:, free].toarray()
    lam = np.linalg.eigvalsh(Kf)
    assert np.allclose(lam, F["lam"], rtol=1e-8, atol=1e-12 * lam[-1])
    S = F["U_star"]
    assert (S[:, mask] == 0).all()
    res_ = Kf @ S[:, free].T - F["F"][:, free].T
    assert np.abs(res_).max() <= 1e-9 * np.abs(F["F"]).max()          # U* is the FE solution
    quad = lambda X: np.einsum("ld,ld->l", X, (K @ X.T).T)                # noqa: E731
    close = lambda a, b, rtol=1e-9: np.allclose(a, b, rtol=rtol, atol=0)  # noqa: E731
    assert close(quad(S), F["uK_star"]) and close((S * S).sum(axis=1), F["u2_star"], 1e-12)
    pi = lambda U: 0.5 * quad(U) - np.einsum("ld,ld->l", F["F"], U)      # noqa: E731
    assert close(pi(S), F["pi_star"])
    rq = F["uK_star"] / F["u2_star"]
    idx = np.searchsorted(mod.LOG_EDGES, np.log10(F["lam"][None, :] / rq[:, None]), side="right")

    def spectra(C2):
        return (np.stack([np.bincount(idx[l], weights=C2[l], minlength=70) for l in range(LOADS)]),
                np.stack([np.bincount(idx[l], weights=C2[l] * F["lam"], minlength=70)
                          for l in range(LOADS)]))

    S2s, SKs = spectra(F["C2_star"])
    assert close(S2s, z["S2_log_star"][i], 1e-9) and close(SKs, z["SK_log_star"][i], 1e-9)
    area, _, _ = _geometry(nodes, tris)
    assert close(F["area"], area, 1e-12)
    vm = lambda U: np.stack([element_von_mises(nodes, tris, U[l], material)   # noqa: E731
                             for l in range(LOADS)])
    vm_s = vm(S)
    assert close(F["vm_ref"], vm_s, 1e-6)                         # stored as float32
    a = np.abs(F["pi_star"])
    for ri, row in enumerate(ROWS):
        U, Ui = F[f"U_{row}"], F[f"U_ieee_{row}"]
        e, ei, d = U - S, Ui - S, U - Ui
        at = (ri, 0, i)
        assert close((e * e).sum(axis=1), z["e2"][at], 1e-12) and close(quad(e), z["eK"][at])
        assert close((ei * ei).sum(axis=1), z["e2_ieee"][at], 1e-12)
        assert close(quad(ei), z["eK_ieee"][at])
        assert close((d * d).sum(axis=1), z["d2_tf32"][at], 1e-12)
        assert close(quad(d), z["dK_tf32"][at])
        assert close(pi(U), F[f"pi_{row}"]) and close((pi(U) - F["pi_star"]) / a, F[f"rel_{row}"], 1e-8)
        c = np.einsum("ld,ld->l", F["F"], U) / quad(U)              # c* = F^T u / u^T K u
        assert close(c, F[f"c_{row}"])
        assert close((pi(c[:, None] * U) - F["pi_star"]) / a, F[f"rel_c_{row}"], 1e-7)
        C2 = F[f"C2_{row}"]
        assert close(C2.sum(axis=1), z["e2"][at]) and close(C2 @ F["lam"], z["eK"][at])
        S2, SK = spectra(C2)
        assert close(S2, z["S2_log"][at], 1e-9) and close(SK, z["SK_log"][at], 1e-9)
        vm_u = vm(U)
        assert close(F[f"vm_{row}"], vm_u, 1e-6)
        dv = vm_u - vm_s
        assert close(np.linalg.norm(dv, axis=1) / np.linalg.norm(vm_s, axis=1),
                     z["vm_elem"][at], 1e-10)
        assert close(np.sqrt((dv ** 2) @ area / ((vm_s ** 2) @ area)), z["vm_area"][at], 1e-10)
