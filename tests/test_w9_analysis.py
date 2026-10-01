"""wp9 Stage 0a: the C0 reading functions (fejepa.analysis.w9) and the
session-1 decision rules (scripts/w9_session1_decisions.py)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from fejepa.analysis.w9 import (amplitude_row, cg_energy_drops, ensemble_row, energy,
                                gap_estimate_row, k_norm2, spearman)
from fejepa.data.archive import InstanceArchive

ROOT = Path(__file__).resolve().parents[1]


def _arch(n=30, L=3, seed=0):
    """A random SPD 'instance' with two Dirichlet dofs, labelled."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    K = sp.csr_matrix(A @ A.T + n * np.eye(n))
    F = rng.standard_normal((L, n))
    mask = np.zeros(n, dtype=bool)
    mask[:2] = True
    F[:, mask] = 0.0
    free = ~mask
    U = np.zeros((L, n))
    U[:, free] = np.linalg.solve(K.toarray()[np.ix_(free, free)], F[:, free].T).T
    meta = {"material": {"E": 1.0, "nu": 0.3}, "extra": {"target_h": 0.1}}
    return InstanceArchive(nodes=np.zeros((n // 2, 2)), elements=np.zeros((1, 3), int), K=K,
                           F=F, dirichlet_mask=mask, meta=meta, U_star=U)


def _preds(arch, S=3, scale=0.2, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for s in range(S):
        U = arch.U_star * (1 + 0.1 * s) + scale * rng.standard_normal(arch.U_star.shape)
        U[:, arch.dirichlet_mask] = 0.0
        out[s] = U
    return out


def test_ensemble_identity_is_exact():
    a = _arch()
    row = ensemble_row(a, _preds(a))
    assert row["identity_residual"] < 1e-12
    assert math.isclose(row["gap_seed_mean"], row["gap_ensemble"] + row["D"], rel_tol=1e-10)
    assert row["D"] > 0 and row["D_labelfree"] > 0


def test_cg_energy_drops_match_scipy_and_bound_the_gap():
    from fejepa.fe.solve import cg_k_steps

    a = _arch(n=40, seed=3)
    U = _preds(a, S=1, scale=0.5)[0]
    free = a.free_mask
    Kff = a.K[free][:, free]
    for li in range(a.n_loads):
        drops = cg_energy_drops(Kff, a.F[li][free], U[li][free], (1, 3, 5, 10))
        gap = 0.5 * k_norm2(Kff, (U[li] - a.U_star[li])[free])[0]
        vals = [drops[k] for k in (1, 3, 5, 10)]
        assert all(0 <= v <= gap * (1 + 1e-12) for v in vals)        # lower bounds
        assert all(x <= y + 1e-15 for x, y in zip(vals, vals[1:]))   # non-decreasing
        for k in (1, 3, 5):                                          # same iterates as scipy
            xk = cg_k_steps(a.K, a.F[li], free, U[li], k)[free]
            pk = energy(Kff, a.F[li][free], xk)[0]
            p0 = energy(Kff, a.F[li][free], U[li][free])[0]
            assert math.isclose(p0 - pk, drops[k], rel_tol=1e-6, abs_tol=1e-12)
    row = gap_estimate_row(a, U, (1, 5, 10))
    assert set(row["est_rel"]) == {"1", "5", "10"}
    assert all(0 <= row["captured"][k] <= 1 + 1e-12 for k in row["captured"])


def test_amplitude_row_recovers_a_pure_amplitude_error():
    a = _arch(seed=4)
    row = amplitude_row(a, 0.8 * a.U_star)           # right shape, 80% amplitude
    assert np.allclose(row["c_star"], 1.25, rtol=1e-10)
    assert math.isclose(row["c_battery"], 1.25, rel_tol=1e-10)
    assert row["disp_c"] < 1e-10 and row["egap_c"] < 1e-12
    assert math.isclose(row["disp"], 0.2, rel_tol=1e-10)


def test_spearman_handles_ties_and_degenerate_input():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 2, 3], [3, 2, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert math.isnan(spearman([1, 2], [1, 2]))


# ------------------------------------------------------------- decisions --

def _dec():
    spec = importlib.util.spec_from_file_location("dec", ROOT / "scripts" / "w9_session1_decisions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _inputs(ratio=2.0, cstar=1.2, on=52.0, off=30.0, valid=True, rss=(1e9, 4e9, 12.5e9, 25e9),
            usable=100e9, trainval=0.95, rho=0.95, gpu=None):
    marks = [{"n": n, "rss": r} for n, r in zip((1024, 4096, 12800, 25600), rss, strict=False)]
    memory = {"limits": {"usable": usable}, "workers": 3, "marks": marks}
    if gpu is not None:                 # (total, context, reserved at 25,600, step on, step off)
        total, ctx, res25, s_on, s_off = gpu
        for m in marks:
            m["gpu_reserved"] = res25 * m["n"] / 25600
        memory["gpu"] = {"total": total, "context_estimate": ctx,
                         "step": {"ckpt_on": {"reserved_overhead": s_on},
                                  "ckpt_off": {"reserved_overhead": s_off}}}
    return dict(
        amp2d={"summary": {"ratio_disp_F5_over_inband": ratio, "c_star_median_F5": cstar}},
        timing={"variants": {"worker3": {"ar": {"ms_per_step_per_unit": on, "valid": valid}},
                             "worker3_no_ckpt": {"ar": {"ms_per_step_per_unit": off,
                                                        "valid": True}}}},
        memory=memory,
        trainval={"summary": {"egap_train_over_val": trainval}},
        val={"summary": {"cg_estimates": {"10": {"spearman_seed_mean": rho}}}})


def test_decision_S_enters_needs_both_conditions():
    d = _dec().decide
    assert d(**_inputs())["S_enters"]["value"] is True
    assert d(**_inputs(ratio=1.4))["S_enters"]["value"] is False
    assert d(**_inputs(cstar=1.05))["S_enters"]["value"] is False
    assert d(**_inputs(ratio=1.5, cstar=1.10))["S_enters"]["value"] is True     # inclusive
    assert d(**_inputs(ratio=1.4999, cstar=1.10))["S_enters"]["value"] is False
    assert d(**_inputs(ratio=1.5, cstar=1.0999))["S_enters"]["value"] is False


def test_decision_checkpointing():
    d = _dec().decide
    assert d(**_inputs(on=52, off=30))["checkpointing_off"]["value"] is True
    assert d(**_inputs(on=52, off=45))["checkpointing_off"]["value"] is False   # 1.16x
    assert d(**_inputs(on=65, off=50))["checkpointing_off"]["value"] is True    # 1.3x: inclusive
    assert d(**_inputs(on=64.9, off=50))["checkpointing_off"]["value"] is False
    assert d(**_inputs(valid=False))["checkpointing_off"]["value"] is False     # invalid: stays on
    t = _inputs()
    t["timing"] = {"variants": {}}
    assert d(**t)["checkpointing_off"]["value"] is False
    # 32 GB card, 1 GB context, 4 GB packs at 25,600: three workers fit with the
    # 2 GB step (3 x 7 = 21 <= 25.6) but not with an 8 GB step (3 x 13 = 39):
    # checkpointing stays on rather than costing the parallel schedule
    g = (32e9, 1e9, 4e9, 2e9, 8e9)
    r = d(**_inputs(gpu=g))
    assert (r["pool_max"]["n"], r["pool_max"]["mode"]) == (25600, "parallel")
    assert r["checkpointing_off"]["value"] is False and r["checkpointing_off"]["fits_without"] is False
    r = d(**_inputs(gpu=(32e9, 1e9, 4e9, 2e9, 3e9)))                     # 3 x 8 = 24: fits
    assert r["checkpointing_off"]["value"] is True


def test_decision_pool_cap_keeps_the_design_before_the_schedule():
    d = _dec().decide
    # 3 x 25 GB = 75 <= 80: parallel at 25,600
    p = d(**_inputs())["pool_max"]
    assert (p["n"], p["mode"]) == (25600, "parallel")
    # 3 x 25 = 75 > 64 but 25 <= 64: one worker at a time at 25,600 (before 12,800 x 3)
    p = d(**_inputs(usable=80e9))["pool_max"]
    assert (p["n"], p["mode"]) == (25600, "sequential")
    # 25 > 16 at 25,600 for one worker; 3 x 12.5 = 37.5 > 16; 12.5 <= 16: sequential 12,800
    p = d(**_inputs(usable=20e9))["pool_max"]
    assert (p["n"], p["mode"]) == (12800, "sequential")
    # nothing fits: decided, and the design is revisited
    p = d(**_inputs(usable=10e9))["pool_max"]
    assert p["n"] is None and p["value"] == {"n": None, "mode": None} and "error" not in p
    # the boundary is inclusive: 3 x 20 = 60 = 0.8 x 75
    p = d(**_inputs(rss=(1e9, 4e9, 10e9, 20e9), usable=75e9))["pool_max"]
    assert (p["n"], p["mode"]) == (25600, "parallel")
    p = d(**_inputs(rss=(1e9, 4e9, 10e9, 20.001e9), usable=75e9))["pool_max"]
    assert (p["n"], p["mode"]) == (25600, "sequential")
    # the audit stopped at 4,096: residency is extrapolated from its last two marks
    p = d(**_inputs(rss=(1e9, 4e9), usable=200e9))["pool_max"]
    assert p["host_per_worker_bytes"]["25600"] == pytest.approx(4e9 + (25600 - 4096) * 3e9 / 3072)
    # the device binds although the host does not: 3 x (1 + 9 + 2) = 36 > 25.6;
    # one worker fits at 25,600
    p = d(**_inputs(usable=1e12, gpu=(32e9, 1e9, 9e9, 2e9, 2e9)))["pool_max"]
    assert (p["n"], p["mode"]) == (25600, "sequential")
    assert p["gpu_per_worker_bytes_ckpt_on"]["12800"] == pytest.approx(1e9 + 4.5e9 + 2e9)


def test_decision_memory_counts_the_steps_growth_and_the_later_context():
    d = _dec().decide
    t = _inputs(usable=100e9, gpu=(32e9, 1e9, 4e9, 2e9, 2e9))
    # host: 3 x 25 = 75 <= 80, but the steps grew the process by 2 GB: 3 x 27 = 81 > 80
    t["memory"]["rss_after_step"] = 25e9 + 2e9
    p = d(**t)["pool_max"]
    assert (p["n"], p["mode"]) == (25600, "sequential")
    assert p["host_per_worker_bytes"]["25600"] == pytest.approx(27e9)
    # device: a context read after the steps (2.5 GB) replaces the earlier 1 GB
    t = _inputs(usable=1e12, gpu=(32e9, 1e9, 4e9, 2e9, 2e9))
    t["memory"]["gpu"]["context_after_step"] = 2.5e9
    p = d(**t)["pool_max"]
    assert p["gpu_per_worker_bytes_ckpt_on"]["25600"] == pytest.approx(2.5e9 + 4e9 + 2e9)
    assert (p["n"], p["mode"]) == (25600, "parallel")             # 3 x 8.5 = 25.5 <= 25.6
    t["memory"]["gpu"]["context_after_step"] = 2.6e9               # 3 x 8.6 = 25.8 > 25.6
    assert d(**t)["pool_max"]["mode"] == "sequential"


def test_decision_records_c04_and_c2():
    d = _dec().decide
    assert "overfits" in d(**_inputs(trainval=0.7))["C0_4_prediction"]["reading"]
    assert "parity" in d(**_inputs(trainval=0.9))["C0_4_prediction"]["reading"]
    assert "parity" in d(**_inputs(trainval=0.8))["C0_4_prediction"]["reading"]  # < 0.8 only
    assert d(**_inputs(rho=0.95))["C2_score_usable"]["value"] is True
    assert d(**_inputs(rho=0.9))["C2_score_usable"]["value"] is True             # inclusive
    assert d(**_inputs(rho=0.85))["C2_score_usable"]["value"] is False
    assert d(**_inputs(rho=float("nan")))["C2_score_usable"]["value"] is None
    t = _inputs()
    t["val"] = {"summary": {"cg_estimates": {"5": {"spearman_seed_mean": 0.99}}}}
    assert d(**t)["C2_score_usable"]["value"] is None


def test_decision_rules_fail_one_at_a_time():
    """A missing or malformed input leaves only the rules that read it undecided."""
    d = _dec().decide
    t = _inputs(gpu=(32e9, 1e9, 4e9, 2e9, 2e9))
    del t["memory"]["gpu"]["step"]                 # the step measurement failed on the box
    t["memory"]["gpu"]["step_error"] = "RuntimeError: CUDA out of memory"
    r = d(**t)
    assert r["pool_max"]["value"] is None and "out of memory" in r["pool_max"]["error"]
    assert r["checkpointing_off"]["value"] is None and "rule 2" in r["checkpointing_off"]["error"]
    assert r["S_enters"]["value"] is True and r["C2_score_usable"]["value"] is True
    t = _inputs()
    t["trainval"] = None
    r = d(**t)
    assert r["C0_4_prediction"]["value"] is None and "missing" in r["C0_4_prediction"]["error"]
    assert r["S_enters"]["value"] is True and r["pool_max"]["n"] == 25600
