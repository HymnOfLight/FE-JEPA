"""The higher-fidelity T-stub model (fejoint.tstub_hf) and the baseline extensions of fejoint.tstub.

Fixtures are synthetic T-stubs (VT1, VT2), not specimens of the gate: no test of Girao Coelho et al. (2004)
is solved here."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dataclasses import replace
import numpy as np
import pytest
from fejoint.tstub_hf import WeldedTStub, analyse, BOLTS
from fejoint.tstub import TStubPair, analyse as base_analyse
from fejoint import linsolve

ROOT = pathlib.Path(__file__).resolve().parents[1]
VT2 = WeldedTStub(name="VT2", b=100.0, w=100.0, n=35.0, a_w=4.0, rows=(22.5, 77.5), bolt=16,
                  E_f=205000.0, E_w=211000.0, E_b=214000.0, t_f=12.0, t_w=8.0)
VT1 = WeldedTStub(name="VT1", b=80.0, w=100.0, n=35.0, a_w=4.0, rows=(32.0,), bolt=20,
                  E_f=205000.0, E_w=211000.0, E_b=214000.0, t_f=12.0, t_w=8.0)


def kkt_ok(k, tol=1e-8):
    return (k["min_gap"] > -1e-12 and k["min_lambda"] > -tol and k["max_complementarity"] < tol
            and k["max_lambda_inactive"] < tol and k["max_residual_unconstrained"] < tol)


@pytest.fixture(scope="module")
def vt2():
    return analyse(VT2, "coarse", return_data=True)


@pytest.fixture(scope="module")
def vt1():
    return analyse(VT1, "coarse")


def test_bolt_calibration_hits_agerskov_length(vt2, vt1):
    for o, g in ((vt2, VT2), (vt1, VT1)):
        cal = o["calibration"]
        assert abs(cal["rel_error"]) < 1e-9
        bs = BOLTS[g.bolt]
        A_b = np.pi * bs.d ** 2 / 4
        L = bs.As * (1.43 * 2 * g.t_f + 0.71 * bs.m + 2 * 0.1 * bs.m) / A_b      # Bursi and Jaspart eqs (1), (2)
        assert abs(cal["target_mm_per_N"] - L / 2 / (g.E_b * bs.As)) < 1e-15
    assert abs(replace(VT2, bolt_length="ec3").L_bolt - (24.0 + 0.5 * (10.0 + 14.8))) < 1e-12


def test_equilibrium_optimality_and_contact_pattern(vt2, vt1):
    for o in (vt2, vt1):
        assert abs(o["equilibrium_rel"]) < 1e-9          # applied share + prying = head force
        assert abs(o["shank_reaction_rel"]) < 1e-9       # head force = force through the shank's mid-section
        assert kkt_ok(o["kkt"])
        assert o["prying_over_F"] > 0.05                 # the flange tips bear on each other
        assert 0.0 < o["contact_fraction_flanges"] < 0.6
        assert o["contact_fraction_heads"] > 0.5
        assert o["bearing_faces_dropped"] == 0
    # off-centre row: the far end opens more than the near end
    assert vt1["gaps_mm"][1] > vt1["gaps_mm"][0] > 0


def test_head_gaps_and_forces_from_the_fields(vt2):
    d = vt2["_data"]
    msh, u, sol, pos = d["msh"], d["u"], d["sol"], d["pos"]
    pr = msh["pairs"][0]
    s = u[pr["a"], 0] - u[pr["f"], 0]                   # separations recomputed from the displacements
    assert s.min() > -1e-12 * np.abs(u).max()
    mu = sol["lam"][pos[3 * pr["f"]]]
    assert mu.min() > -1e-9 * mu.max()
    assert np.abs(mu * s).max() < 1e-9 * mu.max() * np.abs(u).max()


def test_homogeneity():
    a = analyse(VT2, "coarse", F=10000.0)
    b = analyse(VT2, "coarse", F=37000.0)
    assert abs(a["K_kN_per_mm"] / b["K_kN_per_mm"] - 1) < 1e-9
    assert abs(a["prying_over_F"] - b["prying_over_F"]) < 1e-9


def test_quarter_model_equals_whole_length(vt2):
    full = analyse(replace(VT2, full=True), "coarse")
    assert abs(full["K_kN_per_mm"] / vt2["K_kN_per_mm"] - 1) < 1e-7
    assert abs(full["gaps_mm"][0] / full["gaps_mm"][1] - 1) < 1e-7
    assert abs(full["prying_over_F"] - vt2["prying_over_F"]) < 1e-7
    assert abs(full["calibration"]["E_shank"] / vt2["calibration"]["E_shank"] - 1) < 1e-9


def test_stiffer_bolt_material_stiffens_and_bolt_length_order():
    a = analyse(VT2, "coarse")
    b = analyse(replace(VT2, bolt_length="ec3"), "coarse")
    # the longer of the two bolt lengths (here Agerskov's, 37.3 mm against 36.4 mm) gives the softer T-stub
    La, Lb = VT2.L_bolt, replace(VT2, bolt_length="ec3").L_bolt
    assert La != Lb and (a["K_kN_per_mm"] < b["K_kN_per_mm"]) == (La > Lb)
    c = analyse(replace(VT2, E_f=2 * VT2.E_f, E_w=2 * VT2.E_w, E_b=2 * VT2.E_b), "coarse")
    assert abs(c["K_kN_per_mm"] / a["K_kN_per_mm"] - 2) < 1e-7    # all moduli doubled: K doubles


# ---- baseline (fejoint.tstub) --------------------------------------------------------------------

def test_baseline_defaults_reproduce_the_stored_bursi_jaspart_runs(monkeypatch):
    ref = json.load(open(ROOT / "probes" / "tstub_bj1997.json"))
    T1 = TStubPair(t_f=10.7, t_w=7.1, b=150.0, L=80.0, w=90.0, bolt_y=(20.0,), H=220.0, corner="root", r=15.0,
                   d=12.0, As=84.3, head=7.5, nut=10.8, bearing_r=12.0, n_washers=2, washer_t=2.5, x_meas=150.0)
    o = base_analyse(T1, h=5.0, n_tf=2)                  # current solver (PARDISO if available)
    assert abs(o["K_kN_per_mm"] / ref[0]["K_kN_per_mm"] - 1) < 1e-10
    assert abs(o["bolt_force_total_N"] / ref[0]["bolt_force_total_N"] - 1) < 1e-10
    monkeypatch.setattr(linsolve, "_PARDISO", None)     # the solver the stored runs used: bit for bit
    o = base_analyse(T1, h=5.0, n_tf=2)
    assert o["K_kN_per_mm"] == ref[0]["K_kN_per_mm"]
    assert o["bolt_force_total_N"] == ref[0]["bolt_force_total_N"]


BASE = TStubPair(t_f=12.0, t_w=8.0, b=170.0, L=100.0, w=100.0, bolt_y=(22.5,), H=80.0, corner="weld", r=5.657,
                 d=16.0, As=157.0, head=10.0, nut=14.8, bearing_r=11.245, x_meas=42.0,
                 E_f=205000.0, E_w=211000.0, E_b=214000.0, measure="gap")


def test_baseline_gap_reading_quarter_equals_whole_length():
    q = base_analyse(BASE, h=5.0, n_tf=2)
    f = base_analyse(replace(BASE, full_length=True, bolt_y=(22.5, 77.5)), h=5.0, n_tf=2)
    assert abs(f["K_kN_per_mm"] / q["K_kN_per_mm"] - 1) < 1e-7
    assert abs(f["gaps_mm"][0] / f["gaps_mm"][1] - 1) < 1e-7
    assert kkt_ok(q["kkt"]) and kkt_ok(f["kkt"])
