"""Bolt heads in unilateral contact with the end plate (change of variables, see docs/contact_energy_note.md,
Section 7), the other bolt models, and the small examples of Sections 4 and 5 of that note.

Joint fixture: a coarse mesh of the verification geometry VER1 (not one of the paper's specimens)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
import pytest
from fejoint import linsolve
from fejoint.contact import solve_signorini
from fejoint.joint import JointGeometry, analyse

RNG = np.random.default_rng(20261010)
VER1 = JointGeometry(tp=12.0, hp=300.0, bp=100.0, eX=25.0, p=70.0, p23=135.0, w=60.0, LX=60.0,
                     hb=200.0, bb=100.0, tfb=8.5, twb=5.6, L_beam=1000.0, L_load=800.0, x_dt1=720.0,
                     t_fc=25.0, d=16.0, As=157.0, head=10.0, nut=13.0, washer_r=15.0)
COARSE = dict(h=10.0, n_tp=1, n_tf=1)


def Pi(A, b, v):
    return 0.5 * v @ (A @ v) - b @ v


def kkt_ok(k, tol=1e-9):
    return (k["min_gap"] > -1e-12 and k["min_lambda"] > -tol and k["max_complementarity"] < tol
            and k["max_lambda_inactive"] < tol and k["max_residual_unconstrained"] < tol)


def head_checks(out, d, tol=1e-6):
    """Recompute the head gaps from the fields and check each head's force and moment balance:
    k_axial a = sum(mu_i - kreg_i s_i) and k_rot (b, c) = sum((dy_i, dz_i) (mu_i - kreg_i s_i))."""
    g = VER1
    nodes, T, red_of, pos, nh, ndof = d["nodes"], d["T"], d["red_of"], d["pos"], d["nh"], d["ndof"]
    zr = np.zeros(T.shape[1]); free = np.nonzero(pos >= 0)[0]; zr[free] = d["sol"]["x"]
    lam_r = np.zeros(T.shape[1]); lam_r[free] = d["sol"]["lam"]
    uf = T @ zr
    for k, (pt, r) in enumerate(zip(d["patches"], g.rows)):
        hs = d["head_slots"][k]
        a = uf[hs]; b_ = uf[hs + 1] if nh == 3 else 0.0; c_ = uf[hs + 2] if nh == 3 else 0.0
        dy, dz = nodes[pt, 1] - r, nodes[pt, 2] - g.w / 2
        s_fields = a + b_ * dy + c_ * dz - uf[3 * pt]
        s_slot = zr[red_of[3 * pt]]
        assert np.max(np.abs(s_fields - s_slot)) <= 1e-12 * max(1.0, np.abs(uf).max())
        mu = lam_r[red_of[3 * pt]]
        assert mu.min() >= -tol * max(1.0, np.abs(mu).max())
        net = mu - d["kreg"][red_of[3 * pt]] * s_slot
        assert abs(d["head_k"][nh * k] * a - net.sum()) <= tol * max(1.0, abs(net).sum())
        if nh == 3:
            for kk, arm in ((1, dy), (2, dz)):
                assert abs(d["head_k"][nh * k + kk] * uf[hs + kk] - (arm * net).sum()) <= tol * max(1.0, np.abs(arm * net).sum())
    return True


@pytest.fixture(scope="module")
def head():
    return analyse(VER1, washer="head_contact", k_rot_factor=4.0, **COARSE)


def test_kkt_and_both_kinds_of_contact(head):
    out, d = head
    assert kkt_ok(out["kkt"])
    act = d["sol"]["active"]; nc = d["n_column"]
    assert 0 < act[:nc].sum() < nc                   # plate partly on, partly off the column flange
    assert act[nc:].any() and not act[nc:].all()     # some washer nodes bear on their heads, some lift off
    assert out["bolt_heads"]["s_min_mm"] >= -1e-12


def test_transformation_and_head_equilibrium(head):
    assert head_checks(*head)


@pytest.mark.parametrize("factor", [None, 0.0, 12.0])
def test_other_head_variants(factor):
    out, d = analyse(VER1, washer="head_contact", k_rot_factor=factor, **COARSE)
    assert kkt_ok(out["kkt"])
    assert head_checks(out, d)


def test_tied_support_keeps_heads_unilateral():
    out, d = analyse(VER1, washer="head_contact", k_rot_factor=4.0, support="tied", **COARSE)
    assert d["n_column"] == 0 and d["cidx"].size == d["s_slots"].size
    assert out["contact_fraction"] == 1.0 and kkt_ok(out["kkt"])
    assert head_checks(out, d)
    rig, _ = analyse(VER1, washer="rigid", support="tied", **COARSE)
    assert rig["kkt"] is None and np.isfinite(rig["S_a_kNm_per_mrad"]) and rig["S_a_kNm_per_mrad"] > 0


def test_statements_hold_in_transformed_variables(head):
    _, d = head
    A, b, cidx, sol = d["A"], d["b"], d["cidx"], d["sol"]
    u, lam = sol["x"], sol["lam"]
    Pu, nu_ = Pi(A, b, u), np.sqrt(u @ (A @ u))
    assert abs(lam @ u) <= 1e-9 * abs(b @ u)                                  # Proposition 2
    assert abs(Pu / (-0.5 * nu_ ** 2) - 1) < 1e-9
    sol3 = solve_signorini(A, 3.0 * b, cidx)                                   # proportionality
    assert np.array_equal(sol3["active"], sol["active"])
    assert np.linalg.norm(sol3["x"] - 3 * u) <= 1e-9 * np.linalg.norm(3 * u)
    act = sol["active"]
    for k in range(120):
        s = np.abs(u).max() * 10.0 ** RNG.uniform(-3, 0.5)
        if k % 3 == 0:
            v = u + s * RNG.standard_normal(u.size)
        elif k % 3 == 1:
            v = RNG.uniform(0.2, 1.8) * u + 0.1 * s * RNG.standard_normal(u.size)
        else:
            v = u.copy(); lift = np.zeros(cidx.size); lift[act] = s * RNG.uniform(0, 1, act.sum()); v[cidx] += lift
        v[cidx] = np.maximum(v[cidx], 0.0)
        gap = Pi(A, b, v) - Pu; e2 = (v - u) @ (A @ (v - u))
        assert abs(gap - 0.5 * e2 - lam @ v) <= 1e-8 * max(abs(gap), abs(Pu))  # Lemma 1
        assert gap - 0.5 * e2 >= -1e-8 * abs(Pu)                                # Proposition 1
        if Pi(A, b, v) <= 0:
            assert np.sqrt(e2) <= nu_ * (1 + 1e-9)                             # Proposition 3
        tp = max(0.0, (b @ v) / (v @ (A @ v)))
        assert Pi(A, b, tp * v) <= 1e-12 * abs(Pu)                             # Proposition 5


def test_no_tilt_tied_reproduces_rigid_washer():
    a, _ = analyse(VER1, washer="rigid", **COARSE)
    t, _ = analyse(VER1, washer="tilt_tied", k_rot_factor=None, **COARSE)
    assert abs(t["S_a_kNm_per_mrad"] / a["S_a_kNm_per_mrad"] - 1) < 1e-10
    assert abs(t["S_b_kNm_per_mrad"] / a["S_b_kNm_per_mrad"] - 1) < 1e-10


def test_regularisation_small_and_bonding_limit(head):
    out, _ = head
    lo, _ = analyse(VER1, washer="head_contact", k_rot_factor=4.0, reg=1e-9, **COARSE)
    assert abs(lo["S_a_kNm_per_mrad"] / out["S_a_kNm_per_mrad"] - 1) < 1e-6
    hi, _ = analyse(VER1, washer="head_contact", k_rot_factor=4.0, reg=1e4, **COARSE)
    tied, _ = analyse(VER1, washer="tilt_tied", k_rot_factor=4.0, **COARSE)
    assert kkt_ok(tied["kkt"])
    assert abs(hi["S_a_kNm_per_mrad"] / tied["S_a_kNm_per_mrad"] - 1) < 1e-3


def test_iterative_branch_matches_direct(head, monkeypatch):
    out, _ = head
    monkeypatch.setattr(linsolve, "_PARDISO", None)
    monkeypatch.setattr(linsolve, "DIRECT_MAX", 1000)    # force AMG-preconditioned CG inside the active set loop
    it, _ = analyse(VER1, washer="head_contact", k_rot_factor=4.0, **COARSE)
    assert abs(it["S_a_kNm_per_mrad"] / out["S_a_kNm_per_mrad"] - 1) < 1e-7
    assert abs(it["S_b_kNm_per_mrad"] / out["S_b_kNm_per_mrad"] - 1) < 1e-7
    assert kkt_ok(it["kkt"], tol=1e-8)


def test_pardiso_branch_matches_superlu(monkeypatch):
    if linsolve._PARDISO is None:
        pytest.skip("pypardiso not available")
    with monkeypatch.context() as m:                     # SuperLU throughout
        m.setattr(linsolve, "_PARDISO", None)
        m.setattr(linsolve, "DIRECT_MAX", 10 ** 9)
        lu, _ = analyse(VER1, washer="head_contact", k_rot_factor=4.0, **COARSE)
    monkeypatch.setattr(linsolve, "PARDISO_MIN", 0)      # PARDISO throughout
    pa, _ = analyse(VER1, washer="head_contact", k_rot_factor=4.0, **COARSE)
    assert abs(pa["S_a_kNm_per_mrad"] / lu["S_a_kNm_per_mrad"] - 1) < 1e-9
    assert abs(pa["S_b_kNm_per_mrad"] / lu["S_b_kNm_per_mrad"] - 1) < 1e-9
    assert kkt_ok(pa["kkt"], tol=1e-8)


# ---- the small examples in the note (n = 2, K = I) -------------------------------------------

def _solve2(f, g=0.0):
    A = sp.identity(2, format="csr"); b = np.asarray(f, float)
    return A, b, solve_signorini(A, b, np.array([0]), g=np.array([g]))


def test_example_fail_direction_bound_is_sharp():
    A, b, sol = _solve2([-1.0, 1.0])
    assert np.allclose(sol["x"], [0, 1]) and np.allclose(sol["lam"], [1, 0])
    v = np.array([0.7, 1.0])                        # t between sqrt(2) - 1 and 1
    assert Pi(A, b, v) > 0                          # flagged by the zero-field screen
    assert np.linalg.norm(v - sol["x"]) < np.linalg.norm(sol["x"])   # yet better than the zero field
    t0 = np.sqrt(2) - 1                             # the bound of Proposition 4: Pi = 0 there
    assert abs(Pi(A, b, np.array([t0, 1.0]))) < 1e-14


def test_example_ranking_flips():
    A, b, sol = _solve2([-1.0, 1.0])
    v1, v2 = np.array([0.3, 1.0]), np.array([0.0, 1.5])
    assert Pi(A, b, v2) < Pi(A, b, v1)
    assert np.linalg.norm(v2 - sol["x"]) > np.linalg.norm(v1 - sol["x"])


def test_example_energy_and_regression_choose_differently():
    A, b, sol = _solve2([-1.0, 1.0])
    ts = np.linspace(-0.5, 1.0, 150001)             # model class v(t) = (t + 1/2, 1 - t), feasible for t >= -1/2
    V = np.stack([ts + 0.5, 1 - ts], axis=1)
    E = 0.5 * (V ** 2).sum(1) - V @ b
    R = ((V - sol["x"]) ** 2).sum(1)
    assert abs(ts[E.argmin()] + 0.5) < 1e-9 and abs(R[E.argmin()] - 0.25) < 1e-9
    assert abs(ts[R.argmin()] + 0.25) < 1e-4 and abs(R.min() - 0.125) < 1e-8


def test_example_initial_gap():
    A, b, sol = _solve2([-3.0, 0.0], g=1.0)
    u = sol["x"]
    assert np.allclose(u, [-1, 0]) and np.allclose(sol["lam"], [2, 0])
    mu_g, q, nu_ = 2.0, 2.0, 1.0
    v = np.array([-1.0, 2.0])                      # feasible: v_1 >= -1
    assert Pi(A, b, v) <= 0 and np.linalg.norm(v - u) > nu_              # pass direction fails
    assert np.linalg.norm(v - u) ** 2 <= nu_ ** 2 + 2 * mu_g + 1e-12      # what survives
    assert abs(Pi(A, b, u) - (-0.5 * nu_ ** 2 - mu_g)) < 1e-12           # equation (2)
    tp = min(1.0, max(0.0, (b @ v) / (v @ v)))      # clipped rescaling
    w = tp * v
    assert abs(tp - 0.6) < 1e-12 and abs(Pi(A, b, w) + 0.9) < 1e-12
    assert abs(np.linalg.norm(w - u) - np.sqrt(1.6)) < 1e-12 and np.linalg.norm(w - u) > nu_
    assert abs(np.sqrt(q ** 2 + nu_ ** 2 + 2 * mu_g) - q - nu_) < 1e-12  # gap form of Proposition 4 equals |u*| here


def test_beam_segment_reproduces_full_beam():
    full, _ = analyse(VER1, washer="head_contact", k_rot_factor=4.0, **COARSE)
    for segl in (400.0, 200.0):
        cut, d = analyse(VER1, washer="head_contact", k_rot_factor=4.0, beam_segment=segl, **COARSE)
        assert kkt_ok(cut["kkt"])
        assert cut["mesh"]["dof_reduced"] < full["mesh"]["dof_reduced"]
        assert abs(cut["S_paper_kNm_per_mrad"] / full["S_paper_kNm_per_mrad"] - 1) < 5e-3
        assert abs(cut["S_a_kNm_per_mrad"] / full["S_a_kNm_per_mrad"] - 1) < 5e-3
