"""The higher-fidelity joint model (fejoint.joint_hf). Fixture: the synthetic joint VER1 (not a specimen)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dataclasses import replace
import numpy as np
import scipy.sparse as sp
import pytest
from fejoint.joint import JointGeometry
from fejoint.joint_hf import analyse, build, L_eff_agerskov, MESHES
from fejoint.linsolve import SPDSolver

VER1 = JointGeometry(tp=12.0, hp=300.0, bp=100.0, eX=25.0, p=70.0, p23=135.0, w=60.0, LX=60.0,
                     hb=200.0, bb=100.0, tfb=8.5, twb=5.6, L_beam=1000.0, L_load=800.0, x_dt1=720.0,
                     t_fc=25.0, d=16.0, As=157.0, head=10.0, nut=14.8, washer_r=11.245,
                     E_plate=205000.0, E_flange=211000.0, E_web=207000.0, E_bolt=214000.0)
COARSE = MESHES["coarse"]


def kkt_ok(k, tol=1e-8):
    return (k["min_gap"] > -1e-12 and k["min_lambda"] > -tol and k["max_complementarity"] < tol
            and k["max_lambda_inactive"] < tol and k["max_residual_unconstrained"] < tol)


@pytest.fixture(scope="module")
def plain():
    return analyse(VER1, COARSE, return_data=True)


@pytest.fixture(scope="module")
def welded():
    return analyse(VER1, COARSE, a_flange=4.0, a_web=3.0)


def test_equilibrium_optimality_and_bolt_forces(plain, welded):
    for o in (plain, welded):
        assert abs(o["vertical_equilibrium_rel"]) < 1e-8
        assert abs(o["horizontal_equilibrium_rel"]) < 1e-8
        assert kkt_ok(o["kkt"])
        assert np.allclose(o["bolt_force_N"], o["bolt_force_from_spring_N"], rtol=1e-8, atol=1e-6)
        assert o["bearing_faces_dropped"] == 0
        f = o["bolt_force_N"]
        assert f[1] > f[0] > f[2] >= 0           # inner tension row most, the compression row least
        assert 0 < o["contact_fraction_column"] < 1


def test_whole_bolt_compliance_is_agerskov(plain):
    """One bolt alone (row 2), its bottom plane on its springs to the column, loaded by a uniform pressure on
    its bearing ring: the ring's mean displacement over the load is L_eff / (E_b A_s)."""
    d = plain["_data"]
    msh, K, diag, head_k = d["msh"], d["K"], d["diag"], d["head_k"]
    nodes, body = msh["nodes"], msh["body"]
    i = 1
    bn = np.nonzero(body == 1 + i)[0]
    bot = d["bottoms"][i]
    r, zc = VER1.rows[i], VER1.w / 2
    full = np.r_[3 * bn, 3 * bn + 1, 3 * bn + 2]                 # bolt dofs, then a, b, c
    nf = full.size
    pos = {dof: k for k, dof in enumerate(full)}
    keep = [k for k, dof in enumerate(full) if dof not in set(3 * bot)]
    col = {k: j for j, k in enumerate(keep)}
    nr = len(keep) + 3
    rows_, cols_, vals_ = [], [], []
    for k in keep:
        rows_.append(k); cols_.append(col[k]); vals_.append(1.0)
    for j in bot:                                               # u_x = a + b (y - r) + c (z - zc)
        k = pos[3 * j]
        rows_ += [k, k, k]; cols_ += [nr - 3, nr - 2, nr - 1]
        vals_ += [1.0, nodes[j, 1] - r, nodes[j, 2] - zc]
    Tb = sp.csr_matrix((vals_, (rows_, cols_)), shape=(nf, nr))
    Kb = (K + sp.diags(diag))[full][:, full]
    Kloc = (Tb.T @ Kb @ Tb).tocsr() + sp.diags(np.r_[np.zeros(nr - 3), head_k[3 * i:3 * i + 3]])
    p = msh["pairs"][i]
    w = p["area"] / p["area"].sum()
    f = np.zeros(nf)
    for j, wt in zip(p["a"], w):
        f[pos[3 * j]] += wt
    u = Tb @ SPDSolver(Kloc.tocsr()).solve(Tb.T @ f)
    c = float(sum(wt * u[pos[3 * j]] for j, wt in zip(p["a"], w)))
    target = d["L_eff"] / (VER1.E_bolt * VER1.As)
    assert abs(c / target - 1) < 1e-6


def test_homogeneity(plain):
    """The contact problem has no initial gap, so the field scales with the load. (The paper's rotation takes
    an arctan of the deflection, so S_paper itself drifts by about 1e-5 between these loads.)"""
    o = analyse(VER1, COARSE, P_full=37000.0, return_data=True)
    body = plain["_data"]["msh"]["body"]
    Ua, Ub = plain["_data"]["U"][body == 0], o["_data"]["U"][body == 0] / 3.7
    assert np.abs(Ua - Ub).max() < 1e-8 * np.abs(Ua).max()
    assert abs(o["gap_DT9_mm"] / 3.7 / plain["gap_DT9_mm"] - 1) < 1e-8
    assert abs(o["S_paper_kNm_per_mrad"] / plain["S_paper_kNm_per_mrad"] - 1) < 1e-4


def test_welds_stiffen_and_are_close_to_their_volume(plain, welded):
    assert welded["S_paper_kNm_per_mrad"] > plain["S_paper_kNm_per_mrad"]
    assert 0.8 < welded["welds"]["volume_over_exact"] < 1.25
    assert plain["welds"]["cells"] == 0


def test_stiffer_bolt_bending_stiffens(plain):
    o = analyse(VER1, COARSE, rot_factor=4.0)
    assert o["S_paper_kNm_per_mrad"] > plain["S_paper_kNm_per_mrad"]


def test_mesh_has_holes_and_heads_clear_of_beam():
    msh = build(VER1, **COARSE, a_flange=4.0, a_web=3.0)
    nodes, hexes, mat = msh["nodes"], msh["hexes"], msh["mat"]
    c = nodes[hexes[mat == 0]].mean(axis=1)
    for r in VER1.rows:
        assert np.hypot(c[:, 1] - r, c[:, 2] - VER1.w / 2).min() >= msh["radii"]["rh"]
