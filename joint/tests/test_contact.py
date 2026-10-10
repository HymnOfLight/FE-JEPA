"""Contact solver and the energy statements under frictionless contact (see docs/contact_energy_note.md).

Test problem: one flange of a T-stub (an end plate strip) on a rigid flat support, two bolt springs,
pulled up at the centre; incompatible-mode hexahedra; linear elasticity; no friction, no preload,
zero initial gap, so the feasible set is a closed convex cone."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
import pytest
from fejoint import hex8i as H
from fejoint.contact import solve_signorini, solve_tied

RNG = np.random.default_rng(20261009)


@pytest.fixture(scope="module")
def tstub():
    L, t, d = 120.0, 10.0, 30.0
    nodes, hexes = H.box_mesh(np.linspace(0, L, 49), np.linspace(0, t, 3), np.linspace(0, d, 13))
    K = H.assemble(nodes, hexes, 210000.0, 0.3)
    x, y, z = nodes.T
    n = nodes.shape[0]
    top = np.nonzero(np.isclose(y, t))[0]; bottom = np.nonzero(np.isclose(y, 0))[0]
    springs = np.zeros(3 * n)
    for xb in (L / 2 - 35, L / 2 + 35):
        patch = top[np.hypot(x[top] - xb, z[top] - 15) <= 7.5]
        springs[3 * patch + 1] += 210000.0 * 157.0 / 45.0 / patch.size
    K = (K + sp.diags(springs)).tocsr()
    strip = top[np.abs(x[top] - L / 2) <= 5.0 + 1e-9]
    f = np.zeros(3 * n); f[3 * strip + 1] = 1000.0 / strip.size
    fixed = np.zeros(3 * n, bool)
    fixed[3 * np.nonzero(np.isclose(x, L / 2))[0]] = True
    fixed[3 * np.nonzero(np.isclose(z, 0))[0] + 2] = True
    free = np.nonzero(~fixed)[0]
    A = K[free][:, free].tocsr(); b = f[free]
    pos = -np.ones(3 * n, int); pos[free] = np.arange(free.size)
    cidx = pos[3 * bottom + 1]
    sol = solve_signorini(A, b, cidx)
    return A, b, cidx, sol


def Pi(A, b, v):
    return 0.5 * v @ (A @ v) - b @ v


def feasible_samples(u, cidx, act, n=200):
    out = []
    for k in range(n):
        s = np.abs(u).max() * 10.0 ** RNG.uniform(-3, 0.5)
        kind = k % 4
        if kind == 0:
            v = u + s * RNG.standard_normal(u.size)
        elif kind == 1:
            v = RNG.uniform(0.2, 1.8) * u + 0.1 * s * RNG.standard_normal(u.size)
        elif kind == 2:
            v = u.copy(); lift = np.zeros(cidx.size); lift[act] = s * RNG.uniform(0, 1, act.sum()); v[cidx] += lift
        else:
            v = s * RNG.standard_normal(u.size)
        v[cidx] = np.maximum(v[cidx], 0.0)
        out.append(v)
    return out


def test_kkt_and_partial_contact(tstub):
    A, b, cidx, sol = tstub
    k = sol["kkt"]
    assert k["min_gap"] > -1e-12 and k["min_lambda"] > -1e-10
    assert k["max_complementarity"] < 1e-10 and k["max_residual_unconstrained"] < 1e-10
    assert k["max_lambda_inactive"] < 1e-10
    assert 0 < sol["active"].sum() < cidx.size          # prying: part of the flange bears, part lifts off


def test_homogeneity_and_solution_energy(tstub):
    A, b, cidx, sol = tstub
    u = sol["x"]
    sol3 = solve_signorini(A, 3.0 * b, cidx)
    assert np.array_equal(sol3["active"], sol["active"])
    assert np.linalg.norm(sol3["x"] - 3 * u) <= 1e-10 * np.linalg.norm(3 * u)
    assert abs(sol["lam"] @ u) <= 1e-10 * abs(b @ u)
    assert abs(Pi(A, b, u) / (-0.5 * u @ (A @ u)) - 1) < 1e-10


def test_decomposition_bound_floor_and_rescaling(tstub):
    A, b, cidx, sol = tstub
    u, lam = sol["x"], sol["lam"]
    Pu, nu_ = Pi(A, b, u), np.sqrt(u @ (A @ u))
    for v in feasible_samples(u, cidx, sol["active"]):
        gap = Pi(A, b, v) - Pu; e2 = (v - u) @ (A @ (v - u))
        assert abs(gap - 0.5 * e2 - lam @ v) <= 1e-9 * max(abs(gap), abs(Pu))
        assert gap - 0.5 * e2 >= -1e-9 * abs(Pu)
        if Pi(A, b, v) <= 0:
            assert np.sqrt(e2) <= nu_ * (1 + 1e-9)
        tp = max(0.0, (b @ v) / (v @ (A @ v)))
        assert Pi(A, b, tp * v) <= 1e-12 * abs(Pu)


def test_floor_fail_direction_and_ranking_no_longer_exact(tstub):
    A, b, cidx, sol = tstub
    u, lam = sol["x"], sol["lam"]
    Pu, nu_ = Pi(A, b, u), np.sqrt(u @ (A @ u))
    from fejoint.linsolve import SPDSolver
    w = SPDSolver(A).solve(lam)
    q = np.sqrt(lam @ w)
    bound = np.sqrt(q ** 2 + nu_ ** 2) - q
    flagged = None
    for alpha in np.linspace(0, 4, 4001)[1:]:
        v = u + alpha * (nu_ / q) * w; v[cidx] = np.maximum(v[cidx], 0)
        if Pi(A, b, v) > 0:
            flagged = v; break
    assert flagged is not None
    err = np.sqrt((flagged - u) @ (A @ (flagged - u)))
    assert err < nu_                       # flagged, yet better than zero displacement
    assert err >= bound * (1 - 1e-3)       # and no better than the bound allows
    V = feasible_samples(u, cidx, sol["active"], n=150)
    vals = [(Pi(A, b, v), np.sqrt((v - u) @ (A @ (v - u)))) for v in V]
    flips = sum(1 for i in range(len(V)) for j in range(i + 1, len(V))
                if (vals[i][0] - vals[j][0]) * (vals[i][1] - vals[j][1]) < 0)
    assert flips > 0                       # energy order is no longer the error order


def test_tied_is_stiffer_and_identity_exact(tstub):
    A, b, cidx, sol = tstub
    tied = solve_tied(A, b, cidx)
    ut, uc = tied["x"], sol["x"]
    assert b @ ut < b @ uc                 # compliance of the bonded model is smaller
    for _ in range(30):
        v = ut + RNG.standard_normal(ut.size) * np.abs(ut).max(); v[cidx] = 0.0
        lhs = Pi(A, b, v) - Pi(A, b, ut); rhs = 0.5 * (v - ut) @ (A @ (v - ut))
        assert abs(lhs - rhs) <= 1e-9 * max(abs(lhs), abs(Pi(A, b, ut)))
