"""Checks of the two-dimensional T-stub used by the exploratory network probe (python -I -m pytest -q here)."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tstub2d as T


def test_element_rigid_modes_and_constant_strain():
    xy = np.array([[0.0, 0.0], [3.0, 0.4], [2.6, 2.2], [-0.3, 1.9]])          # distorted
    D = T.D_plane_strain()
    Ke = T.q4i_stiffness(xy, D)
    w = np.linalg.eigvalsh(Ke)
    assert np.sum(np.abs(w) < 1e-8 * w.max()) == 3 and w.min() > -1e-8 * w.max()
    # constant strain: u = A x; energy must equal 1/2 eps^T D eps * area (patch test on one element)
    A = np.array([[1e-3, 2e-4], [-5e-4, 3e-4]])
    u = (xy @ A.T).ravel()
    eps = np.array([A[0, 0], A[1, 1], A[0, 1] + A[1, 0]])
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert np.isclose(0.5 * u @ Ke @ u, 0.5 * eps @ D @ eps * area, rtol=1e-10)


def test_cantilever_bending_one_layer():
    """Pure bending of a slender rectangular strip, one element through the thickness: exact for this element."""
    L, t, nel = 100.0, 5.0, 10
    xs = np.linspace(0, L, nel + 1)
    nodes = np.array([(x, y) for y in (0.0, t) for x in xs])
    els = np.array([[i, i + 1, nel + 2 + i, nel + 1 + i] for i in range(nel)])
    D = T.D_plane_strain()
    n = 2 * nodes.shape[0]
    K = np.zeros((n, n))
    for e in els:
        dofs = np.ravel([[2 * a, 2 * a + 1] for a in e])
        K[np.ix_(dofs, dofs)] += T.q4i_stiffness(nodes[e], D)
    M = 1000.0                                                              # end moment as a couple of forces
    f = np.zeros(n); f[2 * nel] = M / t; f[2 * (2 * nel + 1)] = -M / t
    fixed = [0, 1, 2 * (nel + 1)]                                           # clamp the root in x, pin in y
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n); u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    Ep = T.E_STEEL / (1 - T.NU ** 2)
    v_beam = M * L ** 2 / (2 * Ep * t ** 3 / 12)
    v = 0.5 * (u[2 * nel + 1] + u[2 * (2 * nel + 1) + 1])
    assert abs(v / v_beam - 1) < 1e-6


def test_signorini_solution_and_energy_identity():
    g = T.TStub2D(t_f=12.0, m=30.0, n=25.0, k_b=1e4)
    o = T.solve(g)
    k = o["kkt"]
    assert k["min_gap"] > -1e-12 and k["min_lambda"] > -1e-10 and k["max_complementarity"] < 1e-10
    assert k["max_residual_unconstrained"] < 1e-9
    K, f, u = o["K"], o["f"], o["u"]
    c = 2 * o["sets"]["contact"] + 1
    act = u[c] <= 1e-12 * np.abs(u).max()
    assert act.any() and (~act).any(), "prying: part of the underside in contact, part separated"
    assert u[2 * o["sets"]["centre"] + 1] > 0                              # the flange lifts under the web
    # Pi(v) - Pi* = 1/2 ||v - u*||_K^2 + lambda*^T v_c for any v with v_x = 0 on the symmetry line
    rng = np.random.default_rng(0)
    v = u + 1e-3 * rng.standard_normal(u.size) * np.abs(u).max()
    v[2 * o["sets"]["symmetry"]] = 0.0
    lam = K @ u - f
    lhs = T.energy(K, f, v) - o["Pi"]
    rhs = 0.5 * (v - u) @ (K @ (v - u)) + lam[c] @ v[c]
    assert np.isclose(lhs, rhs, rtol=1e-9, atol=1e-12 * abs(o["Pi"]))
    # Clapeyron for the cone: Pi* = -1/2 f^T u*
    assert np.isclose(o["Pi"], -0.5 * f @ u, rtol=1e-10)


def test_topology_fixed_across_geometries():
    a = T.mesh(T.TStub2D(8.0, 20.0, 15.0, 3e3)); b = T.mesh(T.TStub2D(20.0, 45.0, 40.0, 3e4))
    assert a[0].shape == b[0].shape and np.array_equal(a[1], b[1])
    for k in ("symmetry", "contact", "load"):
        assert np.array_equal(a[2][k], b[2][k])
    assert a[2]["bolt"] == b[2]["bolt"] and a[2]["centre"] == b[2]["centre"]
