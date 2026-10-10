"""Tests for the incompatible-mode hexahedron (run with: python -I -m pytest -q tests)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fejoint import hex8i as H

RNG = np.random.default_rng(7)


def distorted_hex():
    X = H.XI.copy() * np.array([1.3, 0.8, 1.1])
    X += 0.15 * RNG.standard_normal(X.shape)
    return X


def test_rigid_body_modes_and_psd():
    Ke = H.condensed_stiffness(distorted_hex()[None], H.elastic_D(210000.0, 0.3))[0]
    w = np.linalg.eigvalsh(Ke)
    tol = 1e-9 * w.max()
    assert np.sum(np.abs(w) < tol) == 6
    assert w.min() > -tol
    assert np.sum(w > tol) == 18


def test_patch_test_distorted():
    nodes, hexes = H.box_mesh(np.linspace(0, 1, 4), np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    interior = np.all((nodes > 1e-9) & (nodes < 1 - 1e-9), axis=1)
    nodes = nodes.copy()
    nodes[interior] += 0.08 * RNG.standard_normal((interior.sum(), 3))
    E, nu = 1000.0, 0.27
    K = H.assemble(nodes, hexes, E, nu)
    A = np.array([[1e-3, 2e-4, -3e-4], [5e-4, -7e-4, 1e-4], [-2e-4, 3e-4, 4e-4]])
    b = np.array([1e-3, -2e-3, 5e-4])
    u_exact = (nodes @ A.T + b).ravel()
    fixed = np.repeat(~interior, 3)
    free = ~fixed
    u = u_exact.copy()
    u[free] = 0.0
    rhs = -(K[free][:, fixed] @ u_exact[fixed])
    u[free] = spla.spsolve(K[free][:, free].tocsc(), rhs)
    assert np.max(np.abs(u - u_exact)) < 1e-12
    eps = H.gauss_strains(nodes, hexes, u, E, nu)
    S = 0.5 * (A + A.T)
    target = np.array([S[0, 0], S[1, 1], S[2, 2], 2 * S[0, 1], 2 * S[1, 2], 2 * S[2, 0]])
    assert np.max(np.abs(eps - target)) < 1e-12


def test_pure_bending_exact_on_rectangular_elements():
    L, h, b = 10.0, 1.0, 0.5
    E, nu = 1.0, 0.3
    nodes, hexes = H.box_mesh(np.linspace(0, L, 6), np.linspace(-h / 2, h / 2, 2), np.linspace(0, b, 2))
    K = H.assemble(nodes, hexes, E, nu)
    n = nodes.shape[0]
    x, y, z = nodes.T
    # consistent nodal forces for a linear traction sigma_xx = -M y / I on the end face x = L
    M = 1.0; I = b * h ** 3 / 12.0
    f = np.zeros(3 * n)
    end = np.nonzero(np.isclose(x, L))[0]
    # bilinear face: integrate N_a * (-M y / I) over the face, exactly with 2x2 Gauss on each face quad
    ys, zs = np.unique(y[end]), np.unique(z[end])
    gp = np.array([-1, 1]) / np.sqrt(3)
    for j in range(len(ys) - 1):
        for k in range(len(zs) - 1):
            y0, y1, z0, z1 = ys[j], ys[j + 1], zs[k], zs[k + 1]
            for s in gp:
                for t in gp:
                    yy = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * s
                    zz = 0.5 * (z0 + z1) + 0.5 * (z1 - z0) * t
                    w = 0.25 * (y1 - y0) * (z1 - z0)
                    tr = -M * yy / I
                    for (yc, zc, Ny, Nz) in ((y0, z0, (1 - s) / 2, (1 - t) / 2), (y1, z0, (1 + s) / 2, (1 - t) / 2),
                                             (y1, z1, (1 + s) / 2, (1 + t) / 2), (y0, z1, (1 - s) / 2, (1 + t) / 2)):
                        a = end[np.isclose(y[end], yc) & np.isclose(z[end], zc)][0]
                        f[3 * a] += Ny * Nz * tr * w
    # minimal supports: u_x = 0 on x = 0; u_y = 0 at (0, 0?) line; u_z = 0 at one node
    fixed = np.zeros(3 * n, bool)
    left = np.nonzero(np.isclose(x, 0))[0]
    fixed[3 * left] = True
    a0 = left[np.argmin(np.abs(y[left]) + np.abs(z[left]))]
    fixed[3 * left + 1] = np.isclose(y[left], y[a0]) & False
    fixed[3 * a0 + 1] = True
    fixed[3 * a0 + 2] = True
    # the bottom-left nodes are at y = -h/2; also block the y rotation by fixing u_y at a second left node
    a1 = left[np.argmax(z[left] - np.abs(y[left] - y[a0]))]
    fixed[3 * a1 + 1] = True
    free = ~fixed
    u = np.zeros(3 * n)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), f[free])
    # exact solution: u_y(x) = M x^2 / (2 E I) on the neutral axis (y = 0 not a node; use the mean of the faces)
    tip = end
    slope_exact = M * L / (E * I)            # rotation at x = L
    ux_top = u[3 * tip[np.isclose(y[tip], h / 2)]].mean()
    ux_bot = u[3 * tip[np.isclose(y[tip], -h / 2)]].mean()
    rot = (ux_bot - ux_top) / h
    assert abs(rot / slope_exact - 1) < 1e-8


def test_identity_holds_with_condensed_stiffness():
    nodes, hexes = H.box_mesh(np.linspace(0, 3, 5), np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    nodes = nodes.copy(); nodes[:, 1] += 0.1 * np.sin(nodes[:, 0])
    K = H.assemble(nodes, hexes, 1.0, 0.3)
    fixed = np.repeat(np.isclose(nodes[:, 0], 0), 3)
    free = ~fixed
    A = K[free][:, free].tocsc()
    f = RNG.standard_normal(A.shape[0])
    ustar = spla.spsolve(A, f)
    np.linalg.cholesky(A.toarray())  # SPD after removing rigid-body modes
    Pi = lambda v: 0.5 * v @ (A @ v) - f @ v  # noqa: E731
    r = A @ ustar - f                          # solver residual, kept explicitly
    for _ in range(20):
        v = ustar + RNG.standard_normal(ustar.size) * 10 ** RNG.uniform(-3, 1)
        lhs = Pi(v) - Pi(ustar)
        rhs = 0.5 * (v - ustar) @ (A @ (v - ustar)) + r @ (v - ustar)
        assert abs(lhs - rhs) <= 1e-12 * (abs(Pi(v)) + abs(Pi(ustar)))
