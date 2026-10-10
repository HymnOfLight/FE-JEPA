"""structured_tet_mesh must tile its box exactly and conformingly; P1 energy of a linear field must be exact."""
import numpy as np

from fejepa.fe.tet3d import structured_tet_mesh, assemble_tet, _elastic_D


def _vols(nodes, tets):
    p = nodes[tets]
    return np.abs(np.linalg.det(p[:, 1:] - p[:, :1])) / 6.0


def test_tiles_the_box_exactly_and_conformingly():
    w, h, d, nx, ny, nz = 2.0, 1.3, 0.9, 3, 2, 2
    nodes, tets = structured_tet_mesh(w, h, d, nx, ny, nz)
    vol = _vols(nodes, tets)
    assert vol.min() > 1e-12 * w * h * d
    assert abs(vol.sum() - w * h * d) <= 1e-12 * w * h * d
    tri = np.sort(np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    _, counts = np.unique(tri, axis=0, return_counts=True)
    assert set(np.unique(counts)) <= {1, 2}                       # no triangle in three tets
    assert (counts == 1).sum() == 4 * (ny * nz + nx * nz + nx * ny)  # boundary triangles only


def test_random_points_covered_once():
    rng = np.random.default_rng(0)
    w, h, d = 1.7, 1.1, 0.8
    nodes, tets = structured_tet_mesh(w, h, d, 3, 2, 2)
    P = rng.uniform([0, 0, 0], [w, h, d], size=(4000, 3))
    p = nodes[tets]
    T = np.transpose(p[:, 1:] - p[:, :1], (0, 2, 1))              # columns: edges from vertex 0
    lam = np.einsum("eij,nej->nei", np.linalg.inv(T), P[:, None, :] - p[None, :, 0, :])
    inside = (lam >= -1e-12).all(-1) & (lam.sum(-1) <= 1 + 1e-12)
    assert (inside.sum(1) == 1).all()                              # once (a.s. on no shared face)


def test_p1_energy_of_a_linear_field_is_exact():
    w, h, d = 2.0, 1.3, 0.9
    nodes, tets = structured_tet_mesh(w, h, d, 3, 2, 2)
    material = {"E": 1.0, "nu": 0.3, "plane": "3d"}
    K = assemble_tet(nodes, tets, material)
    G = np.array([[1e-3, 2e-4, -3e-4], [5e-4, -7e-4, 1e-4], [-2e-4, 3e-4, 4e-4]])
    u = (nodes @ G.T).ravel()
    S = 0.5 * (G + G.T)
    eps = np.array([S[0, 0], S[1, 1], S[2, 2], 2 * S[0, 1], 2 * S[1, 2], 2 * S[2, 0]])
    exact = 0.5 * eps @ _elastic_D(material) @ eps * w * h * d
    assert abs(0.5 * u @ (K @ u) / exact - 1) < 1e-12
