"""Plane-stress/strain linear elasticity assembly (scikit-fem backend).

Plan v2.0 mapping: Sec.2.1 / Sec.2.5 verified asset. The math is byte-for-byte the
audited v1 pipeline: skfem ``ElementVector(ElementTriP1)`` assembly, reordered to the
node-major convention (dof ``2*i + c``), a load *battery* sharing one K (only F changes),
homogeneous Dirichlet on a selector. Reference solves are NOT performed here -- the
runner's labelling stage owns them so the solve ledger stays truthful (plan WP5).

Requires the ``gen`` extra (scikit-fem). Import is deferred so the rest of the package
works without it (synthetic backend, analysis-only environments).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .stress import lame

LOAD_NAMES = ["tip_down", "tip_axial", "top_shear", "gravity"]


def assemble_plate(nodes: np.ndarray, elements: np.ndarray, material: dict,
                   width: float, height: float,
                   traction_scales: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble (K, F-battery, dirichlet_mask) in node-major ordering.

    traction_scales: (4,) magnitudes for [tip_down, tip_axial, top_shear, gravity].
    Left edge (x=0) is clamped in both components.
    """
    from skfem import Basis, ElementTriP1, ElementVector, FacetBasis, LinearForm, MeshTri, asm
    from skfem.models.elasticity import linear_elasticity

    mesh = MeshTri(nodes.T.copy(), elements.T.copy())
    element = ElementVector(ElementTriP1())
    basis = Basis(mesh, element)

    lam, mu = lame(material["E"], material["nu"], material["plane"])
    K_g = asm(linear_elasticity(lam, mu), basis).tocsr()

    def right(x):
        return np.isclose(x[0], width)

    def top(x):
        return np.isclose(x[1], height)

    def left(x):
        return np.isclose(x[0], 0.0)

    def traction(fb, t):
        return asm(LinearForm(lambda v, w: t[0] * v[0] + t[1] * v[1]), fb)

    fb_right = FacetBasis(mesh, element, facets=mesh.facets_satisfying(right))
    fb_top = FacetBasis(mesh, element, facets=mesh.facets_satisfying(top))
    ts = np.asarray(traction_scales, dtype=np.float64)
    F_g = np.stack([
        traction(fb_right, (0.0, -ts[0])),
        traction(fb_right, (ts[1], 0.0)),
        traction(fb_top, (ts[2], 0.0)),
        asm(LinearForm(lambda v, w: 0.0 * v[0] - ts[3] * v[1]), basis),
    ])

    dirichlet_dofs = basis.get_dofs(facets=mesh.facets_satisfying(left)).all()

    # node-major permutation: perm[2*i + c] = global dof of component c at node i
    perm = basis.nodal_dofs.T.reshape(-1)
    K = K_g[perm][:, perm].tocsr()
    F = F_g[:, perm]
    mask_g = np.zeros(K_g.shape[0], dtype=bool)
    mask_g[np.asarray(dirichlet_dofs, dtype=np.int64)] = True
    dirichlet_mask = mask_g[perm]
    return K, F, dirichlet_mask
