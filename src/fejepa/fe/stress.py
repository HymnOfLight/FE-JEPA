r"""Stress / von Mises recovery for 2D linear-elasticity (constant-strain triangles).

``ElementTriP1`` gives a constant strain per element (CST), so the strain and
stress are piecewise constant.  For a triangle with engineering strain
:math:`(\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy}) = B\,u_e`, the stress is
recovered with the **same** Lame parameters used to assemble ``K`` (so that the
recovered strain energy matches :math:`\tfrac12 u^\top K u` exactly):

.. math::

    \sigma_{xx} = \lambda(\varepsilon_{xx}+\varepsilon_{yy}) + 2\mu\varepsilon_{xx},
    \quad
    \sigma_{yy} = \lambda(\varepsilon_{xx}+\varepsilon_{yy}) + 2\mu\varepsilon_{yy},
    \quad
    \sigma_{xy} = \mu\,\gamma_{xy}.

The von Mises stress uses the out-of-plane component appropriate to the plane
assumption (``sigma_zz = 0`` for plane stress; ``nu(sigma_xx+sigma_yy)`` for plane
strain).
"""

from __future__ import annotations

import numpy as np

from fejepa.fe.elasticity import Material


def element_areas_and_B(nodes: np.ndarray, elements: np.ndarray):
    """Return per-element area ``(Ne,)`` and CST strain-displacement ``B`` ``(Ne,3,6)``.

    The element dof ordering of ``B`` is node-major ``[u0x,u0y,u1x,u1y,u2x,u2y]``.
    """

    p = nodes[elements]  # (Ne, 3, 2)
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]

    det = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    area = 0.5 * np.abs(det)
    inv2A = 1.0 / det  # signed; B uses 1/(2A_signed) consistently

    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1)  # (Ne,3)
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1)  # (Ne,3)

    Ne = elements.shape[0]
    B = np.zeros((Ne, 3, 6))
    for i in range(3):
        B[:, 0, 2 * i] = b[:, i] * inv2A
        B[:, 1, 2 * i + 1] = c[:, i] * inv2A
        B[:, 2, 2 * i] = c[:, i] * inv2A
        B[:, 2, 2 * i + 1] = b[:, i] * inv2A
    return area, B


def element_strain_stress(
    nodes: np.ndarray, elements: np.ndarray, u_nodal: np.ndarray, material: Material
):
    """Return ``(strain(Ne,3), stress(Ne,3), von_mises(Ne,))`` for displacement ``u``.

    ``u_nodal`` is the node-major displacement vector ``(2*n_nodes,)``.
    """

    lam, mu = material.lame()
    _, B = element_areas_and_B(nodes, elements)

    u = u_nodal.reshape(-1)
    # gather element dof vectors (Ne, 6) in node-major order
    dof = np.empty((elements.shape[0], 6), dtype=np.int64)
    dof[:, 0::2] = 2 * elements
    dof[:, 1::2] = 2 * elements + 1
    ue = u[dof]  # (Ne, 6)

    strain = np.einsum("eij,ej->ei", B, ue)  # (Ne,3): [exx, eyy, gxy]
    exx, eyy, gxy = strain[:, 0], strain[:, 1], strain[:, 2]
    tr = exx + eyy
    sxx = lam * tr + 2 * mu * exx
    syy = lam * tr + 2 * mu * eyy
    sxy = mu * gxy
    stress = np.stack([sxx, syy, sxy], axis=1)

    if material.plane == "stress":
        szz = np.zeros_like(sxx)
    else:  # plane strain
        szz = material.nu * (sxx + syy)
    vm = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3.0 * sxy**2
    )
    return strain, stress, vm


def strain_energy(
    nodes: np.ndarray, elements: np.ndarray, u_nodal: np.ndarray, material: Material
) -> float:
    r"""Recovered strain energy :math:`\tfrac12\int \sigma:\varepsilon\,dx`.

    Equals :math:`\tfrac12 u^\top K u` for the assembled ``K`` (consistency check).
    """

    area, _ = element_areas_and_B(nodes, elements)
    strain, stress, _ = element_strain_stress(nodes, elements, u_nodal, material)
    # density = 1/2 (sxx exx + syy eyy + sxy gxy)  [engineering strain]
    dens = 0.5 * np.einsum("ei,ei->e", stress, strain)
    return float(np.sum(dens * area))
