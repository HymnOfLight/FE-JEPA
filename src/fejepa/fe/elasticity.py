"""2D linear-elasticity finite-element assembly.

This module assembles the discrete system ``K U = F`` for plane linear
elasticity on an unstructured triangular mesh using ``scikit-fem``.  It exposes
the objects the FE-JEPA energy anchor needs:

* the global stiffness matrix ``K`` (sparse, symmetric, positive definite after
  Dirichlet elimination -- see Lemma 1 of the proposal),
* one right-hand side ``F`` per load case in a *battery* that shares ``K``,
* a boolean mask of the constrained (Dirichlet) degrees of freedom, and
* the reference solution ``U*`` for the labelled evaluation split.

All vectors/matrices are returned in **node-major** ordering, i.e. degree of
freedom ``2 * i + c`` is component ``c in {0, 1}`` of node ``i``.  This makes the
mapping between a decoder that emits an ``(n_nodes, 2)`` displacement field and
the assembled operators a trivial reshape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
import skfem
from skfem import Basis, ElementTriP1, ElementVector, FacetBasis, MeshTri, condense, solve
from skfem.models.elasticity import linear_elasticity

FacetSelector = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Material:
    """Isotropic linear-elastic material."""

    E: float = 1.0
    nu: float = 0.3
    plane: str = "stress"  # "stress" or "strain"

    def lame(self) -> tuple[float, float]:
        return lame_from_E_nu(self.E, self.nu, self.plane)


@dataclass
class LoadCase:
    """A single right-hand side: a body force plus boundary tractions.

    ``tractions`` is a list of ``(selector, (tx, ty))`` pairs where ``selector``
    maps facet midpoint coordinates ``x`` of shape ``(2, n_facets)`` to a boolean
    mask selecting the facets that carry the constant traction ``(tx, ty)``.
    """

    name: str = "load"
    body: tuple[float, float] = (0.0, 0.0)
    tractions: list[tuple[FacetSelector, tuple[float, float]]] = field(default_factory=list)


@dataclass
class FEProblem:
    """An assembled FE instance (geometry + material + load battery).

    Attributes are stored in node-major ordering.  ``U_star`` may be ``None`` for
    unlabelled instances where the solve is intentionally skipped.
    """

    nodes: np.ndarray  # (n_nodes, 2)
    elements: np.ndarray  # (n_elems, 3) int
    K: sp.csr_matrix  # (ndof, ndof), ndof = 2 * n_nodes
    F: np.ndarray  # (n_loads, ndof)
    dirichlet_mask: np.ndarray  # (ndof,) bool, True == constrained
    material: Material
    load_names: list[str]
    U_star: np.ndarray | None = None  # (n_loads, ndof) or None
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_dof(self) -> int:
        return self.K.shape[0]

    @property
    def n_loads(self) -> int:
        return self.F.shape[0]

    @property
    def free_mask(self) -> np.ndarray:
        return ~self.dirichlet_mask


def lame_from_E_nu(E: float, nu: float, plane: str = "stress") -> tuple[float, float]:
    """Return the Lame parameters ``(lambda, mu)``.

    For ``plane == "stress"`` the in-plane effective ``lambda`` is used so that the
    2D constitutive law matches plane-stress elasticity.
    """

    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    if plane == "stress":
        lam = 2.0 * lam * mu / (lam + 2.0 * mu)
    elif plane != "strain":
        raise ValueError(f"plane must be 'stress' or 'strain', got {plane!r}")
    return lam, mu


def _node_major_permutation(basis: Basis) -> np.ndarray:
    """Permutation mapping node-major dof index -> skfem global dof index.

    ``perm[2 * i + c]`` is the global dof of component ``c`` at node ``i``.
    """

    nodal = basis.nodal_dofs  # (2, n_nodes)
    if nodal.shape[0] != 2:
        raise ValueError("expected a 2-component vector basis")
    n_nodes = nodal.shape[1]
    perm = np.empty(2 * n_nodes, dtype=np.int64)
    perm[0::2] = nodal[0]
    perm[1::2] = nodal[1]
    return perm


def _assemble_load(basis: Basis, mesh: MeshTri, element, load: LoadCase) -> np.ndarray:
    ndof = basis.N
    f = np.zeros(ndof)

    bx, by = load.body
    if bx != 0.0 or by != 0.0:

        @skfem.LinearForm
        def body_form(v, w):
            return bx * v[0] + by * v[1]

        f = f + body_form.assemble(basis)

    for selector, (tx, ty) in load.tractions:
        facets = mesh.facets_satisfying(selector, boundaries_only=True)
        if facets.size == 0:
            continue
        fbasis = FacetBasis(mesh, element, facets=facets)

        @skfem.LinearForm
        def traction_form(v, w):
            return tx * v[0] + ty * v[1]

        f = f + traction_form.assemble(fbasis)

    return f


def assemble_problem(
    mesh: MeshTri,
    material: Material,
    dirichlet_selector: FacetSelector,
    load_cases: Sequence[LoadCase],
    solve_reference: bool = True,
    meta: dict | None = None,
) -> FEProblem:
    """Assemble ``K`` once and one ``F`` per load case (node-major ordering).

    Parameters
    ----------
    mesh:
        A ``skfem.MeshTri`` instance.
    material:
        Isotropic linear-elastic :class:`Material`.
    dirichlet_selector:
        Maps facet midpoint coordinates ``x`` (shape ``(2, n_facets)``) to a
        boolean mask selecting clamped (homogeneous Dirichlet) boundary facets.
    load_cases:
        The load battery; all share the same ``K``.
    solve_reference:
        If ``True`` the reference FE solution ``U*`` is computed (labelled split).
        Set ``False`` for unlabelled pretraining instances.
    """

    if len(load_cases) == 0:
        raise ValueError("need at least one load case")

    element = ElementVector(ElementTriP1())
    basis = Basis(mesh, element)
    perm = _node_major_permutation(basis)

    lam, mu = material.lame()
    K_global = linear_elasticity(lam, mu).assemble(basis).tocsr()

    F_global = np.stack(
        [_assemble_load(basis, mesh, element, lc) for lc in load_cases], axis=0
    )

    dirichlet_dofs = basis.get_dofs(dirichlet_selector).all()

    # Reorder to node-major.
    K = K_global[perm][:, perm].tocsr()
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.size)
    F = F_global[:, perm]

    dirichlet_mask = np.zeros(K.shape[0], dtype=bool)
    dirichlet_mask[inv_perm[dirichlet_dofs]] = True

    U_star = None
    if solve_reference:
        U_rows = []
        for j in range(F_global.shape[0]):
            u = solve(*condense(K_global, F_global[j], D=dirichlet_dofs))
            U_rows.append(u[perm])
        U_star = np.stack(U_rows, axis=0)

    nodes = mesh.p.T.copy()
    elements = mesh.t.T.copy()

    return FEProblem(
        nodes=nodes,
        elements=elements,
        K=K,
        F=F,
        dirichlet_mask=dirichlet_mask,
        material=material,
        load_names=[lc.name for lc in load_cases],
        U_star=U_star,
        meta=meta or {},
    )
