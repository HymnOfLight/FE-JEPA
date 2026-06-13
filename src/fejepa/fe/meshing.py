"""Mesh construction helpers (gmsh-backed, with a pure-skfem fallback)."""

from __future__ import annotations

import numpy as np
from skfem import MeshTri


def meshtri_from_gmsh_current_model() -> MeshTri:
    """Build a :class:`skfem.MeshTri` from the currently active gmsh model.

    The caller is responsible for initialising gmsh, building geometry, and
    calling ``gmsh.model.mesh.generate(2)`` before invoking this function.
    """

    import gmsh

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]
    node_tags = np.asarray(node_tags, dtype=np.int64)

    # Remap arbitrary gmsh node tags to a contiguous 0..N-1 index space.
    tag_to_idx = np.zeros(node_tags.max() + 1, dtype=np.int64)
    tag_to_idx[node_tags] = np.arange(node_tags.size)

    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    tris = None
    for etype, enodes in zip(elem_types, elem_node_tags):
        if etype == 2:  # 3-node triangle
            tris = np.asarray(enodes, dtype=np.int64).reshape(-1, 3)
            break
    if tris is None or tris.size == 0:
        raise RuntimeError("gmsh model contains no triangular elements")

    t = tag_to_idx[tris].T  # (3, n_elems)
    p = coords.T  # (2, n_nodes)
    return MeshTri(p, t)


def plate_with_holes_mesh(
    width: float,
    height: float,
    holes: list[tuple[float, float, float]],
    mesh_size: float,
    verbose: bool = False,
) -> MeshTri:
    """Mesh a rectangular plate ``[0, width] x [0, height]`` with circular holes.

    ``holes`` is a list of ``(cx, cy, r)`` tuples.  Holes that are fully inside
    the plate are subtracted from the domain.
    """

    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("plate")
        occ = gmsh.model.occ

        rect = occ.addRectangle(0.0, 0.0, 0.0, width, height)
        domain = [(2, rect)]
        if holes:
            tools = [(2, occ.addDisk(cx, cy, 0.0, r, r)) for (cx, cy, r) in holes]
            out, _ = occ.cut(domain, tools)
            domain = out
        occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.model.mesh.generate(2)
        return meshtri_from_gmsh_current_model()
    finally:
        gmsh.finalize()
