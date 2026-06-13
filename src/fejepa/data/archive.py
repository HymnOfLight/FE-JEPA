"""Per-instance archive format: one compressed ``.npz`` per FE instance.

A dataset directory contains ``instance_00000.npz`` files plus a
``manifest.json`` describing the split.  Each archive stores the geometry, the
node-major sparse stiffness ``K`` (CSR triplet), the load battery ``F``, the
Dirichlet mask, optional reference solution ``U*``, and metadata.  The expensive
``U*`` solve is the *only* solve in the project's training economy and is stored
only for labelled instances.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fejepa.fe.elasticity import FEProblem, Material


@dataclass
class InstanceArchive:
    """Lightweight in-memory view of an instance archive."""

    nodes: np.ndarray
    elements: np.ndarray
    K: sp.csr_matrix
    F: np.ndarray
    dirichlet_mask: np.ndarray
    U_star: np.ndarray | None
    material: Material
    load_names: list[str]
    meta: dict

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
    def labelled(self) -> bool:
        return self.U_star is not None

    @property
    def free_mask(self) -> np.ndarray:
        return ~self.dirichlet_mask

    def to_problem(self) -> FEProblem:
        return FEProblem(
            nodes=self.nodes,
            elements=self.elements,
            K=self.K,
            F=self.F,
            dirichlet_mask=self.dirichlet_mask,
            material=self.material,
            load_names=list(self.load_names),
            U_star=self.U_star,
            meta=dict(self.meta),
        )


def save_problem(path: str | os.PathLike, problem: FEProblem) -> Path:
    """Persist an :class:`FEProblem` to a compressed ``.npz`` archive."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    K = problem.K.tocsr()

    payload: dict[str, np.ndarray] = {
        "nodes": problem.nodes.astype(np.float64),
        "elements": problem.elements.astype(np.int64),
        "K_data": K.data.astype(np.float64),
        "K_indices": K.indices.astype(np.int64),
        "K_indptr": K.indptr.astype(np.int64),
        "K_shape": np.asarray(K.shape, dtype=np.int64),
        "F": problem.F.astype(np.float64),
        "dirichlet_mask": problem.dirichlet_mask.astype(np.bool_),
    }
    if problem.U_star is not None:
        payload["U_star"] = problem.U_star.astype(np.float64)

    meta = {
        "material": {
            "E": problem.material.E,
            "nu": problem.material.nu,
            "plane": problem.material.plane,
        },
        "load_names": list(problem.load_names),
        "extra": problem.meta,
    }
    payload["meta_json"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )

    np.savez_compressed(path, **payload)
    return path


def load_problem(path: str | os.PathLike) -> InstanceArchive:
    """Load an :class:`InstanceArchive` from a ``.npz`` archive."""

    with np.load(path, allow_pickle=False) as data:
        K = sp.csr_matrix(
            (data["K_data"], data["K_indices"], data["K_indptr"]),
            shape=tuple(int(s) for s in data["K_shape"]),
        )
        meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
        material = Material(**meta["material"])
        U_star = data["U_star"] if "U_star" in data.files else None
        return InstanceArchive(
            nodes=data["nodes"],
            elements=data["elements"],
            K=K,
            F=data["F"],
            dirichlet_mask=data["dirichlet_mask"],
            U_star=U_star,
            material=material,
            load_names=list(meta["load_names"]),
            meta=meta["extra"],
        )


def write_manifest(directory: str | os.PathLike, manifest: dict) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def read_manifest(directory: str | os.PathLike) -> dict:
    return json.loads((Path(directory) / "manifest.json").read_text())
