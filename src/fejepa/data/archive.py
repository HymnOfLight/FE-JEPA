"""Instance archives and dataset manifests.

Plan v2.0 mapping:
  - Sec.2.1 / Sec.2.5: the archive format ``(mesh, K, F-battery, [U*])`` is a verified asset
    and is preserved unchanged (node-major dofs: dof ``2*i + c`` is component ``c`` of node ``i``).
  - B1 (provenance): manifests are hashable files; :func:`manifest_sha256` feeds the
    report provenance block.
  - WP5 (data economy): archives are written *unlabelled* by default; labels are added
    later by the runner's labelling stage via :func:`add_labels` + :func:`mark_labelled`,
    so every reference solve is accounted by the solve ledger.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

MANIFEST_NAME = "manifest.json"


@dataclass
class InstanceArchive:
    """One boundary-value problem: mesh, assembled operators, load battery, optional labels."""

    nodes: np.ndarray            # (N, 2) float64
    elements: np.ndarray         # (E, 3) int64, indices into nodes
    K: sp.csr_matrix             # (ndof, ndof), node-major, symmetric
    F: np.ndarray                # (L, ndof) load battery sharing this K
    dirichlet_mask: np.ndarray   # (ndof,) bool, True on constrained dofs
    meta: dict                   # {"material": {...}, "extra": {...}, "loads": [...]}
    U_star: np.ndarray | None = None   # (L, ndof) reference FE solutions, if labelled
    path: Path | None = None

    # -- conveniences -------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_loads(self) -> int:
        return int(self.F.shape[0])

    @property
    def ndof(self) -> int:
        return int(self.F.shape[1])

    @property
    def free_mask(self) -> np.ndarray:
        return ~self.dirichlet_mask

    @property
    def labelled(self) -> bool:
        return self.U_star is not None


def save_instance(arch: InstanceArchive, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        nodes=arch.nodes.astype(np.float64),
        elements=arch.elements.astype(np.int64),
        K_data=arch.K.data,
        K_indices=arch.K.indices,
        K_indptr=arch.K.indptr,
        K_shape=np.asarray(arch.K.shape, dtype=np.int64),
        F=arch.F.astype(np.float64),
        dirichlet_mask=arch.dirichlet_mask.astype(bool),
        meta_json=np.frombuffer(json.dumps(arch.meta).encode("utf-8"), dtype=np.uint8),
    )
    if arch.U_star is not None:
        payload["U_star"] = arch.U_star.astype(np.float64)
    # R9a: atomic write (temp file + os.replace) so a power cut never leaves a
    # truncated archive; numpy is given a file handle so it does not rename.
    import os

    path = str(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **payload)
    os.replace(tmp, path)


def load_instance(path: Path) -> InstanceArchive:
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        K = sp.csr_matrix(
            (d["K_data"], d["K_indices"], d["K_indptr"]),
            shape=tuple(int(s) for s in d["K_shape"]),
        )
        meta = json.loads(bytes(d["meta_json"].tobytes()).decode("utf-8"))
        U = d["U_star"] if "U_star" in d.files else None
        return InstanceArchive(
            nodes=d["nodes"], elements=d["elements"], K=K, F=d["F"],
            dirichlet_mask=d["dirichlet_mask"], meta=meta, U_star=U, path=path,
        )


def add_labels(path: Path, U_star: np.ndarray) -> None:
    """Attach reference solutions to an existing archive (WP5 labelling stage)."""
    arch = load_instance(path)
    arch.U_star = np.asarray(U_star, dtype=np.float64)
    save_instance(arch, path)


# -- manifests ---------------------------------------------------------------

def write_manifest(data_dir: Path, records: list[dict], extra: dict) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(version=2, **extra, n_instances=len(records), instances=records)
    p = data_dir / MANIFEST_NAME
    p.write_text(json.dumps(manifest, indent=1))
    return p


def load_manifest(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / MANIFEST_NAME).read_text())


def instance_files(data_dir: Path) -> list[Path]:
    """Files in manifest order -- the split-determinism contract (audit V4)."""
    m = load_manifest(data_dir)
    return [Path(data_dir) / r["file"] for r in m["instances"]]


def mark_labelled(data_dir: Path, filenames: set[str]) -> None:
    m = load_manifest(data_dir)
    for r in m["instances"]:
        if r["file"] in filenames:
            r["labelled"] = True
    (Path(data_dir) / MANIFEST_NAME).write_text(json.dumps(m, indent=1))


def manifest_sha256(data_dir: Path) -> str:
    return hashlib.sha256((Path(data_dir) / MANIFEST_NAME).read_bytes()).hexdigest()
