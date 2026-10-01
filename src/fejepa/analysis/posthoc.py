"""wp8-lejepa Stage 1.36: post-hoc readings of trained states (no training).

Measurements that interpret the E-series and Phase-2b results from the
states they already produced. None of them feeds a pre-registered verdict;
every one is reported as a post-hoc reading.

* Energy-optimal amplitude. For a prediction u of one load case, the scalar

      c* = F^T u / (u^T K u)                        (free dofs only)

  minimises Pi_h(c u) over c. Since Pi_h(c u) - Pi_h(U*) = 0.5 ||c u - U*||_K^2
  (Lemma 1), c* u is the energy-norm projection of U* onto span{u}; c = 1 is
  admissible, so gap(c* u) <= gap(u). It needs no label: one mat-vec. A
  prediction that is right in shape but off in amplitude by a factor a has
  c* = 1/a exactly.
* Error anatomy (3D tetrahedra). The error field e = u - U* has an exact
  element decomposition of its energy, 0.5 e^T K e = sum_e 0.5 sigma(e):eps(e)
  vol_e, so the share of the error energy carried by any set of elements
  (near cavities, near the loaded nodes, near token-cell boundaries) is well
  defined and non-negative; it is compared with the same set's share of the
  reference solution's energy (an enrichment ratio of 1 means the error is
  no more concentrated there than the solution's own energy).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# ------------------------------------------------------------ amplitude --

def free_mask(arch) -> np.ndarray:
    """(ndof,) True on the free (non-Dirichlet) degrees of freedom."""
    return np.asarray(arch.free_mask, dtype=bool).ravel()


def amplitude_factor(U: np.ndarray, K, F: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Per-load c* = F^T u / (u^T K u) over the free dofs; NaN where u^T K u = 0."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))[:, free]
    F2 = np.atleast_2d(np.asarray(F, dtype=np.float64))[:, free]
    Kf = sp.csr_matrix(K)[free][:, free]
    KU = (Kf @ U2.T).T
    num = np.einsum("ld,ld->l", F2, U2)
    den = np.einsum("ld,ld->l", U2, KU)
    out = np.full(U2.shape[0], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def l2_amplitude(U: np.ndarray, U_star: np.ndarray) -> np.ndarray:
    """Per-load label-based reference: argmin_c ||c u - U*||_2 = <u, U*> / ||u||^2."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))
    S2 = np.atleast_2d(np.asarray(U_star, dtype=np.float64))
    den = np.einsum("ld,ld->l", U2, U2)
    out = np.full(U2.shape[0], np.nan)
    ok = den > 0
    out[ok] = np.einsum("ld,ld->l", U2, S2)[ok] / den[ok]
    return out


def energy_norms(U: np.ndarray, K, free: np.ndarray) -> np.ndarray:
    """Per-load energy norm ||u||_K over the free dofs."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))[:, free]
    Kf = sp.csr_matrix(K)[free][:, free]
    return np.sqrt(np.maximum(np.einsum("ld,ld->l", U2, (Kf @ U2.T).T), 0.0))


def battery_amplitude(U: np.ndarray, K, F: np.ndarray, free: np.ndarray) -> float:
    """One energy-optimal factor for the whole load battery,
    c_b = sum_l F_l^T u_l / sum_l u_l^T K u_l -- the level at which `fscale`
    acts (it is a battery-level scale)."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))[:, free]
    F2 = np.atleast_2d(np.asarray(F, dtype=np.float64))[:, free]
    Kf = sp.csr_matrix(K)[free][:, free]
    den = float(np.einsum("ld,ld->", U2, (Kf @ U2.T).T))
    return float(np.einsum("ld,ld->", F2, U2) / den) if den > 0 else float("nan")


def apply_amplitude(U: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scale each load case by its factor; a NaN factor leaves the case unchanged."""
    c = np.where(np.isfinite(c), c, 1.0)
    return np.atleast_2d(U) * c[:, None]


# ---------------------------------------------------------- 3D anatomy --

def element_energy(nodes: np.ndarray, tets: np.ndarray, u: np.ndarray,
                   material: dict) -> np.ndarray:
    """(E,) element strain energies 0.5 sigma:eps vol of a node-major P1 field;
    they sum to 0.5 u^T K u for K = assemble_tet(nodes, tets, material)."""
    from ..fe.tet3d import _tet_geometry, tet_strains, tet_stresses

    vol, _ = _tet_geometry(nodes, tets)
    eps = tet_strains(nodes, tets, u)
    sig = tet_stresses(nodes, tets, u, material)
    return 0.5 * np.einsum("ei,ei->e", sig, eps) * vol


def cavity_distance(points: np.ndarray, holes) -> np.ndarray:
    """Distance from each point to the nearest spherical cavity surface
    (holes as (x, y, z, r)); +inf when there is no cavity."""
    pts = np.asarray(points, dtype=np.float64)
    if not holes:
        return np.full(pts.shape[0], np.inf)
    h = np.asarray(holes, dtype=np.float64).reshape(-1, 4)
    d = np.linalg.norm(pts[:, None, :] - h[None, :, :3], axis=2) - h[None, :, 3]
    return np.abs(d).min(axis=1)


def low_order_fraction(nodes: np.ndarray, e: np.ndarray, degree: int = 2) -> float:
    """Share of ||e||_2^2 captured by the best (least-squares) polynomial field
    of total degree <= `degree` in the node coordinates, fitted per component:
    1 for a globally smooth error, near 0 for a local or oscillatory one."""
    x = np.asarray(nodes, dtype=np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-30)
    d = x.shape[1]
    cols = [np.ones(len(x))]
    if degree >= 1:
        cols += [x[:, i] for i in range(d)]
    if degree >= 2:
        cols += [x[:, i] * x[:, j] for i in range(d) for j in range(i, d)]
    A = np.stack(cols, axis=1)
    E = np.asarray(e, dtype=np.float64).reshape(len(x), -1)
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    tot = float((E ** 2).sum())
    return float(((A @ coef) ** 2).sum() / tot) if tot > 0 else float("nan")


def region_share(weights: np.ndarray, mask: np.ndarray) -> float:
    """Share of a non-negative per-element quantity carried by the masked elements."""
    w = np.asarray(weights, dtype=np.float64)
    tot = float(w.sum())
    return float(w[mask].sum() / tot) if tot > 0 else float("nan")


def seed_margin(nbr_rel) -> np.ndarray:
    """Per-node relative margin to the nearest token-cell boundary of a
    bottleneck pack: (d2 - d1) / (d2 + d1), from the distances to the two
    nearest seeds (0 on a cell boundary, 1 at a seed). `nbr_rel` is
    the pack's (N, k, dim) node-minus-seed offsets."""
    rel = np.asarray(nbr_rel, dtype=np.float64)
    if rel.ndim != 3 or rel.shape[1] < 2:             # decode_k = 1: no second seed
        return np.full(rel.shape[0], np.nan)
    d = np.sort(np.linalg.norm(rel, axis=2), axis=1)
    return (d[:, 1] - d[:, 0]) / np.maximum(d[:, 1] + d[:, 0], 1e-300)


# ------------------------------------------------------ run inputs --------

def run_seeds(report: dict) -> list:
    return [int(s) for s in (report.get("provenance") or {}).get("seeds", [])]


def verified_states(report: dict, states_dir, seeds=None) -> dict:
    """{seed: path} of the run's AR states in `states_dir`, each file's SHA-256
    checked against the report's d9_restart record (refuses any other state)."""
    import hashlib
    from pathlib import Path

    rec = report["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    pool = max(int(p) for p in report["config"]["experiments"]["e8"]["pool_sizes"])
    out = {}
    for s in (run_seeds(report) if seeds is None else seeds):
        p = Path(states_dir) / f"ar_p{pool}_s{s}.pt"
        if not p.exists():
            raise SystemExit(f"{p}: missing (the run's seed-{s} AR state)")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        want = (rec.get(f"s{s}") or {}).get("sha256")
        if sha != want:
            raise SystemExit(f"{p}: SHA-256 {sha[:12]} is not the state the report "
                             f"trained for seed {s} ({str(want)[:12]})")
        out[int(s)] = p
    return out


def verified_dir(report: dict, data_dir) -> str:
    """The corpus directory, refused unless its manifest SHA-256 is the one the
    report recorded for it."""
    from pathlib import Path

    from ..data.archive import manifest_sha256

    want = {Path(str(d.get("dir") or d.get("path"))).name: d.get("manifest_sha256")
            for d in (report.get("provenance") or {}).get("datasets", [])}
    name = Path(str(data_dir)).name
    got = manifest_sha256(Path(data_dir))
    if want.get(name) != got:
        raise SystemExit(f"{data_dir}: manifest {got[:12]} is not the report's "
                         f"{str(want.get(name))[:12]}")
    return str(data_dir)


def run_files(report: dict, which: str, n: int | None = None, data_dir=None) -> list:
    """The run's own instance lists: 'val' (in-band validation split), 'train'
    (the AR training prefix, pool[:pool_size]) or 'fine' (the P3 evaluation
    set, manifest order[:n_eval])."""
    from pathlib import Path

    from ..data.archive import instance_files
    from .common import instance_files as split_files

    cfg = report["config"]
    if which in ("val", "train"):
        d = verified_dir(report, data_dir or cfg["data"]["dir"])
        files = split_files(d, split=cfg["split"], subset="val" if which == "val" else "pool")
        if which == "train":
            files = files[:max(int(p) for p in cfg["experiments"]["e8"]["pool_sizes"])]
    elif which == "fine":
        dt = cfg["data_transfer"]
        d = verified_dir(report, data_dir or dt["dir"])
        files = instance_files(Path(d))[:int((dt.get("split") or {}).get("n_eval", 256))]
    else:
        raise ValueError(which)
    return files if n is None else files[:n]


def run_model(report: dict, state_path, seed: int, device: str):
    """The report's own model (config embedded in the report), loaded strictly
    from a verified state, in eval mode under the run's TF32 policy."""
    from ..report import config_sha256
    from ..runtime import setup_torch
    from .common import build_model_from_config

    cfg = report["config"]
    if config_sha256(cfg) != report["provenance"]["config_sha256"]:
        raise SystemExit("the report's embedded configuration does not hash to its "
                         "recorded config_sha256")
    setup_torch(device, tf32=bool(cfg.get("tf32", True)))
    return build_model_from_config(cfg["model"], state_path=str(state_path), seed=int(seed),
                                   device=device)
