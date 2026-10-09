#!/usr/bin/env python3
"""cmame-paper Stage 7: the spectral content of the errors of CM2D's models,
in two dimensions, from the run's own states on its own validation instances.
Post hoc: specified after CM2D's verdict, reported only; no verdict reads it,
and nothing in PREREG_CM2D does.

Why. By Corollary "Modewise contraction", in the eigenbasis of the stiffness
matrix the energy norm weights the error in mode m by its eigenvalue and the
Euclidean displacement norm weights all modes equally. CM2D's H2b changed
only the norm of the supervised loss, from the relative Euclidean error (L_D,
row `labels`) to the relative stiffness-norm error (L_K, row
`labels_knorm`), and the von Mises error halved. This export measures what
the change of norm did to the error itself: how its squared norm is spread
over the eigenmodes, for both rows, the label-free row and the graph network.

Models (seeds 0-2), at the largest label budget (1,024) and the label-free
pool (1,024):

* `ar` (label-free; E1's states, evaluated by CM2D, not retrained):
  `ar_p{pool}_s{seed}.pt` in `--ar-states-dir` (default: the report's
  `reuse_from.states_dir`), SHA-256 checked against the report's d9_restart
  record and against the return's provenance file;
* `labels` (L_D), `labels_knorm` (L_K) and `mgn` (the graph network):
  `{row}_b{budget}_s{seed}.pt` in `--states-dir`, SHA-256 checked against
  the return's provenance file (its `sha256sum` lines of the kept states).

A state whose SHA-256 differs from its record is refused before any model
runs, and so is a report whose SHA-256 is not the one the provenance file
records. Every model is also checked by content: its five per-instance
metrics on the validation split, recomputed here with the evaluation's own
function, against the report's arrays for that row and seed (expected: 0 or
round-off for the transformers; for the graph network, whose CUDA scatter
reductions are not bitwise reproducible, about 1e-4 in 3D). A model whose
median relative deviation of the displacement error or of the relative
energy gap exceeds CONTENT_TOL is recorded as a content mismatch, and the
script ends with exit status 5 after writing everything.

Spectra. Per validation instance, the free block K_ff of the stiffness matrix
(at most about 3,600 free dofs here) is diagonalised once,
K_ff = W diag(lambda) W^T with lambda ascending and W orthonormal. For a
field x (an error e = u - U* or the reference U*) of one load case, with
coefficients x_hat = W^T x_f, the spectra are the sums of x_hat_m^2 (S2) and
of lambda_m x_hat_m^2 (SK) over two binnings of the modes:

* by stiffness relative to the solution (`_log`): mode m of load case l falls
  in the bin of log10(lambda_m / RQ*_l) on LOG_EDGES (1/8 decade from
  10^-2 to 10^6.5, with one bin below and one above), where
  RQ*_l = ||U*_l||_K^2 / ||U*_l||_2^2 is the solution's own Rayleigh
  quotient. This is the axis on which the two losses differ: the
  S2-weighted mean of lambda_m / RQ*_l is the error's normalised Rayleigh
  quotient, and L_K weights mode m by lambda_m relative to L_D;
* by rank (`_rank`): RANK_BINS bins of (nearly) equal mode count, bin b
  holding the modes floor(b n / RANK_BINS) to floor((b + 1) n / RANK_BINS)
  - 1. Nearly all of a smooth field's Euclidean norm falls in the first bin;
  recorded for completeness.

Summed over the bins either spectrum gives ||x||_2^2 and ||x||_K^2 (checked
against direct computations; the constrained dofs are zero in every
prediction and in U*, recorded).

Rounding. Every prediction is made as the run made it, with TF32 matrix
products (the configuration's policy); the rounding of TF32 (about 5e-4
relative) is rough, and the stiffness norm weights rough perturbations
heavily, so on the most accurate predictions it could account for part of
the stiff end of a spectrum. Each model therefore also predicts every
instance once with TF32 off (`fejepa.runtime.setup_torch(tf32=False)`, IEEE
float32 products, the attention without fused kernels), after which the
run's policy is restored: the error of that prediction is recorded beside
the other (`_ieee`: its norms and log spectra), and so is their difference
u - u_ieee (`_tf32`: its norms and its stiffness log spectrum), the TF32
rounding up to the kernels' own reordering of sums (about 1e-7 relative). The content check reads
the run's own (TF32) predictions only. The normalised Rayleigh quotient compares
an error with the solution, whose quotient is close to the smallest
eigenvalue (within a factor of a few tens on these plates, recorded), so
every error's exceeds 1 by far; the reading is the comparison between rows.

Outputs (`--out`, a directory):

* `spectra_val.npz`: per row, seed, validation instance and load case:
  Pi_h(u), the relative energy gap g = (Pi_h(u) - Pi_h(U*)) / |Pi_h(U*)|,
  the energy-optimal amplitude c* = F^T u / u^T K u and the relative gap of
  c* u; ||e||_2^2 and ||e||_K^2 (their ratio over U*'s is the error's
  normalised Rayleigh quotient); the relative von Mises error over the
  element values (the evaluation's vm_rel_l2) and weighted by element area
  (the L^2 norm of Proposition "Energy gap and stress error"); the
  stress-energy integral of e, the sum over elements of
  area (s_vm(e)^2 / (3G) + p(e)^2 / B) with p the mean stress of the plane
  stress state and B the three-dimensional bulk modulus, equal to ||e||_K^2
  by that proposition (a check); the largest absolute prediction on a
  constrained dof; the spectra of e in both binnings; the IEEE prediction's
  ||e||_2^2, ||e||_K^2 and log spectra; and the rounding u - u_ieee: its
  squared Euclidean and stiffness norms and its stiffness log spectrum. Per
  instance and load case of U*: Pi_h(U*), ||U*||_2^2, ||U*||_K^2, the area
  integrals of s_vm(U*)^2 and p(U*)^2,
  gamma* = 3G ||p(U*)||^2 / (B ||s_vm(U*)||^2), RQ* over the smallest
  eigenvalue, and the spectra of U*. Per instance: the number of free dofs,
  the smallest and largest eigenvalue, the eigenvalue at each rank bin's
  first mode, G and B.
* `fig2d.npz`: the instance chosen by RULES: the mesh, the load battery, U*,
  each row's seed-0 prediction and its IEEE prediction, element von Mises
  stresses (reference and rows), the element areas, the per-load energies,
  c*, RQ* and gamma*, the eigenvalues and, per load case, the squared
  coefficients of U* and of each row's seed-0 error in every mode.
* `spectra.json`: inputs and their hashes, the state and content checks, the
  selection, the counts of load cases worse than the zero field, the
  diagnostics' summaries (see `summary`), the eigenvalue ranges and timings.

    python scripts/cm2d_spectra.py --report records/cmame/cm2d/return/report.json \\
        --provenance records/cmame/cm2d/return/provenance.txt \\
        --states-dir runs/cm2d/e8_states --out runs/cmame/spectra/export
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROWS = ("ar", "labels", "labels_knorm", "mgn")
RANK_BINS = 40
"""Mode-rank bins: 40 aggregate exactly into halves, quarters, eighths,
tenths and twentieths."""
LOG_EDGES = -2.0 + 0.125 * np.arange(69)
"""log10(lambda_m / RQ*): 1/8-decade edges from -2 to 6.5 (exact in binary
floating point), so that whole decades and quarter decades are edges. On
CM2D's validation meshes the modes span about 10^-1.62 to 10^5.44 RQ*."""
LOG_BINS = LOG_EDGES.size + 1
"""70 bins: below the first edge, the 68 between edges, at or above the last."""
DECADES = (1, 2, 3, 4)
"""The summaries' thresholds: modes with lambda_m >= 10^k RQ*."""
CONTENT_TOL = 1e-3
"""Largest median relative deviation of a model's recomputed displacement
error and relative energy gap from the report's arrays."""
PAIRS = (("labels", "labels_knorm"), ("labels", "ar"), ("labels_knorm", "ar"),
         ("mgn", "labels"), ("mgn", "ar"))
RULES = {"fig2d": "the validation instance at 0-based rank (n - 1) // 2 of the labels row's "
                  "relative energy gaps at the largest budget, seed 0, in ascending order, "
                  "in the report's per-instance arrays"}
METRICS = ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err", "crit_recall")
GATED = ("disp_rel_l2", "energy_gap_rel")
DIAG = ("pi", "rel", "c", "rel_c", "e2", "eK", "vm_elem", "vm_area", "stress_energy",
        "u_dirichlet_max")
REF = ("pi_star", "u2_star", "uK_star", "gamma_star", "vm2_area_star", "p2_area_star",
       "rq_star_over_lam_min", "ustar_dirichlet_max")
SPECTRA = {"S2_rank": RANK_BINS, "SK_rank": RANK_BINS, "S2_log": LOG_BINS, "SK_log": LOG_BINS}
IEEE = ("e2_ieee", "eK_ieee", "d2_tf32", "dK_tf32")
"""Per prediction: the IEEE prediction's error norms, and the rounding's."""
IEEE_SPECTRA = {"S2_log_ieee": LOG_BINS, "SK_log_ieee": LOG_BINS, "SK_log_tf32": LOG_BINS}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------- inputs --

def budgets(report: dict) -> tuple:
    """(largest label budget, label-free pool) as the report's cells name
    them; every supervised row must have a cell at that budget."""
    cells = report["results"]["e8"]["metrics"]["cells"]
    b = str(max(int(x) for x in cells["labels"]))
    for row in ROWS[1:]:
        if b not in cells.get(row, {}):
            raise SystemExit(f"the report has no {row} cell at {b} labels")
    pool = str(int(report["config"]["experiments"]["e8"]["pool_sizes"][0]))
    if pool not in cells["ar"]:
        raise SystemExit(f"the report has no label-free cell at pool {pool}")
    return b, pool


def report_arrays(report: dict, row: str, seed_index: int) -> dict:
    """The report's per-instance validation arrays of one row and seed."""
    cells = report["results"]["e8"]["metrics"]["cells"]
    b, pool = budgets(report)
    return cells[row][pool if row == "ar" else b]["per_seed_eval"][seed_index]["per_instance"]


def provenance_hashes(path) -> dict:
    """{path as listed: SHA-256} of the `sha256sum` lines of a return's
    provenance file."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line.strip())
        if m:
            out[m.group(2)] = m.group(1)
    return out


def select(report: dict) -> dict:
    """The instance of RULES, from the report's own arrays."""
    lab = np.asarray(report_arrays(report, "labels", 0)["energy_gap_rel"], float)
    i = int(np.argsort(lab, kind="stable")[(len(lab) - 1) // 2])
    return {"fig2d": {"set": "val", "index": i, "value": float(lab[i])}}


def row_model(report: dict, row: str, state, seed: int, dev: str):
    """The report's model of `row` (the graph network is the run's own
    `kind: mgn` build), strictly loaded, in eval mode, under the run's TF32
    policy."""
    from fejepa.analysis.common import build_model_from_config
    from fejepa.analysis.posthoc import run_model
    from fejepa.report import config_sha256
    from fejepa.runtime import setup_torch

    if row != "mgn":
        return run_model(report, state, seed, dev)
    cfg = report["config"]
    if config_sha256(cfg) != report["provenance"]["config_sha256"]:
        raise SystemExit("the report's embedded configuration does not hash to its "
                         "recorded config_sha256")
    setup_torch(dev, tf32=bool(cfg.get("tf32", True)))
    return build_model_from_config(dict(cfg["model"], kind="mgn"), state_path=str(state),
                                   seed=int(seed), device=dev)


def predict(model, arch, dev: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        u = model.forward_instance(model.prepare_instance(arch, dev))
    return u.detach().cpu().numpy()


@contextmanager
def ieee_fp32(dev: str, tf32: bool):
    """Float32 products without TF32 (the codebase's own switch) and the
    attention restricted to the math backend, so that nn.MultiheadAttention
    takes its own unfused route, whose products are then those products (a
    fused kernel might round otherwise); then the run's policy `tf32` and the
    default attention backends again, whatever happens inside."""
    from fejepa.runtime import setup_torch

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        math_only = sdpa_kernel(SDPBackend.MATH)
    except ImportError:                       # an older torch: the products only
        math_only = nullcontext()
    setup_torch(dev, tf32=False)
    try:
        with math_only:
            yield
    finally:
        setup_torch(dev, tf32=tf32)


# ------------------------------------------------------------- the math --

def tri_ops(arch):
    """The P1-triangle operator of a plane-stress instance (areas, shape
    function derivatives, Lame parameters, shear and bulk moduli), built once
    and shared by every model and seed; None for another kind of mesh or
    material (the stress diagnostics are then NaN)."""
    from fejepa.fe.stress import _geometry, lame

    nodes, tris = np.asarray(arch.nodes, np.float64), np.asarray(arch.elements)
    m = arch.meta["material"]
    if nodes.shape[1] != 2 or tris.shape[1] != 3 or m.get("plane") != "stress":
        return None
    area, b, c = _geometry(nodes, tris)
    E, nu = float(m["E"]), float(m["nu"])
    lam, mu = lame(E, nu, "stress")
    return {"area": area, "b": b, "c": c, "tris": tris, "lam": lam, "mu": mu,
            "G": E / (2 * (1 + nu)), "Bk": E / (3 * (1 - 2 * nu))}


def tri_stress(op: dict, U: np.ndarray) -> np.ndarray:
    """(L, E, 3) plane stresses (sxx, syy, sxy) of node-major fields U, as
    fejepa.fe.stress.element_stresses computes them."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))
    t = op["tris"]
    ux, uy = U2[:, 2 * t], U2[:, 2 * t + 1]                       # (L, E, 3)
    exx = np.einsum("ei,lei->le", op["b"], ux)
    eyy = np.einsum("ei,lei->le", op["c"], uy)
    gxy = np.einsum("ei,lei->le", op["c"], ux) + np.einsum("ei,lei->le", op["b"], uy)
    tr = exx + eyy
    return np.stack([op["lam"] * tr + 2 * op["mu"] * exx, op["lam"] * tr + 2 * op["mu"] * eyy,
                     op["mu"] * gxy], axis=-1)


def vm_p(sig: np.ndarray) -> tuple:
    """Von Mises and mean stress of (..., 3) plane stresses (sigma_zz = 0)."""
    sxx, syy, sxy = sig[..., 0], sig[..., 1], sig[..., 2]
    return np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3.0 * sxy ** 2), (sxx + syy) / 3.0


def quad(K, X: np.ndarray) -> np.ndarray:
    """Per row x of X, x^T K x."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.einsum("ld,ld->l", X, (K @ X.T).T)


def eigenbasis(arch) -> dict:
    """The eigendecomposition of the instance's free stiffness block and its
    mode-rank bin edges."""
    from fejepa.analysis.posthoc import free_block, free_mask

    free = free_mask(arch)
    kf = free_block(arch.K, free)
    lam, W = np.linalg.eigh(kf.toarray())
    n = lam.size
    edges = np.floor(np.arange(RANK_BINS + 1) * n / RANK_BINS).astype(np.int64)
    return {"free": free, "Kf": kf, "lam": lam, "W": W, "edges": edges}


def log_index(lam: np.ndarray, rq_star) -> np.ndarray:
    """(L, n) log bin of every mode for each load case: lambda_m / RQ*_l on
    LOG_EDGES (0 below the first edge, LOG_BINS - 1 at or above the last)."""
    rq = np.asarray(rq_star, dtype=np.float64)
    if not np.all(rq > 0):
        raise SystemExit("a load case whose reference solution has no stiffness-norm energy")
    if not np.all(lam > 0):
        raise SystemExit("a free stiffness block that is not positive definite")
    return np.searchsorted(LOG_EDGES, np.log10(lam[None, :] / rq[:, None]), side="right")


def edge_bin(k: float) -> int:
    """The first log bin of the modes with lambda_m >= 10^k RQ* (k an edge)."""
    j = int(np.searchsorted(LOG_EDGES, k))
    if not np.isclose(LOG_EDGES[j], k):
        raise ValueError(f"10^{k} is not an edge")
    return j + 1


def binned(eig: dict, X: np.ndarray, log_idx: np.ndarray) -> dict:
    """The squared coefficients (L, n) of fields X and their spectra in both
    binnings."""
    C2 = (np.atleast_2d(np.asarray(X, dtype=np.float64))[:, eig["free"]] @ eig["W"]) ** 2
    lam, e = eig["lam"], eig["edges"]
    CK = C2 * lam
    return {"C2": C2,
            "S2_rank": np.stack([C2[:, a:b].sum(axis=1) for a, b in zip(e[:-1], e[1:])], axis=1),
            "SK_rank": np.stack([CK[:, a:b].sum(axis=1) for a, b in zip(e[:-1], e[1:])], axis=1),
            "S2_log": np.stack([np.bincount(log_idx[j], weights=C2[j], minlength=LOG_BINS)
                                for j in range(C2.shape[0])]),
            "SK_log": np.stack([np.bincount(log_idx[j], weights=CK[j], minlength=LOG_BINS)
                                for j in range(C2.shape[0])])}


def reference(arch, op, eig) -> dict:
    """Per load case of U*: Pi_h(U*), ||U*||_2^2, ||U*||_K^2, the area
    integrals of s_vm(U*)^2 and p(U*)^2 and gamma* (with a plane-stress
    triangle mesh, and its stresses), RQ* over the smallest eigenvalue, the
    largest |U*| on a constrained dof, the log bin of every mode and the
    spectra of U*."""
    from fejepa.anchor.energy import pi_h

    S = np.atleast_2d(np.asarray(arch.U_star, dtype=np.float64))
    fixed = ~eig["free"]
    u2, uK = np.einsum("ld,ld->l", S, S), quad(arch.K, S)
    idx = log_index(eig["lam"], uK / u2)
    nan = np.full(S.shape[0], np.nan)
    out = {"pi_star": pi_h(S, arch.K, arch.F), "u2_star": u2, "uK_star": uK,
           "gamma_star": nan, "vm2_area_star": nan.copy(), "p2_area_star": nan.copy(),
           "rq_star_over_lam_min": (uK / u2) / eig["lam"][0],
           "ustar_dirichlet_max": (np.abs(S[:, fixed]).max(axis=1) if fixed.any()
                                   else np.zeros(S.shape[0])),
           "log_idx": idx, "sig": None, "vm": None, **binned(eig, S, idx)}
    if op is not None:
        out["sig"] = tri_stress(op, S)
        vm, p = vm_p(out["sig"])
        A = op["area"]
        out["vm"] = vm
        out["vm2_area_star"], out["p2_area_star"] = (vm ** 2) @ A, (p ** 2) @ A
        out["gamma_star"] = 3.0 * op["G"] * out["p2_area_star"] / (op["Bk"] * out["vm2_area_star"])
    return out


def diagnostics(U: np.ndarray, arch, op, ref: dict, eig: dict) -> dict:
    """Per load case of a prediction u, with e = u - U*: the energies, c*,
    the error norms, the von Mises errors, the stress-energy integral, the
    largest |u| on a constrained dof and the spectra of e."""
    from fejepa.analysis.posthoc import amplitude_factor, apply_amplitude
    from fejepa.anchor.energy import pi_h

    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))
    Er = U2 - np.atleast_2d(np.asarray(arch.U_star, dtype=np.float64))
    free = eig["free"]
    pi_u = pi_h(U2, arch.K, arch.F)
    c = amplitude_factor(U2, arch.K, arch.F, free, Kf=eig["Kf"])
    pi_c = pi_h(apply_amplitude(U2, c), arch.K, arch.F)
    a = np.abs(ref["pi_star"])
    out = {"pi": pi_u, "rel": (pi_u - ref["pi_star"]) / a, "c": c,
           "rel_c": (pi_c - ref["pi_star"]) / a,
           "e2": np.einsum("ld,ld->l", Er, Er), "eK": quad(arch.K, Er),
           "u_dirichlet_max": (np.abs(U2[:, ~free]).max(axis=1) if (~free).any()
                               else np.zeros(U2.shape[0])),
           **binned(eig, Er, ref["log_idx"])}
    nan = np.full(U2.shape[0], np.nan)
    if op is None:
        out.update(vm_elem=nan, vm_area=nan.copy(), stress_energy=nan.copy())
        return out
    A, vm_s = op["area"], ref["vm"]
    sig_u = tri_stress(op, U2)
    vm_u, _ = vm_p(sig_u)
    d = vm_u - vm_s
    vm_e, p_e = vm_p(sig_u - ref["sig"])          # linear: sigma(e) = sigma(u) - sigma(U*)
    out["vm_elem"] = np.linalg.norm(d, axis=1) / (np.linalg.norm(vm_s, axis=1) + 1e-30)
    out["vm_area"] = np.sqrt(((d ** 2) @ A) / ((vm_s ** 2) @ A))
    out["stress_energy"] = (vm_e ** 2 / (3.0 * op["G"]) + p_e ** 2 / op["Bk"]) @ A
    return out


def ieee_diagnostics(U: np.ndarray, U_ieee: np.ndarray, arch, ref: dict, eig: dict) -> dict:
    """Per load case: the IEEE prediction's error norms and log spectra, and
    the rounding U - U_ieee: its squared Euclidean and stiffness norms and its
    stiffness log spectrum."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))
    V2 = np.atleast_2d(np.asarray(U_ieee, dtype=np.float64))
    Ei = V2 - np.atleast_2d(np.asarray(arch.U_star, dtype=np.float64))
    Dl = U2 - V2
    bi, bd = binned(eig, Ei, ref["log_idx"]), binned(eig, Dl, ref["log_idx"])
    return {"e2_ieee": np.einsum("ld,ld->l", Ei, Ei), "eK_ieee": quad(arch.K, Ei),
            "S2_log_ieee": bi["S2_log"], "SK_log_ieee": bi["SK_log"],
            "d2_tf32": np.einsum("ld,ld->l", Dl, Dl), "dK_tf32": quad(arch.K, Dl),
            "SK_log_tf32": bd["SK_log"]}


def _ratio(a, b) -> np.ndarray:
    """a / b where b > 0, NaN elsewhere."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=np.broadcast_to(b > 0, out.shape))
    return out


def rayleigh_ratio(D: dict, R: dict) -> np.ndarray:
    """(rows, seeds, n, L) normalised Rayleigh quotients
    (||e||_K^2 / ||e||_2^2) / (||U*||_K^2 / ||U*||_2^2)."""
    return _ratio(_ratio(D["eK"], D["e2"]), _ratio(R["uK_star"], R["u2_star"])[None, None])


def tail_shares(S: np.ndarray) -> np.ndarray:
    """(..., bins) share of a binned spectrum's sum in bins b and above."""
    tail = np.cumsum(S[..., ::-1], axis=-1)[..., ::-1]
    return _ratio(tail, S.sum(axis=-1, keepdims=True))


def _median_curve(X: np.ndarray) -> list:
    """Per bin, the median over every other axis (NaN-aware)."""
    return [float(v) for v in np.nanmedian(X.reshape(-1, X.shape[-1]), axis=0)]


def _above(S: np.ndarray) -> dict:
    """Median share of a log spectrum in modes with lambda_m >= 10^k RQ*."""
    t = tail_shares(S)
    return {f"1e{k}": float(np.nanmedian(t[..., edge_bin(k)])) for k in DECADES}


def _median_or_none(x) -> float | None:
    """Median of the finite values; None when there are none."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else None


def _gm(r) -> float:
    """Geometric mean of positive values."""
    return float(np.exp(np.mean(np.log(r))))


def _pos(*z) -> np.ndarray:
    """Where every array is finite and positive."""
    return np.logical_and.reduce([np.isfinite(v) & (v > 0) for v in z])


def _seed_geomean(x: np.ndarray) -> np.ndarray:
    """(n, L) geometric mean over the seed axis of positive values (NaN-aware)."""
    lx = np.full(x.shape, np.nan)
    ok = _pos(x)
    lx[ok] = np.log(x[ok])
    with np.errstate(all="ignore"):
        return np.exp(np.nanmean(lx, axis=0))


def _pair_core(rq, g, d2, ia: int, ib: int, both: np.ndarray) -> dict | None:
    """The readings of one pair of rows that the TF32 and the IEEE predictions
    share: the share of triples on which the first row's normalised Rayleigh
    quotient is the larger, the same on the seed geometric means (free of the
    seed pairing), the median ratio, and the geometric-mean ratios of the
    relative gap, the quotient and the squared displacement error (per load
    case g = rq d^2 exactly, so the first is the product of the other two)."""
    x, y = rq[ia][both], rq[ib][both]
    ga, gb, da, db = g[ia][both], g[ib][both], d2[ia][both], d2[ib][both]
    m = _pos(x, y, ga, gb, da, db)
    if not m.any():
        return None
    xs, ys = _seed_geomean(x), _seed_geomean(y)
    ms = _pos(xs, ys)
    return {"n": int(m.sum()), "share_first_above": float(np.mean(x[m] > y[m])),
            "share_first_above_seed_geomean": (float(np.mean(xs[ms] > ys[ms]))
                                               if ms.any() else None),
            "median_ratio": float(np.median(x[m] / y[m])),
            "geomean_ratio_gap": _gm(ga[m] / gb[m]),
            "geomean_ratio_rayleigh": _gm(x[m] / y[m]),
            "geomean_ratio_disp_sq": _gm(da[m] / db[m])}


def _pair(D: dict, rq, g, d2, ia: int, ib: int, both: np.ndarray) -> dict | None:
    out = _pair_core(rq, g, d2, ia, ib, both)
    if out is None:
        return None
    x, y = rq[ia][both], rq[ib][both]
    ga, gb, da, db = g[ia][both], g[ib][both], d2[ia][both], d2[ib][both]
    va, vb = D["vm_elem"][ia][both], D["vm_elem"][ib][both]
    wa, wb = D["vm_area"][ia][both], D["vm_area"][ib][both]
    m = _pos(x, y, ga, gb, da, db)
    mv, mw = _pos(va, vb), _pos(wa, wb)
    seeds = [si for si in range(both.size) if both[si]]
    out.update({
        # the verdict's kind of comparison: ratio of the means over every
        # (seed, instance, load case), i.e. of the seed means
        "ratio_of_means_gap": float(np.mean(ga[m]) / np.mean(gb[m])),
        "ratio_of_means_vm_elem": (float(np.mean(va[mv]) / np.mean(vb[mv]))
                                   if mv.any() else None),
        "geomean_ratio_vm_elem_sq": _gm((va[mv] / vb[mv]) ** 2) if mv.any() else None,
        "geomean_ratio_vm_area_sq": _gm((wa[mw] / wb[mw]) ** 2) if mw.any() else None,
        "share_first_gap_lower": float(np.mean(ga[m] < gb[m])),
        "share_first_disp_lower": float(np.mean(da[m] < db[m])),
        # opposite rankings (Remark "Opposite rankings")
        "share_first_disp_lower_gap_higher": float(np.mean((da[m] < db[m]) & (ga[m] > gb[m]))),
        "share_first_disp_higher_gap_lower": float(np.mean((da[m] > db[m]) & (ga[m] < gb[m]))),
        "by_seed": []})
    for k, si in enumerate(seeds):
        ms = m[k]
        out["by_seed"].append(
            {"seed_index": si, "n": int(ms.sum()),
             "geomean_ratio_gap": _gm(ga[k][ms] / gb[k][ms]) if ms.any() else None,
             "geomean_ratio_rayleigh": _gm(x[k][ms] / y[k][ms]) if ms.any() else None,
             "geomean_ratio_disp_sq": _gm(da[k][ms] / db[k][ms]) if ms.any() else None})
    return out


def summary(D: dict, R: dict, used: np.ndarray) -> dict:
    """The readings of the arrays.

    Per row, over its (seed, instance, load case) triples: the median, the
    10th and 90th percentiles and the per-seed medians of the normalised
    Rayleigh quotient; the median shares of ||e||_2^2 and of ||e||_K^2 in the
    modes with lambda_m >= 10^k RQ* (k in DECADES), and per bin the median
    share in that bin and above, for the log and the rank spectra; the
    medians of the two von Mises errors and of the relative gap before and
    after c*; the largest and median ratio vm_area^2 / ((1 + gamma*) g)
    (Proposition "Energy gap and stress error": at most 1); the largest
    relative deviations of ||e||_K^2 from the stress-energy integral and from
    2 (Pi_h(u) - Pi_h(U*)), and, per binning, of ||e||_K^2 and ||e||_2^2 from
    the sums of the spectra; the largest |u| on a constrained dof; the
    medians of the quotient per load case (RQ* differs between load cases by
    up to about a decade relative to the smallest eigenvalue). The same
    model's IEEE prediction (`ieee`): the quotient, the relative gap and the
    shares above 10^k RQ* and per bin. The rounding u - u_ieee (`tf32`): the
    median (and 90th percentile and largest) ratio of its squared stiffness
    norm to the error's, the same ratio in the modes with
    lambda_m >= 10^k RQ* (k = 2, 3, 4), and the median ratio of its squared
    Euclidean norm to the error's.

    Per pair of rows (PAIRS), on the same seed index, instance and load case:
    the share on which the first row's error has the larger normalised
    Rayleigh quotient and the median ratio; the ratios (first / second) of
    the means of the relative gap and of the von Mises error (the verdict's
    kind of comparison); since per load case the relative gap is the normalised
    Rayleigh quotient times the squared relative displacement error, the
    geometric-mean ratios of the three (the first is the product of the
    other two), overall and per seed, and of the squared von Mises errors;
    the shares on which the first row has the lower gap, the lower
    displacement error, and each of the two opposite rankings; the share of
    instance and load cases on which the seed geometric mean of the first
    row's quotient is the larger, which does not depend on the pairing of
    seeds; and the readings the IEEE predictions share (`ieee`). At a given
    seed the three transformer rows start from the same initial weights and
    visit the instances in the same order (PREREG_CM2D Sec. 2), so their
    pairing is a control; the graph network's seeds are unrelated to the
    transformers', so for its pairs only the ratios of means, the pooled
    geometric means (not the per-seed ones) and the share on the seed
    geometric means do not depend on the pairing. The factorisation holds
    for the geometric means, not for the ratios of means.

    Of U*: the median shares above 10^k RQ* and per bin, the median RQ*, the
    median RQ* over the smallest eigenvalue, the identities' largest
    deviations (the spectra's sums; -2 Pi_h(U*) = ||U*||_K^2; and
    (1 + gamma*) ||s_vm(U*)||^2 / (3G) = ||U*||_K^2, which checks gamma*
    itself), the median gamma* and the largest |U*| on a constrained dof."""
    rq = rayleigh_ratio(D, R)
    g = _ratio(D["eK"], R["uK_star"][None, None])
    d2 = _ratio(D["e2"], R["u2_star"][None, None])          # squared relative disp. error
    rq_i = rayleigh_ratio({"eK": D["eK_ieee"], "e2": D["e2_ieee"]}, R)
    g_i = _ratio(D["eK_ieee"], R["uK_star"][None, None])
    d2_i = _ratio(D["e2_ieee"], R["u2_star"][None, None])
    noise = _ratio(D["dK_tf32"], D["eK"])                   # rounding over error, K-norm
    noise_above = {}
    for k in (2, 3, 4):
        j = edge_bin(k)
        noise_above[f"1e{k}"] = _ratio(D["SK_log_tf32"][..., j:].sum(axis=-1),
                                       D["SK_log"][..., j:].sum(axis=-1))
    bound = _ratio(D["vm_area"] ** 2, (1.0 + R["gamma_star"])[None, None] * g)
    gap = 2.0 * (D["pi"] - R["pi_star"][None, None])
    fin = lambda x: x[np.isfinite(x)]                                      # noqa: E731
    dev = lambda a, b: float(np.nanmax(np.abs(a - b) / np.abs(b)))         # noqa: E731
    out = {"rows": {}, "pairs": {}}
    for ri, row in enumerate(ROWS):
        ok = used[ri]
        q = fin(rq[ri][ok]) if ok.any() else np.empty(0)
        if not q.size:
            out["rows"][row] = None
            continue
        sel = lambda X: X[ri][ok]                                          # noqa: E731
        out["rows"][row] = {
            "n": int(q.size),
            "rayleigh_ratio_median": float(np.median(q)),
            "rayleigh_ratio_p10_p90": [float(np.percentile(q, p)) for p in (10, 90)],
            "rayleigh_ratio_median_by_seed": [
                (float(np.nanmedian(rq[ri][si])) if ok[si] else None) for si in range(len(ok))],
            "rayleigh_ratio_median_by_load": [float(v) for v in
                                              np.nanmedian(sel(rq).reshape(-1, rq.shape[-1]), 0)],
            "share_e2_above_median": _above(sel(D["S2_log"])),
            "share_eK_above_median": _above(sel(D["SK_log"])),
            "tail_log_e2_median": _median_curve(tail_shares(sel(D["S2_log"]))),
            "tail_log_eK_median": _median_curve(tail_shares(sel(D["SK_log"]))),
            "tail_rank_e2_median": _median_curve(tail_shares(sel(D["S2_rank"]))),
            "tail_rank_eK_median": _median_curve(tail_shares(sel(D["SK_rank"]))),
            "vm_elem_median": float(np.nanmedian(sel(D["vm_elem"]))),
            "vm_area_median": float(np.nanmedian(sel(D["vm_area"]))),
            "rel_gap_median": float(np.nanmedian(sel(D["rel"]))),
            "rel_gap_after_cstar_median": float(np.nanmedian(sel(D["rel_c"]))),
            "prop1_bound_ratio_max": (float(np.max(fin(sel(bound))))
                                      if fin(sel(bound)).size else None),
            "prop1_bound_ratio_median": float(np.nanmedian(sel(bound))),
            "stress_identity_max_rel_dev": dev(sel(D["stress_energy"]), sel(D["eK"])),
            "gap_identity_max_rel_dev": dev(sel(gap), sel(D["eK"])),
            "spectral_eK_max_rel_dev": {b: dev(sel(D[f"SK_{b}"]).sum(axis=-1), sel(D["eK"]))
                                        for b in ("rank", "log")},
            "spectral_e2_max_rel_dev": {b: dev(sel(D[f"S2_{b}"]).sum(axis=-1), sel(D["e2"]))
                                        for b in ("rank", "log")},
            "u_dirichlet_max": float(np.nanmax(sel(D["u_dirichlet_max"]))),
            # the same model with TF32 off (IEEE float32 products)
            "ieee": {"rayleigh_ratio_median": float(np.nanmedian(sel(rq_i))),
                     "rayleigh_ratio_p10_p90": [float(np.nanpercentile(sel(rq_i), p))
                                                for p in (10, 90)],
                     "rel_gap_median": float(np.nanmedian(sel(g_i))),
                     "share_e2_above_median": _above(sel(D["S2_log_ieee"])),
                     "share_eK_above_median": _above(sel(D["SK_log_ieee"])),
                     "tail_log_e2_median": _median_curve(tail_shares(sel(D["S2_log_ieee"]))),
                     "tail_log_eK_median": _median_curve(tail_shares(sel(D["SK_log_ieee"])))},
            # the rounding u - u_ieee against the error e of the TF32 prediction
            "tf32": {"dK_over_eK_median": float(np.nanmedian(sel(noise))),
                     "dK_over_eK_p90_max": [float(np.nanpercentile(sel(noise), 90)),
                                            float(np.nanmax(sel(noise)))],
                     "d2_over_e2_median": float(np.nanmedian(sel(_ratio(D["d2_tf32"], D["e2"])))),
                     "dK_over_eK_above_median": {k: _median_or_none(sel(v))
                                                 for k, v in noise_above.items()},
                     # load cases with error energy above 10^k RQ* (a load case
                     # whose stiffest mode lies below it has none and is left out)
                     "dK_over_eK_above_n": {k: int(np.isfinite(sel(v)).sum())
                                            for k, v in noise_above.items()}}}
    for a, b in PAIRS:
        ia, ib = ROWS.index(a), ROWS.index(b)
        pr = _pair(D, rq, g, d2, ia, ib, used[ia] & used[ib])
        if pr is not None:
            pr["ieee"] = _pair_core(rq_i, g_i, d2_i, ia, ib, used[ia] & used[ib])
        out["pairs"][f"{a}_vs_{b}"] = pr
    vm_id = (1.0 + R["gamma_star"]) * R["vm2_area_star"] / (3.0 * R["G"][:, None])
    out["reference"] = {
        "share_u2_above_median": _above(R["S2_log"]),
        "share_uK_above_median": _above(R["SK_log"]),
        "tail_log_u2_median": _median_curve(tail_shares(R["S2_log"])),
        "tail_log_uK_median": _median_curve(tail_shares(R["SK_log"])),
        "tail_rank_u2_median": _median_curve(tail_shares(R["S2_rank"])),
        "tail_rank_uK_median": _median_curve(tail_shares(R["SK_rank"])),
        "rayleigh_median": float(np.nanmedian(_ratio(R["uK_star"], R["u2_star"]))),
        "rayleigh_over_lam_min_median": float(np.nanmedian(R["rq_star_over_lam_min"])),
        "rayleigh_over_lam_min_min_max": [float(np.nanmin(R["rq_star_over_lam_min"])),
                                          float(np.nanmax(R["rq_star_over_lam_min"]))],
        "spectral_uK_max_rel_dev": {b: dev(R[f"SK_{b}"].sum(axis=-1), R["uK_star"])
                                    for b in ("rank", "log")},
        "spectral_u2_max_rel_dev": {b: dev(R[f"S2_{b}"].sum(axis=-1), R["u2_star"])
                                    for b in ("rank", "log")},
        "energy_identity_max_rel_dev": dev(-2.0 * R["pi_star"], R["uK_star"]),
        "gamma_identity_max_rel_dev": dev(vm_id, R["uK_star"]),
        "gamma_star_median": float(np.nanmedian(R["gamma_star"])),
        "ustar_dirichlet_max": float(np.nanmax(R["ustar_dirichlet_max"]))}
    return out


def counts(D: dict, used: np.ndarray) -> dict:
    out = {}
    for ri, row in enumerate(ROWS):
        ok = used[ri]
        if not ok.any():
            out[row] = None
            continue
        pi, rel, rel_c = D["pi"][ri][ok], D["rel"][ri][ok], D["rel_c"][ri][ok]
        out[row] = {"load_cases": int(pi.size),
                    "load_cases_pi_positive": int(np.sum(pi > 0)),
                    "load_cases_rel_gap_above_1": int(np.sum(rel > 1)),
                    "instances": int(rel.shape[0] * rel.shape[1]),
                    "instances_mean_rel_gap_above_1": int(np.sum(rel.mean(axis=-1) > 1)),
                    "load_cases_rel_gap_above_1_after_cstar": int(np.sum(rel_c > 1 + 1e-9)),
                    "load_cases_cstar_increased_gap":
                        int(np.sum(rel_c > rel * (1 + 1e-9) + 1e-12))}
    return out


def content_check(got: dict, ref: dict) -> dict:
    out = {"n": len(got["disp_rel_l2"])}
    for k in METRICS:
        g = np.asarray(got[k], float)
        r = np.asarray(ref[k][:len(g)], float)
        dev = np.abs(g - r) / np.maximum(np.abs(r), 1e-30)
        out[f"{k}_median_rel_dev"] = float(np.median(dev))
        out[f"{k}_max_rel_dev"] = float(np.max(dev))
    out["ok"] = all(out[f"{k}_median_rel_dev"] <= CONTENT_TOL for k in GATED)
    return out


def _torch_info(dev: str) -> dict:
    import torch

    out = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": dev,
           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
           "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if str(dev).startswith("cuda"):
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


# ---------------------------------------------------------------- main --

def verify_inputs(report: dict, report_path, prov: dict, states_dir, ar_dir, seeds) -> tuple:
    """The report's and every state's SHA-256 against the provenance file
    (and the label-free states against the report's d9_restart record);
    refuses on any difference. Returns ({(row, seed): path}, records)."""
    from fejepa.analysis.posthoc import verified_states

    b, pool = budgets(report)
    want = [v for k, v in prov.items() if k.endswith("/report.json") or k == "report.json"]
    got = sha256(report_path)
    if want != [got]:
        raise SystemExit(f"{report_path}: SHA-256 {got[:12]} is not the report the provenance "
                         f"file records ({', '.join(w[:12] for w in want) or 'none'})")
    by_name = {}
    for k, v in prov.items():
        by_name.setdefault(Path(k).name, set()).add(v)
    ar = verified_states(report, ar_dir, seeds=seeds)             # refuses on mismatch
    paths, recs = {}, {}
    for row in ROWS:
        for s in seeds:
            p = Path(ar[s]) if row == "ar" else Path(states_dir) / f"{row}_b{b}_s{s}.pt"
            if not p.exists():
                raise SystemExit(f"{p}: missing (the run's seed-{s} {row} state)")
            h, rec = sha256(p), by_name.get(p.name, set())
            if rec != {h}:
                raise SystemExit(f"{p}: SHA-256 {h[:12]} is not the state the provenance file "
                                 f"records ({', '.join(sorted(x[:12] for x in rec)) or 'none'})")
            paths[(row, s)] = p
            recs[f"{row}_s{s}"] = {"file": p.name, "dir": str(p.parent), "sha256": h,
                                   "sha256_ok": True,
                                   "checked_against": (["provenance", "d9_restart"]
                                                       if row == "ar" else ["provenance"])}
    return paths, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="the run's report (the records copy)")
    ap.add_argument("--provenance", required=True,
                    help="the return's provenance file (its sha256sum lines)")
    ap.add_argument("--states-dir", required=True, help="the run's kept supervised states")
    ap.add_argument("--ar-states-dir", default=None,
                    help="the label-free states (default: the report's reuse_from.states_dir)")
    ap.add_argument("--n-val", type=int, default=None, help="validation instances (all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    if a.n_val is not None and a.n_val < 1:
        raise SystemExit("--n-val: at least one validation instance")

    from fejepa.analysis.common import resolve_device, sha256_of, write_json
    from fejepa.analysis.posthoc import run_files, run_seeds
    from fejepa.data.archive import LazyArchives, has_labels, load_instance
    from fejepa.metrics import evaluate_fields
    from fejepa.report import _git_describe

    report = json.loads(Path(a.report).read_text())
    dev = resolve_device(a.device)
    seeds = run_seeds(report)
    b, pool = budgets(report)
    ar_dir = a.ar_states_dir or (report.get("reuse_from") or {}).get("states_dir")
    if not ar_dir:
        raise SystemExit("--ar-states-dir: the report names no reuse_from.states_dir")
    out_dir = Path(a.out)
    tf32 = bool(report["config"].get("tf32", True))          # the run's policy

    # ---- inputs first: the instance list, the figure instance, the states --
    files = run_files(report, "val")
    val_files = files[:a.n_val] if a.n_val else files
    sel = select(report)
    for name, rule in sel.items():
        if not 0 <= rule["index"] < len(files):
            raise SystemExit(f"{name}: index {rule['index']} outside the validation set "
                             f"({len(files)} instances)")
        f = Path(files[rule["index"]])
        if not f.exists() or not has_labels(f):
            raise SystemExit(f"{name}: {f} is missing or holds no labels")
        rule["file"] = f.name
    paths, res_states = verify_inputs(report, a.report, provenance_hashes(a.provenance),
                                      a.states_dir, ar_dir, seeds)
    models = {k: row_model(report, k[0], p, k[1], dev) for k, p in paths.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"what": "cmame-paper Stage 7: spectral content of the errors of CM2D's models "
                   "(post hoc; reported only)",
           "git": _git_describe(), "torch": _torch_info(dev), "report": a.report,
           "report_sha256": sha256_of(a.report), "provenance": a.provenance,
           "provenance_sha256": sha256_of(a.provenance),
           "config_sha256": report["provenance"]["config_sha256"], "budget": int(b),
           "pool": int(pool), "seeds": seeds, "rows": list(ROWS), "rank_bins": RANK_BINS,
           "log_edges": [float(x) for x in LOG_EDGES], "decades": list(DECADES),
           "pairs": [list(p) for p in PAIRS], "rules": RULES, "selections": sel,
           "content_tol": CONTENT_TOL, "states": res_states,
           "precision": {"tf32_policy": tf32,
                         "ieee_pass": "every model predicts every instance a second time "
                                      "under fejepa.runtime.setup_torch(tf32=False), the "
                                      "attention by the math SDPA backend; the run's policy "
                                      "and the default backends are restored after each "
                                      "instance"},
           "val": {"n": len(val_files), "files": [Path(f).name for f in val_files]}}
    write_json(out_dir / "spectra.json", res)

    # ---- every validation instance x row x seed ------------------------------
    n, keys = len(val_files), [(row, s) for row in ROWS for s in seeds]
    s0 = seeds[0]
    keep = {rule["index"]: name for name, rule in sel.items()}
    kept = {}
    D, R = None, None
    metrics = {k: {m: [] for m in METRICS} for k in keys}
    t0, t_inf, t_eig = time.time(), {f"{r}_s{s}": 0.0 for r, s in keys}, 0.0
    t_ieee = 0.0
    for i, arch in enumerate(LazyArchives(val_files)):
        L = arch.n_loads
        if D is None:
            D = {k: np.full((len(ROWS), len(seeds), n, L), np.nan) for k in DIAG}
            D.update({k: np.full((len(ROWS), len(seeds), n, L, nb), np.nan)
                      for k, nb in SPECTRA.items()})
            D.update({k: np.full((len(ROWS), len(seeds), n, L), np.nan) for k in IEEE})
            D.update({k: np.full((len(ROWS), len(seeds), n, L, nb), np.nan)
                      for k, nb in IEEE_SPECTRA.items()})
            R = {k: np.full((n, L), np.nan) for k in REF}
            R.update({k: np.full((n, L, nb), np.nan) for k, nb in SPECTRA.items()})
            R.update({"n_free": np.zeros(n, np.int64), "lam_min": np.full(n, np.nan),
                      "lam_max": np.full(n, np.nan), "G": np.full(n, np.nan),
                      "Bk": np.full(n, np.nan),
                      "lam_rank_start": np.full((n, RANK_BINS), np.nan)})
        elif L != D["pi"].shape[-1]:
            raise SystemExit(f"{val_files[i]}: {L} load cases, not {D['pi'].shape[-1]}")
        t = time.perf_counter()
        eig = eigenbasis(arch)
        t_eig += time.perf_counter() - t
        lam = eig["lam"]
        R["n_free"][i], R["lam_min"][i], R["lam_max"][i] = lam.size, lam[0], lam[-1]
        R["lam_rank_start"][i] = lam[np.minimum(eig["edges"][:-1], lam.size - 1)]
        op = tri_ops(arch)
        if op is not None:
            R["G"][i], R["Bk"][i] = op["G"], op["Bk"]
        ref = reference(arch, op, eig)
        for k in (*REF, *SPECTRA):
            R[k][i] = ref[k]
        if i in keep:
            kept[(keep[i], "lam")], kept[(keep[i], "ref")] = lam, ref
        preds = {}
        for (row, s) in keys:
            t = time.perf_counter()
            preds[(row, s)] = predict(models[(row, s)], arch, dev)
            t_inf[f"{row}_s{s}"] += time.perf_counter() - t
        t = time.perf_counter()
        with ieee_fp32(dev, tf32):
            preds_ieee = {k: predict(models[k], arch, dev) for k in keys}
        t_ieee += time.perf_counter() - t
        for (row, s) in keys:
            U = preds[(row, s)]
            dg = diagnostics(U, arch, op, ref, eig)
            di = ieee_diagnostics(U, preds_ieee[(row, s)], arch, ref, eig)
            ri, si = ROWS.index(row), seeds.index(s)
            for k in (*DIAG, *SPECTRA):
                D[k][ri, si, i] = dg[k]
            for k in (*IEEE, *IEEE_SPECTRA):
                D[k][ri, si, i] = di[k]
            if s == s0 and i in keep:
                kept[(keep[i], row)] = (U, dg, preds_ieee[(row, s)])
            ev = evaluate_fields(U, arch)
            for m in METRICS:
                metrics[(row, s)][m].append(ev[m])
        if (i + 1) % 16 == 0 or i + 1 == n:
            print(f"[spectra] val {i + 1}/{n} | {time.time() - t0:.0f} s", flush=True)

    checks, mismatch = {}, []
    for (row, s) in keys:
        chk = content_check(metrics[(row, s)], report_arrays(report, row, seeds.index(s)))
        res_states[f"{row}_s{s}"]["content"] = chk
        checks[f"{row}_s{s}"] = chk
        if not chk["ok"]:
            mismatch.append(f"{row}_s{s}")
    used = np.ones((len(ROWS), len(seeds)), dtype=bool)          # every state passed its hash
    np.savez_compressed(out_dir / "spectra_val.npz", rows=np.array(ROWS), seeds=np.array(seeds),
                        files=np.array([Path(f).name for f in val_files]),
                        rank_bins=np.int64(RANK_BINS), log_edges=LOG_EDGES,
                        **{k: D[k] for k in (*DIAG, *SPECTRA, *IEEE, *IEEE_SPECTRA)},
                        **{k: R[k] for k in REF},
                        **{f"{k}_star": R[k] for k in SPECTRA},
                        **{k: R[k] for k in ("n_free", "lam_min", "lam_max", "lam_rank_start",
                                             "G", "Bk")})
    diag = summary(D, R, used)
    lam_min, lam_max = R["lam_min"], R["lam_max"]
    res.update(content_checks=checks, content_mismatch=mismatch, counts=counts(D, used),
               diagnostics=diag,
               eigen={"n_free_min_median_max": [int(R["n_free"].min()),
                                                 float(np.median(R["n_free"])),
                                                 int(R["n_free"].max())],
                      "lam_min_min": float(lam_min.min()),
                      "sqrt_kappa_min_max": [float(np.sqrt(lam_max / lam_min).min()),
                                             float(np.sqrt(lam_max / lam_min).max())],
                      "seconds": round(t_eig, 1)},
               inference_seconds=t_inf, inference_seconds_ieee=round(t_ieee, 1),
               torch_after=_torch_info(dev), val_seconds=round(time.time() - t0, 1))
    write_json(out_dir / "spectra.json", res)

    # ---- the figure instance --------------------------------------------------
    figs = {}
    for name, rule in sel.items():
        f = files[rule["index"]]
        arch = load_instance(f)
        op = tri_ops(arch)
        if (name, "lam") not in kept:                     # beyond --n-val
            eig = eigenbasis(arch)
            ref = reference(arch, op, eig)
            kept[(name, "lam")], kept[(name, "ref")] = eig["lam"], ref
            for row in ROWS:
                U = predict(models[(row, s0)], arch, dev)
                with ieee_fp32(dev, tf32):
                    U_ieee = predict(models[(row, s0)], arch, dev)
                kept[(name, row)] = (U, diagnostics(U, arch, op, ref, eig), U_ieee)
        ref = kept[(name, "ref")]
        payload = {"nodes": np.asarray(arch.nodes, np.float64),
                   "elements": np.asarray(arch.elements, np.int64),
                   "dirichlet_mask": np.asarray(arch.dirichlet_mask, bool),
                   "F": np.asarray(arch.F, np.float64),
                   "U_star": np.asarray(arch.U_star, np.float64),
                   "lam": kept[(name, "lam")], "C2_star": ref["C2"],
                   "pi_star": ref["pi_star"], "u2_star": ref["u2_star"],
                   "uK_star": ref["uK_star"], "rq_star": ref["uK_star"] / ref["u2_star"],
                   "gamma_star": ref["gamma_star"], "log_edges": LOG_EDGES}
        if op is not None:
            payload["area"] = op["area"]
            payload["vm_ref"] = ref["vm"].astype(np.float32)
        for row in ROWS:
            U, dg, U_ieee = kept[(name, row)]
            payload.update({f"U_{row}": np.asarray(U, np.float64), f"C2_{row}": dg["C2"],
                            f"U_ieee_{row}": np.asarray(U_ieee, np.float64),
                            f"pi_{row}": dg["pi"], f"rel_{row}": dg["rel"], f"c_{row}": dg["c"],
                            f"rel_c_{row}": dg["rel_c"]})
            if op is not None:
                payload[f"vm_{row}"] = vm_p(tri_stress(op, U))[0].astype(np.float32)
        np.savez_compressed(out_dir / f"{name}.npz", **payload,
                            meta=np.array(json.dumps({"figure": name, "rule": RULES[name],
                                                      "set": "val", "index": rule["index"],
                                                      "file": Path(f).name, "seed": s0,
                                                      "material": arch.meta["material"]})))
        figs[name] = dict(rule, n_nodes=int(arch.n_nodes),
                          n_elements=int(arch.elements.shape[0]),
                          n_free=int(kept[(name, "lam")].size))
        print(f"[spectra] {name}: val #{rule['index']} {Path(f).name} ({arch.n_nodes} nodes)",
              flush=True)
    res["figures"] = figs
    res["seconds"] = round(time.time() - t0, 1)
    write_json(out_dir / "spectra.json", res)
    print(json.dumps({"content_median_rel_dev": {
                          k: [c["disp_rel_l2_median_rel_dev"], c["energy_gap_rel_median_rel_dev"]]
                          for k, c in checks.items()},
                      "content_mismatch": mismatch,
                      "rayleigh_ratio_median": {r: (v or {}).get("rayleigh_ratio_median")
                                                for r, v in diag["rows"].items()},
                      "pairs": {k: (v or {}).get("share_first_above")
                                for k, v in diag["pairs"].items()},
                      "prop1_bound_ratio_max": {r: (v or {}).get("prop1_bound_ratio_max")
                                                for r, v in diag["rows"].items()},
                      "figures": {k: (v["file"], v["index"]) for k, v in figs.items()}}),
          flush=True)
    if mismatch:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
