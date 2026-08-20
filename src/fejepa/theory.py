"""WP6 -- the theory note's internal falsification pass, made executable.

Plan v2.0 mapping: WP6's acceptance is "3-5 pages surviving an internal
falsification pass". The note itself is LaTeX; every claim with numeric content is
verified here on real instances (GPU-free, numpy/scipy only), so the pass is a
command (``fejepa theory``) and a report block, not a promise:

  C5(b) conditioning lemma
      ||u - U*||_2 <= sqrt(2 * gap(u) / lambda_min(K_ff))     (checked; tightness
      reported), and gradient flow on Pi_h contracts eigencomponent i at exactly
      the rate (1 - eta * lambda_i) per step (checked to machine precision --
      the numeric heart of "displacement is the wrong primary metric").

  C5(c) Chebyshev polish bound
      gap_k / gap_0 <= 4 * rho^(2k),  rho = (sqrt(kappa)-1)/(sqrt(kappa)+1),
      for k CG steps (checked against measured cg_k_steps ratios).

  C5(d) Proposition 1 scoping
      * within-geometry premise: the physics targets ||dU*||_K across the load
        battery are well separated (what E6's rho relies on);
      * the NAIVE cross-geometry extension is refuted by explicit numeric
        counterexample search: two different geometries under the same load can be
        closer (direction metric, interpolated to a common frame) than two loads on
        one geometry -- so only the descriptor-conditioned form (E6's cross
        variant) is tenable.

A violated inequality trips the C5 kill (theory wrong, or a code bug -- either way
the note may not ship). The counterexample being FOUND is the expected outcome and
is not a kill: it selects the scoped statement.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .anchor.energy import energy_gap
from .fe.solve import cg_k_steps
from .models.features import geometry_descriptor, normalized_coords
from .experiments.protocol import kill, result

PLAN_REF = "plan v2.0 WP6, C5(b)(c)(d); E6 is Prop.1's empirical face"


# ------------------------------------------------------------- spectral core --

def lambda_extremes(K: sp.spmatrix, free_mask: np.ndarray) -> dict:
    """(lambda_min, lambda_max, kappa) of the constrained operator K_ff (SPD)."""
    free = np.asarray(free_mask, dtype=bool)
    Kff = sp.csr_matrix(K)[free][:, free].tocsc()
    lo = float(spla.eigsh(Kff, k=1, sigma=0, which="LM",
                          return_eigenvectors=False)[0])
    hi = float(spla.eigsh(Kff, k=1, which="LM",
                          return_eigenvectors=False)[0])
    return {"lambda_min": lo, "lambda_max": hi, "kappa": hi / lo}


# --------------------------------------------------- C5(b): conditioning lemma --

def check_conditioning_lemma(arch, n_samples: int = 8,
                             rng: np.random.Generator | None = None,
                             spec: dict | None = None) -> dict:
    """||e||_2 <= sqrt(2 gap / lambda_min) for random perturbations e of U*.

    ``spec``: optional precomputed :func:`lambda_extremes` (shared with the
    Chebyshev check to factor K_ff for eigsh only once per instance)."""
    rng = rng or np.random.default_rng(0)
    spec = spec or lambda_extremes(arch.K, arch.free_mask)
    ratios = []
    for _ in range(n_samples):
        j = int(rng.integers(0, arch.n_loads))
        e = rng.standard_normal(arch.ndof) * arch.free_mask
        e *= rng.uniform(0.01, 1.0) * np.linalg.norm(arch.U_star[j]) \
            / (np.linalg.norm(e) + 1e-30)
        u = arch.U_star[j] + e
        gap = float(energy_gap(u, arch.U_star[j], arch.K, arch.F[j])[0])
        bound = np.sqrt(2.0 * max(gap, 0.0) / spec["lambda_min"])
        ratios.append(np.linalg.norm(e) / (bound + 1e-300))
    max_ratio = float(np.max(ratios))
    return {"holds": bool(max_ratio <= 1.0 + 1e-8), "max_ratio": max_ratio,
            "median_tightness": float(np.median(ratios)), **spec}


def check_mode_contraction(arch, n_modes: int = 2) -> dict:
    """One gradient step on Pi_h with rate eta contracts eigencomponent i by
    exactly (1 - eta * lambda_i): the 'rate proportional to lambda_i' claim."""
    free = arch.free_mask
    Kff = sp.csr_matrix(arch.K)[free][:, free].tocsc()
    lo_w, lo_v = spla.eigsh(Kff, k=n_modes, sigma=0, which="LM")
    hi_w, hi_v = spla.eigsh(Kff, k=1, which="LM")
    eta = 0.5 / float(hi_w[0])
    errs = []
    for w, v in [*zip(lo_w, lo_v.T, strict=True), (hi_w[0], hi_v[:, 0])]:
        stepped = v - eta * (Kff @ v)
        coeff = float(v @ stepped)
        errs.append(max(abs(coeff - (1.0 - eta * float(w))),
                        float(np.linalg.norm(stepped - coeff * v))))
    max_err = float(np.max(errs))
    return {"holds": bool(max_err < 1e-9), "max_err": max_err, "eta": eta,
            "checked_lambdas": [float(w) for w in lo_w] + [float(hi_w[0])]}


# ------------------------------------------------ C5(c): Chebyshev polish bound --

def check_chebyshev_polish(arch, ks=(1, 3, 5, 10), n_inits: int = 4,
                           rng: np.random.Generator | None = None,
                           spec: dict | None = None) -> dict:
    """Measured gap_k / gap_0 from cg_k_steps vs the bound 4 rho^(2k)."""
    rng = rng or np.random.default_rng(1)
    spec = spec or lambda_extremes(arch.K, arch.free_mask)
    rho = (np.sqrt(spec["kappa"]) - 1.0) / (np.sqrt(spec["kappa"]) + 1.0)
    worst = 0.0
    for _ in range(n_inits):
        j = int(rng.integers(0, arch.n_loads))
        e0 = rng.standard_normal(arch.ndof) * arch.free_mask
        u0 = arch.U_star[j] + e0
        gap0 = float(energy_gap(u0, arch.U_star[j], arch.K, arch.F[j])[0])
        for k in ks:
            uk = cg_k_steps(arch.K, arch.F[j], arch.free_mask, u0, int(k))
            gapk = float(energy_gap(uk, arch.U_star[j], arch.K, arch.F[j])[0])
            bound = 4.0 * rho ** (2 * int(k))
            worst = max(worst, (gapk / gap0) / bound)
    return {"holds": bool(worst <= 1.0 + 1e-8),
            "worst_measured_over_bound": float(worst),
            "rho": float(rho), **spec}


# ------------------------------------------- C5(d): Proposition 1 scoping ------

def prop1_within_geometry_premise(archs) -> dict:
    """The separation E6 relies on: min pairwise ||dU*||_K across the battery,
    normalized by the mean load K-norm, per geometry."""
    mins = []
    for a in archs:
        norms = [float(np.sqrt(max(0.0, a.U_star[i] @ (a.K @ a.U_star[i]))))
                 for i in range(a.n_loads)]
        scale = float(np.mean(norms)) + 1e-30
        seps = [float(np.sqrt(max(0.0, d @ (a.K @ d)))) / scale
                for i in range(a.n_loads) for k in range(i + 1, a.n_loads)
                for d in [a.U_star[i] - a.U_star[k]]]
        mins.append(min(seps))
    return {"min_separation": float(np.min(mins)),
            "median_separation": float(np.median(mins)),
            "n_geometries": len(archs)}


def _direction_field(arch, j: int) -> np.ndarray:
    f = arch.U_star[j] * arch.free_mask
    return f / (np.linalg.norm(f) + 1e-30)


def _interp_to(src_arch, field_src: np.ndarray, dst_arch) -> np.ndarray:
    """Transport a node-major field between meshes via normalized coordinates
    (linear inside the hull, nearest fill outside) -- the only honest common
    frame; the K-norm is not even defined across meshes, which is itself half
    of the counterexample."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    pts, xq = normalized_coords(src_arch.nodes), normalized_coords(dst_arch.nodes)
    sd = int(src_arch.nodes.shape[1])                      # WP7 3D-P0
    vals = field_src.reshape(-1, sd)
    out = np.zeros((dst_arch.n_nodes, sd))
    for c in range(sd):
        y = LinearNDInterpolator(pts, vals[:, c])(xq)
        bad = np.isnan(y)
        if bad.any():
            y[bad] = NearestNDInterpolator(pts, vals[:, c])(xq[bad])
        out[:, c] = y
    return out.reshape(-1)


def prop1_cross_geometry_counterexample(archs, load_idx: int = 0) -> dict:
    """Search for (A, B) with different geometry descriptors whose same-load
    solution directions are closer than A's own closest load pair -- refuting
    any unconditioned cross-geometry alignment claim."""
    within = []
    for a in archs:
        d = [np.linalg.norm(_direction_field(a, i) - _direction_field(a, k))
             for i in range(a.n_loads) for k in range(i + 1, a.n_loads)]
        within.append(float(np.min(d)))
    best = None
    for ia, a in enumerate(archs):
        fa = _direction_field(a, load_idx)
        for ib, b in enumerate(archs):
            if ib == ia:
                continue
            if np.allclose(geometry_descriptor(a.meta),
                           geometry_descriptor(b.meta)):
                continue
            fb = _interp_to(b, _direction_field(b, load_idx), a)
            fb = fb / (np.linalg.norm(fb) + 1e-30)
            d_cross = float(np.linalg.norm(fa - fb))
            if d_cross < within[ia] and (best is None or d_cross < best["d_cross"]):
                best = {"a_index": ia, "b_index": ib, "d_cross": d_cross,
                        "a_within_min": within[ia]}
    return {"naive_extension_falsified": best is not None, "witness": best,
            "metric": "unit-norm direction L2, fields interpolated in "
                      "normalized coordinates (cross-mesh K-norm is undefined)"}


# ----------------------------------------------------------------- aggregate --

def run_theory_checks(archs, cfg: dict | None = None) -> dict:
    """The WP6 falsification pass over a labelled probe set (val subset)."""
    cfg = cfg or {}
    n = min(int(cfg.get("n_check", 8)), len(archs))
    probe = archs[:n]
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    specs = [lambda_extremes(a.K, a.free_mask) for a in probe]
    cond = [check_conditioning_lemma(a, rng=rng, spec=s)
            for a, s in zip(probe, specs, strict=True)]
    mode = [check_mode_contraction(a) for a in probe]
    cheb = [check_chebyshev_polish(a, rng=rng, spec=s)
            for a, s in zip(probe, specs, strict=True)]
    metrics = {
        "conditioning": {"holds": all(c["holds"] for c in cond),
                         "max_ratio": max(c["max_ratio"] for c in cond),
                         "kappa_range": [min(c["kappa"] for c in cond),
                                         max(c["kappa"] for c in cond)]},
        "mode_contraction": {"holds": all(m["holds"] for m in mode),
                             "max_err": max(m["max_err"] for m in mode)},
        "chebyshev_polish": {"holds": all(c["holds"] for c in cheb),
                             "worst_measured_over_bound":
                                 max(c["worst_measured_over_bound"]
                                     for c in cheb)},
        "prop1_premise": prop1_within_geometry_premise(probe),
        "prop1_naive_cross": prop1_cross_geometry_counterexample(probe),
    }
    all_hold = all(metrics[k]["holds"] for k in
                   ("conditioning", "mode_contraction", "chebyshev_polish"))
    k = kill("C5 numeric falsification pass: any verified inequality violated",
             triggered=bool(not all_hold),
             note=("all inequalities hold" if all_hold else
                   "violation -- theory wrong or code bug; note must not ship"))
    proto = {"n_check": n, "seed": int(cfg.get("seed", 0)),
             "gpu_free": True,
             "note": "the LaTeX note cites these numbers; the counterexample "
                     "being found is expected and selects the scoped Prop.1"}
    return result("WP6-theory", PLAN_REF, proto, metrics, [k])
