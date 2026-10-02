"""wp9 Stage 0a: zero-training readings (C0) of trained 2D states.

Reported only; the session-1 decision rules that read some of them are code
(`scripts/w9_session1_decisions.py`) committed before the session.

Notation (one instance, one load case; free dofs only): K the stiffness, F
the load, U* = K^{-1} F, Pi(u) = 0.5 u^T K u - F^T u, gap(u) = Pi(u) - Pi(U*)
= 0.5 ||u - U*||_K^2, and the relative gap gap(u) / |Pi(U*)| =
||u - U*||_K^2 / ||U*||_K^2 (the records' `energy_gap_rel`).

* Ensemble decomposition (C0.1). For seed predictions u_s, s = 1..S, and
  their mean u_bar, the parallel-axis identity in the K inner product gives
      mean_s ||u_s - U*||_K^2 = ||u_bar - U*||_K^2 + mean_s ||u_s - u_bar||_K^2,
  so the seed-mean relative gap equals the ensemble's plus the disagreement
  D = mean_s ||u_s - u_bar||_K^2 / ||U*||_K^2 >= 0 (exact; the residual is
  reported). The unnormalised disagreement needs no label.
* Energy drop of k CG steps (C0.3). From x_0 = u, k steps of unpreconditioned
  CG on K_ff x = F_f give Delta_k = Pi(x_0) - Pi(x_k) = gap(x_0) - gap(x_k), a
  label-free lower bound on gap(u) (Strakos and Tichy 2002); Delta_k /
  |Pi(x_k)| is its relative, label-free counterpart: with G = gap(u),
  G_k = gap(x_k) and P = |Pi(U*)|, it is (G - G_k) / |P - G_k|, a lower bound
  of the relative gap G / P whenever G <= P, i.e. Pi(u) <= 0 (u no worse than
  u = 0); beyond that it can exceed it. Session 1 reads it through its rank
  correlation with the true gap, which needs no such bound.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------- basics ---

def free_parts(arch):
    """(free mask, K_ff (CSR), F_f (L, nfree), U*_f or None)."""
    free = np.asarray(arch.free_mask, dtype=bool).ravel()
    Kff = sp.csr_matrix(arch.K)[free][:, free]
    F = np.atleast_2d(np.asarray(arch.F, dtype=np.float64))[:, free]
    Us = (None if arch.U_star is None
          else np.atleast_2d(np.asarray(arch.U_star, dtype=np.float64))[:, free])
    return free, Kff, F, Us


def k_norm2(Kff, X) -> np.ndarray:
    """Per-row ||x||_K^2 of an (L, nfree) array."""
    X = np.atleast_2d(X)
    return np.einsum("ld,ld->l", X, (Kff @ X.T).T)


def energy(Kff, F, X) -> np.ndarray:
    """Per-row Pi(x) = 0.5 x^T K x - F^T x."""
    X = np.atleast_2d(X)
    return 0.5 * k_norm2(Kff, X) - np.einsum("ld,ld->l", np.atleast_2d(F), X)


def predictions(models: dict, arch, device) -> dict:
    """{seed: U (L, ndof) float64}. One prepared pack serves every seed: the
    pack depends on the configuration and the instance, not on the weights
    (wp8 Stage 1.37)."""
    import torch

    def _prep_key(m):                    # what prepare_instance depends on (wp9 S)
        c = getattr(m, "cfg", None)
        return (getattr(c, "decode_scale", "max"), getattr(c, "decode_scale_factor", 1.0),
                repr(getattr(c, "features", None)))

    if len({_prep_key(m) for m in models.values()}) > 1:
        raise ValueError("predictions: the models prepare instances differently (decode "
                         "scale or features); one shared pack would mis-scale some of them")
    pack = next(iter(models.values())).prepare_instance(arch, device)
    out = {}
    for s, m in models.items():
        with torch.no_grad():
            out[s] = m.forward_instance(pack).detach().cpu().numpy().astype(np.float64)
    return out


# ------------------------------------------------------------ C0.1 ------

def ensemble_row(arch, preds: dict) -> dict:
    """One labelled instance: per load case, the seed relative gaps, the
    ensemble's, the disagreement D and the identity residual; instance values
    are the means over load cases (as the records' per-instance arrays)."""
    free, Kff, _, Us = free_parts(arch)
    U = np.stack([np.atleast_2d(preds[s])[:, free] for s in sorted(preds)])   # (S, L, nf)
    ubar = U.mean(axis=0)
    ref = k_norm2(Kff, Us)
    g_seed = np.stack([k_norm2(Kff, U[i] - Us) / ref for i in range(U.shape[0])])   # (S, L)
    g_ens = k_norm2(Kff, ubar - Us) / ref
    d_abs = np.mean([k_norm2(Kff, U[i] - ubar) for i in range(U.shape[0])], axis=0)
    D = d_abs / ref
    resid = g_seed.mean(axis=0) - g_ens - D
    pi_bar = energy(Kff, np.atleast_2d(arch.F)[:, free], ubar)
    return {"gap_seed": g_seed.mean(axis=1).tolist(), "gap_seed_mean": float(g_seed.mean()),
            "gap_ensemble": float(g_ens.mean()), "D": float(D.mean()),
            # label-free disagreement, normalised by the ensemble's own |Pi|
            "D_labelfree": float(np.mean(0.5 * d_abs / np.maximum(np.abs(pi_bar), 1e-300))),
            "identity_residual": float(np.max(np.abs(resid)))}


# ------------------------------------------------------------ C0.3 ------

def cg_energy_drops(Kff, b: np.ndarray, x0: np.ndarray, ks) -> dict:
    """{k: Pi(x_0) - Pi(x_k)} for unpreconditioned CG from x0 on Kff x = b,
    read at the iterations in `ks` (one run to max(ks); an exact solve before
    k leaves the later drops at the full gap)."""
    ks = sorted(int(k) for k in ks)
    x = np.array(x0, dtype=np.float64)
    r = b - Kff @ x
    p = r.copy()
    rr = float(r @ r)
    e0 = 0.5 * float(x @ (Kff @ x)) - float(b @ x)
    out, k_done = {}, 0
    bnorm = float(np.linalg.norm(b)) or 1.0
    for k in ks:
        while k_done < k and np.sqrt(rr) > 1e-15 * bnorm:
            Ap = Kff @ p
            alpha = rr / float(p @ Ap)
            x += alpha * p
            r -= alpha * Ap
            rr_new = float(r @ r)
            p = r + (rr_new / rr) * p
            rr = rr_new
            k_done += 1
        out[k] = e0 - (0.5 * float(x @ (Kff @ x)) - float(b @ x))
    return out


def gap_estimate_row(arch, U: np.ndarray, ks) -> dict:
    """One labelled instance, one seed: the true relative gap and, per k, the
    label-free relative estimate Delta_k / |Pi(x_k)| and the captured fraction
    Delta_k / gap (instance values: means over load cases)."""
    free, Kff, F, Us = free_parts(arch)
    X0 = np.atleast_2d(U)[:, free]
    gap = 0.5 * k_norm2(Kff, X0 - Us)
    pi_star = np.abs(energy(Kff, F, Us))
    est, frac = {k: [] for k in ks}, {k: [] for k in ks}
    pi0 = energy(Kff, F, X0)
    for li in range(X0.shape[0]):
        drops = cg_energy_drops(Kff, F[li], X0[li], ks)
        for k in ks:
            pik = pi0[li] - drops[k]
            est[k].append(drops[k] / max(abs(pik), 1e-300))
            frac[k].append(drops[k] / max(gap[li], 1e-300))
    return {"gap_rel": float(np.mean(gap / pi_star)),
            "est_rel": {str(k): float(np.mean(est[k])) for k in ks},
            "captured": {str(k): float(np.mean(frac[k])) for k in ks}}


# ------------------------------------------------------- amplitude ------

def amplitude_row(arch, U: np.ndarray) -> dict:
    """c* per load case, the battery form c_b, energy norms, and the errors
    before and after scaling by c* (as the wp8 post-hoc amplitude reading)."""
    from ..metrics import displacement_errors, energy_gap_rel
    from ..models.features import battery_fscale
    from .posthoc import (amplitude_factor, apply_amplitude, battery_amplitude,
                          energy_norms, free_block, free_mask)

    free = free_mask(arch)
    Kf = free_block(arch.K, free)
    c = amplitude_factor(U, arch.K, arch.F, free, Kf=Kf)
    Uc = apply_amplitude(U, c)
    ex = arch.meta.get("extra") or {}
    row = {"n_nodes": int(arch.n_nodes), "target_h": float(ex.get("target_h", np.nan)),
           "fscale": float(battery_fscale(arch.F)), "c_star": c.tolist(),
           "c_battery": float(battery_amplitude(U, arch.K, arch.F, free, Kf=Kf)),
           "u_norm_K": energy_norms(U, arch.K, free, Kf=Kf).tolist()}
    if arch.U_star is not None:
        row.update({"ustar_norm_K": energy_norms(arch.U_star, arch.K, free, Kf=Kf).tolist(),
                    "disp": float(displacement_errors(U, arch).mean()),
                    "disp_c": float(displacement_errors(Uc, arch).mean()),
                    "egap": float(energy_gap_rel(U, arch).mean()),
                    "egap_c": float(energy_gap_rel(Uc, arch).mean())})
    return row


def amplitude_summary(rows: list) -> dict:
    cs = np.array([r["c_star"] for r in rows], dtype=float)
    out = {"n_instances": len(rows), "c_star_median": float(np.nanmedian(cs)),
           "frac_c_star_gt_1": float(np.mean(cs[np.isfinite(cs)] > 1.0)),
           "c_battery_median": float(np.nanmedian([r["c_battery"] for r in rows]))}
    for k in ("disp", "disp_c", "egap", "egap_c"):
        if rows and k in rows[0]:
            v = np.array([r[k] for r in rows], dtype=float)
            out[k] = float(v.mean())
            out[k + "_median"] = float(np.median(v))
    return out


# --------------------------------------------------------------- stats --

def spearman(x, y) -> float:
    """Tie-aware Spearman rank correlation; NaN with fewer than three pairs or
    a constant variable."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(x[ok]), rankdata(y[ok]))[0, 1])
