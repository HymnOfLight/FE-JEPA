"""Constant-strain-triangle stress/strain recovery.

Plan v2.0 mapping: Sec.2.1 (verified asset) and Sec.4 (primary metrics): the von-Mises
suite and critical-region recall are computed from this recovery, which uses the *same*
Lame parameters as assembly so that the recovered strain energy equals 0.5 u^T K u
exactly (audit V1: agreement to 1.9e-12; preserved by test_stress_energy.py).
"""

from __future__ import annotations

import numpy as np


def lame(E: float, nu: float, plane: str) -> tuple[float, float]:
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    if plane == "stress":
        lam = 2.0 * lam * mu / (lam + 2.0 * mu)
    return lam, mu


def _geometry(nodes: np.ndarray, elements: np.ndarray):
    p = nodes[elements]                                    # (E, 3, 2)
    x1, y1 = p[:, 0, 0], p[:, 0, 1]
    x2, y2 = p[:, 1, 0], p[:, 1, 1]
    x3, y3 = p[:, 2, 0], p[:, 2, 1]
    det = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    area = 0.5 * np.abs(det)
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / det[:, None]
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / det[:, None]
    return area, b, c


def element_strains(nodes: np.ndarray, elements: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Per-element engineering strains (eps_xx, eps_yy, gamma_xy). u is node-major (ndof,)."""
    _, b, c = _geometry(nodes, elements)
    dof = np.empty((elements.shape[0], 6), dtype=np.int64)
    dof[:, 0::2] = 2 * elements
    dof[:, 1::2] = 2 * elements + 1
    ue = u[dof].reshape(-1, 3, 2)                          # (E, 3, 2)
    exx = np.einsum("ei,ei->e", b, ue[:, :, 0])
    eyy = np.einsum("ei,ei->e", c, ue[:, :, 1])
    gxy = np.einsum("ei,ei->e", c, ue[:, :, 0]) + np.einsum("ei,ei->e", b, ue[:, :, 1])
    return np.stack([exx, eyy, gxy], axis=1)


def element_stresses(nodes, elements, u, material: dict) -> np.ndarray:
    eps = element_strains(nodes, elements, u)
    lam, mu = lame(material["E"], material["nu"], material["plane"])
    tr = eps[:, 0] + eps[:, 1]
    sxx = lam * tr + 2.0 * mu * eps[:, 0]
    syy = lam * tr + 2.0 * mu * eps[:, 1]
    sxy = mu * eps[:, 2]
    return np.stack([sxx, syy, sxy], axis=1)


def element_von_mises(nodes, elements, u, material: dict) -> np.ndarray:
    s = element_stresses(nodes, elements, u, material)
    sxx, syy, sxy = s[:, 0], s[:, 1], s[:, 2]
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def strain_energy(nodes, elements, u, material: dict) -> float:
    """Total strain energy from the recovered fields; equals 0.5 u^T K u (tested)."""
    area, _, _ = _geometry(nodes, elements)
    eps = element_strains(nodes, elements, u)
    sig = element_stresses(nodes, elements, u, material)
    dens = 0.5 * (sig[:, 0] * eps[:, 0] + sig[:, 1] * eps[:, 1] + sig[:, 2] * eps[:, 2])
    return float(np.sum(dens * area))
