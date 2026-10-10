"""Pre-probe for stage C1.3 (exploratory, not pre-registered): how do feasibility parametrisations of the
contact unknowns behave when a field is fitted by minimising the total potential energy with Adam?

Not a network: each unknown is optimised directly, so this isolates the parametrisation from shared weights.
Problem: the hex8i T-stub of tests/test_contact.py (flange on a rigid support, two bolt springs, pulled at
the centre). Parametrisations of the constrained (contact) unknowns v_c = phi(z_c):
  relu      phi(z) = max(0, z)                       (dead where z < 0: zero gradient)
  softplus  phi(z) = s * log(1 + exp(z / s))         (s = 1e-3 of the solution's largest displacement)
  square    phi(z) = z^2 / scale                     (gradient vanishes at contact)
  project   v_c clipped to >= 0 after every step     (projected Adam, the reference)
  free      no constraint                            (penetration allowed: shows why feasibility is needed)
Readings: relative energy gap, relative energy-norm error, contact-set agreement, dead fraction.
"""
import sys, json, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
from fejoint import hex8i as H
from fejoint.contact import solve_signorini


def tstub():
    L, t, d = 120.0, 10.0, 30.0
    nodes, hexes = H.box_mesh(np.linspace(0, L, 49), np.linspace(0, t, 3), np.linspace(0, d, 13))
    K = H.assemble(nodes, hexes, 210000.0, 0.3)
    x, y, z = nodes.T; n = nodes.shape[0]
    top = np.nonzero(np.isclose(y, t))[0]; bottom = np.nonzero(np.isclose(y, 0))[0]
    springs = np.zeros(3 * n)
    for xb in (L / 2 - 35, L / 2 + 35):
        patch = top[np.hypot(x[top] - xb, z[top] - 15) <= 7.5]
        springs[3 * patch + 1] += 210000.0 * 157.0 / 45.0 / patch.size
    K = (K + sp.diags(springs)).tocsr()
    strip = top[np.abs(x[top] - L / 2) <= 5.0 + 1e-9]
    f = np.zeros(3 * n); f[3 * strip + 1] = 1000.0 / strip.size
    fixed = np.zeros(3 * n, bool)
    fixed[3 * np.nonzero(np.isclose(x, L / 2))[0]] = True
    fixed[3 * np.nonzero(np.isclose(z, 0))[0] + 2] = True
    free = np.nonzero(~fixed)[0]
    A = K[free][:, free].tocsr(); b = f[free]
    pos = -np.ones(3 * n, int); pos[free] = np.arange(free.size)
    return A, b, pos[3 * bottom + 1]


def run(A, b, cidx, ustar, kind, iters, lr, seed=0):
    rng = np.random.default_rng(seed)
    n = b.size; scale = np.abs(ustar).max()
    Pi = lambda v: 0.5 * v @ (A @ v) - b @ v  # noqa: E731
    Pstar = Pi(ustar); estar = np.sqrt(ustar @ (A @ ustar))
    z = 0.01 * scale * rng.standard_normal(n)
    if kind == "square":
        z[cidx] = np.sqrt(np.abs(z[cidx]) * scale)
    m = np.zeros(n); v2 = np.zeros(n); b1, b2, eps = 0.9, 0.999, 1e-30
    s = 1e-3 * scale
    def field(z):
        v = z.copy(); dv = np.ones(n)
        zc = z[cidx]
        if kind == "relu":
            v[cidx] = np.maximum(0, zc); dv[cidx] = (zc > 0).astype(float)
        elif kind == "softplus":
            v[cidx] = s * np.logaddexp(0, zc / s); dv[cidx] = 1 / (1 + np.exp(-zc / s))
        elif kind == "square":
            v[cidx] = zc ** 2 / scale; dv[cidx] = 2 * zc / scale
        return v, dv
    hist = []
    for k in range(1, iters + 1):
        v, dv = field(z)
        g = (A @ v - b) * dv
        m = b1 * m + (1 - b1) * g; v2 = b2 * v2 + (1 - b2) * g * g
        z -= lr * scale * (m / (1 - b1 ** k)) / (np.sqrt(v2 / (1 - b2 ** k)) + eps)
        if kind == "project":
            z[cidx] = np.maximum(z[cidx], 0)
        if k in (100, 1000, 3000, 10000, 30000, iters):
            v, _ = field(z)
            e = v - ustar
            on_true = ustar[cidx] <= 1e-9 * scale
            on_v = v[cidx] <= 1e-3 * scale
            hist.append({"iter": k, "rel_gap": float((Pi(v) - Pstar) / abs(Pstar)),
                         "rel_err_energy": float(np.sqrt(e @ (A @ e)) / estar),
                         "penetration_max_rel": float(max(0.0, -v[cidx].min()) / scale),
                         "contact_set_agreement": float((on_true == on_v).mean()),
                         "dead_fraction": float(((z[cidx] <= 0) & ~on_true).mean()) if kind == "relu" else None})
    return hist


if __name__ == "__main__":
    A, b, cidx = tstub()
    sol = solve_signorini(A, b, cidx)
    ustar = sol["x"]
    out = {"n": int(b.size), "n_contact": int(cidx.size), "active_fraction": float(sol["active"].mean()), "runs": {}}
    for kind in ("project", "relu", "softplus", "square", "free"):
        for lr in (1e-3, 1e-2):
            t0 = time.perf_counter()
            h = run(A, b, cidx, ustar, kind, 30000, lr)
            out["runs"][f"{kind}_lr{lr:g}"] = h
            last = h[-1]
            print(f"{kind:9s} lr {lr:<6g} gap {last['rel_gap']:+.2e} err {last['rel_err_energy']:.2e} "
                  f"pen {last['penetration_max_rel']:.1e} agree {last['contact_set_agreement']:.3f} dead {last['dead_fraction']} "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    json.dump(out, open(pathlib.Path(__file__).with_suffix(".json"), "w"), indent=1)
