#!/usr/bin/env python3
"""wp9 Stage 0e and cmame-paper Stage 1: the field figures, the per-load
energies and the error diagnostics of the CMAME paper, from Phase-2b's own
states on the run's own instances. Reported only; no verdict reads it, and
nothing in PREREG_W9 does.

Models (in `--states-dir`, the run's state directory), all at the largest
label budget and the run's pool:

* `ar` (label-free, trained on the discrete energy): `ar_p{pool}_s{seed}.pt`,
  SHA-256 checked against the report's d9_restart record and refused
  otherwise (as in Stage 0d);
* `labels` (labels only, the same network) and `mgn` (the graph network):
  `labels_b{b}_s{seed}.pt` and `mgn_b{b}_s{seed}.pt`, the P3 shared
  checkpoints. No report records their hashes; `--sup-hashes` does (the
  listing of the box's state directory returned on 22 September 2026,
  before Phase-2b, which reused these units).

Every model is also checked by content: its per-instance displacement error
and relative energy gap on the validation split, recomputed here with the
evaluation's own functions, against the report's per-instance arrays for
that model and seed (expected: 0 or round-off for the transformer states,
about 1e-4 for the graph network, whose CUDA scatter reductions are not
bitwise reproducible). A supervised state is used if its hash matches or its
content does (median relative deviation of both quantities at most
CONTENT_TOL); a state that fails both is left out and recorded, and the
script ends with exit status 4 after writing everything else. A state used
on its hash whose content does not match is recorded as a content mismatch
(exit status 5, unless 4 applies). The figure instances are resolved and
checked (present, labelled) before any model runs.

Outputs (`--out`, a directory):

* `energies_val.npz`: per model, seed, validation instance and load case,
  Pi_h(u), the relative energy gap (Pi_h(u) - Pi_h(U*)) / |Pi_h(U*)| (the
  zero field has 1), the energy-optimal amplitude c* = F^T u / u^T K u and
  the relative gap of c* u; Pi_h(U*) per instance and load case. Per load
  case, Pi_h(u) > 0 exactly when u is further from U* in the energy norm
  than the zero field, and Pi_h(c* u) <= min(Pi_h(u), 0).
  cmame-paper Stage 1 adds, per model, seed, validation instance and load
  case, with e = u - U*: the squared Euclidean and stiffness norms of the
  error, ||e||_2^2 and ||e||_K^2 = e^T K e (their ratio is the error's
  Rayleigh quotient, its weight in stiff modes); the relative von Mises
  error over the element values, as the evaluation computes it, and weighted
  by element volume, the L^2 norm of Proposition 1; and the stress-energy
  integral of e, the sum over elements of vol (s_vm(e)^2 / (3G) + p(e)^2 / B),
  equal to ||e||_K^2 by Proposition 1 (a check). Per instance and load case
  of U*: ||U*||_2^2, ||U*||_K^2 and gamma* = 3G ||p(U*)||^2 / (B ||s_vm(U*)||^2)
  (volume-weighted), so that Proposition 1's bound, vm_vol^2 <= (1 + gamma*) g,
  can be read on every prediction; and the largest absolute prediction on a
  constrained dof (0: K is stored unconstrained, and e^T K e is the free
  block's value only because the models zero those dofs). P1 tetrahedra only
  (the run is 3D).
* `fig5.npz`, `fig6.npz`, `fig7.npz`: the instances chosen by RULES, with
  the mesh, the load battery, the reference U*, each model's seed-0
  prediction, element von Mises stresses (reference and models; von Mises
  is positively homogeneous, so that of c* u is |c*| times that of u), the
  element volumes and the per-load energies and c*.
* `fields.json`: inputs and their hashes, the checks, the selections (index,
  file, and the choice of every seed for the rules that read seed 0), the
  counts of load cases and instances worse than the zero field, the
  diagnostics' summaries (medians, the largest ratio to Proposition 1's
  bound, the two identities' largest deviations, and how often a supervised
  model's error has the larger Rayleigh quotient than the label-free one's
  on the same instance, load case and seed), timings.

    python scripts/export_fields.py --report records/wp8/e2/baseline/report_phase2b.json \\
        --states-dir runs/phase2/e8_states --out runs/cmame/timing/fields
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ARMS = ("ar", "labels", "mgn")
CONTENT_TOL = 1e-3
"""A supervised state whose hash differs is accepted when the median relative
deviation of its recomputed per-instance displacement error and relative
energy gap from the report's arrays are both at most this."""
RULES = {
    "fig5": "the validation instance with the largest relative energy gap of the labels-only "
            "arm at the largest budget, seed 0, in the report's per-instance arrays",
    "fig6": "the validation instance at 0-based rank (n - 1) // 2 of the label-free arm's "
            "relative energy gaps, seed 0, in ascending order, in the report's arrays",
    "fig7": "the fine-set instance at 0-based rank (n - 1) // 2 of the label-free arm's "
            "displacement errors, seed 0, in ascending order, in the report's P3 arrays",
}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUP_HASHES = ROOT / "records" / "wp9" / "phase2_supervised_states.json"


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budgets(report: dict) -> tuple:
    """(largest label budget, AR pool) as the report's cells name them."""
    cells = report["results"]["e8"]["metrics"]["cells"]
    return (str(max(int(b) for b in cells["labels"])),
            str(int(report["config"]["experiments"]["e8"]["pool_sizes"][0])))


def report_arrays(report: dict, arm: str, seed_index: int) -> dict:
    """The report's per-instance validation arrays of one model and seed."""
    cells = report["results"]["e8"]["metrics"]["cells"]
    b, pool = budgets(report)
    cell = cells[arm][pool if arm == "ar" else b]
    return cell["per_seed_eval"][seed_index]["per_instance"]


def select(report: dict) -> dict:
    """The three instances of RULES, from the report's own arrays."""
    seeds = list(range(len(report["provenance"]["seeds"])))
    lab = [np.asarray(report_arrays(report, "labels", i)["energy_gap_rel"], float) for i in seeds]
    ar = np.asarray(report_arrays(report, "ar", 0)["energy_gap_rel"], float)
    try:
        fine = np.asarray(report["results"]["p3_transfer"]["metrics"]["ar"]["fine"]
                          ["per_seed_eval"][0]["per_instance"]["disp_rel_l2"], float)
    except (KeyError, IndexError, TypeError) as exc:
        raise SystemExit("the report records no P3 fine-set arrays (a Phase-2b report "
                         "is expected)") from exc
    mid = lambda v: int(np.argsort(v, kind="stable")[(len(v) - 1) // 2])  # noqa: E731
    return {"fig5": {"set": "val", "index": int(np.argmax(lab[0])),
                     "per_seed_choice": [int(np.argmax(v)) for v in lab],
                     "value": float(lab[0].max())},
            "fig6": {"set": "val", "index": mid(ar), "value": float(ar[mid(ar)])},
            "fig7": {"set": "fine", "index": mid(fine), "value": float(fine[mid(fine)])}}


def arm_model(report: dict, arm: str, state, seed: int, dev: str):
    """The report's model of `arm`, strictly loaded, in eval mode, under the
    run's TF32 policy (the graph network is the run's own `kind: mgn` build)."""
    from fejepa.analysis.common import build_model_from_config
    from fejepa.analysis.posthoc import run_model
    from fejepa.report import config_sha256
    from fejepa.runtime import setup_torch

    if arm != "mgn":
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


def load_energies(U: np.ndarray, arch) -> dict:
    """Per load case: Pi_h(u), relative gap, c*, relative gap of c* u."""
    from fejepa.analysis.posthoc import amplitude_factor, apply_amplitude, free_block, free_mask
    from fejepa.anchor.energy import pi_h

    free = free_mask(arch)
    kf = free_block(arch.K, free)
    pi_star = pi_h(arch.U_star, arch.K, arch.F)
    pi_u = pi_h(U, arch.K, arch.F)
    c = amplitude_factor(U, arch.K, arch.F, free, Kf=kf)
    pi_c = pi_h(apply_amplitude(U, c), arch.K, arch.F)
    return {"pi": pi_u, "pi_star": pi_star, "rel": (pi_u - pi_star) / np.abs(pi_star),
            "c": c, "rel_c": (pi_c - pi_star) / np.abs(pi_star)}


DIAG = ("e2", "eK", "vm_elem", "vm_vol", "stress_energy", "u_dirichlet_max")
"""Per-prediction diagnostics of cmame-paper Stage 1 (see the module docstring)."""
REF = ("u2", "uK", "gamma")
"""Per-instance diagnostics of the reference U*."""


def element_ops(arch):
    """The P1-tetrahedron operator of the instance's mesh, with the shear and
    bulk moduli, built once and shared by every model and seed; None for a
    mesh of another kind (the stress diagnostics are then NaN)."""
    if int(arch.nodes.shape[1]) != 3 or int(arch.elements.shape[1]) != 4:
        return None
    from fejepa.analysis.posthoc import element_operator

    m = arch.meta["material"]
    op = element_operator(arch.nodes, arch.elements, m)
    E, nu = float(m["E"]), float(m["nu"])
    op["G"], op["Bk"] = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    return op


def _stress(op: dict, U: np.ndarray) -> np.ndarray:
    """(L, E, 6) Voigt stresses (tensor shears) of node-major fields U."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))
    eps = np.einsum("eij,lej->lei", op["B"], U2[:, op["dof"]])
    return eps @ op["D"].T


def _vm_p(sig: np.ndarray) -> tuple:
    """Von Mises and mean stress of (..., 6) Voigt stresses."""
    sxx, syy, szz, sxy, syz, szx = (sig[..., i] for i in range(6))
    vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                 + 3.0 * (sxy ** 2 + syz ** 2 + szx ** 2))
    return vm, (sxx + syy + szz) / 3.0


def _quad(K, X: np.ndarray) -> np.ndarray:
    """Per row x of X, x^T K x."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return np.einsum("ld,ld->l", X, (K @ X.T).T)


def reference_diagnostics(arch, op) -> dict:
    """Per load case of U*: ||U*||_2^2, ||U*||_K^2, gamma* and (with a
    tetrahedral mesh) its element von Mises stresses."""
    S = np.atleast_2d(np.asarray(arch.U_star, dtype=np.float64))
    out = {"u2": np.einsum("ld,ld->l", S, S), "uK": _quad(arch.K, S),
           "gamma": np.full(S.shape[0], np.nan), "vm": None, "sig": None}
    if op is not None:
        out["sig"] = _stress(op, S)
        vm, p = _vm_p(out["sig"])
        V = op["vol"]
        out["vm"] = vm
        out["gamma"] = 3.0 * op["G"] * ((p ** 2) @ V) / (op["Bk"] * ((vm ** 2) @ V))
    return out


def load_diagnostics(U: np.ndarray, arch, op, ref: dict) -> dict:
    """Per load case of a prediction u, with e = u - U*: ||e||_2^2, ||e||_K^2,
    the relative von Mises error over the element values (the evaluation's
    vm_rel_l2) and weighted by element volume, and the stress-energy integral
    of e (equal to ||e||_K^2 by Proposition 1)."""
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))
    Er = U2 - np.atleast_2d(np.asarray(arch.U_star, dtype=np.float64))
    # K is the unconstrained assembly; e^T K e is the free block's value because
    # every model zeroes the constrained dofs (recorded: u_dirichlet_max = 0)
    fixed = ~np.asarray(arch.free_mask, dtype=bool).ravel()
    out = {"e2": np.einsum("ld,ld->l", Er, Er), "eK": _quad(arch.K, Er),
           "u_dirichlet_max": (np.abs(U2[:, fixed]).max(axis=1) if fixed.any()
                               else np.zeros(U2.shape[0]))}
    nan = np.full(Er.shape[0], np.nan)
    if op is None:
        out.update(vm_elem=nan, vm_vol=nan.copy(), stress_energy=nan.copy())
        return out
    V, vm_s = op["vol"], ref["vm"]
    sig_u = _stress(op, U2)
    vm_u, _ = _vm_p(sig_u)
    d = vm_u - vm_s
    vm_e, p_e = _vm_p(sig_u - ref["sig"])        # linear: sigma(e) = sigma(u) - sigma(U*)
    out["vm_elem"] = np.linalg.norm(d, axis=1) / (np.linalg.norm(vm_s, axis=1) + 1e-30)
    out["vm_vol"] = np.sqrt(((d ** 2) @ V) / ((vm_s ** 2) @ V))
    out["stress_energy"] = (vm_e ** 2 / (3.0 * op["G"]) + p_e ** 2 / op["Bk"]) @ V
    return out


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a / b where b > 0, NaN elsewhere."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=np.broadcast_to(b > 0, out.shape))
    return out


def diagnostic_summary(D: dict, R: dict, E: dict, used: np.ndarray) -> dict:
    """Per model: the medians of the error's normalised Rayleigh quotient
    (||e||_K^2 / ||e||_2^2 over ||U*||_K^2 / ||U*||_2^2) and of the two von
    Mises errors; the largest ratio of vm_vol^2 to (1 + gamma*) g (at most 1
    by Proposition 1); the largest relative deviations of ||e||_K^2 from the
    stress-energy integral and from 2 (Pi_h(u) - Pi_h(U*)); and, for each
    supervised model, the share of (seed, instance, load case) triples on
    which its error has the larger normalised Rayleigh quotient than the
    label-free model's (same seed index, instance and load case); and the
    largest absolute prediction on a constrained dof (0: the models zero
    them, so that e^T K e with the unconstrained K is the free block's value)."""
    rq = _ratio(_ratio(D["eK"], D["e2"]), _ratio(R["uK"], R["u2"])[None, None])
    g = _ratio(D["eK"], R["uK"][None, None])
    bound = _ratio(D["vm_vol"] ** 2, (1.0 + R["gamma"])[None, None] * g)
    gap = 2.0 * (E["pi"] - E["pi_star_b"])
    out = {}
    for ai, arm in enumerate(ARMS):
        ok = used[ai]
        if not ok.any():
            out[arm] = None
            continue
        sel = lambda X: X[ai][ok]                                          # noqa: E731
        fin = lambda x: x[np.isfinite(x)]                                  # noqa: E731
        rec = {"rayleigh_ratio_median": float(np.nanmedian(sel(rq))),
               "rayleigh_ratio_p10_p90": [float(np.nanpercentile(sel(rq), q)) for q in (10, 90)],
               "vm_elem_median": float(np.nanmedian(sel(D["vm_elem"]))),
               "vm_vol_median": float(np.nanmedian(sel(D["vm_vol"]))),
               "prop1_bound_ratio_max": (float(np.max(fin(sel(bound))))
                                         if fin(sel(bound)).size else None),
               "prop1_bound_ratio_median": float(np.nanmedian(sel(bound))),
               "stress_identity_max_rel_dev": float(np.nanmax(
                   np.abs(sel(D["stress_energy"]) - sel(D["eK"])) / sel(D["eK"]))),
               "gap_identity_max_rel_dev": float(np.nanmax(
                   np.abs(sel(gap) - sel(D["eK"])) / sel(D["eK"]))),
               "u_dirichlet_max": float(np.nanmax(sel(D["u_dirichlet_max"])))}
        if arm != "ar" and used[0].any():
            both = ok & used[0]
            a, r = rq[ai][both], rq[0][both]
            m = np.isfinite(a) & np.isfinite(r)
            rec["rayleigh_above_label_free_share"] = (float(np.mean(a[m] > r[m]))
                                                      if m.any() else None)
        out[arm] = rec
    return out


def figure_payload(arch, preds: dict) -> dict:
    """The arrays of one figure instance."""
    from fejepa.anchor.energy import pi_h

    if int(arch.nodes.shape[1]) == 3:
        from fejepa.fe.tet3d import tet_von_mises as von_mises
    else:
        from fejepa.fe.stress import element_von_mises as von_mises
    m = arch.meta["material"]
    vm = lambda U: np.stack([von_mises(arch.nodes, arch.elements, u, m)  # noqa: E731
                             for u in np.atleast_2d(U)]).astype(np.float32)
    out = {"nodes": np.asarray(arch.nodes, np.float64),
           "tets": np.asarray(arch.elements, np.int64),
           "dirichlet_mask": np.asarray(arch.dirichlet_mask, bool),
           "F": np.asarray(arch.F, np.float64), "U_star": np.asarray(arch.U_star, np.float64),
           "vm_ref": vm(arch.U_star), "pi_star": pi_h(arch.U_star, arch.K, arch.F)}
    op = element_ops(arch)
    if op is not None:
        out["vol"] = np.asarray(op["vol"], np.float64)
    for arm, U in preds.items():
        e = load_energies(U, arch)
        out.update({f"U_{arm}": np.asarray(U, np.float64), f"vm_{arm}": vm(U),
                    f"pi_{arm}": e["pi"], f"rel_{arm}": e["rel"], f"c_{arm}": e["c"],
                    f"rel_c_{arm}": e["rel_c"]})
    return out


def content_check(got: dict, ref: dict) -> dict:
    out = {"n": len(got["disp_rel_l2"])}
    for k in ("disp_rel_l2", "energy_gap_rel"):
        g = np.asarray(got[k], float)
        r = np.asarray(ref[k][:len(g)], float)
        dev = np.abs(g - r) / np.maximum(np.abs(r), 1e-30)
        out[f"{k}_median_rel_dev"] = float(np.median(dev))
        out[f"{k}_max_rel_dev"] = float(np.max(dev))
    out["ok"] = all(out[f"{k}_median_rel_dev"] <= CONTENT_TOL
                    for k in ("disp_rel_l2", "energy_gap_rel"))
    return out


def _torch_info(dev: str) -> dict:
    import torch

    out = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": dev,
           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
           "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if str(dev).startswith("cuda"):
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="the run's report (the records copy)")
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--sup-hashes", default=str(DEFAULT_SUP_HASHES),
                    help="SHA-256 of the supervised states (JSON with a 'sha256' map)")
    ap.add_argument("--n-val", type=int, default=None, help="validation instances (all)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, help="output directory")
    a = ap.parse_args()
    if a.n_val is not None and a.n_val < 1:
        raise SystemExit("--n-val: at least one validation instance")

    from fejepa.analysis.common import resolve_device, sha256_of, write_json
    from fejepa.analysis.posthoc import run_files, run_seeds, verified_states
    from fejepa.data.archive import LazyArchives, has_labels, load_instance
    from fejepa.metrics import displacement_errors, energy_gap_rel
    from fejepa.report import _git_describe

    report = json.loads(Path(a.report).read_text())
    dev = resolve_device(a.device)
    seeds = run_seeds(report)
    b, pool = budgets(report)
    sup = json.loads(Path(a.sup_hashes).read_text())["sha256"]
    out_dir = Path(a.out)

    # ---- inputs first: the instance lists, the figure instances, the states --
    sets = {"val": run_files(report, "val"), "fine": run_files(report, "fine")}
    val_files = sets["val"][:a.n_val] if a.n_val else sets["val"]
    sel = select(report)
    for name, rule in sel.items():
        files = sets[rule["set"]]
        if not 0 <= rule["index"] < len(files):
            raise SystemExit(f"{name}: index {rule['index']} outside the {rule['set']} set "
                             f"({len(files)} instances)")
        f = Path(files[rule["index"]])
        if not f.exists() or not has_labels(f):
            raise SystemExit(f"{name}: {f} is missing or holds no labels")
        rule["file"] = f.name
    ar_states = verified_states(report, a.states_dir, seeds=seeds)      # refuses on mismatch
    states, models, res_states = {}, {}, {}
    for arm in ARMS:
        for s in seeds:
            if arm == "ar":
                p = ar_states[s]
                rec = {"file": p.name, "sha256": sha256(p), "sha256_ok": True,
                       "expected": "the report's d9_restart record"}
            else:
                p = Path(a.states_dir) / f"{arm}_b{b}_s{s}.pt"
                if not p.exists():
                    raise SystemExit(f"{p}: missing (the run's seed-{s} {arm} state)")
                got, want = sha256(p), sup.get(p.name)
                rec = {"file": p.name, "sha256": got, "sha256_ok": got == want,
                       "expected": want}
            states[(arm, s)] = p
            res_states[f"{arm}_s{s}"] = rec
    for (arm, s), p in states.items():
        models[(arm, s)] = arm_model(report, arm, p, s, dev)

    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"what": "wp9 Stage 0e and cmame-paper Stage 1: field figures, per-load energies "
                   "and error diagnostics of Phase-2b's models (CMAME paper); reported only",
           "git": _git_describe(), "torch": _torch_info(dev), "report": a.report,
           "report_sha256": sha256_of(a.report),
           "config_sha256": report["provenance"]["config_sha256"],
           "sup_hashes": a.sup_hashes, "sup_hashes_sha256": sha256_of(a.sup_hashes),
           "budget": int(b), "pool": int(pool), "seeds": seeds, "rules": RULES,
           "selections": sel, "content_tol": CONTENT_TOL, "states": res_states,
           "val": {"n": len(val_files), "files": [Path(f).name for f in val_files]}}
    write_json(out_dir / "fields.json", res)

    # ---- every validation instance x model x seed: per-load energies --------
    n, keys = len(val_files), [(arm, s) for arm in ARMS for s in seeds]
    s0 = seeds[0]
    keep_idx = {rule["index"]: name for name, rule in sel.items() if rule["set"] == "val"}
    kept = {}                    # the val figures' seed-0 predictions, as in the energies
    L = None
    E = {k: None for k in ("pi", "rel", "c", "rel_c")}
    D = {k: None for k in DIAG}
    R = {k: None for k in REF}
    pi_star, metrics = None, {k: {"disp_rel_l2": [], "energy_gap_rel": []} for k in keys}
    t0, t_inf = time.time(), {f"{arm}_s{s}": 0.0 for arm, s in keys}
    for i, arch in enumerate(LazyArchives(val_files)):
        if L is None:
            L = arch.n_loads
            E = {k: np.full((len(ARMS), len(seeds), n, L), np.nan) for k in E}
            D = {k: np.full((len(ARMS), len(seeds), n, L), np.nan) for k in D}
            R = {k: np.full((n, L), np.nan) for k in R}
            pi_star = np.full((n, L), np.nan)
        op = element_ops(arch)
        ref = reference_diagnostics(arch, op)
        for k in REF:
            R[k][i] = ref[k]
        for (arm, s) in keys:
            t = time.perf_counter()
            U = predict(models[(arm, s)], arch, dev)
            t_inf[f"{arm}_s{s}"] += time.perf_counter() - t
            if s == s0 and i in keep_idx:
                kept[(keep_idx[i], arm)] = U
            e = load_energies(U, arch)
            ai, si = ARMS.index(arm), seeds.index(s)
            for k in ("pi", "rel", "c", "rel_c"):
                E[k][ai, si, i] = e[k]
            pi_star[i] = e["pi_star"]
            dg = load_diagnostics(U, arch, op, ref)
            for k in DIAG:
                D[k][ai, si, i] = dg[k]
            metrics[(arm, s)]["disp_rel_l2"].append(float(displacement_errors(U, arch).mean()))
            metrics[(arm, s)]["energy_gap_rel"].append(float(energy_gap_rel(U, arch).mean()))
        if (i + 1) % 16 == 0 or i + 1 == n:
            print(f"[fields] val {i + 1}/{n} | {time.time() - t0:.0f} s", flush=True)

    left_out, mismatch, checks = [], [], {}
    for (arm, s) in keys:
        chk = content_check(metrics[(arm, s)], report_arrays(report, arm, seeds.index(s)))
        rec = res_states[f"{arm}_s{s}"]
        rec["content"] = chk
        rec["used"] = bool(rec["sha256_ok"] or chk["ok"])
        if not rec["used"]:
            left_out.append(f"{arm}_s{s}")
        elif not chk["ok"]:
            mismatch.append(f"{arm}_s{s}")
        checks[f"{arm}_s{s}"] = chk
    used = np.array([[res_states[f"{arm}_s{s}"]["used"] for s in seeds] for arm in ARMS])
    for k in E:
        E[k][~used] = np.nan
    for k in D:
        D[k][~used] = np.nan
    np.savez_compressed(out_dir / "energies_val.npz", arms=np.array(ARMS), seeds=np.array(seeds),
                        files=np.array([Path(f).name for f in val_files]), pi_star=pi_star, **E,
                        **D, u2_star=R["u2"], uK_star=R["uK"], gamma_star=R["gamma"])
    counts = {}
    for ai, arm in enumerate(ARMS):
        ok = used[ai]
        if not ok.any():
            counts[arm] = None
            continue
        pi, rel, rel_c = E["pi"][ai][ok], E["rel"][ai][ok], E["rel_c"][ai][ok]
        counts[arm] = {"load_cases": int(pi.size),
                       "load_cases_pi_positive": int(np.sum(pi > 0)),
                       "load_cases_rel_gap_above_1": int(np.sum(rel > 1)),
                       "instances": int(rel.shape[0] * rel.shape[1]),
                       "instances_mean_rel_gap_above_1": int(np.sum(rel.mean(axis=-1) > 1)),
                       "load_cases_rel_gap_above_1_after_cstar": int(np.sum(rel_c > 1 + 1e-9)),
                       "load_cases_cstar_increased_gap":
                           int(np.sum(rel_c > rel * (1 + 1e-9) + 1e-12)),
                       "median_rel_gap": float(np.median(rel)),
                       "median_rel_gap_after_cstar": float(np.median(rel_c))}
    diag = diagnostic_summary(D, R, dict(E, pi_star_b=pi_star[None, None]), used)
    res.update(content_checks=checks, counts=counts, diagnostics=diag, left_out=left_out,
               content_mismatch=mismatch, inference_seconds=t_inf,
               val_seconds=round(time.time() - t0, 1))
    write_json(out_dir / "fields.json", res)

    # ---- the three figure instances ------------------------------------------
    figs = {}
    for name, rule in sel.items():
        f = sets[rule["set"]][rule["index"]]
        arch = load_instance(f)
        preds = {}
        for arm in ARMS:
            if not res_states[f"{arm}_s{s0}"]["used"]:
                continue
            preds[arm] = kept.get((name, arm))
            if preds[arm] is None:               # fine set, or beyond --n-val
                preds[arm] = predict(models[(arm, s0)], arch, dev)
        np.savez_compressed(out_dir / f"{name}.npz", **figure_payload(arch, preds),
                            meta=np.array(json.dumps({"figure": name, "rule": RULES[name],
                                                      "set": rule["set"], "index": rule["index"],
                                                      "file": Path(f).name, "seed": s0,
                                                      "material": arch.meta["material"]})))
        figs[name] = dict(rule, models=sorted(preds), n_nodes=int(arch.n_nodes),
                          n_elements=int(arch.elements.shape[0]),
                          from_energy_pass=all((name, arm) in kept for arm in preds))
        print(f"[fields] {name}: {rule['set']} #{rule['index']} {Path(f).name} "
              f"({arch.n_nodes} nodes)", flush=True)
    res["figures"] = figs
    res["seconds"] = round(time.time() - t0, 1)
    write_json(out_dir / "fields.json", res)
    print(json.dumps({"content_median_rel_dev": {
                          k: [c["disp_rel_l2_median_rel_dev"], c["energy_gap_rel_median_rel_dev"]]
                          for k, c in checks.items()},
                      "counts": counts, "diagnostics": diag, "left_out": left_out,
                      "content_mismatch": mismatch,
                      "figures": {k: (v["file"], v["index"]) for k, v in figs.items()}}),
          flush=True)
    if left_out:
        raise SystemExit(4)
    if mismatch:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
