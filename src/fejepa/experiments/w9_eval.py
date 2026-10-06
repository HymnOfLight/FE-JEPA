"""wp9 Stage 0b: the evaluation additions of the wp9 runs.

Everything here is opt-in through a configuration's top-level ``evaluation``
block (and ``experiments.e8.reuse_from``); a configuration without them never
reaches this module, so every existing configuration evaluates exactly as
before.

``evaluation``:
  holdouts   {name: {"dir": <family directory>, "family": <manifest family>}}
             evaluation-only sets (wp9: OOD-2D v1 F1-F5 and the remesh set R).
             Each is verified before any training: the manifest names the
             family, every file matches the SHA-256 its manifest records, and
             every instance is labelled. The trained models are evaluated on
             them with the frozen metric suite; nothing reads them for
             training or selection.
  amplitude  true: every evaluation (validation split and holdouts) also
             records, per instance, the energy-optimal amplitude c* = F^T u /
             u^T K u per load case, the battery form c_b, the errors after
             scaling each load case by its c* (label-free post-processing;
             secondary), and the energy norms of prediction and solution.

``experiments.e8.reuse_from`` = {"report": <a finished run's report>,
"states_dir": <its e8_states>}: the AR units do not train; each loads the
state that run trained for the same seed and pool size -- refused unless the
file's SHA-256 is the one that report records, and unless this configuration
equals that report's configuration outside the evaluation-only keys
(`REUSE_FREE_KEYS`) -- and evaluates it like a freshly trained one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..metrics import FIELD_KEYS, evaluate_fields

AMP_KEYS = ("c_star", "c_battery", "disp_c", "egap_c", "u_norm_K", "ustar_norm_K")
REUSE_FREE_KEYS = ("_comment", "out", "prereg_file", "prereg_guard", "evaluation")
"""Top-level keys in which a configuration may differ from the run whose
states it reuses (besides `experiments.e8.reuse_from` itself): none of them
reaches training."""
REUSE_SUPERVISED_KEYS = ("ar_only", "budgets", "include_mgn", "mgn_budgets", "include_ar_ft",
                         "include_anchor", "include_knorm", "sup_epochs", "sup_lr",
                         "include_naive_baselines")
"""cmame-paper (`reuse_from.supervised_grid`): the `experiments.e8` keys that only
shape the supervised grid and the naive baselines -- none of them reaches the AR
units whose states are reused -- may also differ."""


# -------------------------------------------------------------- holdouts --

def resolve_holdouts(evaluation: dict | None) -> dict:
    """{name: {"files": [...], "provenance": {...}}} for the configuration's
    holdout sets, each verified (family, per-file SHA-256, labels)."""
    from ..data.archive import instance_files, load_manifest, manifest_sha256
    from ..fe.ood2d import verify_manifest_files

    out = {}
    for name, spec in ((evaluation or {}).get("holdouts") or {}).items():
        d = Path(spec["dir"])
        try:
            m = load_manifest(d)
        except (OSError, ValueError) as exc:
            raise ValueError(f"holdout {name}: no readable manifest in {d} "
                             f"({type(exc).__name__}); generate it first") from exc
        if m.get("family") != spec.get("family"):
            raise ValueError(f"holdout {name}: {d} holds family {m.get('family')!r}, "
                             f"the configuration names {spec.get('family')!r}")
        bad = verify_manifest_files(d)
        if bad:
            raise ValueError(f"holdout {name}: {len(bad)} files of {d} are missing or "
                             f"differ from their manifest (first {bad[0]})")
        unlabelled = [r["file"] for r in m["instances"] if not r.get("labelled")]
        if unlabelled:
            raise ValueError(f"holdout {name}: {len(unlabelled)} unlabelled instances "
                             f"(first {unlabelled[0]})")
        out[name] = {"files": [str(f) for f in instance_files(d)],
                     "provenance": {"dir": str(d), "family": m.get("family"),
                                    "seed": m.get("seed"), "n_instances": m["n_instances"],
                                    "manifest_sha256": manifest_sha256(d),
                                    "gmsh_version": m.get("gmsh_version"),
                                    "label_ledger": m.get("ledger")}}
    return out


# ----------------------------------------------------------- evaluation --

def amplitude_record(U: np.ndarray, arch) -> dict:
    """One instance's amplitude readings (AMP_KEYS)."""
    from ..analysis.w9 import amplitude_row

    row = amplitude_row(arch, U)
    return {k: row[k] for k in AMP_KEYS}


def amplitude_summary(per: dict) -> dict:
    cs = np.array([c for row in per["c_star"] for c in row], dtype=float)
    fin = cs[np.isfinite(cs)]
    return {"disp_c": float(np.mean(per["disp_c"])), "egap_c": float(np.mean(per["egap_c"])),
            "c_star_median": float(np.median(fin)) if fin.size else float("nan"),
            "frac_c_star_gt_1": float(np.mean(fin > 1.0)) if fin.size else float("nan"),
            "c_battery_median": float(np.nanmedian(per["c_battery"]))}


def evaluate_model_w9(predict_fn, archs, amplitude: bool = False) -> dict:
    """`metrics.evaluate_model` (the same means and per-instance arrays, from
    the same computation) plus, with `amplitude`, the amplitude readings of
    the same predictions."""
    per = {k: [] for k in FIELD_KEYS}
    amp = {k: [] for k in AMP_KEYS} if amplitude else None
    n = 0
    for a in archs:
        U = predict_fn(a)
        vals = evaluate_fields(U, a)
        for k in FIELD_KEYS:
            per[k].append(vals[k])
        if amp is not None:
            rec = amplitude_record(U, a)
            for k in AMP_KEYS:
                amp[k].append(rec[k])
        n += 1
    out = {k: float(np.mean(per[k])) for k in FIELD_KEYS}
    out["per_instance"] = {k: [float(x) for x in per[k]] for k in FIELD_KEYS}
    out["n_val"] = n
    if amp is not None:
        out["amplitude"] = {"per_instance": amp, "summary": amplitude_summary(amp)}
    return out


# ----------------------------------------------------------- reuse_from --

def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strip(cfg: dict, supervised_grid: bool = False) -> dict:
    c = json.loads(json.dumps(cfg))
    for k in REUSE_FREE_KEYS:
        c.pop(k, None)
    e8 = (c.get("experiments") or {}).get("e8")
    if isinstance(e8, dict):
        e8.pop("reuse_from", None)
        if supervised_grid:
            for k in REUSE_SUPERVISED_KEYS:
                e8.pop(k, None)
    return c


def verify_reuse(cfg: dict, reuse: dict) -> dict:
    """{"states": {seed: {pool: (path, sha256)}}, "provenance": {...}} for an
    evaluation-only E8: refuses unless this configuration equals the source
    report's outside REUSE_FREE_KEYS and every state file has the SHA-256
    that report records for it."""
    rp = Path(reuse["report"])
    src = json.loads(rp.read_text())
    grid = bool(reuse.get("supervised_grid"))
    a, b = _strip(cfg, grid), _strip(src["config"], grid)
    if a != b:
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        sub = []
        for k in ("experiments", "model", "data", "split", "pretrain"):
            if a.get(k) != b.get(k) and isinstance(a.get(k), dict):
                sub += [f"{k}.{j}" for j in sorted(set(a[k]) | set(b.get(k) or {}))
                        if a[k].get(j) != (b.get(k) or {}).get(j)]
        raise ValueError(f"e8.reuse_from: this configuration differs from {rp}'s outside "
                         f"the evaluation-only keys: {diff} {sub}")
    e8 = src["results"]["e8"]
    recorded = e8["metrics"]["d9_restart"]["ar_states"]         # one state per seed
    seeds = sorted(int(k[1:]) for k in recorded)
    pools = [int(p) for p in e8["protocol"]["pool_sizes"]]
    if len(pools) != 1:
        raise ValueError("e8.reuse_from: the source run must have one pool size "
                         f"(its d9 block records one state per seed), has {pools}")
    states = {}
    for s in seeds:
        want = recorded[f"s{s}"]["sha256"]
        path = Path(reuse["states_dir"]) / f"ar_p{pools[0]}_s{s}.pt"
        if not path.is_file():
            raise ValueError(f"e8.reuse_from: {path} not found")
        got = _sha256(path)
        if got != want:
            raise ValueError(f"e8.reuse_from: {path} is not the state {rp} trained "
                             f"(SHA-256 {got[:12]}..., recorded {str(want)[:12]}...)")
        states[s] = {pools[0]: (str(path), got)}
    return {"states": states,
            "provenance": {"report": str(rp), "report_sha256": _sha256(rp),
                           "states_dir": str(reuse["states_dir"]),
                           "states_sha256": {f"s{s}": v[pools[0]][1]
                                             for s, v in states.items()},
                           "source_config_sha256": src["provenance"].get("config_sha256"),
                           **({"supervised_grid": True} if grid else {})}}
