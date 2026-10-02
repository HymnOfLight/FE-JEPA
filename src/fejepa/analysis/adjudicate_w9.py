"""Executable adjudication of PREREG_W9 (Sec. 6-8): H1 (unlabelled pool size,
C1) and H2 (mesh-independent scale, S), with the secondary readings.

Inputs: the arms' reports by role -- `n1024` (E1's states, evaluation only),
`nmax`, `n4096`, `b1024` (E1's base retrained with fresh seeds: H2's reference
and a second 1,024 arm, run in every session 2 since r3) and, when session
1's rule 1 admitted S, `s`; E1's base report (the
records copy); session 1's decision file and the readings it was computed
from (recomputed here with the frozen rules); session 1's OOD-2D record;
optionally session 2's plan and status file, and the remesh set's manifest.

Everything that makes a verdict invalid is refused (ValueError). Run-time
choices that change scheduling and memory only (workers, checkpointing),
restarts, a missing arm, and a changed software stack are recorded as
deviations; a primary arm without a report, or a non-finite value in H2's
fresh reference, leaves only its own hypothesis not evaluated. Secondary
readings never block a verdict: a reading that cannot be computed is
reported as not evaluated."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import numpy as np

from .adjudicate import _finite, noise_guarded_worse, require_stamped

GUARD_BAND = 0.10
GUARD_K = 2.0
TAIL_Q = 90
REPRO_MAX = 1e-4
SETS = ("val", "IB", "F1", "F2", "F3", "F4", "F5", "R")
METRICS = ("energy_gap_rel", "disp_rel_l2")
INBAND = ("val", "IB")
"""H1's in-band set since r3: E1's validation split and the in-band holdout IB
(2,048 fresh training-family instances) together; a seed's value is its mean
over all their instances. H2 and rule 1 read the validation split alone."""
ALLOWED_DIFF = {("_comment",), ("out",), ("prereg_file",), ("evaluation",),
                ("experiments", "e8", "pool_sizes"), ("experiments", "e8", "ar_epochs"),
                ("experiments", "e8", "reuse_from"), ("experiments", "e8", "seed_offset"),
                ("model", "decode_scale"), ("model", "decode_scale_factor"),
                ("model", "features", "load_density")}
"""The keys in which a wp9 configuration may differ from E1's base (PREREG_W9
Sec. 2); `prereg_guard` is on in both."""
S_FACTOR = 0.015625
RULE1_RATIO_MIN, RULE1_CSTAR_MIN = 1.5, 1.10
"""Rule 1's thresholds (scripts/w9_session1_decisions.py; a test keeps the two
in agreement): H2's uninformative case reads the fresh baseline against them."""
BOOT_N, BOOT_SEED = 2000, 20261002
"""r3 (Sec. 7): the instance-resampling interval's resamples and generator
seed, fixed so that the readings are deterministic."""
TRAIN_H_MIN = 0.05
"""The finest training mesh size (OOD-2D v1: training target_h 0.05-0.12): R's
growth reading divides its finest mesh's error by each training-range mesh
size's (r3, Sec. 7)."""


def _spec(pool, epochs, reuse=False, offset=0, s=False) -> dict:
    return {"pool": pool, "epochs": epochs, "reuse": reuse, "seed_offset": offset,
            "decode_scale": "l1" if s else "max",
            "decode_scale_factor": S_FACTOR if s else 1.0, "load_density": s}


DEFAULT_SPECS = {
    "w9_c1_n1024": _spec(1024, 200, reuse=True),
    "w9_c1_n4096": _spec(4096, 50),
    "w9_c1_n25600": _spec(25600, 8),
    "w9_c1_n12800": _spec(12800, 16),
    "w9_b_n1024": _spec(1024, 200, offset=3),
    "w9_s_n1024": _spec(1024, 200, offset=3, s=True),
}
"""Each stamped configuration's identity (scripts/make_w9_configs.py; a test
keeps the two in agreement), keyed by its label in PREREG_W9."""
ROLE_LABELS = {"n1024": ("w9_c1_n1024",), "nmax": ("w9_c1_n25600", "w9_c1_n12800"),
               "n4096": ("w9_c1_n4096",), "b1024": ("w9_b_n1024",), "s": ("w9_s_n1024",)}
SESSION1_INPUTS = ("amp2d", "timing", "memory", "trainval", "val")
"""The decision script's inputs, by its argument names (scripts/w9_session1_decisions.py)."""


def label_of(rep: dict) -> str:
    """The configuration a report ran, from its output directory
    (runs/w9/<arm>/report.json is configs/w9_<arm>.json's)."""
    return "w9_" + Path(rep["config"]["out"]).parent.name


# ----------------------------------------------------------------- access --

def _maxpool(d: dict):
    return d[max(d, key=int)]


def cell(report: dict, set_name: str) -> dict:
    m = report["results"]["e8"]["metrics"]
    if set_name == "val":
        return _maxpool(m["cells"]["ar"])
    hold = m.get("holdouts") or {}
    if set_name not in hold:
        raise ValueError(f"adjudication refused: the report has no holdout {set_name!r}")
    return _maxpool(hold[set_name])


def seed_means(c: dict, metric: str) -> list:
    return [float(e[metric]) for e in c["per_seed_eval"]]


def seed_tails(c: dict, metric: str = "energy_gap_rel", q: float = TAIL_Q) -> list:
    return [float(np.percentile(np.asarray(e["per_instance"][metric], float), q))
            for e in c["per_seed_eval"]]


def seed_amp(c: dict, key: str) -> list:
    out = []
    for e in c["per_seed_eval"]:
        amp = e.get("amplitude")
        if amp is None:
            raise ValueError("an evaluation without amplitude readings")
        out.append(float(amp["summary"][key]))
    return out


def seed_ratios(rep: dict, num_set: str = "F5", den_set: str = "val",
                metric: str = "disp_rel_l2") -> list:
    """Per seed: the num_set / den_set ratio of the seed's mean `metric`."""
    a, b = seed_means(cell(rep, num_set), metric), seed_means(cell(rep, den_set), metric)
    return [x / y if y else float("nan") for x, y in zip(a, b, strict=True)]


def per_instance(c: dict, metric: str) -> list:
    """Per seed: the per-instance values of `metric` (one float array each)."""
    return [np.asarray(e["per_instance"][metric], float) for e in c["per_seed_eval"]]


def seed_medians(c: dict, metric: str) -> list:
    """Per seed: the median of `metric` over the set's instances."""
    return [float(np.median(x)) for x in per_instance(c, metric)]


def pooled_seed_means(rep: dict, sets=INBAND, metric: str = "energy_gap_rel") -> list:
    """Per seed: the mean of `metric` over the instances of `sets` together
    (each set's seed mean weighted by its number of instances). A non-finite
    seed mean of any set makes the seed's value non-finite."""
    cells = [cell(rep, s) for s in sets]
    n = {len(c["per_seed_eval"]) for c in cells}
    if len(n) != 1:
        raise ValueError(f"adjudication refused: the sets {sets} hold different numbers of seeds")
    out = []
    for k in range(n.pop()):
        tot = cnt = 0
        for c in cells:
            e = c["per_seed_eval"][k]
            m = len(e["per_instance"][metric])
            tot += float(e[metric]) * m
            cnt += m
        out.append(tot / cnt)
    return out


def pooled_per_instance(rep: dict, sets, metric: str) -> list:
    """Per seed: the per-instance values of `metric` over `sets`, concatenated."""
    per = [per_instance(cell(rep, s), metric) for s in sets]
    return [np.concatenate(xs) for xs in zip(*per, strict=True)]


# ------------------------------------------------------------------ guard --

def guard(base_vals, new_vals, band: float = GUARD_BAND, k: float = GUARD_K) -> dict:
    """PREREG_W9 Sec. 6: rel = mean(new)/mean(base) - 1, tau = max(band, k x
    SE_rel); `lower` iff rel < -tau, `worse` iff rel > tau. A non-finite base
    value is refused; a non-finite new value is a divergence (counted as
    worse, never as lower)."""
    b = _finite(base_vals, "the reference's per-seed values")
    a = [float(v) for v in new_vals]
    if not a:
        raise ValueError("adjudication refused: the new arm has no per-seed values")
    if any(not math.isfinite(v) for v in a):
        return {"base_mean": statistics.fmean(b), "new_mean": None, "rel_change": None,
                "se_rel": None, "threshold": band, "lower": False, "worse": True,
                "diverged": True, "base_per_seed": b, "new_per_seed": a}
    w = noise_guarded_worse(b, a, band=band, k=k)
    return {"base_mean": w["base_mean"], "new_mean": w["new_mean"],
            "rel_change": w["rel_change"], "se_rel": w["se_rel"], "threshold": w["threshold"],
            "lower": bool(w["rel_change"] < -w["threshold"]), "worse": bool(w["fires"]),
            "diverged": False, "base_per_seed": b, "new_per_seed": a}


def _soft(fn, *args, **kw):
    """A secondary reading: computed, or reported as not evaluated."""
    try:
        return fn(*args, **kw)
    except Exception as exc:                              # noqa: BLE001
        return {"not_evaluated": f"{type(exc).__name__}: {exc}"}


# ------------------------------------------------- robustness (r3, Sec. 7) --

def welch_interval(base_vals, new_vals, level: float = 0.95) -> dict:
    """rel = mean(new)/mean(base) - 1 with a Welch interval: rel +/- t(df) x
    SE_rel (the guard's SE_rel, Welch-Satterthwaite degrees of freedom)."""
    from scipy.stats import t as student_t

    b, a = np.asarray(base_vals, float), np.asarray(new_vals, float)
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(a))):
        raise ValueError("non-finite per-seed values")
    vb, va = b.var(ddof=1) / len(b), a.var(ddof=1) / len(a)
    rel = float(a.mean() / b.mean() - 1)
    if vb + va == 0:
        return {"rel_change": rel, "low": rel, "high": rel, "df": None, "level": level}
    df = (vb + va) ** 2 / (vb ** 2 / (len(b) - 1) + va ** 2 / (len(a) - 1))
    half = float(student_t.ppf(0.5 + level / 2, df) * math.sqrt(vb + va) / b.mean())
    return {"rel_change": rel, "low": rel - half, "high": rel + half, "df": float(df),
            "level": level}


def instance_bootstrap(base: list, new: list, base_den: list | None = None,
                       new_den: list | None = None, n: int = BOOT_N, seed: int = BOOT_SEED,
                       level: float = 0.95) -> dict:
    """An interval for rel from resampling the instances, the seeds kept: per
    resample, the same instances are drawn for both arms and every seed; each
    seed's value is its mean over them, the arm's the seed mean. With
    `*_den` (the in-band arrays) a seed's value is the ratio of its two sets'
    means, each set drawn on its own."""
    def stack(xs):
        arr = np.stack([np.asarray(x, float) for x in xs])
        if not np.all(np.isfinite(arr)):
            raise ValueError("non-finite per-instance values")
        return arr

    rng = np.random.default_rng(seed)
    B, A = stack(base), stack(new)
    if B.shape[1] != A.shape[1]:
        raise ValueError("the arms' instance counts differ")
    idx = rng.integers(0, B.shape[1], size=(n, B.shape[1]))
    vb, va = B[:, idx].mean(axis=2), A[:, idx].mean(axis=2)
    if base_den is not None:
        Bd, Ad = stack(base_den), stack(new_den)
        if Bd.shape[1] != Ad.shape[1]:
            raise ValueError("the arms' instance counts differ")
        jdx = rng.integers(0, Bd.shape[1], size=(n, Bd.shape[1]))
        vb, va = vb / Bd[:, jdx].mean(axis=2), va / Ad[:, jdx].mean(axis=2)
    rel = va.mean(axis=0) / vb.mean(axis=0) - 1
    lo, hi = np.percentile(rel, [50 * (1 - level), 50 * (1 + level)])
    return {"low": float(lo), "high": float(hi), "resamples": n, "level": level}


def robustness(b: dict, a: dict, set_name, metric: str, ratio: bool = False) -> dict:
    """Beside a verdict's comparison of report `a` with reference `b` (no
    criterion): the guard on per-seed medians over the instances, rel's Welch
    interval, and the instance-resampling interval. `set_name`: one set, or a
    tuple of sets read together (H1's in-band set). `ratio`: the comparison
    is of the per-seed set_name / validation ratio (medians: the ratio of the
    two sets' medians)."""
    if not isinstance(set_name, str):
        pb, pa = pooled_per_instance(b, set_name, metric), pooled_per_instance(a, set_name, metric)
        return {"medians": _soft(guard, [float(np.median(x)) for x in pb],
                                 [float(np.median(x)) for x in pa]),
                "welch_95": _soft(welch_interval, pooled_seed_means(b, set_name, metric),
                                  pooled_seed_means(a, set_name, metric)),
                "instance_resampling_95": _soft(instance_bootstrap, pb, pa)}
    cb, ca = cell(b, set_name), cell(a, set_name)
    if ratio:
        vb, va = cell(b, "val"), cell(a, "val")
        med_b = [x / y for x, y in zip(seed_medians(cb, metric), seed_medians(vb, metric),
                                       strict=True)]
        med_a = [x / y for x, y in zip(seed_medians(ca, metric), seed_medians(va, metric),
                                       strict=True)]
        means_b = seed_ratios(b, set_name, "val", metric)
        means_a = seed_ratios(a, set_name, "val", metric)
        boot = (per_instance(cb, metric), per_instance(ca, metric),
                per_instance(vb, metric), per_instance(va, metric))
    else:
        med_b, med_a = seed_medians(cb, metric), seed_medians(ca, metric)
        means_b, means_a = seed_means(cb, metric), seed_means(ca, metric)
        boot = (per_instance(cb, metric), per_instance(ca, metric))
    return {"medians": _soft(guard, med_b, med_a),
            "welch_95": _soft(welch_interval, means_b, means_a),
            "instance_resampling_95": _soft(instance_bootstrap, *boot)}


# ----------------------------------------------------------- consistency --

def config_diff(a, b, path=()) -> set:
    """Key paths where two configuration trees differ (a whole subtree is one
    path when one side lacks it or the types differ)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.add(path + (k,))
            else:
                out |= config_diff(a[k], b[k], path + (k,))
        return out
    return set() if a == b else {path}


def _arm_identity(cfg: dict) -> dict:
    e8 = cfg["experiments"]["e8"]
    m = cfg.get("model") or {}
    return {"pool": int(e8["pool_sizes"][0]) if len(e8["pool_sizes"]) == 1 else None,
            "epochs": int(e8["ar_epochs"]), "reuse": "reuse_from" in e8,
            "seed_offset": int(e8.get("seed_offset", 0) or 0),
            "decode_scale": m.get("decode_scale", "max"),
            "decode_scale_factor": float(m.get("decode_scale_factor", 1.0)),
            "load_density": bool((m.get("features") or {}).get("load_density", False))}


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repro(rep: dict, e1: dict) -> float:
    """Largest relative deviation of the 1,024 arm's in-band per-instance
    arrays from E1's report's (the same states, evaluated again)."""
    got, ref = cell(rep, "val")["per_seed_eval"], cell(e1, "val")["per_seed_eval"]
    if len(got) != len(ref):
        raise ValueError("adjudication refused: the 1,024 arm and E1 differ in seeds")
    dev = 0.0
    for g, r in zip(got, ref, strict=True):
        for k in METRICS:
            a, b = np.asarray(g["per_instance"][k], float), np.asarray(r["per_instance"][k], float)
            if a.shape != b.shape:
                raise ValueError("adjudication refused: the 1,024 arm's validation arrays "
                                 "are not E1's (length)")
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                return float("inf")                       # a non-finite value never reproduces
            dev = max(dev, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))))
    return dev


def check_arm(role: str, rep: dict, e1: dict, specs: dict, prereg_entries: dict | None,
              expected_git: str | None, e1_report_sha256: str | None) -> dict:
    """Refuse an arm that is not what PREREG_W9 stamped for `role`; return
    its identity, run-time settings and what the deviations need."""
    require_stamped(rep, role, "PREREG_W9.md")
    cfg = rep["config"]
    extra = config_diff(cfg, e1["config"]) - ALLOWED_DIFF
    if extra:
        raise ValueError(f"adjudication refused: the {role} arm differs from E1's base "
                         f"configuration beyond PREREG_W9 Sec. 2: "
                         f"{sorted('.'.join(map(str, p)) for p in extra)}")
    ident = _arm_identity(cfg)
    label = label_of(rep)
    if label not in ROLE_LABELS[role]:
        raise ValueError(f"adjudication refused: the {role} report is {label}'s, not one of "
                         f"{ROLE_LABELS[role]}")
    if ident != specs[label]:
        raise ValueError(f"adjudication refused: the {role} arm is {ident}, PREREG_W9 "
                         f"stamps {specs[label]} as {label}")
    if prereg_entries is not None:
        want = prereg_entries.get(label)
        if want is None or rep["prereg"]["config_sha256"] != want:
            raise ValueError(f"adjudication refused: the {role} arm's verified hash is not "
                             f"PREREG_W9's CONFIG_SHA256[{label}]")
    prov, e1p = rep["provenance"], e1["provenance"]
    if expected_git is not None and prov.get("git") != expected_git:
        raise ValueError(f"adjudication refused: the {role} arm ran on {prov.get('git')!r}, "
                         f"not {expected_git!r}")
    if [d.get("manifest_sha256") for d in prov["datasets"]] != \
            [d.get("manifest_sha256") for d in e1p["datasets"]]:
        raise ValueError(f"adjudication refused: the {role} arm ran on another corpus")
    want_seeds = [int(s) + ident["seed_offset"] for s in e1p.get("seeds")]
    if [int(s) for s in prov.get("seeds")] != want_seeds:
        raise ValueError(f"adjudication refused: the {role} arm's seeds {prov.get('seeds')} "
                         f"are not {want_seeds}")
    ev = rep.get("evaluation") or {}
    if not ev.get("amplitude") or sorted(ev.get("holdouts") or {}) != sorted(SETS[1:]):
        raise ValueError(f"adjudication refused: the {role} arm's evaluation is not "
                         "PREREG_W9's (holdouts IB, F1-F5 and R, amplitude readings)")
    sizes = {"val": int(cfg["split"]["n_val"]),
             **{k: int(v["n_instances"]) for k, v in ev["holdouts"].items()}}
    for s in SETS:
        if len(cell(rep, s)["per_seed_eval"]) != len(want_seeds):
            raise ValueError(f"adjudication refused: the {role} arm's {s} cell does not hold "
                             f"one evaluation per seed")
        for e in cell(rep, s)["per_seed_eval"]:          # the verdicts read these arrays
            per = e.get("per_instance") or {}
            if any(len(per.get(m) or ()) != sizes[s] for m in METRICS):
                raise ValueError(f"adjudication refused: the {role} arm's {s} cell does not "
                                 f"hold per-instance values for its {sizes[s]} instances")
    e8 = rep["results"]["e8"]
    d9 = e8["metrics"]["d9_restart"]
    out = {"identity": ident, "label": label, "workers": e8["protocol"].get("workers"),
           "activation_checkpointing": (rep.get("runtime_overrides") or {}).get(
               "activation_checkpointing", (cfg.get("model") or {}).get(
                   "activation_checkpointing", True)),
           "resumed": d9.get("units_resumed_from_epoch") or {},
           "reuse_states": bool(rep.get("d9_reuse_states")),
           "states": {k: {"sha256": v.get("sha256"), "reused": v.get("reused")}
                      for k, v in d9["ar_states"].items()},
           "holdout_manifests": {k: v["manifest_sha256"]
                                 for k, v in (ev.get("holdouts") or {}).items()},
           "ledger_total": int((rep.get("solve_ledger") or {}).get("total", 0)),
           "torch": (prov.get("versions") or {}).get("torch"),
           "gpu": (rep.get("runtime_policy") or {}).get("gpu"),
           "tf32": (rep.get("runtime_policy") or {}).get("tf32"),
           "config_sha256": prov.get("config_sha256"), "git": prov.get("git")}
    if ident["reuse"]:
        want = {k: v["sha256"] for k, v in e1["results"]["e8"]["metrics"]["d9_restart"]
                ["ar_states"].items()}
        got = {k: v["sha256"] for k, v in d9["ar_states"].items()}
        if not e8["protocol"].get("eval_only") or got != want:
            raise ValueError("adjudication refused: the 1,024 arm did not evaluate E1's "
                             f"states (eval_only={e8['protocol'].get('eval_only')}, states "
                             f"{ {k: v[:12] for k, v in got.items()} })")
        if e1_report_sha256 is not None and \
                (rep.get("reuse_from") or {}).get("report_sha256") != e1_report_sha256:
            raise ValueError("adjudication refused: the 1,024 arm reused states through "
                             "another report")
        dev = _repro(rep, e1)
        if not dev <= REPRO_MAX:
            raise ValueError(f"adjudication refused: the 1,024 arm does not reproduce E1's "
                             f"validation arrays (max relative deviation {dev:.3g})")
        out["reproduction_max_rel_dev"] = dev
    elif e8["protocol"].get("eval_only"):
        raise ValueError(f"adjudication refused: the {role} arm did not train")
    return out


def check_session1(decisions: dict, inputs: dict | None, input_sha256: dict | None,
                   decide=None, rules_sha256: str | None = None,
                   expected_git: str | None = None) -> list:
    """Refuse a decision file that is not the frozen rules' output on the
    session's own readings; return notes."""
    notes = []
    if rules_sha256 is not None and decisions.get("rules_sha256") != rules_sha256:
        raise ValueError("adjudication refused: the decisions were not computed by the frozen "
                         "rules (rules_sha256 differs)")
    if expected_git is not None and decisions.get("git") != expected_git:
        raise ValueError(f"adjudication refused: the decisions ran on {decisions.get('git')!r}, "
                         f"not {expected_git!r}")
    if input_sha256 is not None:
        if decisions.get("inputs_sha256") != input_sha256:
            raise ValueError("adjudication refused: the decisions' recorded inputs are not "
                             "the session-1 files given")
    if decide is not None and inputs is not None:
        again = decide(**inputs)
        d = decisions["decisions"]
        for rule in ("S_enters", "pool_max", "checkpointing_off"):
            if json.dumps(again[rule].get("value"), sort_keys=True) != \
                    json.dumps(d[rule].get("value"), sort_keys=True):
                raise ValueError(f"adjudication refused: rule {rule} recomputed from the "
                                 f"session-1 readings gives {again[rule].get('value')}, the "
                                 f"decision file says {d[rule].get('value')}")
        notes.append("decisions recomputed from the session-1 readings: identical")
    return notes


# ------------------------------------------------------------- secondary --

def arm_table(rep: dict) -> dict:
    """Per set: seed means and per-seed values of the main metrics, the tail,
    and the amplitude summaries (each reading computed or not evaluated)."""
    out = {}
    for s in SETS:
        row = {}
        c = cell(rep, s)
        for m in METRICS:
            row[m] = seed_means(c, m)
        row["tail_energy_gap_rel"] = _soft(seed_tails, c)
        for k in ("egap_c", "disp_c", "c_star_median", "c_battery_median"):
            row[k] = _soft(seed_amp, c, k)
        out[s] = {k: ({"seed_mean": float(np.mean(v)), "per_seed": v}
                      if isinstance(v, list) else v) for k, v in row.items()}
    out["inband"] = {}                       # H1's set (r3): validation split and IB together
    for m in METRICS:
        v = _soft(pooled_seed_means, rep, INBAND, m)
        out["inband"][m] = ({"seed_mean": float(np.mean(v)), "per_seed": v}
                            if isinstance(v, list) else v)
    v = _soft(lambda: [float(np.percentile(x, TAIL_Q))
                       for x in pooled_per_instance(rep, INBAND, "energy_gap_rel")])
    out["inband"]["tail_energy_gap_rel"] = ({"seed_mean": float(np.mean(v)), "per_seed": v}
                                            if isinstance(v, list) else v)
    out["ratio_F5_over_val_disp"] = _soft(lambda: {
        "per_seed": seed_ratios(rep), "seed_mean": float(np.mean(seed_ratios(rep)))})
    out["ratio_F5_over_val_disp_of_seed_means"] = _soft(   # rule 1's form (r3, Sec. 5)
        lambda: float(out["F5"]["disp_rel_l2"]["seed_mean"] / out["val"]["disp_rel_l2"]["seed_mean"]))
    return out


def remesh_readings(rep: dict, manifest: dict) -> dict:
    """On R, per mesh size and seed: the median c_b, and the median energy
    norm relative to each geometry's coarsest mesh, of the prediction and of
    the exact solution (the discretisation's own change)."""
    recs = manifest["instances"]
    c = cell(rep, "R")
    hs = sorted({float(r["target_h"]) for r in recs}, reverse=True)
    coarsest = {}
    for i, r in enumerate(recs):
        g = r["geometry"]
        if g not in coarsest or float(r["target_h"]) > float(recs[coarsest[g]]["target_h"]):
            coarsest[g] = i
    out = {repr(h): {"c_battery_median": [], "u_norm_ratio_median": [],
                     "ustar_norm_ratio_median": []} for h in hs}
    for e in c["per_seed_eval"]:
        amp = e["amplitude"]["per_instance"]
        if len(amp["c_battery"]) != len(recs):
            raise ValueError("R's readings and manifest differ in length")
        for h in hs:
            idx = [i for i, r in enumerate(recs) if float(r["target_h"]) == h]

            def ratio(key):
                v = [u / b for i in idx for u, b in zip(amp[key][i],
                                                       amp[key][coarsest[recs[i]["geometry"]]],
                                                       strict=True) if b > 0]
                return float(np.median(v))
            out[repr(h)]["c_battery_median"].append(float(np.nanmedian(
                [amp["c_battery"][i] for i in idx])))
            out[repr(h)]["u_norm_ratio_median"].append(ratio("u_norm_K"))
            out[repr(h)]["ustar_norm_ratio_median"].append(ratio("ustar_norm_K"))
    return out


def remesh_growth(rep: dict, manifest: dict, metric: str = "disp_rel_l2") -> dict:
    """r3 (Sec. 7): per seed, R's mean `metric` at its finest mesh size over
    its mean at each training-range mesh size (only the mesh changes; rule 1
    never read R's errors)."""
    recs = manifest["instances"]
    hs = sorted({float(r["target_h"]) for r in recs})
    vals = per_instance(cell(rep, "R"), metric)
    if any(len(v) != len(recs) for v in vals):
        raise ValueError("R's readings and manifest differ in length")
    at = {h: [i for i, r in enumerate(recs) if float(r["target_h"]) == h] for h in hs}
    return {repr(h): [float(np.mean(v[at[hs[0]]]) / np.mean(v[at[h]])) for v in vals]
            for h in hs if h >= TRAIN_H_MIN and h != hs[0]}


def comparisons(base_tab: dict, new_tab: dict) -> dict:
    """Exploratory: every metric and set against the reference, with the
    Sec. 6 guard computed (no verdict attached; about a hundred comparisons)."""
    keys = (*METRICS, "tail_energy_gap_rel", "egap_c", "disp_c")
    out = {}
    for s in (*SETS, "inband"):
        out[s] = {}
        for k in keys:
            if s == "inband" and k not in base_tab[s] and k not in new_tab[s]:
                continue
            b, n = base_tab[s].get(k, {}), new_tab[s].get(k, {})
            out[s][k] = (_soft(guard, b["per_seed"], n["per_seed"])
                         if "per_seed" in b and "per_seed" in n else
                         {"not_evaluated": "reading missing"})
    return out


def _status_attempts(status_text: str | None) -> dict:
    """{log: number of attempts} from session 2's status file."""
    out = {}
    for line in (status_text or "").splitlines():
        m = re.match(r"(\S+\.log) (start|exit=)", line.strip())
        if m and m.group(2) == "start":
            out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


# ------------------------------------------------------------ adjudicate --

def h2_verdict(b: dict, s: dict, tab_b: dict, tab_s: dict) -> dict:
    """PREREG_W9 Sec. 6, H2: S against the fresh baseline -- (i) F5
    displacement lower, (ii) the F5 / in-band displacement ratio lower, (iii)
    K1: neither in-band metric worse; each beyond the guard. A non-finite
    value of S fails the condition it enters (diverged); a non-finite
    baseline value raises (the comparison is void)."""
    f5 = guard(tab_b["F5"]["disp_rel_l2"]["per_seed"], tab_s["F5"]["disp_rel_l2"]["per_seed"])
    ratio = guard(seed_ratios(b), seed_ratios(s))
    k1_egap = guard(tab_b["val"]["energy_gap_rel"]["per_seed"],
                    tab_s["val"]["energy_gap_rel"]["per_seed"])
    k1_disp = guard(tab_b["val"]["disp_rel_l2"]["per_seed"], tab_s["val"]["disp_rel_l2"]["per_seed"])
    fails = []
    if not f5["lower"]:
        fails.append("(i) F5 displacement error not lower beyond the guard"
                     + (" -- diverged" if f5["diverged"] else ""))
    if not ratio["lower"]:
        fails.append("(ii) F5 / in-band displacement ratio not lower beyond the guard"
                     + (" -- diverged" if ratio["diverged"] else ""))
    if k1_egap["worse"] or k1_disp["worse"]:
        fails.append("(iii) K1: in-band "
                     + " and ".join(x for x, g in (("energy gap", k1_egap),
                                                     ("displacement error", k1_disp))
                                    if g["worse"]) + " worse beyond the guard"
                     + (" -- diverged" if k1_egap["diverged"] or k1_disp["diverged"] else ""))
    return {"F5_displacement": f5, "F5_over_inband_ratio": ratio,
            "K1_inband_energy_gap": k1_egap, "K1_inband_displacement": k1_disp,
            "verdict": "SUPPORTED" if not fails else "NOT SUPPORTED: " + "; ".join(fails)}


def _fresh_baseline_flag(g: dict) -> str | None:
    """r3 (Sec. 7): the fresh baseline against E1's states on the in-band
    energy gap -- a coarse check of the reuse (other seeds; the guard's floor
    is 10%): a difference beyond the guard, or a diverged baseline, is
    reported with H1 as a flag and never changes its verdict."""
    if not isinstance(g, dict) or "not_evaluated" in g:
        return None
    if g.get("diverged"):
        return ("the fresh baseline's in-band energy gap is non-finite (diverged): the coarse "
                "check of the reuse could not be made; H1's verdict stands")
    if not (g.get("lower") or g.get("worse")):
        return None
    return (f"the fresh baseline's in-band energy gap is {'lower' if g['lower'] else 'worse'} "
            f"than E1's states beyond the guard (rel {g['rel_change']:+.1%}, tau "
            f"{g['threshold']:.1%}): the coarse check of the reuse shows a difference; H1's "
            "verdict stands")


def _uninformative(tab_b: dict) -> str | None:
    """r3 (Sec. 7): the fresh baseline does not show the deficit rule 1
    admitted S on -- its F5 / in-band ratio of seed means below rule 1's
    line, or its median c* on F5 below rule 1's line."""
    ratio = tab_b.get("ratio_F5_over_val_disp_of_seed_means")
    cstar = (tab_b["F5"].get("c_star_median") or {}).get("seed_mean")
    low = []
    if isinstance(ratio, float) and ratio < RULE1_RATIO_MIN:
        low.append(f"F5 / in-band ratio {ratio:.3g} < {RULE1_RATIO_MIN}")
    if isinstance(cstar, float) and cstar < RULE1_CSTAR_MIN:
        low.append(f"median c* on F5 {cstar:.3g} < {RULE1_CSTAR_MIN}")
    if not low:
        return None
    return ("the fresh baseline does not show the deficit rule 1 admitted S on ("
            + "; ".join(low) + "): NOT SUPPORTED is uninformative about S's mechanism; "
            "the verdict stands")


ROLE_OF_ARM = {"c1_n1024": "n1024", "c1_n25600": "nmax", "c1_n12800": "nmax",
               "c1_n4096": "n4096", "b_n1024": "b1024", "s_n1024": "s"}
"""Session 2's arm directories (and log names) by role."""


def _status_finished(status_text: str | None) -> set:
    """The arms session 2's status file shows finished: the last attempt
    exited 0, or the arm was skipped because its report existed."""
    last, skipped = {}, set()
    for line in (status_text or "").splitlines():
        m = re.match(r"(\S+)\.log exit=(\d+)", line.strip())
        if m:
            last[m.group(1)] = int(m.group(2))
        m = re.match(r"(\S+): report exists, skipped", line.strip())
        if m:
            skipped.add(m.group(1))
    return {a for a, rc in last.items() if rc == 0} | skipped


def adjudicate_w9(reports: dict, e1: dict, decisions: dict, ood_record: dict,
                  r_manifest: dict | None = None, specs: dict | None = None,
                  prereg_entries: dict | None = None, expected_git: str | None = None,
                  e1_report_sha256: str | None = None, session1_inputs: dict | None = None,
                  session1_sha256: dict | None = None, decide=None,
                  rules_sha256: str | None = None, plan: dict | None = None,
                  status_text: str | None = None, r_manifest_sha256: str | None = None,
                  returned_arms: set | None = None) -> dict:
    """PREREG_W9 Sec. 6-8. Session-level faults refuse the adjudication
    (ValueError); a report that fails its own checks is refused -- left out
    and recorded -- and a hypothesis without its valid reports is NOT
    EVALUATED. `returned_arms`: the arm directories the session-2 return
    holds (default: the reports' own)."""
    specs = specs or DEFAULT_SPECS
    notes = check_session1(decisions, session1_inputs, session1_sha256, decide,
                           rules_sha256, expected_git)
    dec = decisions["decisions"]
    for rule in ("S_enters", "pool_max"):
        if dec.get(rule, {}).get("value") is None:
            raise ValueError(f"adjudication refused: session 1's {rule} is undecided "
                             f"({dec.get(rule, {}).get('error')})")
    if dec["pool_max"].get("n") is None:
        raise ValueError("adjudication refused: rule 2 found no pool (no session 2 was due)")
    s_admitted = bool(dec["S_enters"]["value"])
    want_manifests = {k: v.get("manifest_sha256") for k, v in ood_record["families"].items()}
    if sorted(want_manifests) != sorted(SETS[1:]) or not all(want_manifests.values()):
        raise ValueError("adjudication refused: session 1's OOD-2D record is incomplete "
                         f"({ {k: bool(v) for k, v in want_manifests.items()} })")
    from ..fe.ood2d import DEFAULT_SEEDS, SET_SIZES

    off = {k: [v.get("n_instances"), v.get("seed")] for k, v in ood_record["families"].items()
           if [v.get("n_instances"), v.get("seed")] != [SET_SIZES[k], DEFAULT_SEEDS[k]]}
    if off:
        raise ValueError("adjudication refused: session 1's evaluation sets are not PREREG_W9's "
                         f"(instances, seed): {off}, Sec. 3 fixes "
                         f"{ {k: [SET_SIZES[k], DEFAULT_SEEDS[k]] for k in off} }")
    if r_manifest_sha256 is not None and r_manifest_sha256 != want_manifests["R"]:
        raise ValueError("adjudication refused: the R manifest given is not session 1's")
    if plan is not None and plan.get("decisions_sha256") and decisions.get("_sha256") and \
            plan["decisions_sha256"] != decisions["_sha256"]:
        raise ValueError("adjudication refused: session 2 was planned from another decision file")
    if status_text is not None:
        have = set(returned_arms) if returned_arms is not None else {
            Path(r["config"]["out"]).parent.name for r in reports.values()}
        lost = sorted(_status_finished(status_text) - have)
        if lost:
            raise ValueError(f"adjudication refused: the status file shows {lost} finished, but "
                             "the return holds no report for them -- return the complete session")

    arms, refused = {}, {}
    nmax_label = f"w9_c1_n{int(dec['pool_max']['n'])}"
    for role, rep in reports.items():
        try:
            if role == "s" and not s_admitted:
                raise ValueError("adjudication refused: rule 1 did not admit S")
            a = check_arm(role, rep, e1, specs, prereg_entries, expected_git, e1_report_sha256)
            if role == "nmax" and a["label"] != nmax_label:
                raise ValueError(f"adjudication refused: rule 2 selected {nmax_label}, the "
                                 f"report is {a['label']}'s")
            if a["holdout_manifests"] != want_manifests:
                raise ValueError("adjudication refused: the holdout manifests are not "
                                 "session 1's")
            arms[role] = a
        except (ValueError, KeyError, TypeError) as exc:
            refused[role] = f"{type(exc).__name__}: {exc}" if not isinstance(exc, ValueError) \
                else str(exc)
    valid = {r: reports[r] for r in arms}

    deviations = [f"{role}: report refused -- {why}" for role, why in sorted(refused.items())]
    expected_roles = {"n1024", "nmax", "n4096", "b1024"} | ({"s"} if s_admitted else set())
    for role in sorted(expected_roles - set(reports)):
        deviations.append(f"{role}: no report (planned arm not run or not returned)")
    rule3 = dec.get("checkpointing_off", {}).get("value")
    if rule3 is None:
        deviations.append("rule 3 undecided: checkpointing kept on (memory and time only)")
    flags = {}
    if plan is not None:
        for a in plan.get("arms", []):
            role = {"w9_c1_n1024": "n1024", "w9_c1_n4096": "n4096", "w9_b_n1024": "b1024",
                    "w9_s_n1024": "s"}.get(Path(a["config"]).stem, "nmax")
            f = a.get("flags", [])
            flags[role] = {"workers": int(f[f.index("--workers") + 1]) if "--workers" in f
                           else None,
                           "ckpt": (f[f.index("--activation-checkpointing") + 1] == "on")
                           if "--activation-checkpointing" in f else None}
    else:
        deviations.append("no session-2 plan given: run-time settings checked against the "
                          "rules directly")
    want_ckpt_rule = not bool(rule3)
    nmax_workers = (1 if dec["pool_max"].get("mode") == "sequential"
                    else int(e1["config"].get("workers", 1)))
    for role, a in arms.items():
        if a["identity"]["reuse"]:
            continue
        fl = flags.get(role, {})
        want_ckpt = fl.get("ckpt") if fl.get("ckpt") is not None else want_ckpt_rule
        if bool(a["activation_checkpointing"]) != want_ckpt:
            deviations.append(f"{role}: activation checkpointing "
                              f"{'on' if a['activation_checkpointing'] else 'off'}, the plan chose "
                              f"{'on' if want_ckpt else 'off'} (memory and time only)")
        want_w = fl.get("workers") if fl.get("workers") is not None else (
            nmax_workers if role == "nmax" else int(e1["config"].get("workers", 1)))
        if a["workers"] is not None and int(a["workers"]) != int(want_w):
            deviations.append(f"{role}: {a['workers']} workers, the plan chose {want_w} "
                              "(scheduling only)")
        if a["resumed"]:
            deviations.append(f"{role}: units resumed from epoch checkpoints {a['resumed']}")
        if a["reuse_states"]:
            deviations.append(f"{role}: run in restart mode (--reuse-states)")
    for role, a in arms.items():
        if a["ledger_total"]:
            deviations.append(f"{role}: solve ledger {a['ledger_total']} (expected 0)")
        e1pol = e1.get("runtime_policy") or {}
        e1torch = (e1["provenance"].get("versions") or {}).get("torch")
        if a["torch"] != e1torch or a["gpu"] != e1pol.get("gpu") or a["tf32"] != e1pol.get("tf32"):
            deviations.append(f"{role}: software or hardware differs from E1's run (torch "
                              f"{a['torch']} vs {e1torch}, GPU {a['gpu']} vs {e1pol.get('gpu')}, "
                              f"tf32 {a['tf32']} vs {e1pol.get('tf32')}) -- H1 compares with E1's "
                              "states trained on E1's stack")
    for log, n in _status_attempts(status_text).items():
        if n > 1:
            deviations.append(f"{log}: {n} attempts (session 2 status file)")

    def missing(roles):
        return "; ".join(f"{r}: {'refused' if r in refused else 'no report'}"
                         for r in roles if r not in arms)

    tabs = {role: arm_table(rep) for role, rep in valid.items()}
    h1 = {"verdict": f"NOT EVALUATED ({missing(('n1024', 'nmax'))})",
          "rel_change": None, "threshold": None}
    if {"n1024", "nmax"} <= set(arms):
        try:
            h1 = guard(pooled_seed_means(valid["n1024"]), pooled_seed_means(valid["nmax"]))
            h1["verdict"] = ("SUPPORTED" if h1["lower"] else
                             "NOT SUPPORTED (diverged)" if h1["diverged"] else
                             "NOT SUPPORTED (worse beyond the guard)" if h1["worse"] else
                             "NOT SUPPORTED")
            h1["n_max"] = arms["nmax"]["identity"]["pool"]
            h1["n_max_configuration"] = arms["nmax"]["label"]
            h1["inband_sets"] = list(INBAND)
            h1["robustness"] = _soft(robustness, valid["n1024"], valid["nmax"], INBAND,
                                     "energy_gap_rel")
        except ValueError as exc:                         # a non-finite reference value
            h1 = {"verdict": "NOT EVALUATED (a non-finite value of the 1,024 arm voids the "
                             f"comparison: {exc})", "rel_change": None, "threshold": None}
            deviations.append("n1024: a non-finite value in H1's reference -- H1 not evaluated")
    out = {"what": "PREREG_W9 adjudication", "H1": h1, "notes": notes,
           "arms": {r: {k: v for k, v in a.items() if k != "holdout_manifests"}
                    for r, a in arms.items()},
           "refused_reports": refused,
           "tables": tabs,
           "comparisons_vs_n1024_exploratory": {r: comparisons(tabs["n1024"], t)
                                                for r, t in tabs.items() if r != "n1024"}
           if "n1024" in tabs else {"not_evaluated": "no valid n1024 report"},
           "deviations": deviations}
    pred = dec.get("C0_4_prediction") or {}
    out["c0_4_prediction"] = {"reading": pred.get("reading"),
                              "predicted_overfit": pred.get("value"),
                              "observed_inband_rel_change": h1.get("rel_change")}
    if not s_admitted:
        out["H2"] = {"verdict": "NOT RUN (rule 1 did not admit S)"}
    elif {"b1024", "s"} <= set(arms):
        try:
            out["H2"] = h2_verdict(valid["b1024"], valid["s"], tabs["b1024"], tabs["s"])
        except ValueError as exc:                         # a non-finite reference value
            out["H2"] = {"verdict": "NOT EVALUATED (a non-finite value of the fresh baseline "
                                    f"voids the comparison: {exc})"}
            deviations.append("b1024: a non-finite value in H2's reference -- H2 not "
                              "evaluated; H1 is unaffected")
        else:                     # secondary readings beside the verdict (r3, Sec. 7)
            out["H2"]["robustness"] = {
                k: _soft(robustness, valid["b1024"], valid["s"], *args)
                for k, args in (("F5_displacement", ("F5", "disp_rel_l2")),
                                ("F5_over_inband_ratio", ("F5", "disp_rel_l2", True)),
                                ("K1_inband_energy_gap", ("val", "energy_gap_rel")),
                                ("K1_inband_displacement", ("val", "disp_rel_l2")))}
            note = _soft(_uninformative, tabs["b1024"])
            if isinstance(note, str) and out["H2"]["verdict"] != "SUPPORTED":
                out["H2"]["uninformative"] = note
        out["comparisons_s_vs_b1024_exploratory"] = comparisons(tabs["b1024"], tabs["s"])
    else:
        out["H2"] = {"verdict": f"NOT EVALUATED ({missing(('b1024', 's'))})"}
    if s_admitted:
        out["S_mechanism"] = {r: {"c_star_median_F5": tabs[r]["F5"].get("c_star_median"),
                                  "ratio_F5_over_val_disp": tabs[r]["ratio_F5_over_val_disp"]}
                              for r in ("b1024", "s", "n1024") if r in tabs}
    if {"b1024", "n1024"} <= set(tabs):
        out["replication_b1024_vs_n1024"] = comparisons(tabs["n1024"], tabs["b1024"])
        flag = _soft(_fresh_baseline_flag,
                     out["replication_b1024_vs_n1024"]["inband"]["energy_gap_rel"])
        if isinstance(flag, str):
            h1["fresh_baseline_flag"] = flag
    if {"b1024", "nmax"} <= set(tabs):
        out["nmax_vs_b1024_exploratory"] = comparisons(tabs["b1024"], tabs["nmax"])
    if r_manifest is not None:
        out["remesh"] = {r: _soft(remesh_readings, rep, r_manifest) for r, rep in valid.items()}
        out["remesh_growth"] = {r: _soft(remesh_growth, rep, r_manifest)
                                for r, rep in valid.items()}
        gb, gs = out["remesh_growth"].get("b1024"), out["remesh_growth"].get("s")
        if gb is not None and gs is not None:
            out["remesh_growth_s_vs_b1024_exploratory"] = (
                {h: _soft(guard, gb[h], gs[h]) for h in gb}
                if "not_evaluated" not in gb and "not_evaluated" not in gs else
                {"not_evaluated": "a growth reading is missing"})
    return out


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


def file_sha256(path) -> str:
    return _sha256(path)
