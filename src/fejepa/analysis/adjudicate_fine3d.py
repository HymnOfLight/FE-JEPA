"""Executable adjudication of PREREG_FINE3D (Sec. 5-8): R2a (S on the
three-dimensional token bottleneck) and R3 (label-free fine-tuning at the
fine mesh against labelled fine-tuning run in the same session), with the
secondary readings.

Inputs: R2a's two reports (`b`: E2's M = 1,024 configuration trained again on
the fresh seeds 3-5; `s`: the same with S) and, for the mechanism readings
only, their post-hoc amplitude records (scripts/posthoc_amplitude.py); R3's
two reports (scripts/r3_finetune.py: the energy arm and the labelled arm,
its reference); the Phase-2b report (the records copy: R3's starting states,
and the recorded labelled fine-tuning beside its re-run) and E2's M = 1,024
report (the records copy: the corpus, and a reading of the fresh baseline
beside the seeds E2 trained); the stamped pre-registration's CONFIG_SHA256
lines.

R2a and R3 are adjudicated independently: whatever makes one of them invalid
is refused for that one alone (its verdicts read NOT EVALUATED with the
reason) and the other is still adjudicated. Secondary readings never block a
verdict: one that cannot be computed is reported as not evaluated."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..metrics import FIELD_KEYS
from .adjudicate import require_stamped
from .adjudicate_w9 import _soft, config_diff, guard, instance_bootstrap, welch_interval

PREREG_NAME = "PREREG_FINE3D.md"
TAG = "prereg-fine3d"
R2A_LABELS = {"b": "r2a_b_m1024", "s": "r2a_s_m1024"}
R3_LABELS = {"energy": "r3_eft_b16", "labelled": "r3_lft_b16"}
R3_LOSS = {"energy": "ar", "labelled": "labels"}
R3_TRAINER = {"energy": "pretrain", "labelled": "train_supervised"}
"""R3's two arms: their configurations, losses and trainers."""
S_KEYS = {("model", "decode_scale"), ("model", "decode_scale_factor"),
          ("model", "features", "load_density")}
ALLOWED_DIFF_R2A = {("_comment",), ("out",), ("prereg_file",),
                    ("experiments", "e8", "seed_offset"), *S_KEYS}
"""The keys in which an R2a configuration may differ from E2's M = 1,024
configuration (PREREG_FINE3D Sec. 2); the guard is on in both."""
FRESH_SEEDS = [3, 4, 5]
R3_SEEDS = [0, 1, 2]
N_FINE, N_INBAND = 256, 256
R3_TRAIN_N, R3_STEPS = 16, 800
"""R3's budget (P3's labelled fine-tuning at 16 is its reference) and steps
(50 epochs x 16 instances)."""
R3_TRAIN_FILES = [f"instance_{i:05d}.npz" for i in range(256, 272)]
"""The first 16 instances of the fine few-shot prefix: the fine corpus in
manifest order, after the 256 of the evaluation set (P3's partition)."""
PHASE2B_REPORT_SHA256 = "320b6db5060ecae9f4747327228c7705d30877bece67d09af2f68d6c466f2794"
E2_M1024_CONFIG_SHA256 = "526a5688eb49f3332e6f0ced926dcd21227f2bebe7e419f10e9a3aeae0b0b088"
UNINFORMATIVE_RATIO_MIN, UNINFORMATIVE_CSTAR_MIN = 1.5, 1.10
"""PREREG_FINE3D Sec. 7 (W9's rule-1 lines): a fresh baseline whose fine /
in-band ratio of seed means is below 1.5, or whose median c* on the fine set
(the seed mean of each seed's median) is below 1.10, does not show the
deficit S is meant to remove."""
R2B_FINE_REL_MAX = -0.4
"""PREREG_FINE3D Sec. 6: R2b is recommended iff R2a is SUPPORTED and S's fine
displacement error is at most 60% of the fresh baseline's (rel <= -40% in
condition (i)): from there R2b, on the transformer, is predicted to be
SUPPORTED more often than not (records/wp9/fine3d/simulations.json, `r2b`)."""
HIGHER_BETTER = ("crit_recall",)
RUNTIME_KEYS = ("torch", "gpu", "tf32")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path) -> dict:
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------- access --

def _maxpool(d: dict):
    return d[max(d, key=int)]


def e8_cell(rep: dict) -> dict:
    """E8's AR cell (the in-band validation split, the largest pool)."""
    return _maxpool(rep["results"]["e8"]["metrics"]["cells"]["ar"])


def p3_cell(rep: dict, which: str) -> dict:
    """P3's zero-shot AR cell on the fine evaluation set or on the in-band split."""
    return rep["results"]["p3_transfer"]["metrics"]["ar"][which]


def seed_means(c: dict, metric: str) -> list:
    return [float(e[metric]) for e in c["per_seed_eval"]]


def per_instance(c: dict, metric: str) -> list:
    return [np.asarray(e["per_instance"][metric], float) for e in c["per_seed_eval"]]


def seed_medians(c: dict, metric: str) -> list:
    return [float(np.median(x)) for x in per_instance(c, metric)]


def ratio_per_seed(num: list, den: list) -> list:
    return [x / y if y else float("nan") for x, y in zip(num, den, strict=True)]


def _cells_ok(c: dict, n_seeds: int, n: int, what: str) -> None:
    if len(c.get("per_seed_eval") or ()) != n_seeds:
        raise ValueError(f"{what} does not hold one evaluation per seed")
    for e in c["per_seed_eval"]:
        per = e.get("per_instance") or {}
        if any(len(per.get(m) or ()) != n for m in FIELD_KEYS):
            raise ValueError(f"{what} does not hold per-instance values for its {n} instances")


def _direction(g: dict, metric: str) -> str:
    """'better' / 'worse' / 'within the guard' for a guard on `metric`."""
    if not isinstance(g, dict) or "not_evaluated" in g:
        return "not evaluated"
    if g.get("diverged"):
        return "diverged"
    if metric in HIGHER_BETTER:
        return "worse" if g["lower"] else ("better" if g["worse"] else "within the guard")
    return "better" if g["lower"] else ("worse" if g["worse"] else "within the guard")


def _runtime(rep: dict) -> dict:
    pol = rep.get("runtime_policy") or {}
    return {"torch": pol.get("torch") or ((rep.get("provenance") or {}).get("versions") or {})
            .get("torch"), "gpu": pol.get("gpu"), "tf32": pol.get("tf32")}


# ------------------------------------------------------------------- R2a --

def check_r2a(role: str, rep: dict, e2: dict, prereg_entries: dict | None,
              expected_git: str | None, s_factor: float | None) -> dict:
    """Refuse an R2a report that is not what PREREG_FINE3D stamped for
    `role` ('b' or 's'); return what the record needs."""
    label = R2A_LABELS[role]
    require_stamped(rep, f"R2a {role} report", PREREG_NAME)
    cfg = rep["config"]
    if Path(cfg["out"]).parent.name != label:
        raise ValueError(f"the R2a {role} report is {Path(cfg['out']).parent.name}'s, "
                         f"not {label}'s")
    if e2["provenance"]["config_sha256"] != E2_M1024_CONFIG_SHA256:
        raise ValueError("the E2 report is not the stamped M = 1,024 run")
    diff = config_diff(cfg, e2["config"])
    extra = diff - ALLOWED_DIFF_R2A
    if extra:
        raise ValueError(f"the R2a {role} configuration differs from E2's beyond PREREG_FINE3D "
                         f"Sec. 2: {sorted('.'.join(map(str, p)) for p in extra)}")
    m = cfg["model"]
    if int(cfg["experiments"]["e8"].get("seed_offset", 0) or 0) != FRESH_SEEDS[0]:
        raise ValueError(f"the R2a {role} configuration's seed offset is not {FRESH_SEEDS[0]}")
    has_s = bool(diff & S_KEYS)
    if role == "b" and has_s:
        raise ValueError("the R2a baseline carries S's keys")
    if role == "s" and not (m.get("decode_scale") == "l1"
                            and (m.get("features") or {}).get("load_density") is True
                            and (s_factor is None
                                 or float(m.get("decode_scale_factor")) == float(s_factor))):
        raise ValueError(f"the R2a S configuration is not S (decode_scale "
                         f"{m.get('decode_scale')!r}, factor {m.get('decode_scale_factor')!r}, "
                         f"load_density "
                         f"{(m.get('features') or {}).get('load_density')!r})")
    if prereg_entries is not None:
        want = prereg_entries.get(label)
        if want is None or rep["prereg"]["config_sha256"] != want:
            raise ValueError(f"the R2a {role} report's verified hash is not "
                             f"PREREG_FINE3D's CONFIG_SHA256[{label}]")
    prov = rep["provenance"]
    if expected_git is not None and prov.get("git") != expected_git:
        raise ValueError(f"the R2a {role} report ran on {prov.get('git')!r}, not "
                         f"{expected_git!r}")
    if [d.get("manifest_sha256") for d in prov["datasets"]] != \
            [d.get("manifest_sha256") for d in e2["provenance"]["datasets"]]:
        raise ValueError(f"the R2a {role} report ran on another corpus")
    if [int(s) for s in prov.get("seeds")] != FRESH_SEEDS or \
            [int(s) for s in rep["results"]["p3_transfer"]["protocol"]["seeds"]] != FRESH_SEEDS:
        raise ValueError(f"the R2a {role} report's seeds are {prov.get('seeds')} (P3: "
                         f"{rep['results']['p3_transfer']['protocol']['seeds']}), not "
                         f"{FRESH_SEEDS}")
    n = len(FRESH_SEEDS)
    _cells_ok(e8_cell(rep), n, N_INBAND, f"the R2a {role} report's E8 AR cell")
    _cells_ok(p3_cell(rep, "fine"), n, N_FINE, f"the R2a {role} report's P3 fine cell")
    _cells_ok(p3_cell(rep, "inband"), n, N_INBAND, f"the R2a {role} report's P3 in-band cell")
    d9 = rep["results"]["e8"]["metrics"]["d9_restart"]
    return {"label": label, "config_sha256": prov["config_sha256"], "git": prov.get("git"),
            "states": {k: v.get("sha256") for k, v in d9["ar_states"].items()},
            "resumed": d9.get("units_resumed_from_epoch") or {},
            "reuse_states": bool(rep.get("d9_reuse_states")),
            "ledger_total": int((rep.get("solve_ledger") or {}).get("total", 0)),
            "timestamp_utc": prov.get("timestamp_utc"), **_runtime(rep)}


def r2a_values(rep: dict) -> dict:
    """The per-seed values R2a's verdict reads."""
    fine = seed_means(p3_cell(rep, "fine"), "disp_rel_l2")
    inb = seed_means(p3_cell(rep, "inband"), "disp_rel_l2")
    return {"fine_disp": fine, "p3_inband_disp": inb,
            "ratio": ratio_per_seed(fine, inb),
            "ratio_of_seed_means": (float(np.mean(fine) / np.mean(inb))
                                    if all(map(math.isfinite, fine + inb)) else float("nan")),
            "inband_egap": seed_means(e8_cell(rep), "energy_gap_rel"),
            "inband_disp": seed_means(e8_cell(rep), "disp_rel_l2")}


def r2a_verdict(vb: dict, vs: dict) -> dict:
    """PREREG_FINE3D Sec. 6, R2a: S against the fresh baseline -- (i) the fine
    displacement error lower, (ii) the fine / in-band displacement ratio lower,
    (iii) K1: neither in-band metric worse; each beyond the guard. A
    non-finite value of S fails the condition it enters (diverged); a
    non-finite baseline value raises (the comparison is void)."""
    fine = guard(vb["fine_disp"], vs["fine_disp"])
    ratio = guard(vb["ratio"], vs["ratio"])
    k1_egap = guard(vb["inband_egap"], vs["inband_egap"])
    k1_disp = guard(vb["inband_disp"], vs["inband_disp"])
    fails = []
    if not fine["lower"]:
        fails.append("(i) fine displacement error not lower beyond the guard"
                     + (" -- diverged" if fine["diverged"] else ""))
    if not ratio["lower"]:
        fails.append("(ii) fine / in-band displacement ratio not lower beyond the guard"
                     + (" -- diverged" if ratio["diverged"] else ""))
    if k1_egap["worse"] or k1_disp["worse"]:
        fails.append("(iii) K1: in-band "
                     + " and ".join(x for x, g in (("energy gap", k1_egap),
                                                     ("displacement error", k1_disp))
                                    if g["worse"]) + " worse beyond the guard"
                     + (" -- diverged" if k1_egap["diverged"] or k1_disp["diverged"] else ""))
    return {"fine_displacement": fine, "fine_over_inband_ratio": ratio,
            "K1_inband_energy_gap": k1_egap, "K1_inband_displacement": k1_disp,
            "failing": fails,
            "verdict": "SUPPORTED" if not fails else "NOT SUPPORTED: " + "; ".join(fails)}


def r2b_recommendation(v: dict) -> dict:
    """PREREG_FINE3D Sec. 6: the R2b rule, fixed before R2a ran."""
    if v["verdict"] != "SUPPORTED":
        return {"recommended": False, "why": "R2a is not SUPPORTED"}
    rel = v["fine_displacement"]["rel_change"]
    if rel <= R2B_FINE_REL_MAX:
        return {"recommended": True,
                "why": f"R2a SUPPORTED and S's fine displacement error rel {rel:+.1%} <= "
                       f"{R2B_FINE_REL_MAX:+.0%}"}
    return {"recommended": False,
            "why": f"R2a SUPPORTED but S's fine displacement error rel {rel:+.1%} > "
                   f"{R2B_FINE_REL_MAX:+.0%} (the reduction is below the R2b line)"}


def check_amp(amp: dict, rep_sha: str | None) -> None:
    """A post-hoc amplitude record of this report (when its SHA-256 is given),
    every set of every seed complete. `{"unreadable": why}` stands for a record
    the script could not parse."""
    if "unreadable" in amp:
        raise ValueError(f"the amplitude record is unreadable: {amp['unreadable']}")
    if rep_sha is not None and amp.get("report_sha256") != rep_sha:
        raise ValueError("the amplitude record was measured on another report")
    seeds = amp.get("seeds") or {}
    if not seeds:
        raise ValueError("the amplitude record holds no seed")
    for s, v in seeds.items():
        for name in ("inband", "fine"):
            summ = (v.get(name) or {}).get("summary") or {}
            if not summ.get("complete"):
                raise ValueError(f"the amplitude record's {s} {name} pass is incomplete")


def mechanism(amp: dict) -> dict:
    """Seed means of the amplitude readings, and the remesh pass by lc."""
    seeds = sorted(amp["seeds"])
    out = {}
    for name in ("inband", "fine"):
        summ = [amp["seeds"][s][name]["summary"] for s in seeds]
        out[name] = {k: {"per_seed": [float(x[k]) for x in summ],
                         "seed_mean": float(np.mean([x[k] for x in summ]))}
                     for k in ("c_star_median", "c_battery_median", "amp_ratio_median",
                               "disp", "disp_c", "egap", "egap_c")}
        out[name]["reproduction"] = {s: amp["seeds"][s][name].get("reproduction")
                                     for s in seeds}
    rem = [amp["seeds"][s].get("remesh") for s in seeds]
    if all(isinstance(r, dict) and "c_battery_median_by_lc" in r for r in rem):
        out["remesh"] = {k: {lc: float(np.mean([r[k][lc] for r in rem])) for lc in rem[0][k]}
                         for k in ("c_battery_median_by_lc", "u_norm_K_ratio_median_by_lc",
                                   "fscale_ratio_median_by_lc")}
        out["remesh"]["n_geometries"] = [int(r.get("n_geometries", 0)) for r in rem]
    return out


def _uninformative(vb: dict, mech_b: dict | None) -> str | None:
    low = []
    ratio = vb["ratio_of_seed_means"]
    if math.isfinite(ratio) and ratio < UNINFORMATIVE_RATIO_MIN:
        low.append(f"fine / in-band ratio of seed means {ratio:.3g} < {UNINFORMATIVE_RATIO_MIN}")
    if isinstance(mech_b, dict) and "fine" in mech_b:
        c = mech_b["fine"]["c_star_median"]["seed_mean"]
        if c < UNINFORMATIVE_CSTAR_MIN:
            low.append(f"median c* on the fine set {c:.3g} < {UNINFORMATIVE_CSTAR_MIN}")
    if not low:
        return None
    return ("the fresh baseline does not show the deficit S is meant to remove ("
            + "; ".join(low) + "): a NOT SUPPORTED R2a is uninformative about S's "
            "mechanism; the verdict stands")


def r2a_robustness(b: dict, s: dict) -> dict:
    """Beside each condition (no criterion): the guard on per-seed medians, a
    Welch interval and an instance-resampling interval."""
    out = {}
    fb, fs = p3_cell(b, "fine"), p3_cell(s, "fine")
    ib, is_ = p3_cell(b, "inband"), p3_cell(s, "inband")
    out["fine_displacement"] = {
        "medians": _soft(guard, seed_medians(fb, "disp_rel_l2"), seed_medians(fs, "disp_rel_l2")),
        "welch_95": _soft(welch_interval, seed_means(fb, "disp_rel_l2"),
                          seed_means(fs, "disp_rel_l2")),
        "instance_resampling_95": _soft(instance_bootstrap, per_instance(fb, "disp_rel_l2"),
                                        per_instance(fs, "disp_rel_l2"))}
    out["fine_over_inband_ratio"] = {
        "medians": _soft(guard, ratio_per_seed(seed_medians(fb, "disp_rel_l2"),
                                               seed_medians(ib, "disp_rel_l2")),
                         ratio_per_seed(seed_medians(fs, "disp_rel_l2"),
                                        seed_medians(is_, "disp_rel_l2"))),
        "welch_95": _soft(welch_interval, r2a_values(b)["ratio"], r2a_values(s)["ratio"]),
        "instance_resampling_95": _soft(instance_bootstrap, per_instance(fb, "disp_rel_l2"),
                                        per_instance(fs, "disp_rel_l2"),
                                        per_instance(ib, "disp_rel_l2"),
                                        per_instance(is_, "disp_rel_l2"))}
    for k, metric in (("K1_inband_energy_gap", "energy_gap_rel"),
                      ("K1_inband_displacement", "disp_rel_l2")):
        cb, cs = e8_cell(b), e8_cell(s)
        out[k] = {"medians": _soft(guard, seed_medians(cb, metric), seed_medians(cs, metric)),
                  "welch_95": _soft(welch_interval, seed_means(cb, metric), seed_means(cs, metric)),
                  "instance_resampling_95": _soft(instance_bootstrap, per_instance(cb, metric),
                                                  per_instance(cs, metric))}
    return out


def r2a_comparisons(b: dict, s: dict) -> dict:
    """Exploratory: every metric on the fine set, P3's in-band pass and E8's
    cell, S against the fresh baseline, with the guard computed."""
    out = {}
    for name, get in (("fine", lambda r: p3_cell(r, "fine")),
                      ("p3_inband", lambda r: p3_cell(r, "inband")), ("e8_inband", e8_cell)):
        out[name] = {}
        for m in FIELD_KEYS:
            g = _soft(guard, seed_means(get(b), m), seed_means(get(s), m))
            out[name][m] = {"guard": g, "S_is": _direction(g, m)}
    return out


def e2_reading(b: dict, e2: dict) -> dict:
    """The fresh baseline (seeds 3-5) beside E2's M = 1,024 run (seeds 0-2,
    the same configuration otherwise): a coarse check of the training path;
    a difference beyond the guard is a flag, never a verdict."""
    out = {}
    for name, get in (("fine_displacement",
                       lambda r: seed_means(p3_cell(r, "fine"), "disp_rel_l2")),
                      ("inband_energy_gap", lambda r: seed_means(e8_cell(r), "energy_gap_rel")),
                      ("inband_displacement", lambda r: seed_means(e8_cell(r), "disp_rel_l2"))):
        out[name] = _soft(guard, get(e2), get(b))
    flags = [f"{k}: the fresh baseline is {'lower' if g['lower'] else 'higher'} than E2's seeds "
             f"0-2 beyond the guard (rel {g['rel_change']:+.1%}, tau {g['threshold']:.1%})"
             for k, g in out.items() if isinstance(g, dict) and "not_evaluated" not in g
             and not g.get("diverged") and (g["lower"] or g["worse"])]
    out["flags"] = flags
    return out


def adjudicate_r2a(b: dict, s: dict, e2: dict, prereg_entries: dict | None = None,
                   expected_git: str | None = TAG, s_factor: float | None = None,
                   amp_b: dict | None = None, amp_s: dict | None = None,
                   b_sha256: str | None = None, s_sha256: str | None = None) -> dict:
    checks = {"b": check_r2a("b", b, e2, prereg_entries, expected_git, s_factor),
              "s": check_r2a("s", s, e2, prereg_entries, expected_git, s_factor)}
    vb, vs = r2a_values(b), r2a_values(s)
    try:
        verdict = r2a_verdict(vb, vs)
    except ValueError as exc:      # a non-finite value of the fresh baseline: the comparison is void
        verdict = {"verdict": "NOT EVALUATED: a per-seed value of the fresh baseline is not "
                              f"finite: the comparison is void ({exc})", "void": True}
    mech = {}
    for role, amp, sha in (("b", amp_b, b_sha256), ("s", amp_s, s_sha256)):
        if amp is None:
            mech[role] = {"not_evaluated": "no amplitude record given"}
            continue
        try:
            check_amp(amp, sha)
            mech[role] = mechanism(amp)
        except Exception as exc:                          # noqa: BLE001
            mech[role] = {"not_evaluated": f"{type(exc).__name__}: {exc}"}
    out = {"verdict": verdict["verdict"], "conditions": verdict, "values": {"b": vb, "s": vs}}
    if verdict.get("void"):
        out["uninformative_note"] = None
    elif verdict["verdict"] != "SUPPORTED":
        ok_b = "not_evaluated" not in mech["b"]
        out["uninformative_note"] = _uninformative(vb, mech["b"] if ok_b else None)
        if not ok_b:
            out["uninformative_cstar"] = ("not evaluated: no usable amplitude record of the "
                                          "fresh baseline (" + mech["b"]["not_evaluated"] + ")")
    else:
        out["uninformative_note"] = None
    deviations = [f"R2a {r}: solve ledger {c['ledger_total']} (expected 0)"
                  for r, c in checks.items() if c["ledger_total"]]
    deviations += [f"R2a {r}: restarted (units resumed {c['resumed']}, reuse_states "
                   f"{c['reuse_states']})"
                   for r, c in checks.items() if c["resumed"] or c["reuse_states"]]
    diff = [k for k in RUNTIME_KEYS if checks["b"][k] != checks["s"][k]]
    if diff:
        deviations.append("R2a: the two arms ran with different " + ", ".join(
            f"{k} ({checks['b'][k]!r} / {checks['s'][k]!r})" for k in diff))
    out.update({"r2b": ({"recommended": False, "why": "R2a not evaluated"}
                        if verdict.get("void") else r2b_recommendation(verdict)),
                "robustness": r2a_robustness(b, s),
                "comparisons": r2a_comparisons(b, s), "fresh_baseline_vs_e2": e2_reading(b, e2),
                "mechanism": mech, "checks": checks, "deviations": deviations})
    return out


# -------------------------------------------------------------------- R3 --

def check_r3(rep: dict, p2b: dict, prereg_entries: dict | None,
             expected_git: str | None, arm: str = "energy") -> dict:
    """Refuse an R3 report that is not what PREREG_FINE3D stamped for `arm`
    ('energy' or 'labelled')."""
    label = R3_LABELS[arm]
    pr = rep.get("prereg") or {}
    prov = rep["provenance"]
    if not pr.get("config_sha256") or pr["config_sha256"] != prov.get("config_sha256") \
            or Path(str(pr.get("file", ""))).name != PREREG_NAME:
        raise ValueError(f"the R3 {arm} report is not a stamped, guard-verified run of "
                         f"{PREREG_NAME}")
    if prereg_entries is not None and prereg_entries.get(label) != pr["config_sha256"]:
        raise ValueError(f"the R3 {arm} report's verified hash is not PREREG_FINE3D's "
                         f"CONFIG_SHA256[{label}]")
    cfg = rep["config"]
    if Path(cfg["out"]).parent.name != label or cfg["train"].get("loss") != R3_LOSS[arm]:
        raise ValueError(f"the R3 {arm} report is another configuration's "
                         f"({Path(cfg['out']).parent.name}, loss {cfg['train'].get('loss')!r})")
    if expected_git is not None and prov.get("git") != expected_git:
        raise ValueError(f"the R3 {arm} report ran on {prov.get('git')!r}, not "
                         f"{expected_git!r}")
    if prov.get("source_report_sha256") != PHASE2B_REPORT_SHA256:
        raise ValueError(f"the R3 {arm} report started from another source report")
    if [int(s) for s in prov.get("seeds") or ()] != R3_SEEDS or sorted(rep["per_seed"]) != \
            [f"s{s}" for s in R3_SEEDS]:
        raise ValueError(f"the R3 {arm} report's seeds are {prov.get('seeds')}, not {R3_SEEDS}")
    if {d.get("manifest_sha256") for d in prov.get("datasets") or ()} != \
            {d.get("manifest_sha256") for d in p2b["provenance"]["datasets"]}:
        raise ValueError(f"the R3 {arm} report ran on another corpus")
    rec = p2b["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    p3 = p2b["config"]["experiments"]["p3_transfer"]
    epochs = int(p3["fewshot_epochs"])
    if int(cfg["train"]["epochs"]) != epochs or \
            float(cfg["train"]["lr"]) != float(p3["fewshot_lr"]) or \
            int(cfg["train"]["n"]) != R3_TRAIN_N:
        raise ValueError(f"the R3 {arm} schedule is not the labelled few-shot fine-tuning's at "
                         f"{R3_TRAIN_N}")
    out = {"config_sha256": prov.get("config_sha256"), "git": prov.get("git"), "seeds": {}}
    for s in R3_SEEDS:
        r = rep["per_seed"][f"s{s}"]
        t = r["train"]
        if r["source_state"]["sha256"] != rec[f"s{s}"]["sha256"]:
            raise ValueError(f"R3 {arm} seed {s} did not start from the Phase-2b state")
        if r["content_check"]["passed"] is not True:
            raise ValueError(f"R3 {arm} seed {s}'s content check did not pass")
        if r.get("git") != prov.get("git") or r.get("config_sha256") != prov.get("config_sha256"):
            raise ValueError(f"R3 {arm} seed {s} ran on {r.get('git')!r} with configuration "
                             f"{str(r.get('config_sha256'))[:12]}, not the report's")
        if t.get("loss") != R3_LOSS[arm] or t.get("trainer") != R3_TRAINER[arm]:
            raise ValueError(f"R3 {arm} seed {s} trained with loss {t.get('loss')!r} and "
                             f"{t.get('trainer')!r}, not {R3_LOSS[arm]!r} and "
                             f"{R3_TRAINER[arm]!r}")
        if int(t.get("seed", -1)) != s:
            raise ValueError(f"R3 {arm} seed {s} trained with the seed {t.get('seed')!r}")
        if t["files"] != R3_TRAIN_FILES or int(t["steps"]) != R3_STEPS or \
                int(t.get("epochs_run", -1)) != epochs or \
                len(t.get("loss_by_epoch") or ()) != epochs:
            raise ValueError(f"R3 {arm} seed {s} trained on {t['files'][:2]}... for "
                             f"{t['steps']} steps ({t.get('epochs_run')} epochs), not the first "
                             f"{R3_TRAIN_N} of the fine few-shot prefix for {R3_STEPS} steps")
        for name, n in (("fine", N_FINE), ("inband", N_INBAND)):
            per = (r.get(name) or {}).get("per_instance") or {}
            if any(len(per.get(m) or ()) != n for m in FIELD_KEYS):
                raise ValueError(f"R3 {arm} seed {s}'s {name} evaluation does not hold "
                                 f"per-instance values for its {n} instances")
        out["seeds"][f"s{s}"] = {"state_sha256": r["state"]["sha256"],
                                 "content_check": r["content_check"]["max_rel_dev"],
                                 "resumed_from_epoch": t.get("resumed_from_epoch"),
                                 "from_disk": bool(r.get("from_disk")),
                                 "order_sha256": t.get("order_sha256"),
                                 "machines": _machines(r.get("machine")),
                                 "seconds": r.get("seconds")}
    out.update(_runtime(rep))
    return out


def _machines(m) -> list:
    """A seed's machines (host, GPU driver) at its start and its end."""
    if isinstance(m, dict) and set(m) == {"start", "end"}:
        return [m["start"], m["end"]]
    return [m]


def check_r3_pair(ce: dict, cl: dict) -> list:
    """The two arms as one session: the same code and the same recorded epoch
    orders per seed (the orders each seed determines; refused otherwise).
    Returns the deviations: another torch, GPU or TF32 setting between the
    arms, or seeds run on more than one machine (host, GPU driver)."""
    if ce["git"] != cl["git"]:
        raise ValueError(f"R3's arms ran on different code ({ce['git']!r} / {cl['git']!r})")
    for s in R3_SEEDS:
        if ce["seeds"][f"s{s}"]["order_sha256"] != cl["seeds"][f"s{s}"]["order_sha256"]:
            raise ValueError(f"R3's arms recorded different epoch orders for seed {s}")
    diff = [k for k in RUNTIME_KEYS if ce[k] != cl[k]]
    dev = ([f"R3: the two arms ran with different " + ", ".join(
        f"{k} ({ce[k]!r} / {cl[k]!r})" for k in diff)] if diff else [])
    machines = {json.dumps(m, sort_keys=True)
                for c in (ce, cl) for s in R3_SEEDS for m in c["seeds"][f"s{s}"]["machines"]}
    if len(machines) > 1:
        dev.append("R3: the seeds of the two arms ran on more than one machine: "
                   + "; ".join(sorted(machines)))
    return dev


def recorded_ft(p2b: dict) -> dict:
    """The labelled few-shot fine-tuning at 16 as the Phase-2b report records
    it (P3, seeds 0-2): the run R3's labelled arm repeats."""
    return p2b["results"]["p3_transfer"]["metrics"]["fewshot"][str(R3_TRAIN_N)]["finetune"]


def r3_cell(rep: dict, which: str) -> dict:
    """An R3 arm's evaluations as a cell (per_seed_eval in seed order)."""
    return {"per_seed_eval": [rep["per_seed"][f"s{s}"][which] for s in R3_SEEDS]}


def r3_verdicts(energy: dict, labelled: dict) -> dict:
    """PREREG_FINE3D Sec. 6, R3-E and R3-V: the energy-fine-tuned networks'
    fine energy gap / von Mises error lower than the labelled-fine-tuned
    networks' of the same session beyond the guard. A non-finite per-seed
    value of the energy arm fails its verdict (diverged); one of the labelled
    arm voids that verdict (NOT EVALUATED), and the other still stands."""
    ref, new = r3_cell(labelled, "fine"), r3_cell(energy, "fine")
    out = {}
    for name, metric in (("R3-E", "energy_gap_rel"), ("R3-V", "vm_rel_l2")):
        base = seed_means(ref, metric)
        if not all(math.isfinite(v) for v in base):
            out[name] = {"metric": metric,
                         "verdict": f"NOT EVALUATED: the labelled arm's per-seed {metric} "
                                    f"{base} is not finite: the comparison is void"}
            continue
        g = guard(base, seed_means(new, metric))
        out[name] = {"metric": metric, "guard": g,
                     "verdict": "SUPPORTED" if g["lower"] else
                     ("NOT SUPPORTED (diverged)" if g["diverged"] else
                      "NOT SUPPORTED" + (": worse beyond the guard" if g["worse"] else ""))}
    return out


def _amp_readings(rep: dict) -> dict:
    return {name: {k: [float(e["amplitude"]["summary"][k])
                       for e in r3_cell(rep, name)["per_seed_eval"]]
                   for k in ("c_star_median", "c_battery_median", "disp_c", "egap_c")}
            for name in ("fine", "inband")}


def _compare(ref: dict, new: dict, paired: bool = False) -> dict:
    """Every metric, `new` against `ref`, with the guard (and, for two arms of
    the same seeds, the paired per-seed relative differences)."""
    out = {}
    for m in FIELD_KEYS:
        g = _soft(guard, seed_means(ref, m), seed_means(new, m))
        out[m] = {"guard": g, "new_is": _direction(g, m)}
        if paired:                                  # None where either value is not finite
            out[m]["paired_per_seed_rel"] = _soft(lambda: [
                float(n / r - 1) if math.isfinite(n) and math.isfinite(r) and r else None
                for r, n in zip(seed_means(ref, m), seed_means(new, m), strict=True)])
    return out


def r3_readings(energy: dict, labelled: dict, p2b: dict) -> dict:
    """PREREG_FINE3D Sec. 7: secondary readings, no verdict."""
    ef, lf = r3_cell(energy, "fine"), r3_cell(labelled, "fine")
    ei, li = r3_cell(energy, "inband"), r3_cell(labelled, "inband")
    out = {"vs_labelled": _compare(lf, ef, paired=True),
           "inband_vs_labelled": _compare(li, ei, paired=True),
           "vs_zero_shot": {"energy": _compare(p3_cell(p2b, "fine"), ef),
                            "labelled": _compare(p3_cell(p2b, "fine"), lf)},
           "inband_vs_before": {"energy": _compare(e8_cell(p2b), ei),
                                "labelled": _compare(e8_cell(p2b), li)},
           "labelled_vs_record": _compare(recorded_ft(p2b), lf, paired=True),
           "energy_vs_record": _compare(recorded_ft(p2b), ef, paired=True)}
    out["record_flags"] = [
        f"{m}: the labelled arm is {'lower' if v['guard']['lower'] else 'higher'} than the "
        f"recorded labelled fine-tuning beyond the guard (rel {v['guard']['rel_change']:+.1%})"
        for m, v in out["labelled_vs_record"].items()
        if "not_evaluated" not in v["guard"] and not v["guard"].get("diverged")
        and (v["guard"]["lower"] or v["guard"]["worse"])]
    out["fine_over_inband_disp_ratio_per_seed"] = {
        name: _soft(ratio_per_seed, seed_means(f, "disp_rel_l2"), seed_means(i, "disp_rel_l2"))
        for name, f, i in (("energy", ef, ei), ("labelled", lf, li))}
    out["amplitude"] = {"energy": _soft(_amp_readings, energy),
                        "labelled": _soft(_amp_readings, labelled)}
    rob = {}
    for name, metric in (("R3-E", "energy_gap_rel"), ("R3-V", "vm_rel_l2")):
        rob[name] = {"medians": _soft(guard, seed_medians(lf, metric), seed_medians(ef, metric)),
                     "welch_95": _soft(welch_interval, seed_means(lf, metric),
                                       seed_means(ef, metric)),
                     "instance_resampling_95": _soft(instance_bootstrap, per_instance(lf, metric),
                                                     per_instance(ef, metric))}
    out["robustness"] = rob
    return out


def adjudicate_r3(energy: dict, labelled: dict, p2b: dict, prereg_entries: dict | None = None,
                  expected_git: str | None = TAG) -> dict:
    checks = {"energy": check_r3(energy, p2b, prereg_entries, expected_git, "energy"),
              "labelled": check_r3(labelled, p2b, prereg_entries, expected_git, "labelled")}
    dev = check_r3_pair(checks["energy"], checks["labelled"])
    for arm, c in checks.items():
        dev += [f"R3 {arm} seed {s}: resumed from epoch {x['resumed_from_epoch']}"
                for s, x in c["seeds"].items() if x["resumed_from_epoch"]]
        dev += [f"R3 {arm} seed {s}: taken from an earlier attempt's result on disk"
                for s, x in c["seeds"].items() if x["from_disk"]]
    ref = _runtime(p2b)
    diff = [k for k in RUNTIME_KEYS if checks["labelled"][k] != ref[k]]
    notes = ([f"R3 ran with another {', '.join(diff)} than the recorded labelled fine-tuning "
              f"({ {k: checks['labelled'][k] for k in diff} } against "
              f"{ {k: ref[k] for k in diff} })"] if diff else [])
    readings = r3_readings(energy, labelled, p2b)
    notes += readings["record_flags"]
    return {"verdicts": r3_verdicts(energy, labelled), "readings": readings, "checks": checks,
            "reference_runtime": ref, "deviations": dev, "notes": notes}


# ------------------------------------------------------------------ both --

def adjudicate_fine3d(r2a: dict | None, r3: dict | None, p2b: dict, e2: dict,
                      prereg_entries: dict | None = None, expected_git: str | None = TAG,
                      s_factor: float | None = None, missing: dict | None = None) -> dict:
    """`r2a`: {"b": report, "s": report, "amp_b"?, "amp_s"?, "b_sha256"?,
    "s_sha256"?} or None; `r3`: {"energy": report, "labelled": report} or None;
    `missing`: why a part has no input (from the script). Each part is
    adjudicated on its own; a refusal, or any error while reading a part,
    leaves only that part NOT EVALUATED."""
    missing = missing or {}
    out = {"prereg": PREREG_NAME, "tag": expected_git}
    if r2a is None:
        out["R2a"] = {"verdict": "NOT EVALUATED: " + missing.get("R2a", "no R2a reports"),
                      "r2b": {"recommended": False, "why": "R2a not evaluated"}}
    else:
        try:
            out["R2a"] = adjudicate_r2a(r2a["b"], r2a["s"], e2, prereg_entries, expected_git,
                                        s_factor, r2a.get("amp_b"), r2a.get("amp_s"),
                                        r2a.get("b_sha256"), r2a.get("s_sha256"))
        except Exception as exc:                          # noqa: BLE001
            out["R2a"] = {"verdict": f"NOT EVALUATED: refused: {type(exc).__name__}: {exc}",
                          "r2b": {"recommended": False, "why": "R2a not evaluated"}}
    if r3 is None:
        why = missing.get("R3", "no R3 reports")
        out["R3"] = {"verdicts": {k: {"verdict": "NOT EVALUATED: " + why}
                                  for k in ("R3-E", "R3-V")}}
    else:
        try:
            out["R3"] = adjudicate_r3(r3["energy"], r3["labelled"], p2b, prereg_entries,
                                      expected_git)
        except Exception as exc:                          # noqa: BLE001
            out["R3"] = {"verdicts": {k: {"verdict": f"NOT EVALUATED: refused: "
                                                     f"{type(exc).__name__}: {exc}"}
                                      for k in ("R3-E", "R3-V")}}
    return out
