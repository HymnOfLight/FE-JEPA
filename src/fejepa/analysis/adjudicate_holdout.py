"""Executable adjudication of PREREG_HOLDOUT (Sec. 5-7): the manuscript's
claims re-read on the holdout sets.

Every claim is read from the per-instance arrays that `scripts/holdout_run.py`
writes (`eval_arrays.npz`: per row, seed and holdout instance the frozen
suite's five metrics, and per load case the relative energy gap and
Pi_h(u); the same for the validation split). A claim reads only states that
passed their content check; a claim that needs a state that did not is NOT
EVALUATED. A non-finite value that a claim reads leaves it NOT EVALUATED,
also when it is of the row the claim says is better (T1, T2, T3 and D2),
except in the claims H1, H2a, H2b and H2c (H2c without the guard) and the
reading H3, which follow PREREG_CM2D Sec. 4: a non-finite per-seed value of
the reference row leaves the claim NOT EVALUATED, one of the new row alone
is a divergence and the claim is NOT REPLICATED. Each claim is otherwise REPLICATED or NOT
REPLICATED by the rule the pre-registration fixes for it; the secondary
readings, each beside its value on the validation instances, never decide
anything.

Claims (PREREG_HOLDOUT Sec. 5):
  T1  3D: the label-free transformer better than the supervised transformer
      on all five metrics, in each seed (seed-matched) and in the seed mean;
  T2  3D: the graph network's displacement error lower than the label-free
      transformer's on more than half of the instances in each seed, and its
      von Mises error higher on at least 95% of the instance-seed pairs;
  T3  3D: each supervised network worse than the zero field on at least one
      load case, and every such load case, and no other, with Pi_h(u) > 0
      (an identity of the computation: the count is the information);
  H1  2D: the label-free transformer's relative energy gap lower than L_D's
      beyond the guard;
  H2a 2D: L_K's relative energy gap lower than L_D's beyond the guard;
  H2b 2D: L_K's von Mises error lower than L_D's beyond the guard;
  H2c 2D: L_K's seed-mean von Mises error less than half of L_D's;
  D2  2D: the label-free transformer's displacement error higher than L_D's
      and its other four metrics better, in each seed and in the seed mean;
  H3  2D: L_K against the label-free transformer, a reading without verdict.
"""

from __future__ import annotations

import math

import numpy as np

from .adjudicate_w9 import guard, instance_bootstrap, welch_interval
from .holdout import HIGHER_BETTER, METRICS

REPLICATED, NOT_REPLICATED, NOT_EVALUATED = "REPLICATED", "NOT REPLICATED", "NOT EVALUATED"
CLAIMS = ("T1", "T2", "T3", "H1", "H2a", "H2b", "H2c", "D2")
READINGS = ("H3",)
COMPARISONS = {"h3d": (("ar", "labels"), ("mgn", "ar")),
               "h2d": (("ar", "labels"), ("labels_knorm", "labels"), ("mgn", "labels"),
                       ("labels_knorm", "ar"), ("mgn", "ar"))}
"""The secondary readings' comparisons, (new, reference) (PREREG_HOLDOUT Sec. 6)."""


class Arrays:
    """One set's arrays (the holdout instances, or with `prefix="val_"` the
    validation instances) with the rows and seeds that passed their content
    check (`used`)."""

    def __init__(self, npz: dict, used: dict, prefix: str = ""):
        self.rows = [str(r) for r in npz["rows"]]
        self.seeds = [int(s) for s in npz["seeds"]]
        if [str(m) for m in npz["metrics"]] != list(METRICS):
            raise ValueError(f"the arrays' metrics are {list(npz['metrics'])}, "
                             f"expected {list(METRICS)}")
        self.values = np.asarray(npz[prefix + "values"], dtype=float)      # (R, S, N, 5)
        self.gap_load = np.asarray(npz[prefix + "gap_load"], dtype=float)  # (R, S, N, L)
        self.pi_load = np.asarray(npz[prefix + "pi_load"], dtype=float)
        self.used = used                                                   # {(row, seed): bool}

    def ok(self, *rows) -> bool:
        return all(self.used.get((r, s), False) for r in rows for s in self.seeds)

    def per_instance(self, row: str, metric: str) -> np.ndarray:
        """(S, N) per-instance values of one row and metric."""
        return self.values[self.rows.index(row), :, :, METRICS.index(metric)]

    def per_seed(self, row: str, metric: str) -> list:
        return [float(x) for x in self.per_instance(row, metric).mean(axis=1)]


def arrays_from_report(report: dict, rows, export: dict, names_key: str,
                       cell: str = "1024") -> Arrays:
    """The validation arrays of a run: its report's per-instance arrays of
    `rows` at `cell`, with the per-load relative gaps (`rel`) and energies
    (`pi`) of a post-hoc export of the same states on the same instances
    (`records/cmame/timing/fields/energies_val.npz` in 3D, `arms`;
    `records/cmame/spectra/export/spectra_val.npz` in 2D, `rows`)."""
    cells = report["results"]["e8"]["metrics"]["cells"]
    seeds = [int(s) for s in report["provenance"]["seeds"]]
    sp, n = report["config"]["split"], int(report["provenance"]["datasets"][0]["n_instances"])
    perm = np.random.default_rng(int(sp["seed"])).permutation(n)
    if [str(f) for f in export["files"]] != [f"instance_{int(i):05d}.npz"
                                             for i in perm[:int(sp["n_val"])]]:
        raise ValueError("the export's instances are not the report's validation split")
    V = np.stack([np.stack([np.stack([np.asarray(cells[r][cell]["per_seed_eval"][i]
                                                 ["per_instance"][m], float) for m in METRICS],
                                     axis=-1) for i in range(len(seeds))]) for r in rows])
    names = [str(a) for a in export[names_key]]
    G = np.stack([np.asarray(export["rel"][names.index(r)], float) for r in rows])
    P = np.stack([np.asarray(export["pi"][names.index(r)], float) for r in rows])
    d = {"rows": np.array(rows), "seeds": np.array(seeds), "metrics": np.array(METRICS),
         "values": V, "gap_load": G, "pi_load": P}
    return Arrays(d, {(r, s): True for r in rows for s in seeds})


def resampled(A: Arrays, idx: np.ndarray) -> Arrays:
    """The same rows and seeds on the instances `idx` (the same instances
    for every row and seed)."""
    d = {"rows": np.array(A.rows), "seeds": np.array(A.seeds), "metrics": np.array(METRICS),
         "values": A.values[:, :, idx], "gap_load": A.gap_load[:, :, idx],
         "pi_load": A.pi_load[:, :, idx]}
    return Arrays(d, dict(A.used))


def _better(a: float, b: float, metric: str) -> bool:
    return bool(a > b) if metric in HIGHER_BETTER else bool(a < b)


def _finite(*xs) -> bool:
    return all(np.all(np.isfinite(np.asarray(x, dtype=float))) for x in xs)


def _not_evaluated(why: str) -> dict:
    return {"verdict": NOT_EVALUATED, "why": why}


# ------------------------------------------------------------------- 3D --

def claim_t1(A: Arrays) -> dict:
    if not A.ok("ar", "labels"):
        return _not_evaluated("a label-free or supervised-transformer state failed its "
                              "content check")
    per_metric, ok = {}, True
    for m in METRICS:
        a, b = A.per_seed("ar", m), A.per_seed("labels", m)
        if not _finite(a, b):
            return _not_evaluated(f"non-finite values in {m}")
        seeds = [_better(x, y, m) for x, y in zip(a, b, strict=True)]
        mean = _better(float(np.mean(a)), float(np.mean(b)), m)
        per_metric[m] = {"label_free_per_seed": a, "supervised_per_seed": b,
                         "better_in_seed": seeds, "better_in_mean": mean}
        ok = ok and all(seeds) and mean
    return {"verdict": REPLICATED if ok else NOT_REPLICATED, "per_metric": per_metric}


def claim_t2(A: Arrays, most: float, almost_all: float) -> dict:
    if not A.ok("ar", "mgn"):
        return _not_evaluated("a label-free or graph-network state failed its content check")
    da, dm = A.per_instance("ar", "disp_rel_l2"), A.per_instance("mgn", "disp_rel_l2")
    va, vm = A.per_instance("ar", "vm_rel_l2"), A.per_instance("mgn", "vm_rel_l2")
    if not _finite(da, dm, va, vm):
        return _not_evaluated("non-finite values")
    disp_share = [float(np.mean(dm[s] < da[s])) for s in range(dm.shape[0])]
    vm_share = float(np.mean(vm > va))
    a_ok = all(x > most for x in disp_share)
    b_ok = vm_share >= almost_all
    return {"verdict": REPLICATED if (a_ok and b_ok) else NOT_REPLICATED,
            "disp_lower_share_per_seed": disp_share, "disp_rule": f"> {most:g} in every seed",
            "disp_ok": a_ok, "vm_higher_share_pooled": vm_share,
            "vm_rule": f">= {almost_all:g} of the instance-seed pairs", "vm_ok": b_ok,
            "pairs": int(vm.size)}


def claim_t3(A: Arrays, rows=("labels", "mgn")) -> dict:
    from .holdout import worse_than_zero

    if not A.ok(*rows):
        return _not_evaluated("a supervised state failed its content check")
    per, ok = {}, True
    for r in rows:
        ri = A.rows.index(r)
        w = worse_than_zero(A.gap_load[ri], A.pi_load[ri])
        if w["finite"] != w["load_cases"]:
            return _not_evaluated(f"non-finite per-load values for {r}")
        per[r] = w
        ok = ok and w["worse"] >= 1 and w["sign_mismatch"] == 0
    return {"verdict": REPLICATED if ok else NOT_REPLICATED, "per_row": per,
            "rule": "at least one load case worse than the zero field for each supervised "
                    "network, and Pi_h(u) > 0 on exactly those load cases (an identity of "
                    "the computation)"}


# ------------------------------------------------------------------- 2D --

def _guard_claim(A: Arrays, new: str, base: str, metric: str, band: float, k: float) -> dict:
    if not A.ok(new, base):
        return _not_evaluated(f"a {new} or {base} state failed its content check")
    b, a = A.per_seed(base, metric), A.per_seed(new, metric)
    if not all(math.isfinite(x) for x in b):
        return _not_evaluated(f"a non-finite per-seed value of {base}")
    g = guard(b, a, band=band, k=k)                 # a non-finite new value: diverged
    return {"verdict": REPLICATED if g["lower"] else NOT_REPLICATED, "new": new,
            "base": base, "metric": metric, **g}


def claim_halved(A: Arrays, halved: float, new: str = "labels_knorm", base: str = "labels",
                 metric: str = "vm_rel_l2") -> dict:
    """H2c: rel = mean(new)/mean(base) - 1 of the seed means below `halved`
    (the point value, no guard), with H1, H2a and H2b's non-finite rule."""
    if not A.ok(new, base):
        return _not_evaluated(f"a {new} or {base} state failed its content check")
    b, a = A.per_seed(base, metric), A.per_seed(new, metric)
    if not all(math.isfinite(x) for x in b):
        return _not_evaluated(f"a non-finite per-seed value of {base}")
    out = {"new": new, "base": base, "metric": metric, "threshold": float(halved),
           "base_per_seed": b, "new_per_seed": a}
    if not all(math.isfinite(x) for x in a):
        return {"verdict": NOT_REPLICATED, "diverged": True, "rel_change": None, **out}
    rel = float(np.mean(a) / np.mean(b) - 1.0)
    return {"verdict": REPLICATED if rel < float(halved) else NOT_REPLICATED,
            "diverged": False, "rel_change": rel, "base_mean": float(np.mean(b)),
            "new_mean": float(np.mean(a)), **out}


def claim_d2(A: Arrays) -> dict:
    """The label-free transformer's displacement error higher than L_D's and
    each other metric better, in each seed (seed s against seed s) and in the
    seed mean: 20 comparisons."""
    if not A.ok("ar", "labels"):
        return _not_evaluated("a label-free or L_D state failed its content check")
    per, ok = {}, True
    for m in METRICS:
        a, b = A.per_seed("ar", m), A.per_seed("labels", m)
        if not _finite(a, b):
            return _not_evaluated(f"non-finite values in {m}")
        if m == "disp_rel_l2":
            seeds = [bool(x > y) for x, y in zip(a, b, strict=True)]
            mean = bool(np.mean(a) > np.mean(b))
        else:
            seeds = [_better(x, y, m) for x, y in zip(a, b, strict=True)]
            mean = _better(float(np.mean(a)), float(np.mean(b)), m)
        per[m] = {"label_free_per_seed": a, "supervised_per_seed": b,
                  "label_free_mean": float(np.mean(a)), "supervised_mean": float(np.mean(b)),
                  "expected": "label-free higher" if m == "disp_rel_l2"
                  else "label-free better", "holds_in_seed": seeds, "holds_in_mean": mean}
        ok = ok and all(seeds) and mean
    return {"verdict": REPLICATED if ok else NOT_REPLICATED, "per_metric": per}


def reading_h3(A: Arrays, band: float, k: float) -> dict:
    """PREREG_CM2D's H3 labels: L_K (new) against the label-free row (base)."""
    if not A.ok("labels_knorm", "ar"):
        return {"reading": NOT_EVALUATED}
    b, a = A.per_seed("ar", "energy_gap_rel"), A.per_seed("labels_knorm", "energy_gap_rel")
    if not all(math.isfinite(x) for x in b):
        return {"reading": NOT_EVALUATED}
    g = guard(b, a, band=band, k=k)
    if g["diverged"]:
        label = "the stiffness-norm transformer diverged"
    elif g["lower"]:
        label = "the stiffness-norm transformer lower beyond the guard"
    elif g["worse"]:
        label = "the label-free transformer lower beyond the guard"
    else:
        label = "no difference shown (within the guard)"
    swapped = guard(a, b, band=band, k=k) if all(math.isfinite(x) for x in a) else None
    return {"reading": label, **g, "roles_exchanged": swapped}


# ----------------------------------------------------------- secondary --

def _seeds_ok(A: Arrays, row: str) -> list:
    return [bool(A.used.get((row, s), False)) for s in A.seeds]


def worse_counts(A: Arrays, row: str) -> dict:
    """Load cases of one row worse than the zero field, per seed (each with
    its content check) and over the seeds that passed it, with the sign check
    (`fejepa.analysis.holdout.worse_than_zero`)."""
    from .holdout import worse_than_zero

    ri, ok = A.rows.index(row), _seeds_ok(A, row)
    out = {f"s{s}": {**worse_than_zero(A.gap_load[ri, si], A.pi_load[ri, si]),
                     "content_ok": ok[si]} for si, s in enumerate(A.seeds)}
    keep = [si for si in range(len(A.seeds)) if ok[si]]
    out["all_seeds"] = {**worse_than_zero(A.gap_load[ri][keep], A.pi_load[ri][keep]),
                        "seeds": [A.seeds[si] for si in keep]}
    return out


def mean_gap_above_one(A: Arrays, row: str) -> dict:
    """Instance-seed predictions whose relative energy gap, the mean over the
    instance's load cases, exceeds 1 (the manuscript's count, its Sec. 5.3
    and 6.1), per seed (each with its content check) and over the seeds that
    passed it."""
    g = A.per_instance(row, "energy_gap_rel")
    ok = _seeds_ok(A, row)
    per = [int(np.sum(g[si] > 1.0)) for si in range(g.shape[0])]
    keep = [si for si in range(g.shape[0]) if ok[si]]
    return {"per_seed": per, "content_ok": ok, "total": int(sum(per[si] for si in keep)),
            "pairs": int(g.shape[1] * len(keep)),
            "non_finite": int(np.sum(~np.isfinite(g[keep])))}


def pair_shares(A: Arrays, new: str, base: str) -> dict:
    """Share of instance-seed pairs on which `new` is better than `base`, per metric."""
    out = {}
    for m in METRICS:
        x, y = A.per_instance(new, m), A.per_instance(base, m)
        better = (x > y) if m in HIGHER_BETTER else (x < y)
        out[m] = float(np.mean(better))
    return out


def guard_readings(A: Arrays, new: str, base: str, band: float, k: float) -> dict:
    """The guard on every metric (no criterion), with the guard on per-seed
    medians, a Welch interval and an instance-resampling interval."""
    out = {}
    for m in METRICS:
        b, a = A.per_seed(base, m), A.per_seed(new, m)
        rec = {}
        try:
            rec["guard"] = guard(b, a, band=band, k=k)
            pb, pa = A.per_instance(base, m), A.per_instance(new, m)
            rec["guard_medians"] = guard([float(np.median(x)) for x in pb],
                                         [float(np.median(x)) for x in pa], band=band, k=k)
            rec["welch"] = welch_interval(b, a)
            rec["bootstrap"] = instance_bootstrap(list(pb), list(pa))
        except (ValueError, ZeroDivisionError) as exc:
            rec["not_evaluated"] = f"{type(exc).__name__}: {exc}"
        if m in HIGHER_BETTER and "guard" in rec:
            rec["note"] = "higher is better: 'worse' in the guard's words is better here"
        out[m] = rec
    return out


def set_readings(A: Arrays, name: str, band: float, k: float) -> dict:
    """The secondary readings of one set's arrays (PREREG_HOLDOUT Sec. 6):
    per row and seed for every row and seed, each flagged with its state's
    content check (the counts over the seeds take the seeds that passed);
    the comparisons of two rows only when every seed of both passed."""
    out = {"per_seed_means": {r: {**{m: A.per_seed(r, m) for m in METRICS},
                                  "content_ok": _seeds_ok(A, r)} for r in A.rows},
           "worse_than_zero": {r: worse_counts(A, r) for r in A.rows},
           "mean_gap_above_one": {r: mean_gap_above_one(A, r) for r in A.rows}}
    for new, base in COMPARISONS[name]:
        if A.ok(new, base):
            out[f"pairs_{new}_vs_{base}"] = pair_shares(A, new, base)
            out[f"guard_{new}_vs_{base}"] = guard_readings(A, new, base, band, k)
        else:
            out[f"pairs_{new}_vs_{base}"] = out[f"guard_{new}_vs_{base}"] = \
                {"not_evaluated": "a state of the comparison failed its content check"}
    return out


def ratios(H: Arrays, V: Arrays) -> dict:
    """Per row and metric, the mean over the seeds that passed their content
    check of the holdout set's values over that of the validation values."""
    out = {}
    for r in H.rows:
        keep = [si for si, ok in enumerate(_seeds_ok(H, r)) if ok]
        if not keep:
            continue
        out[r] = {}
        for m in METRICS:
            h = float(np.mean([H.per_seed(r, m)[si] for si in keep]))
            v = float(np.mean([V.per_seed(r, m)[si] for si in keep]))
            out[r][m] = h / v if v != 0 and math.isfinite(h) and math.isfinite(v) else None
    return out


def claims_of(h3d: Arrays | None, h2d: Arrays | None, criteria: dict,
              why: dict | None = None) -> tuple:
    """The eight claims and the reading H3 of one pair of arrays (`why`: per
    set without arrays, the reason its claims are NOT EVALUATED)."""
    band, k = float(criteria["guard_band"]), float(criteria["guard_k"])
    why = why or {}
    claims = {}
    if h3d is None:
        for c in ("T1", "T2", "T3"):
            claims[c] = _not_evaluated(why.get("h3d", "no h3d return"))
    else:
        claims["T1"] = claim_t1(h3d)
        claims["T2"] = claim_t2(h3d, float(criteria["most"]), float(criteria["almost_all"]))
        claims["T3"] = claim_t3(h3d)
    if h2d is None:
        for c in ("H1", "H2a", "H2b", "H2c", "D2"):
            claims[c] = _not_evaluated(why.get("h2d", "no h2d return"))
        readings = {"H3": {"reading": NOT_EVALUATED, "why": why.get("h2d", "no h2d return")}}
    else:
        claims["H1"] = _guard_claim(h2d, "ar", "labels", "energy_gap_rel", band, k)
        claims["H2a"] = _guard_claim(h2d, "labels_knorm", "labels", "energy_gap_rel", band, k)
        claims["H2b"] = _guard_claim(h2d, "labels_knorm", "labels", "vm_rel_l2", band, k)
        claims["H2c"] = claim_halved(h2d, float(criteria["halved"]))
        claims["D2"] = claim_d2(h2d)
        readings = {"H3": reading_h3(h2d, band, k)}
    return {c: claims[c] for c in CLAIMS}, readings


def adjudicate_sets(h3d: Arrays | None, h2d: Arrays | None, criteria: dict,
                    h3d_val: Arrays | None = None, h2d_val: Arrays | None = None,
                    why: dict | None = None) -> dict:
    """Every claim and reading of PREREG_HOLDOUT Sec. 5-6 from the two sets'
    arrays (a set that did not return or is not adjudicable leaves its claims
    NOT EVALUATED, for the reason in `why`); with the validation arrays,
    every secondary reading beside its validation value, the
    holdout-over-validation ratios, and the claims' rules applied to the
    validation instances for reference (no verdict)."""
    band, k = float(criteria["guard_band"]), float(criteria["guard_k"])
    claims, readings = claims_of(h3d, h2d, criteria, why)
    secondary = {}
    for name, H, V in (("h3d", h3d, h3d_val), ("h2d", h2d, h2d_val)):
        if H is None:
            continue
        sec = {"holdout": set_readings(H, name, band, k)}
        if V is not None:
            sec["validation"] = set_readings(V, name, band, k)
            sec["ratio_holdout_over_validation"] = ratios(H, V)
        secondary[name] = sec
    tally = {v: sum(1 for c in claims.values() if c["verdict"] == v)
             for v in (REPLICATED, NOT_REPLICATED, NOT_EVALUATED)}
    out = {"claims": claims, "readings": readings, "secondary": secondary, "tally": tally}
    if h3d_val is not None or h2d_val is not None:
        vc, vr = claims_of(h3d_val, h2d_val, criteria)
        out["validation_reference"] = {
            "what": "the rules of Sec. 5 applied to the validation instances that this run "
                    "recomputed (no verdict)",
            "claims": {c: v["verdict"] for c, v in vc.items()},
            "H3": vr["H3"]["reading"]}
    return out
