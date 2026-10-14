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

And V (not a claim of the manuscript): whether the per-instance values of
each row and metric (the mean over the row's seeds) are distributed alike on
the holdout set and on the validation instances, by a two-sided
Mann-Whitney test per quantity (which detects a difference in the share of
holdout values above validation values, not one in the means) and Holm's
procedure at a family-wise level. REPRESENTATIVE or NOT REPRESENTATIVE. The
holdout-over-validation ratios of the means, with two resampling intervals,
are readings beside it.
"""

from __future__ import annotations

import math

import numpy as np

from .adjudicate_w9 import guard, instance_bootstrap, welch_interval
from .holdout import HIGHER_BETTER, METRICS

REPLICATED, NOT_REPLICATED, NOT_EVALUATED = "REPLICATED", "NOT REPLICATED", "NOT EVALUATED"
REPRESENTATIVE, NOT_REPRESENTATIVE = "REPRESENTATIVE", "NOT REPRESENTATIVE"
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


def _all_pairs(a: list, b: list, metric: str, higher: bool = False) -> dict:
    """The pairing-free reading of a per-seed comparison (a reading, no
    verdict): `a` better than `b` (or, with `higher`, `a` higher than `b`)
    for every one of the 3 x 3 pairs of seeds, and the relative difference
    x / y - 1 of the least favourable pair (the closest when every pair
    holds; negative when `a`'s value is the lower; None when y is zero)."""
    if higher or metric in HIGHER_BETTER:
        holds, x, y = min(a) > max(b), min(a), max(b)
    else:
        holds, x, y = max(a) < min(b), max(a), min(b)
    return {"better_all_pairs": bool(holds),
            "worst_pair_rel": float(x / y - 1.0) if y != 0 else None}


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
                         "better_in_seed": seeds, "better_in_mean": mean,
                         **_all_pairs(a, b, m)}
        ok = ok and all(seeds) and mean
    return {"verdict": REPLICATED if ok else NOT_REPLICATED, "per_metric": per_metric,
            "all_pairs_reading": all(v["better_all_pairs"] for v in per_metric.values())}


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
        pf = _all_pairs(a, b, m, higher=m == "disp_rel_l2")
        per[m] = {"label_free_per_seed": a, "supervised_per_seed": b,
                  "label_free_mean": float(np.mean(a)), "supervised_mean": float(np.mean(b)),
                  "expected": "label-free higher" if m == "disp_rel_l2"
                  else "label-free better", "holds_in_seed": seeds, "holds_in_mean": mean,
                  "holds_all_pairs": pf["better_all_pairs"],
                  "worst_pair_rel": pf["worst_pair_rel"]}
        ok = ok and all(seeds) and mean
    return {"verdict": REPLICATED if ok else NOT_REPLICATED, "per_metric": per,
            "all_pairs_reading": all(v["holds_all_pairs"] for v in per.values())}


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


def ratio_intervals(H: Arrays, V: Arrays, n_boot: int = 2000, seed: int = 0,
                    chunk: int = 200) -> dict:
    """Per row and metric, the holdout mean over the validation mean (the
    per-instance means over the seeds that passed their content check, as
    `ratios`), with two 95% percentile intervals from `n_boot` instance
    resamples with replacement (`seed` fixed; drawn `chunk` at a time; the
    same draws for every row and metric): `interval_95_holdout`, the holdout
    set alone resampled over the validation mean as it is (how far that mean
    is from the distribution's, as the holdout set estimates it), and
    `interval_95_both`, each side resampled on its own (how large a
    difference between the two sets' means their sampling leaves possible;
    wide where a mean is carried by a few instances). None where a value is
    not finite or the validation mean is zero. Readings, no verdict."""
    quantities = []
    for r in H.rows:
        keep = [si for si, ok in enumerate(_seeds_ok(H, r)) if ok]
        for m in METRICS if keep else ():
            h = H.per_instance(r, m)[keep].mean(axis=0)
            v = V.per_instance(r, m)[keep].mean(axis=0)
            ok = bool(np.all(np.isfinite(h)) and np.all(np.isfinite(v)) and v.mean() != 0)
            quantities.append((r, m, h, v, ok))
    rng = np.random.default_rng(seed)
    nh, nv = H.values.shape[2], V.values.shape[2]
    both = {(r, m): [] for r, m, _, _, ok in quantities if ok}
    alone = {(r, m): [] for r, m, _, _, ok in quantities if ok}
    for b in [chunk] * (n_boot // chunk) + ([n_boot % chunk] if n_boot % chunk else []):
        ih, iv = rng.integers(0, nh, size=(b, nh)), rng.integers(0, nv, size=(b, nv))
        for r, m, h, v, ok in quantities:
            if ok:
                mh = h[ih].mean(axis=1)
                alone[(r, m)].append(mh / v.mean())
                with np.errstate(divide="ignore", invalid="ignore"):
                    both[(r, m)].append(mh / v[iv].mean(axis=1))
    out = {}
    for r, m, h, v, ok in quantities:
        if not ok:
            out.setdefault(r, {})[m] = None
            continue
        x, y = np.concatenate(both[(r, m)]), np.concatenate(alone[(r, m)])
        out.setdefault(r, {})[m] = {
            "ratio": float(h.mean() / v.mean()),
            "interval_95_holdout": [float(q) for q in np.quantile(y, [0.025, 0.975])],
            "interval_95_both": [float(q) for q in np.quantile(x[np.isfinite(x)],
                                                               [0.025, 0.975])],
            "resamples": int(n_boot), "non_finite_resamples": int(np.sum(~np.isfinite(x)))}
    return out


def pairing_free(res: dict) -> dict | None:
    """The pairing-free reading of a T1 or D2 result (None when the claim was
    not evaluated): per metric whether every one of the nine pairs of seeds
    holds, and the least favourable pair's relative difference."""
    if "per_metric" not in res:
        return None
    keys = ("better_all_pairs", "holds_all_pairs", "worst_pair_rel")
    return {"all_pairs_reading": res["all_pairs_reading"],
            "per_metric": {m: {k: r[k] for k in keys if k in r}
                           for m, r in res["per_metric"].items()}}


def failing_components(claim: str, res: dict) -> list:
    """The comparisons that a claim read NOT REPLICATED failed: per metric and
    seed or the seed mean (T1, D2), the displacement or von Mises part (T2),
    the supervised network (T3); the claim itself for the single comparisons
    (H1, H2a, H2b, H2c). Empty for any other verdict."""
    if res.get("verdict") != NOT_REPLICATED:
        return []
    if claim in ("T1", "D2"):
        out = []
        for m, r in res["per_metric"].items():
            seeds = r.get("better_in_seed", r.get("holds_in_seed"))
            mean = r.get("better_in_mean", r.get("holds_in_mean"))
            out += [f"{claim} {m} seed {i}" for i, ok in enumerate(seeds) if not ok]
            out += [f"{claim} {m} seed mean"] if not mean else []
        return out
    if claim == "T2":
        return [f"T2 {k}" for k in ("disp_ok", "vm_ok") if not res.get(k)]
    if claim == "T3":
        return [f"T3 {r}" for r, w in res.get("per_row", {}).items()
                if not (w["worse"] >= 1 and w["sign_mismatch"] == 0)]
    return [claim]


def holm(pvalues: dict, alpha: float) -> dict:
    """Holm's step-down procedure at family-wise level `alpha`: {name:
    rejected}. The smallest p-value is compared with alpha / m, the next with
    alpha / (m - 1), and so on: each at or below its level is rejected, and
    the procedure stops at the first above it (ties broken by name, so that
    the result does not depend on the order of the dictionary; a p-value
    that is not a number is never rejected)."""
    items = sorted(pvalues.items(),
                   key=lambda kv: (kv[1] if math.isfinite(kv[1]) else math.inf, kv[0]))
    out = {k: False for k in pvalues}
    for i, (k, p) in enumerate(items):
        if not p <= alpha / (len(items) - i):
            break
        out[k] = True
    return out


def representativeness(sets: dict, alpha: float) -> dict:
    """V (PREREG_HOLDOUT Sec. 5): for each set that returned and is
    adjudicable (`sets`: {name: (holdout Arrays, validation Arrays)}), row
    and metric, the per-instance values (the mean over the row's seeds that
    passed their content check) on the holdout set against those on the
    validation instances, by a two-sided Mann-Whitney test (asymptotic, with
    the tie and continuity corrections; one call per set, which equals the
    calls per quantity); Holm's procedure over all the quantities at
    family-wise level `alpha`. A row without a seed that passed, and a
    quantity with a non-finite per-instance value on either side (counted per
    side), are left out of the family and listed: they are not read.
    REPRESENTATIVE iff no quantity is rejected; `holdout_worse` is decided by
    the share of holdout values above validation values (ties counting
    half), not by the means."""
    from scipy.stats import mannwhitneyu

    quantities, left_out = {}, []
    for name in sorted(sets):
        H, V = sets[name]
        keys, hs, vs, seeds = [], [], [], {}
        for r in H.rows:
            keep = [si for si, ok in enumerate(_seeds_ok(H, r)) if ok]
            if not keep:
                left_out.append({"quantity": f"{name}:{r}",
                                 "why": "no state of the row passed its content check"})
                continue
            for m in METRICS:
                h = H.per_instance(r, m)[keep].mean(axis=0)
                v = V.per_instance(r, m)[keep].mean(axis=0)
                if not (np.all(np.isfinite(h)) and np.all(np.isfinite(v))):
                    left_out.append({"quantity": f"{name}:{r}:{m}",
                                     "why": "non-finite per-instance values",
                                     "non_finite_holdout": int(np.sum(~np.isfinite(h))),
                                     "non_finite_validation": int(np.sum(~np.isfinite(v)))})
                    continue
                keys.append((r, m))
                hs.append(h)
                vs.append(v)
                seeds[r] = [H.seeds[si] for si in keep]
        if not keys:
            continue
        hs, vs = np.stack(hs), np.stack(vs)
        test = mannwhitneyu(hs, vs, axis=1, alternative="two-sided", method="asymptotic",
                            use_continuity=True)
        for (r, m), h, v, u, p in zip(keys, hs, vs, np.atleast_1d(test.statistic),
                                     np.atleast_1d(test.pvalue), strict=True):
            share = float(u) / (h.size * v.size)          # P(H > V) + P(H = V) / 2
            quantities[f"{name}:{r}:{m}"] = {
                "seeds": seeds[r], "n_holdout": int(h.size), "n_validation": int(v.size),
                "holdout_mean": float(h.mean()), "validation_mean": float(v.mean()),
                "ratio_of_means": float(h.mean() / v.mean()) if v.mean() != 0 else None,
                "holdout_median": float(np.median(h)),
                "validation_median": float(np.median(v)), "u": float(u),
                "prob_holdout_higher": share, "p": float(p),
                "holdout_worse": share != 0.5 and (share > 0.5) != (m in HIGHER_BETTER)}
    out = {"what": "V: the per-instance values of every row and metric on the holdout set "
                   "against the validation instances (two-sided Mann-Whitney, Holm)",
           "alpha": float(alpha), "sets": sorted(sets), "left_out": left_out}
    if not quantities:
        return {**out, "verdict": NOT_EVALUATED, "why": "no quantity could be read",
                "quantities": {}, "rejected": [], "rejected_holdout_worse": []}
    rej = holm({k: q["p"] for k, q in quantities.items()}, alpha)
    for k, q in quantities.items():
        q["rejected"] = rej[k]
    rejected = sorted(k for k, x in rej.items() if x)
    return {**out, "verdict": NOT_REPRESENTATIVE if rejected else REPRESENTATIVE,
            "quantities": quantities, "rejected": rejected,
            "rejected_holdout_worse": [k for k in rejected if quantities[k]["holdout_worse"]]}


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
    for c, v in claims.items():
        v["failing"] = failing_components(c, v)
    return {c: claims[c] for c in CLAIMS}, readings


def adjudicate_sets(h3d: Arrays | None, h2d: Arrays | None, criteria: dict,
                    h3d_val: Arrays | None = None, h2d_val: Arrays | None = None,
                    why: dict | None = None) -> dict:
    """Every claim and reading of PREREG_HOLDOUT Sec. 5-6 from the two sets'
    arrays (a set that did not return or is not adjudicable leaves its claims
    NOT EVALUATED, for the reason in `why`); with the validation arrays, V
    over the sets that have both, every secondary reading beside its
    validation value, the holdout-over-validation ratios with their
    resampling intervals, and the claims' rules applied to the validation
    instances for reference, with T1's and D2's pairing-free readings (no
    verdict)."""
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
            sec["ratio_interval"] = ratio_intervals(H, V)
        secondary[name] = sec
    tally = {v: sum(1 for c in claims.values() if c["verdict"] == v)
             for v in (REPLICATED, NOT_REPLICATED, NOT_EVALUATED)}
    pairs = {name: (H, V) for name, H, V in (("h3d", h3d, h3d_val), ("h2d", h2d, h2d_val))
             if H is not None and V is not None}
    if pairs:
        v = representativeness(pairs, float(criteria["representative_alpha"]))
    else:
        v = {"verdict": NOT_EVALUATED, "why": "no set returned with its validation arrays"}
    out = {"claims": claims, "readings": readings, "V": v, "secondary": secondary,
           "tally": tally}
    if h3d_val is not None or h2d_val is not None:
        vc, vr = claims_of(h3d_val, h2d_val, criteria)
        out["validation_reference"] = {
            "what": "the rules of Sec. 5 applied to the validation instances that this run "
                    "recomputed (no verdict)",
            "claims": {c: v["verdict"] for c, v in vc.items()},
            "H3": vr["H3"]["reading"],
            "pairing_free": {c: pairing_free(vc[c]) for c in ("T1", "D2")}}
    return out
