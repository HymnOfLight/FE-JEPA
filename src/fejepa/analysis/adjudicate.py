"""Executable adjudication rules of PREREG_E1 Sec. 5 and PREREG_E2 Sec. 4."""

from __future__ import annotations

import math
import statistics
from pathlib import Path


def _finite(vals, what: str) -> list:
    """Stage 1.31: every per-seed input must be a finite number -- a NaN makes
    `change > threshold` False and max(band, NaN) silently return the band, so
    a diverged seed would pass parity instead of being refused."""
    out = [float(v) for v in vals]
    bad = [v for v in out if not math.isfinite(v)]
    if bad or not out:
        raise ValueError(f"adjudication refused: {what} has non-finite or missing values "
                         f"({out}); a diverged or unmeasured seed is not adjudicated")
    return out


def require_stamped(report: dict, role: str, prereg_name: str | None = None) -> str:
    """Stage 1.31: a verdict is issued only on a stamped, guard-verified run --
    the report's verified pre-registration hash must equal its own config
    SHA-256 (and, when given, be verified against `prereg_name`)."""
    pr = report.get("prereg") or {}
    got = (report.get("provenance") or {}).get("config_sha256")
    if not pr.get("config_sha256") or pr.get("config_sha256") != got:
        raise ValueError(f"adjudication refused: the {role} is not a stamped, guard-verified "
                         f"run (verified {str(pr.get('config_sha256'))[:12]} vs config "
                         f"{str(got)[:12]})")
    if prereg_name and Path(str(pr.get("file", ""))).name != prereg_name:
        raise ValueError(f"adjudication refused: the {role} was verified against "
                         f"{pr.get('file')!r}, not {prereg_name}")
    return str(got)


def ar_per_seed(report: dict, metric: str) -> list:
    """Per-seed values of the AR cell at the largest pool size."""
    cells = report["results"]["e8"]["metrics"]["cells"]["ar"]
    cell = cells[max(cells, key=int)]
    return [float(v) for v in cell[metric]["per_seed"]]


def transfer_ratio(report: dict):
    p3 = (report["results"].get("p3_transfer") or {}).get("metrics")
    if not p3:
        return None
    return float(p3["ar"]["fine_disp_mean"] / (p3["ar"]["inband_disp_mean"] + 1e-30))


def _rel_change(new: float, old: float) -> float:
    return new / (old + 1e-30) - 1.0


NOISE_GUARD_K = 2.0
"""Multiplier of the standard error in the parity kills (PI decision, 29 Sep
2026, recorded in PREREG_E1 r13 / PREREG_E2 r7)."""


def noise_guarded_worse(base_vals, new_vals, band: float = 0.10,
                        k: float = NOISE_GUARD_K) -> dict:
    """Parity-kill test on seed means (the Phase-2 aggregation rule) with a
    noise guard: fires iff the relative change of the seed means exceeds
    max(band, k x SE_rel), where SE_rel is the standard error of the
    difference of the two seed means -- sample SDs (n - 1) of both arms,
    sqrt(s_b^2/n_b + s_n^2/n_n) -- divided by the baseline mean. `threshold`
    is the resolution actually achieved and is reported with the verdict."""
    b = _finite(base_vals, "the baseline's per-seed values")
    a = _finite(new_vals, "the new arm's per-seed values")
    mb, ma = statistics.fmean(b), statistics.fmean(a)
    var = ((statistics.variance(b) / len(b)) if len(b) > 1 else 0.0) + \
          ((statistics.variance(a) / len(a)) if len(a) > 1 else 0.0)
    se_rel = var ** 0.5 / (abs(mb) + 1e-30)
    tau = max(float(band), float(k) * se_rel)
    change = _rel_change(ma, mb)
    return {"base_mean": mb, "new_mean": ma, "rel_change": change, "se_rel": se_rel,
            "threshold": tau, "fires": bool(change > tau)}


PHASE2B_CONFIG_SHA256 = "316f5e6e282db9c11509d8d1d2ed54d229ded9d6367c63b3a13abd7709ba4899"
"""E2's baseline (PREREG_PHASE2B r2 stamp). The Phase-2 report (config
e3bdd1e8...) must never serve as a baseline: its AR cells are the D14
instrument defect (prediction = u* x fscale), against which any architecture
passes parity trivially."""


def _signature(report: dict, model_ignore=("kind", "n_tokens")) -> dict:
    """Everything that shapes an AR cell except the architecture kind and the
    loss specification: corpus identity (manifest SHA-256s), split, seeds,
    labels, schedule, numeric policy, model dims and features."""
    cfg = report.get("config") or {}
    e8 = (cfg.get("experiments") or {}).get("e8") or {}
    pre = {k: v for k, v in (cfg.get("pretrain") or {}).items() if k != "loss_spec"}
    prov = report.get("provenance") or {}
    return {"datasets": [(d.get("dir"), d.get("manifest_sha256"))
                         for d in prov.get("datasets", [])],
            "seeds": prov.get("seeds"),
            "data": cfg.get("data"), "data_transfer": cfg.get("data_transfer"),
            "split": cfg.get("split"), "labels": cfg.get("labels"),
            "pretrain": pre, "tf32": cfg.get("tf32"), "runtime": cfg.get("runtime"),
            "model": {k: v for k, v in (cfg.get("model") or {}).items()
                      if k not in model_ignore},
            # Stage 1.31: ar_lr is the learning rate the E8 AR units actually use
            "e8": {k: e8.get(k) for k in ("pool_sizes", "ar_epochs", "ar_lr", "seeds")}}


def comparability(base: dict, other: dict) -> list:
    """Keys on which two reports differ beyond architecture kind and loss spec
    (empty = comparable). A verdict on incomparable reports is refused."""
    sb, so = _signature(base), _signature(other)
    return sorted(k for k in sb if sb[k] != so[k])


def adjudicate_e1(base: dict, shaped: dict, s_base: list, s_shaped: list,
                  band: float = 0.10, min_effect_abs: float = 0.02,
                  min_effect_seed_sd: float = 2.0) -> dict:
    """K1 parity (PREREG_E1 r13): the shaped arm's seed-mean displacement error
    or seed-mean energy gap is worse than the AR arm's by more than
    max(band, 2 x SE_rel) (`noise_guarded_worse`); K2 no effect (S delta <= 0
    at every seed); GO = no K1 and S improves at every seed by at least the
    pre-declared effect floor -- max(min_effect_abs, min_effect_seed_sd x the
    AR arm's seed-to-seed sample standard deviation of S) -- and the transfer
    ratio does not worsen beyond `band`. Without the floor, three same-signed
    noise-level deltas would pass as GO.

    Stage 1.31 refusals: reports that are not stamped runs of PREREG_E1.md;
    a base report that carries a loss specification or a shaped report that
    is not the SIGReg(head) arm (swapped inputs flip the sign of every kill);
    non-finite per-seed metrics."""
    bad = [v for v in list(s_base) + list(s_shaped) if not math.isfinite(float(v))]
    if bad:
        raise ValueError("E1 adjudication refused: a separation statistic is not finite "
                         f"({len(bad)} value(s)); the measurement is invalid (bins with "
                         "< 2 instances?) -- fix the measurement, do not adjudicate")
    require_stamped(base, "E1 base report", "PREREG_E1.md")
    require_stamped(shaped, "E1 shaped report", "PREREG_E1.md")
    base_spec = ((base.get("config") or {}).get("pretrain") or {}).get("loss_spec")
    spec = ((shaped.get("config") or {}).get("pretrain") or {}).get("loss_spec") or {}
    if base_spec is not None:
        raise ValueError(f"E1 adjudication refused: the base report carries a loss "
                         f"specification {base_spec} -- it is not the AR arm (inputs swapped?)")
    lam, width = spec.get("lambda_reg"), spec.get("sigreg_head_width")
    if (spec.get("reg_mode") != "sigreg_ep_head" or not isinstance(lam, (int, float))
            or not math.isfinite(float(lam)) or float(lam) <= 0.0
            or not isinstance(width, int) or width < 0):
        raise ValueError(f"E1 adjudication refused: the shaped report is not the AR+SIGReg(head) "
                         f"arm with a filled lambda and head width (loss_spec {spec})")
    diff = comparability(base, shaped)
    if diff:
        raise ValueError(f"E1 adjudication refused: the two reports differ beyond the loss "
                         f"specification on {diff} -- not the same corpus/split/seeds/schedule")
    per_seed = []                                  # informational since r13
    for i, (db, ds, eb, es) in enumerate(zip(ar_per_seed(base, "disp_rel_l2"),
                                             ar_per_seed(shaped, "disp_rel_l2"),
                                             ar_per_seed(base, "energy_gap_rel"),
                                             ar_per_seed(shaped, "energy_gap_rel"), strict=True)):
        per_seed.append({"seed": i, "disp_rel_change": _rel_change(ds, db),
                         "egap_rel_change": _rel_change(es, eb)})
    k1_detail = {m: noise_guarded_worse(ar_per_seed(base, m), ar_per_seed(shaped, m), band)
                 for m in ("disp_rel_l2", "energy_gap_rel")}
    k1 = any(d["fires"] for d in k1_detail.values())
    deltas = [b - a for a, b in zip(s_base, s_shaped, strict=True)]
    k2 = all(d <= 0.0 for d in deltas)
    # Stage 1.28: the SAMPLE standard deviation (n - 1), pinned in PREREG_E1
    # r12; the population form (n) understated the floor by sqrt(2/3) at 3 seeds
    sd_base = statistics.stdev(s_base) if len(s_base) > 1 else 0.0
    floor = max(float(min_effect_abs), float(min_effect_seed_sd) * sd_base)
    above_floor = all(d >= floor for d in deltas)
    rb, rs = transfer_ratio(base), transfer_ratio(shaped)
    if rb is None or rs is None:
        ratio_ok, guard = True, "not evaluated (no P3 transfer block in a run; 2D stage)"
    else:
        ratio_ok = _rel_change(rs, rb) <= band
        guard = "passed" if ratio_ok else f"failed (ratio worsened by {_rel_change(rs, rb):.3f} > {band})"
    go = (not k1) and above_floor and ratio_ok
    return {"band": band, "per_seed": per_seed, "shaped_loss_spec": spec,
            "K1_rule": f"seed means; kill iff relative change > max({band}, "
                       f"{NOISE_GUARD_K} x SE_rel of the difference) (PREREG_E1 r13)",
            "K1_detail": k1_detail,
            "S_base": s_base, "S_shaped": s_shaped,
            "S_delta": deltas, "S_effect_floor": floor, "S_base_seed_sd": sd_base,
            "S_base_seed_sd_ddof": 1, "comparability": "identical beyond loss_spec",
            "S_above_floor_all_seeds": above_floor,
            "transfer_ratio_base": rb, "transfer_ratio_shaped": rs,
            "transfer_guard": guard,
            "K1_parity": k1, "K2_no_effect": k2, "GO": go,
            "verdict": "GO" if go else ("KILLED" if (k1 or k2) else "NO-GO")}


def bench_fine_step_s(bench: dict, m_tokens: int) -> tuple:
    """The set-up-free fine-scale step time (s) of the bottleneck at M tokens,
    with every precondition PREREG_E2 Sec. 4 puts on the bench (Stage 1.31):
    the phase exists, is differential and valid (every repeated estimate
    positive), was measured on CUDA outside smoke mode at this M, and the
    step time is a finite positive number. Returns (seconds, phase)."""
    tag = f"bottleneck{int(m_tokens)}_fine"
    phase = (bench.get("phases") or {}).get(tag)
    if phase is None:
        raise ValueError(f"E2 adjudication refused: the bench has no {tag} phase "
                         "(bench this M with --bottleneck-tokens M)")
    if phase.get("timing") != "differential":
        raise ValueError("E2 adjudication refused: the bench's bottleneck phase is not "
                         "setup-free (re-bench with Stage >= 1.28: differential timing)")
    if phase.get("valid") is not True:
        raise ValueError("E2 adjudication refused: the bench's differential estimate is not "
                         f"valid (estimates {phase.get('estimates_ms')} ms; re-bench with "
                         "Stage >= 1.31 on an idle GPU)")
    if bench.get("device") != "cuda" or bench.get("smoke"):
        raise ValueError(f"E2 adjudication refused: the bench ran on {bench.get('device')!r}"
                         f"{' in smoke mode' if bench.get('smoke') else ''}, not the box's GPU")
    if int(phase.get("n_tokens", -1)) != int(m_tokens):
        raise ValueError(f"E2 adjudication refused: the {tag} phase records "
                         f"n_tokens={phase.get('n_tokens')}")
    ms = float(phase.get("ms_per_step", float("nan")))
    if not math.isfinite(ms) or ms <= 0.0:
        raise ValueError(f"E2 adjudication refused: fine step time {ms} ms is not a positive "
                         "finite number")
    return ms / 1000.0, phase


def adjudicate_e2(base: dict, e2: dict, bench: dict, m_tokens: int, band: float = 0.10,
                  kill_s: float = 2.0, go_s: float = 1.0,
                  expect_base_config_sha: str | None = PHASE2B_CONFIG_SHA256) -> dict:
    """K1 accuracy (PREREG_E2 r7): the in-band energy gap or the fine zero-shot
    displacement, compared on seed means, worse than the baseline by more than
    max(band, 2 x SE_rel) (`noise_guarded_worse`; the achieved resolution is
    reported); K2 speed (set-up-free fine step time not below `kill_s`); GO =
    no K1 and fine step time below `go_s`. Refuses a baseline other than the
    Phase-2b report, reports that are not comparable, and a bench without
    set-up-free timing.

    Stage 1.31 refusals: an E2 report that is not a stamped run of
    PREREG_E2.md (and an unstamped baseline); a bench without the
    bottleneck<M>_fine phase (formerly read as "no speed case" = KILLED),
    without a valid differential estimate, not measured on CUDA, a smoke
    bench, or a phase of another M; a non-finite or non-positive step time;
    non-finite per-seed metrics."""
    got = (base.get("provenance") or {}).get("config_sha256")
    if expect_base_config_sha and got != expect_base_config_sha:
        raise ValueError(f"E2 adjudication refused: the baseline report's config SHA-256 is "
                         f"{str(got)[:12]}..., expected {expect_base_config_sha[:12]}... "
                         "(the Phase-2b report; the Phase-2 AR cells are D14-invalid)")
    require_stamped(base, "E2 baseline report")
    require_stamped(e2, "E2 report", "PREREG_E2.md")
    step_s, bench_phase = bench_fine_step_s(bench, m_tokens)
    mk = (e2.get("config") or {}).get("model") or {}
    if mk.get("kind") != "bottleneck" or int(mk.get("n_tokens", -1)) != int(m_tokens):
        raise ValueError(f"E2 adjudication refused: the E2 report is kind={mk.get('kind')!r} "
                         f"n_tokens={mk.get('n_tokens')}, expected bottleneck M={m_tokens}")
    diff = comparability(base, e2)
    if diff:
        raise ValueError(f"E2 adjudication refused: baseline and E2 reports differ beyond the "
                         f"architecture on {diff}")

    def fine_per_seed(rep):
        return [float(v) for v in
                rep["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["disp_rel_l2"]["per_seed"]]

    eg = noise_guarded_worse(ar_per_seed(base, "energy_gap_rel"),
                             ar_per_seed(e2, "energy_gap_rel"), band)
    fd = noise_guarded_worse(fine_per_seed(base), fine_per_seed(e2), band)
    spread = {"egap_seed_sd_base": float(statistics.stdev(ar_per_seed(base, "energy_gap_rel"))),
              "egap_seed_sd_e2": float(statistics.stdev(ar_per_seed(e2, "energy_gap_rel")))}
    k1 = eg["fires"] or fd["fires"]
    k2 = step_s >= kill_s
    go = (not k1) and step_s < go_s
    return {"M": m_tokens, "band": band, "kill_s": kill_s, "go_s": go_s,
            "bench_fine_phase": {k: bench_phase.get(k) for k in
                                 ("n_nodes", "n_tokens", "ms_per_step", "estimates_ms",
                                  "steps", "pairs", "prepare_ms", "peak_gib")},
            "bench_git": bench.get("git"),
            "egap_mean_base": eg["base_mean"], "egap_mean_e2": eg["new_mean"],
            "egap_rel_change": eg["rel_change"], "egap_threshold": eg["threshold"],
            "fine_disp_base": fd["base_mean"], "fine_disp_e2": fd["new_mean"],
            "fine_disp_rel_change": fd["rel_change"], "fine_disp_threshold": fd["threshold"],
            "fine_step_s": step_s, **spread,
            "aggregation": "seed means with noise guard (both K1 quantities)",
            "K1_rule": f"kill iff relative change of seed means > max({band}, "
                       f"{NOISE_GUARD_K} x SE_rel of the difference) (PREREG_E2 r7)",
            "resolution": {"egap": eg["threshold"], "fine_disp": fd["threshold"]},
            "base_config_sha256": got, "comparability": "identical beyond the architecture",
            "K1_accuracy": k1, "K2_speed": k2, "GO": go,
            "verdict": "GO" if go else ("KILLED" if (k1 or k2) else "NO-GO")}
