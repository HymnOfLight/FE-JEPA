"""Executable adjudication rules of PREREG_E1 Sec. 5 and PREREG_E2 Sec. 4."""

from __future__ import annotations

import statistics


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
            "e8": {k: e8.get(k) for k in ("pool_sizes", "ar_epochs", "seeds")}}


def comparability(base: dict, other: dict) -> list:
    """Keys on which two reports differ beyond architecture kind and loss spec
    (empty = comparable). A verdict on incomparable reports is refused."""
    sb, so = _signature(base), _signature(other)
    return sorted(k for k in sb if sb[k] != so[k])


def adjudicate_e1(base: dict, shaped: dict, s_base: list, s_shaped: list,
                  band: float = 0.10, min_effect_abs: float = 0.02,
                  min_effect_seed_sd: float = 2.0) -> dict:
    """K1 parity per seed (disp or energy gap worse than `band`); K2 no effect
    (S delta <= 0 at every seed); GO = S improves at every seed by at least the
    pre-declared effect floor -- max(min_effect_abs, min_effect_seed_sd x the
    AR arm's seed-to-seed standard deviation of S) -- and the transfer ratio
    does not worsen beyond `band`. Without the floor, three same-signed
    noise-level deltas would pass as GO."""
    import math

    bad = [v for v in list(s_base) + list(s_shaped) if not math.isfinite(float(v))]
    if bad:
        raise ValueError("E1 adjudication refused: a separation statistic is not finite "
                         f"({len(bad)} value(s)); the measurement is invalid (bins with "
                         "< 2 instances?) -- fix the measurement, do not adjudicate")
    diff = comparability(base, shaped)
    if diff:
        raise ValueError(f"E1 adjudication refused: the two reports differ beyond the loss "
                         f"specification on {diff} -- not the same corpus/split/seeds/schedule")
    per_seed, k1 = [], False
    for i, (db, ds, eb, es) in enumerate(zip(ar_per_seed(base, "disp_rel_l2"),
                                             ar_per_seed(shaped, "disp_rel_l2"),
                                             ar_per_seed(base, "energy_gap_rel"),
                                             ar_per_seed(shaped, "energy_gap_rel"), strict=True)):
        cd, ce = _rel_change(ds, db), _rel_change(es, eb)
        k1 |= (cd > band) or (ce > band)
        per_seed.append({"seed": i, "disp_rel_change": cd, "egap_rel_change": ce})
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
    return {"band": band, "per_seed": per_seed, "S_base": s_base, "S_shaped": s_shaped,
            "S_delta": deltas, "S_effect_floor": floor, "S_base_seed_sd": sd_base,
            "S_base_seed_sd_ddof": 1, "comparability": "identical beyond loss_spec",
            "S_above_floor_all_seeds": above_floor,
            "transfer_ratio_base": rb, "transfer_ratio_shaped": rs,
            "transfer_guard": guard,
            "K1_parity": k1, "K2_no_effect": k2, "GO": go,
            "verdict": "GO" if go else ("KILLED" if (k1 or k2) else "NO-GO")}


def adjudicate_e2(base: dict, e2: dict, bench: dict, m_tokens: int, band: float = 0.10,
                  kill_s: float = 2.0, go_s: float = 1.0,
                  expect_base_config_sha: str | None = PHASE2B_CONFIG_SHA256) -> dict:
    """K1 accuracy (in-band energy gap, or fine zero-shot displacement, worse
    than the baseline by more than `band`, BOTH at the seed median -- PREREG_E2
    Sec. 4); K2 speed (fine step time not below `kill_s`); GO = parity and fine
    step time below `go_s`. Refuses a baseline other than the Phase-2b report,
    reports that are not comparable, and a bench without setup-free timing."""
    got = (base.get("provenance") or {}).get("config_sha256")
    if expect_base_config_sha and got != expect_base_config_sha:
        raise ValueError(f"E2 adjudication refused: the baseline report's config SHA-256 is "
                         f"{str(got)[:12]}..., expected {expect_base_config_sha[:12]}... "
                         "(the Phase-2b report; the Phase-2 AR cells are D14-invalid)")
    mk = (e2.get("config") or {}).get("model") or {}
    if mk.get("kind") != "bottleneck" or int(mk.get("n_tokens", -1)) != int(m_tokens):
        raise ValueError(f"E2 adjudication refused: the E2 report is kind={mk.get('kind')!r} "
                         f"n_tokens={mk.get('n_tokens')}, expected bottleneck M={m_tokens}")
    diff = comparability(base, e2)
    if diff:
        raise ValueError(f"E2 adjudication refused: baseline and E2 reports differ beyond the "
                         f"architecture on {diff}")

    def ar_median(rep, metric):
        return float(statistics.median(ar_per_seed(rep, metric)))

    def fine_disp(rep):
        return float(statistics.median(
            rep["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["disp_rel_l2"]["per_seed"]))

    eg_b, eg_e = ar_median(base, "energy_gap_rel"), ar_median(e2, "energy_gap_rel")
    spread = {"egap_seed_sd_base": float(statistics.pstdev(ar_per_seed(base, "energy_gap_rel"))),
              "egap_seed_sd_e2": float(statistics.pstdev(ar_per_seed(e2, "energy_gap_rel")))}
    fd_b, fd_e = fine_disp(base), fine_disp(e2)
    egap_change, fine_change = _rel_change(eg_e, eg_b), _rel_change(fd_e, fd_b)
    k1 = (egap_change > band) or (fine_change > band)
    phase = bench["phases"].get(f"bottleneck{m_tokens}_fine")
    if phase is not None and phase.get("timing") != "differential":
        raise ValueError("E2 adjudication refused: the bench's bottleneck phase is not "
                         "setup-free (re-bench with Stage >= 1.28: differential timing)")
    step_s = float(phase["ms_per_step"]) / 1000.0 if phase else None
    k2 = step_s is None or step_s >= kill_s
    go = (not k1) and (step_s is not None) and step_s < go_s
    return {"M": m_tokens, "band": band, "egap_median_base": eg_b, "egap_median_e2": eg_e,
            "egap_rel_change": egap_change, "fine_disp_base": fd_b, "fine_disp_e2": fd_e,
            "fine_disp_rel_change": fine_change, "fine_step_s": step_s, **spread,
            "aggregation": "seed median (both K1 quantities)",
            "base_config_sha256": got, "comparability": "identical beyond the architecture",
            "K1_accuracy": k1, "K2_speed": k2, "GO": go,
            "verdict": "GO" if go else ("KILLED" if (k1 or k2) else "NO-GO")}
