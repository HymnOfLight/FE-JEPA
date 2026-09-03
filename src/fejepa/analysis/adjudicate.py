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


def adjudicate_e1(base: dict, shaped: dict, s_base: list, s_shaped: list,
                  band: float = 0.10) -> dict:
    """K1 parity per seed (disp or energy gap worse than `band`); K2 no effect
    (S delta <= 0 at every seed); GO = S improves at every seed and the
    transfer ratio does not worsen beyond `band`."""
    import math

    bad = [v for v in list(s_base) + list(s_shaped) if not math.isfinite(float(v))]
    if bad:
        raise ValueError("E1 adjudication refused: a separation statistic is not finite "
                         f"({len(bad)} value(s)); the measurement is invalid (bins with "
                         "< 2 instances?) -- fix the measurement, do not adjudicate")
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
    rb, rs = transfer_ratio(base), transfer_ratio(shaped)
    ratio_ok = (rb is None or rs is None) or (_rel_change(rs, rb) <= band)
    go = (not k1) and all(d > 0.0 for d in deltas) and ratio_ok
    return {"band": band, "per_seed": per_seed, "S_base": s_base, "S_shaped": s_shaped,
            "S_delta": deltas, "transfer_ratio_base": rb, "transfer_ratio_shaped": rs,
            "K1_parity": k1, "K2_no_effect": k2, "GO": go,
            "verdict": "GO" if go else ("KILLED" if (k1 or k2) else "NO-GO")}


def adjudicate_e2(base: dict, e2: dict, bench: dict, m_tokens: int, band: float = 0.10,
                  kill_s: float = 2.0, go_s: float = 1.0) -> dict:
    """K1 accuracy (in-band energy gap at the seed median, or fine zero-shot
    displacement, worse than the baseline by more than `band`); K2 speed (fine
    step time not below `kill_s`); GO = parity and fine step time below `go_s`."""
    def ar_median(rep, metric):
        return float(statistics.median(ar_per_seed(rep, metric)))

    def fine_disp(rep):
        return float(rep["results"]["p3_transfer"]["metrics"]["ar"]["fine_disp_mean"])

    eg_b, eg_e = ar_median(base, "energy_gap_rel"), ar_median(e2, "energy_gap_rel")
    fd_b, fd_e = fine_disp(base), fine_disp(e2)
    egap_change, fine_change = _rel_change(eg_e, eg_b), _rel_change(fd_e, fd_b)
    k1 = (egap_change > band) or (fine_change > band)
    phase = bench["phases"].get(f"bottleneck{m_tokens}_fine")
    step_s = float(phase["ms_per_step"]) / 1000.0 if phase else None
    k2 = step_s is None or step_s >= kill_s
    go = (not k1) and (step_s is not None) and step_s < go_s
    return {"M": m_tokens, "band": band, "egap_median_base": eg_b, "egap_median_e2": eg_e,
            "egap_rel_change": egap_change, "fine_disp_base": fd_b, "fine_disp_e2": fd_e,
            "fine_disp_rel_change": fine_change, "fine_step_s": step_s,
            "K1_accuracy": k1, "K2_speed": k2, "GO": go,
            "verdict": "GO" if go else ("KILLED" if (k1 or k2) else "NO-GO")}
