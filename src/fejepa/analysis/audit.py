"""Independent audit of a Phase-2 deciding-run report.

Never calls the runner's gate: every G2 condition and kill is re-derived from
the report's cells with the formulas written out, every cell mean is
re-aggregated from its per-seed values, and provenance and accounting are
checked against the expectations recorded at stamping.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AuditExpectations:
    config_sha: str | None = None
    git_prefix: str = "prereg-phase2"
    dataset_shas: list = field(default_factory=list)
    ledger_total: int | None = None
    ar_sha_file: str | None = None


def _cell(cells, row, key):
    r = cells.get(row) or {}
    return r.get(str(key), r.get(key))


def _mean(cell, metric):
    return float(cell[metric]["mean"])


def check_provenance(report: dict, exp: AuditExpectations, chk) -> None:
    prov = report["provenance"]
    if exp.config_sha:
        chk("provenance.config_sha256 == stamped", prov.get("config_sha256") == exp.config_sha,
            prov.get("config_sha256"))
    chk("provenance.git describe carries the tag",
        str(prov.get("git", "")).startswith(exp.git_prefix), prov.get("git"))
    chk("prereg guard recorded", bool(report.get("prereg")), str(report.get("prereg"))[:80])
    chk("runtime tf32 policy", bool((report.get("runtime_policy") or {}).get("tf32")),
        str(report.get("runtime_policy")))
    if exp.dataset_shas:
        shas = [d.get("manifest_sha256") for d in prov.get("datasets", [])]
        chk("dataset manifest SHAs echoed",
            all(any(s.startswith(e) for s in shas) for e in exp.dataset_shas), str(shas))


def check_accounting(report: dict, exp: AuditExpectations, chk) -> None:
    led = report.get("solve_ledger") or {}
    if exp.ledger_total is not None:
        chk(f"solve ledger total == {exp.ledger_total}",
            int(led.get("total", -1)) == exp.ledger_total, json.dumps(led))
    d9 = (report["results"].get("e8") or {}).get("metrics", {}).get("d9_restart") or {}
    chk("d9 restart mode recorded",
        bool(report.get("d9_reuse_states")) == bool(d9.get("reuse_states")),
        json.dumps({k: d9.get(k) for k in ("reuse_states", "sup_units_from_cache",
                                           "units_resumed_from_epoch")}))
    if exp.ar_sha_file:
        want = {}
        for line in Path(exp.ar_sha_file).read_text().splitlines():
            m = re.match(r"([0-9a-f]{64})\s+.*ar_p\d+_s(\d)\.pt", line.strip())
            if m:
                want[f"s{m.group(2)}"] = m.group(1)
        got = {k: v.get("sha256") for k, v in (d9.get("ar_states") or {}).items()}
        chk("AR state SHA-256 chain (report == box sha256sum)",
            bool(want) and all(got.get(k) == v for k, v in want.items()),
            json.dumps({"box": want, "report": got}))


def check_reaggregation(cells: dict, chk) -> None:
    bad = []
    for row, byb in cells.items():
        for b, cell in byb.items():
            for metric, v in cell.items():
                if isinstance(v, dict) and "per_seed" in v and "mean" in v:
                    ps = v["per_seed"]
                    if ps and abs(sum(ps) / len(ps) - v["mean"]) > 1e-9 * max(1.0, abs(v["mean"])):
                        bad.append(f"{row}@{b}:{metric}")
    chk("every cell mean equals the mean of its per-seed values", not bad, ", ".join(bad[:8]))


def derive_gate(report: dict) -> dict:
    """Explicit re-derivation of G2 (a)(b)(c) and KP1-6 from cells."""
    g, k = report["config"]["gate_g2"], report["config"]["kills"]
    cells = report["results"]["e8"]["metrics"]["cells"]
    buds = sorted((cells.get("labels") or {}).keys(), key=int)
    d = {}
    # (a) sanity
    a_ok = True
    for b in buds:
        anc, zero = _cell(cells, "labels_anchor", b), _cell(cells, "zero", b)
        if not anc or not zero:
            a_ok = False; continue
        if _mean(zero, "disp_rel_l2") / (_mean(anc, "disp_rel_l2") + 1e-30) < g["sanity_x"]:
            a_ok = False
        for nv in g["naive_set"]:
            nc = _cell(cells, nv, b)
            if not nc or _mean(anc, "disp_rel_l2") >= _mean(nc, "disp_rel_l2"):
                a_ok = False
    d["a"] = a_ok
    # (b) label efficiency + KP1 / KP2
    ar_cells = cells.get("ar") or {}
    ar = ar_cells[max(ar_cells, key=int)] if ar_cells else None
    lab_max = _cell(cells, "labels", buds[-1]) if buds else None
    if ar and lab_max:
        gap = _mean(ar, "disp_rel_l2") / (_mean(lab_max, "disp_rel_l2") + 1e-30) - 1.0
        advs = {b: 1.0 - _mean(ar, "energy_gap_rel")
                / (_mean(_cell(cells, "labels", b), "energy_gap_rel") + 1e-30) for b in buds}
        d["b"] = bool(gap <= g["parity_band"] and all(v >= g["egap_adv_min"] for v in advs.values()))
        d["KP1"] = bool(gap > k["KP1_parity_pct"])
        d["KP2"] = bool(any(v < k["KP2_egap_adv_min"] for v in advs.values()))
        d["_gap"], d["_advs"] = gap, advs
    else:
        d["b"], d["KP1"], d["KP2"] = False, False, False
    # KP3 from P1 shared cells
    imps = [1.0 - _mean(_cell(cells, "labels_anchor", b), "energy_gap_rel")
            / (_mean(_cell(cells, "labels", b), "energy_gap_rel") + 1e-30)
            for b in buds if _cell(cells, "labels", b) and _cell(cells, "labels_anchor", b)]
    d["KP3"] = bool(imps and all(v < k["KP3_anchor_improv_min"] for v in imps))
    # (c) transfer + KP4
    p3 = (report["results"].get("p3_transfer") or {}).get("metrics")
    if p3:
        ratio = p3["ar"]["fine_disp_mean"] / (p3["ar"]["inband_disp_mean"] + 1e-30)
        naive_beaten = all(p3["ar"]["fine_disp_mean"] < v for v in p3["naive_at_fine"].values())
        d["KP4"] = bool(ratio > k["KP4_transfer_ratio"] or not naive_beaten)
        d["c"] = bool((not d["KP4"]) and ratio <= g["transfer_win"] and naive_beaten)
        d["_ratio"] = ratio
    else:
        d["c"], d["KP4"] = False, False
    # KP5 (wp6) and KP6 (e6)
    wp6r = report["results"].get("wp6")
    if wp6r:
        tk = wp6r.get("kills")
        holds = (not any(x.get("triggered") for x in tk)) if tk else wp6r.get("holds")
        if holds is None:
            holds = all(v.get("holds", True) for v in (wp6r.get("metrics") or {}).values()
                        if isinstance(v, dict))
        d["KP5"] = not bool(holds)
    else:
        d["KP5"] = False
    e6 = (report["results"].get("e6") or {}).get("metrics")
    d["KP6"] = bool(e6) and float(e6.get("rho_within_mean", 1.0)) < k["KP6_rho_within_min"]
    d["any_kill"] = any(d[x] for x in ("KP1", "KP2", "KP3", "KP4", "KP5", "KP6"))
    d["passed"] = bool(d["a"] and d["b"] and d["c"] and not d["any_kill"])
    return d


def audit(report: dict, exp: AuditExpectations) -> dict:
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    check_provenance(report, exp, chk)
    check_accounting(report, exp, chk)
    check_reaggregation(report["results"]["e8"]["metrics"]["cells"], chk)
    derived = derive_gate(report)
    runner = report["gate_g2"]
    for key in ("a", "b", "c"):
        chk(f"condition ({key}) re-derived == runner", derived[key] == bool(runner["conditions"][key]),
            f"derived {derived[key]} vs runner {runner['conditions'][key]}")
    for key in ("KP1", "KP2", "KP3", "KP4", "KP5", "KP6"):
        chk(f"kill {key} re-derived == runner", derived[key] == bool(runner["kills"][key]),
            f"derived {derived[key]} vs runner {runner['kills'][key]}")
    chk("verdict re-derived == runner", derived["passed"] == bool(runner["passed"]),
        f"derived {derived['passed']} vs runner {runner['passed']}")
    return {"checks": checks,
            "derived": {kk: v for kk, v in derived.items() if not kk.startswith("_")},
            "derived_numbers": {kk: v for kk, v in derived.items() if kk.startswith("_")},
            "all_ok": all(c["ok"] for c in checks)}
