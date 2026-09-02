#!/usr/bin/env python3
"""Independent audit of a Phase-2 deciding-run report (wp8-lejepa tooling).

The runner computes its own gate block. This script does NOT call the
runner's gate: it re-derives every G2 condition and kill from the report's
cells with the formulas written out explicitly, re-aggregates every cell mean
from its per-seed values, and checks provenance and accounting against the
expectations recorded at stamping. Agreement with the runner's block is then
a genuine cross-check; disagreement is an adjudication event.

Usage:
    python scripts/audit_phase2_report.py runs/phase2/report_phase2.json \
        --expect-config-sha e3bdd1e8... --expect-ledger 1280 \
        --expect-git-prefix prereg-phase2 \
        [--expect-dataset-sha ffaa... --expect-dataset-sha aae3...] \
        [--ar-sha-file ar_states.sha256]      # Song's `sha256sum` output
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _cell(cells, row, key):
    r = cells.get(row) or {}
    return r.get(str(key), r.get(key))


def _mean(cell, metric):
    return float(cell[metric]["mean"])


def audit(report: dict, args) -> dict:
    checks = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    prov = report["provenance"]
    # ---------------- provenance ----------------
    if args.expect_config_sha:
        chk("provenance.config_sha256 == stamped",
            prov.get("config_sha256") == args.expect_config_sha,
            prov.get("config_sha256"))
    chk("provenance.git describe carries the tag",
        str(prov.get("git", "")).startswith(args.expect_git_prefix), prov.get("git"))
    chk("prereg guard recorded", bool(report.get("prereg")), str(report.get("prereg"))[:80])
    chk("runtime tf32 policy", bool((report.get("runtime_policy") or {}).get("tf32")),
        str(report.get("runtime_policy")))
    if args.expect_dataset_sha:
        shas = [d.get("manifest_sha256") for d in prov.get("datasets", [])]
        chk("dataset manifest SHAs echoed",
            all(any(s.startswith(e) for s in shas) for e in args.expect_dataset_sha),
            str(shas))
    # ---------------- accounting ----------------
    led = report.get("solve_ledger") or {}
    if args.expect_ledger is not None:
        chk(f"solve ledger total == {args.expect_ledger}",
            int(led.get("total", -1)) == args.expect_ledger, json.dumps(led))
    d9 = (report["results"].get("e8") or {}).get("metrics", {}).get("d9_restart") or {}
    chk("d9 restart mode recorded", bool(report.get("d9_reuse_states")) == bool(d9.get("reuse_states")),
        json.dumps({k: d9.get(k) for k in ("reuse_states", "sup_units_from_cache",
                                           "units_resumed_from_epoch")}))
    if args.ar_sha_file:
        want = {}
        for line in Path(args.ar_sha_file).read_text().splitlines():
            m = re.match(r"([0-9a-f]{64})\s+.*ar_p\d+_s(\d)\.pt", line.strip())
            if m:
                want[f"s{m.group(2)}"] = m.group(1)
        got = {k: v.get("sha256") for k, v in (d9.get("ar_states") or {}).items()}
        chk("AR state SHA-256 chain (report == box sha256sum)",
            want and all(got.get(k) == v for k, v in want.items()), json.dumps({"box": want, "report": got}))
    # ---------------- per-seed re-aggregation ----------------
    e8 = report["results"]["e8"]["metrics"]
    cells = e8["cells"]
    bad = []
    for row, byb in cells.items():
        for b, cell in byb.items():
            for metric, v in cell.items():
                if isinstance(v, dict) and "per_seed" in v and "mean" in v:
                    ps = v["per_seed"]
                    if ps and abs(sum(ps) / len(ps) - v["mean"]) > 1e-9 * max(1.0, abs(v["mean"])):
                        bad.append(f"{row}@{b}:{metric}")
    chk("every cell mean equals the mean of its per-seed values", not bad, ", ".join(bad[:8]))
    # ---------------- gate re-derivation ----------------
    g = report["config"]["gate_g2"]
    k = report["config"]["kills"]
    runner = report["gate_g2"]
    buds = sorted((cells.get("labels") or {}).keys(), key=int)
    derived = {}
    # (a) sanity
    a_ok, a_why = True, []
    for b in buds:
        anc, zero = _cell(cells, "labels_anchor", b), _cell(cells, "zero", b)
        if not anc or not zero:
            a_ok = False; a_why.append(f"b={b}: missing"); continue
        x = _mean(zero, "disp_rel_l2") / (_mean(anc, "disp_rel_l2") + 1e-30)
        if x < g["sanity_x"]:
            a_ok = False; a_why.append(f"b={b}: {x:.2f}x over zero < {g['sanity_x']}")
        for nv in g["naive_set"]:
            nc = _cell(cells, nv, b)
            if not nc or _mean(anc, "disp_rel_l2") >= _mean(nc, "disp_rel_l2"):
                a_ok = False; a_why.append(f"b={b}: does not beat {nv}")
    derived["a"] = a_ok
    # (b) label efficiency + KP1/KP2
    ar_cells = cells.get("ar") or {}
    ar = ar_cells[max(ar_cells, key=int)] if ar_cells else None
    lab_max = _cell(cells, "labels", buds[-1]) if buds else None
    if ar and lab_max:
        gap = _mean(ar, "disp_rel_l2") / (_mean(lab_max, "disp_rel_l2") + 1e-30) - 1.0
        advs = {b: 1.0 - _mean(ar, "energy_gap_rel") / (_mean(_cell(cells, "labels", b), "energy_gap_rel") + 1e-30)
                for b in buds}
        derived["b"] = bool(gap <= g["parity_band"] and all(v >= g["egap_adv_min"] for v in advs.values()))
        derived["KP1"] = bool(gap > k["KP1_parity_pct"])
        derived["KP2"] = bool(any(v < k["KP2_egap_adv_min"] for v in advs.values()))
        derived["_gap"], derived["_advs"] = gap, advs
    else:
        derived["b"] = False; derived["KP1"] = derived["KP2"] = False
    # KP3 from P1 shared cells
    imps = []
    for b in buds:
        lab, anc = _cell(cells, "labels", b), _cell(cells, "labels_anchor", b)
        if lab and anc:
            imps.append(1.0 - _mean(anc, "energy_gap_rel") / (_mean(lab, "energy_gap_rel") + 1e-30))
    derived["KP3"] = bool(imps and all(v < k["KP3_anchor_improv_min"] for v in imps))
    # (c) transfer + KP4
    p3 = (report["results"].get("p3_transfer") or {}).get("metrics")
    if p3:
        ratio = p3["ar"]["fine_disp_mean"] / (p3["ar"]["inband_disp_mean"] + 1e-30)
        naive_beaten = all(p3["ar"]["fine_disp_mean"] < v for v in p3["naive_at_fine"].values())
        derived["KP4"] = bool(ratio > k["KP4_transfer_ratio"] or not naive_beaten)
        derived["c"] = bool((not derived["KP4"]) and ratio <= g["transfer_win"] and naive_beaten)
        derived["_ratio"] = ratio
    else:
        derived["c"] = False; derived["KP4"] = False
    # KP5 (wp6 identities) and KP6 (e6 rho)
    wp6r = report["results"].get("wp6")
    if wp6r:
        tk = wp6r.get("kills")
        if tk:
            holds = not any(kk.get("triggered") for kk in tk)
        else:
            holds = wp6r.get("holds")
            if holds is None:
                holds = all(v.get("holds", True) for v in (wp6r.get("metrics") or {}).values()
                            if isinstance(v, dict))
        derived["KP5"] = not bool(holds)
    else:
        derived["KP5"] = False
    e6 = (report["results"].get("e6") or {}).get("metrics")
    derived["KP6"] = bool(e6) and float(e6.get("rho_within_mean", 1.0)) < k["KP6_rho_within_min"]
    derived["any_kill"] = any(derived[x] for x in ("KP1", "KP2", "KP3", "KP4", "KP5", "KP6"))
    derived["passed"] = bool(derived["a"] and derived["b"] and derived["c"] and not derived["any_kill"])
    # ---------------- compare with the runner's block ----------------
    for key in ("a", "b", "c"):
        chk(f"condition ({key}) re-derived == runner", derived[key] == bool(runner["conditions"][key]),
            f"derived {derived[key]} vs runner {runner['conditions'][key]}")
    for key in ("KP1", "KP2", "KP3", "KP4", "KP5", "KP6"):
        chk(f"kill {key} re-derived == runner", derived[key] == bool(runner["kills"][key]),
            f"derived {derived[key]} vs runner {runner['kills'][key]}")
    chk("verdict re-derived == runner", derived["passed"] == bool(runner["passed"]),
        f"derived {derived['passed']} vs runner {runner['passed']}")
    return {"checks": checks, "derived": {kk: v for kk, v in derived.items() if not kk.startswith("_")},
            "derived_numbers": {kk: v for kk, v in derived.items() if kk.startswith("_")},
            "all_ok": all(c["ok"] for c in checks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--expect-config-sha", default=None)
    ap.add_argument("--expect-ledger", type=int, default=None)
    ap.add_argument("--expect-git-prefix", default="prereg-phase2")
    ap.add_argument("--expect-dataset-sha", action="append", default=[])
    ap.add_argument("--ar-sha-file", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = json.loads(Path(a.report).read_text())
    res = audit(report, a)
    for c in res["checks"]:
        print(("PASS " if c["ok"] else "FAIL "), c["check"], ("-- " + c["detail"]) if c["detail"] and not c["ok"] else "")
    print("derived:", json.dumps(res["derived"]))
    print("ALL OK" if res["all_ok"] else "DISCREPANCIES PRESENT")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
