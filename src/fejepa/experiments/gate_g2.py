"""Gate G2 -- the frozen GO/NO-GO of PREREG_PHASE2 (r8 Sec.2), with kills KP1-KP6
(Sec.4) under the Sec.5 definitions. Fails closed: any unevaluated condition
reads False.

  G2 = (a) AND ((b) OR (c))
  (a) repaired sanity: anchored >= sanity_x over zero on displacement AND beats
      every naive in naive_set on displacement, at EVERY budget (P1 cells).
  (b) label efficiency: disp_AR <= (1+parity_band) * disp_labels@max_budget AND
      egap advantage 1 - egap_AR/egap_labels(b) >= egap_adv_min at EVERY budget.
  (c) resolution transfer: seed-mean fine/in-band displacement ratio of the AR
      arm <= transfer_win AND AR beats every naive (strongest form, built from
      the full in-band prefix) evaluated at fine.

All comparisons use seed-mean quantities (Sec.5 aggregation rule); seed means
include diverged runs by construction upstream (the report writer flags them,
never excludes -- GAP-4). P3 report contract consumed here (produced by
experiments/p3_transfer.py):

  p3["metrics"] = {
    "ar": {"inband_disp_mean": float, "fine_disp_mean": float},
    "naive_at_fine": {"knn_field": float, "scale_aware_poly": float},  # disp means
    ...zero-shot rows for labels@1024 / mgn@1024 and the few-shot table are
    reported but not consumed by the gate.
  }
"""

from __future__ import annotations

PLAN_REF = "PREREG_PHASE2 r8: Sec.2 (G2), Sec.4 (KP1-KP6), Sec.5 (definitions)"

_DEF_GATE = {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"],
             "parity_band": 0.10, "egap_adv_min": 0.40, "transfer_win": 1.25,
             "decision_budget": 64}
_DEF_KILL = {"KP1_parity_pct": 0.10, "KP2_egap_adv_min": 0.40,
             "KP3_anchor_improv_min": 0.25, "KP4_transfer_ratio": 1.5,
             "KP6_rho_within_min": 0.3}


def _cell(e8: dict, regime: str, budget) -> dict | None:
    cells = e8["metrics"]["cells"].get(regime, {})
    return cells.get(budget) or cells.get(str(budget))


def _disp(cell: dict) -> float:
    return cell["disp_rel_l2"]["mean"]


def _egap(cell: dict) -> float:
    return cell["energy_gap_rel"]["mean"]


def _budgets(e8: dict) -> list:
    lab = e8["metrics"]["cells"].get("labels", {})
    return sorted(lab.keys(), key=lambda b: int(b))


def gate_g2(e8_result: dict | None, e1_result: dict | None,
            p3_result: dict | None, e6_result: dict | None,
            wp6_result: dict | None, gate_cfg: dict | None = None,
            kill_cfg: dict | None = None) -> dict:
    g = dict(_DEF_GATE); g.update(gate_cfg or {})
    k = dict(_DEF_KILL); k.update(kill_cfg or {})
    reasons: dict = {}
    kills = {name: False for name in
             ("KP1", "KP2", "KP3", "KP4", "KP5", "KP6")}

    # ---------------- (a) repaired sanity + (b) label efficiency (P1) --------
    a = b = False
    if e8_result is None:
        reasons["a_sanity"] = reasons["b_label_efficiency"] = \
            "P1/E8 not run -- condition unmeasured, gate fails closed"
    else:
        buds = _budgets(e8_result)
        # (a)
        ok, why = True, []
        for bud in buds:
            anc = _cell(e8_result, "labels_anchor", bud)
            zero = _cell(e8_result, "zero", bud)
            if anc is None or zero is None:
                ok = False; why.append(f"b={bud}: anchored/zero cell missing"); continue
            x = _disp(zero) / (_disp(anc) + 1e-30)
            if x < g["sanity_x"]:
                ok = False; why.append(f"b={bud}: {x:.2f}x over zero < {g['sanity_x']}x")
            for nv in g["naive_set"]:
                nc = _cell(e8_result, nv, bud)
                if nc is None:
                    ok = False; why.append(f"b={bud}: naive {nv} cell missing")
                elif _disp(anc) >= _disp(nc):
                    ok = False; why.append(f"b={bud}: anchored does not beat {nv}")
        a = ok
        reasons["a_sanity"] = "passed at every budget" if ok else "; ".join(why)

        # (b)
        max_b = buds[-1]
        ar_cells = e8_result["metrics"]["cells"].get("ar", {})
        ar = (ar_cells[max(ar_cells, key=lambda k: int(k))]
              if ar_cells else None)   # keyed by pool size, not budget
        lab_max = _cell(e8_result, "labels", max_b)
        if ar is None or lab_max is None:
            reasons["b_label_efficiency"] = "ar / labels cell missing"
        else:
            gap = _disp(ar) / (_disp(lab_max) + 1e-30) - 1.0
            parity = gap <= g["parity_band"]
            kills["KP1"] = bool(gap > k["KP1_parity_pct"])
            advs = {}
            for bud in buds:
                lab = _cell(e8_result, "labels", bud)
                advs[bud] = 1.0 - _egap(ar) / (_egap(lab) + 1e-30)
            adv_ok = all(v >= g["egap_adv_min"] for v in advs.values())
            kills["KP2"] = bool(any(v < k["KP2_egap_adv_min"] for v in advs.values()))
            b = bool(parity and adv_ok)
            reasons["b_label_efficiency"] = (
                f"parity gap {gap:+.3f} (band {g['parity_band']}); "
                f"egap advantage min {min(advs.values()):.3f} "
                f"(need >= {g['egap_adv_min']} everywhere)")

    # ---------------- KP3: anchor value (P2/E1') -----------------------------
    if e1_result is None:
        reasons["KP3"] = "P2/E1' not run -- anchor claim unevaluated"
    else:
        improvs = []
        for row in e1_result["metrics"]["per_budget"]:
            bud = row["budget"]
            arms = row["arms"]
            policy = "fixed" if int(bud) < 64 else "balanced"
            if "none" in arms and policy in arms:
                improvs.append(1.0 - arms[policy]["egap"]["mean"]
                               / (arms["none"]["egap"]["mean"] + 1e-30))
        kills["KP3"] = bool(improvs and all(v < k["KP3_anchor_improv_min"]
                                            for v in improvs))
        reasons["KP3"] = (f"anchored egap improvements {['%.3f' % v for v in improvs]} "
                          f"(kill if ALL < {k['KP3_anchor_improv_min']})")

    # ---------------- (c) + KP4: resolution transfer (P3) --------------------
    c = False
    zone = "unevaluated"
    if p3_result is None:
        reasons["c_transfer"] = "P3 not run -- condition unmeasured, reads false"
    else:
        m = p3_result["metrics"]
        ratio = m["ar"]["fine_disp_mean"] / (m["ar"]["inband_disp_mean"] + 1e-30)
        naive_beaten = all(m["ar"]["fine_disp_mean"] < v
                           for v in m["naive_at_fine"].values())
        kills["KP4"] = bool(ratio > k["KP4_transfer_ratio"] or not naive_beaten)
        if kills["KP4"]:
            zone = "retired"
        elif ratio <= g["transfer_win"]:
            zone = "win"
        else:
            zone = "weakened"
        c = bool(zone == "win" and naive_beaten)
        reasons["c_transfer"] = (
            f"fine/in-band ratio {ratio:.3f} (win <= {g['transfer_win']}, "
            f"kill > {k['KP4_transfer_ratio']}); naive beaten at fine: {naive_beaten}")

    # ---------------- KP5: falsification pass (P4/WP6) -----------------------
    if wp6_result is None:
        reasons["KP5"] = "P4/WP6 not run -- theory transcription unverified"
    else:
        tk = wp6_result.get("kills")
        if tk:
            holds = not any(k.get("triggered") for k in tk)
        else:
            holds = wp6_result.get("holds")
            if holds is None:
                m = wp6_result.get("metrics", {})
                holds = all(v.get("holds", True) for v in m.values()
                            if isinstance(v, dict))
        kills["KP5"] = not bool(holds)
        reasons["KP5"] = "all inequalities hold" if holds else \
            "VIOLATION -- investigate and publish before any other claim is used"

    # ---------------- KP6: alignment (P5/E6) ---------------------------------
    if e6_result is None:
        reasons["KP6"] = "P5/E6 not run -- alignment claim unevaluated"
    else:
        rho = e6_result["metrics"].get("rho_within_mean",
                                       e6_result["metrics"].get("rho_within"))
        kills["KP6"] = bool(rho is not None and rho < k["KP6_rho_within_min"])
        reasons["KP6"] = f"rho_within {rho} (kill if < {k['KP6_rho_within_min']})"

    passed = bool(a and (b or c))
    return {"plan_ref": PLAN_REF, "logic": "a AND (b OR c)",
            "decision_budget": g["decision_budget"],
            "thresholds": {"gate": g, "kills": k},
            "conditions": {"a": a, "b": b, "c": c},
            "transfer_zone": zone, "kills": kills,
            "any_kill": bool(any(kills.values())),
            "reasons": reasons, "passed": passed}
