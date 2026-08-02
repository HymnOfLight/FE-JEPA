"""Gate G1' -- the frozen GO/NO-GO for Phase 2 (plan Sec.5 item 2).

  (a) repaired sanity: E5' passed at every budget;
  (b) physics value at the decision budget: labels+anchor vs labels-only achieves
      energy-gap reduction >= 50% AND vM rel-L2 reduction >= 25% (from E8's cells);
  (c) measured transfer: AR->FT at the decision budget beats labels-only by >= 10% on
      the energy gap OR >= 5% on displacement (from E8's cells -- measured, not assumed).

The retired v1.0 displacement criterion is echoed for the record (plan Sec.5 item 1).
E2's verdict is intentionally NOT here: it selects the paper branch, not scale-up.
"""

from __future__ import annotations

PLAN_REF = "plan v2.0 Sec.5 item 2 (G1'), item 1 (retirement)"


def _cell(e8: dict, regime: str, budget: int) -> dict | None:
    cells = e8["metrics"]["cells"].get(regime, {})
    return cells.get(budget) or cells.get(str(budget))


def g1_prime(e5_result: dict | None, e8_result: dict | None,
             e1_result: dict | None = None, decision_budget: int = 64,
             thresholds: dict | None = None) -> dict:
    th = {"egap_reduction": 0.50, "vm_reduction": 0.25,
          "transfer_egap": 0.10, "transfer_disp": 0.05}
    th.update(thresholds or {})
    reasons = {}

    if e5_result is None:
        a = False
        reasons["a_sanity"] = "E5' not run -- condition unmeasured, gate fails closed"
    else:
        a = bool(e5_result["metrics"]["passed_all"])
        reasons["a_sanity"] = f"E5' passed_all={a}"

    b = c = False
    if e8_result is None:
        reasons["b_physics"] = reasons["c_transfer"] = \
            "E8 not run -- condition unmeasured, gate fails closed"
    else:
        lab = _cell(e8_result, "labels", decision_budget)
        anc = _cell(e8_result, "labels_anchor", decision_budget)
        ft = _cell(e8_result, "ar_ft", decision_budget)
        if lab and anc:
            egap_red = (lab["energy_gap_rel"]["mean"] - anc["energy_gap_rel"]["mean"]) \
                / (lab["energy_gap_rel"]["mean"] + 1e-30)
            vm_red = (lab["vm_rel_l2"]["mean"] - anc["vm_rel_l2"]["mean"]) \
                / (lab["vm_rel_l2"]["mean"] + 1e-30)
            b = bool(egap_red >= th["egap_reduction"] and vm_red >= th["vm_reduction"])
            reasons["b_physics"] = (f"egap reduction {egap_red:.3f} "
                                    f"(need >= {th['egap_reduction']}), "
                                    f"vM reduction {vm_red:.3f} "
                                    f"(need >= {th['vm_reduction']})")
        else:
            reasons["b_physics"] = "labels / labels_anchor cell missing at decision budget"
        if lab and ft:
            eg = (lab["energy_gap_rel"]["mean"] - ft["energy_gap_rel"]["mean"]) \
                / (lab["energy_gap_rel"]["mean"] + 1e-30)
            dp = (lab["disp_rel_l2"]["mean"] - ft["disp_rel_l2"]["mean"]) \
                / (lab["disp_rel_l2"]["mean"] + 1e-30)
            c = bool(eg >= th["transfer_egap"] or dp >= th["transfer_disp"])
            reasons["c_transfer"] = (f"AR->FT vs labels @ {decision_budget}: "
                                     f"egap +{eg:.3f}, disp +{dp:.3f}")
        else:
            reasons["c_transfer"] = "ar_ft cell missing at decision budget"

    retired = None
    if e1_result is not None:
        retired = e1_result["metrics"]["retired_criterion_report"]

    return {"plan_ref": PLAN_REF, "decision_budget": decision_budget,
            "thresholds": th, "conditions": {"a": a, "b": b, "c": c},
            "reasons": reasons, "passed": bool(a and b and c),
            "retired_displacement_criterion": retired}
