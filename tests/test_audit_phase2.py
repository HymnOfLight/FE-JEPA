"""The independent Phase-2 audit re-derives the gate from cells with explicit
formulas: checked on a hand-built report where every answer is known."""

from fejepa.analysis.audit import AuditExpectations, audit



def _cell(disp, egap, seeds=(1, 1, 1)):
    n = len(seeds)
    return {"disp_rel_l2": {"mean": disp, "per_seed": [disp] * n},
            "energy_gap_rel": {"mean": egap, "per_seed": [egap] * n}}


def _report(ar_disp=0.20, lab_disp=0.20, ar_egap=0.30, lab_egap=0.60, anc_egap=0.40,
            ratio=1.1, wp6_holds=True, rho=0.8):
    buds = ["16", "1024"]
    cells = {"ar": {"1024": _cell(ar_disp, ar_egap)},
             "labels": {b: _cell(lab_disp, lab_egap) for b in buds},
             "labels_anchor": {b: _cell(0.15, anc_egap) for b in buds},
             "zero": {b: _cell(1.0, 1.0) for b in buds},
             "knn_field": {b: _cell(0.5, 0.9) for b in buds},
             "scale_aware_poly": {b: _cell(0.6, 0.9) for b in buds}}
    p3 = {"ar": {"fine_disp_mean": ar_disp * ratio, "inband_disp_mean": ar_disp},
          "naive_at_fine": {"knn_field": 0.9, "scale_aware_poly": 0.9}}
    return {"provenance": {"config_sha256": "abc", "git": "prereg-phase2-3-gxxx", "datasets": []},
            "prereg": {"config_sha256": "abc"}, "runtime_policy": {"tf32": True},
            "d9_reuse_states": True, "solve_ledger": {"total": 1280, "per_stage": {}},
            "config": {"gate_g2": {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"],
                                   "parity_band": 0.10, "egap_adv_min": 0.40, "transfer_win": 1.25,
                                   "decision_budget": 1024},
                       "kills": {"KP1_parity_pct": 0.10, "KP2_egap_adv_min": 0.40,
                                 "KP3_anchor_improv_min": 0.25, "KP4_transfer_ratio": 1.5,
                                 "KP6_rho_within_min": 0.3}},
            "results": {"e8": {"metrics": {"cells": cells, "d9_restart": {"reuse_states": True}}},
                        "p3_transfer": {"metrics": p3},
                        "wp6": {"kills": [{"triggered": not wp6_holds}]},
                        "e6": {"metrics": {"rho_within_mean": rho}}},
            "gate_g2": {"conditions": {"a": True, "b": True, "c": True},
                        "kills": {k: False for k in ("KP1", "KP2", "KP3", "KP4", "KP5", "KP6")},
                        "passed": True}}


ARGS = AuditExpectations(config_sha="abc", git_prefix="prereg-phase2", ledger_total=1280)


def test_go_report_is_reproduced_and_all_checks_pass():
    res = audit(_report(), ARGS)
    assert res["all_ok"], [c for c in res["checks"] if not c["ok"]]
    assert res["derived"]["passed"] is True


def test_each_kill_is_re_derived_from_cells():
    d = audit(_report(ar_disp=0.25), ARGS)["derived"]          # +25% gap
    assert d["KP1"] is True and d["b"] is False
    d = audit(_report(ar_egap=0.50), ARGS)["derived"]          # advantage 1-0.5/0.6 < 0.4
    assert d["KP2"] is True
    d = audit(_report(anc_egap=0.55), ARGS)["derived"]         # anchor improv 1-0.55/0.6 < 0.25
    assert d["KP3"] is True
    d = audit(_report(ratio=1.6), ARGS)["derived"]             # transfer ratio > 1.5
    assert d["KP4"] is True and d["c"] is False
    d = audit(_report(wp6_holds=False), ARGS)["derived"]
    assert d["KP5"] is True
    d = audit(_report(rho=0.1), ARGS)["derived"]
    assert d["KP6"] is True


def test_disagreement_with_runner_block_is_flagged():
    rep = _report(ar_disp=0.25)                                    # truly KP1, runner claims clean
    res = audit(rep, ARGS)
    assert not res["all_ok"]
    assert any("KP1" in c["check"] and not c["ok"] for c in res["checks"])


def test_audit_adjudicates_a_phase2b_shaped_report(tmp_path):
    """PREREG_PHASE2B: with the sanity floor raised, the audit re-derives (a)
    at budgets >= the floor, demands the reference gate, and re-derives it too."""
    import json

    from fejepa.analysis.audit import AuditExpectations, audit
    from fejepa.experiments.runner import run_config
    from fejepa.fe.synthetic import generate_synthetic_dataset

    MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 16, "mgn_depth": 2,
             "features": {"load_summary": True, "geometry": True}}
    d = generate_synthetic_dataset(tmp_path / "c", n=12, seed=3)
    df = generate_synthetic_dataset(tmp_path / "f", n=8, seed=4)
    cfg = {"data": {"dir": str(d), "n": 12, "seed": 3, "backend": "synthetic", "labelled_policy": "economy"},
           "data_transfer": {"dir": str(df), "n": 8, "seed": 4, "backend": "synthetic", "labelled_policy": "economy",
                             "split": {"n_eval": 3, "n_fewshot_prefix": 3}},
           "split": {"n_val": 3, "seed": 1}, "model": MODEL, "sup": {"epochs": 1, "lr": 1e-3},
           "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {"e8": {"enabled": True, "budgets": [2, 4], "pool_sizes": [4], "seeds": 1, "ar_epochs": 1,
                                  "sup_epochs": 1, "include_mgn": True, "include_ar_ft": False, "mgn_budgets": [4]},
                           "p3_transfer": {"enabled": True, "fewshot_budgets": [2], "fewshot_epochs": 1, "naive_budget": 4},
                           "wp6": {"enabled": True, "n_check": 2, "seed": 0}},
           "gate_g2": {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"], "parity_band": 0.10,
                       "egap_adv_min": 0.40, "transfer_win": 1.25, "decision_budget": 4, "sanity_min_budget": 4},
           "kills": {"KP1_parity_pct": 0.10, "KP2_egap_adv_min": 0.40, "KP3_anchor_improv_min": 0.25,
                     "KP4_transfer_ratio": 1.5, "KP6_rho_within_min": 0.3},
           "device": "cpu", "workers": 1, "tf32": False, "runtime": {"compile": False, "amp": False, "precision": "fp32"},
           "seeds": [0], "out": str(tmp_path / "run" / "report_phase2b.json"), "prereg_guard": False}
    p = tmp_path / "cfg.json"; p.write_text(json.dumps(cfg))
    run_config(str(p))
    report = json.loads((tmp_path / "run" / "report_phase2b.json").read_text())
    assert report["gate_g2_reference_all_budgets"] is not None
    out = audit(report, AuditExpectations(git_prefix=""))
    names = [c["check"] for c in out["checks"]]
    assert "reference gate (all budgets) present when the sanity floor is raised" in names
    assert all(c["ok"] for c in out["checks"] if "re-derived" in c["check"]), [c for c in out["checks"] if not c["ok"]]


def test_audit_fails_closed_when_the_floor_exceeds_every_budget():
    from fejepa.analysis.audit import derive_gate

    def cell(v): return {"disp_rel_l2": {"mean": v, "per_seed": [v]}, "energy_gap_rel": {"mean": v, "per_seed": [v]}}
    cells = {"labels_anchor": {"16": cell(0.2)}, "labels": {"16": cell(0.4)}, "zero": {"16": cell(1.0)},
             "knn_field": {"16": cell(0.65)}, "scale_aware_poly": {"16": cell(1.5)}, "ar": {"16": cell(0.9)}}
    report = {"config": {"gate_g2": {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"], "parity_band": 0.1,
                                     "egap_adv_min": 0.4, "transfer_win": 1.25, "decision_budget": 16, "sanity_min_budget": 1024},
                         "kills": {"KP1_parity_pct": 0.1, "KP2_egap_adv_min": 0.4, "KP3_anchor_improv_min": 0.25,
                                   "KP4_transfer_ratio": 1.5, "KP6_rho_within_min": 0.3}},
              "results": {"e8": {"metrics": {"cells": cells}}}}
    assert derive_gate(report)["a"] is False
