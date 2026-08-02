"""RESULTS.md renderer + Figure-1 (numpy-only; matplotlib present in dev extra)."""

import json

from fejepa.results import render_results, write_figures, write_results


def _cell(disp, egap):
    z = {"mean": 0.0, "std": 0.0, "per_seed": [0.0]}
    return {"disp_rel_l2": {"mean": disp, "std": 0.01, "per_seed": [disp]},
            "energy_gap_rel": {"mean": egap, "std": 0.02, "per_seed": [egap]},
            "vm_rel_l2": {"mean": 0.3, "std": 0.0, "per_seed": [0.3]},
            "peak_vm_rel_err": z, "crit_recall": z, "per_seed_eval": [{}]}


def _payload():
    e8 = {"protocol": {"budgets": [16, 64]},
          "metrics": {"cells": {
              "labels": {16: _cell(0.34, 27.0), 64: _cell(0.24, 3.1)},
              "labels_anchor": {16: _cell(0.33, 0.5), 64: _cell(0.24, 0.4)},
              "ar_ft": {16: _cell(0.32, 0.6), 64: _cell(0.23, 0.5)},
              "zero": {16: _cell(1.0, 1.0), 64: _cell(1.0, 1.0)},
              "ar": {1024: _cell(0.22, 0.13)}},
              "label_efficiency_auc_disp": {"labels": 0.29}},
          "kills": [{"condition": "K2: ...", "triggered": False, "note": "ok"}]}
    e1 = {"metrics": {"per_budget": [{
        "budget": 64,
        "arms": {"none": {"disp": {"mean": 0.244, "std": 0.002, "per_seed": []}}},
        "improvements": {
            "balanced": {"disp": {"value": 0.011, "t": 1.2},
                         "egap": {"value": 0.87, "t": 9.0}},
            "fixed": {"disp": {"value": 0.010, "t": 1.1},
                      "egap": {"value": 0.85, "t": 8.0}}},
        "grid_best": {"arm": "grid_0.3", "selection_bias": True,
                      "disp": {"value": 0.03, "t": 4.0},
                      "egap": {"value": 0.8, "t": 7.0}}}],
        "retired_criterion_report": {"fixed_disp_improvement_at_decision":
                                     {"value": 0.0108, "t": 1.2}}},
        "kills": [{"condition": "K1", "triggered": False, "note": ""}]}
    gate = {"passed": True, "decision_budget": 64,
            "conditions": {"a": True, "b": True, "c": True},
            "reasons": {"a_sanity": "ok", "b_physics": "egap -87%", "c_transfer": "ok"},
            "retired_displacement_criterion":
                e1["metrics"]["retired_criterion_report"]}
    return {"config": {"workers": 3},
            "results": {"e1": e1, "e8": e8},
            "gate_g1_prime": gate,
            "runtime_policy": {"device": "cuda", "tf32": True},
            "data_economy": {"labelled_instances": 1280, "labelled_val": 256,
                             "labelled_pool_prefix": 1024,
                             "solves_per_labelled_instance": 4,
                             "reference_solves_total": 5120,
                             "unlabeled_pool_depth_used": 1024,
                             "unlabeled_over_labelled_pool": 1.0,
                             "ledger": {}},
            "solve_ledger": {"per_stage": {"labelling-val": 1024}, "total": 5120,
                             "wall_clock_s": 12.3},
            "provenance": {"config_sha256": "a" * 64, "timestamp_utc": "t",
                           "git": "g", "datasets": [
                               {"dir": "d", "n_instances": 30000,
                                "backend": "gmsh", "manifest_sha256": "b" * 64}]}}


def test_render_go_and_string_key_robustness():
    payload = _payload()
    md = render_results(payload)
    md_json = render_results(json.loads(json.dumps(payload)))  # str keys
    for text in (md, md_json):
        assert "**Verdict: GO**" in text
        assert "| labels | 0.3400±0.0100 | 0.2400±0.0100 |" in text
        assert "AR (unlabeled pool 1024, 0 labels)" in text
        assert "grid_0.3" in text and "retired criterion" in text
        assert "labelled instances: 1280" in text
        assert "## E7" in text and "not run" in text     # guarded sections


def test_write_results_and_figure(tmp_path):
    payload = _payload()
    out = write_results(payload, tmp_path / "RESULTS.md")
    assert out.exists() and "Gate G1'" in out.read_text()
    figs = write_figures(payload, tmp_path)
    assert figs and figs[0].name == "figure1_energy_gap.png"
    assert figs[0].stat().st_size > 5000


def test_figure_skipped_without_e8(tmp_path):
    assert write_figures({"results": {}}, tmp_path) == []


def _wrap(results):
    return {"results": results, "provenance": {}, "config": {},
            "runtime_policy": {}}


def test_render_producer_true_key_shapes():
    """Fragments below mirror each producer's ACTUAL metrics keys (verified
    against the sources); a renderer/producer drift breaks this test, not a
    finished run."""
    frags = {
        "e2": {"metrics": {
            "jepa_vs_ar_improvements": {64: {"disp": 0.02, "egap": 0.06}},
            "jepa_improvement_at_decision_budget": {"disp": 0.02, "egap": 0.06}},
            "kills": []},
        "e3": {"metrics": {"conditions": {}, "ratios": {},
                           "best_std_ratio": 2.1}, "kills": []},
        "e4": {"metrics": {"per_coarsen": [{
            "coarsen": 2.5, "n_train": 32,
            "inv_off": {"err_fine": .2, "err_coarse": .3, "gap": .1,
                        "gap_over_abs_err": .4},
            "inv_on": {"err_fine": .2, "err_coarse": .25, "gap": .05,
                       "gap_over_abs_err": .2},
            "gap_reduction": 0.5}]}, "kills": []},
        "e6": {"metrics": {"rho_within_mean": 0.81,
                           "rho_within_per_geometry": [0.8, 0.82],
                           "rho_cross_descriptor": 0.4}, "kills": []},
        "e7": {"metrics": {
            "iterations_to_tol_mean": {"zero": 200.0, "naive": 150.0,
                                       "learned": 90.0},
            "savings_learned": 0.4, "savings_naive": 0.25,
            "polish_at_k": {0: {"energy_gap_rel": .3, "disp_rel_l2": .3},
                            5: {"energy_gap_rel": .1, "disp_rel_l2": .28}}},
            "kills": []},
    }
    md = render_results(_wrap(frags))
    assert "+2.00%" in md and "best std-rank ratio (reg on/off) = 2.100" in md
    assert "| 2.5 |" in md and "+50.0%" in md
    assert "within-geometry mean rho = 0.810" in md
    assert "learned 90.0" in md and "| 5 | 0.1000 |" in md


def test_render_real_e5_output(instances):
    from fejepa.experiments.e5_sanity import run_e5

    # the exact flat shape the runner injects from E1' (see runner.py)
    anchored = {3: {"mean": 0.053, "per_instance": [0.05, 0.06, 0.05]}}
    res = run_e5(instances[:6], instances[6:],
                 {"budgets": [3], "fit_budget": 6}, anchored=anchored)
    md = render_results(_wrap({"e5": res}))
    assert "baseline disp rel-L2:" in md and "| 3 |" in md
    assert ("PASS" in md) or ("FAIL" in md)
