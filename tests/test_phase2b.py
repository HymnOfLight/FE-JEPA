"""PREREG_PHASE2B mechanics: the sanity floor, the reported reference gate,
the AR-only pilot shape, and the cache surgery that re-runs only the
AR-dependent units while reusing the supervised and scratch units."""

import json
import shutil
import pytest

torch = pytest.importorskip("torch")

from fejepa.experiments.gate_g2 import gate_g2
from fejepa.experiments.runner import run_config
from fejepa.fe.synthetic import generate_synthetic_dataset

MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 16, "mgn_depth": 2,
         "features": {"load_summary": True, "geometry": True}}


def _e8(anc16, anc64):
    def cell(v): return {"disp_rel_l2": {"mean": v, "per_seed": [v]}, "energy_gap_rel": {"mean": v, "per_seed": [v]}}
    cells = {"labels_anchor": {"16": cell(anc16), "64": cell(anc64)},
             "labels": {"16": cell(0.4), "64": cell(0.3)}, "zero": {"16": cell(1.0), "64": cell(1.0)},
             "knn_field": {"16": cell(0.65), "64": cell(0.48)}, "scale_aware_poly": {"16": cell(1.5), "64": cell(1.5)},
             "ar": {"64": cell(0.9)}}
    return {"metrics": {"cells": cells, "budgets": [16, 64]}}


def test_sanity_floor_exempts_budgets_below_it():
    e8 = _e8(anc16=0.369, anc64=0.256)                 # 2.71x at 16 (fails 3.0x), 3.9x at 64
    g_all = gate_g2(e8, None, None, None, None, gate_cfg={"sanity_x": 3.0})
    g_64 = gate_g2(e8, None, None, None, None, gate_cfg={"sanity_x": 3.0, "sanity_min_budget": 64})
    assert g_all["conditions"]["a"] is False and "b=16" in g_all["reasons"]["a_sanity"]
    assert g_64["conditions"]["a"] is True and "not assessed" in g_64["reasons"]["a_sanity"]


def _cfg(tmp_path, d, df, out, extra_gate=None, extra_e8=None):
    cfg = {"data": {"dir": str(d), "n": 12, "seed": 3, "backend": "synthetic", "labelled_policy": "economy"},
           "data_transfer": {"dir": str(df), "n": 8, "seed": 4, "backend": "synthetic", "labelled_policy": "economy",
                             "split": {"n_eval": 3, "n_fewshot_prefix": 3}},
           "split": {"n_val": 3, "seed": 1}, "model": MODEL, "sup": {"epochs": 1, "lr": 1e-3},
           "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {"e8": {"enabled": True, "budgets": [2, 4], "pool_sizes": [4], "seeds": 1, "ar_epochs": 1,
                                  "sup_epochs": 1, "include_mgn": True, "include_ar_ft": False, "mgn_budgets": [4],
                                  **(extra_e8 or {})},
                           "p3_transfer": {"enabled": True, "fewshot_budgets": [2], "fewshot_epochs": 1, "naive_budget": 4},
                           "wp6": {"enabled": True, "n_check": 2, "seed": 0}},
           "gate_g2": {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"], "parity_band": 0.10,
                       "egap_adv_min": 0.40, "transfer_win": 1.25, "decision_budget": 4, **(extra_gate or {})},
           "kills": {"KP1_parity_pct": 0.10, "KP2_egap_adv_min": 0.40, "KP3_anchor_improv_min": 0.25,
                     "KP4_transfer_ratio": 1.5, "KP6_rho_within_min": 0.3},
           "device": "cpu", "workers": 1, "tf32": False, "runtime": {"compile": False, "amp": False, "precision": "fp32"},
           "seeds": [0], "out": str(out), "prereg_guard": False}
    return cfg


def test_reference_gate_reported_and_cache_surgery_reuses_only_valid_units(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "c", n=12, seed=3)
    df = generate_synthetic_dataset(tmp_path / "f", n=8, seed=4)
    run_dir = tmp_path / "run"
    # Phase-2-shaped run (deciding), no sanity floor
    c2 = _cfg(tmp_path, d, df, run_dir / "report_phase2.json")
    p2 = tmp_path / "phase2.json"; p2.write_text(json.dumps(c2))
    r2 = run_config(str(p2))
    assert r2["gate_g2_reference_all_budgets"] is None
    states = run_dir / "e8_states"
    ar = sorted(states.glob("ar_p*_s*.pt")); assert ar, "AR states expected"
    # cache surgery: move the AR states and the AR-dependent fine-tune units aside
    aside = run_dir / "e8_states_phase2_invalid_ar"; aside.mkdir()
    for f in ar: shutil.move(str(f), aside / f.name)
    ft = sorted((states / "unit_cache_p3").glob("P3_finetune_*.pkl")); assert ft
    for f in ft: shutil.move(str(f), aside / f.name)
    sup_before = sorted(p.name for p in (states / "unit_cache").glob("*.pkl"))
    scratch_before = sorted(p.name for p in (states / "unit_cache_p3").glob("P3_scratch_*.pkl"))
    # Phase-2b-shaped run: same e8_states directory, a different report, the sanity floor at the decision budget
    c2b = _cfg(tmp_path, d, df, run_dir / "report_phase2b.json", extra_gate={"sanity_min_budget": 4})
    p2b = tmp_path / "phase2b.json"; p2b.write_text(json.dumps(c2b))
    r2b = run_config(str(p2b), reuse_states=True, label_workers_override=1)
    d9 = r2b["results"]["e8"]["metrics"]["d9_restart"]
    assert all(not v["reused"] for v in d9["ar_states"].values())          # AR retrained
    assert sorted(d9["sup_units_from_cache"]) and len(d9["sup_units_from_cache"]) == len(sup_before)
    assert r2b["gate_g2_reference_all_budgets"] is not None                   # both gates reported
    assert r2b["gate_g2"]["thresholds"]["gate"]["sanity_min_budget"] == 4
    assert sorted(p.name for p in (states / "unit_cache_p3").glob("P3_scratch_*.pkl")) == scratch_before
    assert (run_dir / "report_phase2.json").exists() and (run_dir / "report_phase2b.json").exists()


def test_ar_only_pilot_shape(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "c", n=12, seed=3)
    df = generate_synthetic_dataset(tmp_path / "f", n=8, seed=4)
    cfg = _cfg(tmp_path, d, df, tmp_path / "pilot" / "report.json", extra_e8={"ar_only": True, "seeds": 1})
    for k in ("p3_transfer", "wp6"):
        cfg["experiments"][k]["enabled"] = False
    p = tmp_path / "pilot.json"; p.write_text(json.dumps(cfg))
    r = run_config(str(p))
    cells = r["results"]["e8"]["metrics"]["cells"]
    assert "ar" in cells and "labels" not in cells
    assert r["results"]["e8"]["protocol"]["ar_only"] is True
    key = next(iter(cells["ar"]))                                # keyed by pool size
    assert isinstance(cells["ar"][key]["disp_rel_l2"]["mean"], float)
