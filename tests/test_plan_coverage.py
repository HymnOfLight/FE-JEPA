"""Plan-coverage invariant (numpy-only, runs everywhere).

Plan v2.0 Sec.6 schedules exactly E1'..E8 plus Gate G1'. This test makes the
"every planned experiment has runnable code" guarantee machine-checked:

  1. each battery ID maps to an importable experiments module exposing ``run_*``
     whose PLAN_REF names that ID (modules stay torch-free at import time);
  2. the runner wires every ``run_*`` and the gate;
  3. both shipped configs carry every experiment block, and the smoke config
     enables all eight (the end-to-end runnable path);
  4. the deciding config keeps E2 present-but-disabled, per the pre-registration
     (plan Sec.5 item 6: E2's one-shot verdict belongs to the WP2 run).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BATTERY = {
    "E1'": ("e1_anchor", "run_e1"),
    "E2": ("e2_jepa", "run_e2"),
    "E3'": ("e3_collapse", "run_e3"),
    "E4'": ("e4_meshviews", "run_e4"),
    "E5'": ("e5_sanity", "run_e5"),
    "E6": ("e6_alignment", "run_e6"),
    "E7": ("e7_polish", "run_e7"),
    "E8": ("e8_regimes", "run_e8"),
}


def _configs():
    smoke = json.loads((ROOT / "configs/smoke.json").read_text())
    rec8 = json.loads((ROOT / "configs/phase1_rec8_v2.json").read_text())
    return smoke, rec8


def test_every_battery_experiment_is_importable_and_tagged():
    for plan_id, (mod_name, fn_name) in BATTERY.items():
        mod = importlib.import_module(f"fejepa.experiments.{mod_name}")
        assert hasattr(mod, fn_name), f"{mod_name} lacks {fn_name}"
        assert plan_id in mod.PLAN_REF, f"{mod_name}.PLAN_REF does not cite {plan_id}"


def test_runner_wires_every_experiment_and_the_gate():
    src = (ROOT / "src/fejepa/experiments/runner.py").read_text()
    for _pid, (mod_name, fn_name) in BATTERY.items():
        assert f"from .{mod_name} import {fn_name}" in src
        key = mod_name.split("_")[0]
        assert f'results["{key}"]' in src
    assert "g1_prime" in src


def test_configs_cover_every_experiment():
    smoke, rec8 = _configs()
    for cfg in (smoke, rec8):
        for _pid, (mod_name, _fn) in BATTERY.items():
            assert mod_name.split("_")[0] in cfg["experiments"]
        assert "gate" in cfg
    # smoke exercises the full runnable path
    assert all(v.get("enabled") for v in smoke["experiments"].values())
    # deciding run: E2 (prereg Sec.5 item 6) and WP2 (weeks 2-5) present, disabled
    allowed_disabled = {"e2", "wp2"}
    for k in allowed_disabled:
        assert rec8["experiments"][k]["enabled"] is False
    assert all(v.get("enabled") for k, v in rec8["experiments"].items()
               if k not in allowed_disabled)


def test_work_packages_with_code_demands_are_implemented():
    """Plan WP audit made executable. WP6 (theory note) needs no code; WP7 is
    conditional on G1' and intentionally absent (PLAN_MAP)."""
    import fejepa.polish as polish                                    # WP3
    from fejepa.experiments.runner import data_economy_summary        # WP5
    from fejepa.report import stamp_prereg, verify_prereg             # WP1

    assert callable(polish.polish_battery) and callable(polish.polished)
    assert callable(stamp_prereg) and callable(verify_prereg)
    assert callable(data_economy_summary)

    e7 = (ROOT / "src/fejepa/experiments/e7_polish.py").read_text()
    assert "polish_battery" in e7                                     # WP3 wired

    _smoke, rec8 = _configs()
    assert rec8.get("prereg_guard") is True                           # WP1 armed
    assert "CONFIG_SHA256" in (ROOT / "PREREG.md").read_text()

    gi = (ROOT / ".gitignore").read_text()                            # WP8 hygiene
    assert "runs/" in gi and "*.pt" in gi

    from fejepa.experiments.wp2_masking import run_wp2                # WP2 sweep
    from fejepa.fe.tet3d import assemble_tet, tet_instance            # WP7 contract
    from fejepa.theory import run_theory_checks                       # WP6 pass

    assert callable(run_theory_checks)
    assert callable(tet_instance) and callable(assemble_tet)
    from fejepa.results import write_figures, write_results          # WP1 acceptance

    assert callable(run_wp2) and callable(write_results) and callable(write_figures)
    runner_src = (ROOT / "src/fejepa/experiments/runner.py").read_text()
    assert "run_wp2" in runner_src and "write_results" in runner_src
    assert "run_theory_checks" in runner_src                          # WP6 wired


def test_gate_module_exposes_g1_prime():
    gate = importlib.import_module("fejepa.experiments.gate")
    assert callable(gate.g1_prime)
    assert "G1'" in gate.PLAN_REF
