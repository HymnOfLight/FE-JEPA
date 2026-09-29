"""Stage 1.28: the E-series instruments as the runbook calls them (subprocess,
--smoke): the lambda pilot never touches the E1 validation split and sizes
the head from the pilot's own AR model; the bench times the bottleneck
without per-call set-up (Stage 1.31: median of repeated differential pairs
with a validity flag; the smoke test checks the wiring, not a GPU-noise-free
positive number, so it cannot flake on the box)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}


def _run(args, tmp_path):
    r = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=ENV,
                       cwd=tmp_path, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    return r


def test_lambda_pilot_uses_the_e1_split_without_its_validation_set(tmp_path):
    out = tmp_path / "pilot.json"
    _run([str(ROOT / "scripts" / "e1_lambda_pilot.py"), "--smoke", "--out", str(out)], tmp_path)
    r = json.loads(out.read_text())
    assert "disjoint from E1's validation set" in r["pilot_val"]
    assert r["head_width_source"].startswith("auto")
    assert r["intrinsic_dimension"]["suggested_head_width"] == r["head_width"]
    assert set(r["rows"]) == {"0.0", "0.01", "0.1", "1.0"}
    assert r["numeric_policy"]["tf32"] is True and r["git"]           # the runs' policy, traceable
    assert "sha256" in r["prereg"]
    assert r["lr"] == 1e-3 and "e8.ar_lr" in r["lr_source"]          # the arms' learning rate


def test_bench_times_the_bottleneck_differentially(tmp_path):
    out = tmp_path / "bench.json"
    _run([str(ROOT / "scripts" / "bench_phase2_preconditions.py"),
          str(ROOT / "configs" / "phase2_v1.json"), "--smoke", "--bottleneck-tokens", "4",
          "--repeats", "4", "--bottleneck-steps", "4", "--bottleneck-pairs", "3",
          "--out", str(out)], tmp_path)
    res = json.loads(out.read_text())
    assert res["git"] and res["numeric_policy"]["tf32"] is True
    ph = res["phases"]
    for tag in ("bottleneck4_inband_0", "bottleneck4_fine"):
        p = ph[tag]
        assert p["timing"] == "differential" and p["steps"] == [2, 6]
        assert len(p["pairs"]) == 3 and len(p["estimates_ms"]) == 3
        assert p["ms_per_step"] == sorted(p["estimates_ms"])[1]          # the median pair
        assert isinstance(p["valid"], bool) and p["n_tokens"] == 4
        assert p["decode_k"] == 6                                      # Stage 1.33 decoder
        assert p["prepare_ms"] > 0 and p["ms_per_step_incl_setup"] > 0


def test_differential_estimator_is_the_median_pair_and_flags_jitter():
    from scripts.bench_phase2_preconditions import differential_step

    d = differential_step([(1.00, 3.00), (1.10, 3.00), (0.90, 3.20)], 10, 110)
    assert d["estimates_s"] == pytest.approx([0.020, 0.019, 0.023])
    assert d["step_s"] == pytest.approx(0.020) and d["valid"]
    assert d["setup_s"] == pytest.approx(0.80)                         # median of T1 - n1*step
    assert d["spread_rel"] == pytest.approx(0.20)
    # one pair whose set-up hiccup exceeds the timed steps: the median survives,
    # but the phase is flagged invalid and the adjudicator will refuse it
    j = differential_step([(1.0, 3.0), (6.0, 3.1), (1.0, 3.0)], 10, 110)
    assert j["step_s"] == pytest.approx(0.020) and not j["valid"] and j["spread_rel"] is None
    with pytest.raises(ValueError):
        differential_step([(1.0, 2.0)], 10, 10)
