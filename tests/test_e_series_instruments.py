"""Stage 1.28: the E-series instruments as the runbook calls them (subprocess,
--smoke): the lambda pilot never touches the E1 validation split and sizes
the head from the pilot's own AR model; the bench times the bottleneck
without per-call set-up."""

import json
import subprocess
import sys
from pathlib import Path

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


def test_bench_times_the_bottleneck_differentially(tmp_path):
    out = tmp_path / "bench.json"
    _run([str(ROOT / "scripts" / "bench_phase2_preconditions.py"),
          str(ROOT / "configs" / "phase2_v1.json"), "--smoke", "--bottleneck-tokens", "4",
          "--repeats", "4", "--out", str(out)], tmp_path)
    res = json.loads(out.read_text())
    assert res["git"] and res["numeric_policy"]["tf32"] is True
    ph = res["phases"]
    for tag in ("bottleneck4_inband_0", "bottleneck4_fine"):
        p = ph[tag]
        assert p["timing"] == "differential" and p["steps"][1] > p["steps"][0]
        assert p["ms_per_step"] > 0 and p["setup_ms"] >= 0
