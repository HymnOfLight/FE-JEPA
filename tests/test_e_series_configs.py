"""wp8-lejepa: the E-series configurations are generated from the stamped
configurations, validate under --dry-run without touching data, refuse
unfilled placeholders, and the step plan respects AR-only mode."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from fejepa.experiments.cost import count_steps
from fejepa.experiments.runner import run_config

ROOT = Path(__file__).resolve().parents[1]


def _gen(tmp_path):
    out = tmp_path / "cfgs"
    out.mkdir()
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_e_series_configs.py"),
                        "--phase2", str(ROOT / "configs" / "phase2_v1.json"),
                        "--phase1", str(ROOT / "configs" / "phase1_rec8_v2.json"),
                        "--out-dir", str(out)], capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert r.returncode == 0, r.stderr
    return out


def test_generated_configs_dry_run_without_data(tmp_path):
    out = _gen(tmp_path)
    for name, kind, exps in (("e2_m512", "bottleneck", {"e8", "p3_transfer"}),
                             ("e2_m1024", "bottleneck", {"e8", "p3_transfer"}),
                             ("e1_2d_base", "fejepa", {"e8"})):
        cfg = json.loads((out / f"{name}.json").read_text())
        cfg["data"]["dir"] = str(tmp_path / "does-not-exist")      # must never be created
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(cfg))
        s = run_config(str(p), dry_run=True)
        assert s["dry_run"] and s["model_kind"] == kind and s["ar_only"]
        assert set(s["experiments_enabled"]) == exps
        assert s["label_need_pool_prefix"] == 0
        assert s["prereg_status"].startswith("would refuse")        # drafts not stamped yet
        assert not (tmp_path / "does-not-exist").exists()


def test_shaped_placeholders_are_refused(tmp_path):
    import pytest

    out = _gen(tmp_path)
    cfg = json.loads((out / "e1_2d_shaped.json").read_text())
    assert cfg["pretrain"]["loss_spec"]["lambda_reg"] is None
    p = tmp_path / "shaped.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="unfilled placeholders"):
        run_config(str(p), dry_run=True)


def test_count_steps_respects_ar_only():
    base = {"experiments": {"e8": {"enabled": True, "seeds": 3, "ar_epochs": 200,
                                   "pool_sizes": [1024], "budgets": [16, 64, 256, 1024],
                                   "sup_epochs": 200, "include_mgn": True}}}
    full = count_steps(base)["e8"]
    ar_only = count_steps({"experiments": {"e8": dict(base["experiments"]["e8"], ar_only=True)}})["e8"]
    assert ar_only == 3 * 200 * 1024 and ar_only < full


def test_probe_r2_recovers_linear_structure():
    from fejepa.analysis.separation import probe_r2

    rng = np.random.default_rng(0)
    G = rng.standard_normal((40, 6))
    W = rng.standard_normal((6, 32))
    X = G @ W + 0.01 * rng.standard_normal((40, 32))
    assert probe_r2(X, G) > 0.95
    assert probe_r2(rng.standard_normal((40, 32)), G) < 0.5


def test_committed_configs_are_exactly_the_generator_output(tmp_path):
    """'Generated, not hand-edited': the committed E-series configurations must
    reproduce byte for byte from the generator (a silent hand edit fails here)."""
    out = _gen(tmp_path)
    for name in ("e2_m512", "e2_m1024", "e1_2d_base", "e1_2d_shaped"):
        assert (out / f"{name}.json").read_bytes() == (ROOT / "configs" / f"{name}.json").read_bytes(), name
