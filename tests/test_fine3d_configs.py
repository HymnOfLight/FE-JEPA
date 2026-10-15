"""fine3d (PREREG_FINE3D Sec. 2-3): the three configurations are the
generator's; R2a's differ from E2's stamped M = 1,024 configuration only in
the keys Sec. 2 names, S's factor is the scale-factor record's, and R3's
schedule is the labelled few-shot fine-tuning's in the Phase-2b report."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fejepa.analysis.adjudicate_w9 import config_diff
from fejepa.report import config_sha256

ROOT = Path(__file__).resolve().parents[1]


def _gen():
    spec = importlib.util.spec_from_file_location("make_fine3d_configs",
                                                  ROOT / "scripts" / "make_fine3d_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text())


def test_the_committed_configurations_are_the_generators():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_fine3d_configs.py"),
                        "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert list(_gen().ARMS) == ["r2a_b_m1024", "r2a_s_m1024", "r3_eft_b16"]


def test_the_generator_refuses_another_base(tmp_path, monkeypatch):
    g = _gen()
    base = _cfg("e2_m1024")
    assert config_sha256(base) == g.BASE_SHA256
    pre = (ROOT / "PREREG_E2.md").read_text()
    assert f"CONFIG_SHA256[e2_m1024] = {g.BASE_SHA256}" in pre
    bad = dict(base, workers=3)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "e2_m1024.json").write_text(json.dumps(bad))
    monkeypatch.setattr(g, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="not PREREG_E2's stamped"):
        g.load_base()


def test_r2a_differs_from_e2_only_in_the_named_keys():
    g = _gen()
    base = _cfg("e2_m1024")
    named = {("_comment",), ("out",), ("prereg_file",), ("experiments", "e8", "seed_offset")}
    s_keys = {("model", "decode_scale"), ("model", "decode_scale_factor"),
              ("model", "features", "load_density")}
    b, s = _cfg("r2a_b_m1024"), _cfg("r2a_s_m1024")
    assert config_diff(b, base) == named
    assert config_diff(s, base) == named | s_keys
    for c in (b, s):
        assert c["experiments"]["e8"]["seed_offset"] == 3 and c["prereg_guard"] is True
        assert c["prereg_file"] == "PREREG_FINE3D.md"
        assert c["model"]["kind"] == "bottleneck" and c["model"]["n_tokens"] == 1024
    assert (s["model"]["decode_scale"], s["model"]["decode_scale_factor"],
            s["model"]["features"]["load_density"]) == ("l1", 2.0 ** -10, True)
    assert g.S3_FACTOR == 2.0 ** -10
    assert b["out"] == "runs/fine3d/r2a_b_m1024/report.json"
    assert s["out"] == "runs/fine3d/r2a_s_m1024/report.json"


def test_s_factor_is_the_records_power_of_two():
    g = _gen()
    rec = json.loads((ROOT / "records" / "wp9" / "fine3d" / "scale_factor_3d.json").read_text())
    spec = importlib.util.spec_from_file_location("fine3d_scale_factor",
                                                  ROOT / "scripts" / "fine3d_scale_factor.py")
    sf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sf)
    assert rec["sample"]["factor"] == g.S3_FACTOR == sf.factor_of(rec["sample"]["median"])
    assert sf.factor_of(0.0151) == 1 / 64                     # wp9's rule in 2D


def test_r3_schedule_is_the_labelled_fine_tunings():
    g = _gen()
    rep = json.loads((ROOT / g.PHASE2B_REPORT).read_text())
    p3 = rep["config"]["experiments"]["p3_transfer"]
    c = _cfg("r3_eft_b16")
    assert c["train"]["epochs"] == p3["fewshot_epochs"] == 50
    assert c["train"]["lr"] == p3["fewshot_lr"] == 1.5e-3
    assert c["train"]["n"] == 16 and 16 in p3["fewshot_budgets"]
    from fejepa.train.pretrain import PretrainConfig
    from fejepa.train.supervised import SupervisedConfig

    for cls in (PretrainConfig, SupervisedConfig):            # the trainers' defaults
        d = cls()
        assert (d.weight_decay, d.clip) == (c["train"]["weight_decay"], c["train"]["clip"])
    assert c["seeds"] == rep["provenance"]["seeds"] == [0, 1, 2]
    assert c["source"]["report_sha256"] == g.PHASE2B_REPORT_SHA256
    assert c["evaluation"] == {"fine": 256, "inband": 256, "amplitude": True}
    assert c["out"] == "runs/fine3d/r3_eft_b16/report.json"
    assert c["prereg_file"] == "PREREG_FINE3D.md" and c["prereg_guard"] is True


def test_r2a_configurations_pass_the_runners_checks():
    pytest.importorskip("torch")
    from fejepa.experiments.runner import run_config

    for name in ("r2a_b_m1024", "r2a_s_m1024"):
        out = run_config(ROOT / "configs" / f"{name}.json", dry_run=True)
        assert out["model_kind"] == "bottleneck" and out["ar_only"] is True
        assert out["plan_steps"] == {"e8": 614400, "total": 614400}
        assert out["label_need_pool_prefix"] == 0
