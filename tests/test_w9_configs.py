"""wp9 Stage 0b: the wp9 configurations are the generator's output, derived
from E1's stamped base and differing from it only in PREREG_W9 Sec. 2's keys;
PREREG_W9's labelled lines name exactly these configurations (and, once
stamped, record their hashes); every configuration validates under --dry-run."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fejepa.analysis.adjudicate_w9 import (ALLOWED_DIFF, DEFAULT_SPECS, S_FACTOR, _arm_identity,
                                           config_diff, label_of)
from fejepa.experiments.cost import count_steps
from fejepa.report import PREREG_PLACEHOLDER, config_sha256, read_prereg_entries

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("w9_c1_n1024", "w9_c1_n4096", "w9_c1_n25600", "w9_c1_n12800", "w9_b_n1024",
        "w9_s_n1024")


def _gen():
    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "make_w9_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(arm):
    return json.loads((ROOT / "configs" / f"{arm}.json").read_text())


def test_committed_configurations_are_the_generators():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_w9_configs.py"), "--check"],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout + r.stderr
    mk = _gen()
    assert tuple(mk.ARMS) == ARMS and mk.S_FACTOR == S_FACTOR


def test_configurations_differ_from_e1_only_where_prereg_w9_says():
    base = json.loads((ROOT / "configs" / "e1_2d_base.json").read_text())
    assert config_sha256(base) == _gen().BASE_SHA256
    for arm in ARMS:
        cfg = _cfg(arm)
        assert config_diff(cfg, base) <= ALLOWED_DIFF, arm
        assert _arm_identity(cfg) == DEFAULT_SPECS[arm], arm         # the adjudicator's view
        assert label_of({"config": cfg}) == arm
        e8 = cfg["experiments"]["e8"]
        assert e8["pool_sizes"][0] * e8["ar_epochs"] == 204800
        assert cfg["prereg_file"] == "PREREG_W9.md" and cfg["prereg_guard"] is True
        assert cfg["out"] == f"runs/w9/{arm[3:]}/report.json"
        ev = cfg["evaluation"]
        assert ev["amplitude"] is True
        assert ev["holdouts"] == {h: {"dir": f"runs/w9/ood2d/{h}", "family": h}
                                  for h in ("IB", "F1", "F2", "F3", "F4", "F5", "R")}
    assert _cfg("w9_c1_n1024")["experiments"]["e8"]["reuse_from"] == {
        "report": "records/wp8/e1/e1_2d_base/report.json",
        "states_dir": "runs/e1_2d_base/e8_states"}
    assert count_steps(_cfg("w9_c1_n1024"))["e8"] == 0
    assert count_steps(_cfg("w9_c1_n25600"))["e8"] == 3 * 204800
    b, s = _cfg("w9_b_n1024"), _cfg("w9_s_n1024")
    assert b["experiments"]["e8"]["seed_offset"] == s["experiments"]["e8"]["seed_offset"] == 3
    assert config_diff(b, s) == {("model", "decode_scale"), ("model", "decode_scale_factor"),
                                 ("model", "features", "load_density"), ("out",), ("_comment",)}
    assert s["model"]["decode_scale"] == "l1" and s["model"]["decode_scale_factor"] == 1 / 64
    assert s["model"]["features"]["load_density"] is True
    assert "seed_offset" not in _cfg("w9_c1_n25600")["experiments"]["e8"]


def test_generator_refuses_another_base(tmp_path):
    mk = _gen()
    base = json.loads((ROOT / "configs" / "e1_2d_base.json").read_text())
    base["pretrain"]["lr"] = 2e-3
    p = tmp_path / "b.json"
    p.write_text(json.dumps(base))
    with pytest.raises(SystemExit, match="not E1's stamped base"):
        mk.load_base(str(p))


def test_prereg_lines_name_the_configurations():
    pf = ROOT / "PREREG_W9.md"
    entries = read_prereg_entries(pf)
    assert [lab for lab, _ in entries] == list(ARMS)
    stamped = [v for _, v in entries if v != PREREG_PLACEHOLDER]
    assert len(stamped) in (0, len(ARMS))                 # all open, or all stamped
    if stamped:
        for lab, v in entries:
            assert v == config_sha256(_cfg(lab)), lab
        text = pf.read_text()
        m = re.search(r"PREREG_W9_SHA256 = ([0-9a-f]{64})", text)
        assert m, "a stamped file records its own hash"
        body = text.replace(m.group(1), "<record after commit>")
        assert hashlib.sha256(body.encode()).hexdigest() == m.group(1)


def test_configurations_validate_under_dry_run(tmp_path):
    from fejepa.experiments.runner import run_config

    for arm in ARMS:
        s = run_config(str(ROOT / "configs" / f"{arm}.json"), dry_run=True,
                       activation_checkpointing=False)
        assert s["dry_run"] and s["ar_only"] and s["prereg_guard"]
        assert s["prereg_status"] in ("verified",) or s["prereg_status"].startswith("would refuse")
        assert s["activation_checkpointing"] is False
        # the holdouts (and E1's states) live on the box: refused here, by name
        assert "w9" in s and ("error" in s["w9"] or s["w9"].get("holdouts"))
