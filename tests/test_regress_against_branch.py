"""wp8 Stage 1.36/1.37: the cross-branch regression tool compares exactly
(WP6's ARPACK quantities to round-off) and builds its miniature from the
stamped configuration without touching its structure (the end-to-end run
against wp7-3d is a sandbox record in BRANCH_NOTES; it needs a second
checkout)."""

import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "regress_against_branch", ROOT / "scripts" / "regress_against_branch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_compare_is_exact_and_skips_only_timing_and_paths():
    mod = _mod()
    a = {"e8": {"x": [1.0, float("nan"), 2], "wall_clock_s": 3.0, "state_dir": "/a"}}
    b = {"e8": {"x": [1.0, float("nan"), 2], "wall_clock_s": 9.0, "state_dir": "/b"}}
    assert mod.compare(a, b) == ([], 2)
    c = json.loads(json.dumps(b))
    c["e8"]["x"][0] = math.nextafter(1.0, 2.0)                  # one ulp
    diffs, n = mod.compare(a, c)
    assert n == 2 and len(diffs) == 1 and diffs[0].startswith("/e8/x[0]")
    assert mod.compare({"k": 1}, {"j": 1})[0] == ["/j: present on one side only",
                                                   "/k: present on one side only"]


def test_only_wp6_is_compared_to_round_off():
    mod = _mod()
    a = {"results": {"wp6": {"kappa": 6010.490715984713, "max_err": 8.3e-16},
                     "e8": {"x": 6010.490715984713}}}
    b = {"results": {"wp6": {"kappa": 6010.490715984704, "max_err": 1.4e-15},
                     "e8": {"x": 6010.490715984704}}}
    diffs, n = mod.compare(a, b)
    assert n == 3 and diffs == ["/results/e8/x: 6010.490715984713 vs 6010.490715984704"]
    b["results"]["wp6"]["kappa"] = 6011.0
    assert len(mod.compare(a, b)[0]) == 2


def test_miniature_keeps_the_stamped_structure(tmp_path):
    mod = _mod()
    cfg = json.loads((ROOT / "configs" / "phase2b_v1.json").read_text())
    m = mod.miniature(cfg, tmp_path)
    assert set(m) == set(cfg) and set(m["experiments"]) == set(cfg["experiments"])
    assert m["prereg_guard"] is False and m["device"] == "cpu" and m["workers"] == 1
    assert m["experiments"]["e8"]["include_mgn"] == cfg["experiments"]["e8"]["include_mgn"]
    assert m["experiments"]["e8"]["include_ar_ft"] == cfg["experiments"]["e8"]["include_ar_ft"]
    assert m["model"]["features"] == cfg["model"]["features"]
    assert m["pretrain"]["lr"] == cfg["pretrain"]["lr"] and m["sup"]["lr"] == cfg["sup"]["lr"]
    assert m["experiments"]["p3_transfer"]["fewshot_init"] == "ar_seed_matched"
    assert m["out"] == str(tmp_path / "report.json")
    # both lambda policies of labels_anchor are exercised, and WP6 / E6 run
    from fejepa.experiments.e8_regimes import POLICY_BALANCED_FROM

    bud = m["experiments"]["e8"]["budgets"]
    assert min(bud) < POLICY_BALANCED_FROM <= max(bud)
    assert m["gate_g2"]["decision_budget"] in bud
    assert m["experiments"]["wp6"]["enabled"] and m["experiments"]["e6"]["enabled"]
    assert m["labels"]["inband_prefix"] >= max(bud)
    assert m["data"]["n"] >= max(bud) + m["split"]["n_val"]


def test_src_tree_is_gits_tree_id(tmp_path):
    """wp9 Stage 0c: the summary names each side by `src_tree`, which must be
    git's own id of the files' tree: ignored files and empty directories
    skipped, a directory sorted as its name plus '/', the execute bit kept."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("no git")
    mod = _mod()
    src = tmp_path / "src"
    (src / "pkg" / "a").mkdir(parents=True)
    (src / "pkg" / "__init__.py").write_text("x = 1\n")
    (src / "pkg" / "a" / "z.py").write_bytes("\u00e9 = 2\n".encode())
    (src / "pkg" / "a-b.py").write_text("")             # sorts before the directory "a"
    (src / "pkg" / "tool.sh").write_text("#!/bin/sh\n")
    (src / "pkg" / "tool.sh").chmod(0o755)
    (src / "pkg" / "__pycache__").mkdir()
    (src / "pkg" / "__pycache__" / "m.cpython-311.pyc").write_bytes(b"\0")
    (src / "pkg" / "stray.pyc").write_bytes(b"\0")
    (src / "pkg" / "empty").mkdir()
    (src / "fejepa.egg-info").mkdir()
    (src / "fejepa.egg-info" / "PKG-INFO").write_text("x\n")
    shutil.copyfile(ROOT / ".gitignore", tmp_path / ".gitignore")
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}

    def g(*args):
        return subprocess.run([git, "-C", str(tmp_path), "-c", "core.autocrlf=false",
                               "-c", "core.fileMode=true", *args], capture_output=True,
                              text=True, check=True, env=env).stdout.strip()

    g("init", "-q")
    g("add", "-A")
    assert mod.src_tree(src) == g("write-tree", "--prefix=src/")
    (src / "pkg" / "tool.sh").chmod(0o644)                     # the mode is part of the id
    g("add", "-A")
    assert mod.src_tree(src) == g("write-tree", "--prefix=src/")
