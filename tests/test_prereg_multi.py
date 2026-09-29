"""Stage 1.28: one pre-registration governs several configurations (the E1 and
E2 arms). Labelled lines `CONFIG_SHA256[<config stem>] = ...`; every arm must
be stamped before any arm runs; each config verifies against its own line;
unlabelled single-line files (Phase 1 / 2 / 2b) behave exactly as before."""

import json
from pathlib import Path

import pytest

from fejepa.report import (PREREG_PLACEHOLDER, config_sha256, read_prereg_entries,
                           stamp_prereg, verify_prereg)

TWO = (f"# PREREG_EX\n\nCONFIG_SHA256[arm_a] = {PREREG_PLACEHOLDER}\n"
       f"CONFIG_SHA256[arm_b] = {PREREG_PLACEHOLDER}\n")


def _cfgs():
    return {"x": 1, "arm": "a"}, {"x": 1, "arm": "b"}


def test_every_arm_must_be_stamped_before_any_arm_runs(tmp_path):
    pf = tmp_path / "P.md"; pf.write_text(TWO)
    a, b = _cfgs()
    stamp_prereg(pf, a, label="arm_a")
    with pytest.raises(ValueError, match=r"unstamped \(arm_b\)"):
        verify_prereg(a, pf, label="arm_a")          # arm_b not frozen yet
    stamp_prereg(pf, b, label="arm_b")
    assert verify_prereg(a, pf, label="arm_a") == config_sha256(a)
    assert verify_prereg(b, pf, label="arm_b") == config_sha256(b)
    assert [lab for lab, _ in read_prereg_entries(pf)] == ["arm_a", "arm_b"]


def test_each_config_verifies_only_against_its_own_line(tmp_path):
    pf = tmp_path / "P.md"; pf.write_text(TWO)
    a, b = _cfgs()
    stamp_prereg(pf, a, label="arm_a"); stamp_prereg(pf, b, label="arm_b")
    with pytest.raises(ValueError, match="mismatch"):
        verify_prereg(a, pf, label="arm_b")
    with pytest.raises(ValueError, match="mismatch"):
        verify_prereg(dict(a, x=2), pf, label="arm_a")  # edited after stamping


def test_multi_line_file_refuses_an_unlabelled_stamp_and_unknown_labels(tmp_path):
    pf = tmp_path / "P.md"; pf.write_text(TWO)
    a, _ = _cfgs()
    with pytest.raises(ValueError, match="stamp each with its label"):
        stamp_prereg(pf, a)
    with pytest.raises(ValueError, match=r"CONFIG_SHA256\[arm_c\]"):
        stamp_prereg(pf, a, label="arm_c")
    assert pf.read_text() == TWO                     # nothing written on refusal


def test_single_unlabelled_line_keeps_the_phase2_semantics(tmp_path):
    pf = tmp_path / "P.md"
    pf.write_text(f"CONFIG_SHA256 = {PREREG_PLACEHOLDER}\nPREREG_X_SHA256 = <record>\n")
    a, _ = _cfgs()
    h = stamp_prereg(pf, a, label="anything")        # a label is ignored on unlabelled files
    assert verify_prereg(a, pf, label="whatever") == h
    assert "PREREG_X_SHA256 = <record>" in pf.read_text()


def test_the_drafts_old_placeholder_token_is_not_a_stamp_line(tmp_path):
    """r11/r5 drafts carried `<fill at stamping>`, which the tool does not
    recognise (the Phase-2 r9 lesson): stamping would refuse on the box."""
    pf = tmp_path / "P.md"; pf.write_text("CONFIG_SHA256 = <fill at stamping>\n")
    with pytest.raises(ValueError, match="no CONFIG_SHA256 line"):
        stamp_prereg(pf, {"x": 1})


def test_cli_stamps_the_line_named_after_the_config(tmp_path):
    from fejepa.cli import main

    pf = tmp_path / "P.md"; pf.write_text(TWO)
    a, b = _cfgs()
    (tmp_path / "arm_a.json").write_text(json.dumps(a))
    (tmp_path / "arm_b.json").write_text(json.dumps(b))
    main(["prereg", str(tmp_path / "arm_b.json"), "--stamp", "--prereg-file", str(pf)])
    ents = dict(read_prereg_entries(pf))
    assert ents["arm_b"] == config_sha256(b) and ents["arm_a"] == PREREG_PLACEHOLDER


def test_runner_guard_uses_the_config_stem(tmp_path):
    """Two arms under one PREREG through the production runner (dry run):
    refused until both are stamped, then each verifies."""
    from fejepa.experiments.runner import run_config

    pf = tmp_path / "P.md"
    pf.write_text(f"CONFIG_SHA256[e_base] = {PREREG_PLACEHOLDER}\n"
                  f"CONFIG_SHA256[e_shaped] = {PREREG_PLACEHOLDER}\n")
    base = {"model": {"dim": 16, "depth": 1, "heads": 2,
                      "features": {"load_summary": True, "geometry": True}},
            "data": {"dir": str(tmp_path / "nope"), "n": 4, "seed": 1, "backend": "synthetic"},
            "split": {"n_val": 1, "seed": 1},
            "experiments": {"e8": {"enabled": True, "ar_only": True, "budgets": [2],
                                   "pool_sizes": [2], "seeds": 1, "ar_epochs": 1}},
            "prereg_guard": True, "prereg_file": str(pf), "device": "cpu", "workers": 1}
    shaped = json.loads(json.dumps(base))
    shaped["pretrain"] = {"loss_spec": {"reg_mode": "sigreg_ep_head", "lambda_reg": 0.1,
                                        "sigreg_n_proj": 16, "sigreg_head_width": 0}}
    paths = {}
    for name, c in (("e_base", base), ("e_shaped", shaped)):
        c["out"] = str(tmp_path / name / "report.json")
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(c))
    stamp_prereg(pf, base, label="e_base")
    s = run_config(str(paths["e_base"]), dry_run=True)
    assert s["prereg_status"].startswith("would refuse") and "e_shaped" in s["prereg_status"]
    with pytest.raises(ValueError, match="unstamped"):
        run_config(str(paths["e_base"]))              # a real run refuses outright
    stamp_prereg(pf, shaped, label="e_shaped")
    for name in ("e_base", "e_shaped"):
        s = run_config(str(paths[name]), dry_run=True)
        assert s["prereg_status"] == "verified", (name, s["prereg_status"])
    assert not (tmp_path / "nope").exists()
