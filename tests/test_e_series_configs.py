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


E_SERIES = ("e2_m512", "e2_m1024", "e1_2d_base", "e1_2d_shaped", "e1_2d_raw_s0")


def _committed_fill():
    """The stamping-time values recorded in the committed shaped config (None
    while unstamped) -- regenerating with them must reproduce the commit."""
    ls = json.loads((ROOT / "configs" / "e1_2d_shaped.json").read_text())["pretrain"]["loss_spec"]
    return ls["lambda_reg"], ls["sigreg_head_width"]


def _gen(tmp_path, lam=None, width=None):
    out = tmp_path / "cfgs"
    out.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(ROOT / "scripts" / "make_e_series_configs.py"),
            "--phase2b", str(ROOT / "configs" / "phase2b_v1.json"),
            "--phase1", str(ROOT / "configs" / "phase1_rec8_v2.json"),
            "--out-dir", str(out)]
    if lam is not None:
        args += ["--e1-lambda", repr(float(lam))]
    if width is not None:
        args += ["--e1-head-width", str(int(width))]
    r = subprocess.run(args, capture_output=True, text=True,
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
    reproduce byte for byte from the generator -- run with the stamping-time
    lambda / head width the committed shaped config records (Stage 1.28: the
    fill no longer breaks this test)."""
    lam, width = _committed_fill()
    out = _gen(tmp_path, lam, width)
    for name in E_SERIES:
        assert (out / f"{name}.json").read_bytes() == (ROOT / "configs" / f"{name}.json").read_bytes(), name


def test_filling_at_stamping_touches_only_the_placeholders(tmp_path):
    blank = _gen(tmp_path / "a")
    filled = _gen(tmp_path / "b", 0.1, 24)
    for name in ("e2_m512", "e2_m1024", "e1_2d_base"):
        assert (blank / f"{name}.json").read_bytes() == (filled / f"{name}.json").read_bytes()
    sh = json.loads((filled / "e1_2d_shaped.json").read_text())["pretrain"]["loss_spec"]
    raw = json.loads((filled / "e1_2d_raw_s0.json").read_text())["pretrain"]["loss_spec"]
    assert sh == {"reg_mode": "sigreg_ep_head", "lambda_reg": 0.1, "sigreg_n_proj": 256,
                  "sigreg_head_width": 24}
    assert raw == {"reg_mode": "sigreg_ep", "lambda_reg": 0.1, "sigreg_n_proj": 256}
    for name in ("e1_2d_shaped", "e1_2d_raw_s0"):
        a = json.loads((blank / f"{name}.json").read_text()); a["pretrain"].pop("loss_spec")
        b = json.loads((filled / f"{name}.json").read_text()); b["pretrain"].pop("loss_spec")
        assert a == b, name


def test_e2_configs_are_the_baseline_config_but_for_the_architecture(tmp_path):
    """E2's baseline is the Phase-2b report: every section that shapes the AR
    cells (corpus, split, seeds, pool, epochs, lr, numeric policy, labels,
    features, dims) must be the Phase-2b configuration's, byte for byte."""
    base = json.loads((ROOT / "configs" / "phase2b_v1.json").read_text())
    for m in (512, 1024):
        c = json.loads((ROOT / "configs" / f"e2_m{m}.json").read_text())
        allowed = {"model", "experiments", "out", "prereg_file", "prereg_guard", "_comment"}
        assert set(c) == set(base)
        for k in set(c) - allowed:
            assert c[k] == base[k], k
        arch = ("kind", "n_tokens", "decode_k")
        assert {k: v for k, v in c["model"].items() if k not in arch} == base["model"]
        assert c["model"]["kind"] == "bottleneck" and c["model"]["n_tokens"] == m
        assert c["model"]["decode_k"] == 6          # Stage 1.33: the continuous decoder
        e8, b8 = dict(c["experiments"]["e8"]), dict(base["experiments"]["e8"])
        assert e8.pop("ar_only") is True and e8 == b8
        p3, bp3 = dict(c["experiments"]["p3_transfer"]), dict(base["experiments"]["p3_transfer"])
        assert (p3.pop("fewshot_budgets"), p3.pop("naive_budget")) == ([], 0)
        bp3.pop("fewshot_budgets"); bp3.pop("naive_budget")
        assert p3 == bp3


def test_raw_ablation_config_is_one_seed_and_refuses_its_placeholder(tmp_path):
    import pytest

    c = json.loads((ROOT / "configs" / "e1_2d_raw_s0.json").read_text())
    assert c["experiments"]["e8"]["seeds"] == 1 and c["experiments"]["e8"]["ar_only"] is True
    assert c["pretrain"]["loss_spec"]["reg_mode"] == "sigreg_ep"
    if c["pretrain"]["loss_spec"]["lambda_reg"] is None:
        p = tmp_path / "raw.json"
        p.write_text(json.dumps(c))
        with pytest.raises(SystemExit, match="unfilled placeholders"):
            run_config(str(p), dry_run=True)


# ---- Stage 1.31: the pilot JSON fills the configurations AND PREREG_E1 ------------

def _pilot_json(tmp_path, **over):
    import hashlib

    p1 = ROOT / "configs" / "phase1_rec8_v2.json"
    rows = {"0.0": {"disp_rel_l2": 0.100}, "0.01": {"disp_rel_l2": 0.101},
            "0.1": {"disp_rel_l2": 0.104}, "1.0": {"disp_rel_l2": 0.120}}
    rec = {"selected_lambda": 0.1, "admissible": [0.01, 0.1], "rows": rows,
           "head_width": 20, "head_width_source": "auto: intrinsic-dimension rule",
           "rule": "largest lambda with pilot-val disp <= AR * (1 + 0.05)",
           "epochs": 20, "n_train": 512, "n_val": 128, "smoke": False,
           "pilot_ledger": {"per_stage": {}, "total": 0, "wall_clock_s": 0.0},
           "manifest_sha256_before": "m", "manifest_sha256_after": "m",
           "split": {"n_val": 256, "seed": 1}, "numeric_policy": {"tf32": True},
           "config_sha256": hashlib.sha256(p1.read_bytes()).hexdigest()}
    rec.update(over)
    f = tmp_path / "e1_pilot.json"
    f.write_text(json.dumps(rec))
    return f


def _gen_from_pilot(tmp_path, pilot, prereg=None, extra=()):
    out = tmp_path / "cfgs"
    out.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, str(ROOT / "scripts" / "make_e_series_configs.py"),
            "--phase2b", str(ROOT / "configs" / "phase2b_v1.json"),
            "--phase1", str(ROOT / "configs" / "phase1_rec8_v2.json"),
            "--out-dir", str(out), "--e1-from-pilot", str(pilot), *extra]
    if prereg is not None:
        args += ["--fill-prereg", str(prereg)]
    return out, subprocess.run(args, capture_output=True, text=True, cwd=tmp_path,
                                env={"PYTHONPATH": str(ROOT / "src"),
                                     "PATH": "/usr/bin:/bin:/usr/local/bin"})


def test_generator_fills_configs_and_prereg_from_the_pilot_json(tmp_path):
    import hashlib

    from scripts.make_e_series_configs import PLACEHOLDER, PREREG_E1_PARAMS

    pilot = _pilot_json(tmp_path)
    prereg = tmp_path / "PREREG_E1.md"
    # Stage 1.33: start from the placeholders even after the committed file is
    # filled at stamping (the test must stay green on the stamped head)
    blank = f"LAMBDA = `{PLACEHOLDER}`; WIDTH = `{PLACEHOLDER}`; pilot record SHA-256 = `{PLACEHOLDER}`"
    prereg.write_text(PREREG_E1_PARAMS.sub(blank, (ROOT / "PREREG_E1.md").read_text()))
    out, r = _gen_from_pilot(tmp_path, pilot, prereg)
    assert r.returncode == 0, r.stderr
    sh = json.loads((out / "e1_2d_shaped.json").read_text())["pretrain"]["loss_spec"]
    raw = json.loads((out / "e1_2d_raw_s0.json").read_text())["pretrain"]["loss_spec"]
    assert (sh["lambda_reg"], sh["sigreg_head_width"], raw["lambda_reg"]) == (0.1, 20, 0.1)
    text = prereg.read_text()
    sha = hashlib.sha256(pilot.read_bytes()).hexdigest()
    assert f"LAMBDA = `0.1`; WIDTH = `20`; pilot record SHA-256 = `{sha}`" in text
    # identical to the explicit-value path, byte for byte
    same = _gen(tmp_path / "explicit", 0.1, 20)
    for name in E_SERIES:
        assert (out / f"{name}.json").read_bytes() == (same / f"{name}.json").read_bytes(), name
    # re-filling with the same pilot is a no-op; another pilot's values are refused
    _, r2 = _gen_from_pilot(tmp_path, pilot, prereg)
    assert r2.returncode == 0 and prereg.read_text() == text
    od = tmp_path / "o"
    od.mkdir()
    rows1 = {"0.0": {"disp_rel_l2": 0.1}, "0.01": {"disp_rel_l2": 0.1},
             "0.1": {"disp_rel_l2": 0.1}, "1.0": {"disp_rel_l2": 0.1}}
    _, r3 = _gen_from_pilot(od, _pilot_json(od, selected_lambda=1.0, rows=rows1,
                                            admissible=[0.01, 0.1, 1.0]), prereg)
    assert r3.returncode != 0 and "already filled" in r3.stderr
    assert prereg.read_text() == text


def test_generator_refuses_pilots_that_cannot_configure_e1(tmp_path):
    cases = [({"selected_lambda": None}, "NO-GO-AT-PILOT"),
             ({"smoke": True}, "smoke pilot"),
             ({"pilot_ledger": {"total": 5}}, "bought labels"),
             ({"manifest_sha256_after": "changed"}, "manifest changed"),
             ({"split": {"n_val": 128, "seed": 1}}, "not the E1 split"),
             ({"numeric_policy": {"tf32": False}}, "numeric policy"),
             ({"config_sha256": "0" * 64}, "not this Phase-1 config"),
             # Stage 1.33: the pre-registered protocol and a selection that follows from the rows
             ({"epochs": 1}, "epochs = 1"),
             ({"n_train": 256}, "n_train = 256"),
             ({"rows": {"0.0": {"disp_rel_l2": 0.1}, "0.1": {"disp_rel_l2": 0.1}}}, "grid"),
             ({"rule": "largest lambda with pilot-val disp <= AR * (1 + 0.1)"}, "5% tolerance"),
             ({"head_width_source": "fixed by --head-width"}, "intrinsic-dimension rule"),
             ({"selected_lambda": 1.0}, "do not follow from the rows"),
             ({"admissible": [0.01]}, "do not follow from the rows")]
    for i, (over, msg) in enumerate(cases):
        d = tmp_path / f"c{i}"
        d.mkdir()
        _, r = _gen_from_pilot(d, _pilot_json(d, **over))
        assert r.returncode != 0 and msg in r.stderr, (over, r.stderr[-300:])
    d = tmp_path / "contra"
    d.mkdir()
    _, r = _gen_from_pilot(d, _pilot_json(d), extra=("--e1-lambda", "0.01"))
    assert r.returncode != 0 and "contradicts the pilot" in r.stderr


def test_committed_prereg_e1_parameters_agree_with_the_committed_configs():
    """PREREG_E1 Sec. 2's LAMBDA / WIDTH line and the committed shaped / raw
    configurations are one fact: placeholders with null values before
    stamping, identical numbers after."""
    from scripts.make_e_series_configs import PLACEHOLDER, PREREG_E1_PARAMS

    hits = list(PREREG_E1_PARAMS.finditer((ROOT / "PREREG_E1.md").read_text()))
    assert len(hits) == 1
    m = hits[0]
    sh = json.loads((ROOT / "configs" / "e1_2d_shaped.json").read_text())["pretrain"]["loss_spec"]
    raw = json.loads((ROOT / "configs" / "e1_2d_raw_s0.json").read_text())["pretrain"]["loss_spec"]
    if m.group("lam") == PLACEHOLDER:
        assert m.group("width") == PLACEHOLDER and m.group("sha") == PLACEHOLDER
        assert sh["lambda_reg"] is None and sh["sigreg_head_width"] is None
        assert raw["lambda_reg"] is None
    else:
        assert float(m.group("lam")) == sh["lambda_reg"] == raw["lambda_reg"]
        assert int(m.group("width")) == sh["sigreg_head_width"]
        assert len(m.group("sha")) == 64
