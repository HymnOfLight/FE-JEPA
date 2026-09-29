"""E1 adjudication instrument: the separation statistic is exact on
constructed cases and the script's real path runs kind-aware."""


import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.analysis import separation as ls


def test_silhouette_and_bins_on_constructed_clusters():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=float)
    labels = np.repeat(np.arange(4), 25)
    x = centres[labels] + 0.1 * rng.standard_normal((100, 2))
    assert ls.silhouette(x, labels) > 0.95                    # tight, far clusters
    assert ls.loo_1nn_accuracy(x, labels) == 1.0
    mixed = rng.standard_normal((100, 2))
    assert abs(ls.silhouette(mixed, labels)) < 0.15           # no structure ~ 0
    bins = ls.quartile_bins(np.arange(100, dtype=float))
    assert np.bincount(bins).tolist() == [25, 25, 25, 25]


def test_kind_guard_refuses_fejepa_only_probes_under_other_kinds(tmp_path):
    import json

    from fejepa.experiments.runner import run_config

    cfg = {"model": {"kind": "bottleneck", "dim": 16},
           "experiments": {"e6": {"enabled": True}}, "out": str(tmp_path / "r.json")}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="FE-JEPA-only"):
        run_config(str(p))


def test_instance_files_returns_the_runs_val_split(tmp_path):
    from fejepa.analysis.common import instance_files
    from fejepa.experiments.protocol import load_split
    from fejepa.fe.synthetic import generate_synthetic_dataset

    d = generate_synthetic_dataset(tmp_path / "vs", n=10, seed=8)
    split = {"n_val": 3, "seed": 1}
    val = instance_files(d, split=split, subset="val")
    assert [str(f) for f in val] == [str(f) for f in load_split(d, 3, seed=1).val_files]
    pool = instance_files(d, split=split, subset="pool")
    assert len(pool) == 7 and not set(map(str, pool)) & set(map(str, val))
    assert len(instance_files(d)) == 10                      # no split: whole pool


def test_separation_reports_validity_for_tiny_sets(tmp_path):
    from fejepa.analysis.common import build_model_from_config
    from fejepa.analysis.separation import measure_separation
    from fejepa.data.archive import load_instance
    from fejepa.experiments.protocol import load_split
    from fejepa.fe.synthetic import generate_synthetic_dataset

    mcfg = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}
    m = build_model_from_config(mcfg)
    d = generate_synthetic_dataset(tmp_path / "t", n=16, seed=6)
    files = load_split(d, 0, 1).pool_files
    small = measure_separation(m, [load_instance(f) for f in files[:3]])
    assert small["S_valid"] is False and small["S_invalid_reason"]
    big = measure_separation(m, [load_instance(f) for f in files])
    assert big["S_valid"] is True and big["bins"] == [4, 4, 4, 4]


def test_bootstrap_ci_brackets_the_point_estimate():
    from fejepa.analysis.separation import bootstrap_silhouette, silhouette

    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [6, 0], [0, 6], [6, 6]], dtype=float)
    labels = np.repeat(np.arange(4), 30)
    x = centres[labels] + 0.8 * rng.standard_normal((120, 2))
    s = silhouette(x, labels)
    lo, hi = bootstrap_silhouette(x, labels, n_boot=100)
    assert lo <= s <= hi and (hi - lo) < 0.3


def test_interval_has_no_duplicate_inflation_and_tracks_sampling_spread():
    """Stage 1.28: on clouds with a known population, the interval contains
    the point estimate and its half-width is of the order of the true
    sampling spread (population redraws); the retired with-replacement
    bootstrap put its whole interval above S."""
    from fejepa.analysis.separation import bootstrap_silhouette, silhouette

    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(4), 64)
    for shift in (0.0, 0.4):
        centres = rng.standard_normal((4, 64)) * shift
        x = centres[labels] + rng.standard_normal((256, 64))
        s = silhouette(x, labels)
        lo, hi = bootstrap_silhouette(x, labels, n_boot=80)
        assert lo <= s <= hi
        redraw = [silhouette(centres[labels] + np.random.default_rng(100 + k).standard_normal((256, 64)),
                             labels) for k in range(30)]
        half = 0.5 * (hi - lo)
        sd = float(np.std(redraw))
        assert 0.5 * 1.96 * sd < half < 2.5 * 1.96 * sd, (half, sd)


def test_stage131_silhouette_is_exact_under_a_large_shared_offset():
    """Pooled latents can share a large offset (small inter-instance distances
    relative to the latent norm). The float32 Gram expansion cancelled
    catastrophically there; distances are now float64 on centred data, so S
    does not depend on the offset."""
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(4), 32)
    x = rng.standard_normal((4, 64))[labels] * 0.5 + rng.standard_normal((128, 64))
    ref = ls.silhouette(x.astype(np.float64), labels)
    shifted = (x + 3e3).astype(np.float32)                     # offset ~ 3000 x spread
    assert ls.silhouette(shifted, labels) == pytest.approx(ref, abs=1e-5)
    assert ls.loo_1nn_accuracy(shifted, labels) == ls.loo_1nn_accuracy(x, labels)


def _sep_file(tmp_path, name, **kw):
    import json

    rec = {"state": "", "subset": "val", "S_silhouette": 0.1, "S_valid": True,
           "n_instances": 256, "smoke": False, "config_sha256": "cfgA", "state_sha256": None,
           "data_manifest_sha256": "corpusA"}
    rec.update(kw)
    f = tmp_path / name
    f.write_text(json.dumps(rec))
    return str(f)


def test_stage131_separation_files_are_tied_to_the_exact_state_config_and_split(tmp_path):
    from scripts.adjudicate_e1 import _separation_by_seed

    run = tmp_path / "e1_2d_base"
    (run / "e8_states").mkdir(parents=True)
    rpath = run / "report.json"
    states = {f"s{s}": {"sha256": f"st{s}"} for s in range(3)}
    report = {"config": {"split": {"n_val": 256, "seed": 1}},
              "provenance": {"config_sha256": "cfgA", "seeds": [0, 1, 2],
                             "datasets": [{"dir": "runs/data2d", "manifest_sha256": "corpusA"}]},
              "results": {"e8": {"metrics": {"d9_restart": {"ar_states": states}}}}}

    def files(over=None):
        over = over or {}
        out = []
        for s in range(3):
            kw = {"state": str(run / "e8_states" / f"ar_p1024_s{s}.pt"),
                  "state_sha256": f"st{s}", "S_silhouette": 0.1 * (s + 1)}
            kw.update(over.get(s, {}))
            out.append(_sep_file(tmp_path, f"sep_s{s}.json", **kw))
        return out

    assert _separation_by_seed(files(), str(rpath), report) == pytest.approx([0.1, 0.2, 0.3])
    bad = [({1: {"state_sha256": "overwritten"}}, "not the state the report trained"),
           ({2: {"config_sha256": "cfgB"}}, "measured with configuration"),
           ({0: {"n_instances": 128}}, "measured on 128 instances"),
           ({0: {"S_valid": False, "S_invalid_reason": "bins"}}, "not a valid reading"),
           ({1: {"smoke": True}}, "not a valid reading"),
           ({1: {"subset": "pool"}}, "PREREG_E1 fixes 'val'"),
           ({2: {"data_manifest_sha256": "corpusB"}}, "not one of the report's datasets")]
    for over, msg in bad:
        with pytest.raises(SystemExit, match=msg):
            _separation_by_seed(files(over), str(rpath), report)
    # Stage 1.33: a non-finite S (diverged latents) is passed through for
    # adjudicate_e1 to judge; an invalid reading with a finite S is refused
    got = _separation_by_seed(files({1: {"S_valid": False, "S_silhouette": float("nan")}}),
                              str(rpath), report)
    assert got[0] == pytest.approx(0.1) and got[1] != got[1]


def test_stage131_separation_script_records_state_and_config_hashes(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    from fejepa.analysis.common import build_model_from_config
    from fejepa.fe.synthetic import generate_synthetic_dataset
    from fejepa.report import config_sha256

    root = Path(__file__).resolve().parents[1]
    d = generate_synthetic_dataset(tmp_path / "c", n=20, seed=6)
    mcfg = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}
    cfg = {"model": mcfg, "split": {"n_val": 8, "seed": 1}}
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps(cfg))
    spath = tmp_path / "ar_p1024_s0.pt"
    torch.save(build_model_from_config(mcfg).state_dict(), spath)
    out = tmp_path / "sep.json"
    r = subprocess.run([sys.executable, str(root / "scripts" / "latent_separation.py"),
                        "--config", str(cpath), "--state", str(spath), "--data", str(d),
                        "--n-instances", "8", "--device", "cpu", "--out", str(out)],
                       capture_output=True, text=True, cwd=tmp_path,
                       env={"PYTHONPATH": str(root / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert r.returncode == 0, r.stderr[-2000:]
    rec = json.loads(out.read_text())
    assert rec["state_sha256"] == hashlib.sha256(spath.read_bytes()).hexdigest()
    assert rec["config_sha256"] == config_sha256(cfg) and rec["n_instances"] == 8
    from fejepa.data.archive import manifest_sha256
    assert rec["data_manifest_sha256"] == manifest_sha256(d) and rec["data"] == str(d)
    assert rec["subset"] == "val" and rec["smoke"] is False
