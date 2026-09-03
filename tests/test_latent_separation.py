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
