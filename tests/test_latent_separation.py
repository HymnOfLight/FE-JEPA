"""E1 adjudication instrument: the separation statistic is exact on
constructed cases and the script's real path runs kind-aware."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "latent_separation", Path(__file__).resolve().parents[1] / "scripts" / "latent_separation.py")
ls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ls)


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
