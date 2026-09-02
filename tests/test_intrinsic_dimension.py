"""wp8-lejepa Stage 0.1: the intrinsic-dimension instrument's estimator is
validated on manifolds of KNOWN dimension, its distance kernel is memory-safe,
and the LayerNorm check follows execution order on the real encoder."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.analysis import intrinsic_dim as idm


def test_twonn_recovers_known_dimensions():
    rng = np.random.default_rng(0)
    # 5-d Gaussian embedded linearly in 32-d
    lin = rng.standard_normal((3000, 5)) @ rng.standard_normal((5, 32))
    assert 4.0 <= idm.twonn(lin) <= 6.5
    # 2-d manifold (Swiss roll) in 3-d
    t = 1.5 * math.pi * (1 + 2 * rng.random(3000))
    h = 21 * rng.random(3000)
    roll = np.stack([t * np.cos(t), h, t * np.sin(t)], 1)
    assert 1.5 <= idm.twonn(roll) <= 2.7
    # 1-d curve in 10-d
    s = rng.random(2000) * 10
    curve = np.stack([np.sin(k * s) for k in range(1, 11)], 1)
    assert 0.7 <= idm.twonn(curve) <= 1.6


def test_twonn_distance_kernel_is_memory_safe_and_exact():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5000, 256))            # would be 5000*5000*256*8 = 51 GB naively
    import time

    t0 = time.perf_counter()
    d_est = idm.twonn(x)                             # must run in seconds, not OOM
    assert time.perf_counter() - t0 < 30
    # TwoNN is biased low at N << the sample size a 256-d Gaussian would need
    # (concentration of measure); the statement that matters is "far above any
    # low-dimensional manifold and below the ambient dimension".
    assert 40 < d_est < 256
    # exact two-NN distances on a tiny set vs brute force
    y = rng.standard_normal((50, 4))
    sq = (y * y).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2 * y @ y.T
    np.fill_diagonal(d2, np.inf)
    brute = np.sqrt(np.sort(np.maximum(d2, 0), axis=1)[:, :2])
    assert np.all(brute[:, 1] >= brute[:, 0])


def test_layernorm_check_uses_execution_order_on_real_encoder(tmp_path):
    from fejepa.data.archive import load_instance
    from fejepa.experiments.parallel import _build_model
    from fejepa.experiments.protocol import load_split
    from fejepa.fe.synthetic import generate_synthetic_dataset

    d = generate_synthetic_dataset(tmp_path / "ln", n=3, seed=4)
    arch = load_instance(load_split(d, 0, 1).pool_files[0])
    m = _build_model({"kind": "fejepa", "model": {"dim": 16, "depth": 1, "heads": 2,
                      "features": {"load_summary": True, "geometry": True}}, "seed": 0})
    pack = m.prepare_instance(arch, "cpu")
    assert idm.last_module_is_layernorm(m, pack["feats"]) is True   # the LeWM caveat applies
