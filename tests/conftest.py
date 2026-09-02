"""Shared fixtures: small labelled synthetic instances (no gmsh/skfem/torch needed).

The synthetic backend (fejepa.fe.synthetic) mirrors the audited FE conventions, so the
numpy test suite preserves the plan Sec.2.1 invariants as executable checks on any
machine. Torch/skfem/gmsh-dependent tests importorskip their extras and run on the box.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fejepa.fe.solve import SolveLedger  # noqa: E402
from fejepa.fe.synthetic import synthetic_instance  # noqa: E402


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def instances():
    """8 small labelled instances (varied geometry via the generator's sampling)."""
    r = np.random.default_rng(123)
    return [synthetic_instance(r, nx=6, ny=4, labelled=True) for _ in range(8)]


@pytest.fixture(scope="session")
def arch(instances):
    return instances[0]


@pytest.fixture()
def ledger():
    return SolveLedger()


# ---- wp8-lejepa: shared tiny labelled corpus for the E-series tests ----------
TINY_MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 16, "mgn_depth": 2,
              "features": {"load_summary": True, "geometry": True}}


@pytest.fixture
def tiny_corpus(tmp_path):
    """Factory: tiny_corpus(seed=..., n=..., n_val=..., n_label=...) -> split with
    the val set and the first `n_label` pool instances labelled (canonical
    version of the per-file `_corpus` helpers; use it in new tests)."""
    def make(seed: int = 21, n: int = 6, n_val: int = 2, n_label: int = 3):
        from fejepa.experiments.protocol import load_split
        from fejepa.experiments.runner import _label_files
        from fejepa.fe.synthetic import generate_synthetic_dataset

        d = generate_synthetic_dataset(tmp_path / f"corpus_{seed}", n=n, seed=seed)
        sp = load_split(d, n_val=n_val, seed=1)
        led = SolveLedger()
        _label_files(sp.val_files, led, "v")
        _label_files(sp.pool_files[:n_label], led, "p")
        return sp
    return make
