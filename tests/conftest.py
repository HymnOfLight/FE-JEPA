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
