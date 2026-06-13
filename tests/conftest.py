import numpy as np
import pytest

from fejepa.fe.generator import GeneratorConfig, sample_instance, sample_instance_multires
from fejepa.data.archive import load_problem, save_problem


def _coarse_cfg(max_holes=0):
    # Coarse mesh keeps the unit tests fast.
    return GeneratorConfig(
        width_range=(1.6, 1.8),
        height_range=(0.9, 1.0),
        mesh_size_frac=(0.18, 0.22),
        max_holes=max_holes,
    )


@pytest.fixture(scope="session")
def small_problem():
    rng = np.random.default_rng(0)
    return sample_instance(rng, _coarse_cfg(max_holes=0), labelled=True)


@pytest.fixture(scope="session")
def small_archive(small_problem, tmp_path_factory):
    d = tmp_path_factory.mktemp("inst")
    path = d / "inst.npz"
    save_problem(path, small_problem)
    return load_problem(path)


@pytest.fixture(scope="session")
def multires_pair():
    rng = np.random.default_rng(1)
    cfg = GeneratorConfig(
        width_range=(1.6, 1.8),
        height_range=(0.9, 1.0),
        mesh_size_frac=(0.10, 0.12),
        max_holes=0,
    )
    return sample_instance_multires(rng, cfg, coarsen=2.2, labelled=True)
