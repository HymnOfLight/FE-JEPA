import numpy as np

from fejepa.experiments.falsification import cross_resolution_gap, load_multires_split
from fejepa.fe.generator import GeneratorConfig, generate_multires_dataset
from fejepa.models.fejepa import FEJEPA, FEJEPAConfig


def test_multires_generation_and_split(tmp_path):
    cfg = GeneratorConfig(
        width_range=(1.6, 1.8), height_range=(0.9, 1.0),
        mesh_size_frac=(0.12, 0.14), max_holes=0,
    )
    d = generate_multires_dataset(tmp_path / "mr", n_instances=4, seed=0, coarsen=2.0,
                                  cfg=cfg, verbose=False)
    train, val = load_multires_split(d, n_val=2, seed=0)
    assert len(train) == 2 and len(val) == 2
    fine, coarse = train[0]
    assert fine.n_nodes >= coarse.n_nodes
    # shared geometry
    assert np.isclose(fine.meta["width"], coarse.meta["width"])


def test_cross_resolution_gap_runs(tmp_path):
    cfg = GeneratorConfig(
        width_range=(1.6, 1.8), height_range=(0.9, 1.0),
        mesh_size_frac=(0.14, 0.16), max_holes=0,
    )
    d = generate_multires_dataset(tmp_path / "mr", n_instances=3, seed=1, coarsen=2.0,
                                  cfg=cfg, verbose=False)
    _, val = load_multires_split(d, n_val=2, seed=0)
    model = FEJEPA(FEJEPAConfig(dim=32, depth=2))
    g = cross_resolution_gap(model, val)
    assert {"rel_l2_coarse", "rel_l2_fine", "transfer_gap"} <= set(g)
    assert g["transfer_gap"] >= 0.0
