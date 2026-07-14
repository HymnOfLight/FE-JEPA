"""Archive round-trip, manifests, economy labelling (plan B1/WP5)."""

import numpy as np

from fejepa.data.archive import (add_labels, instance_files, load_instance,
                                 load_manifest, manifest_sha256, mark_labelled)
from fejepa.fe.synthetic import generate_synthetic_dataset


def test_roundtrip(tmp_path, arch):
    from fejepa.data.archive import save_instance

    p = tmp_path / "a.npz"
    save_instance(arch, p)
    back = load_instance(p)
    assert np.allclose(back.nodes, arch.nodes)
    assert np.allclose(back.F, arch.F)
    assert (back.K != arch.K).nnz == 0
    assert np.array_equal(back.dirichlet_mask, arch.dirichlet_mask)
    assert np.allclose(back.U_star, arch.U_star)
    assert back.meta["extra"]["backend"] == "synthetic"


def test_economy_dataset_and_labelling(tmp_path, ledger):
    d = generate_synthetic_dataset(tmp_path / "ds", n=4, seed=0, labelled="none")
    files = instance_files(d)
    assert len(files) == 4
    assert all(not load_instance(f).labelled for f in files)
    assert ledger.total == 0

    a = load_instance(files[0])
    from fejepa.fe.solve import solve_fe_displacement

    U, _ = solve_fe_displacement(a.K, a.F, a.free_mask, method="direct",
                                 ledger=ledger, stage="labelling-val")
    add_labels(files[0], U)
    mark_labelled(d, {files[0].name})
    assert load_instance(files[0]).labelled
    m = load_manifest(d)
    assert m["instances"][0]["labelled"] is True
    assert ledger.as_dict()["per_stage"]["labelling-val"] == a.n_loads


def test_manifest_sha_changes(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "ds", n=2, seed=0)
    s1 = manifest_sha256(d)
    mark_labelled(d, {"instance_00000.npz"})
    assert manifest_sha256(d) != s1
