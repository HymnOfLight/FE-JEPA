"""wp9 Stage 0a: the OOD-2D v1 evaluation families and the 2D remesh set.
Each family changes one attribute of the training family and nothing else;
generation is deterministic; manifests pin every file by SHA-256."""

import json

import numpy as np
import pytest

from fejepa.fe.ood2d import (DEFAULT_SEEDS, F5_TARGET_H, FAMILIES, REMESH_H,
                             sample_family_params)


def _draws(family, n=400, seed=5):
    return [sample_family_params(np.random.default_rng(c), family)
            for c in np.random.SeedSequence(seed).spawn(n)]


def _training_ranges_except(ps, attr):
    """Every attribute other than `attr` stays in the training ranges."""
    checks = {
        "width": all(1.5 <= p["width"] <= 3.0 for p in ps),
        "height": all(0.8 <= p["height"] <= 1.5 for p in ps),
        "nu": all(0.25 <= p["nu"] <= 0.38 for p in ps),
        "target_h": all(0.05 <= p["target_h"] <= 0.12 for p in ps),
        "holes": all(len(p["holes"]) <= 3 for p in ps),
        "radius": all(0.06 <= r / min(p["width"], p["height"]) <= 0.16
                      for p in ps for *_, r in p["holes"]),
        "traction": all(0.025 <= t <= 0.075 for p in ps for t in p["traction_scales"]),
    }
    skip = {"F1": {"holes"}, "F2": {"width", "height"}, "F3": {"holes", "radius"},
            "F4": {"nu"}, "F5": {"target_h"}}[attr]
    return {k: v for k, v in checks.items() if k not in skip}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_each_family_changes_one_attribute(family):
    ps = _draws(family)
    assert all(ok for ok in _training_ranges_except(ps, family).values())
    assert all(p["family"] == family for p in ps)
    for p in ps:                                    # every hole inside the plate
        for cx, cy, r in p["holes"]:
            assert min(cx, p["width"] - cx, cy, p["height"] - cy) > r
    if family == "F1":
        assert {len(p["holes"]) for p in ps} == {4, 5, 6}
    if family == "F2":
        assert all(3.0 <= p["width"] <= 4.5 and 0.6 <= p["height"] <= 0.8 for p in ps)
    if family == "F3":
        assert {len(p["holes"]) for p in ps} == {1, 2, 3}
        for p in ps:
            rel = [r / min(p["width"], p["height"]) for *_, r in p["holes"]]
            lo, hi = (0.16, 0.22) if p["hole_class"] == "large" else (0.03, 0.06)
            assert all(lo <= x <= hi for x in rel)
            if p["hole_class"] == "large":           # clearance to the edge
                for cx, cy, r in p["holes"]:
                    assert min(cx, p["width"] - cx, cy, p["height"] - cy) - r >= \
                        0.02 * min(p["width"], p["height"]) - 1e-12
        n_large = sum(p["hole_class"] == "large" for p in ps)
        assert 150 < n_large < 250                   # one class per instance, p = 1/2
    if family == "F4":
        assert all(0.40 <= p["nu"] <= 0.45 for p in ps)
    if family == "F5":
        assert all(p["target_h"] == F5_TARGET_H for p in ps)


def test_sampling_is_deterministic_and_seeds_are_distinct():
    assert _draws("F3", 20) == _draws("F3", 20)
    assert len(set(DEFAULT_SEEDS.values())) == len(DEFAULT_SEEDS)


def test_generated_family_and_remesh_set(tmp_path):
    pytest.importorskip("gmsh")
    pytest.importorskip("skfem")
    from fejepa.data.archive import instance_files, load_instance, load_manifest, manifest_sha256
    from fejepa.fe.ood2d import generate_family, generate_remesh, verify_manifest_files

    d = generate_family(tmp_path / "F5", "F5", n=2)
    m = load_manifest(d)
    assert m["n_instances"] == 2 and m["ledger"]["total"] == 8 and verify_manifest_files(d) == []
    assert "seconds" not in m and "wall_clock_s" not in m["ledger"]     # no volatile field
    for f in instance_files(d):
        a = load_instance(f)
        assert a.labelled and a.meta["extra"]["family"] == "F5"
        assert a.meta["extra"]["target_h"] == F5_TARGET_H
        # labels solve the system on the free dofs
        free = a.free_mask
        res = (a.K @ a.U_star.T).T[:, free] - a.F[:, free]
        assert np.abs(res).max() < 1e-8 * np.abs(a.F).max()
    assert manifest_sha256(generate_family(tmp_path / "F5b", "F5", n=2)) == manifest_sha256(d)
    (d / "instance_00000.npz").write_bytes(b"tampered")
    assert verify_manifest_files(d) == ["instance_00000.npz"]

    r = generate_remesh(tmp_path / "R", 1, hs=REMESH_H[:3])
    recs = load_manifest(r)["instances"]
    assert [x["target_h"] for x in recs] == list(REMESH_H[:3])
    archs = [load_instance(r / x["file"]) for x in recs]
    assert len({a.n_nodes for a in archs}) == 3                 # the meshes differ ...
    res = [a.F.reshape(4, -1, 2).sum(axis=1) for a in archs]
    # ... the loads do not: the three edge tractions exactly; gravity up to the
    # area of the polygonal holes, which the mesh resolves more finely
    assert all(np.allclose(x[:3], res[0][:3], rtol=1e-9, atol=1e-15) for x in res)
    assert all(np.allclose(x[3], res[0][3], rtol=1e-2) for x in res)
    ex = [a.meta["extra"] for a in archs]
    assert all(e["width"] == ex[0]["width"] and e["holes"] == ex[0]["holes"] for e in ex)
    assert json.loads((r / "manifest.json").read_text())["hs"] == list(REMESH_H[:3])


def test_inband_holdout_is_the_training_family(tmp_path):
    """IB (PREREG_W9 r3): instance i is the training sampler's draw for child i
    of SeedSequence(91007), meshed and labelled as the corpus is."""
    pytest.importorskip("gmsh")
    pytest.importorskip("skfem")
    from fejepa.data.archive import instance_files, load_instance, load_manifest, manifest_sha256
    from fejepa.fe.generator import sample_params
    from fejepa.fe.ood2d import IB_N, generate_inband, verify_manifest_files

    assert IB_N == 2048 and DEFAULT_SEEDS["IB"] == 91007
    d = generate_inband(tmp_path / "IB", n=3)
    m = load_manifest(d)
    assert m["family"] == "IB" and m["seed"] == 91007 and m["n_instances"] == 3
    assert m["ledger"]["total"] == 12 and verify_manifest_files(d) == []
    want = [sample_params(np.random.default_rng(c))
            for c in np.random.SeedSequence(91007).spawn(3)]
    for f, w in zip(instance_files(d), want, strict=True):
        a = load_instance(f)
        ex = a.meta["extra"]
        assert a.labelled and ex["family"] == "IB" and ex["target_h"] == w["target_h"]
        assert ex["width"] == w["width"] and ex["holes"] == w["holes"]
        assert a.meta["material"]["nu"] == w["nu"] and 0.05 <= ex["target_h"] <= 0.12
    assert manifest_sha256(generate_inband(tmp_path / "IB2", n=3)) == manifest_sha256(d)


def test_set_sizes_and_seeds_are_prereg_w9s():
    """PREREG_W9 Sec. 3's sizes and seeds, which scripts/w9_make_ood2d.py
    generates by default and the session-2 plan and the adjudication check."""
    from fejepa.fe.ood2d import SET_SIZES

    assert SET_SIZES == {"F1": 256, "F2": 256, "F3": 256, "F4": 256, "F5": 256, "R": 80,
                         "IB": 2048}
    assert DEFAULT_SEEDS == {"F1": 91001, "F2": 91002, "F3": 91003, "F4": 91004, "F5": 91005,
                             "R": 91006, "IB": 91007}


def test_manifest_check_reports_missing_and_changed_files(tmp_path):
    import hashlib

    from fejepa.data.archive import write_manifest
    from fejepa.fe.ood2d import verify_manifest_files

    recs = []
    for name in ("a.npz", "b.npz", "c.npz"):
        (tmp_path / name).write_bytes(name.encode())
        recs.append({"file": name, "sha256": hashlib.sha256(name.encode()).hexdigest()})
    write_manifest(tmp_path, recs, {"family": "F5"})
    assert verify_manifest_files(tmp_path) == []
    (tmp_path / "a.npz").unlink()
    (tmp_path / "c.npz").write_bytes(b"changed")
    assert verify_manifest_files(tmp_path) == ["a.npz", "c.npz"]
