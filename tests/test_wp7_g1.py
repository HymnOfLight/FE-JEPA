"""WP7 3D-G1 -- gmsh corpus backend tests (envelope memo of 21 August 2026).

gmsh-gated where meshing is involved; the SimJEB schema-audit test is pure
filesystem. Mirrors the tet3d contract battery on unstructured meshes: the
whole point of G1 is that everything downstream of the archive schema is
already dimension- and structure-agnostic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from fejepa.fe.gmsh3d import (generate_gmsh3d_dataset, gmsh3d_instance,
                              mesh_box_with_cavities, plane_face_traction,
                              sample_params3d)
from fejepa.fe.tet3d import _tet_geometry, tet_strain_energy
from fejepa.models.features import (FeatureSpec, build_features_battery,
                                    geometry_descriptor)

RNG = lambda s: np.random.default_rng(s)


def test_sampling_ranges_and_cavity_separation():
    rng = RNG(0)
    for _ in range(20):
        p = sample_params3d(rng)
        assert 1.5 <= p["width"] <= 3.0 and 0.8 <= p["height"] <= 1.5
        assert 0.6 <= p["depth"] <= 1.2 and 0.25 <= p["nu"] <= 0.38
        for i, (x, y, z, r) in enumerate(p["holes"]):
            assert r < min(p["width"], p["height"], p["depth"])
            for xx, yy, zz, rr in p["holes"][i + 1:]:
                dist = ((x-xx)**2 + (y-yy)**2 + (z-zz)**2) ** 0.5
                assert dist >= r + rr           # non-overlapping


def _cavity_params():
    return {"width": 2.4, "height": 1.2, "depth": 1.0, "nu": 0.3,
            "holes": [(0.8, 0.6, 0.5, 0.15), (1.7, 0.6, 0.5, 0.12)]}


def test_mesh_with_cavities_valid_and_deterministic():
    p = _cavity_params()
    n1, t1 = mesh_box_with_cavities(p, lc=0.28)
    n2, t2 = mesh_box_with_cavities(p, lc=0.28)
    assert np.array_equal(n1, n2) and np.array_equal(t1, t2)
    vol, _ = _tet_geometry(n1, t1)
    assert (vol > 0).all()
    box = p["width"] * p["height"] * p["depth"]
    cav = sum(4/3 * np.pi * r**3 for *_, r in p["holes"])
    assert abs(vol.sum() - (box - cav)) / box < 0.02      # mesh volume sanity


def test_instance_contract_identities():
    a = gmsh3d_instance(RNG(13), lc=0.30, labelled=True,
                        params=_cavity_params())
    free = ~a.dirichlet_mask
    res = max(np.linalg.norm((a.K @ a.U_star[j] - a.F[j])[free])
              / max(1.0, np.linalg.norm(a.F[j])) for j in range(a.n_loads))
    assert res < 1e-8
    u = a.U_star[0]
    e_rec = tet_strain_energy(a.nodes, a.elements, u, a.meta["material"])
    e_quad = 0.5 * float(u @ (a.K @ u))
    assert abs(e_rec - e_quad) <= 1e-10 * abs(e_quad)
    asym = abs(a.K - a.K.T).max()
    assert asym <= 1e-12 * abs(a.K).max()                 # machine tolerance


def test_traction_quadrature_exact_on_plane():
    a = gmsh3d_instance(RNG(5), lc=0.35, labelled=False)
    ex = a.meta["extra"]
    f = plane_face_traction(a.nodes, a.elements, 0, ex["width"],
                            np.array([0.0, 1.0, 0.0]))
    area = ex["height"] * ex["depth"]                     # cavities are interior
    assert abs(f[1::3].sum() - area) <= 1e-9 * area
    assert abs(f[0::3].sum()) < 1e-12 and abs(f[2::3].sum()) < 1e-12


def test_descriptor_and_features_carry_over():
    p = _cavity_params()
    a = gmsh3d_instance(RNG(7), lc=0.30, labelled=False, params=p)
    g = geometry_descriptor(a.meta)
    assert g.shape == (6,)
    assert np.isclose(g[0], p["width"] / 3.0) and np.isclose(g[2], p["depth"] / 1.2)
    assert np.isclose(g[4], len(p["holes"]) / 3.0)
    cav = sum(4/3 * np.pi * r**3 for *_, r in p["holes"])
    assert np.isclose(g[5], cav / (p["width"] * p["height"] * p["depth"]))
    feats = build_features_battery(a, FeatureSpec(load_summary=True,
                                                  geometry=True, spatial_dim=3))
    assert feats.shape == (a.n_loads, a.n_nodes, 20)


def test_dataset_roundtrip_and_manifest(tmp_path):
    from fejepa.data.archive import load_instance

    out = generate_gmsh3d_dataset(tmp_path / "d", n=3, seed=2, labelled="none",
                                  lc=0.4)
    m = json.load(open(out / "manifest.json"))
    assert m["backend"] == "gmsh3d" and m["n_instances"] == 3
    assert m["lc"] == 0.4 and m["load_names"][0] == "face_down"
    rec = m["instances"][0]
    a = load_instance(out / rec["file"])
    assert a.meta["extra"]["backend"] == "gmsh3d" and a.n_nodes == rec["n_nodes"]
    assert not rec["labelled"]


def test_runner_and_cli_wiring(tmp_path):
    from fejepa.experiments.runner import _ensure_dataset
    from fejepa.fe.solve import SolveLedger

    dcfg = {"dir": str(tmp_path / "g"), "n": 2, "seed": 3,
            "backend": "gmsh3d", "lc": 0.4}
    ddir = _ensure_dataset(dcfg, SolveLedger())
    assert (ddir / "manifest.json").exists()
    m = json.load(open(ddir / "manifest.json"))
    assert m["backend"] == "gmsh3d" and m["n_instances"] == 2

    from fejepa.experiments.runner import _ensure_multires
    mdir = _ensure_multires({"backend": "gmsh3d", "dir": str(tmp_path / "g"),
                             "lc": 0.4, "seed": 3}, 2.0, 2, SolveLedger())
    mm = json.load(open(mdir / "manifest.json"))
    assert len(mm["pairs"]) == 2 and mm["backend"] == "gmsh3d"


def test_simjeb_schema_audit_on_mock_tree(tmp_path):
    from scripts.simjeb_schema_audit import audit_tree

    root = tmp_path / "simjeb"
    (root / "meshes").mkdir(parents=True)
    (root / "meshes" / "0001.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    (root / "splits.json").write_text(json.dumps({"train": [1], "test": [2]}))
    (root / "labels.csv").write_text("id,mass\n1,0.5\n")
    rep = audit_tree(root)
    assert rep["counts_by_ext"][".obj"] == 1
    assert rep["split_files"] and "splits.json" in rep["split_files"][0]
    obj = rep["samples"][".obj"]
    assert obj["vertices"] == 3 and obj["faces"] == 1
    assert rep["samples"][".csv"]["header"] == ["id", "mass"]
