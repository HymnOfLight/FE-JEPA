"""WP7 3D-P0: the four shared-module patches (RUN_PLAN_2026-08-05 §3.2) plus the
S1-enablement dispatch points (decoder/MGN I/O, baselines, theory interpolation,
tet3d corpus backend).

Contract, per the guard-rail in the run plan: every patch (a) works on (E, 4)
tetrahedra and (b) leaves the 2D path bit-identical -- the golden references
below are the literal v2.1.5 implementations, inlined."""

import numpy as np
import pytest

from fejepa.baselines import (GlobalPolyBaseline, KNNFieldBaseline,
                              ScaleAwarePolyBaseline, zero_predictor)
from fejepa.fe.tet3d import generate_tet3d_dataset, tet_instance, tet_von_mises
from fejepa.metrics import evaluate_fields, evaluate_model, vm_suite
from fejepa.models.features import (FeatureSpec, GEOMETRY_DIM, base_dim,
                                    build_features, geometry_descriptor,
                                    load_summary, load_summary_dim,
                                    spatial_dim_of)
from fejepa.models.fejepa import element_edges, mesh_adjacency
from fejepa.theory import (_interp_to, prop1_cross_geometry_counterexample,
                           run_theory_checks)


@pytest.fixture(scope="module")
def arch3d():
    return tet_instance(np.random.default_rng(7), nx=3, ny=3, nz=2, labelled=True)


@pytest.fixture(scope="module")
def archs3d():
    r = np.random.default_rng(21)
    return [tet_instance(r, nx=3, ny=2, nz=2, labelled=True) for _ in range(4)]


# ------------------------------------------------------------------- P0.1 ----

def _mesh_adjacency_v215(elements, n_nodes):
    """Literal v2.1.5 triangle-only body (the golden 2D reference)."""
    nbrs = [set() for _ in range(n_nodes)]
    for a, b, c in elements:
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    return [np.fromiter(s, dtype=np.int64) for s in nbrs]


def _gnn_edges_v215(elements):
    """Literal v2.1.5 triangle-only edge builder (the golden 2D reference)."""
    e = np.concatenate([elements[:, [0, 1]], elements[:, [1, 2]],
                        elements[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    return np.concatenate([e, e[:, ::-1]], axis=0).T


def test_element_edges_single_tet():
    e = element_edges(np.array([[0, 1, 2, 3]]))
    und = {tuple(sorted(c)) for c in e.T.tolist()}
    assert und == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    assert e.shape == (2, 12)                       # 6 undirected, both directions


def test_mesh_adjacency_accepts_tets(arch3d):
    adj = mesh_adjacency(arch3d.elements, arch3d.n_nodes)   # v2.1.5: crashed here
    assert len(adj) == arch3d.n_nodes
    pairs = {(min(a, b), max(a, b))
             for a, nb in enumerate(adj) for b in nb.tolist()}
    ref = {tuple(sorted(c)) for c in element_edges(arch3d.elements).T.tolist()}
    assert pairs == ref                              # adjacency == element edge set
    for a, nb in enumerate(adj):                     # symmetric, no self-loops
        assert a not in nb.tolist()
        assert all(a in adj[b].tolist() for b in nb.tolist())


def test_mesh_adjacency_2d_bit_identical(instances):
    a = instances[0]
    new = mesh_adjacency(a.elements, a.n_nodes)
    old = _mesh_adjacency_v215(a.elements, a.n_nodes)
    assert all(np.array_equal(x, y) for x, y in zip(new, old, strict=True))


def test_element_edges_2d_bit_identical(instances):
    a = instances[0]
    assert np.array_equal(element_edges(a.elements), _gnn_edges_v215(a.elements))


# ------------------------------------------------------------------- P0.2 ----

def test_geometry_descriptor_3d(arch3d):
    g = geometry_descriptor(arch3d.meta)
    ex = arch3d.meta["extra"]
    assert g.shape == (GEOMETRY_DIM,) and np.isfinite(g).all()
    assert np.allclose(g, [ex["width"] / 3.0, ex["height"] / 1.5,
                           ex["depth"] / 1.2, arch3d.meta["material"]["nu"],
                           0.0, 0.0])


def test_geometry_descriptor_2d_unchanged(instances):
    m = instances[0].meta
    ex = m["extra"]
    holes = ex.get("holes", []) or []
    hole_area = float(sum(np.pi * r * r for _, _, r in holes))
    mean_r = float(np.mean([r for _, _, r in holes])) if holes else 0.0
    golden = np.array([ex["width"] / 3.0, ex["height"] / 1.5,
                       m["material"]["nu"], len(holes) / 3.0,
                       hole_area / (ex["width"] * ex["height"]),
                       mean_r / min(ex["width"], ex["height"])])
    assert np.array_equal(geometry_descriptor(m), golden)


# ------------------------------------------------------------------- P0.3 ----

def test_load_summary_3d_gravity_has_z_and_negative_y(arch3d):
    s = load_summary(arch3d.F, 3, 3)                 # gravity acts in -y
    assert s.shape == (load_summary_dim(3),) == (5,)
    assert s[2] < 0 and np.isclose(s[1], 0.0) and np.isclose(s[3], 0.0)
    assert 0.0 < s[4] <= 1.0


def test_load_summary_2d_bit_identical(instances):
    a = instances[0]
    for j in range(a.n_loads):
        fscale = np.abs(a.F).max() + 1e-12
        f = a.F[j].reshape(-1, 2)
        mag = np.linalg.norm(f, axis=1)
        golden = np.array([mag.sum() / (f.shape[0] * fscale),
                           f[:, 0].sum() / (f.shape[0] * fscale),
                           f[:, 1].sum() / (f.shape[0] * fscale),
                           float((mag > 1e-14 * fscale).mean())])
        assert np.array_equal(load_summary(a.F, j), golden)


def test_featurespec_dims_and_roundtrip():
    assert FeatureSpec().dim == 16                   # frozen v2.1.5 value
    assert FeatureSpec(spatial_dim=3).dim == 9 + 5 + 6 == 20
    assert base_dim(3) == 9 and load_summary_dim(3) == 5
    assert FeatureSpec.from_dict(None).spatial_dim == 2
    d3 = FeatureSpec(load_summary=False, geometry=True, spatial_dim=3)
    assert FeatureSpec.from_dict(d3.to_dict()) == d3


def test_build_features_3d_shape_and_mismatch_guard(arch3d):
    spec = FeatureSpec(spatial_dim=3)
    x = build_features(arch3d, 0, spec)
    assert x.shape == (arch3d.n_nodes, spec.dim) and np.isfinite(x).all()
    assert spatial_dim_of(arch3d) == 3
    with pytest.raises(ValueError, match="spatial_dim"):
        build_features(arch3d, 0, FeatureSpec())     # 2D spec on a 3D instance


def test_build_features_2d_bit_identical(instances):
    a, spec = instances[0], FeatureSpec()
    coords = a.nodes - a.nodes.mean(axis=0, keepdims=True)
    coords = coords / (np.sqrt((coords ** 2).sum(axis=1).mean()) + 1e-8)
    fscale = np.abs(a.F).max() + 1e-12
    golden = np.concatenate([
        coords, a.dirichlet_mask.reshape(-1, 2).astype(np.float64),
        a.F[0].reshape(-1, 2) / fscale,
        np.broadcast_to(load_summary(a.F, 0), (a.n_nodes, 4)),
        np.broadcast_to(geometry_descriptor(a.meta), (a.n_nodes, 6)),
    ], axis=1).astype(np.float32)
    assert np.array_equal(build_features(a, 0, spec), golden)


# ------------------------------------------------------------------- P0.4 ----

def test_vm_suite_3d_dispatch_exact_on_labels(arch3d):
    out = vm_suite(arch3d.U_star, arch3d)
    assert np.allclose(out["vm_rel_l2"], 0.0) and np.allclose(
        out["crit_recall"], 1.0)
    vm_direct = tet_von_mises(arch3d.nodes, arch3d.elements, arch3d.U_star[0],
                              arch3d.meta["material"])
    assert vm_direct.shape == (arch3d.elements.shape[0],)


def test_vm_suite_3d_detects_error(arch3d, rng):
    U = arch3d.U_star + 0.2 * rng.standard_normal(arch3d.U_star.shape) \
        * arch3d.free_mask * np.abs(arch3d.U_star).max()
    out = vm_suite(U, arch3d)
    assert out["vm_rel_l2"].min() > 0.0


def test_evaluate_fields_3d(arch3d):
    vals = evaluate_fields(arch3d.U_star, arch3d)
    assert np.isclose(vals["disp_rel_l2"], 0.0)
    assert np.isclose(vals["energy_gap_rel"], 0.0, atol=1e-12)
    assert all(np.isfinite(v) for v in vals.values())


# ------------------------------------------- S1 enablement: baselines (E5') --

def test_zero_predictor_3d(archs3d):
    out = evaluate_model(zero_predictor, archs3d)
    assert np.isclose(out["disp_rel_l2"], 1.0)
    assert out["n_val"] == len(archs3d)


def test_poly_baselines_fit_predict_3d(archs3d):
    for cls in (GlobalPolyBaseline, ScaleAwarePolyBaseline):
        U = cls().fit(archs3d[:3]).predict(archs3d[3])
        assert U.shape == archs3d[3].F.shape and np.isfinite(U).all()
        assert np.abs(U[:, archs3d[3].dirichlet_mask]).max() == 0.0


def test_knn_field_transports_own_field_3d(archs3d):
    model = KNNFieldBaseline().fit(archs3d[:3])
    a = archs3d[0]                                   # its own nearest neighbour
    U = model.predict(a)
    rel = np.linalg.norm(U - a.U_star) / np.linalg.norm(a.U_star)
    assert rel < 1e-8                                # exact at the data points


# ---------------------------------------- S1 enablement: theory (WP6/Prop.1) --

def test_interp_to_self_is_identity_3d(arch3d):
    f = arch3d.U_star[0]
    back = _interp_to(arch3d, f, arch3d)
    assert np.linalg.norm(back - f) / np.linalg.norm(f) < 1e-10


def test_prop1_cross_geometry_runs_3d(archs3d):
    out = prop1_cross_geometry_counterexample(archs3d)
    assert set(out) >= {"naive_extension_falsified", "witness", "metric"}


def test_wp6_corpus_level_pass_3d(archs3d):
    res = run_theory_checks(archs3d, {"n_check": 3, "seed": 0})
    assert not res["kills"][0]["triggered"]
    m = res["metrics"]
    assert m["conditioning"]["holds"] and m["mode_contraction"]["holds"] \
        and m["chebyshev_polish"]["holds"]
    assert m["prop1_premise"]["min_separation"] > 0.0


# --------------------------------------- S1 enablement: tet3d corpus backend --

def test_generate_tet3d_dataset_roundtrip(tmp_path):
    from fejepa.data.archive import instance_files, load_manifest
    from fejepa.experiments.protocol import load_split

    out = generate_tet3d_dataset(tmp_path / "d3", n=4, seed=5)
    m = load_manifest(out)
    assert m["backend"] == "tet3d" and m["n_instances"] == 4
    assert all(not r["labelled"] for r in m["instances"])   # WP5: unlabelled
    files = instance_files(out)
    assert len(files) == 4
    split = load_split(out, n_val=1, seed=1)
    assert len(split.val_files) == 1 and len(split.pool_files) == 3
    from fejepa.data.archive import load_instance

    a = load_instance(files[0])
    assert a.nodes.shape[1] == 3 and a.elements.shape[1] == 4
    assert a.meta["extra"]["dim"] == 3 and not a.labelled


def test_runner_ensure_dataset_tet3d(tmp_path):
    from fejepa.experiments.runner import _ensure_dataset, _ensure_multires
    from fejepa.fe.solve import SolveLedger

    dcfg = {"dir": str(tmp_path / "d3r"), "n": 3, "seed": 2, "backend": "tet3d",
            "labelled_policy": "economy"}
    ddir = _ensure_dataset(dcfg, SolveLedger())
    assert (ddir / "manifest.json").exists()
    # E4-3D wiring landed: multires now yields a pairs manifest for tet3d
    mdir = _ensure_multires({**dcfg, "nx": 4, "ny": 3, "nz": 3}, 2.0, 2,
                            SolveLedger())
    import json as _json
    mm = _json.load(open(mdir / "manifest.json"))
    assert len(mm["pairs"]) == 2 and mm["coarsen"] == 2.0


# ----------------------------------- torch-gated: encoder/decoder + MGN I/O --

def test_fejepa_forward_3d_shapes(arch3d):
    torch = pytest.importorskip("torch")
    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa, mesh_adjacency

    cfg = FEJEPAConfig(dim=16, depth=1, heads=2,
                       features=FeatureSpec(spatial_dim=3))
    model = build_fejepa(cfg)
    pack = model.prepare_instance(arch3d, "cpu")
    with torch.no_grad():
        U = model.forward_instance(pack)
    assert tuple(U.shape) == (arch3d.n_loads, arch3d.ndof)
    adj = mesh_adjacency(arch3d.elements, arch3d.n_nodes)
    with torch.no_grad():
        loss = model.masked_prediction(pack["feats"][0], adj,
                                       np.random.default_rng(0))
    assert float(loss) >= 0.0


def test_mgn_forward_3d_shapes(arch3d):
    torch = pytest.importorskip("torch")
    from fejepa.models.gnn import build_mesh_gnn

    model = build_mesh_gnn(dim=16, depth=1,
                           features=FeatureSpec(spatial_dim=3))
    pack = model.prepare_instance(arch3d, "cpu")
    with torch.no_grad():
        U = model.forward_instance(pack)
    assert tuple(U.shape) == (arch3d.n_loads, arch3d.ndof)
