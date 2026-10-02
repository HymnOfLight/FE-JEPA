"""wp9 Stage 0b: the S switch -- a mesh-independent decode scale
(`model.decode_scale = "l1"`) and mesh-independent load features
(`model.features.load_density`). Off (the default) every feature and the decode
scale are the wp8 values bit for bit; on, the load features are the same on
every mesh of one geometry, and scaling the loads scales the prediction."""

import numpy as np
import pytest

from fejepa.models.features import (GEOMETRY_DIM, FeatureSpec, battery_fscale, battery_l1,
                                    build_features_battery,
                                    density_load_summary, geometry_descriptor,
                                    load_densities, load_summary, load_summary_dim,
                                    normalized_coords, simplex_geometry, spatial_dim_of)


def _wp8_build_features(arch, load_idx, spec):
    """wp8-lejepa's build_features (5f8e2df), verbatim: the reference the
    default path must reproduce bit for bit."""
    n = arch.n_nodes
    sd = spatial_dim_of(arch)
    coords = normalized_coords(arch.nodes)
    dmask = arch.dirichlet_mask.reshape(-1, sd).astype(np.float64)
    fscale = battery_fscale(arch.F)
    f = arch.F[load_idx].reshape(-1, sd) / fscale
    cols = [coords, dmask, f]
    if spec.load_summary:
        cols.append(np.broadcast_to(load_summary(arch.F, load_idx, sd),
                                    (n, load_summary_dim(sd))))
    if spec.geometry:
        cols.append(np.broadcast_to(geometry_descriptor(arch.meta), (n, GEOMETRY_DIM)))
    return np.concatenate(cols, axis=1).astype(np.float32)


def _synthetic(seed=0, nx=8, ny=6):
    from fejepa.fe.synthetic import synthetic_instance

    return synthetic_instance(np.random.default_rng(seed), nx=nx, ny=ny)


def _tet(seed=0, nx=4, ny=3, nz=3):
    from fejepa.fe.tet3d import tet_instance

    return tet_instance(np.random.default_rng(seed), nx=nx, ny=ny, nz=nz)


def _kuhn(seed=0, n=(3, 2, 2)):
    """A conforming 3D instance (Kuhn split of a structured box, six tetrahedra
    per cell) with consistent P1 loads: uniform tractions on the faces x = w
    and y = h, gravity. (The tet3d smoke backend's five-tetrahedra split covers
    5/6 of each cell and is not conforming, so it cannot serve here.)"""
    from fejepa.data.archive import InstanceArchive

    rng = np.random.default_rng(seed)
    w, h, d = rng.uniform(1.5, 3.0), rng.uniform(0.8, 1.5), rng.uniform(0.6, 1.2)
    nx, ny, nz = n
    xs, ys, zs = np.linspace(0, w, nx + 1), np.linspace(0, h, ny + 1), np.linspace(0, d, nz + 1)
    X = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    paths = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for path in paths:
                    v, cur = [nid(i, j, k)], [i, j, k]
                    for ax in path:
                        cur[ax] += 1
                        v.append(nid(*cur))
                    tets.append(v)
    tets = np.asarray(tets)
    vol, bnd, fm = simplex_geometry(X, tets)
    ts = 0.05 * rng.uniform(0.5, 1.5, size=4)
    F = np.zeros((4, X.shape[0], 3))
    for li, (axis, val, t) in enumerate(((0, w, (0, -ts[0], 0)), (0, w, (ts[1], 0, 0)),
                                         (1, h, (ts[2], 0, 0)))):
        on = np.all(np.isclose(X[bnd][:, :, axis], val), axis=1)
        for f_, a_ in zip(bnd[on], fm[on]):
            F[li, f_] += np.asarray(t) * a_ / 3
    for t_, v_ in zip(tets, vol):
        F[3, t_] += np.array([0.0, -ts[3], 0.0]) * v_ / 4
    dmask = np.zeros(3 * X.shape[0], dtype=bool)
    dmask[(3 * np.nonzero(np.isclose(X[:, 0], 0))[0][:, None] + np.arange(3)).ravel()] = True
    import scipy.sparse as sp_

    meta = {"material": {"E": 1.0, "nu": 0.3, "plane": "3d"},
            "extra": {"width": w, "height": h, "depth": d, "dim": 3, "n_holes": 0, "holes": []}}
    return InstanceArchive(nodes=X, elements=tets, K=sp_.eye(3 * X.shape[0], format="csr"),
                           F=F.reshape(4, -1), dirichlet_mask=dmask, meta=meta)


# ------------------------------------------------------------- default path --

@pytest.mark.parametrize("make", [_synthetic, _tet])
def test_default_features_are_the_wp8_features_bit_for_bit(make):
    for seed in (0, 1):
        a = make(seed)
        sd = spatial_dim_of(a)
        for ls in (True, False):
            for geo in (True, False):
                spec = FeatureSpec(load_summary=ls, geometry=geo, spatial_dim=sd)
                got = build_features_battery(a, spec)
                ref = np.stack([_wp8_build_features(a, j, spec) for j in range(a.n_loads)])
                assert got.dtype == ref.dtype and got.tobytes() == ref.tobytes()


def test_default_spec_and_config_are_unchanged():
    from fejepa.models.fejepa import FEJEPAConfig

    assert FeatureSpec().to_dict() == {"load_summary": True, "geometry": True, "spatial_dim": 2}
    assert FeatureSpec.from_dict({}).load_density is False
    on = FeatureSpec.from_dict({"load_density": True})
    assert on.load_density and on.to_dict()["load_density"] is True
    assert FeatureSpec.from_dict(on.to_dict()) == on
    assert on.dim == FeatureSpec().dim                      # same encoder input width
    cfg = FEJEPAConfig()
    assert cfg.decode_scale == "max"
    assert FEJEPAConfig.from_dict(dict(cfg.to_dict(), decode_scale="l1")).decode_scale == "l1"


def test_default_decode_scale_is_the_largest_nodal_force():
    torch = pytest.importorskip("torch")
    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa

    a = _synthetic(3)
    torch.manual_seed(0)
    m = build_fejepa(FEJEPAConfig(dim=16, depth=1, heads=2))
    pack = m.prepare_instance(a, "cpu")
    assert float(pack["fscale"]) == float(np.float32(battery_fscale(a.F)))
    ref = torch.as_tensor(np.stack([_wp8_build_features(a, j, FeatureSpec())
                                    for j in range(a.n_loads)]))
    assert torch.equal(pack["feats"], ref)


def test_invalid_settings_are_refused():
    from fejepa.models.bottleneck import BottleneckConfig
    from fejepa.models.fejepa import FEJEPAConfig
    from fejepa.models.gnn import build_mesh_gnn

    with pytest.raises(ValueError, match="decode_scale"):
        FEJEPAConfig(decode_scale="mean")
    with pytest.raises(ValueError, match="scale_decode"):
        FEJEPAConfig(decode_scale="l1", scale_decode=False)
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="fejepa' only"):
        build_mesh_gnn(features=FeatureSpec(load_density=True))
    with pytest.raises(ValueError, match="fejepa' only"):
        BottleneckConfig.from_dict({"features": {"load_density": True}})


# ------------------------------------------------------------ geometry -------

def test_simplex_geometry_2d_and_3d():
    a = _kuhn(1)
    vol, bnd, fm = simplex_geometry(a.nodes, a.elements)
    w, h, d = (a.meta["extra"][k] for k in ("width", "height", "depth"))
    assert np.isclose(vol.sum(), w * h * d) and np.isclose(fm.sum(), 2 * (w * h + w * d + h * d))
    sq = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    vol, bnd, fm = simplex_geometry(sq, tri)
    assert np.allclose(vol, [1.0, 1.0]) and len(bnd) == 4
    assert np.isclose(fm.sum(), 6.0)                       # perimeter; the diagonal is inside
    tet = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vol, bnd, fm = simplex_geometry(tet, np.array([[0, 1, 2, 3]]))
    assert np.isclose(vol[0], 1 / 6) and len(bnd) == 4
    assert np.isclose(fm.sum(), 1.5 + np.sqrt(3) / 2)


# ------------------------------------------------------------ S: densities ---

@pytest.mark.parametrize("make", [_synthetic, _kuhn])
def test_densities_are_exact_for_uniform_tractions_and_body_forces(make):
    a = make(4)
    sd = spatial_dim_of(a)
    dd = load_densities(a)
    assert dd.kind == ("traction", "traction", "traction", "body")
    for li in range(a.n_loads):
        loaded = dd.measure[li] > 0
        vals = dd.density[li][loaded]
        assert np.allclose(vals, vals[0], rtol=1e-9, atol=0)    # one uniform density
        # the consistent nodal forces are density x the lumped measure
        F = a.F[li].reshape(-1, sd)
        assert np.allclose(F[loaded], vals[0] * dd.measure[li][loaded, None], rtol=1e-9)
        assert np.all(np.abs(F[~loaded]) == 0)
    assert np.isclose(dd.loaded_fraction[3], 1.0)              # gravity loads the body
    assert np.isclose(dd.dscale, np.abs(dd.density).max(), rtol=1e-9)


def test_s_features_do_not_change_with_the_mesh():
    """Same plate and loads on structured meshes of 6x4 to 24x16 cells: the S
    load columns, the S summary and the l1 decode scale agree; wp8's change."""
    spec = FeatureSpec(load_density=True)
    archs = [_synthetic(7, nx, ny) for nx, ny in ((6, 4), (12, 8), (24, 16))]
    l1 = [battery_l1(a.F) for a in archs]
    assert np.allclose(l1, l1[0], rtol=1e-9)
    fs = [battery_fscale(a.F) for a in archs]
    assert fs[0] / fs[2] > 3.5                                  # wp8's scale shrinks with h
    dds = [load_densities(a) for a in archs]
    for li in range(4):
        s = [density_load_summary(a.F, li, dd, 2) for a, dd in zip(archs, dds)]
        assert np.allclose(s, s[0], rtol=1e-9, atol=1e-12)
        old = [load_summary(a.F, li, 2) for a in archs]
        assert not np.allclose(old, old[0], rtol=1e-2)
        uniq = [np.unique(np.round(dd.density[li][dd.measure[li] > 0], 12), axis=0)
                for dd in dds]
        assert all(len(u) == 1 and np.allclose(u, uniq[0]) for u in uniq)
    feats = [build_features_battery(a, spec) for a in archs]
    for li in range(4):                                         # summary columns broadcast
        cols = slice(6, 6 + load_summary_dim(2))
        assert np.allclose([f[li, 0, cols] for f in feats], feats[0][li, 0, cols], atol=1e-6)


def test_s_features_on_gmsh_remeshes_with_holes():
    pytest.importorskip("gmsh")
    pytest.importorskip("skfem")
    from fejepa.fe.generator import build_instance, sample_params

    rng = np.random.default_rng(12)
    p = sample_params(rng)
    while len(p["holes"]) < 2:
        p = sample_params(rng)
    archs = [build_instance(dict(p, target_h=h)) for h in (0.12, 0.05)]
    assert archs[1].n_nodes > 3 * archs[0].n_nodes
    dds = [load_densities(a) for a in archs]
    ts = p["traction_scales"]
    want = [(0.0, -ts[0]), (ts[1], 0.0), (ts[2], 0.0), (0.0, -ts[3])]
    for dd in dds:
        for li, w in enumerate(want):
            vals = dd.density[li][dd.measure[li] > 0]
            assert np.allclose(vals, w, rtol=1e-9, atol=1e-15)
    l1 = [battery_l1(a.F) for a in archs]
    assert abs(l1[0] / l1[1] - 1) < 1e-2                        # gravity: polygonal holes
    s0 = [density_load_summary(archs[0].F, li, dds[0], 2) for li in range(4)]
    s1 = [density_load_summary(archs[1].F, li, dds[1], 2) for li in range(4)]
    assert np.allclose(s0, s1, rtol=2e-2, atol=1e-3)


# --------------------------------------------------------- S: equivariance ---

def test_s_prediction_scales_with_the_loads():
    torch = pytest.importorskip("torch")
    from copy import deepcopy

    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa

    cfg = FEJEPAConfig(dim=16, depth=2, heads=2, decode_scale="l1",
                       features=FeatureSpec(load_density=True))
    torch.manual_seed(0)
    m = build_fejepa(cfg).eval()
    a = _synthetic(5)
    outs = []
    for s in (1.0, 3.7, 0.01):
        b = deepcopy(a)
        b.F = a.F * s
        pack = m.prepare_instance(b, "cpu")
        if s == 1.0:
            base_feats, base_scale = pack["feats"], float(pack["fscale"])
        else:
            assert torch.allclose(pack["feats"], base_feats, atol=1e-6)
            assert np.isclose(float(pack["fscale"]), s * base_scale, rtol=1e-6)
        with torch.no_grad():
            outs.append((s, m.forward_instance(pack).numpy().astype(np.float64)))
    u1 = outs[0][1]
    for s, u in outs[1:]:
        assert np.allclose(u, s * u1, rtol=1e-4, atol=1e-7 * s * np.abs(u1).max())


def test_s_trains_through_the_unit_path():
    """The AR unit trains an S model end to end (same state-dict layout as wp8's)."""
    torch = pytest.importorskip("torch")
    from fejepa.experiments.parallel import _build_model, pretrain_unit
    from fejepa.experiments.protocol import load_split
    from fejepa.fe.synthetic import generate_synthetic_dataset

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        d = generate_synthetic_dataset(Path(td) / "d", n=5, seed=3)
        files = [str(f) for f in load_split(d, 1, 1).pool_files[:3]]
        model = {"dim": 16, "depth": 1, "heads": 2, "decode_scale": "l1",
                 "features": {"load_summary": True, "geometry": True, "load_density": True}}
        out = pretrain_unit({"kind": "fejepa", "model": model, "seed": 0, "tf32": False,
                             "files": files, "loss": "ar",
                             "pre": {"epochs": 2, "lr": 1e-3, "device": "cpu", "log_every": -1},
                             "state_path": str(Path(td) / "s.pt"), "quiet": True})
        sd = torch.load(out["state_path"], map_location="cpu", weights_only=True)
        ref = _build_model({"kind": "fejepa", "model": {"dim": 16, "depth": 1, "heads": 2},
                            "seed": 0}).state_dict()
        assert list(sd) == list(ref) and all(sd[k].shape == ref[k].shape for k in sd)
        assert all(torch.isfinite(v).all() for v in sd.values())


# ------------------------------------------- Stage 0b review: S at model level --

S_MODEL = {"dim": 16, "depth": 1, "heads": 2, "decode_scale": "l1", "decode_scale_factor": 1 / 64,
           "features": {"load_summary": True, "geometry": True, "load_density": True}}


def test_s_model_prepares_the_s_scale_and_features():
    """Both halves of S reach the model: the pack's scale is factor x sum|F|
    and its features are the load-density features (a no-op S would fail)."""
    torch = pytest.importorskip("torch")
    from fejepa.experiments.parallel import _build_model

    a = _synthetic(6)
    m = _build_model({"kind": "fejepa", "model": S_MODEL, "seed": 0})
    pack = m.prepare_instance(a, "cpu")
    assert float(pack["fscale"]) == float(np.float32(battery_l1(a.F) / 64))
    assert float(pack["fscale"]) != float(np.float32(battery_fscale(a.F)))
    want = torch.as_tensor(build_features_battery(a, FeatureSpec(load_density=True)))
    assert torch.equal(pack["feats"], want)
    assert not torch.equal(pack["feats"], torch.as_tensor(build_features_battery(a, FeatureSpec())))


def test_s_ar_loss_scores_exactly_the_inference_field():
    """D14 under S: the anchor inside the AR loss receives bitwise the field
    forward_instance returns (free mask AND the S scale)."""
    torch = pytest.importorskip("torch")
    from fejepa.anchor.energy import AnchorCache
    from fejepa.experiments.parallel import _build_model
    from fejepa.train.losses import AR_CONFIG, compute_loss

    a = _synthetic(7)
    m = _build_model({"kind": "fejepa", "model": S_MODEL, "seed": 0})
    m.train()
    pack = m.prepare_instance(a, "cpu")
    anchor = AnchorCache(device="cpu").get(a)
    seen, real = {}, anchor.energies

    def spy(u):
        seen["u"] = u.detach().clone()
        return real(u)
    anchor.energies = spy
    compute_loss(m, pack, anchor, None, None, np.random.default_rng(0), AR_CONFIG)
    assert torch.equal(seen["u"], m.forward_instance(pack).detach())


def test_decode_scale_factor_is_validated():
    from fejepa.models.fejepa import FEJEPAConfig

    with pytest.raises(ValueError, match="'l1' only"):
        FEJEPAConfig(decode_scale_factor=0.5)
    with pytest.raises(ValueError, match="> 0"):
        FEJEPAConfig(decode_scale="l1", decode_scale_factor=0.0)
    assert FEJEPAConfig(decode_scale="l1", decode_scale_factor=0.25).decode_scale_factor == 0.25


def test_point_load_falls_back_to_the_nodes_own_measure():
    """A load on one boundary node (no facet with all vertices loaded) takes
    the node's share of its adjacent boundary facets."""
    from copy import deepcopy

    a = deepcopy(_synthetic(8))
    sd = 2
    vol, bnd, fm = simplex_geometry(a.nodes, a.elements)
    node = int(bnd[0, 0])
    F = np.zeros_like(a.F)
    F[0, sd * node + 1] = -0.3
    a.F = F
    dd = load_densities(a)
    share = sum(m / 2 for f, m in zip(bnd, fm) if node in f)
    assert dd.kind[0] == "traction" and np.isclose(dd.measure[0][node], share)
    assert np.isclose(dd.density[0][node, 1], -0.3 / share)
    assert dd.kind[1:] == ("none", "none", "none")
