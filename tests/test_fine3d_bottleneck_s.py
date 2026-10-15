"""fine3d (PREREG_FINE3D, R2a): wp9's S on the token bottleneck. Off (the
default) the bottleneck's pack -- features and decode scale -- is E2's bit for
bit; on, both halves of S reach the model, the inputs do not change with the
mesh of one geometry, the prediction scales with the loads, the AR loss
scores exactly the inference field, and the runner accepts S for the
bottleneck and still refuses it for the MGN baseline."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fejepa.models.features import (GEOMETRY_DIM, FeatureSpec, battery_fscale, battery_l1,
                                    build_features_battery, check_decode_scale,
                                    decode_scale_value, density_load_summary,
                                    geometry_descriptor, load_densities, load_summary,
                                    load_summary_dim, normalized_coords, spatial_dim_of)

ROOT = Path(__file__).resolve().parents[1]
BOTTLE = {"kind": "bottleneck", "dim": 16, "depth": 1, "heads": 2, "n_tokens": 8,
          "decode_k": 3, "features": {"load_summary": True, "geometry": True, "spatial_dim": 3}}
S = {"decode_scale": "l1", "decode_scale_factor": 2.0 ** -10}


def _s_model_cfg():
    return dict(BOTTLE, **S, features=dict(BOTTLE["features"], load_density=True))


def _wp8_build_features(arch, load_idx, spec):
    """wp8-lejepa's build_features (5f8e2df), verbatim (as tests/test_w9_s_switch.py
    keeps it): the reference the default path must reproduce bit for bit."""
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


def _gmsh3d(g=0, lc=0.30):
    """A small instance of the 3D training family's geometry (box, cavities)."""
    pytest.importorskip("gmsh")
    from fejepa.fe.gmsh3d import gmsh3d_instance

    return gmsh3d_instance(np.random.default_rng(97000 + g), lc=lc, labelled=False)


def _model(cfg, seed=0):
    pytest.importorskip("torch")
    from fejepa.experiments.parallel import _build_model

    return _build_model({"kind": "bottleneck", "model": cfg, "seed": seed})


# ----------------------------------------------------------- configuration --

def test_decode_scale_helpers_are_the_wp9_rules():
    from fejepa.models.bottleneck import BottleneckConfig

    F = np.array([[0.0, 2.0, -3.0], [1.0, 0.0, 0.5]])
    assert decode_scale_value(F) == battery_fscale(F)
    assert decode_scale_value(F, "l1", 0.25) == 0.25 * battery_l1(F)
    for bad, msg in ((("mean", 1.0, True), "'max' or 'l1'"), (("l1", 1.0, False), "scale_decode"),
                     (("l1", 0.0, True), "> 0"), (("max", 0.5, True), "'l1' only")):
        with pytest.raises(ValueError, match=msg):
            check_decode_scale(*bad)
        with pytest.raises(ValueError, match=msg):
            BottleneckConfig(decode_scale=bad[0], decode_scale_factor=bad[1],
                             scale_decode=bad[2])
    c = BottleneckConfig.from_dict(_s_model_cfg())
    assert (c.decode_scale, c.decode_scale_factor, c.features.load_density) == \
        ("l1", 2.0 ** -10, True)
    d = BottleneckConfig.from_dict(BOTTLE)
    assert (d.decode_scale, d.decode_scale_factor, d.features.load_density) == ("max", 1.0, False)


# ------------------------------------------------------------- default path --

def test_default_pack_is_e2_bit_for_bit():
    torch = pytest.importorskip("torch")
    a = _gmsh3d(1)
    m = _model(BOTTLE)
    pack = m.prepare_instance(a, "cpu")
    assert float(pack["fscale"]) == float(np.float32(battery_fscale(a.F)))
    spec = FeatureSpec(spatial_dim=3)
    ref = torch.as_tensor(np.stack([_wp8_build_features(a, j, spec) for j in range(a.n_loads)]))
    assert torch.equal(pack["feats"], ref)


# ------------------------------------------------------------------------ S --

def test_s_pack_carries_both_halves_of_s():
    torch = pytest.importorskip("torch")
    a = _gmsh3d(2)
    m = _model(_s_model_cfg())
    pack = m.prepare_instance(a, "cpu")
    assert float(pack["fscale"]) == float(np.float32(2.0 ** -10 * battery_l1(a.F)))
    assert float(pack["fscale"]) != float(np.float32(battery_fscale(a.F)))
    want = build_features_battery(a, FeatureSpec(spatial_dim=3, load_density=True))
    assert torch.equal(pack["feats"], torch.as_tensor(want))
    assert not torch.equal(pack["feats"],
                           torch.as_tensor(build_features_battery(a, FeatureSpec(spatial_dim=3))))
    # the token geometry does not depend on S
    ref = _model(BOTTLE).prepare_instance(a, "cpu")
    for k in ("tok_idx", "nbr_idx", "nbr_w", "nbr_rel", "seed_xyz"):
        assert torch.equal(pack[k], ref[k])


def test_s_decode_applies_the_scale_once():
    """The decoded field is the decoder's raw output, masked, times the pack's
    scale (2^-10 x sum|F|) once (hooked at the decoder)."""
    torch = pytest.importorskip("torch")
    a = _gmsh3d(6)
    m = _model(_s_model_cfg()).eval()
    pack = m.prepare_instance(a, "cpu")
    raw = {}
    hook = m.dec.register_forward_hook(lambda mod, inp, out: raw.__setitem__("u", out.detach()))
    with torch.no_grad():
        u = m.forward_instance(pack)
    hook.remove()
    assert float(pack["fscale"]) == float(np.float32(2.0 ** -10 * battery_l1(a.F)))
    assert torch.equal(u, raw["u"].reshape(raw["u"].shape[0], -1) * pack["free"] * pack["fscale"])


def test_s_inputs_do_not_change_with_the_mesh():
    """One geometry and its loads meshed at two sizes: S's decode scale and
    load summary agree (to the gravity load's discretisation of the cavities),
    the densities of every load case are one value on both meshes; the
    default scale shrinks with the element size."""
    archs = [_gmsh3d(3, lc) for lc in (0.30, 0.18)]
    assert archs[1].n_nodes > 2 * archs[0].n_nodes
    l1 = [battery_l1(a.F) for a in archs]
    assert abs(l1[1] / l1[0] - 1) < 0.02
    assert battery_fscale(archs[1].F) / battery_fscale(archs[0].F) < 0.6
    dds = [load_densities(a) for a in archs]
    assert dds[0].kind == dds[1].kind == ("traction", "traction", "traction", "body")
    for li in range(4):
        vals = [np.unique(np.round(dd.density[li][dd.measure[li] > 0] / dd.dscale, 9), axis=0)
                for dd in dds]
        assert all(len(v) == 1 for v in vals) and np.allclose(vals[0], vals[1], atol=1e-9)
        s = [density_load_summary(a.F, li, dd, 3) for a, dd in zip(archs, dds, strict=True)]
        assert np.allclose(s[0], s[1], rtol=0.03, atol=1e-3)


def test_s_prediction_scales_with_the_loads():
    torch = pytest.importorskip("torch")
    m = _model(_s_model_cfg()).eval()
    a = _gmsh3d(4)
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


def test_s_ar_loss_scores_exactly_the_inference_field():
    torch = pytest.importorskip("torch")
    from fejepa.anchor.energy import AnchorCache
    from fejepa.train.losses import AR_CONFIG, compute_loss

    a = _gmsh3d(5)
    m = _model(_s_model_cfg())
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


def test_s_bottleneck_trains_through_the_unit_path(tmp_path):
    torch = pytest.importorskip("torch")
    from fejepa.data.archive import save_instance
    from fejepa.experiments.parallel import pretrain_unit

    files = []
    for g in range(3):
        f = tmp_path / f"i{g}.npz"
        save_instance(_gmsh3d(10 + g), f)
        files.append(str(f))
    out = pretrain_unit({"kind": "bottleneck", "model": _s_model_cfg(), "seed": 3, "tf32": False,
                         "files": files, "loss": "ar",
                         "pre": {"epochs": 2, "lr": 1e-3, "device": "cpu", "log_every": -1},
                         "state_path": str(tmp_path / "s.pt"), "quiet": True})
    sd = torch.load(out["state_path"], map_location="cpu", weights_only=True)
    ref = _model(BOTTLE).state_dict()
    assert list(sd) == list(ref) and all(sd[k].shape == ref[k].shape for k in sd)
    assert all(torch.isfinite(v).all() for v in sd.values())


# ------------------------------------------------------------------- runner --

def test_runner_accepts_s_for_the_bottleneck_only(tmp_path):
    pytest.importorskip("torch")
    from fejepa.experiments.runner import run_config

    cfg = json.loads((ROOT / "configs" / "r2a_s_m1024.json").read_text())
    p = tmp_path / "r2a_s_m1024.json"
    p.write_text(json.dumps(cfg))
    out = run_config(p, dry_run=True)
    assert out["model_kind"] == "bottleneck" and out["experiments_enabled"] == \
        ["e8", "p3_transfer"]
    for kind in ("mgn", "gnn"):
        bad = dict(cfg, model=dict(cfg["model"], kind=kind))
        p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match="implemented for model kinds"):
            run_config(p, dry_run=True)
