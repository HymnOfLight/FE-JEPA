"""wp8-lejepa Stage 1.0: the E1 arm (AR + validated SIGReg on a projector head)
and the E2 bottleneck prototype run through the production paths; legacy
paths are untouched (bitwise regression against the tag is run separately)."""

import pytest

torch = pytest.importorskip("torch")

from fejepa.anchor.energy import AnchorCache
from fejepa.data.archive import load_instance
from fejepa.experiments.parallel import _build_model, pretrain_unit, supervised_unit
from fejepa.experiments.protocol import load_split
from fejepa.experiments.runner import _label_files
from fejepa.fe.solve import SolveLedger
from fejepa.fe.synthetic import generate_synthetic_dataset
from fejepa.metrics import evaluate_model, torch_predictor
from fejepa.models.bottleneck import farthest_point_sampling, nearest_seed
from fejepa.train.losses import ar_sigreg_config
from fejepa.train.pretrain import PretrainConfig, pretrain

MODEL = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}
BOTTLE = {"dim": 16, "depth": 1, "heads": 2, "n_tokens": 8,
          "features": {"load_summary": True, "geometry": True}}


def _corpus(tmp_path, seed=31):
    d = generate_synthetic_dataset(tmp_path / f"c{seed}", n=6, seed=seed)
    sp = load_split(d, n_val=2, seed=1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "v")
    _label_files(sp.pool_files[:3], led, "p")
    return sp


# ------------------------------- E1 -----------------------------------------
def test_e1_ar_sigreg_head_trains_and_state_is_strict_loadable(tmp_path):
    sp = _corpus(tmp_path)
    tr = [load_instance(f) for f in sp.pool_files[:3]]
    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    h = pretrain(m, tr, PretrainConfig(epochs=2, lr=1e-3, seed=0, device="cpu",
                                       loss=ar_sigreg_config(0.1, head=True, n_proj=32),
                                       log_every=-1))
    assert hasattr(m, "sigreg_head") and torch.isfinite(torch.tensor(h["loss"])).all()
    # through the production unit: the saved state has NO head and loads strictly
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
               "files": [str(f) for f in sp.pool_files[:3]],
               "loss": {"reg_mode": "sigreg_ep_head", "lambda_reg": 0.1, "sigreg_n_proj": 32},
               "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
               "state_path": str(tmp_path / "st" / "ar_sig.pt"),
               "eval_val_files": [str(f) for f in sp.val_files], "quiet": True}
    out = pretrain_unit(payload)
    sd = torch.load(out["state_path"], map_location="cpu", weights_only=True)
    assert not any(k.startswith("sigreg_head.") for k in sd)
    fresh = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    fresh.load_state_dict(sd, strict=True)
    assert "val" in out and torch.isfinite(torch.tensor(out["val"]["disp_rel_l2"]))


def test_e1_raw_mode_needs_no_head(tmp_path):
    sp = _corpus(tmp_path)
    tr = [load_instance(f) for f in sp.pool_files[:2]]
    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m, tr, PretrainConfig(epochs=1, lr=1e-3, seed=0, device="cpu",
                                   loss=ar_sigreg_config(0.1, head=False, n_proj=16),
                                   log_every=-1))
    assert not hasattr(m, "sigreg_head")


# ------------------------------- E2 -----------------------------------------
def test_fps_and_assignment_are_deterministic_and_cover():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.random((500, 3))
    s1, s2 = farthest_point_sampling(x, 32), farthest_point_sampling(x, 32)
    assert np.array_equal(s1, s2) and len(set(s1.tolist())) == 32
    a = nearest_seed(x, x[s1])
    assert a.shape == (500,) and a.min() >= 0 and a.max() < 32
    assert np.array_equal(a[s1], np.arange(32))          # seeds map to themselves
    assert farthest_point_sampling(x[:5], 32).shape == (5,)   # n < M handled


def test_bottleneck_matches_pack_contract_and_feeds_the_anchor(tmp_path):
    sp = _corpus(tmp_path)
    arch = load_instance(sp.pool_files[0])
    m = _build_model({"kind": "bottleneck", "model": BOTTLE, "seed": 0})
    pack = m.prepare_instance(arch, "cpu")
    L, N = pack["feats"].shape[0], pack["feats"].shape[1]
    z = m.encode(pack["feats"], pack)
    assert z.shape == (L, pack["n_tok"], 16) and pack["n_tok"] <= min(8, N)
    u = m.forward_instance(pack)
    assert u.shape == (L, pack["free"].numel())
    anc = AnchorCache(device="cpu").get(arch)
    e = anc.energies(u)                                   # the SAME exact anchor
    assert e.shape == (L,) and torch.isfinite(e).all()
    e.mean().backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None for g in grads) and all(torch.isfinite(g).all() for g in grads)


def test_bottleneck_trains_through_ar_and_supervised_units(tmp_path):
    sp = _corpus(tmp_path)
    base = {"kind": "bottleneck", "model": BOTTLE, "seed": 0, "tf32": False}
    out = pretrain_unit({**base, "files": [str(f) for f in sp.pool_files[:3]],
                         "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
                         "state_path": str(tmp_path / "bt" / "ar.pt"),
                         "eval_val_files": [str(f) for f in sp.val_files], "quiet": True})
    assert torch.isfinite(torch.tensor(out["val"]["energy_gap_rel"]))
    res = supervised_unit({**base, "train_files": [str(f) for f in sp.pool_files[:2]],
                           "val_files": [str(f) for f in sp.val_files],
                           "sup": {"epochs": 1, "lr": 1e-3, "device": "cpu",
                                   "anchor_mode": "balanced", "balance_ratio": 1.0,
                                   "log_every": -1}, "tag": "bt b2 s0"})
    assert torch.isfinite(torch.tensor(res["val"]["disp_rel_l2"]))
    m = _build_model(base)
    m.load_state_dict(torch.load(out["state_path"], map_location="cpu", weights_only=True),
                      strict=True)
    ev = evaluate_model(torch_predictor(m, "cpu"), [load_instance(f) for f in sp.val_files])
    assert torch.isfinite(torch.tensor(ev["disp_rel_l2"]))
