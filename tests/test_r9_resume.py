"""R9 hardening: atomic writes, corrupt-artefact fallbacks, and epoch-boundary
checkpoint/resume that reproduces the uninterrupted trajectory bitwise (CPU)."""

import pytest

torch = pytest.importorskip("torch")

from fejepa.data.archive import load_instance
from fejepa.experiments.parallel import (_build_model, cached_supervised_unit,
                                         pretrain_unit)
from fejepa.experiments.protocol import load_split
from fejepa.experiments.runner import _label_files
from fejepa.fe.solve import SolveLedger
from fejepa.fe.synthetic import generate_synthetic_dataset
from fejepa.train.losses import AR_CONFIG
from fejepa.train.pretrain import PretrainConfig, pretrain
from fejepa.train.supervised import SupervisedConfig, train_supervised

MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 16, "mgn_depth": 2,
         "features": {"load_summary": True, "geometry": True}}


def _corpus(tmp_path, seed=21):
    d = generate_synthetic_dataset(tmp_path / f"c{seed}", n=6, seed=seed)
    sp = load_split(d, n_val=2, seed=1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "v")
    _label_files(sp.pool_files[:3], led, "p")
    return d, sp


def _params(model):
    return [p.detach().clone() for p in model.parameters()]


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))


def test_atomic_archive_write_leaves_no_temp_files(tmp_path):
    d, _ = _corpus(tmp_path)
    assert not list(d.glob("*.tmp"))
    assert any(d.glob("*.npz"))


def test_pretrain_resume_is_bitwise_exact(tmp_path):
    _, sp = _corpus(tmp_path)
    tr = [load_instance(f) for f in sp.pool_files[:3]]
    base = dict(epochs=3, lr=1e-3, seed=0, device="cpu", loss=AR_CONFIG,
                log_every=-1)
    m_ref = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m_ref, tr, PretrainConfig(**base))
    ck = str(tmp_path / "pre.ckpt")
    m_a = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m_a, tr, PretrainConfig(**base, ckpt_path=ck, stop_after_epoch=1))
    assert (tmp_path / "pre.ckpt").exists()
    m_b = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})   # fresh
    h = pretrain(m_b, tr, PretrainConfig(**base, ckpt_path=ck, resume=True))
    assert len(h["loss"]) == 3                      # history restored + continued
    assert _same(_params(m_ref), _params(m_b))


def test_supervised_balanced_resume_is_bitwise_exact(tmp_path):
    _, sp = _corpus(tmp_path)
    tr = [load_instance(f) for f in sp.pool_files[:3]]
    val = [load_instance(f) for f in sp.val_files]
    base = dict(epochs=3, lr=1e-3, seed=0, device="cpu", anchor_mode="balanced",
                balance_ratio=1.0, log_every=-1)
    m_ref = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    r_ref = train_supervised(m_ref, tr, val, SupervisedConfig(**base))
    ck = str(tmp_path / "sup.ckpt")
    m_a = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    train_supervised(m_a, tr, val, SupervisedConfig(**base, ckpt_path=ck,
                                                    stop_after_epoch=1))
    m_b = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    r_b = train_supervised(m_b, tr, val, SupervisedConfig(**base, ckpt_path=ck,
                                                          resume=True))
    assert _same(_params(m_ref), _params(m_b))
    assert r_b["balance_scale_mean"] == pytest.approx(r_ref["balance_scale_mean"])


def test_corrupt_state_falls_back_to_retraining(tmp_path):
    _, sp = _corpus(tmp_path)
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
               "files": [str(f) for f in sp.pool_files[:2]],
               "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
               "state_path": str(tmp_path / "st" / "ar.pt"), "quiet": True}
    pretrain_unit(payload)
    (tmp_path / "st" / "ar.pt").write_bytes(b"garbage-from-a-power-cut")
    out = pretrain_unit({**payload, "reuse_existing": True})
    assert out["reused_state"] is False
    sd = torch.load(str(tmp_path / "st" / "ar.pt"), map_location="cpu",
                    weights_only=True)
    assert isinstance(sd, dict) and len(sd) > 0
    assert not (tmp_path / "st" / "ar.ckpt").exists()          # cleaned up


def test_corrupt_cache_falls_back_to_retraining(tmp_path):
    _, sp = _corpus(tmp_path)
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
               "train_files": [str(f) for f in sp.pool_files[:2]],
               "val_files": [str(f) for f in sp.val_files],
               "sup": {"epochs": 1, "lr": 1e-3, "device": "cpu",
                       "anchor_mode": "none", "log_every": -1},
               "tag": "labels b2 s0", "cache_dir": str(tmp_path / "cache"),
               "reuse_existing": True}
    cached_supervised_unit(payload)
    cp = tmp_path / "cache" / "labels_b2_s0.pkl"
    cp.write_bytes(b"\x80\x04garbage")
    res = cached_supervised_unit(payload)
    assert not res.get("from_cache") and "val" in res
    assert not list((tmp_path / "cache").glob("*.tmp"))
    assert not (tmp_path / "cache" / "labels_b2_s0.ckpt").exists()


def test_unit_level_resume_from_checkpoint(tmp_path):
    """A unit interrupted mid-way (ckpt present, no final state) resumes under
    --reuse-states and lands on the uninterrupted unit's parameters."""
    _, sp = _corpus(tmp_path)
    files = [str(f) for f in sp.pool_files[:3]]
    pre = {"epochs": 3, "lr": 1e-3, "device": "cpu", "log_every": -1}
    ref = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
           "files": files, "pre": pre, "state_path": str(tmp_path / "ref" / "ar.pt"),
           "quiet": True}
    pretrain_unit(ref)
    sd_ref = torch.load(ref["state_path"], map_location="cpu", weights_only=True)
    # interrupted attempt: only the epoch-1 checkpoint survives
    sp_path = tmp_path / "run" / "ar.pt"
    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m, [load_instance(f) for f in files],
             PretrainConfig(**pre, loss=AR_CONFIG, seed=0,
                            ckpt_path=str(sp_path.with_suffix(".ckpt")),
                            stop_after_epoch=1))
    assert sp_path.with_suffix(".ckpt").exists() and not sp_path.exists()
    out = pretrain_unit({**ref, "state_path": str(sp_path), "reuse_existing": True})
    assert out["reused_state"] is False                        # trained (resumed)
    sd_new = torch.load(str(sp_path), map_location="cpu", weights_only=True)
    assert all(torch.equal(sd_ref[k], sd_new[k]) for k in sd_ref)
    assert not sp_path.with_suffix(".ckpt").exists()           # cleaned up
