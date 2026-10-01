"""wp9 Stage 0a: D13's per-block activation checkpointing as a configuration
switch (`model.activation_checkpointing`, default true). The switch is
memory-only: label-free (AR) training lands on bitwise identical parameters
with it on or off, through the same unit code path the runner uses."""

import hashlib

import pytest

torch = pytest.importorskip("torch")

from fejepa.experiments.parallel import _build_model, pretrain_unit
from fejepa.experiments.protocol import load_split
from fejepa.fe.synthetic import generate_synthetic_dataset
from fejepa.models.fejepa import FEJEPAConfig

MODEL = {"dim": 16, "depth": 2, "heads": 2, "features": {"load_summary": True, "geometry": True}}


def test_config_default_and_roundtrip():
    assert FEJEPAConfig().activation_checkpointing is True
    assert FEJEPAConfig.from_dict({}).activation_checkpointing is True          # legacy configs
    off = FEJEPAConfig.from_dict({"activation_checkpointing": False})
    assert off.activation_checkpointing is False
    assert FEJEPAConfig.from_dict(off.to_dict()).activation_checkpointing is False


def test_switch_reaches_the_encoder():
    on = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    off = _build_model({"kind": "fejepa", "model": {**MODEL, "activation_checkpointing": False},
                        "seed": 0})
    assert on.encoder.use_checkpoint is True and off.encoder.use_checkpoint is False
    # a plain attribute: the two models have the same state-dict keys and values
    sd_on, sd_off = on.state_dict(), off.state_dict()
    assert list(sd_on) == list(sd_off)
    assert all(torch.equal(sd_on[k], sd_off[k]) for k in sd_on)


def test_ar_training_is_bitwise_identical_with_and_without(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "d", n=5, seed=3)
    files = [str(f) for f in load_split(d, 1, 1).pool_files[:3]]
    digests = []
    for flag in (True, False):
        sp = tmp_path / ("on" if flag else "off") / "ar.pt"   # same name: torch.save records it
        pretrain_unit({"kind": "fejepa", "model": {**MODEL, "activation_checkpointing": flag},
                       "seed": 0, "tf32": False, "files": files, "loss": "ar",
                       "pre": {"epochs": 3, "lr": 1e-3, "device": "cpu", "log_every": -1},
                       "state_path": str(sp), "quiet": True})
        sd = torch.load(str(sp), map_location="cpu", weights_only=True)
        digests.append({k: v.clone() for k, v in sd.items()})
        digests.append(hashlib.sha256(sp.read_bytes()).hexdigest())
    a, sha_a, b, sha_b = digests
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert sha_a == sha_b                       # the saved files are the same bytes
