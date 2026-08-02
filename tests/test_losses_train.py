"""Objectives + one-epoch training smokes (torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.anchor.energy import AnchorCache
from fejepa.models.features import FeatureSpec
from fejepa.models.fejepa import FEJEPAConfig, build_fejepa, mesh_adjacency
from fejepa.models.regularizers import PooledBuffer, sigreg_pooled, vicreg_pooled
from fejepa.train.losses import AR_CONFIG, JEPA_CONFIG, compute_loss, with_inv
from fejepa.train.pretrain import PretrainConfig, amortized_ritz, pretrain
from fejepa.train.supervised import SupervisedConfig, train_supervised


def _model():
    return build_fejepa(FEJEPAConfig(dim=16, depth=1, heads=2,
                                     features=FeatureSpec(True, True)))


def test_compute_loss_parts(arch):
    m = _model()
    pack = m.prepare_instance(arch, "cpu")
    anchor = AnchorCache().get(arch)
    adj = mesh_adjacency(arch.elements, arch.n_nodes)
    rng = np.random.default_rng(0)
    loss, parts = compute_loss(m, pack, anchor, adj, PooledBuffer(), rng, AR_CONFIG)
    assert set(parts) == {"phys"} and torch.isfinite(loss)
    loss, parts = compute_loss(m, pack, anchor, adj, PooledBuffer(), rng, JEPA_CONFIG)
    assert {"phys", "pred", "reg"} <= set(parts)
    loss, parts = compute_loss(m, pack, anchor, adj, PooledBuffer(), rng,
                               with_inv(AR_CONFIG, True), twin_pack=pack)
    assert "inv" in parts and parts["inv"] < 1e-10        # identical views
    loss, parts = compute_loss(m, pack, anchor, None, PooledBuffer(), rng,
                               AR_CONFIG)                 # AR never touches adjacency
    assert set(parts) == {"phys"} and torch.isfinite(loss)


def test_pooled_regularizers_finite():
    buf = PooledBuffer(size=8)
    rows = torch.randn(4, 16)
    for fn in (sigreg_pooled, vicreg_pooled):
        v = fn(rows, buf)
        assert torch.isfinite(v)
    buf.push(rows)
    assert len(buf._items) == 4


def test_supervised_modes_and_finetune(instances):
    train, val = instances[:4], instances[4:6]
    pre = _model()
    amortized_ritz(pre, train, PretrainConfig(epochs=1, seed=0))
    state = pre.state_dict()
    for mode in ("none", "fixed", "balanced"):
        res = train_supervised(_model(), train, val,
                               SupervisedConfig(epochs=1, seed=0, anchor_mode=mode),
                               pretrained_state=state)
        assert res["pretrained_tensors_loaded"] > 0
        assert np.isfinite(res["val"]["disp_rel_l2"])
        if mode == "balanced":
            assert 0.0 <= res["balance_scale_mean"] <= 1.0


def test_empty_training_sets_fail_loud(instances):
    from fejepa.train.pretrain import PretrainConfig, pretrain
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    with pytest.raises(ValueError):
        pretrain(_model(), [], PretrainConfig(epochs=1, seed=0))
    with pytest.raises(ValueError):
        train_supervised(_model(), [], instances[:1], SupervisedConfig(epochs=1))


def test_pretrain_jepa_and_pairs(instances):
    m = _model()
    pretrain(m, instances[:3], PretrainConfig(epochs=1, seed=0, loss=JEPA_CONFIG))
    pairs = [(instances[0], instances[1])]
    pretrain(_model(), None,
             PretrainConfig(epochs=1, seed=0, loss=with_inv(AR_CONFIG, True)),
             pairs=pairs)
