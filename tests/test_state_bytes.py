"""wp8 Stage 1.36: the state an AR unit saves is the state dict itself when
there is no SIGReg head -- byte-identical to the wp7-3d main line's file for
identical tensors (a plain dict dropped `_metadata`, so equal training gave a
different file SHA-256) -- and with a head, the head is gone from both the
tensors and the metadata, and the state loads strictly into a fresh model."""

import io

import pytest

torch = pytest.importorskip("torch")


def _model(seed=0):
    from fejepa.experiments.parallel import _build_model

    return _build_model({"kind": "fejepa", "seed": seed, "model": {
        "dim": 16, "depth": 1, "heads": 2,
        "features": {"load_summary": True, "geometry": True}}})


def _bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def test_without_a_head_the_saved_state_is_the_state_dict_byte_for_byte():
    from fejepa.experiments.parallel import deliverable_state

    sd = _model().state_dict()
    out = deliverable_state(sd)
    assert type(out) is type(sd) and out._metadata == sd._metadata
    assert _bytes(out) == _bytes(sd)


def test_with_a_head_the_head_is_dropped_and_the_state_loads_strictly():
    from fejepa.experiments.parallel import deliverable_state
    from fejepa.train.losses import build_sigreg_head

    m = _model()
    m.sigreg_head = build_sigreg_head(16, 6)
    sd = m.state_dict()
    assert any(k.startswith("sigreg_head.") for k in sd)
    out = deliverable_state(sd)
    assert not any(k.startswith("sigreg_head.") for k in out)
    assert not any(k == "sigreg_head" or k.startswith("sigreg_head.") for k in out._metadata)
    fresh = _model(seed=1)
    fresh.load_state_dict(out, strict=True)
    assert all(torch.equal(fresh.state_dict()[k], out[k]) for k in out)
