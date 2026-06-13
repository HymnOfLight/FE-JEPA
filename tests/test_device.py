import torch

from fejepa.device import cuda_available, describe_device, resolve_device


def test_resolve_auto_matches_hardware():
    resolved = resolve_device("auto")
    expected = "cuda" if cuda_available() else "cpu"
    assert resolved == expected


def test_resolve_cpu_is_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_none_defaults_like_auto():
    assert resolve_device(None) == resolve_device("auto")


def test_cuda_request_falls_back_when_unavailable():
    if cuda_available():
        assert resolve_device("cuda") == "cuda"
    else:
        # portable configs: a cuda request degrades to CPU with a warning
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert resolve_device("cuda") == "cpu"
            assert any("CUDA is unavailable" in str(x.message) for x in w)


def test_describe_device_runs():
    assert isinstance(describe_device(resolve_device("auto")), str)


def test_train_resolves_auto(small_archive):
    # supervised trainer with device="auto" must run and land tensors on a real device
    from fejepa.models.fejepa import FEJEPAConfig
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    cfg = SupervisedConfig(epochs=1, model=FEJEPAConfig(dim=16, depth=1), device="auto")
    out = train_supervised([small_archive], [small_archive], cfg=cfg)
    assert out["val_rel_l2_disp"] is not None
