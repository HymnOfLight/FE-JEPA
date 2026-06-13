import torch

from fejepa.models.sigreg import sigreg_loss


def test_gaussian_lower_than_collapsed():
    torch.manual_seed(0)
    z_gauss = torch.randn(256, 32)
    z_collapsed = torch.randn(256, 32) * 0.01 + 3.0  # near-constant -> collapsed
    s_g = sigreg_loss(z_gauss, n_proj=48).item()
    s_c = sigreg_loss(z_collapsed, n_proj=48).item()
    assert s_c > s_g
    assert s_g >= 0.0


def test_differentiable():
    z = torch.randn(128, 16, requires_grad=True)
    loss = sigreg_loss(z, n_proj=16)
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
