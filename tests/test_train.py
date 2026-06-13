import numpy as np
import torch

from fejepa.models.fejepa import FEJEPAConfig
from fejepa.train.pretrain import PretrainConfig, amortized_ritz
from fejepa.train.schedule import make_scheduler


def test_cosine_schedule_warms_up_and_decays():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = make_scheduler(opt, total_steps=100, schedule="cosine", warmup_frac=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[10]          # warming up
    assert lrs[10] >= lrs[-1]        # decaying after warmup
    assert lrs[-1] < 0.1             # decayed well below base lr


def test_constant_schedule_is_flat():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.5)
    sched = make_scheduler(opt, total_steps=20, schedule="constant")
    for _ in range(20):
        assert abs(opt.param_groups[0]["lr"] - 0.5) < 1e-9
        opt.step()
        sched.step()


def test_amortized_ritz_reduces_energy(small_archive):
    archs = [small_archive]
    cfg = PretrainConfig(epochs=40, lr=2e-3, model=FEJEPAConfig(dim=48, depth=2), seed=0)
    _, history = amortized_ritz(archs, cfg=cfg)
    phys = [h["phys"] for h in history]
    # energy should decrease substantially over training
    assert np.mean(phys[-5:]) < np.mean(phys[:5])
