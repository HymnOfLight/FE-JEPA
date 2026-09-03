"""D10: anchors are CPU-resident and streamed per call; values unchanged."""

import pytest

torch = pytest.importorskip("torch")

from fejepa.anchor.energy import AnchorCache, EnergyAnchor
from fejepa.data.archive import load_instance
from fejepa.experiments.protocol import load_split
from fejepa.experiments.runner import _label_files
from fejepa.fe.solve import SolveLedger
from fejepa.fe.synthetic import generate_synthetic_dataset


def _arch(tmp_path):
    d = generate_synthetic_dataset(tmp_path / "a", n=2, seed=13)
    sp = load_split(d, 1, 1)
    _label_files(sp.val_files, SolveLedger(), "v")
    return load_instance(sp.val_files[0])


def test_cuda_target_constructs_on_cpu_and_stays_cpu_resident(tmp_path):
    """Before D10, requesting device='cuda' built the sparse matrix on the GPU
    at construction (impossible here). Now construction is CPU-only and the
    resident copies stay on the CPU until a prediction on some device calls."""
    a = _arch(tmp_path)
    anc = EnergyAnchor(a.K, a.F, a.dirichlet_mask, device="cuda:0")
    assert anc.K_t.device.type == "cpu" and anc.resident == "cpu"
    assert anc.K_t.layout == torch.sparse_csr            # CUDA target keeps the CSR layout
    cache = AnchorCache(device="cuda:0")
    assert cache.resident == "cpu"


def test_streaming_and_resident_anchors_agree_bitwise(tmp_path):
    a = _arch(tmp_path)
    u = torch.randn(4, a.K.shape[0], dtype=torch.float32, requires_grad=True)
    e_stream = EnergyAnchor(a.K, a.F, a.dirichlet_mask, device="cpu", resident="cpu").energies(u)
    e_resident = EnergyAnchor(a.K, a.F, a.dirichlet_mask, device="cpu", resident="device").energies(u)
    assert torch.equal(e_stream, e_resident)
    g1 = torch.autograd.grad(e_stream.sum(), u, retain_graph=True)[0]
    g2 = torch.autograd.grad(e_resident.sum(), u)[0]
    assert torch.equal(g1, g2)
