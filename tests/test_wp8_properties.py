"""Mathematical properties of the branch's core pieces:
SIGReg is rotation-invariant in expectation (an isotropic sketch); the
bottleneck's output does not depend on the mesh's node numbering."""

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")

from fejepa.data.archive import load_instance
from fejepa.experiments.parallel import _build_model
from fejepa.train.sigreg import sigreg


def test_sigreg_is_rotation_invariant_in_expectation():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(2000, 16, generator=g) * torch.linspace(0.3, 2.0, 16) + 0.5   # anisotropic, shifted
    Q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g))                  # random orthogonal
    a = float(sigreg(z, n_proj=4096, generator=torch.Generator().manual_seed(1)))
    b = float(sigreg(z @ Q, n_proj=4096, generator=torch.Generator().manual_seed(2)))
    assert abs(a - b) <= 0.03 * max(abs(a), abs(b))            # same statistic, different directions


def _permute(arch, perm):
    """Node-permuted copy: nodes, elements, K (dof-major 2 per node), F, mask, U*."""
    sd = arch.nodes.shape[1]
    dof_perm = np.concatenate([[sd * p + k for k in range(sd)] for p in perm])
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    P = sp.csr_matrix((np.ones(len(dof_perm)), (np.arange(len(dof_perm)), dof_perm)),
                      shape=(len(dof_perm), len(dof_perm)))
    return dataclasses.replace(arch, nodes=arch.nodes[perm], elements=inv[arch.elements],
                               K=(P @ arch.K @ P.T).tocsr(), F=arch.F[:, dof_perm],
                               dirichlet_mask=arch.dirichlet_mask[dof_perm],
                               U_star=None if arch.U_star is None else arch.U_star[:, dof_perm])


def test_bottleneck_output_is_independent_of_node_numbering(tiny_corpus):
    sp_ = tiny_corpus(seed=53)
    arch = load_instance(sp_.pool_files[0])
    rng = np.random.default_rng(0)
    perm = rng.permutation(arch.nodes.shape[0])
    parch = _permute(arch, perm)
    m = _build_model({"kind": "bottleneck", "model": {"dim": 16, "depth": 1, "heads": 2, "n_tokens": 6,
                      "features": {"load_summary": True, "geometry": True}}, "seed": 0})
    m.eval()
    sd = arch.nodes.shape[1]
    with torch.no_grad():
        u = m.forward_instance(m.prepare_instance(arch, "cpu")).reshape(-1, arch.nodes.shape[0], sd)
        up = m.forward_instance(m.prepare_instance(parch, "cpu")).reshape(-1, arch.nodes.shape[0], sd)
    assert torch.allclose(up, u[:, perm, :], atol=1e-5, rtol=1e-4)


def test_sigreg_and_head_are_finite_on_collapsed_embeddings():
    """A collapsed (constant) embedding is exactly what the regulariser must
    push away from: value and gradient must be finite there, also through the
    BatchNorm head (zero batch variance -> eps path)."""
    from fejepa.train.losses import build_sigreg_head

    z = torch.full((500, 16), 0.7, requires_grad=True)
    loss = sigreg(z, n_proj=64, generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    head = build_sigreg_head(16, 8).train()
    z2 = torch.full((500, 16), 0.7, requires_grad=True)
    loss2 = sigreg(head(z2), n_proj=64, generator=torch.Generator().manual_seed(0))
    loss2.backward()
    assert torch.isfinite(loss2) and torch.isfinite(z2.grad).all()


def test_bottleneck_units_run_under_spawned_workers(tmp_path, tiny_corpus):
    """E-series runs use workers=1, but the unit path must survive spawn
    pickling of payloads for the new kind (models are built in the worker)."""
    from fejepa.experiments.parallel import map_units, pretrain_unit

    sp_ = tiny_corpus(seed=59)
    files = [str(f) for f in sp_.pool_files[:2]]
    payloads = [{"kind": "bottleneck", "seed": s, "tf32": False, "files": files,
                 "model": {"dim": 16, "depth": 1, "heads": 2, "n_tokens": 6,
                           "features": {"load_summary": True, "geometry": True}},
                 "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
                 "state_path": str(tmp_path / f"w{s}.pt"), "quiet": True} for s in (0, 1)]
    out = map_units(pretrain_unit, payloads, 2, "spawn smoke")
    assert len(out) == 2 and all((tmp_path / f"w{s}.pt").exists() for s in (0, 1))


def test_bottleneck_is_load_scale_blind(tiny_corpus):
    """Scaling the load battery by alpha scales the FE solution by alpha (linear
    elasticity); with scale_decode the prediction must transform exactly the
    same way -- the invariance FE-JEPA rests on, inherited by the bottleneck."""
    sp_ = tiny_corpus(seed=61)
    arch = load_instance(sp_.pool_files[0])
    alpha = 2.37
    twin = dataclasses.replace(arch, F=arch.F * alpha,
                               U_star=None if arch.U_star is None else arch.U_star * alpha)
    m = _build_model({"kind": "bottleneck", "model": {"dim": 16, "depth": 1, "heads": 2, "n_tokens": 6,
                      "scale_decode": True, "features": {"load_summary": True, "geometry": True}}, "seed": 0})
    m.eval()
    with torch.no_grad():
        u = m.forward_instance(m.prepare_instance(arch, "cpu"))
        ut = m.forward_instance(m.prepare_instance(twin, "cpu"))
    assert torch.allclose(ut, alpha * u, rtol=1e-5, atol=1e-7)


def test_sigreg_gradient_matches_finite_differences():
    z = torch.randn(40, 5, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda t: sigreg(t, n_proj=16, generator=torch.Generator().manual_seed(3), proj_chunk=4),
        (z,), eps=1e-3, atol=1e-3, rtol=1e-2, fast_mode=True)
    assert ok


def test_config_hash_covers_nested_loss_spec():
    import json

    from fejepa.report import config_sha256

    cfg = json.loads((__import__("pathlib").Path(__file__).resolve().parents[1]
                      / "configs" / "e1_2d_shaped.json").read_text())
    cfg["pretrain"]["loss_spec"].update({"lambda_reg": 0.1, "sigreg_head_width": 0})
    h = config_sha256(cfg)
    cfg["pretrain"]["loss_spec"]["lambda_reg"] = 0.3
    assert config_sha256(cfg) != h
    cfg["pretrain"]["loss_spec"].update({"lambda_reg": 0.1, "sigreg_head_width": 24})
    assert config_sha256(cfg) != h


def test_dry_run_reports_verified_when_stamped(tmp_path):
    import json
    from pathlib import Path

    from fejepa.experiments.runner import run_config
    from fejepa.report import stamp_prereg

    cfg = {"model": {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}},
           "data": {"dir": str(tmp_path / "nope"), "n": 4, "seed": 1, "backend": "synthetic"},
           "split": {"n_val": 1, "seed": 1},
           "experiments": {"e8": {"enabled": True, "ar_only": True, "budgets": [2], "pool_sizes": [2],
                                  "seeds": 1, "ar_epochs": 1}},
           "prereg_guard": True, "prereg_file": str(tmp_path / "PREREG_X.md"),
           "out": str(tmp_path / "r.json"), "device": "cpu", "workers": 1}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    Path(cfg["prereg_file"]).write_text("# X\n\nCONFIG_SHA256 = <fill before tagging>\n")
    stamp_prereg(Path(cfg["prereg_file"]), cfg)
    s = run_config(str(p), dry_run=True)
    assert s["prereg_status"] == "verified" and s["prereg_verified"]
    assert not (tmp_path / "nope").exists()
