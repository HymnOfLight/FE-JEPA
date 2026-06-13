import numpy as np
import torch

from fejepa.losses import LossConfig, compute_instance_loss
from fejepa.models.encoder import NODE_FEATURE_DIM, build_node_features
from fejepa.models.fejepa import FEJEPA, FEJEPAConfig
from fejepa.models.gnn import MeshGNN, build_edges


def test_encoder_decoder_shapes(small_archive):
    model = FEJEPA(FEJEPAConfig(dim=32, depth=2))
    feats = build_node_features(small_archive, 0)
    assert feats.shape == (small_archive.n_nodes, NODE_FEATURE_DIM)
    latents, disp = model.encode_decode(feats)
    assert latents.shape == (small_archive.n_nodes, 32)
    assert disp.shape == (small_archive.n_nodes, 2)


def test_combined_loss_backward(small_archive, multires_pair):
    fine, coarse = multires_pair
    model = FEJEPA(FEJEPAConfig(dim=32, depth=2))
    total, parts = compute_instance_loss(
        model, fine, LossConfig(), arch_coarse=coarse, rng=np.random.default_rng(0)
    )
    assert {"phys", "pred", "sigreg", "inv", "total"} <= set(parts)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_gnn_baseline_forward(small_archive):
    model = MeshGNN(dim=32, depth=2)
    feats = build_node_features(small_archive, 0)
    edges = build_edges(small_archive)
    coords = torch.as_tensor(small_archive.nodes, dtype=torch.float32)
    _, disp = model.encode_decode(feats, edges, coords)
    assert disp.shape == (small_archive.n_nodes, 2)
