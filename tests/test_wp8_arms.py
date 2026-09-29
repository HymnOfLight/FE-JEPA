"""wp8-lejepa Stage 1.0: the E1 arm (AR + validated SIGReg on a projector head)
and the E2 bottleneck prototype run through the production paths; legacy
paths are untouched (bitwise regression against the tag is run separately)."""

import pytest

torch = pytest.importorskip("torch")

from fejepa.anchor.energy import AnchorCache
from fejepa.data.archive import load_instance
from fejepa.experiments.parallel import _build_model, pretrain_unit, supervised_unit
from fejepa.experiments.protocol import load_split
from fejepa.experiments.runner import _label_files
from fejepa.fe.solve import SolveLedger
from fejepa.fe.synthetic import generate_synthetic_dataset
from fejepa.metrics import evaluate_model, torch_predictor
from fejepa.models.bottleneck import farthest_point_sampling, nearest_seed
from fejepa.train.losses import AR_CONFIG, ar_sigreg_config
from fejepa.train.pretrain import PretrainConfig, pretrain

MODEL = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}
BOTTLE = {"dim": 16, "depth": 1, "heads": 2, "n_tokens": 8,
          "features": {"load_summary": True, "geometry": True}}




# ------------------------------- E1 -----------------------------------------
def test_e1_ar_sigreg_head_trains_and_state_is_strict_loadable(tmp_path, tiny_corpus):
    sp = tiny_corpus(seed=31)
    tr = [load_instance(f) for f in sp.pool_files[:3]]
    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    h = pretrain(m, tr, PretrainConfig(epochs=2, lr=1e-3, seed=0, device="cpu",
                                       loss=ar_sigreg_config(0.1, head=True, n_proj=32),
                                       log_every=-1))
    assert hasattr(m, "sigreg_head") and torch.isfinite(torch.tensor(h["loss"])).all()
    # through the production unit: the saved state has NO head and loads strictly
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
               "files": [str(f) for f in sp.pool_files[:3]],
               "loss": {"reg_mode": "sigreg_ep_head", "lambda_reg": 0.1, "sigreg_n_proj": 32},
               "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
               "state_path": str(tmp_path / "st" / "ar_sig.pt"),
               "eval_val_files": [str(f) for f in sp.val_files], "quiet": True}
    out = pretrain_unit(payload)
    sd = torch.load(out["state_path"], map_location="cpu", weights_only=True)
    assert not any(k.startswith("sigreg_head.") for k in sd)
    fresh = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    fresh.load_state_dict(sd, strict=True)
    assert "val" in out and torch.isfinite(torch.tensor(out["val"]["disp_rel_l2"]))


def test_e1_raw_mode_needs_no_head(tmp_path, tiny_corpus):
    sp = tiny_corpus(seed=31)
    tr = [load_instance(f) for f in sp.pool_files[:2]]
    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m, tr, PretrainConfig(epochs=1, lr=1e-3, seed=0, device="cpu",
                                   loss=ar_sigreg_config(0.1, head=False, n_proj=16),
                                   log_every=-1))
    assert not hasattr(m, "sigreg_head")


# ------------------------------- E2 -----------------------------------------
def test_fps_and_assignment_are_deterministic_and_cover():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.random((500, 3))
    s1, s2 = farthest_point_sampling(x, 32), farthest_point_sampling(x, 32)
    assert np.array_equal(s1, s2) and len(set(s1.tolist())) == 32
    a = nearest_seed(x, x[s1])
    assert a.shape == (500,) and a.min() >= 0 and a.max() < 32
    assert np.array_equal(a[s1], np.arange(32))          # seeds map to themselves
    assert farthest_point_sampling(x[:5], 32).shape == (5,)   # n < M handled


def test_blend_weights_are_continuous_and_reduce_to_the_hard_assignment():
    """Stage 1.32 (PI decision): the decoder blends a node's k nearest tokens
    with Franke-Little weights. They are a partition of unity, sorted nearest
    first, give weight 1 to a point sitting on a seed, and vary continuously
    along any path -- where the hard (k = 1) assignment jumps by a whole token
    at every cell boundary. M <= k falls back to Shepard weights over all seeds."""
    import numpy as np

    from fejepa.models.bottleneck import blend_weights

    rng = np.random.default_rng(0)
    x = rng.random((3000, 3))
    seeds = x[farthest_point_sampling(x, 48)]
    idx, w, rel = blend_weights(x, seeds, 4)
    assert idx.shape == w.shape == (3000, 4) and rel.shape == (3000, 4, 3)
    assert np.allclose(w.sum(1), 1.0) and w.min() >= 0.0
    d = np.linalg.norm(rel, axis=2)
    assert np.all(np.diff(d, axis=1) >= 0.0)                       # nearest first
    assert np.allclose(rel, x[:, None, :] - seeds[idx])
    assert np.array_equal(idx[:, 0], nearest_seed(x, seeds))      # encoder cell = column 0
    on_i, on_w, _ = blend_weights(seeds, seeds, 4)
    assert np.array_equal(on_i[:, 0], np.arange(48)) and np.allclose(on_w[:, 0], 1.0)
    t = np.linspace(0.0, 1.0, 40001)[:, None]
    path = np.hstack([0.05 + 0.9 * t, 0.5 + 0.35 * np.sin(4 * t), 0.2 + 0.6 * t ** 2])

    def dense(kk):
        pi, pw, _ = blend_weights(path, seeds, kk)
        full = np.zeros((path.shape[0], 48))
        np.put_along_axis(full, pi, pw, 1)
        return np.abs(np.diff(full, axis=0)).sum(1)                # L1 change per step

    soft, hard = dense(4), dense(1)
    assert hard.max() == 2.0 and (hard > 0.5).sum() >= 5           # hard: whole-token jumps
    assert soft.max() < 0.02                                       # continuous: no jump
    few_i, few_w, _ = blend_weights(x[:10], seeds[:3], 4)
    assert few_i.shape == (10, 3) and np.allclose(few_w.sum(1), 1.0)


def test_blend_weights_do_not_depend_on_node_numbering():
    import numpy as np

    from fejepa.models.bottleneck import blend_weights

    rng = np.random.default_rng(3)
    x = rng.random((800, 3))
    perm = rng.permutation(800)
    s_a = x[farthest_point_sampling(x, 32)]
    s_b = x[perm][farthest_point_sampling(x[perm], 32)]
    assert np.array_equal(s_a, s_b)                                 # canonical seed order
    ia, wa, _ = blend_weights(x, s_a, 4)
    ib, wb, _ = blend_weights(x[perm], s_b, 4)
    assert np.array_equal(ia[perm], ib) and np.allclose(wa[perm], wb, rtol=0, atol=1e-15)


def test_decode_k1_is_the_former_hard_nearest_token_decoder(tiny_corpus):
    """k = 1 reproduces the pre-1.32 decoder exactly: every node reads its own
    nearest token's latent and the offset to that token's seed."""
    sp = tiny_corpus(seed=33)
    arch = load_instance(sp.val_files[0])
    m = _build_model({"kind": "bottleneck", "model": dict(BOTTLE, decode_k=1), "seed": 0})
    m.eval()
    pack = m.prepare_instance(arch, "cpu")
    assert pack["nbr_idx"].shape[1] == 1 and torch.all(pack["nbr_w"] == 1.0)
    with torch.no_grad():
        z = m.encode(pack["feats"], pack)
        u = m.decode(z, pack)
        h = m.node_embed(pack["feats"])
        zt = z[:, pack["tok_idx"], :]
        r = m.rel_pos(pack["nbr_rel"][:, 0, :]).unsqueeze(0).expand_as(h)
        legacy = m.dec(torch.cat([h, zt, r], dim=-1)).reshape(u.shape[0], -1) * pack["free"]
        legacy = legacy * pack["fscale"]
    assert torch.equal(u, legacy)                                  # bitwise (CPU)
    m4 = _build_model({"kind": "bottleneck", "model": BOTTLE, "seed": 0})
    assert m4.cfg.decode_k == 4 and m4.prepare_instance(arch, "cpu")["nbr_idx"].shape[1] == 4


def test_bottleneck_matches_pack_contract_and_feeds_the_anchor(tmp_path, tiny_corpus):
    sp = tiny_corpus(seed=31)
    arch = load_instance(sp.pool_files[0])
    m = _build_model({"kind": "bottleneck", "model": BOTTLE, "seed": 0})
    pack = m.prepare_instance(arch, "cpu")
    L, N = pack["feats"].shape[0], pack["feats"].shape[1]
    z = m.encode(pack["feats"], pack)
    assert z.shape == (L, pack["n_tok"], 16) and pack["n_tok"] <= min(8, N)
    u = m.forward_instance(pack)
    assert u.shape == (L, pack["free"].numel())
    anc = AnchorCache(device="cpu").get(arch)
    e = anc.energies(u)                                   # the SAME exact anchor
    assert e.shape == (L,) and torch.isfinite(e).all()
    e.mean().backward()
    grads = [p.grad for p in m.parameters()]
    assert all(g is not None for g in grads) and all(torch.isfinite(g).all() for g in grads)


def test_bottleneck_trains_through_ar_and_supervised_units(tmp_path, tiny_corpus):
    sp = tiny_corpus(seed=31)
    base = {"kind": "bottleneck", "model": BOTTLE, "seed": 0, "tf32": False}
    out = pretrain_unit({**base, "files": [str(f) for f in sp.pool_files[:3]],
                         "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
                         "state_path": str(tmp_path / "bt" / "ar.pt"),
                         "eval_val_files": [str(f) for f in sp.val_files], "quiet": True})
    assert torch.isfinite(torch.tensor(out["val"]["energy_gap_rel"]))
    res = supervised_unit({**base, "train_files": [str(f) for f in sp.pool_files[:2]],
                           "val_files": [str(f) for f in sp.val_files],
                           "sup": {"epochs": 1, "lr": 1e-3, "device": "cpu",
                                   "anchor_mode": "balanced", "balance_ratio": 1.0,
                                   "log_every": -1}, "tag": "bt b2 s0"})
    assert torch.isfinite(torch.tensor(res["val"]["disp_rel_l2"]))
    m = _build_model(base)
    m.load_state_dict(torch.load(out["state_path"], map_location="cpu", weights_only=True),
                      strict=True)
    ev = evaluate_model(torch_predictor(m, "cpu"), [load_instance(f) for f in sp.val_files])
    assert torch.isfinite(torch.tensor(ev["disp_rel_l2"]))


# ------------------------- 3D path and resume interplay ----------------------
def test_bottleneck_runs_on_a_real_3d_gmsh_instance(tmp_path, tiny_corpus):
    """E2 targets 3D: the bottleneck must run on a tetrahedral gmsh instance
    (3 coordinate columns, spatial_dim-3 features, ndof = 3N) through forward,
    the exact anchor, a supervised step and evaluation."""
    from fejepa.fe.gmsh3d import generate_gmsh3d_dataset
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    d = generate_gmsh3d_dataset(tmp_path / "g3", 3, 5, labelled="none", lc_range=(0.45, 0.6))
    sp = load_split(str(d), 1, 1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "v")
    _label_files(sp.pool_files[:2], led, "p")
    m3 = {"dim": 16, "depth": 1, "heads": 2, "n_tokens": 16,
          "features": {"load_summary": True, "geometry": True, "spatial_dim": 3}}
    m = _build_model({"kind": "bottleneck", "model": m3, "seed": 0})
    tr = [load_instance(f) for f in sp.pool_files[:2]]
    val = [load_instance(f) for f in sp.val_files]
    pack = m.prepare_instance(tr[0], "cpu")
    assert pack["nbr_rel"].shape[1:] == (4, 3) and pack["seed_xyz"].shape[1] == 3
    assert pack["nbr_idx"].shape == pack["nbr_w"].shape == (tr[0].nodes.shape[0], 4)
    u = m.forward_instance(pack)
    assert u.shape[1] == 3 * tr[0].nodes.shape[0] == pack["free"].numel()
    e = AnchorCache(device="cpu").get(tr[0]).energies(u)
    assert torch.isfinite(e).all()
    res = train_supervised(m, tr, val, SupervisedConfig(epochs=1, lr=1e-3, seed=0, device="cpu",
                                                        anchor_mode="none", log_every=-1))
    assert torch.isfinite(torch.tensor(res["val"]["disp_rel_l2"]))
    ev = evaluate_model(torch_predictor(m, "cpu"), val)
    assert torch.isfinite(torch.tensor(ev["energy_gap_rel"]))


def test_e1_sigreg_resume_is_bitwise_exact(tmp_path, tiny_corpus):
    """SIGReg draws random directions from the global torch RNG every step; the
    R9 checkpoint restores that RNG, so an interrupted E1 run must land on
    the uninterrupted parameters bitwise (head included)."""
    sp = tiny_corpus(seed=41)
    tr = [load_instance(f) for f in sp.pool_files[:3]]
    base = dict(epochs=3, lr=1e-3, seed=0, device="cpu",
                loss=ar_sigreg_config(0.1, head=True, n_proj=32), log_every=-1)
    m_ref = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m_ref, tr, PretrainConfig(**base))
    ck = str(tmp_path / "e1.ckpt")
    m_a = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m_a, tr, PretrainConfig(**base, ckpt_path=ck, stop_after_epoch=1))
    m_b = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m_b, tr, PretrainConfig(**base, ckpt_path=ck, resume=True))
    pa = dict(m_ref.named_parameters())
    pb = dict(m_b.named_parameters())
    assert set(pa) == set(pb) and any(k.startswith("sigreg_head.") for k in pa)
    assert all(torch.equal(pa[k].detach(), pb[k].detach()) for k in pa)


def test_e1_head_width_rule_plumbing(tmp_path, tiny_corpus):
    """The Stage-0 reading rule sizes the projector head by intrinsic dimension:
    a narrower head must be honoured end to end."""
    sp = tiny_corpus(seed=43)
    tr = [load_instance(f) for f in sp.pool_files[:2]]
    m = _build_model({"kind": "fejepa", "model": MODEL, "seed": 0})
    pretrain(m, tr, PretrainConfig(epochs=1, lr=1e-3, seed=0, device="cpu",
                                   loss=ar_sigreg_config(0.1, head=True, n_proj=16,
                                                         head_width=6),
                                   log_every=-1))
    assert m.sigreg_head[1].num_features == 6 and m.sigreg_head[2].out_features == 6


# --------------- E1 and E2 are executable through run-config ---------------
def test_e1_and_e2_arms_run_through_run_config(tmp_path, tiny_corpus):
    """The pre-registration drafts require model.kind = bottleneck (E2) and a
    dict AR loss spec (E1) to be honoured by the production runner: E8 AR
    pretraining, the supervised grid, P3 zero/few-shot and the gate."""
    import json

    from fejepa.experiments.runner import run_config

    d = generate_synthetic_dataset(tmp_path / "corpus", n=10, seed=3)
    df = generate_synthetic_dataset(tmp_path / "fine", n=6, seed=4)
    cfg = {"data": {"dir": str(d), "n": 10, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "economy"},
           "data_transfer": {"dir": str(df), "n": 6, "seed": 4, "backend": "synthetic",
                             "labelled_policy": "economy",
                             "split": {"n_eval": 3, "n_fewshot_prefix": 2}},
           "split": {"n_val": 3, "seed": 1},
           "model": {**BOTTLE, "kind": "bottleneck", "mgn_dim": 16, "mgn_depth": 2},
           "sup": {"epochs": 1, "lr": 1e-3},
           "pretrain": {"epochs": 1, "lr": 1e-3,
                        "loss_spec": {"reg_mode": "sigreg_ep_head", "lambda_reg": 0.1,
                                      "sigreg_n_proj": 16}},
           "experiments": {
               "e8": {"enabled": True, "budgets": [2, 4], "pool_sizes": [4], "seeds": 1,
                      "ar_epochs": 1, "sup_epochs": 1, "include_mgn": True,
                      "include_ar_ft": False, "mgn_budgets": [4]},
               "p3_transfer": {"enabled": True, "fewshot_budgets": [2],
                               "fewshot_epochs": 1, "naive_budget": 4},
               "wp6": {"enabled": False},           # FE-JEPA-only probe: off under bottleneck
               "e1": {"enabled": False, "seeds": 3}},   # disabled: must not count (Stage 1.31)
           "gate_g2": {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"],
                       "parity_band": 0.10, "egap_adv_min": 0.40, "transfer_win": 1.25,
                       "decision_budget": 4},
           "kills": {"KP1_parity_pct": 0.10, "KP2_egap_adv_min": 0.40,
                     "KP3_anchor_improv_min": 0.25, "KP4_transfer_ratio": 1.5,
                     "KP6_rho_within_min": 0.3},
           "device": "cpu", "workers": 1, "tf32": False,
           "runtime": {"compile": False, "amp": False, "precision": "fp32"},
           "seeds": [0], "out": str(tmp_path / "out" / "report.json"), "prereg_guard": False}
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps(cfg))
    r = run_config(str(cpath))
    e8 = r["results"]["e8"]["metrics"]
    assert set(e8["cells"]) >= {"labels", "labels_anchor", "mgn", "ar"}
    assert "p3_transfer" in r["results"] and "gate_g2" in r
    st = torch.load(tmp_path / "out" / "e8_states" / "ar_p4_s0.pt", map_location="cpu",
                    weights_only=True)
    assert not any(k.startswith("sigreg_head.") for k in st)      # E1 head stripped
    assert any(k.startswith("tok_enc.") for k in st)                # E2 architecture
    # Stage 1.31: provenance counts the seeds of ENABLED experiments only (the
    # disabled e1 block's "seeds": 3 made one-seed runs record seeds [0, 1, 2])
    assert r["provenance"]["seeds"] == [0]


def test_bottleneck_token_layers_are_initialised_independently():
    """Stage 1.31: nn.TransformerEncoder deep-copies one initialised layer;
    the bottleneck re-draws layers 1..depth-1 so no two start identical (as
    FE-JEPA's blocks), and the draw is still a function of the seed."""
    cfg = dict(BOTTLE, depth=3)
    m = _build_model({"kind": "bottleneck", "model": cfg, "seed": 0})
    w = [lay.self_attn.in_proj_weight.detach() for lay in m.tok_enc.layers]
    l1 = [lay.linear1.weight.detach() for lay in m.tok_enc.layers]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(w[i], w[j]) and not torch.equal(l1[i], l1[j])
    again = _build_model({"kind": "bottleneck", "model": cfg, "seed": 0})
    for (ka, va), (kb, vb) in zip(m.state_dict().items(), again.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    other = _build_model({"kind": "bottleneck", "model": cfg, "seed": 1})
    assert not torch.equal(other.tok_enc.layers[1].linear1.weight, l1[1])


def test_ar_only_runs_end_to_end_with_bottleneck_and_p3(tmp_path, tiny_corpus):
    """Both E-series drafts execute E8 as AR pretraining only: no supervised
    cells, no naive rows, label-dependent kills marked unevaluated, P3
    zero-shot tolerant of the missing supervised states, gate not crashing."""
    import json

    from fejepa.experiments.runner import run_config

    d = generate_synthetic_dataset(tmp_path / "corpus", n=10, seed=3)
    df = generate_synthetic_dataset(tmp_path / "fine", n=6, seed=4)
    cfg = {"data": {"dir": str(d), "n": 10, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "economy"},
           "data_transfer": {"dir": str(df), "n": 6, "seed": 4, "backend": "synthetic",
                             "labelled_policy": "economy",
                             "split": {"n_eval": 3, "n_fewshot_prefix": 2}},
           "split": {"n_val": 3, "seed": 1},
           "model": {**BOTTLE, "kind": "bottleneck"},
           "sup": {"epochs": 1, "lr": 1e-3},
           "pretrain": {"epochs": 1, "lr": 1e-3,
                        "loss_spec": {"reg_mode": "sigreg_ep_head", "lambda_reg": 0.1,
                                      "sigreg_n_proj": 16}},
           "experiments": {
               "e8": {"enabled": True, "ar_only": True, "budgets": [2, 4], "pool_sizes": [4],
                      "seeds": 1, "ar_epochs": 1, "include_mgn": True},
               "p3_transfer": {"enabled": True, "fewshot_budgets": [2],
                               "fewshot_epochs": 1, "naive_budget": 4},
               "wp6": {"enabled": False}},
           "gate_g2": {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"],
                       "parity_band": 0.10, "egap_adv_min": 0.40, "transfer_win": 1.25,
                       "decision_budget": 4},
           "kills": {"KP1_parity_pct": 0.10, "KP2_egap_adv_min": 0.40,
                     "KP3_anchor_improv_min": 0.25, "KP4_transfer_ratio": 1.5,
                     "KP6_rho_within_min": 0.3},
           "device": "cpu", "workers": 1, "tf32": False,
           "runtime": {"compile": False, "amp": False, "precision": "fp32"},
           "seeds": [0], "out": str(tmp_path / "out" / "report.json"), "prereg_guard": False}
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps(cfg))
    r = run_config(str(cpath))
    e8 = r["results"]["e8"]
    assert set(e8["metrics"]["cells"]) == {"ar"} and e8["protocol"]["ar_only"] is True
    assert all(not k["triggered"] for k in e8["kills"])
    assert "p3_transfer" in r["results"] and "gate_g2" in r
    assert r["solve_ledger"]["total"] > 0     # val labels only; no pool prefix labels bought


def test_e_series_economy_is_zero_beyond_val_and_fine_eval(tmp_path):
    """E-series configuration on FRESH corpora: e8.ar_only + P3 zero-shot only
    (fewshot_budgets [], naive_budget 0) buys exactly the in-band val and the
    fine evaluation labels -- no pool prefix, no fine prefix -- and still
    yields the transfer ratio the E1 GO rule needs."""
    import json

    from fejepa.experiments.runner import run_config

    d = generate_synthetic_dataset(tmp_path / "corpus", n=10, seed=3)
    df = generate_synthetic_dataset(tmp_path / "fine", n=6, seed=4)
    cfg = {"data": {"dir": str(d), "n": 10, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "economy"},
           "data_transfer": {"dir": str(df), "n": 6, "seed": 4, "backend": "synthetic",
                             "labelled_policy": "economy",
                             "split": {"n_eval": 3, "n_fewshot_prefix": 2}},
           "split": {"n_val": 3, "seed": 1}, "model": MODEL,
           "sup": {"epochs": 1, "lr": 1e-3}, "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {
               "e8": {"enabled": True, "ar_only": True, "budgets": [2, 4], "pool_sizes": [4],
                      "seeds": 1, "ar_epochs": 1},
               "p3_transfer": {"enabled": True, "fewshot_budgets": [], "fewshot_epochs": 1,
                               "naive_budget": 0},
               "wp6": {"enabled": False}},
           "gate_g2": {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"],
                       "parity_band": 0.10, "egap_adv_min": 0.40, "transfer_win": 1.25,
                       "decision_budget": 4},
           "kills": {"KP1_parity_pct": 0.10, "KP2_egap_adv_min": 0.40,
                     "KP3_anchor_improv_min": 0.25, "KP4_transfer_ratio": 1.5,
                     "KP6_rho_within_min": 0.3},
           "device": "cpu", "workers": 1, "tf32": False,
           "runtime": {"compile": False, "amp": False, "precision": "fp32"},
           "seeds": [0], "out": str(tmp_path / "out" / "report.json"), "prereg_guard": False}
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps(cfg))
    r = run_config(str(cpath))
    stages = r["solve_ledger"]["per_stage"]
    assert set(stages) == {"labelling-val", "labelling-fine-val"}, stages
    p3 = r["results"]["p3_transfer"]["metrics"]
    assert p3["naive_at_fine"] == {} and p3["fewshot"] == {}
    assert p3["ar"]["fine_disp_mean"] > 0 and p3["ar"]["inband_disp_mean"] > 0


def test_bottleneck_ar_resume_is_bitwise_exact(tmp_path, tiny_corpus):
    """E2's units rely on the R9 resume path: an interrupted bottleneck AR run
    (FPS-seeded packs rebuilt from scratch on resume) must land on the
    uninterrupted parameters bitwise."""
    sp = tiny_corpus(seed=47)
    tr = [load_instance(f) for f in sp.pool_files[:3]]
    base = dict(epochs=3, lr=1e-3, seed=0, device="cpu", loss=AR_CONFIG, log_every=-1)
    m_ref = _build_model({"kind": "bottleneck", "model": BOTTLE, "seed": 0})
    pretrain(m_ref, tr, PretrainConfig(**base))
    ck = str(tmp_path / "bt.ckpt")
    m_a = _build_model({"kind": "bottleneck", "model": BOTTLE, "seed": 0})
    pretrain(m_a, tr, PretrainConfig(**base, ckpt_path=ck, stop_after_epoch=1))
    m_b = _build_model({"kind": "bottleneck", "model": BOTTLE, "seed": 0})
    pretrain(m_b, tr, PretrainConfig(**base, ckpt_path=ck, resume=True))
    pa, pb = dict(m_ref.named_parameters()), dict(m_b.named_parameters())
    assert set(pa) == set(pb) and any(k.startswith("tok_enc.") for k in pa)
    assert all(torch.equal(pa[k].detach(), pb[k].detach()) for k in pa)


def test_reuse_tolerates_compiled_state_prefix(tmp_path, tiny_corpus):
    """A state saved under a torch.compile wrapper carries `_orig_mod.` keys;
    the reuse path (and the analysis loader) normalise it instead of falling
    back to a retrain."""
    from fejepa.analysis.common import build_model_from_config

    sp = tiny_corpus(seed=71)
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
               "files": [str(f) for f in sp.pool_files[:2]],
               "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
               "state_path": str(tmp_path / "st" / "ar.pt"),
               "eval_val_files": [str(f) for f in sp.val_files], "quiet": True}
    pretrain_unit(payload)
    sd = torch.load(payload["state_path"], map_location="cpu", weights_only=True)
    torch.save({"_orig_mod." + k: v for k, v in sd.items()}, payload["state_path"])
    out = pretrain_unit({**payload, "reuse_existing": True})
    assert out["reused_state"] is True
    m = build_model_from_config(MODEL, payload["state_path"])
    assert m is not None
