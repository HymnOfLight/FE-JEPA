"""D9 (Phase-2 attempt-1 OOM) fixes: MGN checkpointing exactness, AR-state
reuse with hash chaining, unit-result cache, and the restart mode end to end."""

import copy
import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

from fejepa.data.archive import load_instance
from fejepa.experiments.parallel import (_build_model, cached_supervised_unit,
                                         pretrain_unit)
from fejepa.experiments.protocol import load_split
from fejepa.experiments.runner import _label_files, run_config
from fejepa.fe.solve import SolveLedger
from fejepa.fe.synthetic import generate_synthetic_dataset

MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 16, "mgn_depth": 2,
         "features": {"load_summary": True, "geometry": True}}


def _tiny(tmp_path, n=6, seed=11):
    d = generate_synthetic_dataset(tmp_path / f"d{seed}", n=n, seed=seed)
    sp = load_split(d, n_val=2, seed=1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "v")
    _label_files(sp.pool_files[:2], led, "p")
    return sp


def test_mgn_checkpoint_is_bitwise_exact(tmp_path):
    sp = _tiny(tmp_path)
    arch = load_instance(sp.pool_files[0])
    m1 = _build_model({"kind": "mgn", "model": MODEL, "seed": 0})
    m2 = copy.deepcopy(m1)
    m1.use_checkpoint, m2.use_checkpoint = True, False
    outs, grads = [], []
    for m in (m1, m2):
        m.train()
        pack = m.prepare_instance(arch, "cpu")
        u = m.forward_instance(pack)
        (u * u).mean().backward()
        outs.append(u.detach().clone())
        grads.append([p.grad.detach().clone() for p in m.parameters()])
    assert torch.equal(outs[0], outs[1])
    assert all(torch.equal(a, b) for a, b in zip(grads[0], grads[1], strict=True))


def test_pretrain_unit_reuses_existing_state(tmp_path):
    sp = _tiny(tmp_path)
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
               "files": [str(f) for f in sp.pool_files[:2]],
               "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
               "state_path": str(tmp_path / "st" / "ar.pt"),
               "eval_val_files": [str(f) for f in sp.val_files], "quiet": True}
    first = pretrain_unit(payload)
    assert first["reused_state"] is False and "val" in first
    second = pretrain_unit({**payload, "reuse_existing": True})
    assert second["reused_state"] is True and "val" in second
    file_sha = hashlib.sha256((tmp_path / "st" / "ar.pt").read_bytes()).hexdigest()
    assert second["state_sha256"] == file_sha == first["state_sha256"]
    # reused model == saved model: identical val metrics
    assert second["val"]["disp_rel_l2"] == pytest.approx(first["val"]["disp_rel_l2"])


def test_cached_supervised_unit_hits_on_second_call(tmp_path):
    sp = _tiny(tmp_path)
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
               "train_files": [str(f) for f in sp.pool_files[:2]],
               "val_files": [str(f) for f in sp.val_files],
               "sup": {"epochs": 1, "lr": 1e-3, "device": "cpu",
                       "anchor_mode": "none", "log_every": -1},
               "tag": "labels b2 s0", "cache_dir": str(tmp_path / "cache")}
    a = cached_supervised_unit(payload)                 # writes the cache
    not_reused = cached_supervised_unit(payload)        # fresh run: must NOT read
    assert not not_reused.get("from_cache")
    payload["reuse_existing"] = True
    assert not a.get("from_cache") and (tmp_path / "cache" / "labels_b2_s0.pkl").exists()
    b = cached_supervised_unit(payload)
    assert b["from_cache"] is True
    assert b["val"]["disp_rel_l2"] == a["val"]["disp_rel_l2"]


def test_restart_mode_end_to_end(tmp_path):
    """The exact D9 path: run once, then run again with reuse_states -- the
    second report must show AR states reused and every supervised unit
    served from the cache."""
    d = generate_synthetic_dataset(tmp_path / "corpus", n=10, seed=3)
    df = generate_synthetic_dataset(tmp_path / "fine", n=6, seed=4)
    cfg = {"data": {"dir": str(d), "n": 10, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "economy"},
           "data_transfer": {"dir": str(df), "n": 6, "seed": 4,
                             "backend": "synthetic", "labelled_policy": "economy",
                             "split": {"n_eval": 3, "n_fewshot_prefix": 2}},
           "split": {"n_val": 3, "seed": 1}, "model": MODEL,
           "sup": {"epochs": 1, "lr": 1e-3}, "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {
               "e8": {"enabled": True, "budgets": [2, 4], "pool_sizes": [4],
                      "seeds": 1, "ar_epochs": 1, "sup_epochs": 1,
                      "include_mgn": True, "include_ar_ft": False,
                      "mgn_budgets": [4]},
               "p3_transfer": {"enabled": True, "fewshot_budgets": [2],
                               "fewshot_epochs": 1, "naive_budget": 4},
               "e6": {"enabled": True, "pool_size": 4},
               "wp6": {"enabled": True, "n_check": 2, "seed": 0}},
           "gate_g2": {"sanity_x": 3.0, "naive_set": ["knn_field", "scale_aware_poly"],
                       "parity_band": 0.10, "egap_adv_min": 0.40,
                       "transfer_win": 1.25, "decision_budget": 4},
           "kills": {"KP1_parity_pct": 0.10, "KP2_egap_adv_min": 0.40,
                     "KP3_anchor_improv_min": 0.25, "KP4_transfer_ratio": 1.5,
                     "KP6_rho_within_min": 0.3},
           "device": "cpu", "workers": 1, "tf32": False,
           "runtime": {"compile": False, "amp": False, "precision": "fp32"},
           "seeds": [0], "out": str(tmp_path / "out" / "report.json"),
           "prereg_guard": False}
    cpath = tmp_path / "cfg.json"
    cpath.write_text(json.dumps(cfg))
    run_config(str(cpath))                       # attempt 1 (completes)
    r2 = run_config(str(cpath), reuse_states=True)   # attempt 2 (restart mode)
    d9 = r2["results"]["e8"]["metrics"]["d9_restart"]
    assert d9["reuse_states"] is True
    assert all(v["reused"] for v in d9["ar_states"].values())
    assert all(v["sha256"] for v in d9["ar_states"].values())
    assert len(d9["sup_units_from_cache"]) == 2 * 2 + 1   # labels/anchor x2 budgets + mgn@4
    assert r2["d9_reuse_states"] is True
    assert "gate_g2" in r2
