"""cmame-paper Stage 2: the supervised stiffness-norm loss and its E8 row, and
an E8 run that trains a supervised grid beside label-free states it reuses.
Every addition is opt-in: a configuration without the new keys runs exactly as
before (the protocol records gain nothing, the rows are the same)."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.anchor.energy import AnchorCache, pi_h, pi_star_abs
from fejepa.experiments.cost import count_steps
from fejepa.experiments.runner import run_config
from fejepa.experiments.w9_eval import verify_reuse
from fejepa.fe.synthetic import generate_synthetic_dataset, synthetic_instance
from fejepa.train.supervised import SupervisedConfig, _knorm_loss, train_supervised

MODEL = {"dim": 16, "depth": 1, "heads": 2, "mgn_dim": 8, "mgn_depth": 1,
         "features": {"load_summary": True, "geometry": True}}


def _instance(seed=0):
    return synthetic_instance(np.random.default_rng(seed), labelled=True)


def test_quad_is_the_stiffness_quadratic_form_with_its_gradient():
    a = _instance()
    anchor = AnchorCache(dtype=torch.float64).get(a)
    rng = np.random.default_rng(1)
    free = ~np.asarray(a.dirichlet_mask, dtype=bool)
    V = rng.normal(size=a.U_star.shape) * free
    v = torch.tensor(V, dtype=torch.float64, requires_grad=True)
    q = anchor.quad(v)
    want = np.einsum("ld,ld->l", V, (a.K @ V.T).T)
    assert np.allclose(q.detach().numpy(), want, rtol=1e-12)
    q.sum().backward()
    assert np.allclose(v.grad.numpy(), 2 * (a.K @ V.T).T * free, rtol=1e-10, atol=1e-12)
    # the constrained dofs are masked: a field there contributes nothing
    W = V + rng.normal(size=V.shape) * ~free
    assert np.allclose(anchor.quad(torch.tensor(W)).numpy(), want, rtol=1e-12)


def test_knorm_loss_is_the_mean_relative_stiffness_norm_error():
    a = _instance(2)
    anchor = AnchorCache(dtype=torch.float64).get(a)
    rng = np.random.default_rng(3)
    free = ~np.asarray(a.dirichlet_mask, dtype=bool)
    U = a.U_star * (1 + 0.2 * rng.normal(size=a.U_star.shape)) * free
    u = torch.tensor(U, dtype=torch.float64, requires_grad=True)
    us = torch.tensor(a.U_star, dtype=torch.float64)
    uk = torch.tensor(np.sqrt(2 * pi_star_abs(a)), dtype=torch.float64)
    loss = _knorm_loss(anchor, u, us, uk)
    # sqrt of the relative energy gap (Lemma 1), averaged over the load cases
    g = (pi_h(U, a.K, a.F) - pi_h(a.U_star, a.K, a.F)) / pi_star_abs(a)
    assert np.isclose(float(loss.detach()), np.mean(np.sqrt(g)), rtol=1e-9)
    loss.backward()
    E = (U - a.U_star) * free
    KE = (a.K @ E.T).T
    nk = np.sqrt(np.einsum("ld,ld->l", E, KE))
    want = KE * free / (nk * np.sqrt(2 * pi_star_abs(a)))[:, None] / a.n_loads
    assert np.allclose(u.grad.numpy(), want, rtol=1e-8, atol=1e-14)


def test_supervised_config_records_the_loss_only_when_it_is_not_the_default():
    assert "loss" not in SupervisedConfig().protocol()
    assert SupervisedConfig(loss="knorm").protocol()["loss"] == "knorm"


def test_knorm_training_runs_and_lowers_its_loss():
    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa

    rng = np.random.default_rng(4)
    archs = [synthetic_instance(rng, labelled=True) for _ in range(3)]
    torch.manual_seed(0)
    model = build_fejepa(FEJEPAConfig.from_dict(MODEL))

    def train_loss():
        anchors = AnchorCache()
        out = []
        with torch.no_grad():
            for a in archs:
                pack = model.prepare_instance(a, "cpu")
                u = model.forward_instance(pack)
                us = torch.as_tensor(a.U_star, dtype=u.dtype)
                uk = torch.as_tensor(np.sqrt(2 * pi_star_abs(a)), dtype=u.dtype)
                out.append(float(_knorm_loss(anchors.get(a), u, us, uk)))
        return float(np.mean(out))

    before = train_loss()
    res = train_supervised(model, archs, archs[:1],
                           SupervisedConfig(epochs=30, lr=3e-3, loss="knorm", log_every=-1))
    assert res["protocol"]["loss"] == "knorm"
    assert np.isfinite(res["val"]["energy_gap_rel"])
    assert train_loss() < 0.8 * before
    for bad in (dict(loss="knorm", anchor_mode="balanced"), dict(loss="energy")):
        with pytest.raises(ValueError):
            train_supervised(model, archs, archs[:1],
                             SupervisedConfig(epochs=1, log_every=-1, **bad))


# ---------------------------------------------------------------- the E8 run --
def _cfg(tmp, name, data, **e8):
    cfg = {"data": {"dir": str(data), "n": 12, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "asis"},
           "split": {"n_val": 4, "seed": 1}, "model": MODEL,
           "sup": {"epochs": 1, "lr": 1e-3}, "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {"e8": {"enabled": True, "budgets": [2, 4], "pool_sizes": [4],
                                  "seeds": 2, "ar_epochs": 2, "sup_epochs": 1,
                                  "include_mgn": True, "ar_only": True, **e8}},
           "device": "cpu", "workers": 1, "tf32": False, "prereg_guard": False,
           "out": str(tmp / name / "report.json")}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(cfg))
    return p


GRID = {"ar_only": False, "include_anchor": False, "include_ar_ft": False,
        "include_knorm": True, "mgn_budgets": [4]}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cm2d")
    data = generate_synthetic_dataset(tmp / "corpus", n=12, seed=3, labelled="all")
    src = run_config(_cfg(tmp, "A", data))                      # an E1-like label-free run
    reuse = {"report": str(tmp / "A" / "report.json"),
             "states_dir": str(tmp / "A" / "e8_states"), "supervised_grid": True}
    rep = run_config(_cfg(tmp, "C", data, reuse_from=reuse, **GRID))
    return tmp, data, src, rep, reuse


def test_the_grid_trains_beside_the_reused_label_free_states(world):
    tmp, _, src, rep, _ = world
    e8 = rep["results"]["e8"]
    cells = e8["metrics"]["cells"]
    assert set(cells) == {"ar", "labels", "labels_knorm", "mgn", "zero", "scale_aware_poly",
                          "knn_field"}
    assert set(cells["labels"]) == set(cells["labels_knorm"]) == {2, 4} or \
        set(cells["labels"]) == set(cells["labels_knorm"]) == {"2", "4"}
    assert len(cells["mgn"]) == 1                                  # its budget subset
    # the label-free rows are the source run's states, evaluated, not trained
    d9 = e8["metrics"]["d9_restart"]
    want = src["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    assert {k: v["sha256"] for k, v in d9["ar_states"].items()} == \
        {k: v["sha256"] for k, v in want.items()}
    assert all(v["reused"] for v in d9["ar_states"].values())
    ar_c = cells["ar"][4] if 4 in cells["ar"] else cells["ar"]["4"]
    ar_a = src["results"]["e8"]["metrics"]["cells"]["ar"]
    ar_a = ar_a[4] if 4 in ar_a else ar_a["4"]
    for s in range(2):
        assert ar_c["per_seed_eval"][s]["per_instance"] == ar_a["per_seed_eval"][s]["per_instance"]
    assert rep["reuse_from"]["supervised_grid"] is True
    assert not list((tmp / "C").glob("e8_states/ar_*.pt"))       # nothing label-free trained
    # the largest-budget states of the three supervised rows are kept
    kept = sorted(p.name for p in (tmp / "C" / "e8_states").glob("*.pt"))
    assert kept == ["labels_b4_s0.pt", "labels_b4_s1.pt", "labels_knorm_b4_s0.pt",
                    "labels_knorm_b4_s1.pt", "mgn_b4_s0.pt", "mgn_b4_s1.pt"]
    proto = e8["protocol"]
    assert proto["include_knorm"] is True and proto["include_anchor"] is False
    assert proto["eval_only"] is True


def test_the_stiffness_norm_row_trains_with_its_own_loss(world):
    tmp, _, _, rep, _ = world
    import pickle

    cache = tmp / "C" / "e8_states" / "unit_cache"
    knorm = pickle.loads((cache / "labels_knorm_b4_s0.pkl").read_bytes())
    labels = pickle.loads((cache / "labels_b4_s0.pkl").read_bytes())
    assert knorm["val"]["per_instance"] != labels["val"]["per_instance"]
    out = (tmp / "C" / "report.json").read_text()
    assert "relative stiffness" in out or "||u - U*||_K" in out


def test_the_planned_steps_are_exact_for_the_new_configurations(world):
    tmp, _, _, rep, _ = world
    # labels and labels_knorm at budgets 2 and 4, the graph network at 4; 1 epoch; 2 seeds
    assert rep["planned_steps"]["e8"] == 2 * (2 * (2 + 4) + 4)
    # an earlier-style configuration is counted as before
    old = {"experiments": {"e8": {"enabled": True, "budgets": [2, 4], "pool_sizes": [4],
                                  "seeds": 2, "ar_epochs": 2, "sup_epochs": 1,
                                  "include_mgn": True, "mgn_budgets": [4]}}}
    assert count_steps(old)["e8"] == 2 * (4 * 2 + 4 * (2 + 4))


def test_reuse_refusals(world, tmp_path):
    tmp, data, _, _, reuse = world
    # a supervised grid without the supervised_grid flag, or with an AR->FT row
    plain = {k: v for k, v in reuse.items() if k != "supervised_grid"}
    with pytest.raises(SystemExit, match="supervised_grid"):
        run_config(_cfg(tmp_path, "X", data, reuse_from=plain, **GRID))
    with pytest.raises(SystemExit, match="include_ar_ft"):
        run_config(_cfg(tmp_path, "Y", data, reuse_from=reuse,
                        **dict(GRID, include_ar_ft=True)))
    # the grid's own keys may differ from the source run; an AR key may not
    cfg = json.loads(_cfg(tmp_path, "Z", data, reuse_from=reuse, **GRID).read_text())
    assert verify_reuse(cfg, reuse)["provenance"]["supervised_grid"] is True
    for path, value in ((("experiments", "e8", "ar_epochs"), 3), (("pretrain", "lr"), 2e-3),
                        (("experiments", "e8", "pool_sizes"), [5])):
        bad = json.loads(json.dumps(cfg))
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
        with pytest.raises(ValueError, match="differs"):
            verify_reuse(bad, reuse)
    # without the flag, the grid's keys count as differences (wp9's rule, unchanged)
    with pytest.raises(ValueError, match="differs"):
        verify_reuse(cfg, plain)


def test_a_default_configuration_gains_nothing(world):
    tmp, _, src, _, _ = world
    proto = src["results"]["e8"]["protocol"]
    assert "include_knorm" not in proto and "include_anchor" not in proto
    assert "supervised_grid" not in json.dumps(src)


# ------------------------------------------- what PREREG_CM2D Sec. 2 claims --
def test_the_trainer_minimises_the_mean_square_root_of_the_relative_energy_gap(monkeypatch):
    """The loss the trainer computes, step by step, is mean_j g_j^(1/2) of the
    prediction it is given (single precision)."""
    import fejepa.train.supervised as sup
    from fejepa.metrics import energy_gap_rel
    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa

    rng = np.random.default_rng(6)
    archs = [synthetic_instance(rng, labelled=True) for _ in range(3)]
    calls = []
    orig = sup._knorm_loss

    def spy(anchor, u, u_star, ustar_k):
        loss = orig(anchor, u, u_star, ustar_k)
        calls.append((u.detach().double().numpy(), u_star.double().numpy(), float(loss)))
        return loss

    monkeypatch.setattr(sup, "_knorm_loss", spy)
    torch.manual_seed(0)
    model = build_fejepa(FEJEPAConfig.from_dict(MODEL))
    train_supervised(model, archs, archs[:1], SupervisedConfig(epochs=2, lr=3e-3, loss="knorm",
                                                               log_every=-1))
    assert len(calls) == 2 * len(archs)
    for U, Us, loss in calls:
        a = next(x for x in archs if np.allclose(x.U_star, Us, rtol=1e-6, atol=1e-12))
        want = float(np.mean(np.sqrt(energy_gap_rel(U, a))))
        assert np.isclose(loss, want, rtol=1e-4), (loss, want)


def test_a_stiffness_norm_training_resumes_from_its_epoch_checkpoint_exactly(tmp_path):
    """RUNBOOK_CMAME B3's usual case for the new row: a stiffness-norm training
    interrupted after an epoch and resumed from its epoch checkpoint ends with
    the uninterrupted training's weights and validation values, bit for bit
    (CPU), and records the epoch it resumed from."""
    from fejepa.experiments.parallel import _build_model

    rng = np.random.default_rng(8)
    train = [synthetic_instance(rng, labelled=True) for _ in range(3)]
    val = [synthetic_instance(rng, labelled=True) for _ in range(2)]
    base = dict(epochs=3, lr=1e-3, seed=0, loss="knorm", log_every=-1)
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0}
    m_ref = _build_model(payload)
    r_ref = train_supervised(m_ref, train, val, SupervisedConfig(**base))
    ck = str(tmp_path / "knorm.ckpt")
    train_supervised(_build_model(payload), train, val,
                     SupervisedConfig(**base, ckpt_path=ck, stop_after_epoch=1))
    m = _build_model(payload)
    r = train_supervised(m, train, val, SupervisedConfig(**base, ckpt_path=ck, resume=True))
    a, b = m_ref.state_dict(), m.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert r["val"] == r_ref["val"] and r["resumed_from_epoch"] == 1


def _capture(monkeypatch, cfg, pool_files, val_files):
    """The unit payloads run_e8 builds for `cfg` (nothing trains)."""
    import fejepa.experiments.e8_regimes as e8m

    class _Stop(Exception):
        pass

    got = {}

    def fake(func, payloads, workers, label):
        got[label] = copy.deepcopy(payloads)
        if "AR" not in label:
            raise _Stop
        return [{"val": None, "state_path": p["state_path"], "reused_state": False,
                 "state_sha256": "0" * 64} for p in payloads]

    monkeypatch.setattr(e8m, "map_units", fake)
    with pytest.raises(_Stop):
        e8m.run_e8(MODEL, pool_files, val_files, cfg)
    return got


def _grid_cfg(tmp, **kw):
    return {"budgets": [2, 4], "pool_sizes": [4], "seeds": 2, "ar_epochs": 2, "sup_epochs": 1,
            "include_mgn": True, "device": "cpu", "workers": 1, "tf32": False,
            "state_dir": str(tmp / "states"), **GRID, **kw}


def test_the_knorm_and_labels_units_differ_only_in_the_loss(world, monkeypatch, tmp_path):
    from fejepa.experiments.protocol import load_split

    _, data, _, _, _ = world
    split = load_split(data, 4, 1)
    got = _capture(monkeypatch, _grid_cfg(tmp_path), split.pool_files, split.val_files)
    units = {p["tag"]: p for p in got["E8 (supervised grid)"]}
    pairs = 0
    for tag, p in units.items():
        if not tag.startswith("labels_knorm"):
            continue
        q = units[tag.replace("labels_knorm", "labels", 1)]
        strip = lambda d: {k: v for k, v in d.items()                     # noqa: E731
                           if k not in ("tag", "state_path", "sup")}
        assert strip(p) == strip(q)
        assert {k: v for k, v in p["sup"].items() if k not in ("loss", "desc")} == \
            {k: v for k, v in q["sup"].items() if k != "desc"}
        assert p["sup"]["loss"] == "knorm" and "loss" not in q["sup"]
        assert (p["state_path"] is None) == (q["state_path"] is None)
        pairs += 1
    assert pairs == 2 * 2


def test_the_supervised_grid_keys_never_reach_the_label_free_units(world, monkeypatch, tmp_path):
    """Every key that `reuse_from.supervised_grid` lets differ from the source
    run leaves the label-free units' payloads unchanged -- the reason they may
    differ (w9_eval.REUSE_SUPERVISED_KEYS)."""
    from fejepa.experiments.protocol import load_split
    from fejepa.experiments.w9_eval import REUSE_SUPERVISED_KEYS

    _, data, _, _, _ = world
    split = load_split(data, 4, 1)
    alt = {"ar_only": True, "budgets": [4], "include_mgn": False, "mgn_budgets": [2, 4],
           "include_ar_ft": True, "include_anchor": True, "include_knorm": False,
           "sup_epochs": 3, "sup_lr": 2e-3, "include_naive_baselines": False}
    base = _capture(monkeypatch, _grid_cfg(tmp_path), split.pool_files,
                    split.val_files)["E8 (AR pretrain)"]
    assert len(base) == 2
    for key in REUSE_SUPERVISED_KEYS:
        cfg = _grid_cfg(tmp_path, **{key: alt.get(key, 0.123)})
        got = _capture(monkeypatch, cfg, split.pool_files, split.val_files)
        assert got["E8 (AR pretrain)"] == base, key


def test_every_other_key_of_the_source_configuration_is_compared(world):
    """A change to any leaf outside the evaluation-only and supervised-grid
    keys is refused by the reuse check."""
    from fejepa.experiments.w9_eval import (REUSE_FREE_KEYS, REUSE_SUPERVISED_KEYS,
                                            verify_reuse)

    tmp, data, src, _, reuse = world
    cfg = json.loads((tmp / "C.json").read_text())
    verify_reuse(cfg, reuse)                                       # the run's own: accepted

    def leaves(d, path=()):
        for k, v in d.items():
            if isinstance(v, dict) and v:
                yield from leaves(v, path + (k,))
            else:
                yield path + (k,), v

    def other(v):
        if isinstance(v, bool):
            return not v
        if isinstance(v, (int, float)):
            return v + 1
        if isinstance(v, str):
            return v + "x"
        if isinstance(v, list):
            return v + v[-1:] if v else [1]
        return "x"

    n = 0
    for path, v in leaves(cfg):
        if path[0] in REUSE_FREE_KEYS or path[:2] == ("experiments", "e8") and (
                len(path) > 2 and (path[2] in REUSE_SUPERVISED_KEYS or path[2] == "reuse_from")):
            continue
        bad = json.loads(json.dumps(cfg))
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = other(v)
        with pytest.raises(ValueError, match="differs"):
            verify_reuse(bad, reuse)
        n += 1
    assert n >= 20


def test_label_free_and_supervised_transformers_share_initial_weights_and_order(monkeypatch):
    from fejepa.anchor import energy as en
    from fejepa.experiments.parallel import _build_model
    from fejepa.train.losses import AR_CONFIG
    from fejepa.train.pretrain import PretrainConfig, pretrain

    rng = np.random.default_rng(5)
    archs = [synthetic_instance(rng, labelled=True) for _ in range(3)]
    payload = {"kind": "fejepa", "model": MODEL, "seed": 7}
    m_ar, m_sup = _build_model(payload), _build_model(dict(payload))
    a, b = m_ar.state_dict(), m_sup.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    seen = []
    orig = en.AnchorCache.get

    def spy(self, arch):
        seen.append(next(i for i, x in enumerate(archs) if x is arch))
        return orig(self, arch)

    monkeypatch.setattr(en.AnchorCache, "get", spy)
    pretrain(m_ar, archs, PretrainConfig(epochs=2, lr=1e-3, seed=7, loss=AR_CONFIG, log_every=-1))
    order_ar = list(seen)
    seen.clear()
    train_supervised(m_sup, archs, archs[:1], SupervisedConfig(epochs=2, lr=1e-3, seed=7,
                                                               loss="knorm", log_every=-1))
    assert len(order_ar) == 2 * len(archs) and seen == order_ar
