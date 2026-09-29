"""E1 / E2 adjudicators: every kill and the GO rule fire on hand-built inputs
exactly as PREREG_E1 Sec. 5 and PREREG_E2 Sec. 4 state them; Stage 1.28 adds
the refusals (wrong baseline, incomparable reports, set-up-inclusive bench)."""

import pytest

from fejepa.analysis.adjudicate import PHASE2B_CONFIG_SHA256, adjudicate_e1, adjudicate_e2

PHASE2_D14 = "e3bdd1e8778d063ff024b30354e1cd95b492953b2baf8dd25bc0aac863ce04da"


def _rep(disp, egap, fine_ratio=1.2, seeds=3, kind=None, n_tokens=None,
         config_sha=PHASE2B_CONFIG_SHA256, loss_spec=None, manifest="m-inband",
         fine_per_seed=None, ar_epochs=200):
    model = {"dim": 256, "depth": 8, "heads": 8, "scale_decode": True}
    if kind:
        model.update(kind=kind, n_tokens=n_tokens)
    pre = {"epochs": 200, "lr": 1e-3}
    if loss_spec:
        pre["loss_spec"] = loss_spec
    fine = fine_per_seed or [disp * fine_ratio] * seeds
    return {"config": {"model": model, "pretrain": pre, "split": {"n_val": 256, "seed": 1},
                       "data": {"dir": "runs/c"}, "tf32": True,
                       "experiments": {"e8": {"pool_sizes": [1024], "ar_epochs": ar_epochs,
                                              "seeds": seeds}}},
            "provenance": {"config_sha256": config_sha, "seeds": list(range(seeds)),
                           "datasets": [{"dir": "runs/c", "manifest_sha256": manifest}]},
            "results": {"e8": {"metrics": {"cells": {"ar": {"1024": {
                "disp_rel_l2": {"mean": disp, "per_seed": [disp] * seeds},
                "energy_gap_rel": {"mean": egap, "per_seed": [egap] * seeds}}}}}},
                "p3_transfer": {"metrics": {"ar": {
                    "fine_disp_mean": sum(fine) / len(fine), "inband_disp_mean": disp,
                    "fine": {"disp_rel_l2": {"per_seed": fine}}}}}}}


def _e2(disp, egap, m=512, **kw):
    return _rep(disp, egap, kind="bottleneck", n_tokens=m, config_sha="e2cfg", **kw)


def _bench(ms, m=512, timing="differential"):
    return {"phases": {f"bottleneck{m}_fine": {"ms_per_step": ms, "timing": timing}}}


SHAPED = {"reg_mode": "sigreg_ep_head", "lambda_reg": 0.1}


def test_e1_go_and_each_kill():
    base = _rep(0.20, 0.30)
    sh = lambda d, e, **kw: _rep(d, e, loss_spec=SHAPED, **kw)  # noqa: E731
    go = adjudicate_e1(base, sh(0.20, 0.30), [0.1, 0.1, 0.1], [0.2, 0.15, 0.13], 0.10)
    assert go["verdict"] == "GO" and not go["K1_parity"] and not go["K2_no_effect"]
    # noise-level improvements at every seed must NOT pass: effect floor
    tiny = adjudicate_e1(base, sh(0.20, 0.30), [0.1, 0.1, 0.1], [0.101, 0.102, 0.1005], 0.10)
    assert tiny["verdict"] == "NO-GO" and not tiny["S_above_floor_all_seeds"]
    # a floor scaled by the AR arm's seed spread: sample SD 0.05 -> floor 0.10
    spread = adjudicate_e1(base, sh(0.20, 0.30), [0.05, 0.10, 0.15], [0.10, 0.15, 0.20], 0.10)
    assert spread["S_effect_floor"] == pytest.approx(0.10) and spread["verdict"] == "NO-GO"
    k1 = adjudicate_e1(base, sh(0.24, 0.30), [0.1] * 3, [0.2] * 3, 0.10)     # +20% disp
    assert k1["K1_parity"] and k1["verdict"] == "KILLED"
    k2 = adjudicate_e1(base, sh(0.20, 0.30), [0.1] * 3, [0.1, 0.05, 0.09], 0.10)
    assert k2["K2_no_effect"] and k2["verdict"] == "KILLED"
    mixed = adjudicate_e1(base, sh(0.20, 0.30), [0.1] * 3, [0.2, 0.05, 0.2], 0.10)
    assert mixed["verdict"] == "NO-GO"                                        # not all seeds improve
    worse_transfer = adjudicate_e1(base, sh(0.20, 0.30, fine_ratio=1.5), [0.1] * 3, [0.2] * 3, 0.10)
    assert worse_transfer["verdict"] == "NO-GO"                              # ratio worsened > band


def test_e1_refuses_incomparable_reports():
    base = _rep(0.20, 0.30)
    for bad in (_rep(0.20, 0.30, loss_spec=SHAPED, manifest="other-corpus"),
                _rep(0.20, 0.30, loss_spec=SHAPED, ar_epochs=100)):
        with pytest.raises(ValueError, match="differ beyond the loss"):
            adjudicate_e1(base, bad, [0.1] * 3, [0.2] * 3, 0.10)


def test_e2_go_and_each_kill():
    base = _rep(0.20, 0.30)
    go = adjudicate_e2(base, _e2(0.21, 0.31), _bench(800.0), 512, 0.10, 2.0, 1.0)
    assert go["verdict"] == "GO"
    k1 = adjudicate_e2(base, _e2(0.20, 0.40), _bench(800.0), 512, 0.10, 2.0, 1.0)  # egap +33%
    assert k1["K1_accuracy"] and k1["verdict"] == "KILLED"
    k2 = adjudicate_e2(base, _e2(0.20, 0.30), _bench(2500.0), 512, 0.10, 2.0, 1.0)
    assert k2["K2_speed"] and k2["verdict"] == "KILLED"
    nogo = adjudicate_e2(base, _e2(0.20, 0.30), _bench(1500.0), 512, 0.10, 2.0, 1.0)
    assert nogo["verdict"] == "NO-GO"                                         # parity but 1-2 s
    missing = adjudicate_e2(base, _e2(0.20, 0.30), {"phases": {}}, 512, 0.10, 2.0, 1.0)
    assert missing["K2_speed"]                                                # no measurement = no case


def test_e2_fine_displacement_is_compared_at_the_seed_median():
    """PREREG_E2 Sec. 4: both K1 quantities at the seed median. Phase-2b's
    fine per-seed values (0.281, 0.287, 0.207) have median 0.281, mean 0.258."""
    base = _rep(0.03, 0.0104, fine_per_seed=[0.2811, 0.2874, 0.2066])
    e2 = _e2(0.03, 0.0104, fine_per_seed=[0.30, 0.30, 0.30])   # +6.7% vs median, +16% vs mean
    r = adjudicate_e2(base, e2, _bench(800.0), 512, 0.10, 2.0, 1.0)
    assert r["fine_disp_base"] == pytest.approx(0.2811)
    assert not r["K1_accuracy"] and r["verdict"] == "GO"


def test_e2_refuses_the_d14_baseline_wrong_arm_incomparable_and_setup_timing():
    good = _rep(0.20, 0.30)
    with pytest.raises(ValueError, match="D14-invalid"):
        adjudicate_e2(_rep(0.99964, 0.99928, config_sha=PHASE2_D14), _e2(0.2, 0.3),
                      _bench(800.0), 512)
    with pytest.raises(ValueError, match="expected bottleneck M=1024"):
        adjudicate_e2(good, _e2(0.2, 0.3, m=512), _bench(800.0, m=1024), 1024)
    with pytest.raises(ValueError, match="differ beyond the architecture"):
        adjudicate_e2(good, _e2(0.2, 0.3, manifest="regenerated"), _bench(800.0), 512)
    with pytest.raises(ValueError, match="setup-free"):
        adjudicate_e2(good, _e2(0.2, 0.3), _bench(800.0, timing=None), 512)


def test_e1_refuses_nonfinite_separation():
    base = _rep(0.20, 0.30)
    with pytest.raises(ValueError, match="not finite"):
        adjudicate_e1(base, _rep(0.20, 0.30, loss_spec=SHAPED), [0.1, float("nan"), 0.1],
                      [0.2, 0.2, 0.2], 0.10)


def test_e2_reports_seed_spreads():
    r = adjudicate_e2(_rep(0.20, 0.30), _e2(0.21, 0.31), _bench(800.0), 512, 0.10, 2.0, 1.0)
    assert "egap_seed_sd_base" in r and "egap_seed_sd_e2" in r
