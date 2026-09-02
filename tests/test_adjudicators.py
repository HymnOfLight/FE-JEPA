"""E1 / E2 adjudicators: every kill and the GO rule fire on hand-built inputs
exactly as PREREG_E1 Sec. 5 and PREREG_E2 Sec. 4 state them."""

from fejepa.analysis.adjudicate import adjudicate_e1, adjudicate_e2


def _rep(disp, egap, fine_ratio=1.2, seeds=3):
    return {"results": {"e8": {"metrics": {"cells": {"ar": {"1024": {
                "disp_rel_l2": {"mean": disp, "per_seed": [disp] * seeds},
                "energy_gap_rel": {"mean": egap, "per_seed": [egap] * seeds}}}}}},
            "p3_transfer": {"metrics": {"ar": {"fine_disp_mean": disp * fine_ratio,
                                                "inband_disp_mean": disp}}}}}


def test_e1_go_and_each_kill():
    base = _rep(0.20, 0.30)
    go = adjudicate_e1(base, _rep(0.20, 0.30), [0.1, 0.1, 0.1], [0.2, 0.15, 0.12], 0.10)
    assert go["verdict"] == "GO" and not go["K1_parity"] and not go["K2_no_effect"]
    k1 = adjudicate_e1(base, _rep(0.24, 0.30), [0.1] * 3, [0.2] * 3, 0.10)     # +20% disp
    assert k1["K1_parity"] and k1["verdict"] == "KILLED"
    k2 = adjudicate_e1(base, _rep(0.20, 0.30), [0.1] * 3, [0.1, 0.05, 0.09], 0.10)
    assert k2["K2_no_effect"] and k2["verdict"] == "KILLED"
    mixed = adjudicate_e1(base, _rep(0.20, 0.30), [0.1] * 3, [0.2, 0.05, 0.2], 0.10)
    assert mixed["verdict"] == "NO-GO"                                        # not all seeds improve
    worse_transfer = adjudicate_e1(base, _rep(0.20, 0.30, fine_ratio=1.5), [0.1] * 3, [0.2] * 3, 0.10)
    assert worse_transfer["verdict"] == "NO-GO"                              # ratio worsened > band


def test_e2_go_and_each_kill():
    base = _rep(0.20, 0.30)
    bench = {"phases": {"bottleneck512_fine": {"ms_per_step": 800.0}}}
    go = adjudicate_e2(base, _rep(0.21, 0.31), bench, 512, 0.10, 2.0, 1.0)
    assert go["verdict"] == "GO"
    k1 = adjudicate_e2(base, _rep(0.20, 0.40), bench, 512, 0.10, 2.0, 1.0)     # egap +33%
    assert k1["K1_accuracy"] and k1["verdict"] == "KILLED"
    slow = {"phases": {"bottleneck512_fine": {"ms_per_step": 2500.0}}}
    k2 = adjudicate_e2(base, _rep(0.20, 0.30), slow, 512, 0.10, 2.0, 1.0)
    assert k2["K2_speed"] and k2["verdict"] == "KILLED"
    mid = {"phases": {"bottleneck512_fine": {"ms_per_step": 1500.0}}}
    nogo = adjudicate_e2(base, _rep(0.20, 0.30), mid, 512, 0.10, 2.0, 1.0)
    assert nogo["verdict"] == "NO-GO"                                         # parity but 1-2 s
    missing = adjudicate_e2(base, _rep(0.20, 0.30), {"phases": {}}, 512, 0.10, 2.0, 1.0)
    assert missing["K2_speed"]                                                # no measurement = no case
