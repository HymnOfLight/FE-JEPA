"""Tests for Gate G2 (PREREG_PHASE2 r8). Synthetic cases shaped on the 2D
deciding-run numbers, plus every kill trigger and the fails-closed paths."""

from fejepa.experiments.gate_g2 import gate_g2

BUDGETS = [16, 64, 256, 1024]


def _cell(disp, egap, vm=0.2):
    return {"disp_rel_l2": {"mean": disp}, "energy_gap_rel": {"mean": egap},
            "vm_rel_l2": {"mean": vm}}


def mk_e8(ar_disp=0.1658, ar_egap=0.0817,
          lab_disp=(0.2721, 0.1926, 0.1756, 0.1600),
          lab_egap=(6.264, 2.718, 1.257, 0.4177),
          anc_disp=(0.2236, 0.1875, 0.1742, 0.1657)):
    cells = {"ar": {1024: _cell(ar_disp, ar_egap)},
             "labels": {}, "labels_anchor": {}, "zero": {},
             "knn_field": {}, "scale_aware_poly": {}}
    knn = (0.6867, 0.4660, 0.4077, 0.4038)
    sap = (1.6562, 1.3679, 1.4121, 1.4219)
    for i, b in enumerate(BUDGETS):
        cells["labels"][b] = _cell(lab_disp[i], lab_egap[i])
        cells["labels_anchor"][b] = _cell(anc_disp[i], 0.2)
        cells["zero"][b] = _cell(1.0, 1.0)
        cells["knn_field"][b] = _cell(knn[i], 5.0)
        cells["scale_aware_poly"][b] = _cell(sap[i], 500.0)
    return {"metrics": {"cells": cells}}


def mk_e1(improvs=(0.980, 0.944, 0.906, 0.740)):
    rows = []
    for i, b in enumerate(BUDGETS):
        none_eg = 1.0
        arm = "fixed" if b < 64 else "balanced"
        rows.append({"budget": b, "arms": {
            "none": {"egap": {"mean": none_eg}},
            arm: {"egap": {"mean": none_eg * (1 - improvs[i])}}}})
    return {"metrics": {"per_budget": rows}}


def mk_p3(ratio=1.10, inband=0.10, knn=0.40, sap=1.4):
    return {"metrics": {"ar": {"inband_disp_mean": inband,
                               "fine_disp_mean": inband * ratio},
                        "naive_at_fine": {"knn_field": knn,
                                          "scale_aware_poly": sap}}}


E6 = {"metrics": {"rho_within": 0.730}}
WP6_OK = {"holds": True}


def test_go_via_b_with_transfer_unevaluated():
    """2D-shaped numbers: a true, b true, c unevaluated -> GO via (b)."""
    r = gate_g2(mk_e8(), mk_e1(), None, E6, WP6_OK)
    assert r["conditions"] == {"a": True, "b": True, "c": False}
    assert r["passed"] is True
    assert r["transfer_zone"] == "unevaluated"
    assert not r["any_kill"]


def test_kp1_parity_kill_breaks_b():
    r = gate_g2(mk_e8(ar_disp=0.20), mk_e1(), None, E6, WP6_OK)
    assert r["kills"]["KP1"] is True and r["conditions"]["b"] is False
    assert r["passed"] is False  # c unevaluated, so (b OR c) fails


def test_kp2_any_budget():
    """Advantage dips below 0.40 at one budget -> KP2, b false."""
    r = gate_g2(mk_e8(lab_egap=(6.264, 2.718, 1.257, 0.12)), mk_e1(),
                None, E6, WP6_OK)
    assert r["kills"]["KP2"] is True and r["conditions"]["b"] is False


def test_go_via_c_when_b_fails():
    """Parity gone but a strong transfer win still carries G2 through (c)."""
    r = gate_g2(mk_e8(ar_disp=0.20), mk_e1(), mk_p3(ratio=1.10), E6, WP6_OK)
    assert r["conditions"]["c"] is True and r["passed"] is True
    assert r["transfer_zone"] == "win"


def test_transfer_three_zones():
    assert gate_g2(mk_e8(), mk_e1(), mk_p3(ratio=1.20), E6, WP6_OK)[
        "transfer_zone"] == "win"
    weak = gate_g2(mk_e8(), mk_e1(), mk_p3(ratio=1.35), E6, WP6_OK)
    assert weak["transfer_zone"] == "weakened" and weak["conditions"]["c"] is False \
        and weak["kills"]["KP4"] is False
    dead = gate_g2(mk_e8(), mk_e1(), mk_p3(ratio=1.60), E6, WP6_OK)
    assert dead["transfer_zone"] == "retired" and dead["kills"]["KP4"] is True


def test_kp4_naive_wins_at_fine():
    r = gate_g2(mk_e8(), mk_e1(), mk_p3(ratio=1.10, knn=0.05), E6, WP6_OK)
    assert r["kills"]["KP4"] is True and r["conditions"]["c"] is False


def test_kp3_all_budgets_required():
    dead = gate_g2(mk_e8(), mk_e1(improvs=(0.1, 0.2, 0.05, 0.15)),
                   None, E6, WP6_OK)
    assert dead["kills"]["KP3"] is True
    alive = gate_g2(mk_e8(), mk_e1(improvs=(0.1, 0.2, 0.30, 0.15)),
                    None, E6, WP6_OK)
    assert alive["kills"]["KP3"] is False


def test_kp5_and_kp6():
    r = gate_g2(mk_e8(), mk_e1(), None, {"metrics": {"rho_within": 0.25}},
                {"holds": False})
    assert r["kills"]["KP5"] is True and r["kills"]["KP6"] is True


def test_fails_closed_everywhere():
    r = gate_g2(None, None, None, None, None)
    assert r["conditions"] == {"a": False, "b": False, "c": False}
    assert r["passed"] is False
    assert "fails closed" in r["reasons"]["a_sanity"]


def test_sanity_binds_every_budget():
    """Anchored misses 3x at b=16 only -> (a) false, G2 false despite b true."""
    r = gate_g2(mk_e8(anc_disp=(0.40, 0.1875, 0.1742, 0.1657)),
                mk_e1(), None, E6, WP6_OK)
    assert r["conditions"]["a"] is False and r["passed"] is False


def test_kp5_consumes_theory_kill_list():
    """R1 fix: the wp6 result's own kill list is authoritative for KP5."""
    tripped = {"kills": [{"name": "C5 numeric falsification pass",
                          "triggered": True}]}
    r = gate_g2(mk_e8(), mk_e1(), None, E6, tripped)
    assert r["kills"]["KP5"] is True
    clean = {"kills": [{"name": "C5", "triggered": False}]}
    assert gate_g2(mk_e8(), mk_e1(), None, E6, clean)["kills"]["KP5"] is False
