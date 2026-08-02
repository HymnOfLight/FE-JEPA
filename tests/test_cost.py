"""Cost model mirrors the runner arithmetic (plan Sec.9) + ledger accounting."""

from fejepa.experiments.cost import count_steps
from fejepa.fe.solve import SolveLedger


def test_count_steps_hand_check():
    cfg = {"sup": {"epochs": 10}, "pretrain": {"epochs": 5},
           "experiments": {
               "e1": {"enabled": True, "budgets": [4, 8], "seeds": 2, "epochs": 10,
                      "grid": [1.0]},
               "e8": {"enabled": True, "budgets": [4], "pool_sizes": [8], "seeds": 2,
                      "sup_epochs": 10, "ar_epochs": 5, "include_mgn": True},
               "e3": {"enabled": True, "steps": 2000, "modes": ["sigreg"],
                      "geometry_conditions": [False, True]},
           }}
    out = count_steps(cfg)
    # e1: 4 arms (none/balanced/fixed/grid_1.0) * 2 seeds * 10 epochs * (4+8) = 960
    assert out["e1"] == 4 * 2 * 10 * 12
    # e8: 2 seeds * (8*5 AR + 4 sup-arms * 10 * 4) = 2 * (40 + 160) = 400
    assert out["e8"] == 2 * (8 * 5 + 4 * 10 * 4)
    # e3: 2 conds * (1 + 1 modes) * 2000 = 8000
    assert out["e3"] == 8000
    assert out["total"] == out["e1"] + out["e8"] + out["e3"]


def test_count_steps_e5_fallback_mirrors_literal_default():
    cfg = {"sup": {"epochs": 3},          # sup epochs must NOT leak into e5 fallback
           "experiments": {"e5": {"enabled": True, "budgets": [4, 8]}}}
    assert count_steps(cfg)["e5"] == 200 * (4 + 8)


def test_ledger():
    led = SolveLedger()
    led.add("labelling-val", n=4, seconds=0.5)
    led.add("labelling-val", n=4)
    led.add("labelling-pool-prefix", n=8)
    d = led.as_dict()
    assert d["per_stage"]["labelling-val"] == 8 and d["total"] == 16
    assert d["wall_clock_s"] == 0.5
