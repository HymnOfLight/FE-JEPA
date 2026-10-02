"""wp9 Stage 0c (PREREG_W9 r3, Sec. 7-8): the readings beside the verdicts --
medians, the Welch interval, the instance-resampling interval, R's growth,
the uninformative case of H2 -- and the adjudicating code compared with the
stamp. Readings only: no criterion depends on them."""

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fejepa.analysis import adjudicate_w9 as adj

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell(arrays: dict) -> dict:
    """An E8 cell: per seed, the metric means and per-instance arrays."""
    n = len(next(iter(arrays.values())))
    return {"1024": {"per_seed_eval": [
        {**{m: float(np.mean(v[k])) for m, v in arrays.items()},
         "per_instance": {m: list(map(float, v[k])) for m, v in arrays.items()}}
        for k in range(n)]}}


def _report(val: dict, holdouts: dict) -> dict:
    return {"results": {"e8": {"metrics": {"cells": {"ar": _cell(val)},
                                           "holdouts": {k: _cell(v) for k, v in holdouts.items()}}}}}


def test_welch_interval():
    from scipy.stats import t

    b, a = [1.0, 1.1, 0.9], [0.8, 0.85, 0.75]
    w = adj.welch_interval(b, a)
    vb, va = np.var(b, ddof=1) / 3, np.var(a, ddof=1) / 3
    df = (vb + va) ** 2 / (vb ** 2 / 2 + va ** 2 / 2)
    half = t.ppf(0.975, df) * np.sqrt(vb + va) / np.mean(b)
    assert np.isclose(w["rel_change"], -0.2) and np.isclose(w["df"], df)
    assert np.isclose(w["low"], -0.2 - half) and np.isclose(w["high"], -0.2 + half)
    w = adj.welch_interval([1.0, 1.0, 1.0], [0.9, 0.9, 0.9])
    assert w["low"] == w["high"] == pytest.approx(-0.1) and w["df"] is None
    with pytest.raises(ValueError, match="non-finite"):
        adj.welch_interval([1.0, float("nan"), 1.0], [1.0, 1.0, 1.0])


def test_instance_resampling_interval():
    rng = np.random.default_rng(0)
    base = [rng.lognormal(-3.3, 0.8, 256) for _ in range(3)]
    new = [0.8 * x for x in base]                         # 20 % lower on every instance
    b = adj.instance_bootstrap(base, new)
    assert b == adj.instance_bootstrap(base, new)         # deterministic (fixed seed)
    assert b["low"] == pytest.approx(-0.2) and b["high"] == pytest.approx(-0.2)
    assert b["resamples"] == adj.BOOT_N
    blown = [x.copy() for x in base]
    blown[1][233] = 0.76                                  # one instance, one seed
    b = adj.instance_bootstrap(base, blown)
    assert 0 <= b["low"] < b["high"]                      # it can only raise the new arm
    r = adj.instance_bootstrap(base, new, base_den=base, new_den=new)     # ratio: unchanged
    assert r["low"] == pytest.approx(0.0, abs=1e-12) and r["high"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="non-finite"):
        adj.instance_bootstrap(base, [x * np.nan for x in base])
    with pytest.raises(ValueError, match="instance counts"):
        adj.instance_bootstrap(base, [x[:100] for x in base])


def test_robustness_beside_a_comparison():
    rng = np.random.default_rng(1)
    v = [rng.lognormal(-3.3, 0.5, 64) for _ in range(3)]
    f5 = [rng.lognormal(-1.0, 0.5, 32) for _ in range(3)]
    b = _report({"energy_gap_rel": v, "disp_rel_l2": v}, {"F5": {"disp_rel_l2": f5}})
    a = _report({"energy_gap_rel": [0.5 * x for x in v], "disp_rel_l2": v},
                {"F5": {"disp_rel_l2": [0.5 * x for x in f5]}})
    r = adj.robustness(b, a, "val", "energy_gap_rel")
    assert r["medians"]["lower"] and r["medians"]["rel_change"] == pytest.approx(-0.5)
    assert r["welch_95"]["rel_change"] == pytest.approx(-0.5)
    assert r["instance_resampling_95"]["high"] == pytest.approx(-0.5)
    r = adj.robustness(b, a, "F5", "disp_rel_l2", ratio=True)
    assert r["medians"]["rel_change"] == pytest.approx(-0.5)        # F5 halved, in band equal
    assert r["welch_95"]["rel_change"] == pytest.approx(-0.5)
    assert r["instance_resampling_95"]["low"] == pytest.approx(-0.5)


def test_remesh_growth():
    hs = (0.12, 0.085, 0.05, 0.035, 0.025)
    recs = [{"geometry": g, "target_h": h} for g in range(2) for h in hs]
    growth = {0.12: 2.0, 0.085: 1.6, 0.05: 1.25, 0.035: 1.1, 0.025: 1.0}
    disp = [np.array([0.1 * (g + 1) / growth[r["target_h"]] for g, r in
                      ((r["geometry"], r) for r in recs)]) * (1 + 0.1 * k) for k in range(3)]
    rep = _report({"disp_rel_l2": disp}, {"R": {"disp_rel_l2": disp}})
    g = adj.remesh_growth(rep, {"instances": recs})
    assert set(g) == {"0.12", "0.085", "0.05"}                     # the training range only
    for h, want in (("0.12", 2.0), ("0.085", 1.6), ("0.05", 1.25)):
        assert g[h] == pytest.approx([want] * 3)
    with pytest.raises(ValueError, match="differ in length"):
        adj.remesh_growth(rep, {"instances": recs[:-1]})


def test_uninformative_case_and_rule1_lines():
    rules = _module("w9_session1_decisions")
    assert (adj.RULE1_RATIO_MIN, adj.RULE1_CSTAR_MIN) == (rules.S_RATIO_MIN, rules.S_CSTAR_MIN)

    def tab(ratio, cstar):
        return {"ratio_F5_over_val_disp_of_seed_means": ratio,
                "F5": {"c_star_median": {"seed_mean": cstar}}}
    assert adj._uninformative(tab(2.0, 1.3)) is None
    assert "F5 / in-band ratio 1.2 < 1.5" in adj._uninformative(tab(1.2, 1.3))
    assert "median c* on F5 1.05 < 1.1" in adj._uninformative(tab(2.0, 1.05))


def test_adjudicating_code_against_the_stamp():
    cli = _module("adjudicate_w9")
    res = cli.code_against("HEAD")
    if not res["compared"]:
        pytest.skip(f"no git checkout ({res.get('note')})")
    for p in cli.ADJUDICATING_FILES:
        clean = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", p]
                               ).returncode == 0
        assert (p in res["differs"]) == (not clean), p
    none = cli.code_against("no-such-ref-w9")
    assert none["compared"] is False and "no no-such-ref-w9" in none["note"]
