"""cmame-paper Stage 11 (PREREG_HOLDOUT Sec. 5-6): the rules of the eight claims
and the secondary readings.

Applied to the manuscript's own validation arrays (the Phase-2b and CM2D
reports' per-instance arrays, and the per-load gaps of the field export and
of the spectral export), with the configuration's thresholds, every rule
reads REPLICATED with the values the manuscript quotes, and the zero-field
counts are the manuscript's; on constructed arrays each rule decides at its
boundary as written (a tie, a gap of exactly 1, a mean gap of exactly 1,
exactly half, the content tolerance); a state that failed its content check
leaves the claims that read it NOT EVALUATED and keeps its own per-seed
readings outside the counts and ratios; a non-finite value leaves a claim
NOT EVALUATED too, except that in H1, H2a, H2b, H2c and H3 a non-finite value
of the new row alone is a divergence (NOT REPLICATED), as in PREREG_CM2D;
the secondary readings compare the new row with its reference and sit beside
the validation arrays' own; and the resampling record that the
pre-registration quotes is what scripts/holdout_resample.py gives."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from fejepa.analysis.adjudicate_holdout import (CLAIMS, NOT_EVALUATED, NOT_REPLICATED,
                                                REPLICATED, Arrays, adjudicate_sets,
                                                arrays_from_report, claim_d2, claim_halved,
                                                claim_t1, claim_t2, claim_t3,
                                                mean_gap_above_one, pair_shares, reading_h3,
                                                set_readings)
from fejepa.analysis.holdout import METRICS, content_check, worse_than_zero

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = json.loads((ROOT / "configs" / "holdout_v1.json").read_text())["criteria"]
"""The configuration's thresholds (tests/test_holdout_config.py pins them)."""


def _export(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _macros() -> dict:
    text = (ROOT / "paper/cmame/generated/numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text))


@pytest.fixture(scope="module")
def manuscript():
    r3 = json.loads((ROOT / "records/wp8/e2/baseline/report_phase2b.json").read_text())
    r2 = json.loads((ROOT / "records/cmame/cm2d/return/report.json").read_text())
    a3 = arrays_from_report(r3, ["ar", "labels", "mgn"],
                            _export(ROOT / "records/cmame/timing/fields/energies_val.npz"),
                            "arms")
    a2 = arrays_from_report(r2, ["ar", "labels", "labels_knorm", "mgn"],
                            _export(ROOT / "records/cmame/spectra/export/spectra_val.npz"),
                            "rows")
    return a3, a2


def test_rules_read_replicated_on_the_manuscripts_own_arrays(manuscript):
    a3, a2 = manuscript
    res = adjudicate_sets(a3, a2, CRITERIA)
    assert list(res["claims"]) == list(CLAIMS)
    assert {c: v["verdict"] for c, v in res["claims"].items()} == {c: REPLICATED for c in CLAIMS}
    assert res["tally"] == {REPLICATED: 8, NOT_REPLICATED: 0, NOT_EVALUATED: 0}
    c = res["claims"]
    # the values the manuscript quotes (paper/cmame/generated/numbers.tex)
    shares = c["T2"]["disp_lower_share_per_seed"]
    assert round(min(shares) * 100, 1) == 84.4 and round(max(shares) * 100, 1) == 92.6
    assert round(c["T2"]["vm_higher_share_pooled"] * 768) == 767 and c["T2"]["pairs"] == 768
    assert c["T3"]["per_row"]["labels"]["worse"] == 126
    assert c["T3"]["per_row"]["mgn"]["worse"] == 106
    assert all(c["T3"]["per_row"][r]["sign_mismatch"] == 0 for r in ("labels", "mgn"))
    assert round(-100 * c["H1"]["rel_change"], 1) == 71.3
    assert round(-100 * c["H2a"]["rel_change"], 1) == 78.2
    assert round(-100 * c["H2b"]["rel_change"], 1) == 54.9
    assert c["H2c"]["rel_change"] == pytest.approx(c["H2b"]["rel_change"], rel=1e-12)
    assert res["readings"]["H3"]["reading"] == "no difference shown (within the guard)"
    d2 = c["D2"]["per_metric"]
    assert round(d2["disp_rel_l2"]["label_free_mean"], 4) == 0.0634
    assert round(d2["disp_rel_l2"]["supervised_mean"], 4) == 0.0505
    assert all(all(v["holds_in_seed"]) and v["holds_in_mean"] for v in d2.values())
    for m in METRICS:
        assert all(c["T1"]["per_metric"][m]["better_in_seed"])


def test_secondary_readings_beside_the_validation_values(manuscript):
    a3, a2 = manuscript
    res = adjudicate_sets(a3, a2, CRITERIA, h3d_val=a3, h2d_val=a2)
    mac = _macros()
    for name, A in (("h3d", a3), ("h2d", a2)):
        sec = res["secondary"][name]
        assert set(sec) == {"holdout", "validation", "ratio_holdout_over_validation"}
        assert sec["holdout"] == sec["validation"]               # the same arrays here
        assert all(v == pytest.approx(1.0) for r in sec["ratio_holdout_over_validation"].values()
                   for v in r.values())
    v3, v2 = res["secondary"]["h3d"]["validation"], res["secondary"]["h2d"]["validation"]
    # the manuscript's counts: predictions with a mean gap above one, load cases worse
    assert v3["mean_gap_above_one"]["labels"]["total"] == int(mac["numThreeLabWorse"])
    assert v3["mean_gap_above_one"]["mgn"]["total"] == int(mac["numThreeMgnWorse"])
    assert v3["mean_gap_above_one"]["ar"]["total"] == 0
    assert v2["mean_gap_above_one"]["mgn"]["total"] == int(mac["numCmMgnWorse"])
    assert all(v2["mean_gap_above_one"][r]["total"] == 0 for r in ("ar", "labels",
                                                                     "labels_knorm"))
    for row, key in (("labels", "numFieldLoadWorseLab"), ("mgn", "numFieldLoadWorseMgn"),
                     ("ar", "numThreeFreeLoadWorse")):
        w = v3["worse_than_zero"][row]
        assert w["all_seeds"]["worse"] == int(mac[key]) == sum(w[f"s{s}"]["worse"]
                                                                 for s in (0, 1, 2))
    for row, key in (("mgn", "numSpecLoadWorseMgn"), ("ar", "numSpecLoadWorseFree"),
                     ("labels", "numSpecLoadWorseLab"), ("labels_knorm", "numSpecLoadWorseKnorm")):
        assert v2["worse_than_zero"][row]["all_seeds"]["worse"] == int(mac[key])
    assert set(k for k in v2 if k.startswith("guard_")) == {
        "guard_ar_vs_labels", "guard_labels_knorm_vs_labels", "guard_mgn_vs_labels",
        "guard_labels_knorm_vs_ar", "guard_mgn_vs_ar"}
    assert res["validation_reference"]["claims"] == {c: REPLICATED for c in CLAIMS}
    assert json.dumps(res, allow_nan=False, default=float)


def _arrays(values: dict, rows=("ar", "labels", "mgn"), seeds=(0, 1, 2), n=10,
            gap=None, pi=None, used=None) -> Arrays:
    V = np.zeros((len(rows), len(seeds), n, len(METRICS)))
    for (r, m), v in values.items():
        V[rows.index(r), :, :, METRICS.index(m)] = v
    G = np.full((len(rows), len(seeds), n, 4), 0.5) if gap is None else gap
    P = np.full(G.shape, -1.0) if pi is None else pi
    d = {"rows": np.array(rows), "seeds": np.array(seeds), "metrics": np.array(METRICS),
         "values": V, "gap_load": G, "pi_load": P}
    return Arrays(d, used or {(r, s): True for r in rows for s in seeds})


def _t1_values(ar=1.0, lab=2.0):
    out = {}
    for m in METRICS:
        hi = m == "crit_recall"
        out[("ar", m)] = lab if hi else ar
        out[("labels", m)] = ar if hi else lab
    return out


def test_t1_needs_every_metric_and_seed_and_a_strict_difference():
    assert claim_t1(_arrays(_t1_values()))["verdict"] == REPLICATED
    A = _arrays(_t1_values())
    A.values[0, 1, :, METRICS.index("vm_rel_l2")] = 2.0          # a tie in one seed
    assert claim_t1(A)["verdict"] == NOT_REPLICATED
    A = _arrays(_t1_values())
    A.values[0, :, :, METRICS.index("crit_recall")] = 0.5         # recall: higher is better
    assert claim_t1(A)["verdict"] == NOT_REPLICATED
    A = _arrays(_t1_values())
    A.values[0, 2, :, METRICS.index("disp_rel_l2")] = 2.5         # one seed worse, mean better
    assert claim_t1(A)["verdict"] == NOT_REPLICATED
    A = _arrays(_t1_values())
    A.values[0, 0, 3, METRICS.index("peak_vm_rel_err")] = np.nan  # non-finite: not evaluated
    assert claim_t1(A)["verdict"] == NOT_EVALUATED


def test_t2_boundaries():
    n = 20

    def make(lower_share, higher_share):
        A = _arrays({}, n=n)
        ri, rm = 0, 2
        k = int(round(lower_share * n))
        A.values[ri, :, :, METRICS.index("disp_rel_l2")] = 1.0
        A.values[rm, :, :, METRICS.index("disp_rel_l2")] = 2.0
        A.values[rm, :, :k, METRICS.index("disp_rel_l2")] = 0.5
        A.values[ri, :, :, METRICS.index("vm_rel_l2")] = 1.0
        A.values[rm, :, :, METRICS.index("vm_rel_l2")] = 0.5
        flat = A.values[rm, :, :, METRICS.index("vm_rel_l2")].reshape(-1)
        flat[:int(round(higher_share * flat.size))] = 2.0
        A.values[rm, :, :, METRICS.index("vm_rel_l2")] = flat.reshape(3, n)
        return A

    assert claim_t2(make(0.55, 0.95), 0.5, 0.95)["verdict"] == REPLICATED  # >= 95% is enough
    assert claim_t2(make(0.5, 1.0), 0.5, 0.95)["verdict"] == NOT_REPLICATED  # half is not most
    assert claim_t2(make(0.55, 56 / 60), 0.5, 0.95)["verdict"] == NOT_REPLICATED  # 93.3%
    A = make(0.55, 1.0)
    A.values[2, 1, 0, METRICS.index("vm_rel_l2")] = np.inf
    assert claim_t2(A, 0.5, 0.95)["verdict"] == NOT_EVALUATED


def test_t3_needs_a_worse_load_case_in_each_network_and_the_sign():
    G = np.full((3, 3, 4, 4), 0.5)
    P = np.full(G.shape, -1.0)
    G[1, 0, 0, 0], P[1, 0, 0, 0] = 1.5, 0.2                       # labels
    G[2, 2, 3, 1], P[2, 2, 3, 1] = 3.0, 1.0                       # mgn
    assert claim_t3(_arrays({}, n=4, gap=G, pi=P))["verdict"] == REPLICATED
    G2 = G.copy()
    G2[2, 2, 3, 1] = 0.5
    P2 = P.copy()
    P2[2, 2, 3, 1] = -1.0
    assert claim_t3(_arrays({}, n=4, gap=G2, pi=P2))["verdict"] == NOT_REPLICATED
    P3 = P.copy()
    P3[1, 0, 0, 0] = -0.1                                         # the sign disagrees
    r = claim_t3(_arrays({}, n=4, gap=G, pi=P3))
    assert r["verdict"] == NOT_REPLICATED and r["per_row"]["labels"]["sign_mismatch"] == 1
    G4 = G.copy()
    G4[1, 1, 1, 1] = np.nan
    assert claim_t3(_arrays({}, n=4, gap=G4, pi=P))["verdict"] == NOT_EVALUATED


ROWS2 = ("ar", "labels", "labels_knorm", "mgn")


def _vals2():
    vals = {}
    for m in METRICS:
        hi = m == "crit_recall"
        vals[("ar", m)] = 0.9 if hi else 0.03
        vals[("labels", m)] = 0.6 if hi else 0.13
        vals[("labels_knorm", m)] = 0.8 if hi else 0.03
        vals[("mgn", m)] = 0.5 if hi else 0.5
    vals[("ar", "disp_rel_l2")], vals[("labels", "disp_rel_l2")] = 0.063, 0.050
    return vals


def test_guard_claims_d2_and_h3():
    vals = _vals2()
    A = _arrays(vals, rows=ROWS2)
    A.values[:, 1] *= 1.02                                        # seed spread
    res = adjudicate_sets(None, A, CRITERIA)
    assert [res["claims"][c]["verdict"] for c in ("H1", "H2a", "H2b", "H2c", "D2")] == \
        [REPLICATED] * 5
    assert res["claims"]["T1"]["verdict"] == NOT_EVALUATED       # no h3d return
    assert res["readings"]["H3"]["reading"] == "no difference shown (within the guard)"
    B = _arrays({**vals, ("ar", "disp_rel_l2"): 0.040}, rows=ROWS2)
    assert claim_d2(B)["verdict"] == NOT_REPLICATED               # displacement no longer higher
    C = _arrays({**vals, ("labels_knorm", "energy_gap_rel"): 0.02}, rows=ROWS2)
    assert reading_h3(C, 0.10, 2.0)["reading"] == \
        "the stiffness-norm transformer lower beyond the guard"
    used = {(r, s): True for r in ROWS2 for s in (0, 1, 2)}
    used[("labels_knorm", 1)] = False                             # content mismatch
    D = _arrays(vals, rows=ROWS2, used=used)
    res = adjudicate_sets(None, D, CRITERIA)
    assert res["claims"]["H1"]["verdict"] == REPLICATED
    assert all(res["claims"][c]["verdict"] == NOT_EVALUATED for c in ("H2a", "H2b", "H2c"))
    assert res["readings"]["H3"]["reading"] == NOT_EVALUATED


def test_d2_reads_each_seed_and_the_mean():
    vals = _vals2()
    A = _arrays(vals, rows=ROWS2)
    A.values[0, 2, :, METRICS.index("peak_vm_rel_err")] = 0.2     # one seed worse than L_D's
    r = claim_d2(A)                                               # 0.03, 0.03, 0.2 against 0.13
    assert r["per_metric"]["peak_vm_rel_err"]["holds_in_mean"]
    assert r["per_metric"]["peak_vm_rel_err"]["holds_in_seed"] == [True, True, False]
    assert r["verdict"] == NOT_REPLICATED
    A = _arrays(vals, rows=ROWS2)
    A.values[0, 0, :, METRICS.index("disp_rel_l2")] = 0.050       # a tie in one seed
    assert claim_d2(A)["verdict"] == NOT_REPLICATED
    A = _arrays(vals, rows=ROWS2)
    A.values[1, 0, 2, METRICS.index("vm_rel_l2")] = np.nan
    assert claim_d2(A)["verdict"] == NOT_EVALUATED


def test_guard_needs_the_effect_beyond_tau_and_halving_is_strict():
    vals = {(r, m): 0.1 for r in ROWS2 for m in METRICS}
    vals[("labels_knorm", "vm_rel_l2")] = 0.095                   # 5% lower: inside 10%
    A = _arrays(vals, rows=ROWS2)
    res = adjudicate_sets(None, A, CRITERIA)
    assert res["claims"]["H2b"]["verdict"] == NOT_REPLICATED
    assert res["claims"]["H2c"]["verdict"] == NOT_REPLICATED
    for knorm, want in ((0.05, NOT_REPLICATED), (0.0499, REPLICATED)):  # exactly half: not more
        A = _arrays({**vals, ("labels_knorm", "vm_rel_l2"): knorm}, rows=ROWS2)
        assert claim_halved(A, -0.5)["verdict"] == want, knorm


def test_non_finite_values_in_the_guard_claims_follow_cm2d():
    vals = _vals2()
    A = _arrays(vals, rows=ROWS2)
    A.values[2, 1, 4, METRICS.index("vm_rel_l2")] = np.nan       # L_K diverged in one seed
    A.values[2, 1, 4, METRICS.index("energy_gap_rel")] = np.inf
    res = adjudicate_sets(None, A, CRITERIA, h2d_val=_arrays(vals, rows=ROWS2))
    for c in ("H2a", "H2b", "H2c"):
        assert res["claims"][c]["verdict"] == NOT_REPLICATED and res["claims"][c]["diverged"]
    assert res["claims"]["H1"]["verdict"] == REPLICATED           # reads neither value
    assert res["readings"]["H3"]["reading"] == "the stiffness-norm transformer diverged"
    assert json.dumps(res, default=str)                           # no exception, a verdict
    B = _arrays(vals, rows=ROWS2)
    B.values[1, 0, 0, METRICS.index("energy_gap_rel")] = np.nan   # L_D, the reference
    res = adjudicate_sets(None, B, CRITERIA)
    for c in ("H1", "H2a"):
        assert res["claims"][c]["verdict"] == NOT_EVALUATED
    assert res["claims"]["H2b"]["verdict"] == REPLICATED          # reads the von Mises error
    assert res["claims"]["D2"]["verdict"] == NOT_EVALUATED
    C = _arrays(vals, rows=ROWS2)
    C.values[0, 2, 1, METRICS.index("energy_gap_rel")] = np.nan   # the label-free row
    res = adjudicate_sets(None, C, CRITERIA)
    assert res["claims"]["H1"]["verdict"] == NOT_REPLICATED and res["claims"]["H1"]["diverged"]
    assert res["readings"]["H3"]["reading"] == NOT_EVALUATED      # H3's reference


def test_the_resampling_record_is_what_the_script_gives():
    spec = importlib.util.spec_from_file_location("holdout_resample",
                                                  ROOT / "scripts" / "holdout_resample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from fejepa.analysis.holdout import load_spec

    rec = json.loads((ROOT / "records/cmame/holdout/resample.json").read_text())
    cfg = load_spec(ROOT / "configs" / "holdout_v1.json")
    assert rec["sizes"] == {k: s["n_holdout"] for k, s in cfg["sets"].items()}
    assert mod.resample(cfg, rec["resamples"], rec["seed"]) == rec
    # what the pre-registration quotes
    text = (ROOT / "PREREG_HOLDOUT.md").read_text(encoding="utf-8")
    pred = rec["schemes"]["predictive"]["replicated"]
    assert all(v == rec["resamples"] for v in rec["schemes"]["direct"]["replicated"].values())
    assert f"{pred['D2']:,} of {rec['resamples']:,}" in text
    assert all(pred[c] == rec["resamples"] for c in CLAIMS if c != "D2")
    # every failure counted by the comparison that failed, as the text says
    fail = rec["schemes"]["predictive"]["failing"]
    assert fail == {"D2 peak_vm_rel_err seed 0": rec["resamples"] - pred["D2"]}
    assert rec["schemes"]["direct"]["failing"] == {}
    n_fail = rec["resamples"] - pred["D2"]
    assert f"each of the other {n_fail} failed through the peak von Mises error in seed 0" \
        in text
    assert f"({n_fail} of {rec['resamples']:,} resamples)" in text and n_fail == 31
    h2c_max = rec["schemes"]["predictive"]["quantiles"]["H2c_rel"]["max"]
    assert f"H2c came closest, its largest rel {100 * h2c_max:.1f}%" in text
    assert h2c_max == max(rec["schemes"][s]["quantiles"][k]["max"] for s in rec["schemes"]
                          for k in ("H2c_rel",))


# ------------------------------------------------------------ boundaries --

def test_content_check_needs_every_value_finite():
    ref = {m: [0.1, 0.2, 0.3] for m in METRICS}
    assert content_check({m: list(v) for m, v in ref.items()}, ref, 1e-3)["ok"]
    got = {m: list(v) for m, v in ref.items()}
    got["vm_rel_l2"][1] = np.nan                    # the two deciding medians are unchanged
    c = content_check(got, ref, 1e-3)
    assert not c["ok"] and not c["finite"]
    assert c["disp_rel_l2_median_rel_dev"] == c["energy_gap_rel_median_rel_dev"] == 0.0
    got = {m: list(v) for m, v in ref.items()}
    got["disp_rel_l2"] = [x * (1 + 1.5e-3) for x in got["disp_rel_l2"]]
    assert not content_check(got, ref, 1e-3)["ok"]
    got["disp_rel_l2"] = [x * (1 + 0.5e-3) for x in ref["disp_rel_l2"]]
    assert content_check(got, ref, 1e-3)["ok"]


def test_zero_field_boundary():
    """A gap of exactly 1 is not worse; within 1e-9 of 1 the sign is not
    checked, beyond it it is."""
    g = np.array([1.0, 1.0 + 1e-10, 1.0 + 1e-8, 0.5])
    p = np.array([0.0, -1e-12, -1e-8, -1.0])
    w = worse_than_zero(g, p)
    assert w == {"load_cases": 4, "finite": 4, "worse": 2, "sign_mismatch": 1}
    w = worse_than_zero(np.array([1.0 - 1e-10, 1.0 + 1e-8]), np.array([1e-12, 1e-8]))
    assert w["worse"] == 1 and w["sign_mismatch"] == 0
    G = np.full((3, 3, 2, 4), 0.5)
    P = np.full(G.shape, -1.0)
    G[1, 0, 0, 0], P[1, 0, 0, 0] = 1.0, 0.0           # labels: exactly the zero field
    G[2, 0, 0, 0], P[2, 0, 0, 0] = 1.5, 0.5           # mgn: worse
    r = claim_t3(_arrays({}, n=2, gap=G, pi=P))
    assert r["verdict"] == NOT_REPLICATED and r["per_row"]["labels"]["worse"] == 0


def test_mean_gap_above_one_is_strict_and_per_seed():
    A = _arrays({("labels", "energy_gap_rel"): 0.5}, n=4)
    ri, gi = 1, METRICS.index("energy_gap_rel")
    A.values[ri, 0, 0, gi] = 1.0                       # exactly one: not counted
    A.values[ri, 0, 1, gi] = 1.0 + 1e-12
    A.values[ri, 2, 3, gi] = 3.0
    m = mean_gap_above_one(A, "labels")
    assert m["per_seed"] == [1, 0, 1] and m["total"] == 2 and m["pairs"] == 12
    used = {(r, s): True for r in ("ar", "labels", "mgn") for s in (0, 1, 2)}
    used[("labels", 2)] = False                        # a failed seed leaves the total
    B = _arrays({}, n=4, used=used)
    B.values[...] = A.values
    m = mean_gap_above_one(B, "labels")
    assert m["per_seed"] == [1, 0, 1] and m["content_ok"] == [True, True, False]
    assert m["total"] == 1 and m["pairs"] == 8


def test_pair_shares_and_guard_readings_read_new_against_reference():
    vals = _vals2()
    A = _arrays(vals, rows=ROWS2)
    A.values[:, 1] *= 1.02
    sh = pair_shares(A, "labels_knorm", "labels")      # L_K better on every pair
    assert all(v == 1.0 for v in sh.values())
    sh = pair_shares(A, "labels", "labels_knorm")
    assert all(v == 0.0 for v in sh.values())
    rd = set_readings(A, "h2d", CRITERIA["guard_band"], CRITERIA["guard_k"])
    g = rd["guard_labels_knorm_vs_labels"]["vm_rel_l2"]["guard"]
    assert g["lower"] and g["rel_change"] < 0         # new (L_K) lower than the reference
    assert rd["guard_labels_knorm_vs_labels"]["crit_recall"]["note"].startswith("higher")
    assert rd["pairs_mgn_vs_ar"]["disp_rel_l2"] == 0.0


def test_a_failed_seed_keeps_its_own_readings():
    vals = _vals2()
    used = {(r, s): True for r in ROWS2 for s in (0, 1, 2)}
    used[("labels_knorm", 1)] = False
    A = _arrays(vals, rows=ROWS2, used=used)
    A.values[2, 1, :, METRICS.index("vm_rel_l2")] = 0.9   # the failed seed's own values
    V = _arrays(vals, rows=ROWS2, used=dict(used))
    rd = set_readings(A, "h2d", CRITERIA["guard_band"], CRITERIA["guard_k"])
    pm = rd["per_seed_means"]["labels_knorm"]
    assert pm["content_ok"] == [True, False, True]
    assert pm["vm_rel_l2"] == pytest.approx([0.03, 0.9, 0.03])
    w = rd["worse_than_zero"]["labels_knorm"]
    assert w["s1"]["content_ok"] is False and w["all_seeds"]["seeds"] == [0, 2]
    assert w["all_seeds"]["load_cases"] == 2 * 10 * 4
    for key in ("pairs_labels_knorm_vs_labels", "guard_labels_knorm_vs_ar"):
        assert "not_evaluated" in rd[key]
    assert "not_evaluated" not in rd["guard_ar_vs_labels"]
    res = adjudicate_sets(None, A, CRITERIA, h2d_val=V)
    ratio = res["secondary"]["h2d"]["ratio_holdout_over_validation"]["labels_knorm"]
    assert ratio["vm_rel_l2"] == pytest.approx(1.0)       # seeds 0 and 2 only


def test_validation_readings_are_the_validation_arrays():
    """Holdout values twice the validation values: every ratio reads 2 and
    the two sets' readings differ."""
    vals = _vals2()
    H = _arrays(vals, rows=ROWS2)
    H.values[:, 1] *= 1.02
    V = _arrays(vals, rows=ROWS2)
    V.values[...] = H.values / 2.0
    res = adjudicate_sets(None, H, CRITERIA, h2d_val=V)
    sec = res["secondary"]["h2d"]
    assert all(v == pytest.approx(2.0) for r in sec["ratio_holdout_over_validation"].values()
               for v in r.values())
    assert sec["holdout"]["per_seed_means"] != sec["validation"]["per_seed_means"]
    assert sec["validation"]["per_seed_means"]["ar"]["disp_rel_l2"] == \
        pytest.approx([0.0315, 0.0315 * 1.02, 0.0315])
    assert res["validation_reference"]["claims"]["T1"] == NOT_EVALUATED   # no h3d arrays


def test_validation_reference_reads_the_validation_arrays():
    """D2 fails on the holdout arrays and holds on the validation arrays: the
    verdict follows the holdout set, the reference the validation set."""
    vals = _vals2()
    V = _arrays(vals, rows=ROWS2)
    H = _arrays({**vals, ("ar", "disp_rel_l2"): 0.040}, rows=ROWS2)
    res = adjudicate_sets(None, H, CRITERIA, h2d_val=V)
    assert res["claims"]["D2"]["verdict"] == NOT_REPLICATED
    assert res["validation_reference"]["claims"]["D2"] == REPLICATED
    res = adjudicate_sets(None, V, CRITERIA, h2d_val=H)
    assert res["claims"]["D2"]["verdict"] == REPLICATED
    assert res["validation_reference"]["claims"]["D2"] == NOT_REPLICATED


def test_t2_ties_are_neither_lower_nor_higher():
    n = 20
    A = _arrays({}, n=n)
    d, v = METRICS.index("disp_rel_l2"), METRICS.index("vm_rel_l2")
    A.values[0, :, :, d] = A.values[2, :, :, d] = 1.0             # ties on 60% ...
    A.values[2, :, :8, d] = 0.5                                     # ... lower on 40%
    A.values[0, :, :, v], A.values[2, :, :, v] = 1.0, 2.0
    r = claim_t2(A, 0.5, 0.95)
    assert r["disp_lower_share_per_seed"] == [0.4] * 3 and r["verdict"] == NOT_REPLICATED
    B = _arrays({}, n=n)
    B.values[0, :, :, d], B.values[2, :, :, d] = 1.0, 2.0
    B.values[2, :, :11, d] = 0.5                                    # 55% lower
    B.values[0, :, :, v] = B.values[2, :, :, v] = 1.0               # ties on 10% ...
    B.values[2, :, 2:, v] = 2.0                                     # ... higher on 90%
    r = claim_t2(B, 0.5, 0.95)
    assert r["disp_ok"] and r["vm_higher_share_pooled"] == pytest.approx(0.9)
    assert r["verdict"] == NOT_REPLICATED


def test_pair_shares_count_ties_as_neither():
    A = _arrays({(r, m): 0.2 for r in ROWS2 for m in METRICS}, rows=ROWS2)
    assert all(v == 0.0 for v in pair_shares(A, "labels_knorm", "labels").values())
    assert all(v == 0.0 for v in pair_shares(A, "labels", "labels_knorm").values())


def test_readings_keep_new_against_reference():
    """H3's exchanged roles, and the medians' guard, the Welch interval and
    the instance resampling of the secondary readings, each read the new row
    against its reference."""
    vals = {**_vals2(), ("labels_knorm", "energy_gap_rel"): 0.02, ("ar", "energy_gap_rel"): 0.03}
    A = _arrays(vals, rows=ROWS2)
    rng = np.random.default_rng(0)
    A.values *= 1.0 + 0.05 * rng.standard_normal(A.values.shape[:3])[..., None]
    h3 = reading_h3(A, CRITERIA["guard_band"], CRITERIA["guard_k"])
    a, b = (np.mean(A.per_seed(r, "energy_gap_rel")) for r in ("labels_knorm", "ar"))
    assert h3["rel_change"] == pytest.approx(a / b - 1) and h3["rel_change"] < 0
    assert h3["roles_exchanged"]["rel_change"] == pytest.approx(b / a - 1)
    rd = set_readings(A, "h2d", CRITERIA["guard_band"], CRITERIA["guard_k"])
    g = rd["guard_labels_knorm_vs_labels"]["vm_rel_l2"]               # L_K 0.03 against 0.13
    assert g["guard"]["rel_change"] < -0.5 and g["guard_medians"]["rel_change"] < -0.5
    assert g["welch"]["rel_change"] < -0.5 and g["welch"]["high"] < 0
    assert g["bootstrap"]["low"] < g["bootstrap"]["high"] < 0


def test_content_check_reads_both_medians():
    ref = {m: [0.1, 0.2, 0.3, 0.4, 0.5] for m in METRICS}
    got = {m: list(v) for m, v in ref.items()}
    got["energy_gap_rel"] = [x * (1 + 2e-3) for x in ref["energy_gap_rel"]]   # the gap alone
    c = content_check(got, ref, 1e-3)
    assert not c["ok"] and c["disp_rel_l2_median_rel_dev"] == 0.0
    got = {m: list(v) for m, v in ref.items()}
    got["disp_rel_l2"][4] = 1.0                       # one instance far off: the median holds
    c = content_check(got, ref, 1e-3)
    assert c["ok"] and c["disp_rel_l2_max_rel_dev"] == pytest.approx(1.0)
