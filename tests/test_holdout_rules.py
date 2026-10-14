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
the validation arrays' own; a claim that does not replicate names the
comparisons that failed; T1 and D2 carry the pairing-free reading (every
pair of seeds); V reads REPRESENTATIVE on the manuscript's arrays against
themselves, rejects a constructed shift in the direction it has, leaves out
what it cannot read and steps down by Holm's procedure; and the resampling
record that the pre-registration quotes is what scripts/holdout_resample.py
gives."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from fejepa.analysis.adjudicate_holdout import (CLAIMS, NOT_EVALUATED, NOT_REPLICATED,
                                                NOT_REPRESENTATIVE, REPLICATED, REPRESENTATIVE,
                                                Arrays, adjudicate_sets, arrays_from_report,
                                                claim_d2, claim_halved, claim_t1, claim_t2,
                                                claim_t3, claims_of, failing_components, holm,
                                                mean_gap_above_one, pair_shares, pairing_free,
                                                ratio_intervals, reading_h3, representativeness,
                                                resampled, set_readings)
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
        assert set(sec) == {"holdout", "validation", "ratio_holdout_over_validation",
                            "ratio_interval"}
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
    # V's calibration, as the text quotes it
    v = rec["V"]
    assert set(v) == {"alpha", "halves", "at_holdout_sizes", "single_quantity"}
    assert v["alpha"] == CRITERIA["representative_alpha"] == 0.05
    hv, sz = v["halves"], v["at_holdout_sizes"]
    assert hv["sizes"] == {k: {"holdout": 128, "validation": 128} for k in ("h2d", "h3d")}
    assert sz["sizes"] == {k: {"holdout": s["n_holdout"], "validation": 256}
                           for k, s in cfg["sets"].items()}
    assert hv["draws"] == sz["draws"] == 1000 and hv["draws_per_factor"] == 200
    assert hv["false_alarm"] <= 0.05 and sz["false_alarm"] <= 0.05

    def pct(x):
        return f"{100 * x:.1f}".rstrip("0").rstrip(".") + "%"

    assert (f"(1,000 splits), V read NOT REPRESENTATIVE in {pct(hv['false_alarm'])} of the "
            "splits") in text
    shifts = ["1.05", "1.10", "1.15", "1.20"]
    assert list(hv["power"]) == list(sz["power"]) == shifts
    said = ", ".join(pct(hv["power"][s]) for s in shifts[:3]) + f" and {pct(hv['power']['1.20'])}"
    assert f"multiplied by 1.05, 1.10, 1.15 and 1.20 on one half in both sets, eight quantities " \
           f"at once, in {said} of 200 splits each" in text
    said = ", ".join(pct(sz["power"][s]) for s in shifts[:3]) + f" and {pct(sz['power']['1.20'])}"
    assert f"in {pct(sz['false_alarm'])} without a change and in {said} of 200 draws" in text
    assert pct(sz["power"]["1.05"]) == "54%"                        # "about half" (Sec. 9)
    assert "a 5% change of the label-free errors was detected in about half" in text
    assert f"V REPRESENTATIVE (in {pct(1 - sz['false_alarm'])} of the calibration's draws at " \
           "the holdout sets' sizes)" in text
    # one quantity alone made worse
    one = v["single_quantity"]
    assert one["sizes"] == sz["sizes"] and one["draws"] == 200
    assert one["factors"] == ["1.02", "1.05", "1.10", "1.20"]
    det = one["detected"]
    recall = [q for q in det if q.endswith(":crit_recall")]
    errors = [q for q in det if q not in recall]
    assert len(det) == 35 and len(errors) == 28 and len(recall) == 7
    for kind, qs in (("error_metrics", errors), ("crit_recall", recall)):
        for f in one["factors"]:
            got = [det[q][f] for q in qs]
            assert one["summary"][kind][f] == {"min": min(got), "median": float(np.median(got)),
                                               "max": max(got)}
    e, r = one["summary"]["error_metrics"], one["summary"]["crit_recall"]
    assert (f"V rejected that quantity, among the 28 quantities of the error metrics, in "
            f"{pct(e['1.10']['min'])} to {pct(e['1.10']['max'])} of the draws at 1.10 (median "
            f"about {100 * e['1.10']['median']:.0f}%) and in {pct(e['1.20']['min'])} to "
            f"{pct(e['1.20']['max'])} at 1.20 (median about {100 * e['1.20']['median']:.0f}%), "
            f"and among the critical-region recalls in {pct(r['1.02']['min'])} to "
            f"{pct(r['1.02']['max'])} at 1.02 (median {pct(r['1.02']['median'])})") in text
    def span(qs):
        got = [det[q]["1.10"] for q in qs]
        return f"{pct(min(got))} to {pct(max(got))}"

    def err(s, rows):
        return [q for q in errors if q.split(":")[0] == s and q.split(":")[1] in rows]

    assert (f"the error metrics of T1's two rows were rejected at 1.10 in "
            f"{span(err('h3d', ('ar', 'labels')))} of the draws and those of D2's in "
            f"{span(err('h2d', ('ar', 'labels')))}") in text
    assert (f"whose four error metrics alone were rejected at 1.10 in {span(err('h3d', ('ar',)))} "
            f"of the draws (the two-dimensional one's in {span(err('h2d', ('ar',)))})") in text
    h1 = det["h2d:ar:energy_gap_rel"]
    assert (f"the two-dimensional label-free transformer's relative energy gap that H1 reads "
            f"(rejected in {pct(h1['1.10'])} of the draws at 1.10 and {pct(h1['1.20'])} at "
            "1.20)") in text
    assert (f"a change confined to one error metric of one row was detected less often, in as "
            f"few as {pct(e['1.10']['min'])} of the draws at 10% and {pct(e['1.20']['min'])} at "
            f"20% (one critical-region recall, by contrast, in most draws already at 2%") in text
    assert r["1.02"]["median"] > 0.5                                # "most draws"


def test_pairing_free_readings_and_v_on_the_manuscripts_own_arrays(manuscript):
    a3, a2 = manuscript
    res = adjudicate_sets(a3, a2, CRITERIA, h3d_val=a3, h2d_val=a2)
    c = res["claims"]
    assert all(v["failing"] == [] for v in c.values())
    assert c["T1"]["all_pairs_reading"] and c["D2"]["all_pairs_reading"]
    d2 = c["D2"]["per_metric"]
    assert all(v["holds_all_pairs"] for v in d2.values())
    # the closest pair: L_D's best seed against the label-free transformer's worst
    peak = d2["peak_vm_rel_err"]
    assert peak["worst_pair_rel"] == pytest.approx(max(peak["label_free_per_seed"])
                                                   / min(peak["supervised_per_seed"]) - 1)
    assert round(100 * peak["worst_pair_rel"], 1) == -3.0
    t1 = c["T1"]["per_metric"]
    closest = min(t1, key=lambda m: abs(t1[m]["worst_pair_rel"]))
    assert closest == "crit_recall"
    text = (ROOT / "PREREG_HOLDOUT.md").read_text(encoding="utf-8")
    assert (f"the least favourable pair is {100 * abs(peak['worst_pair_rel']):.1f}% apart in D2 "
            "(the peak "
            f"von Mises error) and {100 * abs(t1[closest]['worst_pair_rel']):.1f}% in T1 (the "
            "critical-region recall)") in text
    assert min(abs(v["worst_pair_rel"]) for v in d2.values()) == abs(peak["worst_pair_rel"])
    v = res["V"]
    names = sorted(f"{n}:{r}:{m}" for n, A in (("h3d", a3), ("h2d", a2)) for r in A.rows
                   for m in METRICS)
    assert v["verdict"] == REPRESENTATIVE and sorted(v["quantities"]) == names
    assert len(names) == 35 and v["left_out"] == [] and v["rejected"] == []
    assert v["alpha"] == CRITERIA["representative_alpha"] == 0.05
    assert all(q["p"] == 1.0 and q["prob_holdout_higher"] == 0.5 and not q["holdout_worse"]
               and q["seeds"] == [0, 1, 2] for q in v["quantities"].values())
    q = v["quantities"]["h3d:ar:disp_rel_l2"]
    assert (q["n_holdout"], q["n_validation"]) == (256, 256)
    assert q["holdout_mean"] == pytest.approx(np.mean(a3.per_seed("ar", "disp_rel_l2")))


def test_pairing_free_reading_compares_every_pair_of_seeds():
    A = _arrays(_t1_values())
    d, k = METRICS.index("disp_rel_l2"), METRICS.index("crit_recall")
    A.values[0, :, :, d] = np.array([1.0, 2.0, 3.0])[:, None]     # label-free, per seed
    A.values[1, :, :, d] = np.array([1.5, 2.5, 3.5])[:, None]     # supervised
    r = claim_t1(A)
    assert r["verdict"] == REPLICATED                             # seed s against seed s
    pm = r["per_metric"]["disp_rel_l2"]
    assert not pm["better_all_pairs"] and pm["worst_pair_rel"] == pytest.approx(3.0 / 1.5 - 1)
    assert not r["all_pairs_reading"]
    A.values[0, :, :, d] = 1.0
    A.values[0, :, :, k] = np.array([2.0, 1.2, 3.0])[:, None]     # recall: higher is better
    A.values[1, :, :, k] = np.array([1.0, 1.1, 1.3])[:, None]
    r = claim_t1(A)
    assert r["verdict"] == REPLICATED and not r["all_pairs_reading"]
    assert r["per_metric"]["crit_recall"]["worst_pair_rel"] == pytest.approx(1.2 / 1.3 - 1)
    A.values[0, :, :, k] = 2.0
    assert claim_t1(A)["all_pairs_reading"]
    vals = _vals2()
    B = _arrays(vals, rows=ROWS2)
    B.values[0, :, :, d] = np.array([0.060, 0.070, 0.080])[:, None]   # higher than L_D's
    B.values[1, :, :, d] = np.array([0.050, 0.065, 0.075])[:, None]
    r = claim_d2(B)
    assert r["verdict"] == REPLICATED and not r["all_pairs_reading"]
    pm = r["per_metric"]["disp_rel_l2"]
    assert not pm["holds_all_pairs"] and pm["worst_pair_rel"] == pytest.approx(0.06 / 0.075 - 1)


def test_failing_components_name_what_failed():
    A = _arrays(_t1_values())
    A.values[0, 1, :, METRICS.index("vm_rel_l2")] = 2.0           # a tie in seed 1
    A.values[0, :, :, METRICS.index("crit_recall")] = 1.0         # recall tied throughout
    r = claim_t1(A)
    assert failing_components("T1", r) == [
        "T1 vm_rel_l2 seed 1", "T1 crit_recall seed 0", "T1 crit_recall seed 1",
        "T1 crit_recall seed 2", "T1 crit_recall seed mean"]
    vals = _vals2()
    B = _arrays(vals, rows=ROWS2)
    B.values[0, 0, :, METRICS.index("peak_vm_rel_err")] = 0.2     # seed 0 worse than L_D's
    assert failing_components("D2", claim_d2(B)) == ["D2 peak_vm_rel_err seed 0"]
    G = np.full((3, 3, 4, 4), 0.5)
    P = np.full(G.shape, -1.0)
    G[1, 0, 0, 0], P[1, 0, 0, 0] = 1.5, 0.2
    assert failing_components("T3", claim_t3(_arrays({}, n=4, gap=G, pi=P))) == ["T3 mgn"]
    C = _arrays({(r, m): 0.1 for r in ROWS2 for m in METRICS}, rows=ROWS2)
    res = adjudicate_sets(None, C, CRITERIA)
    for c in ("H1", "H2a", "H2b", "H2c"):
        assert res["claims"][c]["verdict"] == NOT_REPLICATED and res["claims"][c]["failing"] == [c]
    assert res["claims"]["T1"]["failing"] == []                   # NOT EVALUATED
    assert all(v["failing"] == failing_components(c, v) for c, v in res["claims"].items())
    assert failing_components("T1", claim_t1(_arrays(_t1_values()))) == []   # REPLICATED


def test_t2_failing_parts():
    n = 20
    A = _arrays({}, n=n)
    A.values[0, :, :, METRICS.index("disp_rel_l2")] = 1.0
    A.values[2, :, :, METRICS.index("disp_rel_l2")] = 0.5         # lower everywhere
    A.values[0, :, :, METRICS.index("vm_rel_l2")] = 1.0
    A.values[2, :, :, METRICS.index("vm_rel_l2")] = 0.5           # von Mises lower: not higher
    assert failing_components("T2", claim_t2(A, 0.5, 0.95)) == ["T2 vm_ok"]
    A.values[2, :, :, METRICS.index("disp_rel_l2")] = 1.0         # a tie: not lower
    assert failing_components("T2", claim_t2(A, 0.5, 0.95)) == ["T2 disp_ok", "T2 vm_ok"]


def test_holm_steps_down_and_stops_at_the_first_kept():
    assert holm({"a": 0.001, "b": 0.02, "c": 0.04}, 0.05) == {"a": True, "b": True, "c": True}
    assert holm({"a": 0.001, "b": 0.03, "c": 0.04}, 0.05) == {"a": True, "b": False, "c": False}
    assert holm({"a": 0.026, "b": 0.03}, 0.05) == {"a": False, "b": False}
    assert holm({"a": 0.025, "b": 0.05}, 0.05) == {"a": True, "b": True}     # at the level
    assert holm({"b": 0.001, "a": float("nan")}, 0.05) == {"a": False, "b": True}
    p = {f"q{i:02d}": 0.05 / (35 - i) for i in range(35)}
    assert all(holm(p, 0.05).values())
    p["q00"] = 0.05 / 35 * 1.0001
    assert not any(holm(p, 0.05).values())
    assert holm(dict(reversed(list(p.items()))), 0.05) == holm(p, 0.05)


def _copy(A: Arrays, used=None) -> Arrays:
    d = {"rows": np.array(A.rows), "seeds": np.array(A.seeds), "metrics": np.array(METRICS),
         "values": A.values.copy(), "gap_load": A.gap_load, "pi_load": A.pi_load}
    return Arrays(d, dict(used or A.used))


def test_v_rejects_a_shift_in_its_direction(manuscript):
    a3, a2 = manuscript
    H = _copy(a2)
    errors = [m for m in METRICS if m != "crit_recall"]
    for m in errors:
        H.values[0, :, :, METRICS.index(m)] *= 1.5                # the label-free row worse
    v = representativeness({"h2d": (H, a2), "h3d": (a3, a3)}, 0.05)
    assert v["verdict"] == NOT_REPRESENTATIVE and v["sets"] == ["h2d", "h3d"]
    assert v["rejected"] == v["rejected_holdout_worse"] == sorted(f"h2d:ar:{m}" for m in errors)
    q = v["quantities"]["h2d:ar:disp_rel_l2"]
    assert q["prob_holdout_higher"] > 0.5 and q["holdout_worse"] and q["p"] < 0.05 / 35
    assert q["ratio_of_means"] == pytest.approx(1.5)
    # the p-value is scipy's asymptotic test with the tie and continuity corrections
    from scipy.stats import mannwhitneyu

    q = v["quantities"]["h2d:labels:disp_rel_l2"]                 # unchanged: p = 1
    h = H.per_instance("ar", "peak_vm_rel_err").mean(axis=0)
    w = a2.per_instance("ar", "peak_vm_rel_err").mean(axis=0)
    got = v["quantities"]["h2d:ar:peak_vm_rel_err"]["p"]
    kw = dict(alternative="two-sided", method="asymptotic")
    assert got == mannwhitneyu(h, w, use_continuity=True, **kw).pvalue
    assert got != mannwhitneyu(h, w, use_continuity=False, **kw).pvalue
    assert q["p"] == 1.0
    H = _copy(a2)
    H.values[0, :, :, METRICS.index("disp_rel_l2")] /= 1.5        # better on the holdout set
    v = representativeness({"h2d": (H, a2)}, 0.05)
    assert v["rejected"] == ["h2d:ar:disp_rel_l2"] and v["rejected_holdout_worse"] == []
    H = _copy(a2)
    H.values[1, :, :, METRICS.index("crit_recall")] *= 0.7        # recall lower: worse
    v = representativeness({"h2d": (H, a2)}, 0.05)
    assert v["rejected"] == v["rejected_holdout_worse"] == ["h2d:labels:crit_recall"]
    assert v["quantities"]["h2d:labels:crit_recall"]["prob_holdout_higher"] < 0.5


def test_v_leaves_out_what_it_cannot_read_and_reads_passed_seeds_only(manuscript):
    _, a2 = manuscript
    used = {(r, s): True for r in a2.rows for s in a2.seeds}
    used.update({("mgn", s): False for s in a2.seeds})            # no seed of mgn passed
    used[("labels", 1)] = False                                   # one seed of L_D failed
    H = _copy(a2, used)
    H.values[1, 1] = 1e6                                          # ... and is not read
    H.values[0, 2, 7, METRICS.index("vm_rel_l2")] = np.nan        # a non-finite value
    v = representativeness({"h2d": (H, _copy(a2, used))}, 0.05)
    assert v["verdict"] == REPRESENTATIVE and v["rejected"] == []
    assert v["left_out"] == [{"quantity": "h2d:ar:vm_rel_l2",
                              "why": "non-finite per-instance values",
                              "non_finite_holdout": 1, "non_finite_validation": 0},
                             {"quantity": "h2d:mgn",
                              "why": "no state of the row passed its content check"}]
    assert len(v["quantities"]) == 20 - 5 - 1
    assert v["quantities"]["h2d:labels:disp_rel_l2"]["seeds"] == [0, 2]
    assert v["quantities"]["h2d:labels:disp_rel_l2"]["p"] == 1.0
    W = _copy(a2)
    W.values[3, 0, 9, METRICS.index("energy_gap_rel")] = np.inf     # on the validation side
    v = representativeness({"h2d": (_copy(a2), W)}, 0.05)
    assert v["left_out"] == [{"quantity": "h2d:mgn:energy_gap_rel",
                              "why": "non-finite per-instance values",
                              "non_finite_holdout": 0, "non_finite_validation": 1}]
    none = {k: False for k in used}
    v = representativeness({"h2d": (_copy(a2, none), _copy(a2, none))}, 0.05)
    assert v["verdict"] == NOT_EVALUATED and len(v["left_out"]) == 4
    assert adjudicate_sets(None, a2, CRITERIA)["V"]["verdict"] == NOT_EVALUATED
    assert adjudicate_sets(None, a2, CRITERIA, h2d_val=a2)["V"]["sets"] == ["h2d"]


def _resample_script():
    spec = importlib.util.spec_from_file_location("holdout_resample",
                                                  ROOT / "scripts" / "holdout_resample.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vs_single_quantity_reading_is_v_with_that_quantity_alone_changed(manuscript):
    """The calibration's reading with one quantity made worse (its p-value
    from the family with every quantity made worse, Holm on the family
    without a change but for it) is what `representativeness` decides when
    that quantity alone is changed."""
    mod = _resample_script()
    a3, a2 = manuscript
    rng = np.random.default_rng(7)
    sets = {k: (resampled(A, rng.integers(0, 256, size=300)),
                resampled(A, rng.integers(0, 256, size=256))) for k, A in (("h3d", a3),
                                                                            ("h2d", a2))}
    p0 = {q: x["p"] for q, x in representativeness(sets, 0.05)["quantities"].items()}
    for f in (1.1, 1.2):
        pf = {q: x["p"] for q, x in representativeness(
            {k: (mod.worse(H, f), W) for k, (H, W) in sets.items()}, 0.05)["quantities"].items()}
        for q in ("h3d:ar:disp_rel_l2", "h3d:mgn:peak_vm_rel_err", "h2d:labels:crit_recall",
                  "h2d:ar:energy_gap_rel"):
            name, row, m = q.split(":")
            H, W = sets[name]
            one = _copy(H)
            ri, mi = H.rows.index(row), METRICS.index(m)
            one.values[ri, :, :, mi] = mod.worse(H, f).values[ri, :, :, mi]
            direct = representativeness({**sets, name: (one, W)}, 0.05)
            assert direct["quantities"][q]["p"] == pf[q]
            assert all(x["p"] == p0[k] for k, x in direct["quantities"].items() if k != q)
            assert (q in direct["rejected"]) == holm({**p0, q: pf[q]}, 0.05)[q]
    W = mod.worse(a2, 1.25)                                       # errors up, recall down
    for m in METRICS:
        x, y = W.per_instance("ar", m), a2.per_instance("ar", m)
        assert np.allclose(x, y / 1.25 if m == "crit_recall" else y * 1.25, rtol=1e-15)


def test_v_takes_its_level_from_the_configuration(manuscript):
    a3, a2 = manuscript
    H = _copy(a2)
    H.values[0, :, :, METRICS.index("disp_rel_l2")] *= 1.5
    assert adjudicate_sets(None, H, CRITERIA, h2d_val=a2)["V"]["verdict"] == NOT_REPRESENTATIVE
    v = adjudicate_sets(None, H, dict(CRITERIA, representative_alpha=0.0), h2d_val=a2)["V"]
    assert v["verdict"] == REPRESENTATIVE and v["alpha"] == 0.0
    assert v["quantities"]["h2d:ar:disp_rel_l2"]["p"] < 1e-6
    mod = _resample_script()
    from fejepa.analysis.holdout import load_spec

    cfg = load_spec(ROOT / "configs" / "holdout_v1.json")
    for alpha, want in ((0.0, 0.0), (0.05, 1.0)):
        rec = mod.resample(dict(cfg, criteria=dict(cfg["criteria"], representative_alpha=alpha)),
                           10, 0)
        assert rec["V"]["alpha"] == alpha
        assert rec["V"]["halves"]["power"]["1.20"] == want            # 2 splits, x 1.2
        assert rec["V"]["single_quantity"]["detected"]["h3d:ar:crit_recall"]["1.20"] == want


def test_ratio_intervals_resample_each_side_on_its_own(manuscript):
    a3, a2 = manuscript
    ri = ratio_intervals(a3, a3)
    assert set(ri) == {"ar", "labels", "mgn"}
    for r in ri.values():
        for x in r.values():
            assert x["ratio"] == 1.0 and x["resamples"] == 2000
            assert x["non_finite_resamples"] == 0
            for k in ("interval_95_holdout", "interval_95_both"):
                assert x[k][0] < 1.0 < x[k][1], k
    # the heavy tail of the supervised transformer's energy gap: an interval that says little
    lab = ri["labels"]["energy_gap_rel"]["interval_95_both"]
    assert lab[0] < 0.01 and lab[1] > 100
    assert ratio_intervals(a3, a3) == ri                          # seeded
    assert ratio_intervals(a3, a3, seed=1) != ri
    # by hand, with sets of different sizes and a remainder chunk: the same draws
    rng = np.random.default_rng(5)
    H = _copy(resampled(a2, rng.integers(0, 256, size=4111)))
    H.values *= 2.0
    h = H.per_instance("ar", "disp_rel_l2").mean(axis=0)
    v = a2.per_instance("ar", "disp_rel_l2").mean(axis=0)
    draw = np.random.default_rng(0)
    alone, both = [], []
    for b in (200, 200, 50):
        ih, iv = draw.integers(0, 4111, size=(b, 4111)), draw.integers(0, 256, size=(b, 256))
        alone.append(h[ih].mean(axis=1) / v.mean())
        both.append(h[ih].mean(axis=1) / v[iv].mean(axis=1))
    got = ratio_intervals(H, a2, n_boot=450)["ar"]["disp_rel_l2"]
    assert got["ratio"] == pytest.approx(h.mean() / v.mean())
    assert 1.8 < got["ratio"] < 2.2
    assert got["interval_95_holdout"] == list(np.quantile(np.concatenate(alone), [0.025, 0.975]))
    assert got["interval_95_both"] == list(np.quantile(np.concatenate(both), [0.025, 0.975]))
    w = got["interval_95_holdout"]                                # the holdout's sampling only:
    assert w[1] - w[0] < got["interval_95_both"][1] - got["interval_95_both"][0]  # narrower
    # the adjudication's reading is this function with its defaults, holdout over validation
    res = adjudicate_sets(None, H, CRITERIA, h2d_val=a2)
    assert res["secondary"]["h2d"]["ratio_interval"] == ratio_intervals(H, a2)
    # the seeds that passed; values that are not finite; a validation mean of zero
    used = {(r, s): True for r in a2.rows for s in a2.seeds}
    used[("labels", 1)] = False
    H = _copy(a2, used)
    H.values[1, 1] = 1e6
    H.values[0, 0, 3, METRICS.index("vm_rel_l2")] = np.nan
    W = _copy(a2, used)
    W.values[2, :, :, METRICS.index("disp_rel_l2")] = 0.0
    ri = ratio_intervals(H, W)
    assert ri["labels"]["disp_rel_l2"]["ratio"] == pytest.approx(1.0)
    assert ri["ar"]["vm_rel_l2"] is None and ri["labels_knorm"]["disp_rel_l2"] is None
    # resamples whose validation mean is zero are counted and left out of the quantiles
    H = _arrays({("ar", "disp_rel_l2"): 1.0}, n=10)
    W = _arrays({("ar", "disp_rel_l2"): 0.0}, n=10)
    W.values[0, :, 0, METRICS.index("disp_rel_l2")] = 1.0         # one instance of ten
    x = ratio_intervals(H, W)["ar"]["disp_rel_l2"]
    assert 400 < x["non_finite_resamples"] < 1000 and all(np.isfinite(x["interval_95_both"]))
    assert x["ratio"] == pytest.approx(10.0) and x["interval_95_holdout"] == [10.0, 10.0]


def test_validation_reference_carries_the_pairing_free_readings(manuscript):
    a3, a2 = manuscript
    H = _copy(a2)
    H.values[0, :, :, METRICS.index("peak_vm_rel_err")] *= 0.9
    res = adjudicate_sets(None, H, CRITERIA, h2d_val=a2)          # the validation side's own
    pf = res["validation_reference"]["pairing_free"]["D2"]
    assert pf == pairing_free(claims_of(None, a2, CRITERIA)[0]["D2"])
    assert pf != pairing_free(res["claims"]["D2"])
    res = adjudicate_sets(a3, a2, CRITERIA, h3d_val=a3, h2d_val=a2)
    pf = res["validation_reference"]["pairing_free"]
    for c in ("T1", "D2"):
        assert pf[c]["all_pairs_reading"] is True
        for m, x in pf[c]["per_metric"].items():
            assert x["worst_pair_rel"] == res["claims"][c]["per_metric"][m]["worst_pair_rel"]
    assert pf["D2"]["per_metric"]["peak_vm_rel_err"]["holds_all_pairs"] is True
    assert pf["T1"]["per_metric"]["crit_recall"]["better_all_pairs"] is True
    res = adjudicate_sets(None, a2, CRITERIA, h2d_val=a2)
    assert res["validation_reference"]["pairing_free"]["T1"] is None   # no h3d arrays


def test_pairing_free_reading_with_a_zero_value():
    A = _arrays(_t1_values())
    d = METRICS.index("disp_rel_l2")
    A.values[1, :, :, d] = np.array([0.0, 2.0, 3.0])[:, None]     # a supervised seed at zero
    r = claim_t1(A)
    assert r["per_metric"]["disp_rel_l2"]["worst_pair_rel"] is None
    assert not r["per_metric"]["disp_rel_l2"]["better_all_pairs"]
    assert json.dumps(r, allow_nan=False)


def test_the_heavy_tail_the_text_quotes(manuscript):
    a3, _ = manuscript
    g = a3.per_instance("labels", "energy_gap_rel")[1]
    text = (ROOT / "PREREG_HOLDOUT.md").read_text(encoding="utf-8")
    assert f"has a mean of {g.mean():.1f} in seed 1, from one instance at {g.max():,.0f}" in text
    assert np.mean(np.delete(g, np.argmax(g))) < 1.0


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
