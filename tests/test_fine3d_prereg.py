"""fine3d: every number PREREG_FINE3D.md quotes is recomputed from the records
it names; its lines are the four configurations' in the generator's order;
it is ASCII; and stamping a copy of the draft does what Sec. 8 says, refusing
anything else. Every test holds on the draft and on the stamped file."""

import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "PREREG_FINE3D.md"
W8 = ROOT / "records" / "wp8"
SIMS = ROOT / "records" / "wp9" / "fine3d" / "simulations.json"


def _j(p):
    return json.loads(Path(p).read_text())


def _text():
    return re.sub(r"\s+", " ", PRE.read_text(encoding="utf-8"))


def _has(*phrases):
    t = _text()
    for p in phrases:
        assert p in t, p


def _pct(x, nd=0):
    return f"{100 * x:.{nd}f}%"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ascii_and_lines():
    from fejepa.report import PREREG_PLACEHOLDER, read_prereg_entries

    raw = PRE.read_text(encoding="utf-8")
    assert raw.isascii()
    entries = read_prereg_entries(PRE)
    assert [lab for lab, _ in entries] == list(_mod("make_fine3d_configs").ARMS)
    st = _mod("stamp_prereg_fine3d")
    tail = raw.rstrip("\n").splitlines()[-1]
    if tail == st.FOOTER_OPEN:                                  # the draft
        assert all(v == PREREG_PLACEHOLDER for _, v in entries)
    else:                                                       # the stamped file
        assert re.fullmatch(r"PREREG_FINE3D_SHA256 = [0-9a-f]{64}", tail) and st.self_hash_ok(raw)
        from fejepa.report import config_sha256

        for lab, v in entries:
            assert v == config_sha256(_j(ROOT / "configs" / f"{lab}.json"))


def test_section_1_numbers():
    p2b = _j(W8 / "e2" / "baseline" / "report_phase2b.json")
    m = p2b["results"]["p3_transfer"]["metrics"]
    _has(f"{m['ratio_fine_over_inband']:.2f} times its in-band error (seed means "
         f"{m['ar']['fine_disp_mean']:.3f} and {m['ar']['inband_disp_mean']:.4f}")
    amp_t = _j(W8 / "posthoc" / "amp_phase2b.json")
    nodes = [r["n_nodes"] for r in amp_t["seeds"]["s0"]["fine"]["per_instance"]]
    assert len(nodes) == 256 and round(float(np.median(nodes)), -3) == 36000
    _has(f"about 36,000 nodes per instance, median; {min(nodes):,} to {max(nodes):,}")
    sm = amp_t["seed_means"]
    cs = [amp_t["seeds"][s]["fine"]["summary"]["c_star_median"] for s in ("s0", "s1", "s2")]
    assert sm["fine"]["c_star_median"] == pytest.approx(np.mean(cs))      # the "median" defined
    _has(f"is {sm['fine']['c_star_median']:.2f} on the fine set against "
         f"{sm['inband']['c_star_median']:.2f} in band",
         f"by {_pct(1 - sm['fine']['disp_c'] / sm['fine']['disp'])}")
    w9 = _j(ROOT / "records" / "wp9" / "w9_verdict.json")["H2"]
    r = w9["F5_over_inband_ratio"]
    _has(f"fell by {_pct(-w9['F5_displacement']['rel_change'], 1)}, and its ratio to the "
         f"in-band error by {_pct(-r['rel_change'], 1)} ({r['base_mean']:.2f} to {r['new_mean']:.2f})")
    e2 = _j(W8 / "e2" / "e2_m1024" / "report.json")
    a = e2["results"]["p3_transfer"]["metrics"]["ar"]
    f, i = np.mean(a["fine"]["disp_rel_l2"]["per_seed"]), np.mean(a["inband"]["disp_rel_l2"]["per_seed"])
    _has(f"the fine error is {f / i:.2f} times the in-band one (seed means {f:.3f} and {i:.4f})")
    assert round(i / m["ar"]["inband_disp_mean"]) == 3                    # "three times"
    amp = _j(W8 / "posthoc" / "amp_e2_m1024.json")
    cs = [amp["seeds"][s]["fine"]["summary"]["c_star_median"] for s in ("s0", "s1", "s2")]
    ci = [amp["seeds"][s]["inband"]["summary"]["c_star_median"] for s in ("s0", "s1", "s2")]
    _has("median c* on the fine set is " + " / ".join(f"{c:.2f}" for c in cs)
         + f" by seed ({np.mean(ci):.2f} in band)",
         f"c* lowers its fine error by {_pct(1 - amp['seed_means']['fine']['disp_c'] / amp['seed_means']['fine']['disp'])}")
    log = (W8 / "e2" / "e2_m1024" / "run.log").read_text()
    h, mi, sec = map(int, re.search(r"done in (\d+):(\d+):(\d+)", log).groups())
    hours = (h + mi / 60 + sec / 3600)
    _has(f"trained in {hours / 3:.1f} h per seed")     # Sec. 9's time: test_section_7_and_9
    assert "27.3 h per seed" in (ROOT / "PREREG_E2.md").read_text()
    ft = m["fewshot"]["16"]["finetune"]
    inb = m["ar"]["inband"]
    _has(f"brought the displacement error down to {ft['disp_rel_l2']['mean']:.4f} (seed mean; "
         f"{ft['disp_rel_l2']['mean'] / inb['disp_rel_l2']['mean']:.1f} times the in-band "
         f"{inb['disp_rel_l2']['mean']:.4f})",
         f"energy gap ({ft['energy_gap_rel']['mean']:.4f}, "
         f"{ft['energy_gap_rel']['mean'] / inb['energy_gap_rel']['mean']:.1f} times the in-band "
         f"{inb['energy_gap_rel']['mean']:.4f})",
         f"von Mises error ({ft['vm_rel_l2']['mean']:.3f}, "
         f"{ft['vm_rel_l2']['mean'] / inb['vm_rel_l2']['mean']:.1f} times "
         f"{inb['vm_rel_l2']['mean']:.4f})")


def test_section_2_and_3_numbers():
    rec = _j(ROOT / "records" / "wp9" / "fine3d" / "scale_factor_3d.json")
    s, r = rec["sample"], rec["remesh"]["0.0374"]
    _has(f"the median is {s['median']:.5f} (10th-90th percentile {s['p10']:.5f}-{s['p90']:.5f}, "
         f"i.e. {s['p10_over_factor']:.2f}-{s['p90_over_factor']:.2f} times 2^-10)",
         f"(rank correlation {s['spearman_ratio_vs_lc']:.2f})",
         f"({s['median_over_factor']:.2f} times)",
         f"max|F| at lc 0.0374 is {r['max_abs_F_ratio_median']:.3f} times its lc 0.0906 value "
         f"(median; range {r['max_abs_F_ratio_range'][0]:.3f}-{r['max_abs_F_ratio_range'][1]:.3f}; "
         f"lc itself {r['lc_over_coarsest']:.3f} times)",
         f"unchanged to within {_pct(max(v['sum_abs_F_ratio_max_dev'] for v in rec['remesh'].values()), 2)}")
    assert s["factor"] == 2.0 ** -10
    _has("of which R1 (branch cmame-paper, PREREG_HOLDOUT) evaluates 686")
    p2b = _j(W8 / "e2" / "baseline" / "report_phase2b.json")
    st = p2b["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    _has(*(f"`{st[k]['sha256'][:8]}...`" for k in ("s0", "s1", "s2")),
         f"SHA-256 `{hashlib.sha256((W8 / 'e2' / 'baseline' / 'report_phase2b.json').read_bytes()).hexdigest()[:8]}...`")
    assert p2b["provenance"]["git"] == "prereg-phase2b-r2"
    assert p2b["provenance"]["timestamp_utc"].startswith("2026-09-28")
    _has("run 22-28 September 2026 at `prereg-phase2b-r2` with the supervised trainer")
    ft = p2b["results"]["p3_transfer"]["metrics"]["fewshot"]["16"]["finetune"]
    for k, name in (("disp_rel_l2", "displacement"), ("energy_gap_rel", "energy gap"),
                    ("vm_rel_l2", "von Mises")):
        _has(f"{name} " + " / ".join(f"{v:.4f}" for v in ft[k]["per_seed"]))
    e2c = _j(ROOT / "configs" / "e2_m1024.json")
    e2r = _j(W8 / "e2" / "e2_m1024" / "report.json")
    _has(f"CONFIG_SHA256 `{e2r['provenance']['config_sha256'][:8]}...`")
    for d in e2r["provenance"]["datasets"]:
        _has(f"`{d['dir']}` (manifest `{d['manifest_sha256'][:12]}...`)"
             if d["dir"].endswith("phase2") else f"`{d['dir']}` (`{d['manifest_sha256'][:12]}...`)")
    assert e2c["experiments"]["e8"]["ar_epochs"] * 1024 == 204800


def test_r3_content_check_tolerance():
    """The tolerance is far above what the states showed on the box (exactly
    zero in wp8's post-hoc pass) and far below a wrong state's deviation."""
    amp = _j(W8 / "posthoc" / "amp_phase2b.json")
    for s in amp["seeds"].values():
        for name in ("inband", "fine"):
            rep = s[name]["reproduction"]
            assert rep["checked"] and rep["disp_max_rel_dev"] == rep["egap_max_rel_dev"] == 0.0
    for arm in ("r3_eft_b16", "r3_lft_b16"):
        cfg = _j(ROOT / "configs" / f"{arm}.json")["content_check"]
        assert cfg == {"n_fine": 8, "n_inband": 8, "max_rel_dev": 1e-3}
    _has("first 8 fine evaluation instances and the first 8 in-band validation instances",
         "to a relative 1e-3 (wp8's post-hoc pass reproduced them exactly")


def test_section_6_r2a_numbers():
    sims = _j(SIMS)
    cv = sims["r2a_cv"]
    _has(f"coefficients of variation {_pct(cv['fine'])} and {_pct(cv['inband_disp'])} for the "
         f"fine and the in-band displacement errors, {_pct(cv['inband_egap'])} for the in-band "
         "energy gap")
    r = sims["r2a"]

    def p(k, key="p_supported", nd=0, block=r):
        return _pct(block[k][key], nd)
    _has(f"occurs in {p('S spread x1.0; inband unchanged; fine -0%', nd=1)} of trials "
         f"({p('S spread x1.5; inband unchanged; fine -0%', nd=1)} if S's seeds spread")
    for tag, phrase in (("inband unchanged", "with the in-band errors unchanged is supported in "),
                        ("inband 10% lower", "condition (ii) needs more at small effects ("),
                        ("inband 10% higher", "K1 fails more often (")):
        vals = " / ".join(p(f"S spread x1.0; {tag}; fine -{d}%") for d in (20, 30, 50, 70))
        _has(phrase + vals)
    k1 = 1 - r["S spread x1.0; inband unchanged; fine -70%"]["p_supported"]
    assert 0.08 < k1 < 0.12                                     # "about one trial in ten"
    sens = sims["r2a_sensitivity"]
    m512 = sims["r2a_cv_m512"]
    _has("with both arms' spreads twice E2's, a true reduction of 30% / 50% / 70% is supported in "
         + " / ".join(p(f"E2 spreads x2; fine -{d}%", block=sens) for d in (30, 50, 70)),
         f"spreads of E2's M = 512 arm ({_pct(m512['fine'])}, {_pct(m512['inband_disp'])} and "
         f"{_pct(m512['inband_egap'])}) in "
         + " / ".join(p(f"E2 M = 512 spreads; fine -{d}%", block=sens) for d in (30, 50, 70)))
    worst_null = max(v["p_supported"] for k, v in sens.items() if k.endswith("fine -0%"))
    _has(f"the rate without an effect staying at or below {_pct(worst_null, 1)}")
    c = sims["r2a_correlated"]
    _has(f"(rho 0.5: {p('rho 0.5; fine -0%', nd=1, block=c)} without an effect; "
         + " / ".join(p(f"rho 0.5; fine -{d}%", block=c) for d in (20, 30, 50, 70)) + ")")
    _has("a seed mean non-finite in only one of P3's in-band pass and E8's cell is refused "
         "instead (Sec. 8)",
         "a seed mean finite in one of the two only fails, and one finite in neither, a "
         "diverged seed, is left to Sec. 6")                  # Sec. 8; the code: adjudication
    assert sims["r2b_fine_rel_max"] == -0.45
    _has("recommendation in "
         + " / ".join(p(f"S spread x1.0; inband unchanged; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " of trials with the in-band errors unchanged ("
         + " / ".join(p(f"S spread x1.0; inband 10% lower; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " if S also lowers them by 10%; "
         + " / ".join(p(f"S spread x1.0; inband 10% higher; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " if it raises them by 10%), one of 30% in "
         + p("S spread x1.0; inband unchanged; fine -30%", "p_r2b_recommended", nd=1) + ".")


def test_section_6_r2b_numbers():
    """The R2b line and its reasons (`r2b`, `r2b_sensitivity`, `r2b_line`),
    from the Phase-2b spreads and wp8's post-hoc records; the decision's
    assumptions and conclusions are read from the record and the
    adjudicator's constant, not restated."""
    from fejepa.analysis.adjudicate_fine3d import R2B_FINE_REL_MAX

    sims = _j(SIMS)
    line_pct = round(100 * R2B_FINE_REL_MAX)
    assert R2B_FINE_REL_MAX == sims["r2b_fine_rel_max"] == -0.45
    from fejepa.analysis.adjudicate_fine3d import R2B_TIE

    _has(f"is **recommended** iff R2a is SUPPORTED and S's fine displacement error is at most "
         f"{100 + line_pct}% of the fresh baseline's (rel <= {line_pct}% in condition (i), "
         f"compared to within {R2B_TIE:g} for floating-point rounding). Otherwise it is not "
         "recommended on this evidence. A recommendation, not a verdict",
         f"Why {line_pct}% (the same script, `r2b`, `r2b_sensitivity` and `r2b_line`)",
         "on three fresh seeds each as here: six runs of about 27 h of training, about a week "
         "of one GPU",
         "(R2b: the transformer with and without S on three fresh seeds each as simulated in "
         "Sec. 6, six runs of 27.3 h of training,")
    assert "27.3 h per seed" in (ROOT / "PREREG_E2.md").read_text()
    b = sims["r2b"]
    p2b = _j(W8 / "e2" / "baseline" / "report_phase2b.json")
    p3 = p2b["results"]["p3_transfer"]["metrics"]["ar"]
    e8 = p2b["results"]["e8"]["metrics"]["cells"]["ar"]["1024"]

    def cv(x):
        x = np.asarray(x, float)
        return x.std(ddof=1) / x.mean()
    assert b["cv"]["fine"] == pytest.approx(cv(p3["fine"]["disp_rel_l2"]["per_seed"]))
    assert b["cv"]["inband_disp"] == pytest.approx(cv(p3["inband"]["disp_rel_l2"]["per_seed"]))
    assert b["cv"]["inband_egap"] == pytest.approx(cv(e8["energy_gap_rel"]["per_seed"]))
    a = sims["r2a_cv"]
    _has("- The decision. R2b is judged by R2a's rule on the seed spreads of the Phase-2b "
         f"transformer (coefficients of variation {_pct(b['cv']['fine'])} for the fine "
         "displacement error, "
         f"{_pct(b['cv']['inband_disp'])} and {_pct(b['cv']['inband_egap'])} for the in-band "
         f"displacement error and energy gap, against the bottleneck's {_pct(a['fine'])}, "
         f"{_pct(a['inband_disp'])} and {_pct(a['inband_egap'])}")
    _has(f"coefficients of variation {_pct(b['cv']['fine'])} for the fine displacement error, "
         f"{_pct(b['cv']['inband_disp'])} and {_pct(b['cv']['inband_egap'])} for the in-band "
         f"displacement error and energy gap, against the bottleneck's {_pct(a['fine'])}, "
         f"{_pct(a['inband_disp'])} and {_pct(a['inband_egap'])}")
    _has("which supports a true fine reduction of 30% / 40% / 50% / 70% in "
         + " / ".join(_pct(b["power"][f"-{d}%"]) for d in (30, 40, 50, 70)) + " of trials")
    cut = b["c_star_fine_cut"]
    for name, path in (("transformer", "amp_phase2b.json"), ("bottleneck", "amp_e2_m1024.json")):
        sm = _j(W8 / "posthoc" / path)["seed_means"]["fine"]
        assert cut[name] == pytest.approx(1 - sm["disp_c"] / sm["disp"])
    assert b["k"] == pytest.approx(cut["transformer"] / cut["bottleneck"])
    line = sims["r2b_line"]
    cs = line["cases"]
    est, drawn = cs["spreads at their estimates"], cs["spreads drawn"]
    pm = drawn["params"]
    assert pm["k"] == b["k"] and pm["in_band"] == 0.0 and pm["prior"] == "uniform"
    assert pm["nb"] == 3 and pm["bottleneck_spreads"] == "estimates"
    assert pm["spreads"] == {"scale": 2.0, "df": 2, "shared": False, "cap": line["spread_cap"]}
    assert est["params"]["spreads"] == "estimates"
    assert line["value_for_the_line"] == 2.0 and line["values"] == [1.5, 2.0, 3.0]
    _has(f"Assume that S's reduction on the transformer is {b['k']:.2f} times its reduction on "
         "the bottleneck (the ratio of the two c* "
         f"reductions, {_pct(cut['transformer'])} and {_pct(cut['bottleneck'])}, Sec. 1), that "
         f"the bottleneck's true reduction is uniform on 0-{_pct(pm['prior_hi'])} (about c*'s "
         f"{_pct(cut['bottleneck'])}) and that S leaves the transformer's in-band errors "
         "unchanged; and value a supported R2b at twice its cost, a NOT SUPPORTED one at nothing",
         f"lies within {100 * line['window']:g} points of the line, passes one half (R2b more "
         "likely supported than not), and the line of the largest expected value lies there")
    near = ("-35%", "-40%", "-45%", "-50%")
    sens = sims["r2b_sensitivity"]
    post = line["posterior"]
    assert post["median_multiplier"] == pytest.approx(np.sqrt(2 / (2 * np.log(2))))
    assert post["p_more_than_twice"] == pytest.approx(1 - np.exp(-0.5 / 2))   # chi2_2 < 0.5
    _has("With the transformer's spreads at their three-seed estimates, that support near "
         "-35% / -40% / -45% / -50% is " + " / ".join(_pct(est["predicted"][x]) for x in near)
         + ", and the line falls near -35%",
         "But three seeds estimate a spread poorly, and the support depends strongly on it "
         f"(near -40%: {_pct(sens['transformer spreads x1.5']['predicted']['-40%'])} with the "
         "transformer's spreads 1.5 times their estimates, "
         f"{_pct(sens['transformer spreads x2']['predicted']['-40%'])} with twice)",
         "With each of the transformer's spreads drawn in every trial from its uncertainty given "
         "three seeds "
         f"(sigma^2 = {pm['spreads']['scale']:g} s^2 / chi2_{pm['spreads']['df']}, the posterior "
         f"under the prior 1/sigma: a spread's median {post['median_multiplier']:.1f} times its "
         f"estimate, more than twice it in {_pct(post['p_more_than_twice'])} of trials; at most "
         f"{line['spread_cap']:g} times its estimate), the support is "
         + " / ".join(_pct(drawn["predicted"][x]) for x in near)
         + f", it passes one half at {line_pct}%, and {line_pct}% is the line of the largest "
         "expected value")
    # "passes one half ... and the line of the largest expected value lies there"
    for c, want in ((est, -35), (drawn, line_pct)):
        assert round(c["half"]) == want and c["best"]["2"] == f"{want}%"

    def h(name):
        return f"{round(cs[name]['half']):d}%"
    priors = sorted(round(cs[k]["half"]) for k in ("spreads drawn, prior 0-60%",
                                                   "spreads drawn, prior half at 0",
                                                   "spreads drawn, prior decreasing"))
    assert cs["spreads drawn, no cap"]["params"]["spreads"]["cap"] is None
    assert cs["spreads drawn, at most 3 times"]["params"]["spreads"]["cap"] == 3.0
    assert cs["spreads drawn, at most 2 times"]["params"]["spreads"]["cap"] == 2.0
    assert cs["spreads drawn, one draw for the three metrics"]["params"]["spreads"]["shared"]
    for nm, m in (("spreads drawn, a prior worth 2 more degrees of freedom", 2),
                  ("spreads drawn, a prior worth 4 more degrees of freedom", 4)):
        sp = cs[nm]["params"]["spreads"]
        assert sp["scale"] == sp["df"] == 2 + m
    sp = cs["spreads drawn, the prior 1/sigma^2"]["params"]["spreads"]
    assert (sp["scale"], sp["df"]) == (2.0, 3)
    two = {h("spreads drawn, at most 2 times"), h("spreads drawn, one draw for the three metrics")}
    assert len(two) == 1
    _has(f"Sensitivity. The line stays near {line_pct}% without the cap or with it at 3 times "
         f"({h('spreads drawn, no cap')} and "
         f"{h('spreads drawn, at most 3 times')}), with the ratio drawn uniform on 0.74-1.0 "
         f"({h('spreads drawn, k uniform 0.74-1.0')}), with the bottleneck's spreads drawn too "
         f"({h('spreads drawn, the bottleneck' + chr(39) + 's too')}) and under three other "
         "priors (0-60%, half the mass at no effect, decreasing to 0 at 80%: "
         f"{priors[-1]}% to {priors[0]}%)",
         f"it moves to {h('spreads drawn, in band -10% to +10%')} if S changes the transformer's "
         f"in-band errors by up to 10% either way, and to {h('spreads drawn, k uniform 0.5-1.0')} "
         "with a lower ratio (uniform on 0.5-1.0)",
         f"Milder models of the spreads' uncertainty move it the other way: {two.pop()} with the "
         "spreads at most 2 times their estimates or one draw for the three metrics, "
         f"{h('spreads drawn, a prior worth 2 more degrees of freedom')} and "
         f"{h('spreads drawn, a prior worth 4 more degrees of freedom')} with a prior worth 2 or "
         f"4 more degrees of freedom at the estimates, {h('spreads drawn, the prior 1/sigma^2')} "
         "under the prior 1/sigma^2 (sigma^2 = 2 s^2 / chi2_3)")
    loss = line["loss"]
    assert sorted(loss["cases"]) == sorted(n for n, c in cs.items() if c["params"]["nb"] == 3)
    worst = {x: loss["by_line"][x]["max_loss"] for x in loss["by_line"]}
    assert min(worst, key=worst.get) == f"{line_pct}%"            # the smallest largest loss
    ev = {g: v["expected_value"]["2"] for g, v in drawn["gates"].items()}
    _has(f"Over these {len(loss['cases'])} three-seed cases, {line_pct}% loses at most "
         f"{_pct(worst['-45%'])} of a case's best expected value, -40% up to "
         f"{_pct(worst['-40%'])} and -50% up to {_pct(worst['-50%'])}; in the main case the "
         f"expected value is flat near its peak (-40% and -50% reach "
         f"{_pct(min(ev['-40%'], ev['-50%']) / ev['-45%'])} of it)",
         f"The value assumed moves the line most: with a supported R2b worth three times its "
         f"cost the best line is {drawn['best']['3']}, with 1.5 times about "
         f"{drawn['best']['1.5']}")
    # "moves the line most": more than any other assumption of the three-seed cases
    shift_v = max(abs(int(drawn["best"][v][:-1]) - line_pct) for v in ("1.5", "3"))
    shift_c = max(abs(round(cs[n]["half"]) - line_pct) for n in loss["cases"])
    assert shift_v > shift_c
    assert int(worst["-50%"] * 100 + 0.5) == 16 and drawn["best"]["1.5"] == "-60%"
    g = drawn["gates"]

    def rates(x):
        return (_pct(g[x]["p_recommended"]), _pct(g[x]["p_supported_given_recommended"]),
                _pct(g[x]["p_not_recommended_supported"]))
    r45, r40, r50 = rates("-45%"), rates("-40%"), rates("-50%")
    _has(f"At {line_pct}% (the spreads drawn, the 0-80% prior) R2b is recommended in {r45[0]} of "
         f"trials and supported after {r45[1]} of the recommendations, and it would be supported "
         f"but is not recommended in {r45[2]} of trials (at -40%: {r40[0]}, {r40[1]} and "
         f"{r40[2]}; at -50%: {r50[0]}, {r50[1]} and {r50[2]})")
    five = cs["spreads drawn, R2b five seeds"]
    assert five["params"]["nb"] == 5
    _has("Five seeds per arm (ten runs) raise the supported share of the recommendations ("
         f"{_pct(five['gates']['-45%']['p_supported_given_recommended'])} at -45%, the spreads "
         "drawn) but lower the expected value at every line, with a supported R2b worth 1.5 to 3 "
         "times the cost of six runs: R2b's own pre-registration should weigh its number of "
         "seeds against its cost.")
    for x, v in g.items():
        assert five["gates"][x]["p_supported_given_recommended"] > v["p_supported_given_recommended"]
        for val in ("1.5", "2", "3"):
            assert five["gates"][x]["expected_value"][val] < v["expected_value"][val]


def test_section_6_r3_numbers():
    sims = _j(SIMS)
    rc, fx, q, pr = sims["r3_cv"], sims["r3_fixed_reference"], sims["r3"], sims["r3_paired"]
    e, v = "energy_gap_rel", "vm_rel_l2"
    _has(f"coefficients of variation {_pct(rc[e])} for the energy gap, {_pct(rc[v])} for the "
         "von Mises error")
    p2b = _j(W8 / "e2" / "baseline" / "report_phase2b.json")
    ft = p2b["results"]["p3_transfer"]["metrics"]["fewshot"]["16"]["finetune"]
    for metric in (e, v):
        x = np.asarray(ft[metric]["per_seed"])
        assert rc[metric] == pytest.approx(x.std(ddof=1) / x.mean())

    def a(metric, cv, d, nd=0):
        return _pct(q[f"{metric}; new-arm cv {cv}; {d}"], nd)
    _has(f"a support without a true effect occurs in {a(e, 'x1', '-0%', 1)} (R3-E) and "
         f"{a(v, 'x1', '-0%', 1)} (R3-V) of trials, in {a(e, 'x1.5', '-0%', 1)} and "
         f"{a(v, 'x1.5', '-0%', 1)} if the energy arm spreads 1.5 times as much, and in "
         f"{a(e, '0.5', '-0%')} and {a(v, '0.5', '-0%')} if it spreads by 50%")
    zs = p2b["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["energy_gap_rel"]["per_seed"]
    _has(f"spread by {_pct(rc['zero_shot_energy_gap_rel'])}: " + " / ".join(f"{x:.3f}" for x in zs))
    _has("A true reduction of 10% / 20% / 30% / 50% is supported in "
         + " / ".join([a(e, "x1", "-10%"), a(e, "x1", "-20%"), a(e, "x1", "-30%"),
                       a(e, "x1", "-50%", 1)]) + " of trials (R3-E) and "
         + " / ".join([a(v, "x1", "-10%"), a(v, "x1", "-20%"), a(v, "x1", "-30%"),
                       a(v, "x1", "-50%")]) + " (R3-V)")

    def c(metric, d, nd=0):
        return _pct(pr[f"{metric}; rho 0.5; {d}"], nd)
    _has(f"a support without an effect occurs in {c(e, '-0%', 1)} (R3-E) and {c(v, '-0%', 1)} "
         "(R3-V) of trials, and a true reduction of 20% / 30% / 50% is supported in "
         + " / ".join([c(e, "-20%"), c(e, "-30%"), c(e, "-50%", 1)]) + " (R3-E) and "
         + " / ".join([c(v, "-20%"), c(v, "-30%", 1), c(v, "-50%")]) + " (R3-V)")
    assert pr[f"{e}; rho 0.5; -0%"] < q[f"{e}; new-arm cv x1; -0%"]
    x2 = sims["r3_spread_x2"]

    def w(metric, d, nd=0):
        return _pct(x2[f"{metric}; both arms' cv x2; {d}"], nd)
    assert w(e, "-0%", 1) == w(v, "-0%", 1)
    _has(f"with both arms' spreads twice the record's, a support without an effect occurs in "
         f"{w(e, '-0%', 1)} of trials (R3-E and R3-V), and a true reduction of 20% / 30% / 50% "
         "is supported in " + " / ".join(w(e, f"-{d}%") for d in (20, 30, 50)) + " (R3-E) and "
         + " / ".join(w(v, f"-{d}%") for d in (20, 30, 50)) + " (R3-V)")

    def f(metric, rec, d, nd=1):
        return _pct(fx[f"{metric}; record {rec}; new-arm cv x1; {d}"], nd)
    assert f(e, "+0%", "-0%") == f(v, "+0%", "-0%")
    _has(f"the rule's rate without an effect is {f(e, '+0%', '-0%')} if the record equals its "
         f"expectation but {f(e, '+10%', '-0%')} (R3-E) and {f(v, '+10%', '-0%', 0)} (R3-V) if it "
         "lies 10% above it")
    se = {m: np.asarray(ft[m]["per_seed"]).std(ddof=1) / np.sqrt(3) / np.mean(ft[m]["per_seed"])
          for m in (e, v)}
    # the record at 1.10 times the arms' true mean: it exceeds that mean by 0.10 / 1.10 of itself
    _has(f"(the excess, {_pct(0.10 / 1.10)} of the record, is about {0.10 / 1.10 / se[e]:.1f} and "
         f"{0.10 / 1.10 / se[v]:.1f} times the standard error of the record's mean)")


def test_section_7_and_9_numbers():
    amp = _j(W8 / "posthoc" / "amp_e2_m1024.json")
    u = [amp["seeds"][s]["remesh"]["u_norm_K_ratio_median_by_lc"]["0.0374"]
         for s in ("s0", "s1", "s2")]
    cb = [amp["seeds"][s]["remesh"]["c_battery_median_by_lc"] for s in ("s0", "s1", "s2")]
    _has(f"fell to {min(u):.2f}-{max(u):.2f} of its coarsest-mesh value at lc 0.0374",
         f"rose from {min(c['0.0906'] for c in cb):.2f}-{max(c['0.0906'] for c in cb):.2f} at lc "
         f"0.0906 to {min(c['0.0374'] for c in cb):.2f}-{max(c['0.0374'] for c in cb):.2f} at 0.0374")
    t = [amp["seeds"]["s0"][k]["summary"]["seconds"] for k in ("inband", "fine")]
    first = _j(W8 / "posthoc" / "amp_phase2b.json")["seeds"]["s0"]["remesh"]["seconds"]
    _has(f"wp8: {t[0]:.0f} s in band, {t[1]:.0f} s fine, and for the remesh pass {first:.0f} s on "
         "its first run, under a minute from its cache")
    assert amp["seeds"]["s0"]["remesh"]["seconds"] < 60
    bench = _j(W8 / "bench_e2_m1024.json")
    _has(f"the transformer's measured {bench['phases']['fine']['ms_per_step'] / 1000:.1f} s per "
         "fine step")
    # R2a's time: E2's arms ran back to back (RUNBOOK_E_SERIES 2c); a report is written at the end
    from datetime import datetime

    log = (W8 / "e2" / "e2_m1024" / "run.log").read_text()
    h, m, s = map(int, re.search(r"\[E8 \(AR pretrain\)\] done in (\d+):(\d+):(\d+)",
                                 log).groups())
    e8 = round((h * 3600 + m * 60 + s) / 60)
    ts = [datetime.fromisoformat(_j(W8 / "e2" / f"e2_m{n}" / "report.json")["provenance"]
                                 ["timestamp_utc"]) for n in (512, 1024)]
    arm = round((ts[1] - ts[0]).total_seconds() / 60)
    _has(f"E2's M = 1,024 arm wrote its report {arm // 60} h {arm % 60} min after the arm run "
         f"just before it: its three AR units took {e8 // 60} h {e8 % 60} min")
    rb = re.sub(r"\s+", " ", (ROOT / "RUNBOOK_FINE3D.md").read_text(encoding="utf-8"))
    assert f"its three AR units took {e8 // 60} h {e8 % 60} min" in rb


# ------------------------------------------------------------------- stamp --

def _draft_of(text, mod):
    from fejepa.report import PREREG_PLACEHOLDER

    m = re.search(r"\*\*Status:\*\* r\d+, stamped (.+?) before any run: .*?prohibited\.", text)
    if not m:
        return text, None
    text = text.replace(m.group(0), mod.DRAFT_STATUS)
    text = re.sub(r"^(CONFIG_SHA256\[[^\]]+\] = )[0-9a-f]{64}$", rf"\g<1>{PREREG_PLACEHOLDER}",
                  text, flags=re.M)
    text = re.sub(r"^PREREG_FINE3D_SHA256 = [0-9a-f]{64}$", mod.FOOTER_OPEN, text, flags=re.M)
    return text, m.group(1)


def test_the_approved_draft_is_pinned():
    mod = _mod("stamp_prereg_fine3d")
    draft, _ = _draft_of(PRE.read_text(encoding="utf-8"), mod)
    assert hashlib.sha256(draft.encode("utf-8")).hexdigest() == mod.DRAFT_SHA256


def test_stamping_the_draft(tmp_path):
    from fejepa.report import config_sha256, read_prereg_entries, verify_prereg

    mod = _mod("stamp_prereg_fine3d")
    committed = PRE.read_text(encoding="utf-8")
    draft, date = _draft_of(committed, mod)
    pf = tmp_path / "PREREG_FINE3D.md"
    pf.write_text(draft, encoding="utf-8")
    ok, here, want = mod.src_tree_ok()
    assert ok, (f"src is tree {here}, not the regression's {want}: run "
                "scripts/regress_against_branch.py again and update records/wp9/README.md")
    res = mod.stamp(pf, ROOT / "configs", date or "18 October 2026")
    text = pf.read_text(encoding="utf-8")
    for label, value in read_prereg_entries(pf):
        assert value == config_sha256(_j(ROOT / "configs" / f"{label}.json")) == \
            res["config_sha256"][label]
    assert mod.self_hash_ok(text) and text.rstrip("\n").endswith(res["prereg_sha256"])
    assert mod.stamped_status(date or "18 October 2026") in text and mod.DRAFT_STATUS not in text
    assert len(text.splitlines()) == len(draft.splitlines())
    if date is not None:
        assert text == committed
    with pytest.raises(SystemExit, match="not the approved draft"):
        mod.stamp(pf, ROOT / "configs", "18 October 2026")
    for label in res["config_sha256"]:                          # the runner's guard accepts
        assert verify_prereg(_j(ROOT / "configs" / f"{label}.json"), pf, label=label) == \
            res["config_sha256"][label]


def test_stamping_refuses_anything_but_the_draft(tmp_path, monkeypatch):
    mod = _mod("stamp_prereg_fine3d")
    draft, _ = _draft_of(PRE.read_text(encoding="utf-8"), mod)
    pf = tmp_path / "PREREG_FINE3D.md"
    pf.write_text(draft, encoding="utf-8")
    with pytest.raises(SystemExit, match="--date"):
        mod.stamp(pf, ROOT / "configs", "2026-10-18")
    monkeypatch.setattr(mod, "src_tree_ok", lambda: (True, "a" * 40, "a" * 40))
    cfgs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", cfgs)
    c = _j(cfgs / "r3_eft_b16.json")
    c["train"]["epochs"] = 60
    (cfgs / "r3_eft_b16.json").write_text(json.dumps(c, indent=1) + "\n")
    with pytest.raises(SystemExit, match="not the generator's output"):
        mod.stamp(pf, cfgs, "18 October 2026")
    edited = draft.replace("rel <= -45% in condition (i)", "rel <= -50% in condition (i)")
    assert edited != draft
    pf.write_text(edited, encoding="utf-8")
    with pytest.raises(SystemExit, match="not the approved draft"):
        mod.stamp(pf, ROOT / "configs", "18 October 2026")
    pf.write_text(draft, encoding="utf-8")
    monkeypatch.setattr(mod, "src_tree_ok", lambda: (False, "b" * 40, "a" * 40))
    with pytest.raises(SystemExit, match="not the tree the regression ran"):
        mod.stamp(pf, ROOT / "configs", "18 October 2026")
    assert pf.read_text(encoding="utf-8").count("<fill before tagging>") == 4   # untouched
