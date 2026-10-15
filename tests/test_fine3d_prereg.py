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
    assert sims["r2b_fine_rel_max"] == -0.4
    _has("recommendation in "
         + " / ".join(p(f"S spread x1.0; inband unchanged; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " of trials with the in-band errors unchanged ("
         + " / ".join(p(f"S spread x1.0; inband 10% lower; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " if S also lowers them by 10%; "
         + " / ".join(p(f"S spread x1.0; inband 10% higher; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " if it raises them by 10%), one of 30% in "
         + p("S spread x1.0; inband unchanged; fine -30%", "p_r2b_recommended", nd=1) + ".")


def test_section_6_r2b_numbers():
    """The R2b line and its rationale (`r2b`, `r2b_sensitivity`), from the
    Phase-2b spreads and wp8's post-hoc records."""
    from fejepa.analysis.adjudicate_fine3d import R2B_FINE_REL_MAX

    sims = _j(SIMS)
    assert R2B_FINE_REL_MAX == sims["r2b_fine_rel_max"] == -0.4
    _has("is at most 60% of the fresh baseline's (rel <= -40% in condition (i))")
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
    _has(f"coefficients of variation {_pct(b['cv']['fine'])} for the fine displacement error, "
         f"{_pct(b['cv']['inband_disp'])} and {_pct(b['cv']['inband_egap'])} for the in-band "
         f"displacement error and energy gap, against the bottleneck's {_pct(a['fine'])}, "
         f"{_pct(a['inband_disp'])} and {_pct(a['inband_egap'])}")
    _has("supports a true fine reduction of 30% / 40% / 50% / 70% in "
         + " / ".join(_pct(b["power"][f"-{d}%"]) for d in (30, 40, 50, 70)) + " of trials")
    cut = b["c_star_fine_cut"]
    for name, path in (("transformer", "amp_phase2b.json"), ("bottleneck", "amp_e2_m1024.json")):
        sm = _j(W8 / "posthoc" / path)["seed_means"]["fine"]
        assert cut[name] == pytest.approx(1 - sm["disp_c"] / sm["disp"])
    assert b["k"] == pytest.approx(cut["transformer"] / cut["bottleneck"])
    _has(f"c* lowers the transformer's fine error by {_pct(cut['transformer'])} and the "
         f"bottleneck's by {_pct(cut['bottleneck'])} (Sec. 1)",
         f"is {b['k']:.2f} times its reduction on the bottleneck",
         f"uniform on 0-80% (about c*'s {_pct(cut['bottleneck'])})")
    pk, p6, p10 = (b["predicted"][k] for k in ("k", "0.6", "1.0"))
    sens = sims["r2b_sensitivity"]
    _has(f"lies within 2.5 points of -40% is followed by a supported R2b in {_pct(pk['-40%'])} of "
         f"trials (near -30%: {_pct(pk['-30%'])}; near -35%: {_pct(pk['-35%'])}; near -50%: "
         f"{_pct(pk['-50%'])}; near -40% with the ratio 0.6 or 1.0 instead: {_pct(p6['-40%'])} "
         f"or {_pct(p10['-40%'])}; with the prior on 0-60%: "
         f"{_pct(sens['prior 0-60%']['predicted']['-40%'])}; with the transformer's spreads "
         f"doubled: {_pct(sens['transformer spreads x2']['predicted']['-40%'])})")
    assert pk["-35%"] < 0.505 < 0.55 < pk["-40%"]               # "passes one half near -35%"
    gx = sens["transformer spreads x2"]["gates"]
    _has("With the transformer's spreads doubled, R2b is supported after about half of the "
         f"recommendations at -40% ({_pct(gx['-40%']['p_supported_given_recommended'])}; "
         f"{_pct(gx['supported']['p_supported_given_recommended'])} on R2a SUPPORTED alone, "
         f"{_pct(gx['-50%']['p_supported_given_recommended'])} at -50%)")
    assert 0.4 < gx["-40%"]["p_supported_given_recommended"] < 0.6   # "about half"
    g = b["gates"]["k"]
    _has(f"recommending at -40% (in {_pct(g['-40%']['p_recommended'])} of trials) is followed by "
         f"a supported R2b in {_pct(g['-40%']['p_supported_given_recommended'])} of the "
         f"recommendations; recommending on R2a SUPPORTED alone, in "
         f"{_pct(g['supported']['p_supported_given_recommended'])} "
         f"({_pct(g['supported']['p_recommended'])} of trials recommended); at -50%, in "
         f"{_pct(g['-50%']['p_supported_given_recommended'])} "
         f"({_pct(g['-50%']['p_recommended'])})",
         f"R2b would be supported but is not recommended in "
         f"{_pct(g['-40%']['p_not_recommended_supported'])} of trials at -40%, "
         f"{_pct(g['supported']['p_not_recommended_supported'])} on R2a SUPPORTED alone and "
         f"{_pct(g['-50%']['p_not_recommended_supported'])} at -50%")


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
    edited = draft.replace("rel <= -40% in condition (i)", "rel <= -50% in condition (i)")
    assert edited != draft
    pf.write_text(edited, encoding="utf-8")
    with pytest.raises(SystemExit, match="not the approved draft"):
        mod.stamp(pf, ROOT / "configs", "18 October 2026")
    pf.write_text(draft, encoding="utf-8")
    monkeypatch.setattr(mod, "src_tree_ok", lambda: (False, "b" * 40, "a" * 40))
    with pytest.raises(SystemExit, match="not the tree the regression ran"):
        mod.stamp(pf, ROOT / "configs", "18 October 2026")
    assert pf.read_text(encoding="utf-8").count("<fill before tagging>") == 4   # untouched
