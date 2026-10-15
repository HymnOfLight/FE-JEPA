"""fine3d: every number PREREG_FINE3D.md quotes is recomputed from the records
it names; its lines are the three configurations' in the generator's order;
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
    cfg = _j(ROOT / "configs" / "r3_eft_b16.json")["content_check"]
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
    assert sims["r2b_fine_rel_max"] == -0.5
    _has("recommendation in "
         + " / ".join(p(f"S spread x1.0; inband unchanged; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " of trials with the in-band errors unchanged ("
         + " / ".join(p(f"S spread x1.0; inband 10% lower; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " if S also lowers them by 10%; "
         + " / ".join(p(f"S spread x1.0; inband 10% higher; fine -{d}%", "p_r2b_recommended")
                      for d in (50, 70)) + " if it raises them by 10%), one of 30% in under 0.01%")
    assert r["S spread x1.0; inband unchanged; fine -30%"]["p_r2b_recommended"] < 1e-4


def test_section_6_r3_numbers():
    sims = _j(SIMS)
    rc, fl, fx, q = sims["r3_cv"], sims["r3_tau_floor"], sims["r3_fixed_reference"], sims["r3"]
    e, v = "energy_gap_rel", "vm_rel_l2"
    _has(f"fixes tau at {_pct(fl[e], 1)} or more for R3-E and {_pct(fl[v], 1)} or more for R3-V",
         f"coefficients of variation {_pct(rc[e])} for the energy gap, {_pct(rc[v])} for the "
         "von Mises error")
    p2b = _j(W8 / "e2" / "baseline" / "report_phase2b.json")
    ft = p2b["results"]["p3_transfer"]["metrics"]["fewshot"]["16"]["finetune"]
    for metric in (e, v):
        x = np.asarray(ft[metric]["per_seed"])
        assert fl[metric] == pytest.approx(2 * x.std(ddof=1) / np.sqrt(3) / x.mean())

    def f(metric, rec, cv, d, nd=1):
        return _pct(fx[f"{metric}; record {rec}; new-arm cv {cv}; {d}"], nd)
    assert f(e, "+0%", "x1", "-0%") == f(v, "+0%", "x1", "-0%")
    _has(f"occurs in {f(e, '+0%', 'x1', '-0%')} of trials (R3-E and R3-V) if the record equals "
         f"its expectation, in {f(e, '+5%', 'x1', '-0%')} and {f(v, '+5%', 'x1', '-0%')} if it "
         f"lies 5% above it, and in {f(e, '+10%', 'x1', '-0%')} and {f(v, '+10%', 'x1', '-0%', 0)} "
         "if it lies 10% above it")
    se = {m: np.asarray(ft[m]["per_seed"]).std(ddof=1) / np.sqrt(3) / np.mean(ft[m]["per_seed"])
          for m in (e, v)}
    # the record at 1.10 times the arms' true mean: it exceeds that mean by 0.10 / 1.10 of itself
    _has(f"(about a {0.10 / 1.10 / se[e]:.1f}- and {0.10 / 1.10 / se[v]:.1f}-standard-error "
         "event for its mean)")
    lo_e = min(fx[f"{e}; record {r}; new-arm cv 0.5; -0%"] for r in ("+0%", "+5%", "+10%"))
    hi_e = max(fx[f"{e}; record {r}; new-arm cv 0.5; -0%"] for r in ("+0%", "+5%", "+10%"))
    lo_v = min(fx[f"{v}; record {r}; new-arm cv 0.5; -0%"] for r in ("+0%", "+5%", "+10%"))
    hi_v = max(fx[f"{v}; record {r}; new-arm cv 0.5; -0%"] for r in ("+0%", "+5%", "+10%"))
    _has(f"in {_pct(lo_e)[:-1]}-{_pct(hi_e)} (R3-E) and {_pct(lo_v)[:-1]}-{_pct(hi_v)} (R3-V)")
    zs = p2b["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["energy_gap_rel"]["per_seed"]
    _has(f"spread by {_pct(rc['zero_shot_energy_gap_rel'])}: " + " / ".join(f"{x:.3f}" for x in zs))
    _has("supported in " + " / ".join([f(e, "+0%", "x1", "-20%", 0), f(e, "+0%", "x1", "-30%", 0),
                                      f(e, "+0%", "x1", "-50%", 2)]) + " of trials (R3-E) and "
         + " / ".join([f(v, "+0%", "x1", "-20%", 0), f(v, "+0%", "x1", "-30%", 1),
                       f(v, "+0%", "x1", "-50%", 0)]) + " (R3-V) with the record at its expectation")
    _has(f"the rates without an effect are {_pct(q[e + '; new-arm cv x1; -0%'], 1)} (R3-E) and "
         f"{_pct(q[v + '; new-arm cv x1; -0%'], 1)} (R3-V), "
         f"{_pct(q[e + '; new-arm cv x1.5; -0%'], 1)} and {_pct(q[v + '; new-arm cv x1.5; -0%'], 1)} "
         f"at 1.5 times the spread, and {_pct(q[e + '; new-arm cv 0.5; -0%'])} and "
         f"{_pct(q[v + '; new-arm cv 0.5; -0%'])} at 50%")


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

    m = re.search(r"\*\*Status:\*\* r1, stamped (.+?) before any run: .*?prohibited\.", text)
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
    edited = draft.replace("rel <= -50% in condition (i)", "rel <= -40% in condition (i)")
    assert edited != draft
    pf.write_text(edited, encoding="utf-8")
    with pytest.raises(SystemExit, match="not the approved draft"):
        mod.stamp(pf, ROOT / "configs", "18 October 2026")
    pf.write_text(draft, encoding="utf-8")
    monkeypatch.setattr(mod, "src_tree_ok", lambda: (False, "b" * 40, "a" * 40))
    with pytest.raises(SystemExit, match="not the tree the regression ran"):
        mod.stamp(pf, ROOT / "configs", "18 October 2026")
    assert pf.read_text(encoding="utf-8").count("<fill before tagging>") == 3   # untouched
