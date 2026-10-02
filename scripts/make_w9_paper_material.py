#!/usr/bin/env python3
"""wp9 Stage 0c: paper material for PREREG_W9, from the adjudication's verdict
file only (records/wp9/w9_verdict.json, written by scripts/adjudicate_w9.py).

Writes to --out (default paper/wp9):
  table_w9_h1.tex       H1: the in-band energy gap of the largest pool against
                        the 1,024 arm, per seed, with the guard and the verdict
  table_w9_c1.tex       C1: energy gap of every C1 arm on every set (in band:
                        the validation split and IB together and each on its
                        own; the in-band tail; F1-F5; R)
  table_w9_c1_disp.tex  C1: the same for the displacement error
  table_w9_h2.tex       H2: S against the fresh baseline -- the three
                        conditions' quantities and the verdict (if S ran)
  table_w9_s_sets.tex   S against the fresh baseline on every set
                        (exploratory; if S ran)
  table_w9_remesh.tex   R: the label-free amplitude c_b per mesh size, every arm
  tables.md             the same tables in Markdown
  sources.json          the verdict file's SHA-256

Every number is read from the verdict file; nothing is typed by hand. The
outputs are deterministic. Seed statistics are the mean and the SAMPLE
standard deviation (n - 1) over the seeds an arm has. The exploratory
comparisons carry the Sec. 6 guard's numbers but no verdict.

    python scripts/make_w9_paper_material.py --verdict records/wp9/w9_verdict.json --out paper/wp9
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = ("inband", "val", "IB", "F1", "F2", "F3", "F4", "F5", "R")
SET_HEAD = {"inband": "In band\n(val + IB)", "val": "Validation\nsplit", "IB": "IB\nin band",
            "F1": "F1\n4-6 holes", "F2": "F2\nslender", "F3": "F3\nhole size",
            "F4": "F4\nPoisson", "F5": "F5\nfine mesh", "R": "R\nremesh"}
C1_ROLES = ("n1024", "b1024", "n4096", "nmax")
S_ROLES = ("b1024", "s")
ROLE_NAME = {"n1024": "E1's states", "n4096": "AR", "nmax": "AR", "b1024": "AR, fresh seeds",
             "s": "AR with S, fresh seeds"}


def _wp8():
    spec = importlib.util.spec_from_file_location(
        "make_wp8_paper_material", ROOT / "scripts" / "make_wp8_paper_material.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W8 = _wp8()
_signed_pct = W8._signed_pct


def _g(x, sig: int = 3) -> str:
    """wp8's number format; a non-finite value (a diverged seed) prints as such."""
    return W8._g(x, sig) if isinstance(x, (int, float)) and math.isfinite(x) else "diverged"


def _ms(v, sig: int = 3, dec=None) -> str:
    """wp8's mean (± sample SD); a cell with a non-finite seed prints as diverged."""
    vals = [float(x) for x in v]
    return W8._ms(vals, sig, dec) if all(math.isfinite(x) for x in vals) else "diverged"


# ---------------------------------------------------------------- reading

def _vals(tab: dict, set_name: str, key: str):
    v = ((tab or {}).get(set_name) or {}).get(key) or {}
    return v.get("per_seed") if isinstance(v, dict) and "per_seed" in v else None


def _cell(tab, set_name, key, sig=3) -> str:
    v = _vals(tab, set_name, key)
    return _ms(v, sig) if v else "n/a"


def _pool(arm: dict) -> str:
    p = (arm.get("identity") or {}).get("pool")
    return f"{int(p):,}" if p else "n/a"


def _epochs(arm: dict) -> str:
    e = (arm.get("identity") or {}).get("epochs")
    return f"{int(e):,}" if e else "n/a"


def _seeds(arm: dict) -> str:
    st = sorted(int(k[1:]) for k in (arm.get("states") or {}))
    return f"{st[0]}-{st[-1]}" if st else "n/a"


def _robust_note(name: str, r) -> str:
    """PREREG_W9 r3, Sec. 7: the readings beside a verdict (no criterion)."""
    if not isinstance(r, dict) or "not_evaluated" in r:
        return f"{name}: robustness readings not evaluated."
    parts = []
    m = r.get("medians") or {}
    if m.get("rel_change") is not None:
        state = "lower" if m.get("lower") else "worse" if m.get("worse") else "within the guard"
        parts.append(f"per-seed medians {_signed_pct(m['rel_change'])} (threshold "
                     f"{_signed_pct(m['threshold']).lstrip('+')}; {state})")
    for key, what in (("welch_95", "Welch 95% interval"),
                      ("instance_resampling_95", "instance-resampling 95% interval")):
        w = r.get(key) or {}
        if "low" in w:
            parts.append(f"{what} {_signed_pct(w['low'])} to {_signed_pct(w['high'])}")
    return (f"{name}, beside the verdict (no criterion): " + "; ".join(parts) + "."
            if parts else f"{name}: robustness readings not evaluated.")


def _guard_note(name: str, g: dict) -> str:
    if not isinstance(g, dict) or g.get("rel_change") is None:
        return f"{name}: {'diverged' if (g or {}).get('diverged') else 'not computed'}."
    state = "lower" if g.get("lower") else "worse" if g.get("worse") else "within the guard"
    return (f"{name}: change {_signed_pct(g['rel_change'])}, threshold "
            f"{_signed_pct(g['threshold']).lstrip('+')} (max(10%, 2 SE_rel)); {state}.")


# --------------------------------------------------------------- the tables

def h1_table(v: dict) -> tuple:
    h1, arms, tabs = v["H1"], v.get("arms", {}), v.get("tables", {})
    head = ["Arm", "Pool", "Epochs", "Seeds", "Energy gap\nper seed", "Energy gap\nmean ± SD",
            "Validation\nsplit", "IB"]
    rows = []
    for role in ("n1024", "nmax", "b1024"):
        if role not in arms:
            continue
        per = _vals(tabs.get(role), "inband", "energy_gap_rel") or []
        rows.append([ROLE_NAME[role], _pool(arms[role]), _epochs(arms[role]), _seeds(arms[role]),
                     " / ".join(_g(x) for x in per), _ms(per) if per else "n/a",
                     _cell(tabs.get(role), "val", "energy_gap_rel"),
                     _cell(tabs.get(role), "IB", "energy_gap_rel")])
    notes = [f"Verdict: {h1['verdict']}."]
    if h1.get("rel_change") is not None:
        notes.append(_guard_note("N_max against 1,024", h1))
        notes.append(_robust_note("N_max against 1,024", h1.get("robustness")))
    if h1.get("fresh_baseline_flag"):
        g = ((v.get("replication_b1024_vs_n1024") or {}).get("inband") or {}).get(
            "energy_gap_rel")
        notes.append("Flag (a coarse check of the reuse; the verdict stands): " + (
            _guard_note("the fresh baseline against E1's states, in-band energy gap", g)
            if isinstance(g, dict) and g.get("rel_change") is not None
            else "the fresh baseline's in-band energy gap is non-finite."))
    notes.append("Relative energy gap in band: per seed, the mean over E1's validation split "
                 "(256 instances) and the in-band holdout IB (2,048 fresh training-family "
                 "instances) together, the verdict's quantity; the last two columns are the two "
                 "sets' seed means ± SD. 204,800 label-free AR steps per seed in every arm; the "
                 "1,024 arm evaluates E1's base states; the fresh baseline is E1's configuration "
                 "trained again on seeds 3-5, a coarse check of that reuse (PREREG_W9 Sec. 2, 7).")
    return head, rows, notes


def c1_table(v: dict, key: str, what: str) -> tuple:
    arms, tabs = v.get("arms", {}), v.get("tables", {})
    head = ["Pool", "Seeds"] + [SET_HEAD[s] for s in SETS]
    if key == "energy_gap_rel":
        head.insert(3, "In band\ntail p90")
    rows = []
    for role in C1_ROLES:
        if role not in arms:
            continue
        row = [_pool(arms[role]), _seeds(arms[role])]
        for s in SETS:
            row.append(_cell(tabs.get(role), s, key))
            if s == "inband" and key == "energy_gap_rel":
                row.append(_cell(tabs.get(role), s, "tail_energy_gap_rel"))
        rows.append(row)
    notes = [f"{what}: seed mean ± sample SD of the per-seed means over each set's instances; "
             "the tail is the per-seed 90th percentile of the per-instance energy gap.",
             "In band: E1's validation split (256) and IB (2,048 fresh training-family "
             "instances) together, H1's set; F1-F5: OOD-2D v1 (256 instances each); R: 16 "
             "geometries at 5 mesh sizes. Evaluation only; secondary readings (PREREG_W9 "
             "Sec. 7)."]
    return head, rows, notes


def h2_table(v: dict) -> tuple:
    h2, arms, tabs = v["H2"], v.get("arms", {}), v.get("tables", {})
    head = ["Arm", "Seeds", "In-band\nenergy gap", "In-band\ndisplacement", "F5\ndisplacement",
            "F5 / in-band\ndisplacement", "Median c*\non F5"]
    rows = []
    for role in S_ROLES:
        if role not in arms:
            continue
        tab = tabs.get(role) or {}
        ratio = (tab.get("ratio_F5_over_val_disp") or {}).get("per_seed")
        rows.append([ROLE_NAME[role], _seeds(arms[role]), _cell(tab, "val", "energy_gap_rel"),
                     _cell(tab, "val", "disp_rel_l2"), _cell(tab, "F5", "disp_rel_l2"),
                     _ms(ratio) if ratio else "n/a", _cell(tab, "F5", "c_star_median")])
    notes = [f"Verdict: {h2['verdict']}."]
    for key, name in (("F5_displacement", "(i) F5 displacement"),
                      ("F5_over_inband_ratio", "(ii) F5 / in-band ratio"),
                      ("K1_inband_energy_gap", "(iii) in-band energy gap"),
                      ("K1_inband_displacement", "(iii) in-band displacement")):
        if key in h2:
            notes.append(_guard_note(name, h2[key]))
            if key in (h2.get("robustness") or {}):
                notes.append(_robust_note(name, h2["robustness"][key]))
    if h2.get("uninformative"):
        notes.append(f"Note: {h2['uninformative']}.")
    notes.append("S: decode scale 1/64 of the battery's summed |F| and per-node load densities; "
                 "the baseline is E1's configuration on the same fresh seeds (PREREG_W9 Sec. 2).")
    return head, rows, notes


def s_sets_table(v: dict) -> tuple:
    tabs, cmp_ = v.get("tables", {}), v.get("comparisons_s_vs_b1024_exploratory") or {}
    head = ["Set", "Baseline\nenergy gap", "S\nenergy gap", "Change", "Baseline\ndisplacement",
            "S\ndisplacement", "Change "]
    rows = []
    for s in SETS:
        row = [SET_HEAD[s].replace("\n", " ")]
        for key in ("energy_gap_rel", "disp_rel_l2"):
            g = (cmp_.get(s) or {}).get(key) or {}
            ch = _signed_pct(g["rel_change"]) if g.get("rel_change") is not None else "n/a"
            if g.get("lower"):
                ch += " (lower)"
            elif g.get("worse"):
                ch += " (worse)"
            row += [_cell(tabs.get("b1024"), s, key), _cell(tabs.get("s"), s, key), ch]
        rows.append(row)
    notes = ["Exploratory (PREREG_W9 Sec. 7): each change is S against the fresh baseline with "
             "the Sec. 6 guard computed (lower / worse beyond max(10%, 2 SE_rel)); no verdict."]
    return head, rows, notes


def remesh_table(v: dict) -> tuple:
    rem, arms = v.get("remesh") or {}, v.get("arms", {})
    roles = [r for r in dict.fromkeys((*C1_ROLES, *S_ROLES))
             if r in arms and isinstance(rem.get(r), dict) and "not_evaluated" not in rem[r]]
    hs = sorted({h for r in roles for h in rem[r]}, key=float, reverse=True)
    head = ["Arm", "Pool"] + [f"h {float(h):g}" for h in hs]
    rows, extra = [], []
    for role in roles:
        rows.append([ROLE_NAME[role], _pool(arms[role])]
                    + [_ms(rem[role][h]["c_battery_median"], dec=2) for h in hs])
        extra.append(f"{ROLE_NAME[role]} ({_pool(arms[role])}): predicted energy norm relative "
                     "to the coarsest mesh, " + " / ".join(
                         _g(sum(rem[role][h]["u_norm_ratio_median"])
                            / len(rem[role][h]["u_norm_ratio_median"])) for h in hs) + ".")
    if roles:
        r0 = rem[roles[0]]
        extra.append("The exact solution's own energy norm relative to the coarsest mesh: "
                     + " / ".join(_g(sum(r0[h]["ustar_norm_ratio_median"])
                                     / len(r0[h]["ustar_norm_ratio_median"])) for h in hs) + ".")
    growth = v.get("remesh_growth") or {}
    for role in roles:
        g = growth.get(role)
        if isinstance(g, dict) and g and "not_evaluated" not in g:
            gh = sorted(g, key=float, reverse=True)
            extra.append(f"{ROLE_NAME[role]} ({_pool(arms[role])}): displacement error at the "
                         "finest mesh over that at h " + " / ".join(f"{float(h):g}" for h in gh)
                         + ": " + " / ".join(_ms(g[h]) for h in gh) + ".")
    for h, gd in sorted((v.get("remesh_growth_s_vs_b1024_exploratory") or {}).items(),
                        key=lambda kv: -float(kv[0]) if kv[0] != "not_evaluated" else 0):
        if isinstance(gd, dict) and gd.get("rel_change") is not None:
            extra.append(_guard_note(f"S against the fresh baseline, growth over h {float(h):g} "
                                     "(exploratory)", gd))
    notes = ["c_b: the battery-level energy-optimal amplitude (label-free); median over the 16 "
             "geometries, then seed mean ± sample SD. Training mesh sizes 0.05-0.12."] + extra
    return head, rows, notes


# ----------------------------------------------------------------- output

_TEX = (("PREREG_W9", r"PREREG\_W9"), ("N_max", r"$N_\mathrm{max}$"), (" < ", r" $<$ "),
        (" > ", r" $>$ "))


def _tex(s: str) -> str:
    """wp9's own names in LaTeX (wp8's escaping refuses a raw underscore)."""
    for a, b in _TEX:
        s = s.replace(a, b)
    return s


def latex_table(head, rows, caption, label, notes) -> str:
    text = W8.latex_table([_tex(h) for h in head], [[_tex(c) for c in r] for r in rows],
                          _tex(caption), label, [_tex(n) for n in notes])
    first, rest = text.split("\n", 1)
    return ("% generated by scripts/make_w9_paper_material.py from records/wp9/w9_verdict.json; "
            "needs booktabs and graphicx\n" + rest)


CAPTIONS = {
    "h1": ("H1 (2D): does a larger unlabelled pool lower the in-band energy gap at a fixed "
           "budget of label-free AR steps?", "tab:w9-h1"),
    "c1": ("C1 (2D): relative energy gap of every pool size on every evaluation set.",
           "tab:w9-c1"),
    "c1_disp": ("C1 (2D): relative displacement error of every pool size on every evaluation set.",
                "tab:w9-c1-disp"),
    "h2": ("H2 (2D): a mesh-independent decode scale and load input (S) against the freshly "
           "trained baseline.", "tab:w9-h2"),
    "s_sets": ("S against the fresh baseline on every evaluation set (exploratory).",
               "tab:w9-s-sets"),
    "remesh": ("Remesh set R (2D): the label-free amplitude c_b on fixed geometries and loads "
               "meshed at five mesh sizes.", "tab:w9-remesh"),
}


def build(verdict: dict, verdict_sha256: str) -> dict:
    parts = [("h1", "table_w9_h1.tex", h1_table(verdict)),
             ("c1", "table_w9_c1.tex", c1_table(verdict, "energy_gap_rel",
                                                "Relative energy gap")),
             ("c1_disp", "table_w9_c1_disp.tex", c1_table(verdict, "disp_rel_l2",
                                                          "Relative displacement error"))]
    if {"b1024", "s"} <= set(verdict.get("arms", {})) and "F5_displacement" in verdict["H2"]:
        parts += [("h2", "table_w9_h2.tex", h2_table(verdict)),
                  ("s_sets", "table_w9_s_sets.tex", s_sets_table(verdict))]
    parts.append(("remesh", "table_w9_remesh.tex", remesh_table(verdict)))
    files = {}
    md = ["# wp9 tables (generated from records/wp9/w9_verdict.json by "
          "scripts/make_w9_paper_material.py; do not edit)", ""]
    for key, name, (head, rows, notes) in parts:
        caption, label = CAPTIONS[key]
        files[name] = latex_table(head, rows, caption, label, notes)
        md.append(W8.md_table(head, rows, caption, notes))
    if "F5_displacement" not in verdict["H2"]:
        md.append(f"H2: {verdict['H2']['verdict']}.\n")
    if verdict.get("refused_reports"):
        md.append("Refused reports: " + "; ".join(f"{k}: {v}" for k, v in
                                                  sorted(verdict["refused_reports"].items())) + "\n")
    md.append("Deviations: " + ("; ".join(verdict.get("deviations") or []) or "none") + "\n")
    files["tables.md"] = "\n".join(md)
    files["sources.json"] = json.dumps({"generator": "scripts/make_w9_paper_material.py",
                                        "verdict_sha256": verdict_sha256}, indent=1) + "\n"
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict", default=str(ROOT / "records" / "wp9" / "w9_verdict.json"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "wp9"))
    a = ap.parse_args()
    raw = Path(a.verdict).read_bytes()
    files = build(json.loads(raw), hashlib.sha256(raw).hexdigest())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    print("\n".join(str(out / n) for n in files))


if __name__ == "__main__":
    main()
