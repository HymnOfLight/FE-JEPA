#!/usr/bin/env python3
"""The CMAME manuscript's tables, figures and in-text numbers, generated from
the committed records only (no number in the manuscript is typed by hand).

Inputs (their SHA-256 values are written to sources.json):

* records/phase1/report_rec8_v2.json -- the two-dimensional deciding run of
  16 July 2026 (configuration SHA-256 62b26ad8...);
* records/phase1/report_diag.json -- the two-dimensional diagnostic run of
  30 July 2026 (exploratory; read for its six further label-free trainings);
* records/phase1/report_wp2_e2.json -- the two-dimensional WP2/E2 run of
  31 July 2026 (read for its pre-registered criterion K3);
* records/wp8/e2/baseline/report_phase2b.json -- the three-dimensional run of
  28 September 2026 (Phase-2b);
* records/wp8/posthoc/amp_*.json and the other wp8 records read through
  scripts/make_wp8_paper_material.py (post-hoc amplitude readings, E1, E2).

Outputs (paper/cmame/generated/):

* table_2d.tex, table_3d.tex -- every metric at the largest label budget;
* table_label_efficiency.tex -- displacement error and energy gap per budget;
* table_transfer.tex -- the finer three-dimensional mesh, zero-shot and few-shot;
* table_criteria.tex -- every pre-registered criterion with its outcome;
* table_amplitude.tex, table_remesh.tex, table_e1.tex, table_e2.tex -- the
  wp8 tables' rows (shared code) with this manuscript's captions;
* numbers.tex -- one macro per number quoted in the text;
* fig_disp_vs_stress.pdf, fig_energy_gap.pdf, fig_label_efficiency.pdf;
* material.md (the tables and numbers in Markdown) and sources.json.

Seed statistics are the mean and the sample standard deviation (n - 1) of the
per-seed values. "Worse than the zero field" means a relative energy gap above
1 (the zero field's is exactly 1); for an instance it is the mean over its
four load cases, as the evaluation records it; for a single load case it is
equivalent to Pi_h(u) > 0 (Corollary 3 of the manuscript).

The text outputs are deterministic: regenerating from the records reproduces
the committed files byte for byte (tests/test_cmame_material.py).

    python scripts/make_cmame_material.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
from math import floor, log10
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "cmame" / "generated"
REC8 = ROOT / "records" / "wp8"
R2D = ROOT / "records" / "phase1" / "report_rec8_v2.json"
RDIAG = ROOT / "records" / "phase1" / "report_diag.json"
RWP2 = ROOT / "records" / "phase1" / "report_wp2_e2.json"
R3D = REC8 / "e2" / "baseline" / "report_phase2b.json"
AMP = {"transformer": REC8 / "posthoc" / "amp_phase2b.json",
       "m512": REC8 / "posthoc" / "amp_e2_m512.json",
       "m1024": REC8 / "posthoc" / "amp_e2_m1024.json"}
E1_BASE = REC8 / "e1" / "e1_2d_base" / "report.json"
MINUS = "−"
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten")
BUDGETS = ("16", "64", "256", "1024")
METRICS = ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err", "crit_recall")
NAMES = {"ar": "Label-free", "labels": "Supervised",
         "labels_anchor": "Supervised + energy term", "ar_ft": "Label-free, then supervised",
         "mgn": "Graph network, supervised", "knn_field": "Nearest-neighbour field",
         "scale_aware_poly": "Scaled polynomial", "zero": "Zero field"}
ARMS_2D = ("ar", "labels", "labels_anchor", "ar_ft", "mgn", "knn_field", "scale_aware_poly",
           "zero")
ARMS_3D = ("ar", "labels", "labels_anchor", "mgn", "knn_field", "scale_aware_poly", "zero")
TRAINED = ("ar", "labels", "labels_anchor", "ar_ft", "mgn")
# figure identity: colour AND marker (validated categorical palette)
STYLE = {"ar": ("#2a78d6", "o"), "labels": ("#e87ba4", "s"),
         "labels_anchor": ("#eda100", "D"), "mgn": ("#4a3aa7", "^")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _wp8():
    spec = importlib.util.spec_from_file_location(
        "make_wp8_paper_material", ROOT / "scripts" / "make_wp8_paper_material.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _wp8()


def _rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def load(path: Path, used: dict) -> dict:
    data = Path(path).read_bytes()
    used[_rel(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


# ---------------------------------------------------------------- reading
def cells(report: dict) -> dict:
    return report["results"]["e8"]["metrics"]["cells"]


def _cell(report: dict, arm: str, b: str) -> dict:
    c = cells(report)[arm]
    return c[b] if b in c else c[max(c, key=int)]   # the label-free cell is keyed by its pool


def seed_values(report: dict, arm: str, b: str, metric: str) -> list:
    return [float(x) for x in _cell(report, arm, b)[metric]["per_seed"]]


def seed_mean(report: dict, arm: str, metric: str, b: str = "1024") -> float:
    return statistics.fmean(seed_values(report, arm, b, metric))


def per_instance(report: dict, arm: str, metric: str, b: str = "1024") -> np.ndarray:
    """(seeds, instances) array of one metric."""
    rows = [e["per_instance"][metric] for e in _cell(report, arm, b)["per_seed_eval"]]
    return np.array(rows, dtype=float)


def worse_than_zero(report: dict, arm: str, b: str = "1024") -> tuple:
    g = per_instance(report, arm, "energy_gap_rel", b)
    return int(np.sum(g > 1.0)), int(g.size)


# ---------------------------------------------------------------- formatting
def num(x: float, sig: int = 3) -> str:
    """`sig` significant digits, fixed notation, thousands separators, typographic minus."""
    if x == 0:
        return "0"
    dec = max(0, sig - 1 - int(floor(log10(abs(x)))))
    return (MINUS if x < 0 else "") + f"{abs(x):,.{dec}f}"


def ms(values, sig: int = 3) -> str:
    """Seed mean ± sample SD at the mean's precision (as the wp8 tables, with
    thousands separators); one value: the value."""
    v = [float(x) for x in values]
    txt = num(statistics.fmean(v), sig)
    if len(v) == 1:
        return txt
    d = len(txt.split(".")[1]) if "." in txt else 0
    return f"{txt} ± {statistics.stdev(v):,.{d}f}"


def pct(x: float, dec: int = 1) -> str:
    return f"{x * 100:.{dec}f}%"


def spct(x: float, dec: int = 1) -> str:
    return f"{x * 100:+.{dec}f}%".replace("-", MINUS)


def times(x: float, dec: int = 1) -> str:
    return f"{x:.{dec}f}"


def words(n: int) -> str:
    """A count as the text writes it: in words up to ten, in digits above."""
    return WORDS[n] if 0 <= n <= 10 else f"{n:,}"


def share(win: np.ndarray) -> str:
    """The share of pairs a comparison won, as the text writes it ("all" when every one)."""
    return "all" if bool(win.all()) else pct(float(win.mean()))


def tex(s: str) -> str:
    """Text-mode LaTeX for a table cell, note or macro (wp8 escaping plus a few symbols)."""
    s = (s.replace("<", "$<$").replace(">", "$>$")
         .replace("≥", "$\\geq$").replace("≤", "$\\leq$").replace("#", "\\#")
         .replace("&", "\\&").replace("⁻³", "$^{-3}$").replace("′", "$'$"))
    s = re.sub(r"\blc (?=[0-9])", "lc = ", s)           # "lc 0.0374" -> "lc = 0.0374"
    s = re.sub(r"\blc\b", "LCMARK", s)                 # the characteristic length l_c
    return W._latex_escape(s).replace("LCMARK", "$l_c$")


# ---------------------------------------------------------------- tables
def table(head: list, rows: list, caption: str, label: str, notes: list,
          align: str | None = None, rules: tuple = ()) -> str:
    cols = align or ("l" + "c" * (len(head) - 1))
    out = [r"% generated by scripts/make_cmame_material.py from the records; do not edit",
           r"\begin{table}[tbp]", r"\centering", r"\footnotesize",
           r"\setlength{\tabcolsep}{4pt}",
           rf"\caption{{{tex(caption)}}}", rf"\label{{{label}}}",
           r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           " & ".join(_head(h) for h in head) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in rules:
            out.append(r"\midrule")
        out.append(" & ".join(tex(c) for c in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}}", "", r"\smallskip",
            r"\begin{minipage}{\linewidth}\footnotesize\raggedright"]
    out += [tex(n) + (r"\par" if i < len(notes) - 1 else "") for i, n in enumerate(notes)]
    out += [r"\end{minipage}", r"\end{table}", ""]
    return "\n".join(out)


RAGGED = r">{\raggedright\arraybackslash}"          # a ragged-right column


def longtable(head: list, rows: list, caption: str, label: str, notes: list, align: str,
              rules: tuple = ()) -> str:
    """A table that may break across pages (needs the longtable package)."""
    hd = " & ".join(_head(h) for h in head) + r" \\"
    out = [r"% generated by scripts/make_cmame_material.py from the records; do not edit",
           r"\begingroup\small",
           rf"\begin{{longtable}}{{{align}}}",
           rf"\caption{{{tex(caption)}}}\label{{{label}}}\\", r"\toprule", hd, r"\midrule",
           r"\endfirsthead",
           rf"\caption[]{{{tex(caption)} (continued)}}\\", r"\toprule", hd, r"\midrule",
           r"\endhead", r"\bottomrule", r"\endfoot"]
    for i, r in enumerate(rows):
        if i in rules:
            out.append(r"\midrule")
        out.append(" & ".join(tex(c) for c in r) + r" \\")
    out += [r"\end{longtable}", r"\endgroup", r"\vspace{-\smallskipamount}",
            r"\noindent\begin{minipage}{\linewidth}\footnotesize\raggedright"]
    out += [tex(n) + (r"\par" if i < len(notes) - 1 else "") for i, n in enumerate(notes)]
    out += [r"\end{minipage}", r"\par\medskip", ""]
    return "\n".join(out)


def _head(h: str) -> str:
    parts = [tex(x) for x in h.split("\n")]
    return parts[0] if len(parts) == 1 else r"\shortstack{" + r" \\ ".join(parts) + "}"


def md(head, rows, title, notes) -> str:
    out = [f"### {title}", "", "| " + " | ".join(h.replace("\n", " ") for h in head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows] + [""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(out)


HEAD_METRICS = ["Model", "Displacement\nerror", "Relative\nenergy gap", "von Mises\nerror",
                "Peak stress\nerror", "Critical-region\nrecall", "Worse than\nzero field"]


def metric_rows(report: dict, arms) -> list:
    rows = []
    for arm in arms:
        vals = [ms(seed_values(report, arm, "1024", m)) for m in METRICS]
        if arm == "zero":                 # every error is 1 by definition; the top 10% of a
            vals[-1] = "–"                # constant field is undefined (ties)
            rows.append([NAMES[arm]] + vals + ["–"])
            continue
        k, n = worse_than_zero(report, arm)
        rows.append([NAMES[arm]] + vals + [f"{k:,}/{n:,}"])
    return rows


def metric_notes(n_val: int, extra: list) -> list:
    return [
        f"{n_val} validation instances, never trained on; four load cases each. Errors are "
        "relative to the reference finite-element solution, per load case, then averaged over "
        "the load cases; the relative energy gap of the zero field is 1. Trained networks: "
        "mean ± sample standard deviation over 3 seeds; the nearest-neighbour field and the "
        "scaled polynomial are deterministic.",
        "Worse than zero field: instance-seed pairs whose relative energy gap exceeds 1, that "
        "is, predictions further from the solution in the energy norm than the zero field.",
        "Critical-region recall: the share of the 10% most stressed elements of the reference "
        "solution that are also among the 10% most stressed of the prediction (higher is "
        "better; in every other column lower is better).",
    ] + extra


def label_efficiency_rows(r2: dict, r3: dict) -> tuple:
    rows, rules = [], []
    for dim, rep in (("2D", r2), ("3D", r3)):
        if rows:
            rules.append(len(rows))
        arms = [a for a in ("labels", "labels_anchor", "ar_ft", "mgn", "knn_field")
                if a in cells(rep)]
        for arm in arms:
            c = cells(rep)[arm]
            row = [dim, NAMES[arm]]
            for b in BUDGETS:
                if b not in c:
                    row.append("–")
                    continue
                row.append(f"{num(seed_mean(rep, arm, 'disp_rel_l2', b))} / "
                           f"{num(seed_mean(rep, arm, 'energy_gap_rel', b))}")
            rows.append(row)
        d, e = seed_mean(rep, "ar", "disp_rel_l2"), seed_mean(rep, "ar", "energy_gap_rel")
        rows.append([dim, "Label-free (no labels)"] + [f"{num(d)} / {num(e)}"] * 4)
    return rows, tuple(rules)


def transfer_rows(r3: dict) -> list:
    p3 = r3["results"]["p3_transfer"]["metrics"]
    rows = []

    def row(name, blk):
        rows.append([name] + [ms(blk[m]["per_seed"]) for m in METRICS])

    row("Label-free, zero-shot", p3["ar"]["fine"])
    row("Supervised (1,024 labels), zero-shot", p3["zero_shot_reported"]["labels@max"]["fine"])
    row("Graph network (1,024 labels), zero-shot", p3["zero_shot_reported"]["mgn@max"]["fine"])
    for nm, key in (("Nearest-neighbour field", "knn_field"),
                    ("Scaled polynomial", "scale_aware_poly")):
        row(nm, p3["naive_rows"][key])
    for b in ("16", "64"):
        fs = p3["fewshot"][b]
        row(f"Label-free, then {b} fine-mesh labels", fs["finetune"])
        row(f"Supervised from scratch, {b} fine-mesh labels", fs["scratch"])
    return rows


# ---------------------------------------------------------------- provenance
RUNS = (("Two-dimensional run", "phase1_rec8_v2", R2D),
        ("Two-dimensional exploratory run", "diag_anchor_b16", RDIAG),
        ("Two-dimensional joint-embedding comparison", "wp2_e2_v2", RWP2),
        ("Three-dimensional amended run", "phase2b_v1", R3D),
        ("Latent regularisation, label-free network", "e1_2d_base", E1_BASE),
        ("Latent regularisation, regularised network", "e1_2d_shaped",
         REC8 / "e1" / "e1_2d_shaped" / "report.json"),
        ("Latent regularisation, regulariser on the latent vectors directly", "e1_2d_raw_s0",
         REC8 / "e1" / "e1_2d_raw_s0" / "report.json"),
        ("Token bottleneck, M = 512", "e2_m512", REC8 / "e2" / "e2_m512" / "report.json"),
        ("Token bottleneck, M = 1,024", "e2_m1024", REC8 / "e2" / "e2_m1024" / "report.json"))


def config_sha256(cfg: dict) -> str:
    """The run guard's canonical configuration hash (fejepa.report.config_sha256)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def hashes_tex(used: dict) -> str:
    """Each run's committed configuration, its hash (recomputed here and checked
    against the hash the run recorded), the record and its SHA-256."""
    out = [r"% generated by scripts/make_cmame_material.py from the records; do not edit",
           r"\begingroup\raggedright", r"\begin{itemize}"]
    for name, stem, rep_path in RUNS:
        cfg_path = ROOT / "configs" / f"{stem}.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        used[_rel(cfg_path)] = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
        rep = load(rep_path, used)
        h = config_sha256(cfg)
        if h != rep["provenance"]["config_sha256"]:
            raise SystemExit(f"{stem}: the committed configuration is not the one the run used")
        day = rep["provenance"]["timestamp_utc"][:10]
        y, mo, d = (int(x) for x in day.split("-"))
        month = ("January February March April May June July August September October "
                 "November December").split()[mo - 1]
        out.append(rf"\item {tex(name)}, {d} {month} {y}: configuration "
                   rf"\path{{configs/{stem}.json}} (SHA-256 \texttt{{{h[:16]}}}\ldots), record "
                   rf"\path{{{_rel(rep_path)}}} (SHA-256 "
                   rf"\texttt{{{used[_rel(rep_path)][:16]}}}\ldots).")
    out += [r"\end{itemize}", r"\par\endgroup", ""]
    return "\n".join(out)


# ---------------------------------------------------------------- wp8 tables
# the wp8 tables' rows and notes in this manuscript's vocabulary; every
# substitution must apply (a change in the wp8 text then stops the build)
RELABEL = {
    "amp": [("Point-token transformer", "Transformer"),
            ("Fine energy gap\n(median)", "Fine relative\nenergy gap (median)"),
            ("Mean fine energy gap over instances", "Mean fine relative energy gap over "
             "instances"),
            ("(energy gap: per-seed median)", "(relative energy gap: per-seed median)"),
            ("of the prediction u on the evaluated mesh, from that mesh's stiffness K and load F",
             "of the prediction $u$ on the evaluated mesh, from that mesh's stiffness matrix $K$ "
             "and load vector $F$"),
            ("Seeds per architecture", "Seeds per network"),
            (" instance-seed rows measured", " instance-seed evaluations measured"),
            ("of the same states", "of the same trained networks"),
            ("never increases the energy error", "never increases the error in the energy "
             "norm"),
            ("In-band: the 256 Phase-2b validation instances", "In-band: the 256 validation "
             "instances"),
            (" (E8, P3; in-band)", " (in-band)"),
            ("Post-hoc reading 4e (E-series runbook, Sec. 4): reported only; no verdict is "
             "revisited.", "Measured after the runs; no verdict is revisited.")],
    "remesh": [("Point-token transformer", "Transformer"),
               ("fscale, the battery's largest nodal load (both architectures multiply their "
                "output by it), relative", "The largest nodal load over the four load cases, "
                "by which both networks multiply their output, relative"),
               ("Post-hoc reading 4e (E-series runbook, Sec. 4): reported only.",
                "Measured after the runs."),
               ("c_b: the battery-level energy-optimal amplitude (label-free)",
                "c_b: the single energy-optimal amplitude of all four load cases of an "
                r"instance (label-free, \cref{sec:transfer:amplitude})")],
    "e2": [("Point-token transformer", "Transformer"),
           ("In-band\nenergy gap", "In-band relative\nenergy gap"),
           (": K1 in-band energy gap", ": accuracy criterion, in-band relative energy gap"),
           ("Verdict: KILLED.", "Outcome: dropped."),
           (", fires)", "; triggered)"),
           (" s line: not triggered", " s limit: not triggered"),
           ("Seeds per architecture", "Seeds per network"),
           ("set-up-free, median of 3 differential pairs of 10 and 110 steps; transformer: one "
            "timed call per phase, its set-up included, and the mean",
            "the median over three repetitions of the time of 110 steps minus that of 10 "
            "steps, divided by 100, which removes the set-up time; transformer: a single timed "
            "step, its set-up included, the mean"),
           ("Resolution: the pre-registered max(10%, 2 × SE_rel) of the difference of seed means.",
            "Resolution: the pre-registered max(10%, 2 × SE_rel), where SE_rel = "
            "$(s_\\mathrm{T}^2/3 + s_\\mathrm{B}^2/3)^{1/2}$ divided by the transformer's seed "
            "mean, with $s_\\mathrm{T}$ and $s_\\mathrm{B}$ the sample standard deviations of "
            "the two networks over the three seeds."),
           ("relative L2 displacement and relative energy gap", "relative displacement "
            "error and relative energy gap"),
           ("; K2 (speed) fine step", "; speed criterion, fine-mesh step"),
           ("In-band: the 256 Phase-2b validation instances", "In-band: the 256 validation "
            "instances"),
           ("one label-free AR training step", "one label-free training step"),
           ("from the E2 bench", "from a dedicated benchmark"),
           ("the bench's instances at these two ends", "the benchmark instances at the two "
            "ends of this range"),
           ("nodes in the bench)", "nodes in the benchmark)"),
           ("its two bench measurements", "its two benchmark measurements")],
    "e1": [("Arm", "Network"),
           ("Energy\ngap", "Relative\nenergy gap"),
           ("(resolution 10%), energy gap", "(resolution 10%), relative energy gap"),
           ("Verdict: NO-GO.", "Outcome: NO-GO; the regulariser was not adopted."),
           ("S: silhouette of the pooled latents over geometry bins", "S: silhouette of the "
            "instance-mean latent vectors over four bins, the quartiles of the first principal "
            "component of the geometry descriptor"),
           ("1-NN: leave-one-out bin accuracy", "1-NN: leave-one-out accuracy of a "
            "nearest-neighbour classifier of the bin"),
           ("SIGReg monitor\n(tokens)", "SIGReg statistic\n(tokens)"),
           ("The raw-token ablation ran one seed and is reported only.",
            "SIGReg statistic (tokens): the regulariser's statistic evaluated on the latent "
            "vectors themselves rather than on the projection head's output (lower is closer "
            "to an isotropic Gaussian). The variant that applies the regulariser to the latent "
            "vectors directly (raw tokens) ran one seed and is reported only."),
           ("AR + SIGReg (head)", "Label-free + SIGReg (head)"),
           ("AR + SIGReg (raw tokens)", "Label-free + SIGReg (raw tokens)"),
           ("K1 (accuracy parity, shaped vs AR)", "Accuracy criterion (regularised against "
            "label-free)"),
           ("S of the shaped arm minus S of the AR arm", "S of the regularised network minus "
            "S of the label-free network"),
           ("(post-hoc reading 4a, same instances): models built from the AR arm's",
            "(measured after the runs, same instances): models built from the label-free "
            "network's")],
}


def relabel(key: str, head: list, rows: list, notes: list) -> tuple:
    def sub(x: str, counts: dict) -> str:
        for i, (a, b) in enumerate(RELABEL[key]):
            if a in x:
                counts[i] = counts.get(i, 0) + x.count(a)
                x = x.replace(a, b)
        return x

    counts: dict = {}
    head = [sub(h, counts) for h in head]
    rows = [["Label-free" if c == "AR" else sub(c, counts) for c in r] for r in rows]
    notes = [sub(n, counts) for n in notes]
    missing = [RELABEL[key][i][0] for i in range(len(RELABEL[key])) if i not in counts]
    if missing:
        raise SystemExit(f"relabel {key}: not found in the wp8 text: {missing}")
    if head[:2] == ["Architecture", "Tokens"]:          # one column: the network
        head = ["Network"] + head[2:]
        rows = [[r[0] if r[1] == "one per node" else f"Bottleneck, M = {r[1]}"] + r[2:]
                for r in rows]
    return head, rows, notes


# ---------------------------------------------------------------- criteria
def _kill(report: dict, exp: str, idx: int = 0) -> dict:
    return report["results"][exp]["kills"][idx]


def criteria_rows(r2: dict, r3: dict, e1v: dict, e2v: dict, pilot: tuple,
                  rw: dict) -> tuple:
    """(rows, rule positions): every pre-registered criterion of the runs the
    manuscript cites, with the measured value and the outcome as recorded, and
    which trainings each criterion read."""
    rows, rules = [], []
    g1 = r2["gate_g1_prime"]
    th = g1["thresholds"]
    rb = g1["reasons"]
    e5 = r2["results"]["e5"]
    if e5["protocol"].get("anchored_source") != "E1' balanced arm":
        raise SystemExit("2D: G1' (a) no longer reads the energy-term sub-experiment")
    xs = [p_["beats_zero_x"] for p_ in e5["metrics"]["per_budget"]]
    naive = sorted(k for k in e5["metrics"]["baselines"] if k != "zero")
    b = re.search(r"egap reduction ([0-9.]+) .*vM reduction ([0-9.]+)", rb["b_physics"])
    c = re.search(r"egap \+([0-9.]+), disp \+([0-9.]+)", rb["c_transfer"])
    adv2 = r2["results"]["e8"]["metrics"]["ar_egap_advantage_by_budget"]
    k1n = _kill(r2, "e1")["note"]
    k1best = [float(x) for x in re.findall(r"[0-9.]+", k1n.split(":", 1)[1])]
    e7 = r2["results"]["e7"]
    e4 = r2["results"]["e4"]
    e4_big = max(e4["metrics"]["per_coarsen"], key=lambda r: r["coarsen"])

    def met(ok):
        return "met" if ok else "not met"

    def fired(k):
        return "triggered" if k else "not triggered"

    rows += [
        ["2D, 16 July 2026", "G1′ (a), energy-term sub-experiment (separately trained "
         f"networks): supervised + energy term at least {e5['protocol']['margin']:g} times "
         f"better than the zero field in displacement, and better than the {len(naive)} naive "
         f"baselines fitted on {e5['protocol']['fit_budget']:,} labelled instances, at every "
         "budget", f"{min(xs):.1f}–{max(xs):.1f} times; naive baselines beaten at every budget",
         met(g1["conditions"]["a"])],
        ["", f"G1′ (b), Tables 1 and 2, 64 labels: the energy term lowers the relative energy "
         f"gap by ≥ {th['egap_reduction'] * 100:.0f}% and the von Mises error by ≥ "
         f"{th['vm_reduction'] * 100:.0f}%",
         f"{float(b.group(1)) * 100:.1f}%, {float(b.group(2)) * 100:.1f}%",
         met(g1["conditions"]["b"])],
        ["", "G1′ (c), Tables 1 and 2, 64 labels: label-free, then supervised beats supervised "
         "by ≥ 10% in relative energy gap or ≥ 5% in displacement",
         f"{float(c.group(1)) * 100:.1f}%, {float(c.group(2)) * 100:.1f}%",
         met(g1["conditions"]["c"])],
        ["", "Gate G1′ = (a) and (b) and (c)", "", "GO" if g1["passed"] else "NO-GO"],
        ["", "K1, energy-term sub-experiment: the better of the fixed and the gradient-scaled "
         "energy term lowers the relative energy gap by < 25% at every budget",
         " / ".join(f"{x * 100:.1f}%" for x in k1best), fired(_kill(r2, "e1")["triggered"])],
        ["", "K2, Table 1: label-free displacement error > 30% above supervised, 1,024 labels",
         spct(seed_mean(r2, "ar", "disp_rel_l2") / seed_mean(r2, "labels", "disp_rel_l2") - 1),
         fired(_kill(r2, "e8", 0)["triggered"])],
        ["", "C1-advantage, Table 2: label-free relative energy-gap advantage over supervised "
         "< 40% at every budget", " / ".join(f"{adv2[k] * 100:.1f}%" for k in BUDGETS),
         fired(_kill(r2, "e8", 1)["triggered"])],
        ["", "K4: a latent-space regulariser raises the standardised effective rank of the "
         "latent vectors by ≤ 1.5 times",
         f"{r2['results']['e3']['metrics']['best_std_ratio']:.2f} times",
         fired(_kill(r2, "e3")["triggered"])],
        ["", f"K5: conjugate gradients from the prediction of a label-free network "
         f"({e7['protocol']['n_eval']} validation instances) save < 20% of the iterations from "
         "zero", spct(e7["metrics"]["savings_learned"]), fired(_kill(r2, "e7")["triggered"])],
        ["", f"K6: a cross-resolution invariance term shrinks the transfer gap by < 10% at "
         f"coarsening {e4_big['coarsen']:g}", spct(e4_big["gap_reduction"]),
         fired(_kill(r2, "e4")["triggered"])],
        ["", "E6 (alignment): within-geometry rank correlation of latent and solution "
         "distances < 0.3", f"{r2['results']['e6']['metrics']['rho_within_mean']:.3f}",
         fired(_kill(r2, "e6")["triggered"])],
        ["", "C5: any pre-registered check of the conditioning bound, the modewise identity or "
         "the conjugate-gradient bound violated", "none", fired(_kill(r2, "wp6")["triggered"])],
    ]
    rules.append(len(rows))
    k3 = _kill(rw, "e2")
    imp = rw["results"]["e2"]["metrics"]["jepa_vs_ar_improvements"]
    bud = list(imp)
    band = float(re.search(r"within ([0-9.]+)%", k3["condition"]).group(1))
    rows += [
        ["2D, 31 July 2026", f"K3: a self-supervised joint-embedding term added to label-free "
         f"pretraining changes the displacement error and the relative energy gap of the "
         f"transformer, fine-tuned with {', '.join(bud[:-1])} and {bud[-1]} labels, by less "
         f"than {band:g}% at every budget (if triggered, the term is dropped)",
         "change with the term, " + " / ".join(bud) + " labels: displacement "
         + " / ".join(spct(imp[bb]["disp"]) for bb in bud) + "; relative energy gap "
         + " / ".join(spct(imp[bb]["egap"]) for bb in bud) + " (positive: more accurate)",
         fired(k3["triggered"])],
    ]
    rules.append(len(rows))
    g2, ref = r3["gate_g2"], r3["gate_g2_reference_all_budgets"]
    t = g2["thresholds"]
    k = t["kills"]
    zero = {bb: seed_mean(r3, "zero", "disp_rel_l2", bb) for bb in BUDGETS}
    anc = {bb: seed_mean(r3, "labels_anchor", "disp_rel_l2", bb) for bb in BUDGETS}
    floor_ = int(t["gate"]["sanity_min_budget"])
    assessed = [bb for bb in BUDGETS if int(bb) >= floor_]
    parity = seed_mean(r3, "ar", "disp_rel_l2") / seed_mean(r3, "labels", "disp_rel_l2") - 1
    adv3 = [1 - seed_mean(r3, "ar", "energy_gap_rel") / seed_mean(r3, "labels",
                                                                   "energy_gap_rel", bb)
            for bb in BUDGETS]
    kp3 = [1 - seed_mean(r3, "labels_anchor", "energy_gap_rel", bb)
           / seed_mean(r3, "labels", "energy_gap_rel", bb) for bb in BUDGETS]
    p3 = r3["results"]["p3_transfer"]["metrics"]
    ratio = p3["ar"]["fine_disp_mean"] / p3["ar"]["inband_disp_mean"]
    naive_fine = min(p3["naive_at_fine"].values())
    rows += [
        ["3D, 21 September 2026", "Initial run: gate G2 and kill conditions as registered; the "
         "label-free objective was defective (deviation D14)", "(a), (b), (c) not met; KP1, "
         "KP2, KP4 triggered", "NO-GO"],
        ["3D, 22 September 2026", f"Pilot of the amendment: label-free displacement error after "
         f"20 epochs < {pilot[1]}", f"{pilot[0]}", met(float(pilot[0]) < float(pilot[1]))],
        ["3D, 28 September 2026", f"G2 (a), budgets ≥ {floor_} (amendment), Table 2: supervised "
         f"+ energy term at least {t['gate']['sanity_x']:.0f} times better than the zero field "
         "in displacement, and better than both naive baselines",
         " / ".join(f"{zero[bb] / anc[bb]:.1f}" for bb in assessed) + " times; naive beaten",
         met(g2["conditions"]["a"])],
        ["", "G2 (a), every budget (the registered form, reported for reference)",
         f"{zero['16'] / anc['16']:.2f} times at 16 labels", met(ref["conditions"]["a"])],
        ["", f"G2 (b), Tables 2 and 3: label-free displacement error within +"
         f"{t['gate']['parity_band'] * 100:.0f}% of supervised at 1,024 labels, and relative "
         f"energy-gap advantage ≥ {t['gate']['egap_adv_min'] * 100:.0f}% at every budget",
         f"{spct(parity)}; " + " / ".join(f"{x * 100:.2f}%" for x in adv3),
         met(g2["conditions"]["b"])],
        ["", f"G2 (c), Table 4: finer mesh, zero-shot displacement error ≤ "
         f"{t['gate']['transfer_win']} times in-band, and better than both naive baselines",
         f"{ratio:.2f} times; best naive {naive_fine:.3f} against "
         f"{p3['ar']['fine_disp_mean']:.3f}", met(g2["conditions"]["c"])],
        ["", "Gate G2 = (a) and ((b) or (c)), amended", "", "GO" if g2["passed"] else "NO-GO"],
        ["", "Gate G2, registered form (reference)", "", "GO" if ref["passed"] else "NO-GO"],
        ["", f"KP1: supervised (1,024 labels) better than label-free by > "
         f"{k['KP1_parity_pct'] * 100:.0f}% in displacement", spct(parity),
         fired(g2["kills"]["KP1"])],
        ["", f"KP2: relative energy-gap advantage < {k['KP2_egap_adv_min'] * 100:.0f}% at any "
         "budget", f"minimum {min(adv3) * 100:.1f}%", fired(g2["kills"]["KP2"])],
        ["", f"KP3: the energy term lowers the relative energy gap by < "
         f"{k['KP3_anchor_improv_min'] * 100:.0f}% at every budget",
         " / ".join(spct(x) for x in kp3), fired(g2["kills"]["KP3"])],
        ["", f"KP4: finer mesh, zero-shot displacement error > {k['KP4_transfer_ratio']} "
         "times in-band, or a naive baseline better", f"{ratio:.2f} times",
         fired(g2["kills"]["KP4"])],
        ["", "KP5: any pre-registered check of the conditioning bound, the modewise identity or "
         "the conjugate-gradient bound violated", "none", fired(g2["kills"]["KP5"])],
        ["", f"KP6 (alignment): within-geometry rank correlation of latent and solution "
         f"distances < {k['KP6_rho_within_min']}",
         f"{r3['results']['e6']['metrics']['rho_within_mean']:.3f}", fired(g2["kills"]["KP6"])],
    ]
    rules.append(len(rows))
    sd = e1v["S_delta"]
    rows += [
        ["2D, 29 September 2026", "Latent regularisation, K1: relative change of the "
         "displacement error or of the relative energy gap beyond the resolution",
         "within the resolution", fired(e1v["K1_parity"])],
        ["", "Latent regularisation, K2: the separation does not rise in any seed",
         "rose in every seed", fired(e1v["K2_no_effect"])],
        ["", f"Latent regularisation, GO: separation raised by ≥ {e1v['S_effect_floor']:g} in "
         "every seed", " / ".join(f"{x:+.3f}".replace("-", MINUS) for x in sd),
         "NO-GO" if not e1v["GO"] else "GO"],
    ]
    rules.append(len(rows))
    for m in (512, 1024):
        v = e2v[m]
        first = "3D, 30 September 2026" if m == 512 else ""
        rows += [
            [first, f"Token bottleneck, M = {m:,}, K1: in-band relative energy gap or "
             "finer-mesh displacement error worse than the transformer's by more than the "
             "resolution", f"{spct(v['egap_rel_change'])} (resolution "
             f"{v['egap_threshold'] * 100:.1f}%), {spct(v['fine_disp_rel_change'])} "
             f"(resolution {v['fine_disp_threshold'] * 100:.1f}%)", fired(v["K1_accuracy"])],
            ["", f"Token bottleneck, M = {m:,}, K2: finer-mesh training step longer than "
             f"{v['kill_s']:g} s", f"{v['fine_step_s']:.3f} s", fired(v["K2_speed"])],
            ["", f"Token bottleneck, M = {m:,}: outcome", "",
             "dropped" if v["verdict"] == "KILLED" else v["verdict"]],
        ]
    return rows, tuple(rules)


# ---------------------------------------------------------------- numbers
class Numbers(dict):
    def put(self, key: str, value: str) -> None:
        if not key.isalpha():
            raise SystemExit(f"macro name {key!r} is not letters only")
        if key in self:
            raise SystemExit(f"macro {key!r} defined twice")
        self[key] = value


def _sci(x: float, sig: int = 2) -> str:
    """x as m x 10^e (LaTeX math, for macros used in text mode)."""
    e = int(floor(log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{sig - 1}f} \\times 10^{{{e}}}$"


def numbers(r2: dict, rd: dict, r3: dict, amps: dict, e1base: dict, rw: dict) -> Numbers:
    """Every number the manuscript's text quotes, by macro name. Statements the
    text makes about the records ("in every seed", "on every metric") are
    checked here and stop the build when the records disagree."""
    N = Numbers()
    tag = {"ar": "Free", "labels": "Lab", "labels_anchor": "Anc", "mgn": "Mgn", "ar_ft": "Ft"}
    mtag = dict(zip(METRICS, ("Disp", "Gap", "Vm", "Peak", "Recall"), strict=True))
    for dim, rep in (("Two", r2), ("Three", r3)):
        for arm in TRAINED:
            if arm not in cells(rep):
                continue
            A = tag[arm]
            for m in METRICS:
                N.put(f"num{dim}{A}{mtag[m]}", num(seed_mean(rep, arm, m)))
            k, n = worse_than_zero(rep, arm)
            N.put(f"num{dim}{A}Worse", f"{k:,}")
            N.put(f"num{dim}{A}GapMedian",
                  num(float(np.median(per_instance(rep, arm, "energy_gap_rel")))))
        N.put(f"num{dim}Pairs", f"{per_instance(rep, 'ar', 'energy_gap_rel').size:,}")
        N.put(f"num{dim}Val", f"{per_instance(rep, 'ar', 'energy_gap_rel').shape[1]:,}")
        N.put(f"num{dim}FreeDispFour", f"{seed_mean(rep, 'ar', 'disp_rel_l2'):.4f}")
        N.put(f"num{dim}LabDispFour", f"{seed_mean(rep, 'labels', 'disp_rel_l2'):.4f}")
    # ------------------------------------------------ two dimensions
    d_ar, d_lab = seed_mean(r2, "ar", "disp_rel_l2"), seed_mean(r2, "labels", "disp_rel_l2")
    N.put("numTwoFreeDispExcess", spct(d_ar / d_lab - 1))
    N.put("numTwoFreeDispExcessAbs", pct(abs(d_ar / d_lab - 1)))
    trained2 = [seed_mean(r2, a, "energy_gap_rel") for a in TRAINED]
    N.put("numTwoGapSpread", f"{max(trained2) / min(trained2):.0f}")
    # solves (labels): 2D offline + in-run ledger; 3D from the labels the run consumed
    de2 = r2["data_economy"]
    off = de2["labelled_instances"] * de2["solves_per_labelled_instance"]
    N.put("numTwoSolvesOffline", f"{off:,}")
    N.put("numTwoSolvesRun", f"{de2['ledger']['total']:,}")
    N.put("numTwoSolves", f"{off + de2['ledger']['total']:,}")
    lp = r3["labels_present"]
    n_l = lp["n_loads"]
    n_all = (lp["inband_val_n"] + lp["inband_prefix_n"] + lp["fine_val_n"]
             + lp["fine_prefix_n"])
    N.put("numThreeSolves", f"{n_all * n_l:,}")
    N.put("numThreeEvalSolves", f"{(lp['inband_val_n'] + lp['fine_val_n']) * n_l:,}")
    N.put("numThreeTrainSolves", f"{lp['inband_prefix_n'] * n_l:,}")
    N.put("numThreeFewSolves", f"{lp['fine_prefix_n'] * n_l:,}")
    if not all(a > b for a, b in zip(seed_values(r2, "ar", "1024", "disp_rel_l2"),
                                     seed_values(r2, "labels", "1024", "disp_rel_l2"),
                                     strict=True)):
        raise SystemExit("2D: the text says the label-free displacement error is above the "
                         "supervised one in every seed; the records disagree")
    # label efficiency statements of Section 5.2
    ar_gap = seed_mean(r2, "ar", "energy_gap_rel")
    for arm in ("labels", "labels_anchor", "ar_ft", "mgn"):
        for bb in BUDGETS:
            if not ar_gap < seed_mean(r2, arm, "energy_gap_rel", bb):
                raise SystemExit(f"2D: label-free energy gap not below {arm} at {bb}")
    passes = [bb for bb in BUDGETS if seed_mean(r2, "labels", "disp_rel_l2", bb) < d_ar]
    if passes != ["1024"]:
        raise SystemExit(f"2D: the supervised network passes the label-free one at {passes}")
    if not (seed_mean(r2, "labels_anchor", "disp_rel_l2", "16")
            > seed_mean(r2, "labels", "disp_rel_l2", "16")):
        raise SystemExit("2D: the energy term did not raise the displacement error at 16")
    N.put("numTwoAncSixteenSeeds", ", ".join(
        num(x) for x in seed_values(r2, "labels_anchor", "16", "disp_rel_l2")))
    N.put("numTwoLabOverFreeGap", times(seed_mean(r2, "labels", "energy_gap_rel")
                                        / seed_mean(r2, "ar", "energy_gap_rel")))
    for arm in ("labels", "labels_anchor", "ar_ft", "mgn"):
        for m in ("energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err"):
            if not seed_mean(r2, "ar", m) < seed_mean(r2, arm, m):
                raise SystemExit(f"2D: label-free not better than {arm} in {m}")
        if not seed_mean(r2, "ar", "crit_recall") > seed_mean(r2, arm, "crit_recall"):
            raise SystemExit(f"2D: label-free recall not above {arm}")
    g1 = r2["gate_g1_prime"]
    b = re.search(r"egap reduction ([0-9.]+) \(need >= ([0-9.]+)\), vM reduction ([0-9.]+) "
                  r"\(need >= ([0-9.]+)\)", g1["reasons"]["b_physics"])
    N.put("numGOneGapReduction", pct(float(b.group(1))))
    N.put("numGOneVmReduction", pct(float(b.group(3))))
    c = re.search(r"egap \+([0-9.]+), disp \+([0-9.]+)", g1["reasons"]["c_transfer"])
    N.put("numGOneFtGap", pct(float(c.group(1))))
    N.put("numGOneFtDisp", pct(float(c.group(2))))
    N.put("numGOneRetired", pct(g1["retired_displacement_criterion"]
                                ["fixed_disp_improvement_at_decision"]["value"]))
    e5 = r2["results"]["e5"]["metrics"]["per_budget"]
    N.put("numGOneSanityMin", f"{min(p['beats_zero_x'] for p in e5):.1f}")
    N.put("numGOneSanityMax", f"{max(p['beats_zero_x'] for p in e5):.1f}")
    margin = r2["results"]["e5"]["protocol"]["margin"]
    x16 = (seed_mean(r2, "zero", "disp_rel_l2", "16")
           / seed_mean(r2, "labels_anchor", "disp_rel_l2", "16"))
    if not x16 < margin:
        raise SystemExit("2D: the table's 16-label energy-term network now meets the sanity "
                         "factor; the text says it would not have")
    N.put("numGOneSanityTableSixteen", f"{x16:.1f}")
    # ------------------------------------------------ the joint-embedding comparison (K3)
    k3 = _kill(rw, "e2")
    if not k3["condition"].startswith("K3") or k3["triggered"]:
        raise SystemExit("K3: the record no longer reads 'not triggered'")
    band = float(re.search(r"within ([0-9.]+)%", k3["condition"]).group(1)) / 100
    imp = rw["results"]["e2"]["metrics"]["jepa_vs_ar_improvements"]
    cells_k3 = [(bb, m, v[m]) for bb, v in imp.items() for m in ("disp", "egap")]
    out = [c_ for c_ in cells_k3 if abs(c_[2]) > band]
    gains = [c_ for c_ in out if c_[2] > 0]
    if [(g_[0], g_[1]) for g_ in gains] != [("64", "egap")]:
        raise SystemExit("K3: the single gain outside the band is no longer the 64-label gap")
    N.put("numKthreeBand", pct(band, 0))
    N.put("numKthreeOutside", words(len(out)))
    N.put("numKthreeWorse", words(len(out) - len(gains)))
    N.put("numKthreeGain", pct(gains[0][2]))
    base = rw["results"]["e2"]["metrics"]["table"]["ar_ft"]["64"]["egap"]["per_seed"]
    N.put("numKthreeBaseSeeds", ", ".join(num(x) for x in base[:-1]) + f" and {num(base[-1])}")
    ref64 = seed_values(r2, "ar_ft", "64", "energy_gap_rel")
    if not max(ref64) < min(base):
        raise SystemExit("K3: the 16 July fine-tuned network no longer beats every seed of the "
                         "31 July baseline at 64 labels")
    N.put("numKthreeBaseRef", ms(ref64))
    adv = r2["results"]["e8"]["metrics"]["ar_egap_advantage_by_budget"]
    N.put("numTwoAdvMax", pct(max(adv.values())))
    N.put("numTwoAdvMin", pct(min(adv.values())))
    e7 = r2["results"]["e7"]
    it = e7["metrics"]["iterations_to_tol_mean"]
    N.put("numKfiveZero", f"{it['zero']:.1f}")
    N.put("numKfiveFree", f"{it['learned']:.1f}")
    N.put("numKfiveNaive", f"{it['naive']:.1f}")
    N.put("numKfiveSaving", spct(e7["metrics"]["savings_learned"]))
    N.put("numKfiveNaiveSaving", spct(e7["metrics"]["savings_naive"]))
    N.put("numKfiveTol", f"$10^{{{int(round(log10(e7['protocol']['tol'])))}}}$")
    N.put("numKfiveEval", str(e7["protocol"]["n_eval"]))
    pk = e7["metrics"]["polish_at_k"]
    N.put("numKfiveStartGap", num(pk["0"]["energy_gap_rel"]))
    N.put("numKfiveFiveGap", num(pk["5"]["energy_gap_rel"]))
    N.put("numKfiveTwentyGap", num(pk["20"]["energy_gap_rel"]))
    g0 = pk["0"]["energy_gap_rel"]
    for eps, E in ((1e-7, "Seven"), (1e-12, "Twelve"), (1e-17, "Seventeen")):
        N.put(f"numWarmFraction{E}", pct(float(np.log(1 / g0) / np.log(4 / eps)), 0))
    e4 = sorted(r2["results"]["e4"]["metrics"]["per_coarsen"], key=lambda r: r["coarsen"])
    for row, C in zip(e4, ("Low", "High"), strict=True):
        N.put(f"numCoarsen{C}", f"{row['coarsen']:g}")
        N.put(f"numCoarsen{C}Coarse", num(row["inv_off"]["err_coarse"]))
        N.put(f"numCoarsen{C}Fine", num(row["inv_off"]["err_fine"]))
        N.put(f"numCoarsen{C}Reduction", spct(row["gap_reduction"]))
        if not row["gap_reduction"] < 0:
            raise SystemExit("2D: the text says the regulariser widened the transfer gap")
        N.put(f"numCoarsen{C}Widen", pct(-row["gap_reduction"]))
    disp9 = (seed_values(r2, "ar", "1024", "disp_rel_l2")
             + seed_values(rd, "ar", "1024", "disp_rel_l2"))
    gap9 = (seed_values(r2, "ar", "1024", "energy_gap_rel")
            + seed_values(rd, "ar", "1024", "energy_gap_rel"))
    N.put("numTwoRuns", words(len(disp9)))
    N.put("numTwoRunsDisp", f"{statistics.fmean(disp9):.4f} ± {statistics.stdev(disp9):.4f}")
    N.put("numTwoRunsGap", f"{statistics.fmean(gap9):.4f} ± {statistics.stdev(gap9):.4f}")
    N.put("numDiagSeeds", words(len(seed_values(rd, "ar", "1024", "disp_rel_l2"))))
    arms = rd["results"]["e1"]["metrics"]["per_budget"][0]["arms"]
    scaled = (arms["balanced"]["disp"]["per_seed"]
              + seed_values(rd, "labels_anchor", "16", "disp_rel_l2"))
    fixed = arms["fixed"]["disp"]["per_seed"]
    N.put("numDiagScaledMin", num(min(scaled), 2))
    N.put("numDiagScaledMax", num(max(scaled), 2))
    N.put("numDiagFixedMin", num(min(fixed), 2))
    N.put("numDiagFixedMax", num(max(fixed), 2))
    N.put("numDiagFixedSeeds", words(len(fixed)))
    eb = e1base["results"]["e8"]["metrics"]["cells"]["ar"]
    eb = eb[max(eb, key=int)]
    N.put("numEOneBaseDisp", num(statistics.fmean(eb["disp_rel_l2"]["per_seed"])))
    N.put("numEOneBaseGap", num(statistics.fmean(eb["energy_gap_rel"]["per_seed"])))
    e1v = json.loads((REC8 / "e1" / "e1_verdict.json").read_text())
    N.put("numEOneFloor", f"{e1v['S_effect_floor']:g}")
    N.put("numEOneDeltaMin", f"{min(e1v['S_delta']):.3f}")
    N.put("numEOneDeltaMax", f"{max(e1v['S_delta']):.3f}")
    kap = [rep["results"]["wp6"]["metrics"]["conditioning"]["kappa_range"] for rep in (r2, r3)]
    N.put("numSqrtKappaMin", f"{min(k[0] for k in kap) ** 0.5:.0f}")
    N.put("numSqrtKappaMax", f"{max(k[1] for k in kap) ** 0.5:.0f}")
    modes: dict = {}
    for rep, D in ((r2, "Two"), (r3, "Three")):
        w = rep["results"]["wp6"]["metrics"]
        lo, hi = w["conditioning"]["kappa_range"]
        N.put(f"numKappa{D}Low", _sci(lo))
        N.put(f"numKappa{D}High", _sci(hi))
        N.put(f"numCond{D}", f"{w['conditioning']['max_ratio']:.3f}")
        N.put(f"numModes{D}", _sci(w["mode_contraction"]["max_err"], 1))
        modes[D] = _sci(w["mode_contraction"]["max_err"], 1)
        N.put(f"numCheb{D}", f"{w['chebyshev_polish']['worst_measured_over_bound']:.3f}")
        wit = w["prop1_naive_cross"]["witness"]
        if not (w["prop1_naive_cross"]["naive_extension_falsified"]
                and wit["d_cross"] < wit["a_within_min"]):
            raise SystemExit(f"{D}: the cross-geometry counterexample is not on record")
        N.put(f"numCross{D}", _sci(wit["d_cross"], 3))
        N.put(f"numWithin{D}", _sci(wit["a_within_min"], 3))
        N.put(f"numPremise{D}", f"{w['prop1_premise']['min_separation']:.3f}")
        N.put(f"numRho{D}", f"{rep['results']['e6']['metrics']['rho_within_mean']:.3f}")
    N.put("numModesPhrase", f"{modes['Two']} in both dimensions"
          if modes["Two"] == modes["Three"]
          else f"{modes['Two']} (two dimensions) and {modes['Three']} (three dimensions)")
    # ------------------------------------------------ three dimensions
    for arm in ("labels", "labels_anchor"):
        for m in METRICS:
            hi_better = m == "crit_recall"
            a, s = seed_mean(r3, "ar", m), seed_mean(r3, arm, m)
            per_seed = [(x > y) if hi_better else (x < y) for x, y in zip(
                seed_values(r3, "ar", "1024", m), seed_values(r3, arm, "1024", m), strict=True)]
            if not (((a > s) if hi_better else (a < s)) and all(per_seed)):
                raise SystemExit(f"3D: the label-free network is not better than {arm} in {m} "
                                 "in the mean and in every seed")
    # statements of Sections 6.1-6.4 about the graph network and the budgets
    for m in ("energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err"):
        if not seed_mean(r3, "ar", m) < seed_mean(r3, "mgn", m):
            raise SystemExit(f"3D: label-free not better than the graph network in {m}")
    if not (seed_mean(r3, "ar", "crit_recall") > seed_mean(r3, "mgn", "crit_recall")
            and seed_mean(r3, "mgn", "disp_rel_l2") < seed_mean(r3, "ar", "disp_rel_l2")
            < seed_mean(r3, "mgn", "disp_rel_l2", "64")):
        raise SystemExit("3D: graph network recall/displacement statements do not hold")
    d3 = seed_mean(r3, "ar", "disp_rel_l2")
    g3 = seed_mean(r3, "ar", "energy_gap_rel")
    for arm in ("labels", "labels_anchor", "mgn"):
        for bb in cells(r3)[arm]:
            if not g3 < seed_mean(r3, arm, "energy_gap_rel", bb):
                raise SystemExit(f"3D: label-free energy gap not below {arm} at {bb}")
            if arm != "mgn" and not d3 < seed_mean(r3, arm, "disp_rel_l2", bb):
                raise SystemExit(f"3D: label-free displacement not below {arm} at {bb}")
    worse_anc = [bb for bb in BUDGETS if seed_mean(r3, "labels_anchor", "disp_rel_l2", bb)
                 > seed_mean(r3, "labels", "disp_rel_l2", bb)]
    if worse_anc != ["256"]:
        raise SystemExit(f"3D: the energy term raised the displacement error at {worse_anc}")
    worse_anc_gap = [bb for bb in BUDGETS if seed_mean(r3, "labels_anchor", "energy_gap_rel", bb)
                     > seed_mean(r3, "labels", "energy_gap_rel", bb)]
    if worse_anc_gap != ["256"]:
        raise SystemExit(f"3D: the energy term raised the energy gap at {worse_anc_gap}")
    up_d = sum(a > b for a, b in zip(seed_values(r3, "labels_anchor", "256", "disp_rel_l2"),
                                     seed_values(r3, "labels", "256", "disp_rel_l2"),
                                     strict=True))
    up_g = sum(a > b for a, b in zip(seed_values(r3, "labels_anchor", "256", "energy_gap_rel"),
                                     seed_values(r3, "labels", "256", "energy_gap_rel"),
                                     strict=True))
    if (up_d, up_g) != (3, 1):
        raise SystemExit(f"3D, 256 labels: the energy term raised the displacement error in "
                         f"{up_d} seeds and the energy gap in {up_g}; the text says 3 and 1")
    N.put("numThreeLabWorseTwoFiveSix", f"{worse_than_zero(r3, 'labels', '256')[0]:,}")
    N.put("numThreeAncWorseTwoFiveSix", f"{worse_than_zero(r3, 'labels_anchor', '256')[0]:,}")
    g1024 = per_instance(r3, "labels", "energy_gap_rel", "1024")
    worst = int(np.argmax(g1024.max(axis=0)))
    if not (np.delete(g1024, worst, axis=1).mean()
            < seed_mean(r3, "labels", "energy_gap_rel", "256")):
        raise SystemExit("3D: the rise of the supervised energy gap is not one instance's")
    if not (seed_mean(r3, "labels", "energy_gap_rel", "1024")
            > seed_mean(r3, "labels", "energy_gap_rel", "256")):
        raise SystemExit("3D: the supervised energy gap did not rise from 256 to 1,024")
    N.put("numThreeFreeDispAdvantage",
          pct(1 - seed_mean(r3, "ar", "disp_rel_l2") / seed_mean(r3, "labels", "disp_rel_l2")))
    N.put("numThreeAncOverFreeGap", times(seed_mean(r3, "labels_anchor", "energy_gap_rel")
                                          / seed_mean(r3, "ar", "energy_gap_rel")))
    N.put("numThreeMgnOverFreeVm", times(seed_mean(r3, "mgn", "vm_rel_l2")
                                         / seed_mean(r3, "ar", "vm_rel_l2")))
    N.put("numThreeMgnOverFreePeak", times(seed_mean(r3, "mgn", "peak_vm_rel_err")
                                           / seed_mean(r3, "ar", "peak_vm_rel_err")))
    N.put("numThreeFreeOverMgnDisp", times(seed_mean(r3, "ar", "disp_rel_l2")
                                           / seed_mean(r3, "mgn", "disp_rel_l2")))
    lab = per_instance(r3, "labels", "energy_gap_rel")
    N.put("numThreeLabGapMs", ms(seed_values(r3, "labels", "1024", "energy_gap_rel")))
    N.put("numThreeLabGapSeeds", ", ".join(num(x) for x in
                                            seed_values(r3, "labels", "1024", "energy_gap_rel")))
    N.put("numThreeLabWorst", f"{lab.max():,.0f}")
    s_worst = int(np.unravel_index(np.argmax(lab), lab.shape)[0])
    N.put("numThreeLabWorstSeed", str(s_worst))
    worst = [int(np.argmax(lab[i])) for i in range(lab.shape[0])]
    if len(set(worst)) != 1:
        raise SystemExit(f"3D: the supervised arm's worst instance differs between seeds: {worst}")
    w = worst[0]
    N.put("numWorstIndex", str(w))
    N.put("numWorstSeedLabDisp", num(per_instance(r3, "labels", "disp_rel_l2")[s_worst, w], 2))
    N.put("numWorstSeedLabVm", num(per_instance(r3, "labels", "vm_rel_l2")[s_worst, w], 2))
    N.put("numWorstSeedLabPeak", num(per_instance(r3, "labels", "peak_vm_rel_err")[s_worst, w], 3))
    for arm in ("ar", "labels", "mgn"):
        A = tag[arm]
        N.put(f"numWorst{A}Disp", num(per_instance(r3, arm, "disp_rel_l2")[0, w], 2))
        N.put(f"numWorst{A}Gap", num(per_instance(r3, arm, "energy_gap_rel")[0, w], 3))
        N.put(f"numWorst{A}Vm", num(per_instance(r3, arm, "vm_rel_l2")[0, w], 2))
    fr = {m: per_instance(r3, "ar", m) for m in METRICS}
    mg = {m: per_instance(r3, "mgn", m) for m in METRICS}
    disp_lower = (mg["disp_rel_l2"] < fr["disp_rel_l2"]).mean(axis=1)
    vm_higher = mg["vm_rel_l2"] > fr["vm_rel_l2"]
    gap_higher = (mg["energy_gap_rel"] > fr["energy_gap_rel"]).mean(axis=1)
    peak_lower = (mg["peak_vm_rel_err"] < fr["peak_vm_rel_err"]).mean(axis=1)
    N.put("numMgnDispLowerMin", pct(disp_lower.min()))
    N.put("numMgnDispLowerMax", pct(disp_lower.max()))
    N.put("numMgnVmHigher", f"{int(vm_higher.sum()):,}")
    N.put("numMgnGapHigherMin", pct(gap_higher.min()))
    N.put("numMgnPeakLowerMin", pct(peak_lower.min(), 0))
    N.put("numMgnPeakLowerMax", pct(peak_lower.max(), 0))
    both = (mg["disp_rel_l2"] < fr["disp_rel_l2"]) & vm_higher
    N.put("numMgnLowerDispHigherVm", f"{int(both.sum()):,}")
    med_ar = float(np.median(per_instance(r3, "ar", "energy_gap_rel")))
    ratios = [float(np.median(per_instance(r3, a, "energy_gap_rel"))) / med_ar
              for a in ("labels", "labels_anchor", "mgn")]
    N.put("numThreeMedianRatioMin", f"{min(ratios):.0f}")
    N.put("numThreeMedianRatioMax", f"{max(ratios):.0f}")
    # per load case, the label-free network (post-hoc records: |u|_K, |U*|_K and cos_K)
    pos = tot = 0
    for s in amps["transformer"]["seeds"].values():
        for row in s["inband"]["per_instance"]:
            u, us, cs = (np.asarray(row[k], float) for k in ("u_norm_K", "ustar_norm_K", "cos_K"))
            r = u / us
            pos += int(np.sum(r * r - 2 * r * cs + 1 > 1))
            tot += r.size
    N.put("numThreeFreeLoadWorse", f"{pos:,}")
    N.put("numThreeFreeLoads", f"{tot:,}")
    # the gate, both readings
    g2, ref = r3["gate_g2"], r3["gate_g2_reference_all_budgets"]
    t = g2["thresholds"]
    zero = {bb: seed_mean(r3, "zero", "disp_rel_l2", bb) for bb in BUDGETS}
    anc = {bb: seed_mean(r3, "labels_anchor", "disp_rel_l2", bb) for bb in BUDGETS}
    N.put("numGTwoSanityX", f"{t['gate']['sanity_x']:.1f}")
    N.put("numGTwoSanitySixteen", f"{zero['16'] / anc['16']:.2f}")
    N.put("numGTwoSanitySixtyFour", f"{zero['64'] / anc['64']:.1f}")
    N.put("numGTwoSanityMax", f"{zero['1024'] / anc['1024']:.1f}")
    N.put("numGTwoFloor", str(t["gate"]["sanity_min_budget"]))
    N.put("numGTwoParity", spct(seed_mean(r3, "ar", "disp_rel_l2")
                                / seed_mean(r3, "labels", "disp_rel_l2") - 1))
    adv3 = [1 - seed_mean(r3, "ar", "energy_gap_rel") / seed_mean(r3, "labels",
                                                                   "energy_gap_rel", bb)
            for bb in BUDGETS]
    N.put("numGTwoAdvMin", pct(min(adv3)))
    if not (g2["passed"] and not ref["passed"] and g2["kills"]["KP4"]):
        raise SystemExit("3D: the text reports amended GO, reference NO-GO and KP4 triggered")
    # transfer
    p3 = r3["results"]["p3_transfer"]["metrics"]
    N.put("numFineRatio", f"{p3['ar']['fine_disp_mean'] / p3['ar']['inband_disp_mean']:.1f}")
    N.put("numFineKill", f"{t['kills']['KP4_transfer_ratio']:g}")
    N.put("numFineWin", f"{t['gate']['transfer_win']:g}")
    for key, A in (("ar", "Free"), ("labels@max", "Lab"), ("mgn@max", "Mgn")):
        blk = p3["ar"]["fine"] if key == "ar" else p3["zero_shot_reported"][key]["fine"]
        for m in ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2"):
            N.put(f"numFine{A}{mtag[m]}", num(blk[m]["mean"]))
    N.put("numFineKnnDisp", num(p3["naive_at_fine"]["knn_field"]))
    for bb, B in (("16", "Sixteen"), ("64", "SixtyFour")):
        fs = p3["fewshot"][bb]
        for m in ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2", "peak_vm_rel_err"):
            N.put(f"numFew{B}{mtag[m]}", num(fs["finetune"][m]["mean"]))
        N.put(f"numFew{B}ScratchDisp", num(fs["scratch"]["disp_rel_l2"]["mean"]))
    few = p3["fewshot"]["64"]["finetune"]
    inb = p3["ar"]["inband"]
    fac = [few[m]["mean"] / inb[m]["mean"] for m in ("energy_gap_rel", "vm_rel_l2",
                                                     "peak_vm_rel_err")]
    N.put("numFewFactorMin", f"{min(fac):.1f}")
    N.put("numFewFactorMax", f"{max(fac):.1f}")
    # post hoc amplitude (transformer)
    A = amps["transformer"]["seeds"]
    fine = [A[s]["fine"]["summary"] for s in sorted(A)]
    inb_ = [A[s]["inband"]["summary"] for s in sorted(A)]

    def mean(rows, k):
        return statistics.fmean(x[k] for x in rows)

    N.put("numAmpFineDisp", num(mean(fine, "disp")))
    N.put("numAmpFineDispC", num(mean(fine, "disp_c")))
    red = [x["disp_reduction"] for x in fine]
    N.put("numAmpFineRedMin", pct(min(red), 0))
    N.put("numAmpFineRedMax", pct(max(red), 0))
    N.put("numAmpFineGapMedian", num(mean(fine, "egap_median")))
    N.put("numAmpFineGapMedianC", num(mean(fine, "egap_c_median")))
    N.put("numAmpFineCstar", f"{mean(fine, 'c_star_median'):.2f}")
    N.put("numAmpInCstar", f"{mean(inb_, 'c_star_median'):.2f}")
    N.put("numAmpInDisp", num(mean(inb_, "disp")))
    N.put("numAmpInDispC", num(mean(inb_, "disp_c")))
    N.put("numAmpInGap", num(mean(inb_, "egap")))
    N.put("numAmpInGapC", num(mean(inb_, "egap_c")))
    N.put("numAmpRatioBoth", f"{mean(fine, 'disp_c') / mean(inb_, 'disp_c'):.1f}")
    N.put("numAmpRatioFineOnly", f"{mean(fine, 'disp_c') / mean(inb_, 'disp'):.1f}")
    # the amplitude identity on every post-hoc row: egap_c = mean over loads of 1 - cos_K^2
    rows = 0
    dev = 0.0
    for d in amps.values():
        for s in d["seeds"].values():
            for st in ("inband", "fine"):
                for row in s[st]["per_instance"]:
                    cs = np.asarray(row["cos_K"], float)
                    dev = max(dev, abs(row["egap_c"] - float(np.mean(1 - cs * cs))))
                    rows += 1
    N.put("numAmpRows", f"{rows:,}")
    N.put("numAmpIdentity", _sci(dev, 1))
    return N


def _hms(text: str) -> float:
    h, m, sec = (int(x) for x in text.split(":"))
    return h + m / 60 + sec / 3600


def timing_numbers(N: Numbers, e2: dict, used: dict) -> None:
    """Training times, read from the committed run logs and the deviation ledger."""
    led = (ROOT / "DEVIATIONS_PHASE2.md")
    txt = led.read_text(encoding="utf-8")
    used[_rel(led)] = hashlib.sha256(led.read_bytes()).hexdigest()
    m = re.search(r"AR 3 x ([0-9.]+) h", txt)
    if not m:
        raise SystemExit("DEVIATIONS_PHASE2.md: the Phase-2b AR training time is not on record")
    per_seed_3d = float(m.group(1))
    flat = " ".join(txt.split())
    d14 = re.search(r"scored disp_rel_l2 ([0-9.]+) and energy_gap_rel ([0-9.]+)", flat)
    alpha = re.search(r"alpha = ([0-9.e-]+) reproduces both", flat)
    pilot = re.search(r"20 AR epochs\): disp ([0-9.]+) < ([0-9.]+), passed", flat)
    audit = re.search(r"Independent audit [^:]*: ([0-9]+)/([0-9]+) after", flat)
    if not (d14 and alpha and pilot and audit):
        raise SystemExit("DEVIATIONS_PHASE2.md: a D14, pilot or audit figure is not on record")
    N.put("numDfourteenDisp", d14.group(1))
    N.put("numDfourteenGap", d14.group(2))
    a_m, a_e = alpha.group(1).split("e")
    N.put("numDfourteenAlpha", f"${a_m} \\times 10^{{{int(a_e)}}}$")
    N.put("numPilotDisp", f"{float(pilot.group(1)):.3f}")
    N.put("numPilotLimit", f"{float(pilot.group(2)):.2f}")
    N.put("numAuditItems", f"{audit.group(1)} of {audit.group(2)}")
    N.put("numTimeThreeSeed", f"{per_seed_3d:.1f}")
    per_seed_e2 = []
    for M in (512, 1024):
        log = REC8 / "e2" / f"e2_m{M}" / "run.log"
        t = log.read_text(encoding="utf-8")
        used[_rel(log)] = hashlib.sha256(log.read_bytes()).hexdigest()
        if "workers=1" not in t:
            raise SystemExit(f"{log}: the seeds did not run one after another")
        d = re.search(r"\[E8 \(AR pretrain\)\] done in ([0-9:]+)", t)
        per_seed_e2.append(_hms(d.group(1)) / 3)
    N.put("numTimeBottleneckSeedMin", f"{min(per_seed_e2):.1f}")
    N.put("numTimeBottleneckSeedMax", f"{max(per_seed_e2):.1f}")
    N.put("numTimeBottleneckSpeedup", f"{per_seed_3d / statistics.fmean(per_seed_e2):.0f}")
    log = REC8 / "e1" / "e1_2d_base" / "run.log"
    t = log.read_text(encoding="utf-8")
    used[_rel(log)] = hashlib.sha256(log.read_bytes()).hexdigest()
    if "workers=3" not in t:
        raise SystemExit(f"{log}: the three seeds did not run in parallel")
    d = re.search(r"\[E8 \(AR pretrain\)\] done in ([0-9:]+)", t)
    mins = round(_hms(d.group(1)) * 60)
    N.put("numTimeTwoParallel", f"{mins // 60} h {mins % 60} min")
    # the bottleneck against the transformer (E2 records)
    a = e2["archs"]
    t_in, t_fi = a["transformer"]["step_ms"]["inband_0"], a["transformer"]["step_ms"]["fine"]
    r_step = [t_in / a[k]["step_ms"]["inband_0"] for k in ("m512", "m1024")]
    r_gap = [statistics.fmean(a[k]["egap"]) / statistics.fmean(a["transformer"]["egap"])
             for k in ("m512", "m1024")]
    r_fine = [statistics.fmean(a[k]["fine_disp"]) / statistics.fmean(a["transformer"]["fine_disp"])
              for k in ("m512", "m1024")]
    N.put("numEtwoStepMin", f"{min(r_step):.0f}")
    N.put("numEtwoStepMax", f"{max(r_step):.0f}")
    N.put("numEtwoGapMin", f"{min(r_gap):.1f}")
    N.put("numEtwoGapMax", f"{max(r_gap):.1f}")
    N.put("numEtwoFineMin", f"{min(r_fine):.1f}")
    N.put("numEtwoFineMax", f"{max(r_fine):.1f}")
    N.put("numEtwoNodesIn", f"{e2['nodes']['inband_0']:,}")
    N.put("numEtwoNodesFine", f"{e2['nodes']['fine']:,}")
    N.put("numEtwoTransformerFineStep", f"{t_fi / 1000:.1f}")


def _params_transformer(f: int, sd: int, D: int = 256, depth: int = 8) -> tuple:
    """Parameters of models/fejepa.py: (encoder + decoder, auxiliary heads), by formula
    (checked against the torch modules in tests/test_cmame_material.py)."""
    enc = f * D + D + depth * (12 * D * D + 13 * D) + 2 * D
    dec = (4 * D * D + 2 * D) + (2 * D * D + D) + (D * sd + sd)
    aux = (f * D + 7 * D * D + 10 * D) + (2 * D * D + 2 * D)
    return enc + dec, aux


def _params_graph(f: int, sd: int, D: int = 256, depth: int = 8) -> int:
    """Parameters of models/gnn.py, by formula (checked against torch in the tests)."""
    mlp = lambda i: i * D + D * D + 4 * D                                       # noqa: E731
    return mlp(f) + mlp(sd + 1) + depth * (mlp(2 * D) + mlp(3 * D)) + (D * D + D + D * sd + sd)


def extra_numbers(N: Numbers, r2: dict, r3: dict, amps: dict) -> None:
    """Mesh sizes, network sizes, paired per-instance comparisons and the residual-test
    bracket of Section 3.6."""
    A = amps["transformer"]["seeds"]
    s0 = A[sorted(A)[0]]
    nin = np.array([r["n_nodes"] for r in s0["inband"]["per_instance"]])
    nfi = np.array([r["n_nodes"] for r in s0["fine"]["per_instance"]])
    N.put("numNodesInMin", f"{nin.min():,}")
    N.put("numNodesInMax", f"{nin.max():,}")
    N.put("numNodesInMedian", f"{int(np.median(nin)):,}")
    N.put("numNodesFineMin", f"{nfi.min():,}")
    N.put("numNodesFineMax", f"{nfi.max():,}")
    N.put("numNodesFineMedian", f"{int(np.median(nfi)):,}")
    used, aux = _params_transformer(f=20, sd=3)
    N.put("numParamsTransformer", f"{used / 1e6:.1f}")
    N.put("numParamsAux", f"{aux / 1e6:.1f}")
    N.put("numParamsGraph", f"{_params_graph(f=20, sd=3) / 1e6:.1f}")
    for dim, rep in (("Two", r2), ("Three", r3)):
        for arm, A_ in (("labels", "Lab"), ("labels_anchor", "Anc")):
            for m, M in (("disp_rel_l2", "Disp"), ("energy_gap_rel", "Gap"),
                         ("vm_rel_l2", "Vm"), ("peak_vm_rel_err", "Peak"),
                         ("crit_recall", "Recall")):
                a, b = per_instance(rep, "ar", m), per_instance(rep, arm, m)
                win = (a > b) if m == "crit_recall" else (a < b)
                N.put(f"numPair{dim}{A_}{M}", share(win))
    w2 = r2["results"]["wp6"]["metrics"]["conditioning"]["kappa_range"]
    tol = r2["results"]["e7"]["protocol"]["tol"]
    N.put("numKappaTauTwo", _sci(w2[1] * tol * tol, 2))
    N.put("numTauSqKappaTwo", _sci(tol * tol / w2[1], 1))
    N.put("numTauSquared", f"$10^{{{int(round(2 * log10(tol)))}}}$")


def numbers_tex(N: dict) -> str:
    out = ["% generated by scripts/make_cmame_material.py from the records; do not edit"]
    for k in sorted(N):
        v = N[k]
        if v.startswith("$") and v.endswith("$") and v.count("$") == 2:
            body = rf"\ensuremath{{{v[1:-1]}}}"        # usable in text and in math
        else:
            body = tex(v)
        out.append(rf"\newcommand{{\{k}}}{{{body}}}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- figures
def _style():
    return {"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
            "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
            "pdf.fonttype": 42, "legend.frameon": False}


def _axes(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out: Path, name: str) -> Path:
    p = out / f"{name}.pdf"
    fig.savefig(p, dpi=300, metadata={"CreationDate": None})
    return p


FIG_NAMES = {"ar": "Label-free", "labels": "Supervised, same network",
             "labels_anchor": "Supervised + energy term, same network",
             "mgn": "Graph network, supervised"}


def figures(r2: dict, r3: dict, out: Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    written = []
    with plt.rc_context(_style()):
        # displacement error against von Mises error, per instance and seed (3D)
        fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
        for arm in ("labels", "mgn", "ar"):
            col, mk = STYLE[arm]
            x = per_instance(r3, arm, "disp_rel_l2").ravel()
            y = per_instance(r3, arm, "vm_rel_l2").ravel()
            ax.scatter(x, y, s=10, marker=mk, color=col, alpha=0.5, linewidths=0,
                       label=FIG_NAMES[arm])
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(fmt)
            axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Relative displacement error (log scale)")
        ax.set_ylabel("Relative von Mises error (log scale)")
        _axes(ax)
        leg = ax.legend(loc="upper left", fontsize=7, markerscale=1.8, handletextpad=0.3)
        for h in leg.legend_handles:
            h.set_alpha(1)
        written.append(_save(fig, out, "fig_disp_vs_stress"))
        plt.close(fig)

        # distribution of the relative energy gap, 2D and 3D
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), constrained_layout=True, sharey=True)
        for ax, (title, rep) in zip(axes, (("Two dimensions", r2), ("Three dimensions", r3)),
                                    strict=True):
            for arm in ("ar", "labels_anchor", "labels", "mgn"):
                col, mk = STYLE[arm]
                v = np.sort(per_instance(rep, arm, "energy_gap_rel").ravel())
                y = np.arange(1, v.size + 1) / v.size
                ax.step(v, y, where="post", color=col, linewidth=1.5, label=FIG_NAMES[arm])
                i = (v.size - 1) // 2
                ax.plot([v[i]], [y[i]], marker=mk, color=col, markersize=5.5,
                        markeredgecolor="white", markeredgewidth=0.8, linestyle="none")
            ax.axvline(1.0, color=INK2, linestyle="--", linewidth=0.8)
            ax.text(1.15, 0.04, "zero field", rotation=90, fontsize=7, color=INK2, va="bottom")
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(title, fontsize=8, color=INK, loc="left")
            ax.set_xlabel("Relative energy gap (log scale)")
            _axes(ax)
        axes[0].set_ylabel("Fraction of instance-seed pairs at or below")
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=2, fontsize=7, handlelength=1.8)
        written.append(_save(fig, out, "fig_energy_gap"))
        plt.close(fig)

        # label efficiency, 2D and 3D, displacement and energy gap
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), constrained_layout=True)
        xs = [int(bb) for bb in BUDGETS]
        for ci, (title, rep) in enumerate((("Two dimensions", r2), ("Three dimensions", r3))):
            for ri, (m, ylab) in enumerate((("disp_rel_l2", "Relative displacement error"),
                                            ("energy_gap_rel", "Relative energy gap"))):
                ax = axes[ri, ci]
                for arm in ("labels", "labels_anchor", "mgn"):
                    col, mk = STYLE[arm]
                    c = cells(rep)[arm]
                    bx = [int(bb) for bb in BUDGETS if bb in c]
                    by = [seed_mean(rep, arm, m, str(bb)) for bb in bx]
                    ax.plot(bx, by, color=col, marker=mk, markersize=5, linewidth=1.4,
                            markeredgecolor="white", markeredgewidth=0.6, label=FIG_NAMES[arm])
                col, _ = STYLE["ar"]
                ax.axhline(seed_mean(rep, "ar", m), color=col, linewidth=1.6,
                           label="Label-free, no labels")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks(xs)
                ax.set_xticklabels([f"{x:,}" for x in xs])
                ax.minorticks_off()
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
                ax.yaxis.set_major_formatter(fmt)
                if ri == 0:
                    ax.set_title(title, fontsize=8, color=INK, loc="left")
                else:
                    ax.set_xlabel("Labelled training instances (log scale)")
                if ci == 0:
                    ax.set_ylabel(ylab + " (log scale)")
                _axes(ax)
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="outside lower center", ncol=2, fontsize=7)
        written.append(_save(fig, out, "fig_label_efficiency"))
        plt.close(fig)
    return written


def fig_e2(e2: dict, out: Path) -> Path:
    """The wp8 cost/accuracy figure of E2 with this manuscript's names (same data,
    same layout as scripts/make_wp8_paper_material.py)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    pal = {None: "#2a78d6", 512: "#eb6834", 1024: "#1baf7a"}
    mk = {None: "o", 512: "s", 1024: "^"}
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
        n = e2["nodes"]
        panels = (("inband_0", "egap", f"In-band ({n['inband_0']:,}-node step)",
                   "Relative energy gap (log scale)", True, (10, 1e4)),
                  ("fine", "fine_disp", f"Finer mesh, zero-shot ({n['fine']:,}-node step)",
                   "Relative displacement error", False, (10, 1e5)))
        place = {("inband_0", 512): (0.30, 0.86), ("inband_0", 1024): (0.30, 0.50),
                 ("fine", 512): (0.25, 0.47), ("fine", 1024): (0.25, 0.90)}
        for ax, (ph, metric, title, ylab, logy, xlim) in zip(axes, panels, strict=True):
            for key, _label, m in W.E2_ARCHS:
                a = e2["archs"][key]
                name = "Transformer" if m is None else f"Bottleneck, M = {m:,}"
                x, ys = a["step_ms"][ph], a[metric]
                ax.scatter([x] * len(ys), ys, s=16, marker=mk[m], color=pal[m], alpha=0.45,
                           linewidths=0, zorder=2)
                ax.scatter([x], [statistics.fmean(ys)], s=56, marker=mk[m], color=pal[m],
                           edgecolors="white", linewidths=1.5, zorder=3, label=name)
                if m is None:
                    ax.annotate(name, (x, statistics.fmean(ys)), xytext=(-9, 0),
                                textcoords="offset points", fontsize=7, color=INK,
                                va="center", ha="right")
                else:
                    ax.annotate(name, (x, statistics.fmean(ys)), xytext=place[(ph, m)],
                                textcoords="axes fraction", fontsize=7, color=INK,
                                va="center", ha="left",
                                arrowprops={"arrowstyle": "-", "color": INK2,
                                            "linewidth": 0.6, "shrinkA": 2, "shrinkB": 5})
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(0.005, 0.1)
                ax.yaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.05, 0.1]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(0, 0.7)
            ax.set_title(title, fontsize=8, color=INK, loc="left")
            ax.set_xlabel("Label-free training step (ms, log scale)")
            ax.set_ylabel(ylab)
            _axes(ax)
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        axes[1].legend(loc="lower left", fontsize=7, handletextpad=0.3, borderaxespad=0.2)
        p = _save(fig, out, "fig_e2_cost_accuracy")
        plt.close(fig)
    return p


# ---------------------------------------------------------------- main
CAP_2D = ("Two-dimensional plates, every metric at the largest label budget. The supervised "
          "networks were trained on 1,024 labelled instances; the label-free network on the "
          "same 1,024 instances without their solutions.")
CAP_3D = ("Three-dimensional solids, every metric at the largest label budget. The supervised "
          "networks were trained on 1,024 labelled instances; the label-free network on the "
          "same 1,024 instances without their solutions.")
CAP_LE = ("Label efficiency: relative displacement error / relative energy gap against the "
          "number of labelled training instances (seed means).")
CAP_TR = ("Three-dimensional solids on the finer mesh (lc = 0.0374), never trained on. "
          "Zero-shot rows use the networks trained in-band; few-shot rows train, with labels, "
          "on the first 16 or on all of the 64 fine-mesh instances reserved for few-shot "
          "training.")
CAP_CR = "Every pre-registered criterion of the runs reported here, with its outcome."
CAP_AMP = ("Post hoc, three-dimensional: errors before and after scaling each prediction by "
           "its energy-optimal amplitude c*, in-band and on the finer mesh (zero-shot).")
CAP_REM = ("Post hoc, three-dimensional: the energy-optimal amplitude c_b of the four load "
           "cases of an instance together, on fixed geometries and loads meshed at four mesh "
           "sizes lc.")
CAP_E2 = ("Token bottleneck against the transformer with one token per node, both trained on "
          "the energy alone with the same corpus, split, seeds and schedule (3D).")
CAP_E1 = ("Isotropic Gaussian regularisation of the latent vectors added to the label-free "
          r"objective (two dimensions, with the later code of \cref{sec:results2d:other}).")


def build() -> dict:
    used: dict = {}
    r2, rd, r3 = load(R2D, used), load(RDIAG, used), load(R3D, used)
    rw = load(RWP2, used)
    amps = {k: load(p, used) for k, p in AMP.items()}
    e1base = load(E1_BASE, used)
    wused: dict = {}
    e1, e2 = W.read_e1(REC8, wused), W.read_e2(REC8, wused)
    ampw = W.read_amplitude(REC8, wused)
    probe_path = W.default_probe(REC8)
    probe = W._load(probe_path, wused) if probe_path else None
    if probe:
        W._check_probe(e1, probe)
    e1v = load(REC8 / "e1" / "e1_verdict.json", used)
    e2v = {m: load(REC8 / "e2" / f"e2_verdict_M{m}.json", used) for m in (512, 1024)}
    used.update(wused)

    N = numbers(r2, rd, r3, amps, e1base, rw)
    timing_numbers(N, e2, used)
    t = ampw["archs"]["transformer"]
    lcs = [float(x) for x in t["lc"]]
    fine_lc = float(e2["corpus"]["lc_fine"])
    cb = {lc: statistics.fmean(v) for lc, v in zip(lcs, t["remesh_cb"], strict=True)}
    inb = [cb[lc] for lc in lcs if lc != fine_lc]
    N.put("numRemeshGeometries", str(W._one([n for a in ampw["archs"].values()
                                              for n in a["n_geometries"]], "geometries")))
    N.put("numRemeshInDev", pct(max(abs(x - 1) for x in inb), 0))
    N.put("numRemeshFine", f"{cb[fine_lc]:.2f}")
    extra_numbers(N, r2, r3, amps)
    n2 = per_instance(r2, "ar", "disp_rel_l2").shape[1]
    n3 = per_instance(r3, "ar", "disp_rel_l2").shape[1]
    rows2, rows3 = metric_rows(r2, ARMS_2D), metric_rows(r3, ARMS_3D)
    notes2 = metric_notes(n2, [
        "Run of 16 July 2026. Its code did not multiply the load scale back onto the output "
        r"(\cref{sec:methods:networks}); we attribute the narrow spread of the displacement "
        r"errors to this, untested (\cref{sec:results2d:other})."])
    notes3 = metric_notes(n3, ["Run of 28 September 2026."])
    hle = ["Dimension", "Model", "16", "64", "256", "1,024"]
    rle, rules_le = label_efficiency_rows(r2, r3)
    nle = ["Seed means over 3 seeds (the nearest-neighbour field is deterministic). '–': a "
           "budget at which the model was not trained (in 3D the graph network was trained at "
           "64 and 1,024 labels only). The label-free row repeats its single value, which uses "
           "the same 1,024 instances without labels."]
    htr = ["Model", "Displacement\nerror", "Relative\nenergy gap", "von Mises\nerror",
           "Peak stress\nerror", "Critical-region\nrecall"]
    rtr = transfer_rows(r3)
    p3 = r3["results"]["p3_transfer"]["metrics"]
    ntr = [f"{per_instance(r3, 'ar', 'disp_rel_l2').shape[1]} fine-mesh evaluation instances "
           f"(median {N['numNodesFineMedian']} nodes), disjoint from the 64 instances reserved "
           "for few-shot training. "
           "Mean ± sample standard deviation over 3 seeds; the naive baselines are "
           "deterministic and built from the 1,024 in-band labelled instances.",
           f"In-band, the label-free network's displacement error is "
           f"{p3['ar']['inband_disp_mean']:.4f}; the zero-shot ratio is therefore "
           f"{p3['ar']['fine_disp_mean'] / p3['ar']['inband_disp_mean']:.2f}.",
           "Few-shot training runs 50 epochs at learning rate 1.5 × 10⁻³ from the trained "
           "label-free network of the same seed, or from random initialisation (scratch); the "
           "few-shot comparison was pre-registered as reported only."]
    hcr = ["Run", "Criterion", "Measured", "Outcome"]
    pilot = (N["numPilotDisp"], N["numPilotLimit"])
    rcr, rules_cr = criteria_rows(r2, r3, e1v, e2v, pilot, rw)
    ncr = ["Labels are those of the pre-registration documents, each of which numbers its own "
           "criteria. Gates (G1′, G2) decide whether the study proceeds to its next stage. Kill "
           "conditions withdraw a claim, or abandon a component, when triggered: in two "
           "dimensions K1–K6, C1-advantage (the claimed advantage in relative energy gap), C5 "
           r"(the pre-registered checks of \cref{sec:theory:checks}) and E6 (latent alignment); "
           "in three dimensions KP1–KP6, with KP5 the counterpart of C5. The "
           "latent-regularisation and token-bottleneck runs number their criteria afresh, with "
           "the meanings given here. A criterion reads the trained networks of the tables it "
           "names; the energy-term sub-experiment of the 2D run of 16 July 2026 trained its own "
           "supervised networks with and without the energy term, with the same configuration. "
           "K4, E6 and KP6 concern latent representations, which this paper does not use. "
           "Standardised effective rank: the participation ratio of the eigenvalues of the "
           "covariance of the latent vectors, each dimension standardised. The 3D amendment, the "
           "deviations and the joint-embedding term are described below, the latent "
           r"regularisation in \ref{app:e1} and the token bottleneck in "
           r"\cref{sec:cost:training}."]
    h_amp, r_amp = W.amp_rows(ampw)
    h_rem, r_rem = W.remesh_rows(ampw, e2)
    h_e2, r_e2 = W.e2_rows(e2)
    h_e1, r_e1 = W.e1_rows(e1, probe)
    files = {
        "table_2d.tex": table(HEAD_METRICS, rows2, CAP_2D, "tab:2d", notes2, rules=(5, 7)),
        "table_3d.tex": table(HEAD_METRICS, rows3, CAP_3D, "tab:3d", notes3, rules=(4, 6)),
        "table_label_efficiency.tex": table(hle, rle, CAP_LE, "tab:labeleff", nle,
                                            align="llcccc", rules=rules_le),
        "table_transfer.tex": table(htr, rtr, CAP_TR, "tab:transfer", ntr, rules=(3, 5)),
        "table_criteria.tex": longtable(hcr, rcr, CAP_CR, "tab:criteria", ncr,
                                        align="@{}" + "".join(
                                            RAGGED + f"p{{{w}\\linewidth}}"
                                            for w in ("0.14", "0.42", "0.21", "0.11"))
                                        + "@{}",
                                        rules=rules_cr),
        "table_amplitude.tex": table(*relabel("amp", h_amp, r_amp, W.amp_notes(ampw, e2))[:2],
                                     CAP_AMP, "tab:amplitude",
                                     relabel("amp", h_amp, r_amp, W.amp_notes(ampw, e2))[2]),
        "table_remesh.tex": table(*relabel("remesh", h_rem, r_rem,
                                           W.remesh_notes(ampw, e2))[:2], CAP_REM, "tab:remesh",
                                  relabel("remesh", h_rem, r_rem, W.remesh_notes(ampw, e2))[2]),
        "table_e2.tex": table(*relabel("e2", h_e2, r_e2, W.e2_notes(e2))[:2], CAP_E2, "tab:e2",
                              relabel("e2", h_e2, r_e2, W.e2_notes(e2))[2]),
        "table_e1.tex": table(*relabel("e1", h_e1, r_e1, W.e1_notes(e1, probe))[:2], CAP_E1,
                              "tab:e1", relabel("e1", h_e1, r_e1, W.e1_notes(e1, probe))[2]),
        "numbers.tex": numbers_tex(N),
        "hashes.tex": hashes_tex(used),
    }
    files["material.md"] = (
        "# CMAME manuscript material (generated from the records by "
        "scripts/make_cmame_material.py; do not edit)\n\n"
        + md(HEAD_METRICS, rows2, CAP_2D, notes2) + "\n" + md(HEAD_METRICS, rows3, CAP_3D, notes3)
        + "\n" + md(hle, rle, CAP_LE, nle) + "\n" + md(htr, rtr, CAP_TR, ntr) + "\n"
        + md(hcr, rcr, CAP_CR, ncr) + "\n## In-text numbers\n\n"
        + "\n".join(f"- `{k}`: {N[k]}" for k in sorted(N)) + "\n")
    files["sources.json"] = json.dumps({"generator": "scripts/make_cmame_material.py",
                                        "inputs_sha256": dict(sorted(used.items()))},
                                       indent=1) + "\n"
    return {"files": files, "r2": r2, "r3": r3, "e2": e2, "numbers": N}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    b = build()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in b["files"].items():
        (out / name).write_text(text, encoding="utf-8")
    written = [out / n for n in b["files"]]
    if not a.no_figures:
        written += figures(b["r2"], b["r3"], out)
        written.append(fig_e2(b["e2"], out))
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
