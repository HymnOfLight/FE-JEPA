#!/usr/bin/env python3
"""wp8 Stage 1.36 (wrap-up item 9; post-hoc tables Stage 1.38): paper material
for the E-series, from the committed records only.

Writes to --out (default paper/wp8):
  table_e1.tex, table_e2.tex   LaTeX (booktabs) tables of E1 and E2
  table_amplitude.tex          post-hoc reading 4e: errors before and after the
                               label-free amplitude c*, in-band and fine
  table_remesh.tex             post-hoc reading 4e: c_b on fixed geometries
                               meshed at four lc
  tables.md                    the same tables in Markdown
  frontier.csv                 every plotted point of the E2 cost/accuracy figure
  sources.json                 SHA-256 of every input file
  fig_e2_cost_accuracy.pdf/.png  the E2 cost/accuracy figure (unless --no-figure)

Every number is read from `records/wp8/` (the box's returned files, byte for
byte); nothing is typed by hand. The text outputs are deterministic:
regenerating from the records reproduces the committed files byte for byte
(tests/test_paper_material.py). Seed statistics are the mean and the SAMPLE
standard deviation (n - 1) over the seeds a run has.

The untrained-model rows of the E1 table come from post-hoc reading 4a
(`records/wp8/posthoc/probe_random_init.json`, used when present;
`--probe-random-init` names another file). The post-hoc tables are reported
readings: no verdict is revisited.

    python scripts/make_wp8_paper_material.py --out paper/wp8
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "records" / "wp8"

E1_ARMS = (("base", "AR", 3), ("shaped", "AR + SIGReg (head)", 3),
           ("raw_s0", "AR + SIGReg (raw tokens)", 1))
E2_ARCHS = (("transformer", "Point-token transformer", None),
            ("m512", "Token bottleneck", 512), ("m1024", "Token bottleneck", 1024))
UNTRAINED_ARMS = (("geometry_input_true", "Untrained, descriptor input"),
                  ("geometry_input_zeroed", "Untrained, descriptor weights zeroed"),
                  ("geometry_input_false", "Untrained, no descriptor input"))
AMP_FILES = {"transformer": ("amp_phase2b.json", ("e2", "baseline", "report_phase2b.json")),
             "m512": ("amp_e2_m512.json", ("e2", "e2_m512", "report.json")),
             "m1024": ("amp_e2_m1024.json", ("e2", "e2_m1024", "report.json"))}


def default_probe(rec: Path) -> Path | None:
    """Post-hoc reading 4a, when it is on record."""
    p = Path(rec) / "posthoc" / "probe_random_init.json"
    return p if p.exists() else None


# ---------------------------------------------------------------- reading
def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _load(path: Path, used: dict) -> dict:
    raw = Path(path).read_bytes()
    used[_rel(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def _ar_cell(report: dict) -> dict:
    cells = report["results"]["e8"]["metrics"]["cells"]["ar"]
    return cells[max(cells, key=int)]


def _seeds(cell: dict, key: str) -> list:
    return [float(v) for v in cell[key]["per_seed"]]


def _n_instances(cell: dict) -> int:
    """Instances each seed was evaluated on (the per-instance arrays; equal across seeds)."""
    ns = {len(e["per_instance"]["disp_rel_l2"]) for e in cell["per_seed_eval"]}
    if len(ns) != 1:
        raise SystemExit(f"seeds evaluated on different instance counts {sorted(ns)}")
    return ns.pop()


def read_e1(rec: Path, used: dict) -> dict:
    out = {"arms": {}, "verdict": _load(rec / "e1" / "e1_verdict.json", used)}
    for arm, _, n in E1_ARMS:
        rep = _load(rec / "e1" / f"e1_2d_{arm}" / "report.json", used)
        cell = _ar_cell(rep)
        seps = [_load(rec / "e1" / f"sep_{arm.replace('_s0', '')}_s{s}.json", used)
                for s in range(n)]
        out["arms"][arm] = {
            "disp": _seeds(cell, "disp_rel_l2"), "egap": _seeds(cell, "energy_gap_rel"),
            "S": [d["S_silhouette"] for d in seps],
            "r2": [d["probe_r2_geometry"] for d in seps],
            "nn": [d["loo_1nn_bin_accuracy"] for d in seps],
            "sig_tok": [d["sigreg_monitor_tokens"] for d in seps],
            "n_val": int(seps[0]["n_instances"]), "bins": [int(b) for b in seps[0]["bins"]],
            "pc1": [float(d["pc1_variance_share"]) for d in seps]}
        if len(out["arms"][arm]["disp"]) != n:
            raise SystemExit(f"E1 {arm}: {len(out['arms'][arm]['disp'])} seeds, expected {n}")
    return out


def read_e2(rec: Path, used: dict) -> dict:
    base = _load(rec / "e2" / "baseline" / "report_phase2b.json", used)
    out = {"archs": {}, "verdicts": {}, "bench": {}}
    reps = {"transformer": base}
    for m in (512, 1024):
        reps[f"m{m}"] = _load(rec / "e2" / f"e2_m{m}" / "report.json", used)
        out["verdicts"][m] = _load(rec / "e2" / f"e2_verdict_M{m}.json", used)
        out["bench"][m] = _load(rec / f"bench_e2_m{m}.json", used)["phases"]
    for key, rep in reps.items():
        cell = _ar_cell(rep)
        fine = rep["results"]["p3_transfer"]["metrics"]["ar"]["fine"]
        out["archs"][key] = {"disp": _seeds(cell, "disp_rel_l2"),
                             "egap": _seeds(cell, "energy_gap_rel"),
                             "fine_disp": _seeds(fine, "disp_rel_l2")}
    b = out["bench"]
    # the transformer was timed in both bench files (same code, same box): mean of the two
    out["archs"]["transformer"]["step_ms"] = {
        ph: statistics.fmean(b[m][ph]["ms_per_step"] for m in (512, 1024))
        for ph in ("inband_0", "fine")}
    out["archs"]["transformer"]["step_ms_both"] = {
        ph: [b[m][ph]["ms_per_step"] for m in (512, 1024)] for ph in ("inband_0", "fine")}
    for m in (512, 1024):
        out["archs"][f"m{m}"]["step_ms"] = {
            "inband_0": b[m][f"bottleneck{m}_inband_0"]["ms_per_step"],
            "fine": b[m][f"bottleneck{m}_fine"]["ms_per_step"]}
    out["nodes"] = {ph: b[512][ph]["n_nodes"] for ph in ("inband_0", "inband_2", "fine")}
    steps = {(*b[m][f"bottleneck{m}_{ph}"]["steps"], len(b[m][f"bottleneck{m}_{ph}"]["estimates_ms"]))
             for m in (512, 1024) for ph in ("inband_0", "fine")}
    if len(steps) != 1:
        raise SystemExit(f"bench: the bottleneck phases used different step pairs {steps}")
    out["bench_steps"] = list(steps.pop())
    cfg = base["config"]
    out["corpus"] = {"lc_range": [float(x) for x in cfg["data"]["lc_range"]],
                     "lc_fine": float(cfg["data_transfer"]["lc"]),
                     # the instance counts actually evaluated (per-instance arrays)
                     "n_val": _n_instances(_ar_cell(base)),
                     "n_fine": _n_instances(base["results"]["p3_transfer"]["metrics"]["ar"]
                                            ["fine"])}
    for key, rep in reps.items():
        got = (_n_instances(_ar_cell(rep)),
               _n_instances(rep["results"]["p3_transfer"]["metrics"]["ar"]["fine"]))
        if got != (out["corpus"]["n_val"], out["corpus"]["n_fine"]):
            raise SystemExit(f"E2 {key}: evaluated on {got} instances, the baseline on "
                             f"{(out['corpus']['n_val'], out['corpus']['n_fine'])}")
    n_seeds = {len(a[k]) for a in out["archs"].values() for k in ("disp", "egap", "fine_disp")}
    if len(n_seeds) != 1:
        raise SystemExit(f"E2: architectures differ in seed count {sorted(n_seeds)}")
    out["n_seeds"] = n_seeds.pop()
    pol = _load(rec / "bench_e2_m512.json", used)["numeric_policy"]
    out["machine"] = {"gpu": pol.get("gpu"), "tf32": bool(pol.get("tf32"))}
    for m in (512, 1024):
        for ph, key in (("inband_0", f"bottleneck{m}_inband_0"), ("fine", f"bottleneck{m}_fine")):
            if b[m][key]["n_nodes"] != out["nodes"][ph] or b[m][ph]["n_nodes"] != out["nodes"][ph]:
                raise SystemExit(f"bench M={m}: {key} timed another instance size")
    return out


AMP_KEYS = ("disp", "disp_c", "egap", "egap_c", "egap_median", "egap_c_median",
            "c_star_median", "n_instances")


def _inrun_deviation(report: dict) -> float:
    """Largest relative per-instance difference between a run's own two
    evaluations of the same states on the in-band set (E8's AR cell and P3's
    in-band block): the run's evaluation nondeterminism."""
    import numpy as np

    cells = report["results"]["e8"]["metrics"]["cells"]["ar"]
    a = cells[max(cells, key=int)]["per_seed_eval"]
    b = report["results"]["p3_transfer"]["metrics"]["ar"]["inband"]["per_seed_eval"]
    worst = 0.0
    for ea, eb in zip(a, b, strict=True):
        for k in ("disp_rel_l2", "energy_gap_rel"):
            x = np.asarray(ea["per_instance"][k], float)
            y = np.asarray(eb["per_instance"][k], float)
            if x.shape != y.shape:
                raise SystemExit("E8 and P3 evaluated different in-band sets")
            worst = max(worst, float(np.max(np.abs(x - y) / np.maximum(np.abs(x), 1e-30))))
    return worst


def read_amplitude(rec: Path, used: dict) -> dict:
    """Post-hoc reading 4e (`posthoc/amp_*.json`): per seed, the in-band and
    fine summaries and the remesh medians. Each file must name the committed
    report it measured (SHA-256) and be complete; the remesh fscale ratios are
    a property of the meshes and must agree between the three files."""
    out = {"archs": {}, "n_rows": 0, "n_egap_up": 0, "repro_max": 0.0, "inrun_max": 0.0}
    fscale = None
    for key, (name, rp) in AMP_FILES.items():
        d = _load(rec / "posthoc" / name, used)
        rep = rec.joinpath(*rp)
        if d["report_sha256"] != hashlib.sha256(rep.read_bytes()).hexdigest():
            raise SystemExit(f"{name}: measured another report than {_rel(rep)}")
        if key != "transformer":
            out["inrun_max"] = max(out["inrun_max"], _inrun_deviation(json.loads(rep.read_bytes())))
        seeds = sorted(d["seeds"], key=lambda s: int(s[1:]))
        a = {}
        for st in ("inband", "fine"):
            summ = [d["seeds"][s][st]["summary"] for s in seeds]
            if not all(x.get("complete") for x in summ):
                raise SystemExit(f"{name}: the {st} set is incomplete")
            for k in AMP_KEYS:
                a[f"{st}_{k}"] = [x[k] for x in summ]
            for s in seeds:
                rows = d["seeds"][s][st]["per_instance"]
                out["n_rows"] += len(rows)
                out["n_egap_up"] += sum(r["egap_c"] > r["egap"] * (1 + 1e-12) for r in rows)
                rp_ = d["seeds"][s][st]["reproduction"]
                if key != "transformer":
                    out["repro_max"] = max(out["repro_max"], rp_["disp_max_rel_dev"],
                                           rp_["egap_max_rel_dev"])
                elif rp_["disp_max_rel_dev"] != 0.0 or rp_["egap_max_rel_dev"] != 0.0:
                    raise SystemExit(f"{name}: the transformer did not reproduce its report")
        rem = [d["seeds"][s]["remesh"] for s in seeds]
        lcs = [float(x) for x in d["remesh_lc"]]
        a["lc"] = lcs
        a["remesh_cb"] = [[r["c_battery_median_by_lc"][str(lc)] for r in rem] for lc in lcs]
        a["remesh_unorm"] = [[r["u_norm_K_ratio_median_by_lc"][str(lc)] for r in rem]
                             for lc in lcs]
        a["n_geometries"] = sorted({r["n_geometries"] for r in rem})
        if any(r["n_failed"] for r in rem):
            raise SystemExit(f"{name}: a remesh geometry failed")
        fs = [rem[0]["fscale_ratio_median_by_lc"][str(lc)] for lc in lcs]
        if fscale is not None and fs != fscale:
            raise SystemExit(f"{name}: remeshed other meshes than the first file")
        fscale = fs
        out["archs"][key] = a
    if out["n_egap_up"]:                     # the theory says never: a record defect
        raise SystemExit(f"c* raised the energy gap in {out['n_egap_up']} rows")
    out["fscale"] = fscale
    out["lc"] = out["archs"]["transformer"]["lc"]
    return out


# ---------------------------------------------------------------- formatting
def _mean(v):
    return statistics.fmean(v)


def _sd(v):
    return statistics.stdev(v) if len(v) > 1 else None


MINUS = "\u2212"


def _g(x: float, sig: int = 3) -> str:
    """`sig` significant digits, fixed notation, no exponent, typographic minus."""
    if x == 0:
        return "0"
    from math import floor, log10

    dec = max(0, sig - 1 - int(floor(log10(abs(x)))))
    return f"{x:.{dec}f}".replace("-", MINUS)


def _ms(v, sig=3, dec=None) -> str:
    """mean (+/- sample SD at the mean's precision); one seed: the value.
    `dec` fixes the decimals (bounded quantities: S, R^2, 1-NN), otherwise
    `sig` significant digits (errors spanning decades)."""
    m, s = _mean(v), _sd(v)
    txt = _g(m, sig) if dec is None else f"{m:.{dec}f}".replace("-", MINUS)
    if s is None:
        return txt
    d = len(txt.split(".")[1]) if "." in txt else 0
    return f"{txt} ± {s:.{d}f}"


def _signed_pct(x: float) -> str:
    return f"{x * 100:+.1f}%".replace("-", MINUS)


def _ms_time(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:.1f}"


def e1_rows(e1: dict, probe: dict | None) -> tuple:
    head = ["Arm", "Seeds", "Displacement\nerror", "Energy\ngap", "S", "Probe R²", "1-NN",
            "SIGReg monitor\n(tokens)"]
    rows = []
    for arm, label, n in E1_ARMS:
        a = e1["arms"][arm]
        rows.append([label, str(n), _ms(a["disp"]), _ms(a["egap"]), _ms(a["S"], dec=3),
                     _ms(a["r2"], dec=3), _ms(a["nn"], dec=2), _ms(a["sig_tok"], 2)])
    if probe:
        for arm, label in UNTRAINED_ARMS:
            rr = [v for k, v in sorted(probe["readings"].items()) if k.startswith(f"{arm}_s")]
            if not rr:                         # a probe file from before the paired arm
                continue
            rows.append([label, str(len(rr)), "–", "–",
                         _ms([r["S_silhouette"] for r in rr], dec=3),
                         _ms([r["probe_r2_geometry"] for r in rr], dec=3),
                         _ms([r["loo_1nn_bin_accuracy"] for r in rr], dec=2),
                         _ms([r["sigreg_monitor_tokens"] for r in rr], 2)])
    return head, rows


def _chance(bins: list) -> str:
    """1-NN chance level of the bin labels: 1 / k for k equal bins, otherwise
    the largest bin's share."""
    if len(set(bins)) == 1:
        return f"{1 / len(bins):.2f} with {len(bins)} bins of {bins[0]}"
    return f"{max(bins) / sum(bins):.2f}, the largest bin's share"


def _check_probe(e1: dict, probe: dict) -> None:
    """The probe must have read the E1 instances: same count, and the PC1
    share (a property of the instances' descriptors alone) equal to the
    recorded one."""
    if probe.get("n_instances") is not None and probe["n_instances"] != e1["arms"]["base"]["n_val"]:
        raise SystemExit(f"probe: {probe['n_instances']} instances, E1 read "
                         f"{e1['arms']['base']['n_val']}")
    pcs = {round(r["pc1_variance_share"], 12) for r in probe["readings"].values()}
    want = {round(x, 12) for x in e1["arms"]["base"]["pc1"]}
    if pcs != want:
        raise SystemExit(f"probe: PC1 share {sorted(pcs)} is not E1's {sorted(want)}")


def e1_notes(e1: dict, probe: dict | None) -> list:
    v = e1["verdict"]
    k1 = v["K1_detail"]
    d, g = k1["disp_rel_l2"], k1["energy_gap_rel"]
    deltas = " / ".join(f"{x:+.3f}".replace("-", MINUS) for x in v["S_delta"])
    notes = [
        f"K1 (accuracy parity, shaped vs AR): displacement {_signed_pct(d['rel_change'])} "
        f"(resolution {d['threshold'] * 100:.0f}%), energy gap {_signed_pct(g['rel_change'])} "
        f"(resolution {g['threshold'] * 100:.0f}%): not triggered.",
        f"Effect (S of the shaped arm minus S of the AR arm, per seed): {deltas}; the "
        f"pre-registered floor is {v['S_effect_floor']} in every seed: not met. "
        f"Verdict: {v['verdict']}.",
        f"{e1['arms']['base']['n_val']} validation instances; seed mean ± sample SD. "
        "S: silhouette of the pooled latents over geometry bins; probe R²: linear probe of "
        "the geometry descriptor; 1-NN: leave-one-out bin accuracy, chance "
        f"{_chance(e1['arms']['base']['bins'])}.",
        "The geometry descriptor is also a per-node model input, so S, the probe and 1-NN "
        "largely read the input back; they are not evidence of learned geometry"
        + (". Untrained rows (post-hoc reading 4a, same instances): models built from the "
           "AR arm's configuration at initialisation with the descriptor input; the same "
           "models with the descriptor's input weights set to zero; models built without "
           "the descriptor input." if probe else
           " (untrained-model reference: post-hoc reading 4a, pending)."),
        "The raw-token ablation ran one seed and is reported only.",
    ]
    return notes


def e2_rows(e2: dict) -> tuple:
    n_in, n_fi = e2["nodes"]["inband_0"], e2["nodes"]["fine"]
    head = ["Architecture", "Tokens", "In-band\ndisplacement", "In-band\nenergy gap",
            "Fine displacement\n(zero-shot)", f"Step (ms),\n{n_in:,} nodes",
            f"Step (ms),\n{n_fi:,} nodes"]
    rows = []
    for key, label, m in E2_ARCHS:
        a = e2["archs"][key]
        rows.append([label, "one per node" if m is None else f"{m:,}", _ms(a["disp"]),
                     _ms(a["egap"]), _ms(a["fine_disp"]), _ms_time(a["step_ms"]["inband_0"]),
                     _ms_time(a["step_ms"]["fine"])])
    return head, rows


def e2_notes(e2: dict) -> list:
    notes = []
    for m in (512, 1024):
        v = e2["verdicts"][m]
        k1 = ", ".join(
            f"{name} {_signed_pct(v[f'{q}_rel_change'])} (resolution "
            f"{v[f'{q}_threshold'] * 100:.0f}%{', fires' if v[f'{q}_rel_change'] > v[f'{q}_threshold'] else ''})"
            for q, name in (("egap", "in-band energy gap"), ("fine_disp", "fine displacement")))
        notes.append(f"M = {m:,}: K1 {k1}; K2 (speed) fine step {v['fine_step_s']:.3f} s "
                     f"against the {v['kill_s']:.1f} s line: not triggered. Verdict: "
                     f"{v['verdict']}.")
    t = e2["archs"]["transformer"]["step_ms_both"]
    c, n, mach, st = e2["corpus"], e2["nodes"], e2["machine"], e2["bench_steps"]
    notes += [
        f"Seeds per architecture: {e2['n_seeds']}; seed mean ± sample SD. In-band: the {c['n_val']} "
        f"Phase-2b validation instances (lc {c['lc_range'][0]}–{c['lc_range'][1]}; the bench's "
        f"instances at these two ends have {n['inband_0']:,} and {n['inband_2']:,} nodes); fine: "
        f"{c['n_fine']} instances at lc {c['lc_fine']} ({n['fine']:,} nodes in the bench), "
        "never trained on. Errors are relative L2 displacement and relative energy gap.",
        "Step time: one label-free AR training step on one instance of the stated size, "
        f"{mach['gpu']}, TF32 {'on' if mach['tf32'] else 'off'}, from the E2 bench "
        f"(bottleneck: set-up-free, median of {st[2]} differential pairs of {st[0]} and {st[1]} "
        "steps; transformer: one timed call per phase, its set-up included, and the mean of "
        "its two bench measurements, "
        f"{t['inband_0'][0]:,.0f} / {t['inband_0'][1]:,.0f} ms and "
        f"{t['fine'][0]:,.0f} / {t['fine'][1]:,.0f} ms).",
        "Resolution: the pre-registered max(10%, 2 × SE_rel) of the difference of seed means.",
    ]
    return notes


def _one(values: list, what: str):
    if len(set(values)) != 1:
        raise SystemExit(f"{what} differs between seeds or runs: {sorted(set(values))}")
    return values[0]


def _arch_label(m) -> str:
    return "M = " + (f"{m:,}" if m else "?")


def amp_rows(amp: dict) -> tuple:
    head = ["Architecture", "Tokens", "In-band\ndisplacement", "In-band\nwith c*",
            "Fine displacement\n(zero-shot)", "Fine\nwith c*", "Fine energy gap\n(median)",
            "Fine\nwith c*", "Median c*\n(fine)"]
    rows = []
    for key, label, m in E2_ARCHS:
        a = amp["archs"][key]
        rows.append([label, "one per node" if m is None else f"{m:,}", _ms(a["inband_disp"]),
                     _ms(a["inband_disp_c"]), _ms(a["fine_disp"]), _ms(a["fine_disp_c"]),
                     _ms(a["fine_egap_median"]), _ms(a["fine_egap_c_median"]),
                     _ms(a["fine_c_star_median"], dec=2)])
    return head, rows


def amp_notes(amp: dict, e2: dict) -> list:
    A = amp["archs"]
    n_in = _one([n for a in A.values() for n in a["inband_n_instances"]], "in-band count")
    n_fi = _one([n for a in A.values() for n in a["fine_n_instances"]], "fine count")
    means = "; ".join(
        f"{'transformer' if m is None else _arch_label(m)}: "
        f"{_g(_mean(A[key]['fine_egap']))} → {_g(_mean(A[key]['fine_egap_c']))}"
        for key, _, m in E2_ARCHS)
    inband = ", ".join(f"{_mean(A[key]['inband_c_star_median']):.2f} "
                       f"({'transformer' if m is None else _arch_label(m)})"
                       for key, _, m in E2_ARCHS)
    return [
        "c* = F^T u / (u^T K u), per load case: the energy-optimal amplitude of the prediction "
        "u on the evaluated mesh, from that mesh's stiffness K and load F (no labels, no "
        "solve). Scaling by c* never increases the energy error (it did not in any of the "
        f"{amp['n_rows']:,} instance-seed rows measured). In-band median c*: {inband}.",
        f"Seeds per architecture: {e2['n_seeds']}; seed mean ± sample SD of the per-seed mean "
        f"over instances (energy gap: per-seed median). In-band: the {n_in} Phase-2b "
        f"validation instances; fine: {n_fi} instances at lc {e2['corpus']['lc_fine']}, never "
        "trained on. The columns without c* reproduce the runs' own per-instance arrays: the "
        "transformer's exactly, the bottleneck's to a relative "
        f"{amp['repro_max']:.2g} at most; the bottleneck run's own two evaluations of the same "
        f"states (E8, P3; in-band) differ by up to {amp['inrun_max']:.2g}, the transformer "
        "run's not at all.",
        f"Mean fine energy gap over instances (a heavy tail), as trained → with c*: {means}.",
        "Post-hoc reading 4e (E-series runbook, Sec. 4): reported only; no verdict is revisited.",
    ]


def remesh_rows(amp: dict, e2: dict) -> tuple:
    fine = float(e2["corpus"]["lc_fine"])
    head = ["Architecture", "Tokens"] + [f"lc {lc}" + ("\n(fine)" if lc == fine else "")
                                         for lc in amp["lc"]]
    rows = []
    for key, label, m in E2_ARCHS:
        a = amp["archs"][key]
        if a["lc"] != amp["lc"]:
            raise SystemExit(f"remesh: {key} used other mesh sizes")
        rows.append([label, "one per node" if m is None else f"{m:,}"]
                    + [_ms(v, dec=2) for v in a["remesh_cb"]])
    return head, rows


def remesh_notes(amp: dict, e2: dict) -> list:
    A, lcs = amp["archs"], amp["lc"]
    lo, hi = e2["corpus"]["lc_range"]
    g = _one([n for a in A.values() for n in a["n_geometries"]], "remesh geometry count")
    lc0 = lcs[0]
    unorm = "; ".join(
        f"{'transformer' if m is None else _arch_label(m)}: "
        + " / ".join(f"{_mean(v):.2f}" for v in A[key]["remesh_unorm"][1:])
        for key, _, m in E2_ARCHS)
    return [
        f"{g} fresh geometries, each meshed at the {len(lcs)} lc with identical geometry and "
        "loads, so only the mesh changes; every mesh evaluated by every seed. c_b: the "
        "battery-level energy-optimal amplitude (label-free); median over the geometries, then "
        f"seed mean ± sample SD. Training range: lc {lo}–{hi}.",
        "fscale, the battery's largest nodal load (both architectures multiply their output "
        f"by it), relative to lc {lc0}, at lc " + " / ".join(str(lc) for lc in lcs[1:]) + ": "
        + " / ".join(f"{x:.3f}" for x in amp["fscale"][1:])
        + f"; for comparison (lc / {lc0})²: "
        + " / ".join(f"{(lc / lc0) ** 2:.3f}" for lc in lcs[1:]) + ".",
        f"Predicted energy norm relative to lc {lc0} (label-free), seed means, lc "
        + " / ".join(str(lc) for lc in lcs[1:]) + f": {unorm}.",
        "Post-hoc reading 4e (E-series runbook, Sec. 4): reported only.",
    ]


def _latex_escape(s: str) -> str:
    s = (s.replace("F^T u / (u^T K u)", r"$F^\top u / (u^\top K u)$")
         .replace("c*", r"$c^*$").replace("c_b", r"$c_b$").replace("→", r"$\rightarrow$"))
    return _check_latex(
        s.replace("%", r"\%").replace("±", r"$\pm$").replace("²", r"$^2$")
        .replace("×", r"$\times$").replace("–", "--").replace("≈", r"$\approx$")
        .replace("~", r"$\sim$").replace(MINUS, r"$-$")
        .replace("SE_rel", r"SE$_\mathrm{rel}$"))


def _latex_head(h: str) -> str:
    parts = [_latex_escape(x) for x in h.split("\n")]
    return parts[0] if len(parts) == 1 else r"\shortstack{" + r" \\ ".join(parts) + "}"


def _check_latex(s: str) -> str:
    """Refuse a string whose text-mode parts (outside $...$) still carry a raw
    ^ or _ or an unescaped % & #: it would break pdflatex."""
    import re

    for i, part in enumerate(s.split("$")):
        if i % 2 == 0 and (re.search(r"(?<!\\)[_%&#]", part) or "^" in part):
            raise SystemExit(f"LaTeX text not escaped: {s[:80]!r}")
    return s


def latex_table(head, rows, caption, label, notes) -> str:
    cols = "l" + "c" * (len(head) - 1)
    out = [r"% generated by scripts/make_wp8_paper_material.py from records/wp8; "
           r"needs booktabs and graphicx",
           r"\begin{table}[t]", r"\centering", r"\small",
           rf"\caption{{{_latex_escape(caption)}}}", rf"\label{{{label}}}",
           r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           " & ".join(_latex_head(h) for h in head) + r" \\", r"\midrule"]
    out += [" & ".join(_latex_escape(c) for c in r) + r" \\" for r in rows]
    out += [r"\bottomrule", r"\end{tabular}}", "",
            r"\smallskip", r"\begin{minipage}{\linewidth}\footnotesize\raggedright"]
    out += [_latex_escape(n) + (r"\par" if i < len(notes) - 1 else "")
            for i, n in enumerate(notes)]
    out += [r"\end{minipage}", r"\end{table}", ""]
    return "\n".join(out)


def md_table(head, rows, title, notes) -> str:
    out = [f"### {title}", "", "| " + " | ".join(h.replace("\n", " ") for h in head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    out += [""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(out)


E1_CAPTION = ("E1 (2D): SIGReg latent shaping of the label-free AR objective. Accuracy of "
              "the AR cell and separation readings of the pooled latents.")
E2_CAPTION = ("E2 (3D): token bottleneck against the point-token transformer under the "
              "same label-free AR objective, corpus, split, seeds and schedule.")
AMP_CAPTION = ("Post-hoc (3D): errors of the label-free AR cells before and after scaling each "
               "prediction by its energy-optimal amplitude c*, in-band and on the fine mesh "
               "(zero-shot).")
REMESH_CAPTION = ("Post-hoc (3D): the label-free amplitude c_b on fixed geometries and loads "
                  "meshed at four mesh sizes lc.")


def frontier_rows(e2: dict) -> list:
    rows = []
    for key, label, m in E2_ARCHS:
        a = e2["archs"][key]
        for s in range(len(a["egap"])):
            rows.append({"architecture": label, "tokens": "per_node" if m is None else m,
                         "seed": s, "step_ms_inband": a["step_ms"]["inband_0"],
                         "step_ms_fine": a["step_ms"]["fine"],
                         "egap_inband": a["egap"][s], "disp_inband": a["disp"][s],
                         "disp_fine": a["fine_disp"][s]})
    return rows


def frontier_csv(rows: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (repr(v) if isinstance(v, float) else v) for k, v in r.items()})
    return buf.getvalue()


# ---------------------------------------------------------------- figure
PALETTE = {"Point-token transformer": "#2a78d6", 512: "#eb6834", 1024: "#1baf7a"}
MARKERS = {"Point-token transformer": "o", 512: "s", 1024: "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def make_figure(e2: dict, out_dir: Path) -> list:
    """Two panels, one per K1 quantity, against the step time at that mesh size.
    Faint marks: seeds; ringed marks: seed means. Identity: colour AND marker
    shape (print / colour-vision safe), a legend and direct labels with thin
    leader lines (the two bottleneck clusters sit within 0.12 decades of each
    other in time, so their labels are placed in the empty band beside them)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    style = {"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
             "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
             "pdf.fonttype": 42}
    with plt.rc_context(style):                    # no global matplotlib state is changed
        return _draw(plt, e2, out_dir, FixedLocator, FuncFormatter, LogLocator, NullFormatter)


def _draw(plt, e2, out_dir, FixedLocator, FuncFormatter, LogLocator, NullFormatter) -> list:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
    n = e2["nodes"]
    panels = (("inband_0", "egap", f"In-band ({n['inband_0']:,}-node step)",
               "Relative energy gap (log scale)", True, (10, 1e4)),
              ("fine", "fine_disp", f"Fine mesh, zero-shot ({n['fine']:,}-node step)",
               "Relative L2 displacement error", False, (10, 1e5)))
    # label anchors in axes-fraction coordinates, per panel and identity
    place = {("inband_0", 512): (0.30, 0.86), ("inband_0", 1024): (0.30, 0.50),
             ("fine", 512): (0.25, 0.47), ("fine", 1024): (0.25, 0.90)}
    for ax, (ph, metric, title, ylab, logy, xlim) in zip(axes, panels, strict=True):
        for key, label, m in E2_ARCHS:
            a = e2["archs"][key]
            ident = label if m is None else m
            name = label if m is None else f"Bottleneck, M = {m:,}"
            x, ys = a["step_ms"][ph], a[metric]
            ax.scatter([x] * len(ys), ys, s=16, marker=MARKERS[ident], color=PALETTE[ident],
                       alpha=0.45, linewidths=0, zorder=2)
            ax.scatter([x], [_mean(ys)], s=56, marker=MARKERS[ident], color=PALETTE[ident],
                       edgecolors="white", linewidths=1.5, zorder=3, label=name)
            if m is None:                 # alone at the right: label to its left
                ax.annotate(name, (x, _mean(ys)), xytext=(-9, 0), textcoords="offset points",
                            fontsize=7, color=INK, va="center", ha="right")
            else:
                ax.annotate(name, (x, _mean(ys)), xytext=place[(ph, m)],
                            textcoords="axes fraction", fontsize=7, color=INK, va="center",
                            ha="left", arrowprops={"arrowstyle": "-", "color": INK2,
                                                   "linewidth": 0.6, "shrinkA": 2,
                                                   "shrinkB": 5})
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
        ax.set_xlabel("Label-free AR training step (ms, log scale)")
        ax.set_ylabel(ylab)
        ax.grid(True, which="major", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[1].legend(loc="lower left", fontsize=7, frameon=False, handletextpad=0.3,
                   borderaxespad=0.2)
    written = []
    for ext in ("pdf", "png"):
        p = out_dir / f"fig_e2_cost_accuracy.{ext}"
        meta = {"CreationDate": None} if ext == "pdf" else {"Software": None}
        fig.savefig(p, dpi=300, metadata=meta)
        written.append(p)
    plt.close(fig)
    return written


# ---------------------------------------------------------------- main
def build(rec: Path, probe_path: str | None) -> tuple:
    used: dict = {}
    e1, e2 = read_e1(rec, used), read_e2(rec, used)
    amp = read_amplitude(rec, used)
    probe = None
    if probe_path:
        probe = _load(Path(probe_path), used)
        _check_probe(e1, probe)
    h1, r1 = e1_rows(e1, probe)
    h2, r2 = e2_rows(e2)
    h3, r3 = amp_rows(amp)
    h4, r4 = remesh_rows(amp, e2)
    n1, n2 = e1_notes(e1, probe), e2_notes(e2)
    n3, n4 = amp_notes(amp, e2), remesh_notes(amp, e2)
    files = {
        "table_e1.tex": latex_table(h1, r1, E1_CAPTION, "tab:e1", n1),
        "table_e2.tex": latex_table(h2, r2, E2_CAPTION, "tab:e2", n2),
        "table_amplitude.tex": latex_table(h3, r3, AMP_CAPTION, "tab:amplitude", n3),
        "table_remesh.tex": latex_table(h4, r4, REMESH_CAPTION, "tab:remesh", n4),
        "tables.md": ("# wp8 E-series tables (generated from records/wp8 by "
                      "scripts/make_wp8_paper_material.py; do not edit)\n\n"
                      + md_table(h1, r1, E1_CAPTION, n1) + "\n"
                      + md_table(h2, r2, E2_CAPTION, n2) + "\n"
                      + md_table(h3, r3, AMP_CAPTION, n3) + "\n"
                      + md_table(h4, r4, REMESH_CAPTION, n4)),
        "frontier.csv": frontier_csv(frontier_rows(e2)),
        "sources.json": json.dumps({"generator": "scripts/make_wp8_paper_material.py",
                                    "inputs_sha256": dict(sorted(used.items()))},
                                   indent=1) + "\n",
    }
    return files, e1, e2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(REC))
    ap.add_argument("--out", default=str(ROOT / "paper" / "wp8"))
    ap.add_argument("--probe-random-init", default=None,
                    help="post-hoc 4a output for the untrained-model rows of the E1 table "
                         "(default: posthoc/probe_random_init.json under --records, if present)")
    ap.add_argument("--no-figure", action="store_true")
    a = ap.parse_args()
    probe = a.probe_random_init or default_probe(Path(a.records))
    files, _, e2 = build(Path(a.records), str(probe) if probe else None)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    written = [out / n for n in files]
    if not a.no_figure:
        written += make_figure(e2, out)
    print("\n".join(str(p) for p in written))


if __name__ == "__main__":
    main()
