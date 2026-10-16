"""Tables and a chart of the element study (exploratory, after gate G-C1.3).

    python -I probes/elements/write_report.py STUDY.jsonl CHECKS.json OUT_DIR
Writes OUT_DIR/tables.md and OUT_DIR/convergence.png.
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fejoint.specimens import SJ_INI                            # noqa: E402

KINDS = ["H8I", "H8", "T4", "H20", "H20R", "H27", "T10"]
LABEL = {"H8I": "8-node hex, incompatible modes (current)", "H8": "8-node hex, full integration",
         "T4": "4-node tet", "H20": "20-node hex, full integration", "H20R": "20-node hex, reduced integration",
         "H27": "27-node hex", "T10": "10-node tet"}
COLOUR = {"H8I": "#2a78d6", "H8": "#eb6834", "T4": "#1baf7a", "H20": "#eda100", "H20R": "#e87ba4",
          "H27": "#008300", "T10": "#6250d6"}
MARK = {"H8I": "o", "H8": "s", "T4": "^", "H20": "D", "H20R": "v", "H27": "P", "T10": "X"}
UNIFORM = [(20.0, 1, 1, 1), (10.0, 1, 1, 1), (5.0, 2, 2, 1), (2.5, 4, 4, 2), (2.5, 8, 4, 2)]
LAYERS = [(5.0, 1, 2, 1), (5.0, 2, 2, 1), (5.0, 4, 2, 1), (5.0, 8, 2, 1)]


def load(path):
    recs = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        m = r["mesh"]
        recs[(r["specimen"], r["kind"], m["h"], m["n_tp"], m["n_tf"], m["n_tw"])] = r
    return recs


def fmt(v, d=2):
    return "" if v is None else f"{v:.{d}f}"


def main(study, checks, outdir):
    out = pathlib.Path(outdir); out.mkdir(parents=True, exist_ok=True)
    R = load(study)
    C = json.loads(pathlib.Path(checks).read_text())
    L = []
    w = L.append
    w("### Checks of the elements\n")
    w("| Element | Rigid-body residual | Patch test, energy error | Cantilever tip deflection / Timoshenko: 10x1x1, 20x2x1, 40x4x2, 80x8x4 |")
    w("|---|---|---|---|")
    for k in KINDS:
        p = C["patch"][k]
        ct = ", ".join(f"{abs(c['tip_over_timoshenko']):.3f}" for c in C["cantilever"][k])
        w(f"| {k} | {p['rigid_residual']:.0e} | {p['patch_energy_rel_err']:.0e} | {ct} |")
    w("")
    for name in ("FS1", "FS3"):
        tests = [v for t, v in SJ_INI.items() if t.startswith(name)]
        w(f"### {name}: initial stiffness S (kNm/mrad); tests {tests[0]} and {tests[1]}\n")
        w("Refinement (h in mm; layers through the end plate, each beam flange, half the web; the last column adds plate layers only):\n")
        w("| Element | " + " | ".join(f"h {h:g}, {a}/{b}/{c}" for h, a, b, c in UNIFORM) + " | Unknowns at the finest |")
        w("|---|" + "---|" * (len(UNIFORM) + 1))
        for k in KINDS:
            vals, dof = [], None
            for (h, a, b, c) in UNIFORM:
                r = R.get((name, k, h, a, b, c))
                vals.append(fmt(r["S_paper_kNm_per_mrad"]) if r else "")
                if r:
                    dof = r["mesh"]["dof_reduced"]
            w(f"| {k} | " + " | ".join(vals) + f" | {dof:,} |" if dof else f"| {k} | " + " | ".join(vals) + " | |")
        w("")
        w("Layers through the end plate at h = 5 mm (2 per flange, 1 per half web):\n")
        w("| Element | " + " | ".join(f"{a} layer{'s' if a > 1 else ''}" for _, a, _, _ in LAYERS) + " |")
        w("|---|" + "---|" * len(LAYERS))
        for k in KINDS:
            vals = [fmt(R[(name, k, h, a, b, c)]["S_paper_kNm_per_mrad"]) if (name, k, h, a, b, c) in R else ""
                    for (h, a, b, c) in LAYERS]
            if any(vals):
                w(f"| {k} | " + " | ".join(vals) + " |")
        w("")
        best = []
        for k in KINDS:
            cands = [r for (s, kk, *_), r in R.items() if s == name and kk == k]
            if cands:
                r = max(cands, key=lambda r: r["mesh"]["dof_reduced"])
                best.append((k, r))
        w("Finest mesh of each element and its ratio to the two tests:\n")
        w("| Element | Mesh | Unknowns | S | S / test a | S / test b | PDAS iterations | Largest optimality residual |")
        w("|---|---|---|---|---|---|---|---|")
        for k, r in best:
            m = r["mesh"]; kk = r["kkt"]
            res = max(abs(kk["min_lambda"]), kk["max_complementarity"], kk["max_lambda_inactive"], kk["max_residual_unconstrained"])
            w(f"| {k} | h {m['h']:g}, {m['n_tp']}/{m['n_tf']}/{m['n_tw']} | {m['dof_reduced']:,} | {r['S_paper_kNm_per_mrad']:.2f} | "
              f"{r['R_tests'][0]:.2f} | {r['R_tests'][1]:.2f} | {r['pdas_iterations']} | {res:.0e} |")
        w("")
    (out / "tables.md").write_text("\n".join(L) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200, sharey=False)
    for ax, name in zip(axes, ("FS1", "FS3")):
        tests = [v for t, v in SJ_INI.items() if t.startswith(name)]
        ax.axhspan(min(tests), max(tests), color="#d8dcdf", lw=0, zorder=0)    # between the two tests
        for tv in tests:
            ax.axhline(tv, color="#8c939a", lw=1.0, zorder=1)
        ax.text(4.6e3, max(tests) + 2.0, f"tests (Table 8): {name}a {tests[0]:g}, {name}b {tests[1]:g}",
                color="#52514e", fontsize=9)
        def finest(k):
            cands = [r for (sp_, kk, *_), r in R.items() if sp_ == name and kk == k]
            return max(cands, key=lambda r: r["mesh"]["dof_reduced"])["S_paper_kNm_per_mrad"]
        conv = float(np.median([finest(k) for k in ("H8I", "H20", "H20R", "H27")]))
        ax.annotate("", xy=(2.2e5, max(tests) + 0.8), xytext=(2.2e5, conv - 1.0),
                    arrowprops=dict(arrowstyle="<->", color="#52514e", lw=1.0))
        ax.text(2.0e5, 0.5 * (conv + max(tests)), f"{conv / max(tests):.1f} to {conv / min(tests):.1f}\ntimes the tests",
                color="#52514e", fontsize=9, ha="right", va="center")
        for k in KINDS:
            pts = [(R[(name, k) + lv]["mesh"]["dof_reduced"], R[(name, k) + lv]["S_paper_kNm_per_mrad"])
                   for lv in UNIFORM if (name, k) + lv in R]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COLOUR[k], lw=2, marker=MARK[k], ms=7, mec="white", mew=1.0, label=LABEL[k])
            # the other plate-layer counts at h = 5 mm, as hollow markers without a line
            extra = [(R[(name, k) + lv]["mesh"]["dof_reduced"], R[(name, k) + lv]["S_paper_kNm_per_mrad"])
                     for lv in LAYERS if lv not in UNIFORM and (name, k) + lv in R]
            if extra:
                ex, ey = zip(*extra)
                ax.plot(ex, ey, ls="none", marker=MARK[k], ms=7, mfc="white", mec=COLOUR[k], mew=1.6, zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(0, 90 if name == "FS1" else 100)
        ax.set_xlim(4e3, 3e5)
        ax.set_xlabel("unknowns (log scale)", color="#52514e")
        ax.set_ylabel("initial stiffness S (kNm/mrad)", color="#52514e")
        ax.set_title(f"{name}: {'10.4 mm plate' if name == 'FS1' else '20.0 mm plate'}", color="#0b0b0b", fontsize=11, loc="left")
        ax.grid(True, which="major", color="#e6e6e3", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#b9b8b3")
        ax.tick_params(colors="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(matplotlib.lines.Line2D([], [], ls="none", marker="o", ms=7, mfc="white", mec="#52514e", mew=1.6))
    labels.append("hollow: h 5 mm with 1, 4 or 8 plate layers")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "convergence.png", facecolor="white")
    print((out / "tables.md").read_text())


if __name__ == "__main__":
    main(*sys.argv[1:4])
