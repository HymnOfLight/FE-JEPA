"""RESULTS.md renderer and the Figure-1 energy-gap curve.

Plan v2.0 mapping:
  - WP1 acceptance: "G1' verdict under frozen criteria; **filled RESULTS.md**" --
    :func:`write_results` fills it from a run report (the runner calls it
    automatically; ``fejepa results <report.json>`` re-renders any report).
  - Deliverables checklist items 3-4: the deciding-run RESULTS.md sections
    (E1'/E3'/E5'/E6 + measured transfer + G1' verdict) and "the Figure-1
    energy-gap curve" -- :func:`write_figures`.

Every section is guarded: experiments that did not run render as "not run", so the
same renderer serves smoke runs, partial runs, and the deciding run. Reports loaded
back from JSON carry *string* dictionary keys; :func:`_get` normalises int/str access
throughout (the same convention Gate G1' uses).

matplotlib is an optional dependency (dev extra); :func:`write_figures` raises a
clear ImportError if it is absent and the runner degrades gracefully.
"""

from __future__ import annotations

import json
from pathlib import Path


# ------------------------------------------------------------- small helpers --

def _get(d: dict, k):
    """int/str-key tolerant access (reports round-trip through JSON)."""
    if d is None:
        return None
    return d.get(k, d.get(str(k)))


def _f(x, p: int = 4) -> str:
    try:
        return f"{float(x):.{p}f}"
    except (TypeError, ValueError):
        return "--"


def _ms(cell: dict | None, key: str, p: int = 4) -> str:
    """'mean±std' for a mean_std record inside an _agg-shaped cell."""
    rec = (cell or {}).get(key)
    if not rec:
        return "--"
    return f"{_f(rec['mean'], p)}±{_f(rec['std'], p)}"


def _imp(rec: dict | None) -> str:
    if not rec:
        return "--"
    return f"{100 * rec['value']:+.2f}% (t={_f(rec['t'], 1)})"


def _load(payload_or_path) -> dict:
    if isinstance(payload_or_path, (str, Path)):
        return json.loads(Path(payload_or_path).read_text())
    return payload_or_path


# ------------------------------------------------------------------ renderer --

def render_results(payload: dict) -> str:
    r = payload.get("results", {})
    prov = payload.get("provenance", {})
    pol = payload.get("runtime_policy", {})
    L: list[str] = []
    add = L.append

    add("# FE-JEPA v2.0 -- RESULTS")
    add("")
    add(f"Run `{prov.get('config_sha256', '?')[:12]}` | {prov.get('timestamp_utc', '?')}"
        f" | git `{prov.get('git', '?')}` | device={pol.get('device', '?')}"
        f" tf32={pol.get('tf32', '?')} | workers={payload.get('config', {}).get('workers', 1)}")
    pre = payload.get("prereg")
    add(f"Pre-registration: verified `{pre['config_sha256'][:12]}` against"
        f" `{pre['file']}`" if pre else
        "Pre-registration: guard not enabled for this run")
    add("")

    # ---- gate ------------------------------------------------------------
    g2 = payload.get("gate_g2")
    if g2:
        add("## Gate G2 (PREREG_PHASE2)")
        c = g2["conditions"]
        add(f"**{'GO' if g2['passed'] else 'NO-GO'}** -- logic {g2['logic']}; "
            f"(a)={c['a']} (b)={c['b']} (c)={c['c']}; "
            f"transfer zone: {g2.get('transfer_zone')}")
        add("")
        add("| kill | triggered |")
        add("|---|---|")
        for k, v in g2["kills"].items():
            add(f"| {k} | {'**YES**' if v else 'no'} |")
        add("")
        add("Reasons:")
        for k, v in g2.get("reasons", {}).items():
            add(f"- `{k}`: {v}")
        add("")

    g = payload.get("gate_g1_prime")
    add("## Gate G1'" if not g2 else "## Gate G1' (legacy; not the Phase-2 gate)")
    if g:
        add(f"**Verdict: {'GO' if g['passed'] else 'NO-GO'}** "
            f"(decision budget {g['decision_budget']})")
        for c in ("a", "b", "c"):
            key = {"a": "a_sanity", "b": "b_physics", "c": "c_transfer"}[c]
            add(f"- ({c}) {'PASS' if g['conditions'][c] else 'FAIL'}: "
                f"{g['reasons'].get(key, '--')}")
        ret = g.get("retired_displacement_criterion")
        if ret and ret.get("fixed_disp_improvement_at_decision"):
            add(f"- retired criterion (Sec.5 item 1): fixed-lambda disp improvement "
                f"@ decision = {_imp(ret['fixed_disp_improvement_at_decision'])}")
    else:
        add("not run")
    add("")

    # ---- kills -----------------------------------------------------------
    add("## Kill conditions")
    rows = [(eid, k) for eid, res in r.items() for k in res.get("kills", [])]
    if rows:
        add("| exp | condition | triggered | note |")
        add("|---|---|---|---|")
        for eid, k in rows:
            add(f"| {eid} | {k['condition']} | "
                f"{'**YES**' if k['triggered'] else 'no'} | {k['note']} |")
    else:
        add("none evaluated")
    add("")

    # ---- E1' ---------------------------------------------------------------
    add("## E1' -- anchor as supervised auxiliary (P-A)")
    e1 = r.get("e1")
    if e1:
        add("| budget | no-anchor disp | balanced d-disp | balanced d-egap "
            "| fixed d-disp | fixed d-egap | grid-best (BIASED) |")
        add("|---|---|---|---|---|---|---|")
        for row in e1["metrics"]["per_budget"]:
            imp, gb = row["improvements"], row.get("grid_best")
            add(f"| {row['budget']} | {_ms(row['arms'].get('none'), 'disp')} "
                f"| {_imp(imp['balanced']['disp'])} | {_imp(imp['balanced']['egap'])} "
                f"| {_imp(imp['fixed']['disp'])} | {_imp(imp['fixed']['egap'])} "
                f"| {(_imp(gb['disp']) + ' [' + gb['arm'] + ']') if gb else '--'} |")
    else:
        add("not run")
    add("")

    # ---- E5' ---------------------------------------------------------------
    add("## E5' -- repaired sanity")
    e5 = r.get("e5")
    if e5:
        base = e5["metrics"]["baselines"]
        add("baseline disp rel-L2: " + ", ".join(
            f"{n}={_f(base[n]['disp_rel_l2'])}" for n in sorted(base)))
        add("")
        add("| budget | anchored disp | x over zero | beats all naive | pass |")
        add("|---|---|---|---|---|")
        for row in e5["metrics"]["per_budget"]:
            add(f"| {row['budget']} | {_f(row['anchored_disp'])} "
                f"| {_f(row['beats_zero_x'], 2)}x | {row['beats_all_naive']} "
                f"| {'PASS' if row['passed'] else 'FAIL'} |")
    else:
        add("not run")
    add("")

    # ---- E8 headline -------------------------------------------------------
    add("## E8 -- regime grid (headline table)")
    e8 = r.get("e8")
    if e8:
        budgets = e8["protocol"]["budgets"]
        cells = e8["metrics"]["cells"]
        order = ["labels", "labels_anchor", "ar_ft", "mgn",
                 "zero", "scale_aware_poly", "knn_field"]
        regimes = [x for x in order if _get(cells, x) is not None]
        for key, title in (("disp_rel_l2", "Displacement rel-L2"),
                           ("energy_gap_rel", "Relative energy gap")):
            add(f"**{title}**")
            add("| regime | " + " | ".join(str(b) for b in budgets) + " |")
            add("|---|" + "---|" * len(budgets))
            for reg in regimes:                # regime keys are always strings;
                col = cells[reg]               # only BUDGET keys need _get()
                add(f"| {reg} | " + " | ".join(_ms(_get(col, b), key)
                                               for b in budgets) + " |")
            add("")
        for p, cell in (_get(cells, "ar") or {}).items():
            add(f"AR (unlabeled pool {p}, 0 labels): "
                f"disp {_ms(cell, 'disp_rel_l2')}, "
                f"egap {_ms(cell, 'energy_gap_rel')}, "
                f"vM {_ms(cell, 'vm_rel_l2')}")
        auc = e8["metrics"].get("label_efficiency_auc_disp", {})
        add("Label-efficiency AUC (disp): "
            + ", ".join(f"{k}={_f(v)}" for k, v in auc.items()))
    else:
        add("not run")
    add("")

    # ---- E2 ------------------------------------------------------------------
    add("## E2 -- JEPA vs AR (P-B, one-shot verdict)")
    e2 = r.get("e2")
    if e2:
        add("| budget | JEPA-vs-AR d-disp | d-egap |")
        add("|---|---|---|")
        for b, d in e2["metrics"]["jepa_vs_ar_improvements"].items():
            add(f"| {b} | {100 * d['disp']:+.2f}% | {100 * d['egap']:+.2f}% |")
        dec = e2["metrics"].get("jepa_improvement_at_decision_budget")
        if dec:
            add(f"decision-budget improvement: disp {100 * dec['disp']:+.2f}%, "
                f"egap {100 * dec['egap']:+.2f}%")
    else:
        add("not run")
    add("")

    # ---- E3' / E4' / E6 / E7 / WP2 -------------------------------------------
    add("## E3' -- collapse (standardized rank)")
    e3 = r.get("e3")
    if e3:
        add(f"best std-rank ratio (reg on/off) = "
            f"{_f(e3['metrics']['best_std_ratio'], 3)} "
            f"(K4 threshold 1.5; see kills table)")
    else:
        add("not run")
    add("")

    add("## E4' -- cross-resolution invariance")
    e4 = r.get("e4")
    if e4:
        add("| coarsen | gap (inv off) | gap (inv on) | reduction | gap/abs-err (on) |")
        add("|---|---|---|---|---|")
        for row in e4["metrics"]["per_coarsen"]:
            add(f"| {row['coarsen']} | {_f(row['inv_off']['gap'])} "
                f"| {_f(row['inv_on']['gap'])} | {100 * row['gap_reduction']:+.1f}% "
                f"| {_f(row['inv_on']['gap_over_abs_err'], 3)} |")
    else:
        add("not run")
    add("")

    add("## E6 -- latent-physics alignment")
    e6 = r.get("e6")
    add(f"within-geometry mean rho = {_f(e6['metrics']['rho_within_mean'], 3)}; "
        f"cross-geometry (descriptor) rho = "
        f"{_f(e6['metrics']['rho_cross_descriptor'], 3)}" if e6 else "not run")
    add("")

    add("## E7 -- learned init + CG polish")
    e7 = r.get("e7")
    if e7:
        it = e7["metrics"]["iterations_to_tol_mean"]
        add(f"iterations-to-tol (mean): zero {_f(it['zero'], 1)}, "
            f"naive {_f(it['naive'], 1)}, learned {_f(it['learned'], 1)} -> "
            f"savings learned {100 * e7['metrics']['savings_learned']:.1f}%, "
            f"naive {100 * e7['metrics']['savings_naive']:.1f}%")
        pol = e7["metrics"]["polish_at_k"]
        add("| k | egap (rel) | disp rel-L2 |")
        add("|---|---|---|")
        for k, row in pol.items():
            add(f"| {k} | {_f(row['energy_gap_rel'])} | {_f(row['disp_rel_l2'])} |")
    else:
        add("not run")
    add("")

    add("## WP2 -- region-mask ratio sweep (pre-E2)")
    w2 = r.get("wp2")
    if w2:
        add("| ratio | held-out masked-pred MSE | pooled std-rank |")
        add("|---|---|---|")
        for row in w2["metrics"]["per_ratio"]:
            add(f"| {row['ratio']} | {_f(row['holdout_pred_mse'], 5)} "
                f"| {_f(row['pooled_std_rank'], 2)} |")
        add(f"recommended ratio: **{w2['metrics']['recommended_ratio']}** "
            "(held-out MSE; rank reported alongside)")
    else:
        add("not run")
    add("")

    add("## WP6 -- theory numeric falsification pass")
    w6 = r.get("wp6")
    if w6:
        m = w6["metrics"]
        add(f"- conditioning max ratio {_f(m['conditioning']['max_ratio'], 4)} "
            f"(bound holds: {m['conditioning']['holds']}; kappa "
            f"{_f(m['conditioning']['kappa_range'][0], 1)}--"
            f"{_f(m['conditioning']['kappa_range'][1], 1)})")
        add(f"- mode contraction max err {m['mode_contraction']['max_err']:.2e} "
            f"(holds: {m['mode_contraction']['holds']})")
        add(f"- Chebyshev polish worst measured/bound "
            f"{_f(m['chebyshev_polish']['worst_measured_over_bound'], 3)} "
            f"(holds: {m['chebyshev_polish']['holds']})")
        add(f"- Prop.1 premise: min within-geometry separation "
            f"{_f(m['prop1_premise']['min_separation'], 3)}")
        cx = m["prop1_naive_cross"]
        add(f"- naive cross-geometry extension falsified: "
            f"{cx['naive_extension_falsified']}"
            + (f" (witness d_cross={_f(cx['witness']['d_cross'], 3)} < "
               f"within-min={_f(cx['witness']['a_within_min'], 3)})"
               if cx.get("witness") else ""))
    else:
        add("not run")
    add("")

    # ---- WP5 / ledger / provenance -------------------------------------------
    add("## Data economy (WP5)")
    de = payload.get("data_economy")
    if de:
        add(f"- labelled instances: {de['labelled_instances']} "
            f"(val {de['labelled_val']} + pool prefix {de['labelled_pool_prefix']}), "
            f"{de['solves_per_labelled_instance']} solves each")
        add(f"- reference solves total: {de['reference_solves_total']}")
        add(f"- unlabeled pool depth used: {de['unlabeled_pool_depth_used']} "
            f"(x{_f(de['unlabeled_over_labelled_pool'], 1)} the labelled prefix)")
    else:
        add("not recorded")
    add("")
    add("## Solve ledger")
    led = payload.get("solve_ledger", {})
    for stage, n in (led.get("per_stage") or {}).items():
        add(f"- {stage}: {n}")
    add(f"- total: {led.get('total', 0)} "
        f"(wall {led.get('wall_clock_s', 0)} s)")
    add("")
    add("## Provenance")
    for d in prov.get("datasets", []):
        add(f"- {d['dir']}: n={d['n_instances']}, backend={d['backend']}, "
            f"manifest `{str(d['manifest_sha256'])[:12]}`")
    add("")
    return "\n".join(L)


def write_results(payload_or_path, out_path) -> Path:
    payload = _load(payload_or_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_results(payload))
    return out


# ---------------------------------------------------------------- Figure 1 ----

def write_figures(payload_or_path, out_dir) -> list[Path]:
    """Deliverable 4: the energy-gap label-efficiency curve from E8's cells."""
    payload = _load(payload_or_path)
    e8 = payload.get("results", {}).get("e8")
    if not e8:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:                       # optional dev-extra dependency
        raise ImportError("write_figures needs matplotlib "
                          "(pip install 'fejepa[dev]')") from e

    budgets = e8["protocol"]["budgets"]
    cells = e8["metrics"]["cells"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for reg, label in (("labels", "labels only"),
                       ("labels_anchor", "labels + anchor (balanced)"),
                       ("ar_ft", "AR -> fine-tune")):
        col = _get(cells, reg)
        if not col:
            continue
        y = [_get(col, b)["energy_gap_rel"]["mean"] for b in budgets]
        s = [_get(col, b)["energy_gap_rel"]["std"] for b in budgets]
        ax.errorbar(budgets, y, yerr=s, marker="o", capsize=3, label=label)
    for p, cell in (_get(cells, "ar") or {}).items():
        ax.axhline(cell["energy_gap_rel"]["mean"], linestyle="--", linewidth=1.2,
                   color="tab:green",
                   label=f"AR, pool {p} (0 labels)")
    ax.axhline(1.0, linestyle=":", linewidth=1.0, color="grey",
               label="zero predictor")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("label budget")
    ax.set_ylabel("relative energy gap (val mean)")
    ax.set_title("Figure 1 -- physics faithfulness vs. labels (E8)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out_dir) / "figure1_energy_gap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out]
