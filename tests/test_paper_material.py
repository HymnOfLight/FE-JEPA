"""wp8 Stage 1.36 (wrap-up item 9; post-hoc tables Stage 1.38): the paper
material is generated from the committed records only, reproduces byte for
byte, and agrees with the verdicts the adjudicators computed from the same
files and with the post-hoc records."""

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "wp8"
REC = ROOT / "records" / "wp8"
TEXT = ("table_e1.tex", "table_e2.tex", "table_amplitude.tex", "table_remesh.tex", "tables.md",
        "frontier.csv", "sources.json")


def _mod():
    spec = importlib.util.spec_from_file_location(
        "make_wp8_paper_material", ROOT / "scripts" / "make_wp8_paper_material.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_material_regenerates_byte_for_byte():
    mod = _mod()
    probe = mod.default_probe(REC)
    assert probe is not None                         # post-hoc reading 4a is on record
    files, _, _ = mod.build(REC, str(probe))
    assert set(files) == set(TEXT)
    for name, text in files.items():
        assert (OUT / name).read_text(encoding="utf-8") == text, name
    md = files["tables.md"]
    for label in ("Untrained, descriptor input", "Untrained, descriptor weights zeroed",
                  "Untrained, no descriptor input"):
        assert f"| {label} | 3 |" in md


def test_material_agrees_with_the_verdicts():
    mod = _mod()
    _, e1, e2 = mod.build(REC, None)
    rec = REC
    v1 = json.loads((rec / "e1" / "e1_verdict.json").read_text())
    k1 = v1["K1_detail"]
    assert statistics.fmean(e1["arms"]["base"]["disp"]) == pytest.approx(
        k1["disp_rel_l2"]["base_mean"], rel=1e-12)
    assert statistics.fmean(e1["arms"]["shaped"]["egap"]) == pytest.approx(
        k1["energy_gap_rel"]["new_mean"], rel=1e-12)
    assert e1["arms"]["base"]["S"] == v1["S_base"] and e1["arms"]["shaped"]["S"] == v1["S_shaped"]
    for m in (512, 1024):
        v = json.loads((rec / "e2" / f"e2_verdict_M{m}.json").read_text())
        a, t = e2["archs"][f"m{m}"], e2["archs"]["transformer"]
        assert statistics.fmean(a["egap"]) == pytest.approx(v["egap_mean_e2"], rel=1e-12)
        assert statistics.fmean(a["fine_disp"]) == pytest.approx(v["fine_disp_e2"], rel=1e-12)
        assert statistics.fmean(t["egap"]) == pytest.approx(v["egap_mean_base"], rel=1e-12)
        assert statistics.fmean(t["fine_disp"]) == pytest.approx(v["fine_disp_base"], rel=1e-12)
        assert a["step_ms"]["fine"] / 1000 == pytest.approx(v["fine_step_s"], rel=1e-12)
    md = (OUT / "tables.md").read_text(encoding="utf-8")
    assert "Verdict: NO-GO" in md and md.count("Verdict: KILLED") == 2
    sources = json.loads((OUT / "sources.json").read_text())["inputs_sha256"]
    assert sources["records/wp8/e2/baseline/report_phase2b.json"].startswith("320b6db5")


def _probe_file(tmp_path, pc1, arms=("true", "zeroed", "false")):
    readings = {}
    for arm in arms:
        base = 0.30 if arm == "true" else -0.05
        for s in range(3):
            readings[f"geometry_input_{arm}_s{s}"] = {
                "S_silhouette": base + 0.01 * s, "probe_r2_geometry": 1.0 if arm == "true" else 0.19,
                "loo_1nn_bin_accuracy": 0.97 if arm == "true" else 0.25, "pc1_variance_share": pc1,
                "sigreg_monitor_pooled": 0.2, "sigreg_monitor_tokens": 0.2}
    pf = tmp_path / "probe.json"
    pf.write_text(json.dumps({"n_instances": 256, "readings": readings}))
    return pf


def test_untrained_rows_from_the_post_hoc_probe(tmp_path):
    mod = _mod()
    pc1 = json.loads((REC / "e1" / "sep_base_s0.json").read_text())["pc1_variance_share"]
    pf = _probe_file(tmp_path, pc1)
    files, _, _ = mod.build(REC, str(pf))
    md = files["tables.md"]
    assert "| Untrained, descriptor input | 3 | – | – | 0.310 ± 0.010 | 1.000 ± 0.000 | 0.97 ± 0.00 |" in md
    assert "| Untrained, descriptor weights zeroed | 3 |" in md
    assert "| Untrained, no descriptor input | 3 |" in md and "Untrained rows (post-hoc reading 4a" in md
    assert "chance 0.25 with 4 bins of 64" in md
    assert str(pf) in json.loads(files["sources.json"])["inputs_sha256"]
    # a probe file from before the paired arm (Stage 1.36 layout) still renders its two rows
    files, _, _ = mod.build(REC, str(_probe_file(tmp_path, pc1, arms=("true", "false"))))
    assert "descriptor weights zeroed" not in files["tables.md"]


def test_a_probe_of_other_instances_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="PC1 share"):
        _mod().build(REC, str(_probe_file(tmp_path, 0.76)))


def test_amplitude_tables_agree_with_the_records():
    mod = _mod()
    used = {}
    amp = mod.read_amplitude(REC, used)
    _, _, e2 = mod.build(REC, None)
    assert amp["n_rows"] == 4608 and amp["n_egap_up"] == 0
    for key in ("transformer", "m512", "m1024"):
        a, b = amp["archs"][key], e2["archs"][key]
        for ours, theirs in (("inband_disp", "disp"), ("inband_egap", "egap"),
                             ("fine_disp", "fine_disp")):
            if key == "transformer":                 # reproduced bitwise
                assert a[ours] == b[theirs]
            else:                                    # atomic scatter-mean
                assert a[ours] == pytest.approx(b[theirs], rel=1e-3)
        assert all(c < d for c, d in zip(a["fine_disp_c"], a["fine_disp"], strict=True))
    tr = amp["archs"]["transformer"]
    assert statistics.fmean(tr["fine_c_star_median"]) > 1.1 > statistics.fmean(
        tr["inband_c_star_median"]) > 0.99
    md = (OUT / "tables.md").read_text(encoding="utf-8")
    assert "| Point-token transformer | one per node | 0.0300 ± 0.0028 | 0.0142 ± 0.0032 | 0.258 ± 0.045 |" in md
    assert amp["lc"] == [0.0906, 0.0742, 0.0579, 0.0374]
    sources = json.loads((OUT / "sources.json").read_text())["inputs_sha256"]
    assert {f"records/wp8/posthoc/amp_{tag}.json"
            for tag in ("phase2b", "e2_m512", "e2_m1024")} <= set(sources)


def test_latex_text_is_checked():
    mod = _mod()
    assert mod._latex_escape("SE_rel 5% c* c_b → F^T u / (u^T K u)").count("$") % 2 == 0
    for bad in ("a_b", "x^2", "R&D", "#1"):
        with pytest.raises(SystemExit, match="LaTeX"):
            mod._latex_escape(bad)


def test_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    mod = _mod()
    _, _, e2 = mod.build(REC, None)
    paths = mod.make_figure(e2, tmp_path)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.stat().st_size > 10_000 for p in paths)
