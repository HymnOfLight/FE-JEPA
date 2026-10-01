"""wp8 Stage 1.36 (wrap-up item 9): the paper material is generated from the
committed records only, reproduces byte for byte, and agrees with the
verdicts the adjudicators computed from the same files."""

import importlib.util
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "wp8"
TEXT = ("table_e1.tex", "table_e2.tex", "tables.md", "frontier.csv", "sources.json")


def _mod():
    spec = importlib.util.spec_from_file_location(
        "make_wp8_paper_material", ROOT / "scripts" / "make_wp8_paper_material.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_material_regenerates_byte_for_byte():
    files, _, _ = _mod().build(ROOT / "records" / "wp8", None)
    assert set(files) == set(TEXT)
    for name, text in files.items():
        assert (OUT / name).read_text(encoding="utf-8") == text, name


def test_material_agrees_with_the_verdicts():
    mod = _mod()
    _, e1, e2 = mod.build(ROOT / "records" / "wp8", None)
    rec = ROOT / "records" / "wp8"
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


def test_untrained_rows_from_the_post_hoc_probe(tmp_path):
    mod = _mod()
    readings = {}
    for geo, base in ((True, 0.30), (False, -0.05)):
        for s in range(3):
            readings[f"geometry_input_{str(geo).lower()}_s{s}"] = {
                "S_silhouette": base + 0.01 * s, "probe_r2_geometry": 1.0 if geo else 0.19,
                "loo_1nn_bin_accuracy": 0.97 if geo else 0.25, "pc1_variance_share": 0.76,
                "sigreg_monitor_pooled": 0.2, "sigreg_monitor_tokens": 0.2}
    pf = tmp_path / "probe.json"
    pf.write_text(json.dumps({"readings": readings}))
    files, _, _ = mod.build(ROOT / "records" / "wp8", str(pf))
    md = files["tables.md"]
    assert "| Untrained, descriptor input | 3 | – | – | 0.310 ± 0.010 | 1.000 ± 0.000 | 0.97 ± 0.00 |" in md
    assert "| Untrained, no descriptor input | 3 |" in md and "(untrained rows)" in md
    assert str(pf) in json.loads(files["sources.json"])["inputs_sha256"]


def test_figure_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    mod = _mod()
    _, _, e2 = mod.build(ROOT / "records" / "wp8", None)
    paths = mod.make_figure(e2, tmp_path)
    assert [p.suffix for p in paths] == [".pdf", ".png"]
    assert all(p.stat().st_size > 10_000 for p in paths)
