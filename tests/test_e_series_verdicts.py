"""wp8-lejepa Stage 1.35: the E1 and E2 verdicts reproduce from the committed records.

The box returned the run reports, separation readings and verdicts of both
pre-registered experiments (`records/wp8/e1/`, `records/wp8/e2/`, with a copy
of the Phase-2b baseline report E2 is judged against). These tests re-derive
each verdict from those files with the adjudicators as committed, check that
every verdict was computed from exactly these files (input SHA-256), and tie
each report to its stamp line (the configuration SHA-256 the run verified is
the one PREREG_E1 / PREREG_E2 records)."""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "records" / "wp8"


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path) -> dict:
    return json.loads(Path(path).read_text())


def _as_written(obj) -> dict:
    return json.loads(json.dumps(obj))           # the verdict files are plain json.dumps


def _stamped(prereg: str) -> dict:
    from fejepa.report import read_prereg_entries

    return {lab: val for lab, val in read_prereg_entries(ROOT / prereg)}


def _check_inputs(hashes: dict, files: dict) -> None:
    """Every input the verdict names (box paths) is the record file mapped to it."""
    assert set(hashes) == set(files), sorted(hashes)
    for name, digest in hashes.items():
        assert _sha(files[name]) == digest, name


def _s_by_seed(paths, report) -> list:
    states = report["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    rows = {}
    for p in paths:
        d = _load(p)
        s = int(re.search(r"_s(\d+)\.pt$", d["state"]).group(1))
        assert d["state_sha256"] == states[f"s{s}"]["sha256"], p.name
        assert d["config_sha256"] == report["provenance"]["config_sha256"], p.name
        assert d["S_valid"] is True and d["subset"] == "val" and d["n_instances"] == 256
        assert s not in rows
        rows[s] = d["S_silhouette"]
    assert sorted(rows) == [int(x) for x in report["provenance"]["seeds"]]
    return [rows[s] for s in sorted(rows)]


def test_e1_verdict_reproduces_from_the_records():
    from fejepa.analysis.adjudicate import adjudicate_e1

    e1 = REC / "e1"
    stamps = _stamped("PREREG_E1.md")
    reports = {a: _load(e1 / f"e1_2d_{a}" / "report.json") for a in ("base", "shaped", "raw_s0")}
    for arm, rep in reports.items():
        assert rep["provenance"]["config_sha256"] == stamps[f"e1_2d_{arm}"], arm
        assert rep["prereg"]["config_sha256"] == stamps[f"e1_2d_{arm}"], arm
        assert rep["solve_ledger"]["total"] == 0, arm
    base, shaped = reports["base"], reports["shaped"]
    s_base = _s_by_seed(sorted(e1.glob("sep_base_s*.json")), base)
    s_shaped = _s_by_seed(sorted(e1.glob("sep_shaped_s*.json")), shaped)
    want = _load(e1 / "e1_verdict.json")
    hashes = want.pop("inputs_sha256")
    assert _as_written(adjudicate_e1(base, shaped, s_base, s_shaped, 0.10)) == want
    assert (want["verdict"], want["K1_parity"], want["K2_no_effect"], want["GO"]) == \
        ("NO-GO", False, False, False)
    files = {f"runs/e1_2d_{a}/report.json": e1 / f"e1_2d_{a}" / "report.json"
             for a in ("base", "shaped")}
    files.update({f"runs/wp8/{p.name}": p for p in e1.glob("sep_*_s*.json")
                  if not p.name.startswith("sep_raw")})
    _check_inputs(hashes, files)


@pytest.mark.parametrize("m", [512, 1024])
def test_e2_verdict_reproduces_from_the_records(m):
    from fejepa.analysis.adjudicate import adjudicate_e2

    e2 = REC / "e2"
    base_path = e2 / "baseline" / "report_phase2b.json"
    assert _sha(base_path) in (ROOT / "PREREG_E2.md").read_text()     # the baseline PREREG_E2 names
    base, rep = _load(base_path), _load(e2 / f"e2_m{m}" / "report.json")
    stamp = _stamped("PREREG_E2.md")[f"e2_m{m}"]
    assert rep["provenance"]["config_sha256"] == rep["prereg"]["config_sha256"] == stamp
    assert rep["solve_ledger"]["total"] == 0
    bench_path = REC / f"bench_e2_m{m}.json"
    want = _load(e2 / f"e2_verdict_M{m}.json")
    hashes = want.pop("inputs_sha256")
    assert _as_written(adjudicate_e2(base, rep, _load(bench_path), m)) == want
    assert (want["verdict"], want["K1_accuracy"], want["K2_speed"], want["GO"]) == \
        ("KILLED", True, False, False)
    _check_inputs(hashes, {"runs/phase2/report_phase2b.json": base_path,
                           f"runs/e2_m{m}/report.json": e2 / f"e2_m{m}" / "report.json",
                           f"records/wp8/bench_e2_m{m}.json": bench_path})
