"""wp8 Stage 1.38: the post-hoc records (RUNBOOK_E_SERIES Sec. 4, returned on
1 October 2026 and copied byte for byte into records/wp8/posthoc/) are
complete and tied to the stamped runs they read: every step exited 0 on the
commit that carried the instruments, every measured state is the one its
run's report records, every report named is the committed record, the probe
read E1's own instances, and the uncorrected amplitude numbers reproduce the
reports (the transformer bitwise)."""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "records" / "wp8"
PH = REC / "posthoc"
HEAD = "0b3cd101961514548c5c2e5d21dabc42d89b7464"      # Stage 1.37, the instruments' commit
TREE = "9ae35919b1d22f5bd8c2398d000ad365be11fb65"
V215 = "a548825da6d02c91c7ccc4b6fcd6814116b2117e"      # tag v2.1.5, the timing reference
RUNS = {"phase2b": ("e2/baseline/report_phase2b.json", "runs/phase2/e8_states"),
        "e2_m512": ("e2/e2_m512/report.json", "runs/e2_m512/e8_states"),
        "e2_m1024": ("e2/e2_m1024/report.json", "runs/e2_m1024/e8_states")}
ARMS = ("geometry_input_true", "geometry_input_zeroed", "geometry_input_false")


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(name: str) -> dict:
    return json.loads((PH / name).read_text())


def _report(rel: str) -> dict:
    return json.loads((REC / rel).read_text())


def test_every_step_exited_zero_on_the_instrument_commit():
    lines = [x for x in (PH / "status.txt").read_text().splitlines() if x.strip()]
    assert len(lines) == 12 and all(x.endswith(" exit=0") for x in lines)
    assert len(list(PH.glob("*.json"))) == 12
    prov = (PH / "provenance.txt").read_text()
    assert f"HEAD {HEAD}" in prov and f"tree {TREE}" in prov
    assert V215 in prov                                    # the v2.1.5 worktree's head
    for tag in RUNS:
        for kind in ("amp", "anat"):
            assert _json(f"{kind}_{tag}.json")["git"].endswith("-g" + HEAD[:7])
    assert _json("probe_random_init.json")["git"].endswith("-g" + HEAD[:7])
    assert _json("profile_2d_head.json")["src_head"] == HEAD
    assert _json("profile_2d_v215.json")["src_head"] == V215
    assert "318 passed" in (PH / "pytest.log").read_text()


def test_measured_states_are_the_runs_own():
    prov = (PH / "provenance.txt").read_text()
    listed = {p: h for h, p in re.findall(r"^([0-9a-f]{64})\s+(\S+)$", prov, re.M)}
    for tag, (rel, states_dir) in RUNS.items():
        rep = _report(rel)
        want = {k: v["sha256"] for k, v in
                rep["results"]["e8"]["metrics"]["d9_restart"]["ar_states"].items()}
        seeds = [int(s) for s in rep["provenance"]["seeds"]]
        for s in seeds:
            assert listed[f"{states_dir}/ar_p1024_s{s}.pt"] == want[f"s{s}"]
        for kind in ("amp", "anat"):
            d = _json(f"{kind}_{tag}.json")
            assert d["report_sha256"] == _sha(REC / rel)
            assert d["states"] == {f"s{s}": want[f"s{s}"] for s in seeds}
    # 4c (intrinsic dimension) does not verify its input in-run: it read these paths,
    # whose SHA-256 the provenance lists and the first loop ties to the Phase-2b report
    for s in (0, 1, 2):
        assert _json(f"id_phase2b_s{s}.json")["state"] == f"runs/phase2/e8_states/ar_p1024_s{s}.pt"


def test_probe_read_e1s_own_instances():
    d = _json("probe_random_init.json")
    assert d["report_sha256"] == _sha(REC / "e1" / "e1_2d_base" / "report.json")
    sep = json.loads((REC / "e1" / "sep_base_s0.json").read_text())
    assert d["pc1_matches_record"] is True and d["pc1_record"] == sep["pc1_variance_share"]
    assert d["n_instances"] == sep["n_instances"] == 256
    assert set(d["readings"]) == {f"{a}_s{s}" for a in ARMS for s in (0, 1, 2)}
    assert {r["pc1_variance_share"] for r in d["readings"].values()} == {sep["pc1_variance_share"]}


def test_uncorrected_amplitude_numbers_reproduce_the_reports():
    for tag, (rel, _) in RUNS.items():
        rep = _report(rel)
        cells = rep["results"]["e8"]["metrics"]["cells"]["ar"]
        sets = {"inband": cells[max(cells, key=int)],
                "fine": rep["results"]["p3_transfer"]["metrics"]["ar"]["fine"]}
        d = _json(f"amp_{tag}.json")
        for st, cell in sets.items():
            for i, s in enumerate(rep["provenance"]["seeds"]):
                summ = d["seeds"][f"s{s}"][st]["summary"]
                assert summ["complete"] and summ["n_instances"] == 256
                for k, rk in (("disp", "disp_rel_l2"), ("egap", "energy_gap_rel")):
                    want = float(cell[rk]["per_seed"][i])
                    if tag == "phase2b":                   # the same evaluation path, bitwise
                        assert summ[k] == want
                    else:                                  # atomic CUDA scatter-mean
                        assert summ[k] == pytest.approx(want, rel=1e-3)


def test_per_instance_reproduction_matches_the_runs_own_spread():
    """Per instance: the transformer exactly; the bottleneck against its own
    run's two evaluations of the same states (E8 and P3, in-band): per-seed
    median deviations within 3x of the run's (observed 0.88-1.36x), the single
    largest within 4x of the run's largest (observed 3.7x). Both bounds guard
    these frozen records and were set after seeing them."""
    import numpy as np

    for tag, (rel, _) in RUNS.items():
        rep = _report(rel)
        cells = rep["results"]["e8"]["metrics"]["cells"]["ar"]
        e8 = cells[max(cells, key=int)]["per_seed_eval"]
        p3 = rep["results"]["p3_transfer"]["metrics"]["ar"]
        d = _json(f"amp_{tag}.json")
        ours, run_max = [], 0.0
        for i, s in enumerate(rep["provenance"]["seeds"]):
            for st, ref in (("inband", e8[i]), ("fine", p3["fine"]["per_seed_eval"][i])):
                rows = d["seeds"][f"s{s}"][st]["per_instance"]
                for k, rk in (("disp", "disp_rel_l2"), ("egap", "energy_gap_rel")):
                    want = np.asarray(ref["per_instance"][rk], float)
                    got = np.asarray([r[k] for r in rows], float)
                    dev = np.abs(got - want) / np.abs(want)
                    if tag == "phase2b":
                        assert np.array_equal(got, want)
                        continue
                    ours.append(dev.max())
                    if st == "inband":
                        x = np.asarray(p3["inband"]["per_seed_eval"][i]["per_instance"][rk], float)
                        run = np.abs(x - want) / np.abs(want)
                        run_max = max(run_max, run.max())
                        assert np.median(dev) <= 3 * np.median(run) + 1e-9
        if tag != "phase2b":
            assert max(ours) <= 4 * run_max


def test_the_amplitude_never_raised_the_energy_gap():
    n = 0
    for tag in RUNS:
        for seed in _json(f"amp_{tag}.json")["seeds"].values():
            for st in ("inband", "fine"):
                for r in seed[st]["per_instance"]:
                    assert r["egap_c"] <= r["egap"] * (1 + 1e-12)
                    n += 1
    assert n == 3 * 3 * 2 * 256
