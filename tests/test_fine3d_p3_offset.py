"""fine3d (PREREG_FINE3D, R2a): P3 takes E8's seed offset. With
`e8.seed_offset` set, P3's zero-shot rows and few-shot units read the states
E8 trained (ar_p<pool>_s<offset + i>) and record those seeds; without it P3
reads seeds 0..n-1 exactly as before (the cross-branch regression,
records/wp9/regression_wp8_e2_m1024.json, holds the default path bitwise);
the post-hoc amplitude pass reads the offset seeds' states against their own
rows of the report; a seed offset written into P3's own block is refused."""

import contextlib
import hashlib
import io
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from fejepa.experiments.runner import run_config
from fejepa.fe.synthetic import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}
MODEL = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}


def _cfg(tmp, name, data, fine, offset=0, fewshot=()):
    e8 = {"enabled": True, "budgets": [2], "pool_sizes": [4], "seeds": 2, "ar_epochs": 1,
          "sup_epochs": 1, "include_mgn": False, "ar_only": True}
    if offset:
        e8["seed_offset"] = offset
    cfg = {"data": {"dir": str(data), "n": 12, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "asis"},
           "data_transfer": {"dir": str(fine), "n": 6, "seed": 4, "backend": "synthetic",
                             "labelled_policy": "asis",
                             "split": {"n_eval": 3, "n_fewshot_prefix": 3}},
           "split": {"n_val": 4, "seed": 1}, "model": MODEL,
           "sup": {"epochs": 1, "lr": 1e-3}, "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {"e8": e8,
                           "p3_transfer": {"enabled": True, "fewshot_budgets": list(fewshot),
                                           "fewshot_epochs": 1, "fewshot_lr": 1e-3,
                                           "naive_budget": 0}},
           "gate_g2": {"sanity_x": 3.0, "naive_set": [], "parity_band": 0.1,
                       "egap_adv_min": 0.4, "transfer_win": 1.25, "decision_budget": 2,
                       "logic": "a AND (b OR c)"},
           "device": "cpu", "workers": 1, "tf32": False, "prereg_guard": False,
           "out": str(tmp / name / "report.json")}
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p3off")
    data = generate_synthetic_dataset(tmp / "corpus", n=12, seed=3, labelled="all")
    fine = generate_synthetic_dataset(tmp / "fine", n=6, seed=4, labelled="all")
    run_config(_cfg(tmp, "A", data, fine))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                  # what the run prints, for its log
        run_config(_cfg(tmp, "O", data, fine, offset=3, fewshot=[2]))
    (tmp / "O.out").write_text(buf.getvalue())
    rep = {k: json.loads((tmp / k / "report.json").read_text()) for k in ("A", "O")}
    return tmp, rep


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_p3_reads_the_states_e8_trained(world):
    tmp, rep = world
    o = rep["O"]
    assert o["provenance"]["seeds"] == [3, 4]
    assert o["results"]["p3_transfer"]["protocol"]["seeds"] == [3, 4]
    assert rep["A"]["results"]["p3_transfer"]["protocol"]["seeds"] == [0, 1]
    assert sorted(p.name for p in (tmp / "O" / "e8_states").glob("ar_*.pt")) == \
        ["ar_p4_s3.pt", "ar_p4_s4.pt"]
    # P3's in-band zero-shot pass evaluates exactly E8's states on the same split
    e8 = o["results"]["e8"]["metrics"]["cells"]["ar"]["4"]["per_seed_eval"]
    p3 = o["results"]["p3_transfer"]["metrics"]["ar"]["inband"]["per_seed_eval"]
    for a, b in zip(e8, p3, strict=True):
        assert a["per_instance"] == b["per_instance"]
    fine = o["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["per_seed_eval"]
    assert [len(e["per_instance"]["disp_rel_l2"]) for e in fine] == [3, 3]


def test_few_shot_units_start_from_the_offset_states(world):
    tmp, _ = world
    cache = tmp / "O" / "e8_states" / "unit_cache_p3"
    for s in (3, 4):
        res = pickle.loads((cache / f"P3_finetune_b2_s{s}.pkl").read_bytes())
        assert res["pretrained_sha256"] == _sha(tmp / "O" / "e8_states" / f"ar_p4_s{s}.pt")
    assert not list(cache.glob("*_s0.pkl")) and not list(cache.glob("*_s1.pkl"))


def test_offset_seeds_train_other_states(world):
    tmp, rep = world
    a = rep["A"]["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    o = rep["O"]["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    assert sorted(o) == ["s3", "s4"]
    assert {v["sha256"] for v in a.values()}.isdisjoint({v["sha256"] for v in o.values()})
    zs = rep["O"]["results"]["p3_transfer"]["metrics"]["ar"]["fine"]["disp_rel_l2"]["per_seed"]
    assert all(np.isfinite(zs))


def test_a_seed_offset_in_p3s_block_is_refused(tmp_path, world):
    tmp, _ = world
    p = _cfg(tmp_path, "X", tmp / "corpus", tmp / "fine")
    c = json.loads(p.read_text())
    c["experiments"]["p3_transfer"]["seed_offset"] = 3
    p.write_text(json.dumps(c))
    with pytest.raises(SystemExit, match="takes E8's seed offset"):
        run_config(p)


def test_the_amplitude_pass_reads_the_offset_seeds(world, tmp_path):
    """scripts/posthoc_amplitude.py on the offset run, as the session runs it on
    R2a's arms (without the remesh pass, defined for gmsh3d corpora only): the
    states of seeds 3 and 4, each reproducing its own rows of the report (0 on
    CPU); the adjudicator accepts the record and reads its mechanism fields."""
    from fejepa.analysis import adjudicate_fine3d as A

    tmp, rep = world
    rp = tmp / "O" / "report.json"
    out = tmp_path / "amp_O.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_amplitude.py"),
                        "--report", str(rp), "--states-dir", str(tmp / "O" / "e8_states"),
                        "--n-inband", "4", "--n-fine", "3", "--device", "cpu", "--out", str(out)],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    amp = json.loads(out.read_text())
    d9 = rep["O"]["results"]["e8"]["metrics"]["d9_restart"]["ar_states"]
    assert sorted(amp["seeds"]) == ["s3", "s4"]
    assert amp["states"] == {k: v["sha256"] for k, v in d9.items()}
    assert amp["report_sha256"] == _sha(rp)
    for s in amp["seeds"].values():
        for name, n in (("inband", 4), ("fine", 3)):
            rpr = s[name]["reproduction"]
            assert rpr["checked"] and rpr["disp_max_rel_dev"] == rpr["egap_max_rel_dev"] == 0.0
            assert s[name]["summary"]["complete"] and len(s[name]["per_instance"]) == n
    A.check_amp(amp, _sha(rp))
    mech = A.mechanism(amp)
    assert set(mech) == {"inband", "fine"}
    assert len(mech["fine"]["c_star_median"]["per_seed"]) == 2
    with pytest.raises(ValueError, match="another report"):
        A.check_amp(amp, "0" * 64)


def test_p3_zero_shot_prints_its_progress(capsys):
    """P3's zero-shot pass prints a line every 32 instances (RUNBOOK_FINE3D
    Sec. 1: the R2a logs stay alive through it); what it evaluates is
    unchanged."""
    from fejepa.experiments.p3_transfer import _Progress

    items = list(range(70))
    p = _Progress(items, "bottleneck s3 fine")
    assert len(p) == 70 and list(p) == items
    assert capsys.readouterr().out.splitlines() == [
        f"[p3] zero-shot bottleneck s3 fine: {i}/70" for i in (0, 32, 64)]
    assert "`[p3] zero-shot ...`" in (ROOT / "RUNBOOK_FINE3D.md").read_text()


def test_the_runs_zero_shot_pass_prints_its_progress(world):
    """run_p3 itself: every seed's fine and in-band zero-shot evaluations print."""
    tmp, _ = world
    out = (tmp / "O.out").read_text().splitlines()
    for s in (3, 4):
        assert f"[p3] zero-shot fejepa s{s} fine: 0/3" in out
        assert f"[p3] zero-shot fejepa s{s} in band: 0/4" in out

