"""fine3d (PREREG_FINE3D Sec. 2 and 5, R2a): the exact R2a path on a CPU
miniature -- the generator's S configuration (the token bottleneck with S,
E8 AR on the fresh seeds, P3 zero-shot), shrunk to gmsh3d corpora of a dozen
instances and run through run-config as the session runs it, then the
post-hoc amplitude pass on its report, with a remesh geometry as the session
runs it. The run trains and evaluates the fresh seeds and reads no instance of
the corpus's tail (PREREG_FINE3D Sec. 4); P3 evaluates exactly the states E8
trained (its in-band pass equals E8's AR cell per instance, bit for bit on
CPU); the amplitude pass reproduces the report's rows exactly. A P3 or
post-hoc model built without S would not."""

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("gmsh")

from fejepa.experiments.runner import run_config
from fejepa.metrics import FIELD_KEYS

ROOT = Path(__file__).resolve().parents[1]
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}


_READ = None


def _audit(event, args):
    """Files opened for reading while _READ is a list (an audit hook cannot be
    removed, so it records only then)."""
    if _READ is not None and event == "open" and isinstance(args[0], (str, os.PathLike)) \
            and (args[1] is None or "r" in str(args[1])):
        _READ.append(os.fspath(args[0]))


sys.addaudithook(_audit)


def _gen():
    spec = importlib.util.spec_from_file_location("make_fine3d_configs",
                                                  ROOT / "scripts" / "make_fine3d_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run_s(tmp_path_factory):
    """configs/r2a_s_m1024.json as the generator makes it from E2's stamped
    configuration, with the corpora, the model and the training shrunk."""
    tmp = tmp_path_factory.mktemp("r2a_path")
    base = json.loads((ROOT / "configs" / "e2_m1024.json").read_text())
    mini = copy.deepcopy(base)
    mini.update(data=dict(base["data"], dir=str(tmp / "data3d"), n=14, lc_range=[0.30, 0.36]),
                data_transfer=dict(base["data_transfer"], dir=str(tmp / "fine3d"), n=8, lc=0.24,
                                   split={"n_eval": 4, "n_fewshot_prefix": 4}),
                split={"n_val": 4, "seed": 1}, workers=1)
    mini["model"].update(dim=16, depth=1, heads=2, mgn_dim=16, mgn_depth=1, n_tokens=8,
                         decode_k=3)
    mini["experiments"]["e8"].update(pool_sizes=[8], seeds=2, ar_epochs=2, sup_epochs=2)
    cfg = _gen().r2a_config(mini, "r2a_s_m1024")
    cfg.update(prereg_guard=False, out=str(tmp / "s" / "report.json"), device="cpu", tf32=False)
    p = tmp / "r2a_s_mini.json"
    p.write_text(json.dumps(cfg))
    from fejepa.data.archive import instance_files

    global _READ
    _READ = []
    try:
        run_config(p)
        read = list(_READ)
    finally:
        _READ = None
    return (tmp, cfg, json.loads((tmp / "s" / "report.json").read_text()),
            {"read": read, "files": instance_files(tmp / "data3d")})


def test_the_s_run_trains_and_evaluates_the_fresh_seeds(run_s):
    tmp, cfg, rep, _ = run_s
    m = rep["config"]["model"]
    assert m["kind"] == "bottleneck" and m["decode_scale"] == "l1"
    assert m["decode_scale_factor"] == 2.0 ** -10 and m["features"]["load_density"] is True
    assert rep["provenance"]["seeds"] == [3, 4]
    assert rep["results"]["p3_transfer"]["protocol"]["seeds"] == [3, 4]
    assert sorted(p.name for p in (tmp / "s" / "e8_states").glob("ar_*.pt")) == \
        ["ar_p8_s3.pt", "ar_p8_s4.pt"]


def test_p3_evaluates_exactly_the_states_e8_trained(run_s):
    """P3 reloads each state from disk into a model built from the report's
    configuration; E8's AR cell evaluated the model it trained. On CPU the two
    agree per instance, bit for bit (on the box to about 3e-5 in the seed
    means: the adjudication refuses more than 1e-3)."""
    from fejepa.analysis import adjudicate_fine3d as A

    _, _, rep, _ = run_s
    p3, e8 = A.p3_cell(rep, "inband"), A.e8_cell(rep)
    assert len(p3["per_seed_eval"]) == len(e8["per_seed_eval"]) == 2
    for a, b in zip(p3["per_seed_eval"], e8["per_seed_eval"], strict=True):
        for k in FIELD_KEYS:
            assert a["per_instance"][k] == b["per_instance"][k], k
    assert A.p3_e8_deviation(rep) == 0.0


def test_the_s_run_reads_no_instance_of_the_tail(run_s):
    """PREREG_FINE3D Sec. 4: the validation split is fixed by its seed and the
    training pool is the first of the split's pool files, so the pool files
    after it (the tail R1's holdout is drawn from) are never opened."""
    from fejepa.experiments.protocol import load_split

    tmp, cfg, _, io = run_s
    split = load_split(tmp / "data3d", int(cfg["split"]["n_val"]), int(cfg["split"]["seed"]))
    pool = int(cfg["experiments"]["e8"]["pool_sizes"][0])
    tail = {str(f) for f in split.pool_files[pool:]}
    assert len(tail) == 2 and len(io["files"]) == 14
    opened = {str(Path(f)) for f in io["read"]}
    assert not opened & tail
    assert {str(f) for f in split.pool_files[:pool]} <= opened      # the pool was read
    assert {str(f) for f in split.val_files} <= opened


def test_the_amplitude_pass_reproduces_the_s_report(run_s, tmp_path):
    """scripts/posthoc_amplitude.py on the S report as the session runs it, with
    one remesh geometry instead of 16: the states of seeds 3 and 4, each
    reproducing its own rows of the report exactly, and the remesh pass on the
    bottleneck with S complete; the adjudicator reads all of it (its remesh
    count set to one)."""
    from fejepa.analysis import adjudicate_fine3d as A

    tmp, _, rep, _ = run_s
    out = tmp_path / "amp_s.json"
    rp = tmp / "s" / "report.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_amplitude.py"),
                        "--report", str(rp), "--states-dir", str(tmp / "s" / "e8_states"),
                        "--n-inband", "4", "--n-fine", "4", "--remesh", "1", "--device", "cpu",
                        "--out", str(out)],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    amp = json.loads(out.read_text())
    assert sorted(amp["seeds"]) == ["s3", "s4"] and amp["model_kind"] == "bottleneck"
    for s in amp["seeds"].values():
        for name in ("inband", "fine"):
            rpr = s[name]["reproduction"]
            assert rpr["checked"] and rpr["disp_max_rel_dev"] == rpr["egap_max_rel_dev"] == 0.0
        rem = s["remesh"]
        assert rem["n_geometries"] == 1 and rem["n_failed"] == 0 and len(rem["rows"]) == 1
        by_lc = rem["rows"][0]["by_lc"]
        assert [b["lc"] for b in by_lc] == list(amp["remesh_lc"])
        assert all(math.isfinite(b["c_battery"]) and b["n_nodes"] > 0 for b in by_lc)
    import hashlib

    sha = hashlib.sha256(rp.read_bytes()).hexdigest()
    real = A.REMESH_N
    try:
        A.REMESH_N = 1
        chk = A.check_amp(amp, sha)
    finally:
        A.REMESH_N = real
    assert chk == {"reproduction_max_rel_dev": 0.0, "remesh_not_evaluated": None}
    mech = A.mechanism(amp)
    assert set(mech["remesh"]) >= {"c_battery_median_by_lc", "max_abs_F_ratio_median_by_lc"}


def test_the_reproduction_skips_instances_diverged_in_both():
    """An instance non-finite in the report and in the record (a diverged
    state) is not compared and is counted; one finite in one of the two only
    deviates by inf."""
    spec = importlib.util.spec_from_file_location("posthoc_amplitude",
                                                  ROOT / "scripts" / "posthoc_amplitude.py")
    pa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pa)
    nan = float("nan")
    cell = {"per_seed_eval": [{"per_instance": {"disp_rel_l2": [0.5, nan, 0.2],
                                                "energy_gap_rel": [0.1, nan, nan]}}]}
    rows = [{"disp": 0.5, "egap": 0.1}, {"disp": nan, "egap": nan},
            {"disp": 0.2, "egap": 0.3}]
    assert pa.reproduction(rows, cell, 0) == {
        "checked": True, "disp_max_rel_dev": 0.0, "disp_not_finite_in_both": 1,
        "egap_max_rel_dev": math.inf, "egap_not_finite_in_both": 1}
    assert pa.reproduction(rows, None, 0) == {"checked": False}
