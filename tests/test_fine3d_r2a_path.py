"""fine3d (PREREG_FINE3D Sec. 2 and 5, R2a): the exact R2a path on a CPU
miniature -- the generator's S configuration (the token bottleneck with S,
E8 AR on the fresh seeds, P3 zero-shot), shrunk to gmsh3d corpora of a dozen
instances and run through run-config as the session runs it, then the
post-hoc amplitude pass on its report. The run trains and evaluates the
fresh seeds; P3 evaluates exactly the states E8 trained (its in-band pass
equals E8's AR cell per instance, bit for bit on CPU); the amplitude pass
reproduces the report's rows exactly. A P3 or post-hoc model built without
S would not."""

import copy
import importlib.util
import json
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
    mini.update(data=dict(base["data"], dir=str(tmp / "data3d"), n=12, lc_range=[0.30, 0.36]),
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
    run_config(p)
    return tmp, cfg, json.loads((tmp / "s" / "report.json").read_text())


def test_the_s_run_trains_and_evaluates_the_fresh_seeds(run_s):
    tmp, cfg, rep = run_s
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

    _, _, rep = run_s
    p3, e8 = A.p3_cell(rep, "inband"), A.e8_cell(rep)
    assert len(p3["per_seed_eval"]) == len(e8["per_seed_eval"]) == 2
    for a, b in zip(p3["per_seed_eval"], e8["per_seed_eval"], strict=True):
        for k in FIELD_KEYS:
            assert a["per_instance"][k] == b["per_instance"][k], k
    assert A.p3_e8_deviation(rep) == 0.0


def test_the_amplitude_pass_reproduces_the_s_report(run_s, tmp_path):
    """scripts/posthoc_amplitude.py on the S report (without the remesh pass,
    which the session adds): the states of seeds 3 and 4, each reproducing its
    own rows of the report exactly."""
    tmp, _, rep = run_s
    out = tmp_path / "amp_s.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_amplitude.py"),
                        "--report", str(tmp / "s" / "report.json"),
                        "--states-dir", str(tmp / "s" / "e8_states"), "--n-inband", "4",
                        "--n-fine", "4", "--device", "cpu", "--out", str(out)],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-2000:]
    amp = json.loads(out.read_text())
    assert sorted(amp["seeds"]) == ["s3", "s4"] and amp["model_kind"] == "bottleneck"
    for s in amp["seeds"].values():
        for name in ("inband", "fine"):
            rpr = s[name]["reproduction"]
            assert rpr["checked"] and rpr["disp_max_rel_dev"] == rpr["egap_max_rel_dev"] == 0.0
