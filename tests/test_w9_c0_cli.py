"""wp9 Stage 0a: the session-1 scripts end to end on a miniature E1-like run
(synthetic corpus, AR only, two seeds, CPU): every C0 subcommand runs on the
run's verified states and corpus, reproduces the run's own arrays, refuses a
tampered evaluation family, and the decision script reads the outputs."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

from fejepa.data.archive import instance_files, load_instance, save_instance, write_manifest
from fejepa.experiments.runner import run_config
from fejepa.fe.synthetic import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}


def _family(out: Path, archs_by_name: dict, extra: dict) -> Path:
    """A family directory in the OOD-2D manifest format (per-file SHA-256)."""
    out.mkdir(parents=True)
    recs = []
    for name, (a, fields) in archs_by_name.items():
        save_instance(a, out / name)
        recs.append({"file": name, "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
                     "n_nodes": a.n_nodes, "labelled": True, **fields})
    write_manifest(out, recs, extra)
    return out


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True,
                       cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w9c0")
    d = generate_synthetic_dataset(tmp / "corpus", n=12, seed=3, labelled="all")
    cfg = {"data": {"dir": str(d), "n": 12, "seed": 3, "backend": "synthetic",
                    "labelled_policy": "asis"},
           "split": {"n_val": 4, "seed": 1}, "model": MODEL,
           "sup": {"epochs": 1, "lr": 1e-3}, "pretrain": {"epochs": 1, "lr": 1e-3},
           "experiments": {"e8": {"enabled": True, "budgets": [2], "pool_sizes": [4],
                                  "seeds": 2, "ar_epochs": 2, "sup_epochs": 1,
                                  "include_mgn": False, "ar_only": True}},
           "device": "cpu", "workers": 1, "tf32": False, "prereg_guard": False,
           "out": str(tmp / "run" / "report.json")}
    (tmp / "cfg.json").write_text(json.dumps(cfg))
    run_config(str(tmp / "cfg.json"))
    # evaluation families: "F5" = finer structured meshes; "R" = one plate at three resolutions
    f5 = generate_synthetic_dataset(tmp / "f5src", n=3, seed=9, labelled="all", nx=14, ny=10)
    fam = {}
    for i, f in enumerate(instance_files(f5)):
        fam[f"instance_{i:05d}.npz"] = (load_instance(f), {"target_h": 0.05})
    _family(tmp / "F5", fam, {"family": "F5", "seed": 9})
    rem = {}
    for nx, ny, h in ((6, 4, 0.12), (10, 7, 0.08), (14, 10, 0.05)):
        src = generate_synthetic_dataset(tmp / f"r{nx}", n=1, seed=21, labelled="all", nx=nx, ny=ny)
        a = load_instance(instance_files(src)[0])
        a.meta["extra"]["target_h"] = h
        rem[f"g000_h{h:.4f}.npz"] = (a, {"geometry": 0, "target_h": h})
    _family(tmp / "R", rem, {"family": "R", "seed": 21})
    return tmp


def test_c0_subcommands_and_decisions(mini):
    rep, st = mini / "run" / "report.json", mini / "run" / "e8_states"
    out = mini / "s1"
    common = ["--report", rep, "--device", "cpu"]
    _run(ROOT / "scripts" / "w9_c0.py", "val", *common, "--states-dir", st, "--ks", 2, 10,
         "--out", out / "c0_val.json")
    _run(ROOT / "scripts" / "w9_c0.py", "trainval", *common, "--states-dir", st,
         "--out", out / "c0_trainval.json")
    _run(ROOT / "scripts" / "w9_c0.py", "amp2d", *common, "--states-dir", st,
         "--family-dir", mini / "F5", "--remesh-dir", mini / "R", "--out", out / "c0_amp2d.json")
    _run(ROOT / "scripts" / "w9_c0.py", "memory", *common, "--checkpoints", 2, 4, 8,
         "--out", out / "c0_memory.json")
    val = json.loads((out / "c0_val.json").read_text())
    assert val["n_instances"] == 4 and len(val["rows"]) == 4
    assert val["summary"]["ensemble"]["identity_residual_max"] < 1e-10
    rep_dev = val["summary"]["reproduction"]
    assert max(rep_dev["disp"].values()) == 0.0 and max(rep_dev["egap"].values()) < 1e-9
    tv = json.loads((out / "c0_trainval.json").read_text())
    assert tv["train"]["n_instances"] == 4 and tv["val"]["n_instances"] == 4
    amp = json.loads((out / "c0_amp2d.json").read_text())
    assert set(amp["sets"]) == {"inband", "F5"} and amp["R"]["family"] == "R"
    by_h = amp["remesh"]["s0"]["by_h"]
    assert list(by_h) == ["0.12", "0.08", "0.05"] and by_h["0.12"]["u_norm_K_ratio_median"] == 1.0
    assert by_h["0.12"]["ustar_norm_K_ratio_median"] == 1.0
    assert all(v["ustar_norm_K_ratio_median"] > 0 for v in by_h.values())
    mem = json.loads((out / "c0_memory.json").read_text())
    assert [m["n"] for m in mem["marks"]] == [2, 4, 8] and mem["limits"]["usable"] > 0
    # the decisions, with a timing file in the profiler's format
    (out / "profile_2d_w9.json").write_text(json.dumps({"variants": {
        "worker3": {"ar": {"ms_per_step_per_unit": 50.0, "valid": True}},
        "worker3_no_ckpt": {"ar": {"ms_per_step_per_unit": 25.0, "valid": True}}}}))
    _run(ROOT / "scripts" / "w9_session1_decisions.py", "--dir", out, "--out", out / "dec.json")
    dec = json.loads((out / "dec.json").read_text())
    r = dec["decisions"]
    assert all("error" not in v for v in r.values())
    assert r["checkpointing_off"]["speedup"] == 2.0 and r["checkpointing_off"]["value"] in (True, False)
    assert r["pool_max"]["gpu_total_bytes"] is None              # a CPU audit: host only
    assert r["pool_max"]["usable_host_source"] in ("cgroup_limit", "meminfo_total",
                                                   "meminfo_available")
    assert r["S_enters"]["F5"] == {"family": "F5", "seed": 9, "n_instances": 3}
    assert r["C2_score_usable"]["value"] in (True, False)
    assert len(dec["inputs_sha256"]) == 5 and "c0_val.json" in dec["inputs_sha256"]
    assert len(dec["rules_sha256"]) == 64
    # an unreadable input: its rules undecided, the others decided, a non-zero exit
    (out / "c0_trainval.json").rename(out / "c0_trainval.json.away")
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "w9_session1_decisions.py"),
                          "--dir", str(out), "--out", str(out / "dec2.json")],
                         capture_output=True, text=True, cwd=str(ROOT))
    (out / "c0_trainval.json.away").rename(out / "c0_trainval.json")
    assert res.returncode != 0 and "C0_4_prediction" in res.stderr
    r2 = json.loads((out / "dec2.json").read_text())["decisions"]
    assert r2["C0_4_prediction"]["value"] is None and r2["S_enters"] == r["S_enters"]


def test_step_overhead_runs_the_ar_step_in_both_modes(mini, monkeypatch):
    """The GPU-only step measurement, on CPU with the allocator queries stubbed
    (the box runs it on the GPU): both modes run the real AR loss, backward and
    optimiser step, and the switch is restored."""
    import importlib.util

    import torch

    from fejepa.analysis.common import build_model_from_config, instance_files
    from fejepa.anchor.energy import AnchorCache

    spec = importlib.util.spec_from_file_location("w9c0", ROOT / "scripts" / "w9_c0.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    calls = []
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, _n=name, **k: calls.append(_n))
    for name in ("memory_reserved", "memory_allocated", "max_memory_reserved",
                 "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: 0)
    report = json.loads((mini / "run" / "report.json").read_text())
    cfg = report["config"]
    model = build_model_from_config(cfg["model"], seed=0, mode="train", device="cpu")
    files = instance_files(Path(cfg["data"]["dir"]), split=cfg["split"], subset="pool")[:3]
    held = [(load_instance(f), None) for f in files]
    held = [(a, model.prepare_instance(a, "cpu")) for a, _ in held]
    seen = []
    orig = type(model.encoder).forward

    def spy(self, *a, **k):
        seen.append(self.use_checkpoint)
        return orig(self, *a, **k)

    monkeypatch.setattr(type(model.encoder), "forward", spy)
    out = mod._step_overhead(model, held, AnchorCache(device="cpu"))
    assert out["n_nodes"] == max(a.n_nodes for a, _ in held)
    assert set(out) == {"n_nodes", "ckpt_on", "ckpt_off"}
    assert seen == [True, True, False, False] and model.encoder.use_checkpoint is True
    assert calls.count("reset_peak_memory_stats") == 2


def test_amp2d_refuses_swapped_families(mini, tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "w9_c0.py"), "amp2d",
                        "--report", str(mini / "run" / "report.json"), "--device", "cpu",
                        "--states-dir", str(mini / "run" / "e8_states"),
                        "--family-dir", str(mini / "R"), "--remesh-dir", str(mini / "F5"),
                        "--out", str(tmp_path / "o.json")], capture_output=True, text=True,
                       cwd=str(ROOT))
    assert r.returncode != 0 and "holds family 'R', not 'F5'" in (r.stdout + r.stderr)


def test_amp2d_refuses_a_tampered_family(mini, tmp_path):
    bad = tmp_path / "F5bad"
    shutil.copytree(mini / "F5", bad)
    (bad / "instance_00001.npz").write_bytes(b"x")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "w9_c0.py"), "amp2d",
                        "--report", str(mini / "run" / "report.json"), "--device", "cpu",
                        "--states-dir", str(mini / "run" / "e8_states"),
                        "--family-dir", str(bad), "--remesh-dir", str(mini / "R"),
                        "--out", str(tmp_path / "o.json")], capture_output=True, text=True,
                       cwd=str(ROOT))
    assert r.returncode != 0 and "differ from their manifest" in (r.stdout + r.stderr)


def test_c0_refuses_another_state(mini, tmp_path):
    st = tmp_path / "states"
    shutil.copytree(mini / "run" / "e8_states", st)
    f = sorted(st.glob("*.pt"))[0]
    f.write_bytes(f.read_bytes() + b"\0")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "w9_c0.py"), "val",
                        "--report", str(mini / "run" / "report.json"), "--device", "cpu",
                        "--states-dir", str(st), "--out", str(tmp_path / "o.json")],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode != 0 and "is not the state the report" in (r.stdout + r.stderr)


def test_make_ood2d_refuses_a_different_family_and_continues(tmp_path):
    """An existing directory whose manifest is not the family asked for is
    refused (not regenerated, not accepted); the other families still run and
    the script exits non-zero."""
    out = tmp_path / "ood"
    _family(out / "F5", {}, {"family": "F5", "seed": 1})        # a leftover smoke family
    _family(out / "F4", {}, {"family": "F4", "seed": 91004})    # right name and seed, n = 0
    rec = tmp_path / "rec.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "w9_make_ood2d.py"), "--out",
                        str(out), "--families", "F5", "F4", "--n", "0", "--record", str(rec)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode != 0 and "not complete: ['F5']" in r.stderr
    got = json.loads(rec.read_text())["families"]
    assert got["F5"]["status"] == "refused" and "'seed': (1, 91005)" in got["F5"]["error"]
    assert got["F4"]["status"] == "verified existing"
