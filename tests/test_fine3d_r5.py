"""fine3d r5 (16 October 2026): the findings of the review of r4, fixed.

- The post-hoc remesh pass records a geometry that cannot be meshed and goes
  on, but stops on an error while the states are evaluated (on the box a GPU
  error, which would otherwise fail every later geometry and leave a record
  the session took for complete).
- An epoch checkpoint whose file cannot be read is moved aside and nothing is
  restored (a true fresh start); one that reads but cannot be restored (another
  model's state, or an error on the device) stops the run and stays.
- A D9 state (run-config --reuse-states) that cannot be read is moved aside and
  retrained; one that reads but does not load, or an error on the device, stops
  the run and stays (it was removed and retrained).
- R3's checkpoint record names the seed, is written atomically and is read
  whatever its machines field holds.
- The report encoder writes numpy booleans.
The session script's changes are tested in tests/test_fine3d_runbook.py and the
adjudication's in tests/test_fine3d_adjudication.py."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fejepa.experiments.parallel import _build_model, pretrain_unit  # noqa: E402
from fejepa.experiments.protocol import load_split  # noqa: E402
from fejepa.experiments.runner import _label_files  # noqa: E402
from fejepa.fe.solve import SolveLedger  # noqa: E402
from fejepa.fe.synthetic import generate_synthetic_dataset  # noqa: E402
from fejepa.fe.tet3d import tet_instance  # noqa: E402
from fejepa.train import checkpoint as C  # noqa: E402
from fejepa.train.schedule import make_scheduler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _failing_gpu(monkeypatch):
    """Moving a module to a CUDA device fails, as on a failing GPU -- on any
    machine: the box's suite runs these tests with a working GPU."""
    real = torch.nn.Module.to

    def to(self, *a, **k):
        dev = a[0] if a else k.get("device")
        if isinstance(dev, (str, torch.device)) and torch.device(dev).type == "cuda":
            raise RuntimeError("CUDA error: an illegal memory access was encountered "
                               "(simulated)")
        return real(self, *a, **k)
    monkeypatch.setattr(torch.nn.Module, "to", to)


# ------------------------------------------------------- the remesh pass --

def _remesh_setup(monkeypatch, fail_mesh=None, fail_forward_after=None):
    pa = _module("posthoc_amplitude")
    arch = tet_instance(np.random.default_rng(1), nx=4, ny=3, nz=3, labelled=False)
    arch.meta.setdefault("extra", {})["n_holes"] = 0

    def mesh(g, lc, cache):
        if g == fail_mesh:
            raise RuntimeError("gmsh: simulated meshing failure")
        return arch, False
    monkeypatch.setattr(pa, "remesh_instance", mesh)
    calls, real = {"n": 0}, pa._forward

    def forward(model, pack):
        calls["n"] += 1
        if fail_forward_after is not None and calls["n"] > fail_forward_after:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real(model, pack)
    monkeypatch.setattr(pa, "_forward", forward)
    m = _build_model({"kind": "bottleneck", "seed": 0, "model": {
        "dim": 16, "depth": 1, "heads": 2, "n_tokens": 8, "decode_k": 2,
        "features": {"load_summary": True, "geometry": True, "spatial_dim": 3}}}).eval()
    return pa, {3: m}


def test_remesh_pass_records_a_geometry_that_cannot_be_meshed(monkeypatch):
    pa, models = _remesh_setup(monkeypatch, fail_mesh=1)
    rows = [r[3] for r in pa.remesh_rows(models, 3, "cpu")]
    assert [("error" in r) for r in rows] == [False, True, False]
    assert "simulated meshing failure" in rows[1]["error"]
    summ = pa.remesh_summary(rows)
    assert summ["n_geometries"] == 2 and summ["n_failed"] == 1


def test_remesh_pass_stops_when_memory_runs_out(monkeypatch):
    """fine3d r5 (review F, round 2, D2): running out of memory while meshing
    is not the geometry's failure (a second pass may mesh it): the pass stops."""
    pa, models = _remesh_setup(monkeypatch)

    def mesh(g, lc, cache):
        raise MemoryError("simulated")
    monkeypatch.setattr(pa, "remesh_instance", mesh)
    with pytest.raises(MemoryError):
        list(pa.remesh_rows(models, 2, "cpu"))


def test_a_remesh_cache_file_that_cannot_be_written_is_skipped(monkeypatch, tmp_path, capsys):
    """fine3d r5 (review F, round 2, D2): a full disk while caching a mesh is
    not the geometry's failure: the mesh is used, not cached, and a line says
    so."""
    import fejepa.data.archive as archive
    import fejepa.fe.gmsh3d as gmsh3d

    pa = _module("posthoc_amplitude")
    arch = tet_instance(np.random.default_rng(1), nx=4, ny=3, nz=3, labelled=False)
    monkeypatch.setattr(gmsh3d, "gmsh3d_instance", lambda rng, lc, labelled: arch)

    def full(a, path):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(archive, "save_instance", full)
    a, cached = pa.remesh_instance(1, pa.REMESH_LC[0], tmp_path)
    assert a is arch and cached is False and not list(tmp_path.iterdir())
    out = capsys.readouterr().out
    assert '"not written": "OSError: [Errno 28] No space left on device"' in out


def test_remesh_pass_stops_on_an_error_while_evaluating(monkeypatch):
    """Geometry 0 (every mesh size) evaluates; the next forward raises: the
    pass stops with the error (it was recorded as a failed geometry, and
    every later one failed the same way)."""
    n_lc = len(_module("posthoc_amplitude").REMESH_LC)
    pa, models = _remesh_setup(monkeypatch, fail_forward_after=n_lc)
    rows = []
    with pytest.raises(RuntimeError, match="illegal memory access"):
        for r in pa.remesh_rows(models, 3, "cpu"):
            rows.append(r[3])
    assert len(rows) == 1 and len(rows[0]["by_lc"]) == n_lc


# -------------------------------------------------------- epoch checkpoints --

def _loop(seed=0, steps=3, out=2):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, out)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = make_scheduler(opt, 20)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        rng.permutation(5)
        loss = model(torch.randn(3, 4)).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model, opt, sched, rng


def test_a_checkpoint_that_cannot_be_read_is_moved_aside_and_nothing_restored(tmp_path, capsys):
    p = tmp_path / "u.ckpt"
    p.write_bytes(b"not a checkpoint")
    m, o, s, r = _loop(seed=3, steps=0)
    w, rs = m.weight.detach().clone(), r.bit_generator.state
    assert C.load_epoch_checkpoint(p, model=m, opt=o, sched=s, rng=r, device="cpu") is None
    assert not p.exists()
    aside = list(tmp_path.glob("u.ckpt.unusable.*"))
    assert len(aside) == 1 and aside[0].read_bytes() == b"not a checkpoint"
    assert torch.equal(m.weight, w) and r.bit_generator.state == rs and not o.state
    assert "moved to u.ckpt.unusable." in capsys.readouterr().out


@pytest.mark.parametrize("what", ["another model", "the device"])
def test_a_checkpoint_that_reads_but_cannot_be_restored_stops_and_stays(tmp_path, what,
                                                                        monkeypatch):
    model, opt, sched, rng = _loop(steps=3)
    p = tmp_path / "u.ckpt"
    C.save_epoch_checkpoint(p, epochs_done=1, step=3, model=model, opt=opt, sched=sched, rng=rng)
    before = _sha(p)
    if what == "another model":
        m, o, s, r = _loop(seed=1, steps=0, out=3)
        with pytest.raises(RuntimeError, match="size mismatch"):
            C.load_epoch_checkpoint(p, model=m, opt=o, sched=s, rng=r, device="cpu")
    else:
        m, o, s, r = _loop(seed=1, steps=0)
        _failing_gpu(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated"):
            C.load_epoch_checkpoint(p, model=m, opt=o, sched=s, rng=r, device="cuda")
    assert p.exists() and _sha(p) == before
    assert not list(tmp_path.glob("u.ckpt.unusable.*"))


def test_a_resumed_checkpoint_restores_everything(tmp_path):
    model, opt, sched, rng = _loop(steps=3)
    p = tmp_path / "u.ckpt"
    C.save_epoch_checkpoint(p, epochs_done=2, step=3, model=model, opt=opt, sched=sched, rng=rng,
                            extra={"k": 1})
    m, o, s, r = _loop(seed=1, steps=0)
    assert C.load_epoch_checkpoint(p, model=m, opt=o, sched=s, rng=r, device="cpu") == \
        (2, 3, {"k": 1})
    assert torch.equal(m.weight, model.weight) and torch.equal(m.bias, model.bias)
    assert s.state_dict() == sched.state_dict() and r.bit_generator.state == rng.bit_generator.state
    assert o.state_dict()["state"].keys() == opt.state_dict()["state"].keys()


# --------------------------------------------------------- D9 reused states --

MODEL = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}


@pytest.fixture(scope="module")
def d9(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("d9")
    d = generate_synthetic_dataset(tmp / "d", n=6, seed=11)
    sp = load_split(d, n_val=2, seed=1)
    led = SolveLedger()
    _label_files(sp.val_files, led, "v")
    payload = {"kind": "fejepa", "model": MODEL, "seed": 0, "tf32": False,
               "files": [str(f) for f in sp.pool_files[:2]],
               "pre": {"epochs": 1, "lr": 1e-3, "device": "cpu", "log_every": -1},
               "eval_val_files": [str(f) for f in sp.val_files], "quiet": True}
    return tmp, payload


def test_a_d9_state_that_cannot_be_read_is_moved_aside_and_retrained(d9, tmp_path, capsys):
    _, payload = d9
    sp = tmp_path / "st" / "ar.pt"
    sp.parent.mkdir()
    sp.write_bytes(b"truncated")
    out = pretrain_unit(dict(payload, state_path=str(sp), reuse_existing=True))
    assert out["reused_state"] is False and sp.exists() and sp.read_bytes() != b"truncated"
    aside = list(sp.parent.glob("ar.pt.unusable.*"))
    assert len(aside) == 1 and aside[0].read_bytes() == b"truncated"
    assert "unusable state" in capsys.readouterr().out


@pytest.mark.parametrize("what", ["another model", "the device"])
def test_a_d9_state_that_reads_but_does_not_load_stops_and_stays(d9, tmp_path, what,
                                                                 monkeypatch):
    _, payload = d9
    sp = tmp_path / "st" / "ar.pt"
    first = pretrain_unit(dict(payload, state_path=str(sp)))         # a usable state
    assert first["reused_state"] is False and sp.exists()
    if what == "another model":
        torch.save({"encoder.weight": torch.zeros(2, 2)}, sp)
        before = _sha(sp)
        with pytest.raises(RuntimeError):
            pretrain_unit(dict(payload, state_path=str(sp), reuse_existing=True))
    else:
        before = _sha(sp)
        _failing_gpu(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated"):
            pretrain_unit(dict(payload, state_path=str(sp), reuse_existing=True,
                               pre=dict(payload["pre"], device="cuda")))
    assert sp.exists() and _sha(sp) == before
    assert not list(sp.parent.glob("ar.pt.unusable.*"))


# ---------------------------------------------------- R3's checkpoint record --

R3 = _module("r3_finetune")
MINE = {"git": "prereg-fine3d", "config_sha256": "c" * 64, "seed": 1}
HERE = {"host": "box", "gpu_driver": "580.95.05"}


def _ckpt(tmp_path, side):
    ck = tmp_path / "eft_b16_s1.ckpt"
    ck.write_bytes(b"checkpoint")
    if side is not None:
        R3._sidecar(ck).write_text(side if isinstance(side, str) else json.dumps(side))
    return ck


def test_r3_checkpoint_record_names_the_seed(tmp_path):
    ck = _ckpt(tmp_path, dict(MINE, seed=0, machines=[HERE]))         # seed 0's, as seed 1's
    assert R3.own_checkpoint(ck, MINE, HERE) == [HERE]
    assert not ck.exists() and len(list(tmp_path.glob("eft_b16_s1.ckpt.unusable.*"))) == 1
    assert json.loads(R3._sidecar(ck).read_text())["seed"] == 1


@pytest.mark.parametrize("machines", [5, "box", None, {"host": "box"}])
def test_r3_checkpoint_record_is_read_whatever_its_machines_hold(tmp_path, machines):
    ck = _ckpt(tmp_path, dict(MINE, machines=machines))
    assert R3.own_checkpoint(ck, MINE, HERE) == [HERE]                # resumed, no machine kept
    assert ck.exists()


@pytest.mark.parametrize("side", [None, "", '{"git": ', "[1, 2]", '"text"'])
def test_r3_checkpoint_without_a_readable_record_is_moved_aside(tmp_path, side):
    ck = _ckpt(tmp_path, side)
    assert R3.own_checkpoint(ck, MINE, HERE) == [HERE]
    assert not ck.exists() and len(list(tmp_path.glob("eft_b16_s1.ckpt.unusable.*"))) == 1


def test_r3_checkpoint_record_is_written_atomically(tmp_path, monkeypatch):
    import fejepa.analysis.common as common

    seen, real = [], common.write_json
    monkeypatch.setattr(common, "write_json", lambda p, o: (seen.append(Path(p)), real(p, o)))
    ck = _ckpt(tmp_path, dict(MINE, machines=[{"host": "first", "gpu_driver": "1"}]))
    assert R3.own_checkpoint(ck, MINE, HERE) == [{"host": "first", "gpu_driver": "1"}, HERE]
    assert seen == [R3._sidecar(ck)] and not list(tmp_path.glob("*.tmp"))
    assert json.loads(R3._sidecar(ck).read_text()) == dict(
        MINE, machines=[{"host": "first", "gpu_driver": "1"}, HERE])


# ---------------------------------------------------------- report encoder --

def test_the_report_encoder_writes_numpy_booleans_and_numbers():
    from fejepa.report import _Encoder

    out = json.loads(json.dumps({"b": np.bool_(True), "i": np.int64(3), "f": np.float32(0.5),
                                 "a": np.arange(2), "p": Path("x")}, cls=_Encoder))
    assert out == {"b": True, "i": 3, "f": 0.5, "a": [0, 1], "p": "x"}
