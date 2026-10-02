"""wp9 Stage 0b: a worker process killed from outside (the system's
out-of-memory killer) -- busy with a unit or idle after one -- makes map_units
raise at once instead of waiting forever; a unit that raises ends the map
without waiting for the others; normal pools are unaffected (results in
payload order); when the parent process is killed, its workers exit (no
orphan unit keeps training)."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _pids(d: Path) -> list:
    """The pids the units wrote (a unit terminated while writing leaves an empty file)."""
    return [int(t) for t in (p.read_text().strip() for p in d.glob("w*.pid")) if t]


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state not in ("Z", "X")                       # a zombie has exited


def _module(tmp_path, monkeypatch):
    (tmp_path / "w9_killer_mod.py").write_text(
        "import os, signal, time\n"
        "def unit(p):\n"
        "    open(os.path.join(p.get('dir', '.'), f\"w{p['i']}.pid\"), 'w').write(str(os.getpid()))\n"
        "    if p.get('kill'):\n"
        "        os.kill(os.getpid(), signal.SIGKILL)\n"
        "    if p.get('raise'):\n"
        "        raise ValueError('unit failed')\n"
        "    while p.get('wait_for') and not os.path.exists(os.path.join(p['dir'], p['wait_for'])):\n"
        "        time.sleep(0.01)\n"
        "    time.sleep(p.get('secs', 0.2))\n"
        "    if p.get('done_marker'):\n"
        "        open(os.path.join(p['dir'], f\"r{p['i']}.done\"), 'w').close()\n"
        "    return p['i']\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import w9_killer_mod

    return w9_killer_mod


def test_map_units_raises_when_a_busy_worker_is_killed(tmp_path, monkeypatch):
    mod = _module(tmp_path, monkeypatch)
    from fejepa.experiments import parallel

    monkeypatch.setattr(parallel, "WORKER_POLL_S", 2.0)

    d = str(tmp_path)
    assert parallel.map_units(mod.unit, [{"i": i, "dir": d} for i in range(4)], 2, "ok") == \
        [0, 1, 2, 3]
    t0 = time.time()
    with pytest.raises(RuntimeError, match="worker process died") as err:
        parallel.map_units(mod.unit, [{"i": i, "dir": d, "kill": i == 1, "secs": 30}
                                      for i in range(3)], 3, "kill test")
    assert time.time() - t0 < 20                          # the others are not waited for
    codes = str(err.value).split("worker exit codes ")[1].split("]")[0] + "]"
    assert sorted(eval(codes)) == [-15, -15, -9]          # the killed one; the two stopped
    time.sleep(1)
    assert not any(_alive(p) for p in _pids(tmp_path))


def test_map_units_raises_when_an_idle_worker_is_killed(tmp_path, monkeypatch):
    """multiprocessing.Pool could block for ever in its shutdown when a worker
    died while waiting for work (holding the task queue's lock)."""
    mod = _module(tmp_path, monkeypatch)
    from fejepa.experiments import parallel

    monkeypatch.setattr(parallel, "WORKER_POLL_S", 2.0)
    d = str(tmp_path)

    def killer():                       # unit 0 returns only once unit 1 runs on the other worker
        while not (tmp_path / "r0.done").exists():
            time.sleep(0.05)
        time.sleep(0.5)                                   # its result is sent: the worker idles
        os.kill(int((tmp_path / "w0.pid").read_text()), signal.SIGKILL)

    import threading

    threading.Thread(target=killer, daemon=True).start()
    t0 = time.time()
    with pytest.raises(RuntimeError, match="worker process died") as err:
        parallel.map_units(mod.unit, [{"i": 0, "dir": d, "secs": 0.0, "wait_for": "w1.pid",
                                       "done_marker": True},
                                      {"i": 1, "dir": d, "secs": 30}], 2, "idle kill")
    assert time.time() - t0 < 30
    assert "-9" in str(err.value)
    assert _pids(tmp_path) and len(set(_pids(tmp_path))) == 2       # two distinct workers


def test_a_failing_unit_ends_the_map_without_waiting(tmp_path, monkeypatch):
    mod = _module(tmp_path, monkeypatch)
    from fejepa.experiments import parallel

    t0 = time.time()
    with pytest.raises(ValueError, match="unit failed"):
        parallel.map_units(mod.unit, [{"i": 0, "dir": str(tmp_path), "raise": True},
                                      {"i": 1, "dir": str(tmp_path), "secs": 60}], 2, "raise")
    assert time.time() - t0 < 20
    time.sleep(1)
    assert not any(_alive(p) for p in _pids(tmp_path))


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_workers_die_with_a_killed_parent(tmp_path):
    (tmp_path / "w9_sleeper_mod.py").write_text(
        "import os, time\n"
        "def unit(p):\n"
        "    open(os.path.join(p['dir'], f\"w{p['i']}.pid\"), 'w').write(str(os.getpid()))\n"
        "    time.sleep(120)\n"
        "    return p['i']\n")
    (tmp_path / "parent.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import w9_sleeper_mod\n"
        "from fejepa.experiments import parallel\n"
        "if __name__ == '__main__':\n"
        f"    parallel.map_units(w9_sleeper_mod.unit, [{{'i': i, 'dir': {str(tmp_path)!r}}} "
        "for i in range(2)], 2, 'orphan test')\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    parent = subprocess.Popen([sys.executable, str(tmp_path / "parent.py")], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while len(_pids(tmp_path)) < 2:
            assert time.time() - t0 < 120 and parent.poll() is None, "workers did not start"
            time.sleep(0.2)
        pids = _pids(tmp_path)
        assert all(_alive(p) for p in pids)
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=30)
        t0 = time.time()
        while any(_alive(p) for p in pids) and time.time() - t0 < 30:
            time.sleep(0.2)
        assert not any(_alive(p) for p in pids), "a worker outlived its killed parent"
    finally:
        if parent.poll() is None:
            parent.kill()
        for p in tmp_path.glob("w*.pid"):
            try:
                os.kill(int(p.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass


def test_e8_ar_units_in_parallel_match_serial(tmp_path, monkeypatch):
    """The session-2 path (AR-only E8 with holdouts and amplitude readings)
    through the process pool gives what the inline path gives: the same
    states, byte for byte, and the same evaluations (single-threaded torch;
    the contract of tests/test_experiments_smoke.py for E1)."""
    import json

    import torch

    from fejepa.experiments.e8_regimes import run_e8
    from fejepa.experiments.protocol import load_split
    from fejepa.experiments.runner import _label_files
    from fejepa.fe.solve import SolveLedger
    from fejepa.fe.synthetic import generate_synthetic_dataset

    monkeypatch.setenv("FEJEPA_WORKER_THREADS", "1")
    torch.set_num_threads(1)
    sp = load_split(generate_synthetic_dataset(tmp_path / "ds", n=10, seed=0), n_val=3, seed=1)
    _label_files(sp.val_files, SolveLedger(), "lv")
    model = {"dim": 16, "depth": 1, "heads": 2, "features": {"load_summary": True, "geometry": True}}
    cfg = {"pool_sizes": [4], "seeds": 2, "seed_offset": 3, "ar_epochs": 2, "ar_only": True,
           "device": "cpu", "amplitude": True,
           "holdouts": {"F5": [str(f) for f in sp.val_files]}}
    out = {}
    for w in (1, 2):
        res = run_e8(model, sp.pool_files, sp.val_files,
                     dict(cfg, workers=w, state_dir=str(tmp_path / f"st{w}")))
        out[w] = json.dumps(res["metrics"], sort_keys=True, default=str)
    assert out[1] == out[2]
    for s in (3, 4):
        assert (tmp_path / "st1" / f"ar_p4_s{s}.pt").read_bytes() == \
            (tmp_path / "st2" / f"ar_p4_s{s}.pt").read_bytes()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX pipes")
def test_a_worker_killed_while_sending_its_result_does_not_hang_the_exit(tmp_path):
    """A worker that dies half-way through sending its (large) result leaves
    the executor's manager thread blocked on the rest of the message; the map
    still raises (the poll sees the dead worker), and the process must then
    exit instead of joining that thread for ever at interpreter exit. The
    worker kills itself while it is sending its 200 MB result."""
    (tmp_path / "w9_sender_mod.py").write_text(
        "import os, signal, sys, threading, time\n"
        "def unit(p):\n"
        "    if p['i'] == 0:\n"
        "        blob = os.urandom(16) * (200_000_000 // 16)\n"
        "        main_id = threading.main_thread().ident\n"
        "        def killer():\n"
        "            while True:\n"
        "                f, names = sys._current_frames().get(main_id), []\n"
        "                while f is not None:\n"
        "                    names.append(f.f_code.co_name)\n"
        "                    f = f.f_back\n"
        "                if 'send_bytes' in names:\n"
        "                    time.sleep(0.005)\n"
        "                    os.kill(os.getpid(), signal.SIGKILL)\n"
        "                time.sleep(0.0005)\n"
        "        threading.Thread(target=killer, daemon=True).start()\n"
        "        return blob\n"
        "    time.sleep(60)\n"
        "    return p['i']\n")
    (tmp_path / "parent.py").write_text(
        "import sys, time\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import w9_sender_mod\n"
        "from fejepa.experiments import parallel\n"
        "parallel.WORKER_POLL_S = 1.0\n"
        "if __name__ == '__main__':\n"
        "    t0 = time.time()\n"
        "    try:\n"
        "        parallel.map_units(w9_sender_mod.unit, [{'i': 0}, {'i': 1}], 2, 'mid-send')\n"
        "        print('NO ERROR', flush=True)\n"
        "    except RuntimeError as e:\n"
        "        print('RAISED', round(time.time() - t0, 1), str(e)[:40], flush=True)\n")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    try:
        r = subprocess.run([sys.executable, str(tmp_path / "parent.py")], env=env,
                           capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("the parent did not exit after the map raised (hang at interpreter exit)")
    assert r.returncode == 0 and "RAISED" in r.stdout, r.stdout + r.stderr[-2000:]
