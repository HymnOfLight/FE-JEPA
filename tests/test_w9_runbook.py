"""wp9-pool Stage 0f: RUNBOOK_W9.md names only scripts that exist and options
they accept; Sec. 0b fetches the branch and the tag directly in one line, and
its bundle route fetches the same refs; Sec. 4 times the paper's CPU solvers in
the box's environment and with one thread, in 2D and in 3D (3D in the
environment again after the one-thread run), and the timing script runs with
exactly those options and records the threads; scripts/blas_threads.py reads
the loaded BLAS libraries' thread pools; the runbook states the suite's
current test count."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "RUNBOOK_W9.md"
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}
ONE = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
REFS = ("+refs/heads/wp9-pool:refs/remotes/origin/wp9-pool "
        "+refs/tags/prereg-w9:refs/tags/prereg-w9")
LATEX = "tests/test_w9_session2.py::test_paper_material_compiles"


def _text() -> str:
    return RUNBOOK.read_text(encoding="utf-8")


def _section(num: str) -> str:
    text = _text()
    start = text.index(f"\n## {num}. ")
    nxt = text.find("\n## ", start + 1)
    return text[start:] if nxt < 0 else text[start:nxt]


def _blocks(text: str) -> list:
    return re.findall(r"```bash\n(.*?)```", text, re.S)


def _commands(text: str) -> list:
    """(script, subcommand or None, options) of every `python scripts/... .py`
    command in the bash blocks of `text`, continuation lines joined."""
    out = []
    for block in _blocks(text):
        block = block.replace("\\\n", " ")
        for m in re.finditer(r"python (scripts/\S+\.py)([^\n]*)", block):
            rest = m.group(2).split()
            sub = rest[0] if rest and not rest[0].startswith(("-", "<", "$")) else None
            out.append((m.group(1), sub, re.findall(r"(--[a-z][a-z0-9-]*)", m.group(2))))
    return out


def test_the_runbook_names_existing_scripts_and_options():
    cmds = _commands(_text())
    assert {c[0] for c in cmds} >= {"scripts/w9_make_ood2d.py", "scripts/w9_c0.py",
                                     "scripts/posthoc_profile_2d.py",
                                     "scripts/w9_session1_decisions.py",
                                     "scripts/w9_session2_plan.py", "scripts/adjudicate_w9.py",
                                     "scripts/time_inference_vs_solve.py"}
    helps = {}
    for script, sub, opts in cmds:
        assert (ROOT / script).is_file(), script
        key = (script, sub)
        if key not in helps:
            args = [sys.executable, str(ROOT / script), *([sub] if sub else []), "--help"]
            res = subprocess.run(args, capture_output=True, text=True, cwd=ROOT, env=ENV)
            assert res.returncode == 0, (key, res.stderr[-2000:])
            helps[key] = res.stdout
        for o in opts:
            assert o in helps[key], (script, sub, o)


def test_the_fetch_is_direct_with_a_bundle_route():
    sec = _section("0")
    b = _blocks(sec)
    assert len(b) == 4                                     # tmux; fetch; bundle; checkout
    fetch = [x.split("#")[0].rstrip() for x in b[1].splitlines()]
    assert fetch == ["cd ~/autodl-tmp/FE-JEPA",
                     f"git fetch origin {REFS} && echo FETCH-OK || echo FETCH-FAILED",
                     "git rev-parse --short origin/wp9-pool",
                     "git rev-parse --short 'prereg-w9^{commit}'"]
    bundle = [x.split("#")[0].rstrip() for x in b[2].splitlines()]
    assert bundle == ["cd ~/autodl-tmp/FE-JEPA", "sha256sum ~/wp9-pool.bundle",
                      f"git fetch ~/wp9-pool.bundle {REFS} && echo FETCH-OK || echo FETCH-FAILED",
                      "git rev-parse --short origin/wp9-pool",
                      "git rev-parse --short 'prereg-w9^{commit}'"]
    assert "`git bundle create wp9-pool.bundle 414a372..wp9-pool prereg-w9`" in sec
    assert "git describe --tags --match prereg-w9        # must print exactly: prereg-w9" in b[3]
    assert "network_turbo" not in "".join(_blocks(_text()))


def test_section_4_times_the_solvers_with_the_environment_and_with_one_thread():
    sec = _section("4")
    block = _blocks(sec)[1].replace("\\\n", " ")
    runs = {m.group(1): " ".join(m.group(2).split())
            for m in re.finditer(r"^run (\w+)\.log (.*)$", block, re.M)}
    one = "env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
    script = "python scripts/time_inference_vs_solve.py "
    d2 = ("--report records/wp8/e1/e1_2d_base/report.json --states-dir runs/e1_2d_base/e8_states "
          "--n-val 8 --solvers direct cg")
    d3 = ("--report records/wp8/e2/baseline/report_phase2b.json --states-dir "
          "runs/phase2/e8_states --n-val 8 --n-fine 2 --solvers direct cg")
    assert runs == {"timing_2d_env": f"{script}{d2} --out $P/timing_2d_env.json",
                    "timing_2d_one": f"{one}{script}{d2} --out $P/timing_2d_one.json",
                    "timing_3d_env": f"{script}{d3} --out $P/timing_3d_env.json",
                    "timing_3d_one": f"{one}{script}{d3} --out $P/timing_3d_one.json",
                    "timing_3d_env2": f"{script}{d3} --out $P/timing_3d_env2.json",
                    "blas_env": "python scripts/blas_threads.py",
                    "blas_one": f"{one}python scripts/blas_threads.py"}
    assert list(runs) == ["timing_2d_env", "timing_2d_one", "timing_3d_env", "timing_3d_one",
                          "timing_3d_env2", "blas_env", "blas_one"]
    for rel in ("records/wp8/e1/e1_2d_base/report.json",
                "records/wp8/e2/baseline/report_phase2b.json"):
        assert (ROOT / rel).is_file(), rel
    assert "P=runs/w9/threads" in block and "OUT=~/wp9_threads_return" in block
    assert "`status.txt` must show\nthe seven steps with `exit=0`" in sec
    assert 'env | grep -E "^(OMP|MKL|OPENBLAS|GOTO)_" | sort' in _blocks(sec)[0]


def test_the_timing_script_runs_with_section_4s_options(tmp_path):
    """Section 4's 3D options on a small run shaped like Phase-2b's, once in the
    test's environment and once with one thread: both complete, only the direct
    solve and CG run, and the one-thread run records its threads."""
    pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location("w9_timing_tests",
                                                  ROOT / "tests" / "test_w9_timing.py")
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    rp, sdir = t._run_3d(tmp_path)
    opts = ["--n-val", "8", "--n-fine", "2", "--solvers", "direct", "cg"]
    assert " ".join(opts) in " ".join(_section("4").split())
    for name, extra in (("env", {}), ("one", ONE)):
        out = tmp_path / f"{name}.json"
        res = subprocess.run([sys.executable, str(t.SCRIPT), "--report", str(rp), "--states-dir",
                              str(sdir), *opts, "--device", "cpu", "--out", str(out)],
                             capture_output=True, text=True, env={**ENV, **extra})
        assert res.returncode == 0, res.stderr[-2000:]
        r = json.loads(out.read_text())
        assert r["settings"]["solvers"] == ["direct", "cg"]
        assert set(r["results"]) == {"val", "fine"}
        for s in ("val", "fine"):
            summ = r["results"][s]["summary"]
            assert summ["complete"] is True and summ["direct_n_solved"] == summ["n_instances"]
            assert summ["cg_n_solved"] == summ["n_instances"] and summ["cg_fallbacks"] == 0
            assert "cg_warm_n_solved" not in summ and "cg_match_n_solved" not in summ
        m = r["machine"]
        if name == "one":
            assert m["env_threads"] == ONE and m["torch_threads"] == 1
        else:
            assert m["env_threads"] == {k: None for k in ONE}


def test_the_runbook_states_the_suite_count():
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                          "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) tests? collected", res.stdout)
    assert m, res.stdout[-2000:]
    n = int(m.group(1))
    assert LATEX in res.stdout                     # the test that skips without pdflatex
    sec = _section("0")
    assert sec.count(f'"{n} passed"') == 1 and sec.count(f"`{n} passed`") == 1
    assert sec.count(f"`{n - 1} passed, 1 skipped`") == 1


def test_the_blas_readout_reads_the_loaded_pools():
    """scripts/blas_threads.py in the test's environment and with one thread: the
    environment recorded as set, and with one thread every pool it finds (the
    BLAS of numpy and of SciPy's SuperLU, here) and torch at one thread."""
    script = ROOT / "scripts" / "blas_threads.py"
    import numpy
    import scipy

    wheels = [d for d in (Path(m.__file__).parents[1] / f"{m.__name__}.libs" for m in (numpy, scipy))
              if any("openblas" in f.name for f in d.glob("*.so*"))]
    for name, extra in (("env", {}), ("one", ONE)):
        res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                             cwd=ROOT, env={**ENV, **extra})
        assert res.returncode == 0, res.stderr[-2000:]
        r = json.loads(res.stdout)
        assert r["env"] == {**{k: None for k in (*ONE, "GOTO_NUM_THREADS")}, **extra}
        assert r["affinity"] >= 1 and r["numpy"] and r["scipy"]
        assert all(p["num_threads"] >= 1 for p in r["pools"])
        if wheels:                                   # pip wheels: both OpenBLAS copies found
            assert {p["kind"] for p in r["pools"]} >= {"openblas"} and len(r["pools"]) >= len(wheels)
        if name == "one":
            assert all(p["num_threads"] == 1 for p in r["pools"])
            assert r["torch_threads"] in (None, 1)
    # the readout itself, on this process's own libraries
    spec = importlib.util.spec_from_file_location("blas_threads", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import numpy  # noqa: F401
    import scipy.sparse.linalg  # noqa: F401
    libs = mod.loaded_libraries()
    assert libs and len(libs) == len(set(libs)) and all(x.startswith("/") for x in libs)
    assert mod.pools(["/nonexistent/libopenblas.so"]) == []
