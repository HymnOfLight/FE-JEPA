"""fine3d: RUNBOOK_FINE3D.md names only scripts and configurations that exist
and options they accept, fetches the branch and the tag directly with a
bundle route, checks the code with the string every report records, states
the suite's count, and runs scripts/fine3d_session.sh; the session script
(run here with stand-ins for git, python and the steps) runs the six steps
in order, skips finished ones, restarts an interrupted R2a arm from its
states or its checkpoints, moves an unreadable report aside, runs a step
again until its output is complete, stops before any step when the code is
not the tag alone and unmodified, and refuses a second copy."""

import json
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "RUNBOOK_FINE3D.md"
REFS = ("+refs/heads/wp9-pool:refs/remotes/origin/wp9-pool "
        "+refs/tags/prereg-fine3d:refs/tags/prereg-fine3d")
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"}
STEPS = ["r2a_b_m1024", "r2a_s_m1024", "amp_r2a_b_m1024", "amp_r2a_s_m1024", "r3_eft_b16",
         "r3_lft_b16"]


def _text():
    return RUNBOOK.read_text(encoding="utf-8")


def _section(num):
    t = _text()
    start = t.index(f"\n## {num}. ")
    nxt = t.find("\n## ", start + 1)
    return t[start:] if nxt < 0 else t[start:nxt]


def _blocks(text):
    return re.findall(r"```bash\n(.*?)```", text, re.S)


def test_ascii():
    assert _text().isascii()


def test_the_runbook_names_existing_scripts_configs_and_options():
    text = "\n".join(_blocks(_text())).replace("\\\n", " ")
    scripts = re.findall(r"python (scripts/\S+\.py)([^\n]*)", text)
    assert {s for s, _ in scripts} == {"scripts/r3_finetune.py", "scripts/adjudicate_fine3d.py"}
    for script, rest in scripts:
        res = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True,
                             text=True, cwd=ROOT, env=ENV)
        assert res.returncode == 0, res.stderr[-2000:]
        for o in re.findall(r"(--[a-z][a-z0-9-]*)", rest):
            assert o in res.stdout, (script, o)
    for c in re.findall(r"configs/(\S+?)\.json", text):
        assert (ROOT / "configs" / f"{c}.json").is_file() or c == "$c", c
    assert "bash scripts/fine3d_session.sh" in text and (ROOT / "scripts/fine3d_session.sh").is_file()
    sess = (ROOT / "scripts" / "fine3d_session.sh").read_text()
    for opt in ("--report", "--states-dir", "--remesh", "--out"):
        assert opt in sess
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "posthoc_amplitude.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT, env=ENV)
    assert all(o in res.stdout for o in ("--report", "--states-dir", "--remesh", "--out"))


def test_0c_finds_no_fine3d_run_and_can_be_repeated(tmp_path):
    """0c's check prints exactly its comment's line on a box without a fine3d
    run, also once 0c has made runs/fine3d/session (a repeated 0c), and
    otherwise names what is there."""
    line = next(l for l in _blocks(_text())[3].splitlines() if "runs/fine3d" in l
                and "mkdir" not in l)
    cmd, want = line.split("#")[0].strip(), line.split("must print exactly: ")[1].strip()

    def out():
        return subprocess.run(["bash", "-c", cmd], cwd=tmp_path, capture_output=True, text=True,
                              env=ENV).stdout
    (tmp_path / "runs" / "e2_m1024").mkdir(parents=True)        # other runs on the box
    assert out() == want + "\n"
    (tmp_path / "runs" / "fine3d" / "session").mkdir(parents=True)
    (tmp_path / "runs" / "fine3d" / "session" / "pytest.log").write_text("x")
    assert out() == want + "\n"
    (tmp_path / "runs" / "fine3d" / "session_old").mkdir()     # not the session's own
    assert out() == "session_old\n"
    (tmp_path / "runs" / "fine3d" / "session_old").rmdir()
    (tmp_path / "runs" / "fine3d" / "r2a_b_m1024").mkdir()
    assert out() == "r2a_b_m1024\n"


def test_the_fetch_is_direct_with_a_bundle_route():
    b = _blocks(_section("0"))
    fetch = [x.split("#")[0].rstrip() for x in b[1].splitlines()]
    assert fetch == ["cd ~/autodl-tmp/FE-JEPA",
                     f"git fetch origin {REFS} && echo FETCH-OK || echo FETCH-FAILED",
                     "git rev-parse --short origin/wp9-pool",
                     "git rev-parse --short 'prereg-fine3d^{commit}'"]
    bundle = [x.split("#")[0].rstrip() for x in b[2].splitlines()]
    assert bundle == ["cd ~/autodl-tmp/FE-JEPA", "sha256sum ~/wp9-pool.bundle",
                      f"git fetch ~/wp9-pool.bundle {REFS} && echo FETCH-OK || echo FETCH-FAILED",
                      "git rev-parse --short origin/wp9-pool",
                      "git rev-parse --short 'prereg-fine3d^{commit}'"]
    check = [x.split("#") for x in b[3].splitlines()]
    lines = {c[0].rstrip(): c[1].strip() if len(c) > 1 else "" for c in check}
    assert lines["git describe --always --dirty --tags"] == "must print exactly: prereg-fine3d"
    assert lines["git tag --points-at HEAD"] == "must print exactly: prereg-fine3d"
    assert lines["git status --porcelain --untracked-files=no"] == "must print nothing"
    # the same check as the session script's and the string every report records
    sess = (ROOT / "scripts" / "fine3d_session.sh").read_text()
    for cmd in ("git describe --always --dirty --tags", "git tag --points-at HEAD",
                "git status --porcelain --untracked-files=no"):
        assert cmd in sess
    import inspect

    from fejepa.report import _git_describe
    assert '["git", "describe", "--always", "--dirty", "--tags"]' in inspect.getsource(_git_describe)


def test_the_runbook_states_the_suite_count():
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                          "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=ROOT)
    n = int(re.search(r"(\d+) tests? collected", res.stdout).group(1))
    notes = (ROOT / "BRANCH_NOTES_wp9-pool.md").read_text(encoding="utf-8")
    assert int(re.findall(r"^- Suite (\d+)", notes, re.M)[-1]) == n
    sec = _section("0")
    assert sec.count(f'"{n - 1} passed, 1 skipped"') == 1
    assert sec.count(f"`{n - 1} passed, 1 skipped`") == 1 and sec.count(f"`{n} passed`") == 1


# ------------------------------------------------------- the session script --

_GIT = """#!/usr/bin/env bash
# stand-in for git: only the session's exact checks read the stub's files
case "$*" in
  "describe --always --dirty --tags") cat "$STUB_DIR/describe"; exit 0;;
  "tag --points-at HEAD") cat "$STUB_DIR/tags"; exit 0;;
  "status --porcelain --untracked-files=no") cat "$STUB_DIR/status"; exit 0;;
esac
[ "$1" = describe ] && { echo prereg-fine3d-1-gabcdef; exit 0; }
exit 1
"""

_PY = r"""#!/usr/bin/env bash
# stand-in for python: checks run the real interpreter; the steps write their outputs
if [ "$1" = "-" ]; then
  body=$(cat)
  if printf '%s' "$body" | grep -q "torch.version.cuda"; then    # the session's version line
    echo "torch stub | CUDA build stub"; exit 0; fi
  if printf '%s' "$body" | grep -q "torch.cuda.is_available"; then   # the GPU probe
    n=$(( $(cat "$STUB_DIR/gpu_checks" 2>/dev/null || echo 0) + 1 )); echo $n > "$STUB_DIR/gpu_checks"
    if [ -n "${STUB_GPU_SLEEP:-}" ] && [ "$n" -ge "${STUB_GPU_SLEEP_FROM:-1}" ]; then
      echo $$ > "$STUB_DIR/probe_pid"; sleep "$STUB_GPU_SLEEP"; echo $$ > "$STUB_DIR/probe_done"; fi
    [ -n "${STUB_NOGPU:-}" ] && { echo "no GPU"; exit 1; }
    [ -n "${STUB_NOGPU_AFTER:-}" ] && [ "$n" -gt "$STUB_NOGPU_AFTER" ] && { echo "no GPU"; exit 1; }
    [ -n "${STUB_BUSY_AFTER:-}" ] && [ "$n" -gt "$STUB_BUSY_AFTER" ] && {
      echo "the GPU is in use (20.0 of 31.4 GiB taken)"; exit 1; }
    exit 0; fi
  # a completeness check that dies (killed, out of memory): STUB_CHECK_KILLED names a
  # word of its code
  [ -n "${STUB_CHECK_KILLED:-}" ] && printf '%s %s' "$body" "$*" | grep -q "$STUB_CHECK_KILLED" && exit 137
  # a completeness check whose reads fail with an I/O error: STUB_CHECK_EIO names a word of it
  if [ -n "${STUB_CHECK_EIO:-}" ] && printf '%s %s' "$body" "$*" | grep -q "$STUB_CHECK_EIO"; then
    body="import builtins, errno
def _eio(*a, **k):
    raise OSError(errno.EIO, 'simulated I/O error')
builtins.open = _eio
$body"; fi
  printf '%s' "$body" | "$REAL_PY" - "${@:2}"; exit $?
fi
if [ "$1" = "-c" ]; then exec "$REAL_PY" "$@"; fi
echo "CALL $*" >> "$STUB_DIR/calls"
if [ "$1" = "-m" ]; then                               # run-config configs/<name>.json ...
  name=$(basename "$4" .json)
  [ "${SLEEP_IN:-}" = "$name" ] && sleep 30
  [ "${FAIL:-}" = "$name" ] && { mkdir -p runs/fine3d/$name/e8_states; : > runs/fine3d/$name/e8_states/${FAIL_FILE:-ar_p1024_s3.pt}; exit 3; }
  mkdir -p runs/fine3d/$name/e8_states; echo '{"provenance": {}}' > runs/fine3d/$name/report.json
  [ "${CHANGE_AFTER:-}" = "$name" ] && echo prereg-fine3d-dirty > "$STUB_DIR/describe"
  exit 0
fi
case "$1" in
  scripts/posthoc_amplitude.py)
    out=""; rep=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; [ "$1" = --report ] && rep=$2; shift; done
    mkdir -p "$(dirname "$out")"
    "$REAL_PY" -c 'import hashlib,json,os,sys; f=int(os.environ.get("AMP_FAILED", "0")); s={"inband":{"summary":{"complete":True}},"fine":{"summary":{"complete":True}},"remesh":{"rows":[{}]*16,"n_geometries":16-f,"n_failed":f}}; json.dump({"report_sha256":hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest(),"seeds":{"s3":s,"s4":s,"s5":s}}, open(sys.argv[1],"w"))' "$out" "$rep"; exit 0;;
  scripts/r3_finetune.py)
    name=$(basename "$2" .json)
    [ "${FAIL:-}" = "$name" ] && exit 3
    mkdir -p runs/fine3d/$name
    [ "${PARTIAL:-}" = "$name" ] && { echo '{"per_seed": {"s0": {}, "s1": {}}}' > runs/fine3d/$name/report.json; exit 0; }
    echo '{"per_seed": {"s0": {}, "s1": {}, "s2": {}}}' > runs/fine3d/$name/report.json; exit 0;;
esac
exit 9
"""


_FLOCK = """#!/usr/bin/env bash
# stand-in for flock: slow when asked (to interrupt the session before it holds the lock)
[ -n "${STUB_FLOCK_SLEEP:-}" ] && sleep "$STUB_FLOCK_SLEEP"
exec {real} "$@"
"""


@pytest.fixture
def box(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "fine3d_session.sh").write_text(
        (ROOT / "scripts" / "fine3d_session.sh").read_text())
    stub = tmp_path / "stub"
    stub.mkdir()
    real_flock = shutil.which("flock", path="/usr/bin:/bin:/usr/local/bin") or "flock"
    for name, body in (("git", _GIT), ("python", _PY),
                       ("flock", _FLOCK.replace("{real}", real_flock))):
        p = stub / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    (stub / "describe").write_text("prereg-fine3d\n")
    (stub / "tags").write_text("prereg-fine3d\n")
    (stub / "status").write_text("")
    env = {"PATH": f"{stub}:/usr/bin:/bin", "STUB_DIR": str(stub), "REAL_PY": sys.executable,
           "HOME": str(tmp_path)}
    return tmp_path, stub, env


def _run(box, **extra):
    tmp, _, env = box
    return subprocess.run(["bash", "scripts/fine3d_session.sh"], cwd=tmp, capture_output=True,
                          text=True, env={**env, **extra}, timeout=120)


def _status(tmp):
    return (tmp / "runs" / "fine3d" / "session" / "status.txt").read_text()


def test_session_runs_the_steps_in_order_then_skips_them(box):
    tmp, stub, _ = box
    r = _run(box)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = (stub / "calls").read_text().splitlines()
    assert [c.split()[4] if c.split()[1] == "-m" else c.split()[1] for c in calls] == [
        "configs/r2a_b_m1024.json", "configs/r2a_s_m1024.json", "scripts/posthoc_amplitude.py",
        "scripts/posthoc_amplitude.py", "scripts/r3_finetune.py", "scripts/r3_finetune.py"]
    assert all("--reuse-states" not in c for c in calls)
    assert "--label-workers 1" in calls[0] and "--remesh 16" in calls[2]
    assert calls[4].endswith("scripts/r3_finetune.py configs/r3_eft_b16.json")
    assert calls[5].endswith("scripts/r3_finetune.py configs/r3_lft_b16.json")
    st = _status(tmp)
    assert [ln.split()[0] for ln in st.splitlines() if "exit=0" in ln] == \
        [f"{s}.log" for s in STEPS]
    (stub / "calls").unlink()
    r = _run(box)
    assert not (stub / "calls").exists()                    # nothing ran again
    st = _status(tmp)
    assert st.count("report exists, skipped") == 4 and st.count("record complete, skipped") == 2


@pytest.mark.parametrize("left", ["ar_p1024_s3.pt", "ar_p1024_s3.ckpt"])
def test_session_restarts_an_interrupted_arm_from_its_states(box, left):
    tmp, stub, _ = box
    r = _run(box, FAIL="r2a_b_m1024", FAIL_FILE=left)
    assert "r2a_b_m1024.log exit=3" in _status(tmp)
    assert "amp_r2a_b_m1024: no report of r2a_b_m1024, skipped" in _status(tmp)
    (stub / "calls").unlink()
    r = _run(box)
    assert r.returncode == 0, r.stdout + r.stderr
    calls = (stub / "calls").read_text().splitlines()
    assert "--reuse-states" in calls[0] and "configs/r2a_b_m1024.json" in calls[0]
    assert len(calls) == 2 and "posthoc_amplitude.py" in calls[1] and "r2a_b_m1024" in calls[1]
    assert "r2a_b_m1024: restarting from its states" in _status(tmp)
    assert len(list((tmp / "runs" / "fine3d" / "session").glob("r2a_b_m1024.log.*"))) == 1
    assert len(re.findall(r"^r2a_b_m1024\.log start", _status(tmp), re.M)) == 2


def test_session_runs_the_labelled_arm_only_after_a_complete_energy_arm(box):
    tmp, stub, _ = box
    _run(box, FAIL="r3_eft_b16")
    st = _status(tmp)
    assert "r3_eft_b16.log exit=3" in st
    assert "r3_lft_b16: not run, r3_eft_b16 has no complete report (R3 needs both arms)" in st
    assert not any("r3_lft_b16" in c for c in (stub / "calls").read_text().splitlines())
    (stub / "calls").unlink()
    r = _run(box)
    assert r.returncode == 0, r.stdout + r.stderr
    assert [c.split()[-1] for c in (stub / "calls").read_text().splitlines()] == \
        ["configs/r3_eft_b16.json", "configs/r3_lft_b16.json"]


def test_session_needs_all_three_seeds_of_the_energy_arm(box):
    """An energy arm's report with two seeds is not complete: the labelled arm
    does not start, and the energy arm runs again next time."""
    tmp, stub, _ = box
    _run(box, PARTIAL="r3_eft_b16")
    st = _status(tmp)
    assert "r3_eft_b16.log exit=0" in st and "r3_lft_b16: not run" in st
    (stub / "calls").unlink()
    _run(box)
    assert [c.split()[-1] for c in (stub / "calls").read_text().splitlines()] == \
        ["configs/r3_eft_b16.json", "configs/r3_lft_b16.json"]


def test_session_runs_the_amplitude_pass_again_for_a_new_report(box):
    """An amplitude record measured on an earlier report of the arm (the arm
    run again) is not a finished step: the pass runs again on the new one."""
    tmp, stub, _ = box
    _run(box)
    rep = tmp / "runs" / "fine3d" / "r2a_s_m1024" / "report.json"
    rep.write_text('{"provenance": {"again": true}}')
    (stub / "calls").unlink()
    _run(box)
    calls = (stub / "calls").read_text().splitlines()
    assert [c.split()[1] for c in calls] == ["scripts/posthoc_amplitude.py"]
    assert "amp_r2a_s_m1024.json" in calls[0]
    assert "amp_r2a_b_m1024: record complete, skipped" in _status(tmp)


def test_session_moves_an_unreadable_report_aside(box):
    tmp, stub, _ = box
    d = tmp / "runs" / "fine3d" / "r2a_b_m1024"
    d.mkdir(parents=True)
    (d / "report.json").write_text('{"per_seed": ')
    _run(box)
    assert "r2a_b_m1024: unreadable report moved aside" in _status(tmp)
    assert [p.read_text() for p in d.glob("report.json.unreadable.*")] == ['{"per_seed": ']
    calls = (stub / "calls").read_text().splitlines()
    assert "configs/r2a_b_m1024.json" in calls[0] and "--reuse-states" not in calls[0]


def test_session_runs_a_step_again_until_its_output_is_complete(box):
    tmp, stub, _ = box
    _run(box)
    f3 = tmp / "runs" / "fine3d"
    (f3 / "r3_eft_b16" / "report.json").write_text('{"per_seed": {"s0": {}, "s1": {}}}')
    amp = json.loads((f3 / "posthoc" / "amp_r2a_s_m1024.json").read_text())
    amp["seeds"]["s4"]["fine"]["summary"]["complete"] = False
    (f3 / "posthoc" / "amp_r2a_s_m1024.json").write_text(json.dumps(amp))
    amp = json.loads((f3 / "posthoc" / "amp_r2a_b_m1024.json").read_text())
    amp["seeds"]["s5"]["remesh"]["rows"] = [{}] * 15
    (f3 / "posthoc" / "amp_r2a_b_m1024.json").write_text(json.dumps(amp))
    (stub / "calls").unlink()
    _run(box)
    calls = (stub / "calls").read_text().splitlines()
    assert [c.split()[1] for c in calls] == ["scripts/posthoc_amplitude.py",
                                             "scripts/posthoc_amplitude.py",
                                             "scripts/r3_finetune.py"]
    assert "amp_r2a_b_m1024.json" in calls[0] and "amp_r2a_s_m1024.json" in calls[1]
    assert calls[2].endswith("configs/r3_eft_b16.json")          # the labelled arm is complete
    assert _status(tmp).count("report exists, skipped") == 3
    assert "r3_lft_b16: report exists, skipped" in _status(tmp)


@pytest.mark.parametrize("what,content", [("describe", "prereg-fine3d-1-gabcdef\n"),
                                          ("describe", "prereg-fine3d-dirty\n"),
                                          ("tags", "prereg-fine3d\nprereg-fine3d-old\n"),
                                          ("tags", ""),
                                          ("status", " M src/fejepa/metrics.py\n")])
def test_session_stops_when_the_code_is_not_the_tag(box, what, content):
    tmp, stub, _ = box
    (stub / what).write_text(content)
    _run(box)
    assert not (stub / "calls").exists()
    assert _status(tmp).count("STOP before") == 6


def test_session_checks_the_code_before_every_step(box):
    tmp, stub, _ = box
    _run(box, CHANGE_AFTER="r2a_b_m1024")
    calls = (stub / "calls").read_text().splitlines()
    assert len(calls) == 1 and "configs/r2a_b_m1024.json" in calls[0]
    st = _status(tmp)
    assert re.findall(r"STOP before (\S+): the code is prereg-fine3d-dirty", st) == STEPS[1:]
    assert "r2a_b_m1024.log exit=0" in st


def test_session_refuses_a_second_copy(box):
    import time

    tmp, _, env = box
    lock = tmp / "runs" / "fine3d" / "session"
    lock.mkdir(parents=True)
    holder = subprocess.Popen(["flock", str(lock / ".lock"), "sleep", "30"])
    try:
        for _ in range(200):                       # wait until the holder has the lock
            if subprocess.run(["flock", "-n", str(lock / ".lock"), "true"]).returncode != 0:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the lock holder did not take the lock within 10 s")
        r = _run(box)
        assert r.returncode == 1 and "already running" in r.stdout
        assert "pgrep -af 'run-config|posthoc_amplitude|r3_finetune'" in r.stdout   # RUNBOOK 1c's
    finally:
        holder.kill()


def test_session_checks_the_gpu_before_every_step_it_runs(box):
    """A GPU lost after the first step: every later step that would run stops
    with a line in the status file (none starts on the CPU); steps with nothing
    to do are reported as before."""
    tmp, stub, _ = box
    r = _run(box, STUB_NOGPU_AFTER="2")        # the session's own check and the first step's
    assert r.returncode == 0
    st = _status(tmp)
    assert "r2a_b_m1024.log exit=0" in st
    assert re.findall(r"STOP before (\S+): no GPU", st) == ["r2a_s_m1024", "amp_r2a_b_m1024",
                                                            "r3_eft_b16"]
    assert "amp_r2a_s_m1024: no report of r2a_s_m1024, skipped" in st
    assert "r3_lft_b16: not run, r3_eft_b16 has no complete report" in st
    assert len((stub / "calls").read_text().splitlines()) == 1
    assert int((stub / "gpu_checks").read_text()) == 5


def test_session_restarts_nothing_without_the_gpu(box):
    """An R2a arm that failed and left its states, the session run again with
    the GPU lost: the arm is not restarted, and the status file says why (not
    that it is restarting); a finished arm is still skipped."""
    tmp, stub, _ = box
    _run(box, FAIL="r2a_b_m1024")                              # leaves a state, exits 3
    before = _status(tmp)
    (stub / "gpu_checks").write_text("0")
    _run(box, STUB_NOGPU_AFTER="1")                            # the session's own check only
    new = _status(tmp)[len(before):]
    assert "STOP before r2a_b_m1024: no GPU" in new
    assert "restarting from its states" not in new
    assert "r2a_s_m1024: report exists, skipped" in new         # finished in the first run
    assert "STOP before r2a_s_m1024" not in new


def test_session_needs_a_gpu_unless_rehearsing(box):
    tmp, stub, _ = box
    r = _run(box, STUB_NOGPU="1")
    assert r.returncode == 1 and "STOP: no GPU" in _status(tmp)
    assert not (stub / "calls").exists()
    r = _run(box, STUB_NOGPU="1", FINE3D_REHEARSAL="1")
    assert r.returncode == 0 and "rehearsal: no GPU check" in _status(tmp)
    assert len((stub / "calls").read_text().splitlines()) == 6
    assert "FINE3D_REHEARSAL" not in RUNBOOK.read_text()


def test_session_says_why_the_gpu_cannot_be_used(box):
    """fine3d r5: the probe's reason (here a GPU taken by another process)
    reaches the status file; the step does not start."""
    tmp, stub, _ = box
    _run(box, STUB_BUSY_AFTER="1")
    st = _status(tmp)
    assert "STOP before r2a_b_m1024: the GPU is in use (20.0 of 31.4 GiB taken)" in st
    assert not (stub / "calls").exists()


def test_the_gpu_probe_allocates_on_the_gpu_and_needs_it_free():
    """fine3d r5: what the probe checks (the stub above stands in for it; the
    tests below check its process group and the lock by behaviour)."""
    body = (ROOT / "scripts" / "fine3d_session.sh").read_text()
    probe = body[body.index("gpu_probe() {"):body.index("\n}\n", body.index("gpu_probe() {"))]
    assert "exec 9>&-" in probe and "timeout --foreground 300 python -" in probe
    for s in ('torch.zeros(1, device="cuda").add_(1)', "torch.cuda.synchronize()",
              "torch.cuda.mem_get_info()", "free < 0.9 * total"):
        assert s in probe
    calls = [ln for ln in body.splitlines() if "gpu_probe" in ln and "gpu_probe()" not in ln
             and "gpu_probe outside" not in ln and not ln.lstrip().startswith("#")]
    assert calls and all("$(gpu_probe)" in ln for ln in calls), calls   # only inside $(...)
    assert '[ "$BASHPID" != "$$" ] || ' in probe              # and refuses to run outside it


def test_session_keeps_an_amplitude_record_with_a_geometry_that_could_not_be_meshed(box):
    """fine3d r5: a geometry that cannot be meshed is a row with an error (the
    pass goes on, and a second pass would not mesh it either): the record with
    its 16 rows is complete for the session, and the adjudication reads it
    without its remesh readings. An error while evaluating stops the pass
    instead (tests/test_fine3d_r5.py), leaving fewer rows."""
    tmp, stub, _ = box
    _run(box, AMP_FAILED="1")
    (stub / "calls").unlink()
    _run(box)
    assert not (stub / "calls").exists()
    assert _status(tmp).count("record complete, skipped") == 2


@pytest.mark.parametrize("word,stop", [
    ("provenance", ["r2a_b_m1024: the check of its report failed (exit 137)",
                    "r2a_s_m1024: the check of its report failed (exit 137)"]),
    ("report_sha256", ["amp_r2a_b_m1024: the check of its record failed (exit 137)",
                       "amp_r2a_s_m1024: the check of its record failed (exit 137)"]),
    ("per_seed", ["r3_eft_b16: the check of its report failed (exit 137)",
                  "r3_lft_b16: the check of its report failed (exit 137)"])])
def test_a_check_that_dies_stops_its_step_and_moves_nothing(box, word, stop):
    """fine3d r5 (review D, F4): a completeness check killed on its own (the
    OOM killer, a failed fork, an I/O error) is not read as an incomplete
    output: no report is moved aside, nothing runs again, the status file says
    STOP before that step; the session goes on to the next."""
    tmp, stub, _ = box
    _run(box)
    before = sorted(p.name for p in (tmp / "runs" / "fine3d").rglob("*") if p.is_file())
    (stub / "calls").unlink()
    _run(box, STUB_CHECK_KILLED=word)
    assert not (stub / "calls").exists()
    st = _status(tmp)
    assert re.findall(r"STOP before (.+)", st) == stop
    assert "moved aside" not in st and "not run" not in st
    after = sorted(p.name for p in (tmp / "runs" / "fine3d").rglob("*") if p.is_file())
    assert after == before


@pytest.mark.parametrize("word,stop", [
    ("provenance", "r2a_b_m1024: the check of its report failed (exit 1)"),
    ("report_sha256", "amp_r2a_b_m1024: the check of its record failed (exit 1)"),
    ("per_seed", "r3_eft_b16: the check of its report failed (exit 1)")])
def test_an_io_error_in_a_check_is_not_an_incomplete_output(box, word, stop):
    """fine3d r5 (review F, round 2, D1): a check whose read of a complete
    output fails with an I/O error stops that step; it is not read as
    unreadable (a complete R2a report moved aside, a step run again)."""
    tmp, stub, _ = box
    _run(box)
    before = sorted(p.name for p in (tmp / "runs" / "fine3d").rglob("*") if p.is_file())
    (stub / "calls").unlink()
    _run(box, STUB_CHECK_EIO=word)
    assert not (stub / "calls").exists()
    st = _status(tmp)
    assert f"STOP before {stop}" in st and "moved aside" not in st and "not run" not in st
    assert sorted(p.name for p in (tmp / "runs" / "fine3d").rglob("*") if p.is_file()) == before


_FAKE_TORCH = """import warnings


class _Cuda:
    def is_available(self):
        if MODE == "nogpu":
            warnings.warn("CUDA initialization: The NVIDIA driver on your system is too old "
                          "(found version 11040).")
            return False
        return True

    def synchronize(self):
        if MODE == "broken":
            raise RuntimeError("CUDA error: an illegal memory access was encountered\\n"
                               "CUDA kernel errors might be asynchronously reported")

    def mem_get_info(self):
        gib = 2 ** 30
        return (int(11.4 * gib), int(31.4 * gib)) if MODE == "busy" else \
            (int(30.8 * gib), int(31.4 * gib))


class _T:
    def add_(self, x):
        return self


cuda = _Cuda()
MODE = open(__file__.replace("__init__.py", "mode")).read().strip()


def zeros(*a, **k):
    return _T()
"""


@pytest.mark.parametrize("mode,rc,said", [
    ("ok", 0, ""),
    ("nogpu", 1, "no GPU (CUDA initialization: The NVIDIA driver on your system is too old "
                 "(found version 11040).)"),
    ("busy", 1, "the GPU is in use (20.0 of 31.4 GiB taken)"),
    ("broken", 1, "the GPU does not work (RuntimeError: CUDA error: an illegal memory access "
                  "was encountered)")])
def test_the_gpu_probe_says_why(tmp_path, mode, rc, said):
    """fine3d r5: the probe's own code (the session tests use a stand-in for
    it), run against a stand-in torch: silent and 0 on a free working GPU, one
    line and 1 otherwise; outside $(...) it refuses to run (it closes the
    session's lock where it runs)."""
    fake = tmp_path / "fake" / "torch"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text(_FAKE_TORCH)
    (fake / "mode").write_text(mode)
    body = (ROOT / "scripts" / "fine3d_session.sh").read_text()
    i = body.index("gpu_probe() {")
    fn = body[i:body.index("\n}\n", i) + 3]
    script = tmp_path / "probe.sh"
    script.write_text(fn + 'why=$(gpu_probe); rc=$?; printf "%s|%s" "$rc" "$why"\n')
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin",
           "PYTHONPATH": str(tmp_path / "fake")}
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env,
                         timeout=120).stdout
    assert out == f"{rc}|{said}"
    script.write_text(fn + 'gpu_probe; echo "|$?"\n')
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env,
                         timeout=120).stdout
    assert out == "gpu_probe outside $(...)\n|2\n"


def test_ctrl_c_before_the_lock_ends_the_session(box):
    """fine3d r5 (review D, note 2): a SIGINT before the session holds its lock
    ends it with exit 130 (the minimal trap; without it the shell died of the
    signal); nothing starts and no status line is written."""
    import os
    import signal

    tmp, stub, _ = box
    p = _start(box, STUB_FLOCK_SLEEP="10")
    try:
        _wait_for(lambda: (tmp / "runs" / "fine3d" / "session" / ".lock").exists(), "the lock file")
        import time
        time.sleep(0.5)                                  # inside the slow flock
        os.killpg(p.pid, signal.SIGINT)
        assert p.wait(timeout=20) == 130
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert not (stub / "calls").exists()
    assert not (tmp / "runs" / "fine3d" / "session" / "status.txt").exists()


def test_a_failed_check_of_the_energy_arm_does_not_read_as_an_incomplete_one(box):
    """fine3d r5 (review D, F4): the labelled arm's check of the energy arm's
    report, killed on its own, is not "R3 needs both arms": STOP before the
    labelled arm, which does not start."""
    tmp, stub, _ = box
    _run(box)
    (tmp / "runs" / "fine3d" / "r3_lft_b16" / "report.json").unlink()
    (stub / "calls").unlink()
    _run(box, STUB_CHECK_KILLED="r3_eft_b16/report.json")
    assert not (stub / "calls").exists()
    st = _status(tmp)
    assert "STOP before r3_eft_b16: the check of its report failed (exit 137)" in st
    assert "STOP before r3_lft_b16: the check of r3_eft_b16's report failed (exit 137)" in st
    assert "not run" not in st


def _start(box, **extra):
    tmp, _, env = box
    return subprocess.Popen(["bash", "scripts/fine3d_session.sh"], cwd=tmp, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**env, **extra}, start_new_session=True)


def _wait_for(pred, what, timeout=20.0):
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return
        time.sleep(0.05)
    pytest.fail(f"{what} did not happen within {timeout:g} s")


def test_ctrl_c_during_a_step_stops_the_session(box):
    """fine3d r5: Ctrl-C (SIGINT to the session's process group) during a step
    ends the session: no later step, an attempt without an exit line, and a
    status line saying so."""
    import os
    import signal

    tmp, stub, _ = box
    p = _start(box, SLEEP_IN="r2a_b_m1024")
    try:
        _wait_for(lambda: (stub / "calls").exists(), "the first step")
        os.killpg(p.pid, signal.SIGINT)
        assert p.wait(timeout=20) == 130
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert len((stub / "calls").read_text().splitlines()) == 1
    st = _status(tmp)
    assert "r2a_b_m1024.log start" in st and "r2a_b_m1024.log exit=" not in st
    assert "session interrupted" in st


def _gone(pid, timeout=5.0):
    import os
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_ctrl_c_during_a_gpu_check_stops_the_session(box):
    """fine3d r5: Ctrl-C while the GPU probe runs (it runs under timeout, which
    without --foreground would have kept it out of the session's process
    group, so that the probe went on and the next step started) reaches the
    probe, starts nothing and writes no STOP line; the restart is not refused."""
    import os
    import signal

    tmp, stub, _ = box
    p = _start(box, STUB_GPU_SLEEP="10", STUB_GPU_SLEEP_FROM="2")   # the probe before a step
    try:
        _wait_for(lambda: (stub / "gpu_checks").exists()
                  and (stub / "gpu_checks").read_text().strip() == "2", "the step's GPU probe")
        os.killpg(p.pid, signal.SIGINT)
        assert p.wait(timeout=20) == 130
        assert _gone(int((stub / "probe_pid").read_text()))      # the probe had the SIGINT
        assert not (stub / "probe_done").exists()                # and did not run to its end
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    assert not (stub / "calls").exists()
    st = _status(tmp)
    assert "session interrupted" in st and "STOP" not in st
    r = _run(box)                                     # the lock was released
    assert r.returncode == 0 and len((stub / "calls").read_text().splitlines()) == 6


def test_a_killed_session_leaves_no_lock_behind(box):
    """fine3d r5: SIGKILL to the session's group while the GPU probe runs; the
    probe (in the group, and not holding the lock) dies with it, and the
    restart runs."""
    import os
    import signal

    tmp, stub, _ = box
    p = _start(box, STUB_GPU_SLEEP="10", STUB_GPU_SLEEP_FROM="2")   # the probe before a step
    try:
        _wait_for(lambda: (stub / "gpu_checks").exists()
                  and (stub / "gpu_checks").read_text().strip() == "2", "the step's GPU probe")
        os.killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=20)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    r = _run(box)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len((stub / "calls").read_text().splitlines()) == 6


def test_a_probe_left_running_does_not_hold_the_lock(box):
    """fine3d r5 (review F, F6 and N3): the session's shell alone killed while
    the GPU probe runs (the probe and its subshell go on): neither holds the
    session's lock, so the restart is not refused."""
    import os
    import signal

    tmp, stub, _ = box
    p = _start(box, STUB_GPU_SLEEP="10", STUB_GPU_SLEEP_FROM="2")
    try:
        _wait_for(lambda: (stub / "gpu_checks").exists()
                  and (stub / "gpu_checks").read_text().strip() == "2", "the step's GPU probe")
        os.kill(p.pid, signal.SIGKILL)                        # the shell only
        p.wait(timeout=20)
        assert not _gone(int((stub / "probe_pid").read_text()), timeout=0.5)   # still running
        r = _run(box)
        assert "already running" not in r.stdout, r.stdout
        assert r.returncode == 0 and len((stub / "calls").read_text().splitlines()) == 6
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:                 # the leftover probe had finished
            pass
