"""cmame-paper: RUNBOOK_CMAME.md names only scripts that exist and options
they accept; Sec. A runs exactly the commands of RUNBOOK_W9.md Sec. 4 apart
from the F5 timing and writes only under runs/cmame/timing/; Sec. B runs
CM2D's stamped configuration on the tag and writes only under runs/cm2d/
(its readiness check under runs/cmame/); Sec. D runs the spectral export on
CM2D's committed return and writes only under runs/cmame/spectra/, and the
figure instance it names is the script's rule applied to the committed
report; the runbook states the suite's current test count with the two LaTeX
builds left out."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "RUNBOOK_CMAME.md"
LATEX = ("tests/test_cmame_material.py::test_manuscript_compiles",
         "tests/test_w9_session2.py::test_paper_material_compiles")
SCRIPTS = {"A": {"scripts/time_inference_vs_solve.py", "scripts/export_fields.py"},
           "B": {"scripts/cm2d_precheck.py"},
           "C": {"scripts/adjudicate_cm2d.py"},
           "D": {"scripts/cm2d_spectra.py"}}
ENV = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}


def _section(letter: str) -> str:
    text = RUNBOOK.read_text(encoding="utf-8")
    start = text.index(f"\n## {letter}. ")
    nxt = text.find("\n## ", start + 1)
    return text[start:] if nxt < 0 else text[start:nxt]


def _lines(text: str, prefix: str = r"python scripts/[\w./-]+\.py") -> list:
    """The commands of a text matching `prefix`, continuation lines joined and
    spaces collapsed."""
    text = text.replace("\\\n", " ")
    return [" ".join(m.group(0).split()) for m in re.finditer(prefix + r"[^\n]*", text)]


def _commands(text: str) -> list:
    """(script, options) of every `python scripts/... .py` command."""
    out = []
    for line in _lines(text):
        m = re.match(r"python (scripts/\S+\.py)(.*)", line)
        out.append((m.group(1), re.findall(r"(--[a-z][a-z0-9-]*)", m.group(2))))
    return out


def _help(*args) -> str:
    res = subprocess.run([sys.executable, *args, "--help"], capture_output=True, text=True,
                         cwd=ROOT, env=ENV)
    assert res.returncode == 0, res.stderr[-2000:]
    return res.stdout


def test_section_a_is_wp9s_section_4_without_f5():
    w9 = (ROOT / "RUNBOOK_W9.md").read_text(encoding="utf-8")
    sec4 = w9[w9.index("## 4. "):w9.index("## 5. ")]
    want = [re.sub(r" --family F5=\S+", "", c) for c in _lines(sec4)]
    assert want and _lines(_section("A")) == want


def test_the_runbook_names_existing_scripts_and_options():
    for letter, scripts in SCRIPTS.items():
        cmds = _commands(_section(letter))
        assert {c[0] for c in cmds} == scripts, letter
        for script, opts in cmds:
            assert (ROOT / script).is_file(), script
            out = _help(str(ROOT / script))
            for o in opts:
                assert o in out, (script, o)
    # the CLI commands (Sec. B) and their options
    cli = _lines(RUNBOOK.read_text(encoding="utf-8"), r"python -m fejepa\.cli [\w-]+")
    assert cli
    for line in cli:
        sub = line.split()[3]
        out = _help("-m", "fejepa.cli", sub)
        for o in re.findall(r"(--[a-z][a-z0-9-]*)", line):
            assert o in out, (line, o)


def test_sections_write_under_their_own_directories():
    for letter, prefixes in (("A", ("runs/cmame/timing",)), ("B", ("runs/cm2d", "runs/cmame")),
                             ("D", ("runs/cmame/spectra",))):
        text = _section(letter).replace("\\\n", " ")
        ps = re.findall(r"\bP=([^\s;]+)", text)
        assert ps and all(p in prefixes for p in ps), (letter, ps)
        for m in re.finditer(r"--out (\S+)", text):
            assert m.group(1).startswith("$P/"), (letter, m.group(0))
        assert "runs/w9/" not in text, letter
    # Sec. D: every file it writes is under its directory or its own return
    d = _section("D").replace("\\\n", " ")
    for m in re.finditer(r"\btee (?:-a )?(\S+)|> (\S+)", d):
        target = m.group(1) or m.group(2)
        assert target.startswith(("$P/", "runs/cmame/spectra/")), m.group(0)
    assert re.findall(r"mkdir -p (\S+)", d) == ["runs/cmame/spectra", "$OUT"]
    assert re.findall(r"\bOUT=(\S+)", d) == ["~/cm2d_spectra_return"]
    helper = re.search(r"\nrun\(\) \{.*?; \}\n", _section("A"), re.S).group(0)
    assert helper in _section("D") and 'exit=${PIPESTATUS[0]}' in helper
    other = re.search(r'OTHER_RUNS = r"([^"]+)"',
                      (ROOT / "scripts" / "cm2d_precheck.py").read_text()).group(1)
    assert 'pgrep -af "' + other + '"' in d                 # the same check as B1c's


def test_section_b_runs_the_stamped_configuration_on_the_tag():
    b = _section("B")
    cli = _lines(b, r"python -m fejepa\.cli [\w-]+")
    assert cli == ["python -m fejepa.cli run-config configs/cm2d_v1.json",
                   "python -m fejepa.cli run-config configs/cm2d_v1.json --reuse-states"]
    assert "git describe --tags --match prereg-cm2d      # must print exactly: prereg-cm2d" in b
    assert ("+refs/heads/cmame-paper:refs/remotes/origin/cmame-paper "
            "+refs/tags/prereg-cm2d:refs/tags/prereg-cm2d") in b
    # the run starts only on the precheck's GO; a restart only on its own GO
    for name, cmd in (("precheck.json", "configs/cm2d_v1.json\n"),
                      ("precheck_restart.json", "configs/cm2d_v1.json --reuse-states\n")):
        assert (f"grep -q '\"go\": true' $P/{name} 2>/dev/null && mv $P/{name} "
                f"$P/{name}.used && run run.log python -m fejepa.cli run-config {cmd}") in b, name
    pre = (ROOT / "scripts" / "cm2d_precheck.py").read_text()
    other = re.search(r'OTHER_RUNS = r"([^"]+)"', pre).group(1)
    assert 'pgrep -af "' + other + '"' in b                 # B1c checks what the precheck does
    assert "python scripts/cm2d_precheck.py --restart --out $P/precheck_restart.json" in \
        b.replace("\\\n", " ").replace("  ", " ")
    prereg = (ROOT / "PREREG_CM2D.md").read_text(encoding="utf-8")
    assert "CONFIG_SHA256[cm2d_v1]" in prereg and "`prereg-cm2d`" in prereg
    # the runtime settings are the configuration's (workers) and the default
    # (activation checkpointing on): PREREG_CM2D Sec. 6
    assert "--workers" not in b and "--activation-checkpointing" not in b


def test_section_d_exports_from_cm2ds_committed_return():
    import importlib.util
    import json

    import numpy as np

    d = _section("D")
    assert _lines(d) == [
        "python scripts/cm2d_spectra.py --report records/cmame/cm2d/return/report.json "
        "--provenance records/cmame/cm2d/return/provenance.txt --states-dir runs/cm2d/e8_states "
        "--out $P/export --device cuda"]
    ret = ROOT / "records" / "cmame" / "cm2d" / "return"
    assert (ret / "report.json").is_file() and (ret / "provenance.txt").is_file()
    # the states it reads are the ones the return lists, twelve of them
    prov = (ret / "provenance.txt").read_text()
    states = re.findall(r"^[0-9a-f]{64}  (runs/\S+\.pt)$", prov, re.M)
    assert len(states) == 12 and all(s.startswith(("runs/cm2d/e8_states/",
                                                   "runs/e1_2d_base/e8_states/")) for s in states)
    assert "| wc -l   # 12" in d
    # the figure instance it names: the script's rule on the committed report,
    # and the file the run's split puts at that index (the split of the 2D
    # timing's records, whose 32 files follow the same permutation)
    spec = importlib.util.spec_from_file_location("cm2d_spectra",
                                                  ROOT / "scripts" / "cm2d_spectra.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    report = json.loads((ret / "report.json").read_text())
    i = mod.select(report)["fig2d"]["index"]
    split = report["config"]["split"]
    perm = np.random.default_rng(int(split["seed"])).permutation(int(report["config"]["data"]["n"]))
    timing = json.loads((ROOT / "records" / "cmame" / "timing" / "timing_2d.json").read_text())
    assert timing["report_sha256"] == report["reuse_from"]["report_sha256"]   # E1's corpus, split
    files = [r["file"] for r in timing["results"]["val"]["per_instance"]]
    assert files == [f"instance_{perm[k]:05d}.npz" for k in range(len(files))]
    assert f"`[spectra] fig2d: val #{i} instance_{perm[i]:05d}.npz (... nodes)`" in d
    assert f'`{{"fig2d": ["instance_{perm[i]:05d}.npz", {i}]}}`' in d
    # the summary line's keys, as the script prints them (tests/test_cm2d_spectra.py)
    for k in ("content_median_rel_dev", "content_mismatch", "rayleigh_ratio_median", "pairs",
              "prop1_bound_ratio_max", "figures"):
        assert f"`{k}`" in d, k


def test_section_d_reuses_the_blocks_that_ran():
    """Sec. D's fetch is A0b's, as it ran on 8 October; its bundle route, its
    checkout and checks, the rotation of earlier outputs, the run helper and
    the return named once per paste are pinned line by line."""
    blocks = lambda s: re.findall(r"```bash\n(.*?)```", s, re.S)          # noqa: E731
    a, d = blocks(_section("A")), blocks(_section("D"))
    cmd = lambda line: line.split("#")[0].rstrip()                        # noqa: E731
    assert len(d) == 6 and d[1] == a[1]                    # tmux; A0b's fetch; bundle; D0c; D1; D2
    assert [cmd(x) for x in d[2].splitlines()] == [
        "cd ~/autodl-tmp/FE-JEPA", "sha256sum ~/cmame-paper.bundle",
        "git fetch ~/cmame-paper.bundle +refs/heads/cmame-paper:refs/remotes/origin/cmame-paper"
        " && echo FETCH-OK || echo FETCH-FAILED", "git rev-parse --short origin/cmame-paper"]
    d0c, a0c = [cmd(x) for x in d[3].splitlines()], [cmd(x) for x in a[2].splitlines()]
    assert d0c[:4] == a0c[:4]                              # cd, checkout, tree, status
    assert 'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"' in d0c
    for line in ("[ -s $P/status.txt ] && mv $P/status.txt $P/status.txt.$(date +%Y%m%d-%H%M%S);"
                 " : > $P/status.txt",
                 "[ -d $P/export ] && mv $P/export $P/export.$(date +%Y%m%d-%H%M%S)",
                 "run spectra.log python scripts/cm2d_spectra.py"):
        assert line in d[4], line
    assert "T=$(date +%Y%m%d-%H%M)" in d[5] and "*" not in d[5].split("tar czf", 1)[1]


def test_the_runbook_states_the_suite_count():
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                          "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) tests? collected", res.stdout)
    assert m, res.stdout[-2000:]
    n = int(m.group(1))
    for t in LATEX:                                         # the deselected tests exist
        assert t in res.stdout, t
    for letter in "ABD":                                    # A0c, B1c, D0c
        sec = _section(letter)
        assert sec.count(f"`{n - len(LATEX)} passed, {len(LATEX)} deselected`") == 1, letter
        assert all(sec.count(f"--deselect {t}") == 1 for t in LATEX), letter


# ---- the suite the runbook runs on the box ends with its summary line ------
def _sigpipe_ignored() -> bool:
    import signal

    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("SigIgn:"):
            return bool(int(line.split()[1], 16) & (1 << (signal.SIGPIPE - 1)))
    raise AssertionError("no SigIgn line")


def test_a_test_that_meshes_with_gmsh_in_the_test_process():
    """gmsh.initialize() restores SIGPIPE's default action (ending the
    process) in the process that meshes; the test after this one checks that
    the next test starts with it ignored again (tests/conftest.py)."""
    pytest = __import__("pytest")
    pytest.importorskip("gmsh")
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc")
    import numpy as np

    from fejepa.fe.generator import mesh_plate

    state = np.random.get_state()             # mesh_plate names its model from it
    try:
        nodes, tris = mesh_plate(1.0, 1.0, [], 0.5)
    finally:
        np.random.set_state(state)
    assert len(nodes) > 3 and len(tris) > 1


def test_every_test_starts_with_sigpipe_ignored():
    """A failed worker pool after a gmsh test (tests/test_w9_parallel_kill.py
    after tests/test_w9_ood2d.py) could otherwise end the whole run by
    SIGPIPE, without the summary line RUNBOOK_CMAME A0c and B1c read."""
    pytest = __import__("pytest")
    if not sys.platform.startswith("linux"):
        pytest.skip("reads /proc")
    assert _sigpipe_ignored()
