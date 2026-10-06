"""cmame-paper Stage 1: RUNBOOK_CMAME.md names only scripts that exist and
options they accept, runs exactly the commands of RUNBOOK_W9.md Sec. 4 apart
from the F5 timing, writes only under runs/cmame/, and states the suite's
current test count with the two LaTeX builds left out."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "RUNBOOK_CMAME.md"
LATEX = ("tests/test_cmame_material.py::test_manuscript_compiles",
         "tests/test_w9_session2.py::test_paper_material_compiles")


def _lines(text: str) -> list:
    """The `python scripts/... .py` commands of a text, continuation lines
    joined and spaces collapsed."""
    text = text.replace("\\\n", " ")
    return [" ".join(m.group(0).split())
            for m in re.finditer(r"python scripts/[\w./-]+\.py[^\n]*", text)]


def _commands() -> list:
    """(script, options) of every `python scripts/... .py` command."""
    out = []
    for line in _lines(RUNBOOK.read_text(encoding="utf-8")):
        m = re.match(r"python (scripts/\S+\.py)(.*)", line)
        out.append((m.group(1), re.findall(r"(--[a-z][a-z0-9-]*)", m.group(2))))
    return out


def test_the_commands_are_wp9s_section_4_without_f5():
    w9 = (ROOT / "RUNBOOK_W9.md").read_text(encoding="utf-8")
    sec4 = w9[w9.index("## 4. "):w9.index("## 5. ")]
    want = [re.sub(r" --family F5=\S+", "", c) for c in _lines(sec4)]
    assert want and _lines(RUNBOOK.read_text(encoding="utf-8")) == want


def test_the_runbook_names_existing_scripts_and_options():
    cmds = _commands()
    assert {c[0] for c in cmds} == {"scripts/time_inference_vs_solve.py",
                                    "scripts/export_fields.py"}
    for script, opts in cmds:
        assert (ROOT / script).is_file(), script
        res = subprocess.run([sys.executable, str(ROOT / script), "--help"],
                             capture_output=True, text=True, cwd=ROOT,
                             env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
        assert res.returncode == 0, res.stderr[-2000:]
        for o in opts:
            assert o in res.stdout, (script, o)


def test_the_runbook_writes_under_runs_cmame_only():
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "P=runs/cmame/timing" in text
    for m in re.finditer(r"--out (\S+)", text.replace("\\\n", " ")):
        assert m.group(1).startswith("$P/"), m.group(0)
    assert "runs/w9/" not in text


def test_the_runbook_states_the_suite_count():
    res = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                          "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=ROOT)
    m = re.search(r"(\d+) tests? collected", res.stdout)
    assert m, res.stdout[-2000:]
    n = int(m.group(1))
    for t in LATEX:                                         # the deselected tests exist
        assert t in res.stdout, t
    text = RUNBOOK.read_text(encoding="utf-8")
    assert f"`{n - len(LATEX)} passed, {len(LATEX)} deselected`" in text
    assert all(f"--deselect {t}" in text for t in LATEX)
