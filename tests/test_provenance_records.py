"""cmame-paper Stage 10a: the export of the hosting service's activity record
(records/provenance/) is the committed one, and it and the commits support what
Appendix A of the manuscript and the addendum of 10 October 2026 to
PROVENANCE_NOTE.md state: main created at b365af5 (an upload through the web
interface, signed by the service) on 14 July 2026 at 03:25 UTC, holding the
v2.1.4 code, the unstamped PREREG.md template and the registered configuration;
nothing else on the repository until the force push of 2 August 2026 at
17:29 UTC, which replaced that history with the one that begins at a548825
(tag v2.1.5)."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PV = ROOT / "records" / "provenance"
EXPORT = PV / "github_activity_2026-10-10.json"
FIRST = "b365af58b5135c6d079b377c26af419bdfe5fcaa"     # first commit of the first history
V215 = "a548825da6d02c91c7ccc4b6fcd6814116b2117e"      # first commit of the present history
TAG = "provenance-2026-07-14"
CONFIG = "configs/phase1_rec8_v2.json"
V214_TO_V215 = ["src/fejepa/__init__.py", "src/fejepa/experiments/protocol.py",
                "src/fejepa/experiments/runner.py", "tests/test_asis_guard.py"]


def _flat(p: Path) -> str:
    return re.sub(r"\s+", " ", p.read_text(encoding="utf-8"))


def _git(*args, text=True):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=text)


def _has(rev: str) -> bool:
    return shutil.which("git") is not None and _git("cat-file", "-e", f"{rev}^{{commit}}").returncode == 0


def test_the_export_is_the_committed_one():
    sha = hashlib.sha256(EXPORT.read_bytes()).hexdigest()
    assert f"| `{EXPORT.name}` | `{sha}` |" in (PV / "README.md").read_text(encoding="utf-8")


def test_the_export_logs_what_the_text_states():
    events = json.loads(EXPORT.read_text(encoding="utf-8"))
    assert len(events) == 103 and len({e["id"] for e in events}) == 103
    assert [e["timestamp"] for e in events] == sorted(e["timestamp"] for e in events)
    kinds = {k: sum(e["activity_type"] == k for e in events)
             for k in ("push", "branch_creation", "force_push", "branch_deletion")}
    assert kinds == {"push": 96, "branch_creation": 5, "force_push": 1, "branch_deletion": 1}
    assert [e["ref"] for e in events if e["activity_type"] == "branch_deletion"] == \
        ["refs/tags/prereg-phase2b"]
    assert not any(e["ref"].startswith("refs/tags/") and e["activity_type"] != "branch_deletion"
                   for e in events)                       # no tag creation is logged
    assert (events[-1]["timestamp"], events[-1]["ref"]) == ("2026-10-10T09:28:44Z", "refs/heads/main")
    early = [(e["timestamp"], e["activity_type"], e["ref"], e["before"], e["after"])
             for e in events if e["timestamp"] < "2026-08-06"]
    assert early == [("2026-07-14T03:25:10Z", "branch_creation", "refs/heads/main", "0" * 40, FIRST),
                     ("2026-08-02T17:29:37Z", "force_push", "refs/heads/main", FIRST, V215)]
    readme = _flat(PV / "README.md")
    for s in ("103 events", "96 pushes", "5 branch creations", "1 force push", "1 deletion",
              "It logs no tag creations"):
        assert s in readme, s
    app = _flat(ROOT / "paper" / "cmame" / "sections" / "appendix_prereg.tex")
    for s in ("at 03:25 UTC that day", "(\\texttt{b365af5})", "2 August 2026 at 17:29 UTC",
              "(\\texttt{a548825}, tagged \\texttt{v2.1.5})", "dated 17:26 UTC",
              f"\\texttt{{{TAG}}}", "logs the creation of the branch at that commit on 14 July"):
        assert s in app, s
    note = _flat(ROOT / "PROVENANCE_NOTE.md")
    for s in (FIRST, V215, "2 August 2026 at 17:29 UTC", "03:25 UTC", TAG, "records/provenance/"):
        assert s in note, s


def test_the_present_history_begins_where_the_text_says():
    """a548825 is a root commit dated 2026-08-03 01:26:23 +0800 (17:26 UTC on 2 August)
    that holds the two stamped documents and the provenance note as compiled on 1 August,
    which records their hashes; the documents are unchanged since."""
    if not _has(V215):
        pytest.skip("history not available (shallow or exported checkout)")
    assert _git("rev-list", "--parents", "-n", "1", V215).stdout.split() == [V215]
    assert _git("show", "-s", "--format=%cI", V215).stdout.strip() == "2026-08-03T01:26:23+08:00"
    note = _git("show", f"{V215}:PROVENANCE_NOTE.md", text=False)
    assert note.returncode == 0 and b"Compiled 1 August 2026." in note.stdout
    flat = re.sub(rb"\s+", b"", note.stdout).decode("utf-8")
    for doc in ("PREREG.md", "PREREG_WP2.md"):
        blob = _git("show", f"{V215}:{doc}", text=False)
        assert blob.returncode == 0, doc
        assert hashlib.sha256(blob.stdout).hexdigest() in flat, doc
        assert _git("rev-parse", f"{V215}:{doc}").stdout == _git("rev-parse", f"HEAD:{doc}").stdout, doc


def test_the_first_commit_is_what_the_text_says():
    """Where the first commit is available (the tag fetched), it is checked directly;
    without it, the record states what was checked."""
    readme = _flat(PV / "README.md")
    for s in (FIRST, "committer GitHub", "B5690EEEBB952194", "03:25:10 UTC", "<fill before tagging>",
              "It did not hold `PREREG_WP2.md`", *V214_TO_V215):
        assert s in readme, s
    if not _has(FIRST):
        return
    from fejepa.report import config_sha256, read_prereg_hash

    head = _git("rev-list", "--parents", "-n", "1", FIRST).stdout.split()
    assert head == [FIRST]                                                     # a root commit
    assert _git("show", "-s", "--format=%cI|%cn|%s", FIRST).stdout.strip() == \
        "2026-07-14T11:25:05+08:00|GitHub|Add files via upload"
    assert "gpgsig" in _git("cat-file", "-p", FIRST).stdout
    old = _git("show", f"{FIRST}:PREREG.md").stdout.splitlines()
    new = _git("show", "HEAD:PREREG.md").stdout.splitlines()
    assert len(old) == len(new)
    diff = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    assert len(diff) == 1 and old[diff[0]].strip() == "CONFIG_SHA256 = <fill before tagging>"
    assert _git("rev-parse", f"{FIRST}:{CONFIG}").stdout == _git("rev-parse", f"HEAD:{CONFIG}").stdout
    cfg = json.loads(_git("show", f"{FIRST}:{CONFIG}").stdout)
    assert config_sha256(cfg) == read_prereg_hash(ROOT / "PREREG.md") == \
        "62b26ad868d424ef5527c8cb7d826c818aa1ba5cebbc76c7bfe665062781f0ce"
    assert _git("cat-file", "-e", f"{FIRST}:PREREG_WP2.md").returncode != 0
    changed = _git("diff", "--name-only", FIRST, V215, "--", "src", "tests", "pyproject.toml").stdout.split()
    assert changed == V214_TO_V215
    tag = _git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}^{{commit}}")
    if tag.returncode == 0:
        assert tag.stdout.strip() == FIRST


SECLOG = PV / "security_log_FE-JEPA_2026-10-10.json"
PRESENT, NAMESAKE = 1299960094, 1268189576


def test_the_security_log_entries_support_the_text():
    """The owner account's security log (its entries naming FE-JEPA, user agents and request
    identifiers removed): the present repository created public on 14 July 2026 at 03:23:30
    UTC, 13 seconds after its public namesake of 13 June 2026 was deleted; no visibility
    change; and the text and the note say so."""
    sha = hashlib.sha256(SECLOG.read_bytes()).hexdigest()
    assert f"| `{SECLOG.name}` | `{sha}` |" in (PV / "README.md").read_text(encoding="utf-8")
    entries = json.loads(SECLOG.read_text(encoding="utf-8"))
    assert len(entries) == 15
    assert all("FE-JEPA" in json.dumps(e) for e in entries)
    assert not any(k in e for e in entries for k in ("user_agent", "request_id",
                                                     "request_access_security_header"))
    assert not any(e["action"] == "repo.access" for e in entries)
    assert all(e.get("visibility", "public") == "public" and e.get("public_repo", True) is True
               for e in entries)
    key = [(e["@timestamp"], e["action"], e.get("repo_id")) for e in entries
           if e["action"] in ("repo.create", "repo.destroy")]
    assert key == [(1781339807792, "repo.create", NAMESAKE),       # 2026-06-13T08:36:47.792Z
                   (1783999397268, "repo.destroy", NAMESAKE),      # 2026-07-14T03:23:17.268Z
                   (1783999410468, "repo.create", PRESENT)]        # 2026-07-14T03:23:30.468Z
    assert round((key[2][0] - key[1][0]) / 1000) == 13
    app = _flat(ROOT / "paper" / "cmame" / "sections" / "appendix_prereg.tex")
    assert "created, public, on 14 July 2026 at 03:23 UTC" in app
    note = _flat(ROOT / "PROVENANCE_NOTE.md")
    for s in ("created, public, on 14 July 2026 at 03:23 UTC", "created on 13 June 2026",
              "03:23:17 UTC"):
        assert s in note, s
