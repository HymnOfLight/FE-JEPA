"""cmame-paper Stage 10e: the repository's transfer and new name (11 October 2026).

The activity record exported after the transfer begins with the 103 events of the export of
10 October 2026, and its compact form is the export as returned; the repository's record
after the transfer keeps the id of the repository whose creation the security log records;
the list of branches and tags gives, by Software Heritage's snapshot hash, the snapshot that
the manuscript, the records' README, the provenance note and CITATION.cff quote, and the
README states what the records show; the Data availability statement gives the address,
the tag, the Zenodo DOI and that snapshot, and no pending box is left in the manuscript;
the provenance note, the README and CITATION.cff agree with the manuscript."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PV = ROOT / "records" / "provenance"
FIRST = PV / "github_activity_2026-10-10.json"
SECOND = PV / "github_activity_2026-10-11.json"
REFS = PV / "refs_2026-10-11.txt"
REPO = PV / "repository_2026-10-11.json"
SECLOG = PV / "security_log_FE-JEPA_2026-10-10.json"
PAPER = ROOT / "paper" / "cmame"
SECTIONS = PAPER / "sections"
URL = "https://github.com/RuifengCao/label-free-fe-surrogates"
OLD_URL = "https://github.com/HymnOfLight/FE-JEPA"
MID_URL = "https://github.com/RuifengCao/FE-JEPA"
DOI = "10.5281/zenodo.23298799"
SNAPSHOT = "swh:1:snp:79f91af6d07e45a45933743bda38253b49c82bb6"
TAG = "cmame-submission"
REPO_ID = 1299960094
RAW_SHA256 = "f771989fa4077487d6e9d41d9601f0976bdb498665508cb634e440573aaabbab"
ARCHIVED_CMAME = "c5f6b84c00b9f700c464319902f7f1ad7d4de840"    # cmame-paper when archived


def _flat(text: str) -> str:
    return " ".join(text.split())


def _own_git() -> str | None:
    """git, if ROOT is the top of a work tree (not a tree unpacked inside another clone)."""
    git = shutil.which("git")
    if git is None:
        return None
    top = subprocess.run([git, "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True,
                         text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT.resolve():
        return None
    return git


def snapshot_swhid(listing: str) -> str:
    """Software Heritage's snapshot identifier of a `git ls-remote --symref` listing, as its
    git loader builds the snapshot: a symbolic ref is an alias, a ref with a peeled line (an
    annotated tag) a release, any other ref a revision; peeled lines are not branches. The
    manifest is that of swh.model (8.4.1), whose identifier for this listing it reproduces."""
    lines = [line.split("\t") for line in listing.splitlines() if line.strip()]
    alias = {name: tgt[len("ref: "):] for tgt, name in lines if tgt.startswith("ref: ")}
    plain = {name: sha for sha, name in lines if not sha.startswith("ref: ")}
    peeled = {name[:-3] for name in plain if name.endswith("^{}")}
    parts = []
    for name in sorted(set(plain) | set(alias), key=lambda s: s.encode()):
        if name.endswith("^{}"):
            continue
        if name in alias:
            kind, target = b"alias", alias[name].encode()
        else:
            kind = b"release" if name in peeled else b"revision"
            target = bytes.fromhex(plain[name])
        parts += [kind, b" ", name.encode(), b"\x00", b"%d:" % len(target), target]
    body = b"".join(parts)
    return "swh:1:snp:" + hashlib.sha1(b"snapshot %d\x00" % len(body) + body).hexdigest()


def _listing():
    listing = REFS.read_text(encoding="ascii")
    lines = [line.split("\t") for line in listing.splitlines()]
    names = {name for _, name in lines}
    heads = sorted(n for n in names if n.startswith("refs/heads/"))
    tags = sorted(n for n in names if n.startswith("refs/tags/") and not n.endswith("^{}"))
    peeled = sorted(n for n in names if n.endswith("^{}"))
    return listing, lines, names, heads, tags, peeled


def test_the_second_export_begins_with_the_first():
    readme = (PV / "README.md").read_text(encoding="utf-8")
    for f in (SECOND, REFS, REPO):
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        assert f"| `{f.name}` | `{sha}` |" in readme, f.name
    first = json.loads(FIRST.read_text(encoding="utf-8"))
    second = json.loads(SECOND.read_text(encoding="utf-8"))
    assert len(first) == 103 and len(second) == 110
    assert second[:103] == first                          # unchanged and in the same order
    # written as the first export was: one-space indentation and a final newline; its
    # compact form is the export as returned, whose hash the README quotes
    assert SECOND.read_text(encoding="utf-8") == json.dumps(second, indent=1) + "\n"
    compact = json.dumps(second, separators=(",", ":")).encode()
    assert hashlib.sha256(compact).hexdigest() == RAW_SHA256
    later = second[103:]
    assert [e["timestamp"] for e in later] == sorted(e["timestamp"] for e in later)
    assert later[0]["timestamp"] > first[-1]["timestamp"]
    assert (later[0]["timestamp"], later[-1]["timestamp"]) == (
        "2026-10-10T11:36:16Z", "2026-10-11T05:31:05Z")
    assert sum(e["activity_type"] == "push" for e in later) == 6
    assert [e["ref"] for e in later if e["activity_type"] == "branch_creation"] == \
        ["refs/heads/wp10-joint"]
    assert {e["activity_type"] for e in later} == {"push", "branch_creation"}
    flat = _flat(readme)
    for s in ("It holds 110 events",
              "the 103 events of `github_activity_2026-10-10.json`, unchanged and in the same "
              "order, followed by 6 pushes and 1 branch creation (of `wp10-joint`) between "
              "10 October 2026 at 11:36 UTC and 11 October 2026 at 05:31 UTC",
              "gh api -X GET repos/RuifengCao/label-free-fe-surrogates/activity \\ "
              "-f per_page=100 -f direction=asc --paginate",
              "exported again after the transfer, at 12:53 UTC that day",
              'the file\'s events written without whitespace (`json.dumps(events, '
              'separators=(",", ":"))` in Python), SHA-256 `' + RAW_SHA256 + "`",
              "An export taken from the old address before the transfer, at 05:52 UTC that "
              "day, gave the same output byte for byte"):
        assert s in flat, s


def test_the_repository_kept_its_id_and_the_readme_states_the_records():
    repo = json.loads(REPO.read_text(encoding="utf-8"))
    assert REPO.read_text(encoding="utf-8") == json.dumps(repo, indent=1) + "\n"
    assert repo == {"id": REPO_ID, "full_name": "RuifengCao/label-free-fe-surrogates",
                    "created_at": "2026-07-14T03:23:29Z", "visibility": "public",
                    "owner": "RuifengCao"}
    created = [e for e in json.loads(SECLOG.read_text(encoding="utf-8"))
               if e["action"] == "repo.create" and e.get("repo_id") == REPO_ID]
    assert len(created) == 1                       # the security log's creation of this id
    logged = created[0]["@timestamp"] / 1000
    api = datetime.strptime(repo["created_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc).timestamp()
    assert 0 <= logged - api < 2                   # the same creation, to the second
    _, _, _, heads, tags, peeled = _listing()
    flat = _flat((PV / "README.md").read_text(encoding="utf-8"))
    for s in ("transferred from `HymnOfLight`, the account that created it, to `RuifengCao` "
              "(at about 12:28 UTC), and then renamed from `FE-JEPA` to "
              "`label-free-fe-surrogates` (before 12:52 UTC)",
              f"read at 14:03 UTC that day: it keeps the id {REPO_ID}",
              f"redirects the old addresses, `{OLD_URL}` and `{MID_URL}`, to `{URL}`",
              f"git ls-remote --symref {URL}.git",
              f"at 13:33 UTC that day: {len(heads)} branches, {len(tags)} tags "
              f"({len(peeled)} of them annotated, with their peeled commits) and `HEAD`, "
              "which points to `main`",
              "from `HymnOfLight/FE-JEPA` before the transfer (05:53 and 07:18 UTC), from it "
              "and `RuifengCao/FE-JEPA` after the transfer (12:28 UTC), from all three "
              "addresses after the renaming (12:52 UTC) and from the new address at 13:33 UTC, "
              "the branches and tags were the same each time",
              "At 13:34 UTC the branch `wp10-joint` moved on, so later lists differ",
              "from the old address before the transfer and from the new address after it, "
              "both before 13:34 UTC",
              f"Both visits gave the snapshot `{SNAPSHOT}`, the snapshot identifier of the "
              "branches and tags of `refs_2026-10-11.txt`",
              "`swh.model` 8.4.1",
              "security log of `HymnOfLight`, the account that created the repository and "
              "owned it until 11 October 2026"):
        assert s in flat, s
    assert "owner account" not in flat


def test_the_snapshot_is_that_of_the_listed_branches_and_tags():
    listing, lines, names, heads, tags, peeled = _listing()
    assert lines[0] == ["ref: refs/heads/main", "HEAD"]
    assert heads == ["refs/heads/cmame-paper", "refs/heads/main", "refs/heads/wp10-joint",
                     "refs/heads/wp7-3d", "refs/heads/wp8-lejepa", "refs/heads/wp9-pool"]
    assert len(tags) == 11 and len(peeled) == 10
    assert all(p[:-3] in tags for p in peeled)
    sha = dict((name, s) for s, name in lines[1:])
    assert sha["HEAD"] == sha["refs/heads/main"]
    assert sha["refs/heads/cmame-paper"] == ARCHIVED_CMAME
    assert sha["refs/tags/provenance-2026-07-14^{}"] == "b365af58b5135c6d079b377c26af419bdfe5fcaa"
    assert sha["refs/tags/v2.1.5^{}"] == "a548825da6d02c91c7ccc4b6fcd6814116b2117e"
    assert snapshot_swhid(listing) == SNAPSHOT
    # where the objects are in this clone, they are of the kinds the list implies, each
    # annotated tag peels to the listed commit, and the archived cmame-paper is an ancestor
    # of the version checked out
    git = _own_git()
    if git is None:
        return
    for s, name in lines[1:]:
        res = subprocess.run([git, "cat-file", "-t", s], cwd=ROOT, capture_output=True,
                             text=True)
        if res.returncode != 0:
            continue                    # not in this clone (a single-branch or shallow one)
        tagged = f"{name}^{{}}" in names
        assert res.stdout.strip() == ("tag" if tagged else "commit"), name
        if tagged:
            peel = subprocess.run([git, "rev-parse", f"{s}^{{commit}}"], cwd=ROOT,
                                  capture_output=True, text=True)
            assert peel.stdout.strip() == sha[f"{name}^{{}}"], name
    have = subprocess.run([git, "cat-file", "-e", f"{ARCHIVED_CMAME}^{{commit}}"], cwd=ROOT,
                          capture_output=True)
    if have.returncode == 0:
        assert subprocess.run([git, "merge-base", "--is-ancestor", ARCHIVED_CMAME, "HEAD"],
                              cwd=ROOT, capture_output=True).returncode == 0


def test_the_manuscript_gives_the_address_tag_doi_and_snapshot():
    tex = (SECTIONS / "declarations.tex").read_text(encoding="utf-8")
    statement = _flat(tex.split(r"\section*{Data availability}", 1)[1].split(r"\section*", 1)[0])
    for s in (rf"are available at \url{{{URL}}}. The version described here is tagged "
              rf"\texttt{{{TAG}}} and archived at Zenodo, \url{{https://doi.org/{DOI}}}.",
              "The repository as it stood on 11 October 2026, when it moved to its present "
              "address, is archived with its full history by Software Heritage as the snapshot",
              rf"\href{{https://archive.softwareheritage.org/{SNAPSHOT}}}"
              rf"{{\nolinkurl{{{SNAPSHOT}}}}}."):
        assert s in statement, s
    texts = {f: f.read_text(encoding="utf-8") for f in (*SECTIONS.glob("*.tex"),
                                                        *PAPER.glob("*.tex"),
                                                        PAPER / "highlights.txt",
                                                        PAPER / "references.bib")}
    for f, text in texts.items():
        assert "owner account" not in text, f.name
        assert "HymnOfLight" not in text and "FE-JEPA" not in text, f.name
        if f.parent == SECTIONS:
            assert r"\pending" not in text, f.name      # no pending box is left
    assert texts[PAPER / "main.tex"].count(r"\pending") <= 1   # the macro's definition at most
    repro = _flat((SECTIONS / "appendix_repro.tex").read_text(encoding="utf-8"))
    for s in ("Beside it are the entries that name the repository in the security log of the "
              "account that created it, exported the same day",
              "On 11 October 2026 the repository was transferred from that account to the "
              "first author's and renamed",
              "begins with the 103 events of the first",
              "which archived the repository from its former address before the transfer and "
              "from its present address after it, recorded the same snapshot both times, the "
              "one given in the Data availability statement"):
        assert s in repro, s
    prereg = _flat((SECTIONS / "appendix_prereg.tex").read_text(encoding="utf-8"))
    assert "The repository was created, public, on 14 July 2026 at 03:23 UTC; the security " \
           "log of the account that created it, exported on 10 October 2026, records no " \
           "change of its visibility up to that day." in prereg


def test_the_note_the_readme_and_the_citation_file_agree():
    raw = (ROOT / "PROVENANCE_NOTE.md").read_bytes()
    assert not re.search(rb"[^\r]\n", raw)                 # CRLF throughout, as compiled
    note = _flat(raw.decode("utf-8"))
    for s in ("addendum on its transfer and new name 11 October 2026.",
              "## Addendum, 11 October 2026: the repository's transfer and new name",
              "transferred from HymnOfLight, the account that created it, to RuifengCao, and "
              "renamed from FE-JEPA to label-free-fe-surrogates",
              f"Its id ({REPO_ID}), commit history and tags are unchanged",
              f"{OLD_URL} redirects to {URL}",
              "begins with the 103 events of the export of 10 October 2026, unchanged",
              "from the old address before the transfer and from the new address after it",
              f"the snapshot {SNAPSHOT}", "records/provenance/refs_2026-10-11.txt",
              "The Python package keeps its name, fejepa"):
        assert s in note, s
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# label-free-fe-surrogates\n")
    opening = _flat(readme.split("\n## ", 1)[0])
    for s in ("This repository was called FE-JEPA until 11 October 2026, when it was "
              "transferred from the account that created it to Ruifeng Cao's and renamed",
              f"links to <{OLD_URL}> lead here",
              "Its commit history and tags are unchanged, and the Python package keeps the "
              "name `fejepa`", "[`CITATION.cff`](CITATION.cff)"):
        assert s in opening, s
    assert re.search(r'^name = "fejepa"$', (ROOT / "pyproject.toml").read_text(), re.M)
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert cff.startswith("cff-version: 1.2.0\n")
    for s in ("type: software\n", 'title: "label-free-fe-surrogates"\n',
              "authors:\n  - family-names: Cao\n    given-names: Ruifeng\n"
              '    affiliation: "School of Engineering, The University of Manchester"\n'
              '    orcid: "https://orcid.org/0000-0001-9077-9561"\nrepository-code: ',
              f'repository-code: "{URL}"\n', "license: PolyForm-Noncommercial-1.0.0\n",
              f"doi: {DOI}\n",
              f"  - type: doi\n    value: {DOI}\n"
              f'    description: "Zenodo archive of the version tagged {TAG}"\n',
              f'  - type: swh\n    value: "{SNAPSHOT}"\n'
              '    description: "Software Heritage snapshot of the repository as it stood on '
              '11 October 2026, when it moved to this address"\n'):
        assert s in cff, s
    # the code's author alone: one entry in the list of authors
    block = cff.split("\nauthors:\n", 1)[1].split("\nrepository-code:", 1)[0]
    assert [line for line in block.splitlines() if line.startswith("  - ")] == \
        ["  - family-names: Cao"]
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").splitlines()[2] == \
        "# PolyForm Noncommercial License 1.0.0"
