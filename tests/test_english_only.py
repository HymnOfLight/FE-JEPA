"""The repository is English-only (PI rule, 1 October 2026): no tracked text
file outside records/ contains CJK, kana or Hangul characters. records/ is
exempt because it holds the box's returned files byte for byte (a directory
listing in a box log may show a file name in another script); editing a
record would break its provenance."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# CJK ideographs (incl. extensions and compatibility forms), radicals, CJK punctuation,
# kana, bopomofo, Hangul, and half/full-width forms
CJK = re.compile(r"[\u1100-\u11ff\u2e80-\u2fff\u3000-\u9fff\ua960-\ua97f"
                 r"\uac00-\ud7ff\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef"
                 r"\U00020000-\U0003134f]")
EXEMPT = ("records/",)


def _tracked() -> list:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True,
                             timeout=30, check=True).stdout.decode("utf-8")
        return [f for f in out.split("\0") if f]
    except (OSError, subprocess.SubprocessError):          # not a git checkout: walk it
        skip = {".git", "runs", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
        return [str(p.relative_to(ROOT)) for p in ROOT.rglob("*")
                if p.is_file() and not skip & set(p.relative_to(ROOT).parts)]


def test_no_cjk_outside_records():
    bad = []
    for f in _tracked():
        if f.startswith(EXEMPT):
            continue
        try:
            text = (ROOT / f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue                                       # binary, or deleted in the worktree
        for i, line in enumerate(text.splitlines(), 1):
            if CJK.search(line):
                bad.append(f"{f}:{i}: {line.strip()[:60]}")
    assert not bad, "non-English text in the repository:\n" + "\n".join(bad[:20])
