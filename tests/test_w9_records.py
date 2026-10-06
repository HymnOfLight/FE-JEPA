"""wp9 Stage 0c: the regression records (PREREG_W9 Sec. 9) name the source
tree they ran; the committed `src` must be that tree, so that the record
speaks for the code that is stamped and run.

cmame-paper Stage 2 changed `src` and ran the regression again
(`records/cmame/`): the committed `src` must be the tree named by the newest
record, and every record must still match its own summaries."""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORD_DIRS = ("records/cmame", "records/wp9")
"""The regression records, newest first."""
SUMMARIES = ("regression_wp8_e1_2d_base.json", "regression_wp8_phase2b.json")


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          timeout=30)


def _tree(readme: str, side: str):
    m = re.search(rf"\| {side} \| [^|]*`src` = tree `([0-9a-f]{{40}})` \|", readme)
    return m.group(1) if m else None


def test_regression_record_names_this_src_tree():
    newest = next((d for d in RECORD_DIRS if (ROOT / d / "README.md").is_file()), None)
    assert newest, f"none of {RECORD_DIRS} holds a README"
    want = _tree((ROOT / newest / "README.md").read_text(), "this side")
    assert want, f"{newest}/README.md names no src tree for this side"
    head = _git("rev-parse", "HEAD:src")
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    if _git("diff", "--quiet", "HEAD", "--", "src").returncode != 0 or \
            _git("ls-files", "--others", "--exclude-standard", "--", "src").stdout.strip():
        pytest.skip("src has local changes: the check applies to committed code")
    assert head.stdout.strip() == want, (
        f"src is not the tree the newest regression ({newest}) ran: run "
        f"scripts/regress_against_branch.py again and update {newest}/README.md")


@pytest.mark.parametrize("records", RECORD_DIRS)
def test_regression_summaries_ran_the_trees_named(records):
    """Each regression summary names the two source trees it ran (git tree ids,
    no machine paths); they are its README's, and the result is identical."""
    readme = (ROOT / records / "README.md").read_text()
    want = [_tree(readme, side) for side in ("this side", "the other side")]
    assert all(want), f"{records}/README.md names no src tree for a side"
    for name in SUMMARIES:
        rec = json.loads((ROOT / records / name).read_text())
        assert [rec["this_src_tree"], rec["other_src_tree"]] == want, name
        assert rec["identical"] is True and rec["n_differences"] == 0, name
        assert not re.search(r"/(home|tmp|root)/", json.dumps(rec)), name
