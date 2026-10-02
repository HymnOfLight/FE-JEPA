"""wp9 Stage 0c: the regression records (PREREG_W9 Sec. 9) name the source
tree they ran; the committed `src` must be that tree, so that the record
speaks for the code that is stamped and run."""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          timeout=30)


def test_regression_record_names_this_src_tree():
    readme = (ROOT / "records" / "wp9" / "README.md").read_text()
    m = re.search(r"\| this side \| [^|]*`src` = tree `([0-9a-f]{40})` \|", readme)
    assert m, "records/wp9/README.md names no src tree for this side"
    head = _git("rev-parse", "HEAD:src")
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    if _git("diff", "--quiet", "HEAD", "--", "src").returncode != 0 or \
            _git("ls-files", "--others", "--exclude-standard", "--", "src").stdout.strip():
        pytest.skip("src has local changes: the check applies to committed code")
    assert head.stdout.strip() == m.group(1), (
        "src is not the tree the regression ran: run scripts/regress_against_branch.py "
        "again and update records/wp9/README.md")


def test_regression_summaries_ran_the_trees_named():
    """Each regression summary names the two source trees it ran (git tree ids,
    no machine paths); they are the README's, and the result is identical."""
    readme = (ROOT / "records" / "wp9" / "README.md").read_text()
    want = [re.search(rf"\| {side} \| [^|]*`src` = tree `([0-9a-f]{{40}})` \|", readme)
            for side in ("this side", "the other side")]
    assert all(want), "records/wp9/README.md names no src tree for a side"
    for name in ("regression_wp8_e1_2d_base.json", "regression_wp8_phase2b.json"):
        rec = json.loads((ROOT / "records" / "wp9" / name).read_text())
        assert [rec["this_src_tree"], rec["other_src_tree"]] == [m.group(1) for m in want], name
        assert rec["identical"] is True and rec["n_differences"] == 0, name
        assert not re.search(r"/(home|tmp|root)/", json.dumps(rec)), name
