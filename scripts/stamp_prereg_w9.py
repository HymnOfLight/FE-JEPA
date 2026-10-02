#!/usr/bin/env python3
"""wp9: stamp PREREG_W9.md (its Sec. 8) in one step, refusing anything but
the approved r3 draft:

  1. the six configurations are the generator's (scripts/make_w9_configs.py);
  2. the file is the r3 draft: every CONFIG_SHA256 line open, the footer
     open, the status line the draft's;
  3. each CONFIG_SHA256 line is filled with its configuration's canonical
     SHA-256 (fejepa.report.stamp_prereg, as `fejepa prereg --stamp` does);
  4. the status line records the stamp date;
  5. the footer records the file's SHA-256, computed over the file with the
     footer reading `PREREG_W9_SHA256 = <record after commit>`;
  6. the result is read back and checked: every line is its configuration's
     hash and the footer verifies.

Then: commit, tag `prereg-w9`, push the branch and the tag (RUNBOOK_W9 Sec. 0
fetches both).

    python scripts/stamp_prereg_w9.py --date "9 October 2026"
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DRAFT_STATUS = ("**Status:** r3 DRAFT (2 October 2026), not stamped; the six CONFIG_SHA256 "
                "lines below are open. Stamping fills them all at once, before session 1 "
                "(Sec. 8); post-stamp criterion changes are prohibited.")
FOOTER_OPEN = "PREREG_W9_SHA256 = <record after commit>"
_FOOTER = re.compile(r"^PREREG_W9_SHA256 = (.+)$", re.M)
_DATE = re.compile(r"^[1-9]\d? (January|February|March|April|May|June|July|August|September|"
                   r"October|November|December) 20\d\d$")


def stamped_status(date: str) -> str:
    return (f"**Status:** r3, stamped {date} before session 1: the six CONFIG_SHA256 lines "
            "below are filled, and the last line is this file's SHA-256 (Sec. 8); "
            "post-stamp criterion changes are prohibited.")


def _generator():
    spec = importlib.util.spec_from_file_location("make_w9_configs",
                                                  ROOT / "scripts" / "make_w9_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def self_hash_ok(text: str) -> bool:
    m = _FOOTER.findall(text)
    if len(m) != 1 or not re.fullmatch(r"[0-9a-f]{64}", m[0]):
        return False
    blank = _FOOTER.sub(FOOTER_OPEN, text, count=1)
    return hashlib.sha256(blank.encode("utf-8")).hexdigest() == m[0]


def stamp(prereg: Path, configs: Path, date: str, check_generator: bool = True) -> dict:
    from fejepa.report import (PREREG_PLACEHOLDER, config_sha256, read_prereg_entries,
                               stamp_prereg)

    if not _DATE.match(date):
        raise SystemExit(f"--date {date!r}: write it as e.g. '9 October 2026'")
    mk = _generator()
    arms = list(mk.ARMS)
    cfgs = {}
    for arm in arms:
        f = configs / f"{arm}.json"
        cfgs[arm] = json.loads(f.read_text())
        if check_generator and f.read_text() != mk.render(mk.w9_config(mk.load_base(), arm)):
            raise SystemExit(f"{f}: not the generator's output (scripts/make_w9_configs.py)")
    text = prereg.read_text(encoding="utf-8")
    entries = read_prereg_entries(prereg)
    if [lab for lab, _ in entries] != arms:
        raise SystemExit(f"{prereg}: the CONFIG_SHA256 lines are {[lab for lab, _ in entries]}, "
                         f"not {arms}")
    if any(v != PREREG_PLACEHOLDER for _, v in entries):
        raise SystemExit(f"{prereg}: already (partly) stamped -- nothing changed")
    if text.count(DRAFT_STATUS) != 1:
        raise SystemExit(f"{prereg}: the status line is not the approved r3 draft's")
    if len(_FOOTER.findall(text)) != 1 or not text.rstrip("\n").endswith(FOOTER_OPEN):
        raise SystemExit(f"{prereg}: the last line is not `{FOOTER_OPEN}`")
    hashes = {arm: stamp_prereg(prereg, cfgs[arm], label=arm) for arm in arms}
    text = prereg.read_text(encoding="utf-8").replace(DRAFT_STATUS, stamped_status(date))
    own = hashlib.sha256(text.encode("utf-8")).hexdigest()
    text = _FOOTER.sub(f"PREREG_W9_SHA256 = {own}", text, count=1)
    prereg.write_text(text, encoding="utf-8")
    back = dict(read_prereg_entries(prereg))
    if any(back[a] != config_sha256(cfgs[a]) for a in arms) or not \
            self_hash_ok(prereg.read_text(encoding="utf-8")):
        raise SystemExit(f"{prereg}: the stamped file does not verify")
    return {"config_sha256": hashes, "prereg_sha256": own, "date": date}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="the stamp date, e.g. '9 October 2026'")
    ap.add_argument("--prereg", default=str(ROOT / "PREREG_W9.md"))
    ap.add_argument("--configs", default=str(ROOT / "configs"))
    a = ap.parse_args()
    res = stamp(Path(a.prereg), Path(a.configs), a.date)
    print(json.dumps(res, indent=1))
    print("next: commit, tag prereg-w9, push the branch and the tag")


if __name__ == "__main__":
    main()
