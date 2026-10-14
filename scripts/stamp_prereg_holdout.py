#!/usr/bin/env python3
"""cmame-paper Stage 11: stamp PREREG_HOLDOUT.md (its Sec. 7) in one step,
refusing a file whose status line, CONFIG_SHA256 line or last line is not the
approved draft's, and a configuration that does not pass its own checks:

  1. configs/holdout_v1.json loads (`fejepa.analysis.holdout.load_spec`);
  2. the file is the draft: its one CONFIG_SHA256 line (holdout_v1) open, the
     footer open, the status line the draft's;
  3. the CONFIG_SHA256 line is filled with the configuration's canonical
     SHA-256 (fejepa.report.stamp_prereg, as `fejepa prereg --stamp` does);
  4. the status line records the stamp date;
  5. the footer records the file's SHA-256, computed over the file with the
     footer reading `PREREG_HOLDOUT_SHA256 = <record after commit>`;
  6. the result is read back and checked.

Then: commit, tag `prereg-holdout`, push the branch and the tag
(RUNBOOK_CMAME Sec. E fetches both).

    python scripts/stamp_prereg_holdout.py --date "18 October 2026"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LABEL = "holdout_v1"
TAG = "prereg-holdout"
DRAFT_STATUS = ("**Status:** r4 DRAFT (14 October 2026), not stamped; the CONFIG_SHA256 line "
                "below is open. Stamping fills it before the run (Sec. 7); post-stamp "
                "criterion changes are prohibited.")
FOOTER_OPEN = "PREREG_HOLDOUT_SHA256 = <record after commit>"
_FOOTER = re.compile(r"^PREREG_HOLDOUT_SHA256 = (.+)$", re.M)
_DATE = re.compile(r"^[1-9]\d? (January|February|March|April|May|June|July|August|September|"
                   r"October|November|December) 20\d\d$")


def stamped_status(date: str) -> str:
    return (f"**Status:** r4, stamped {date} before the run: the CONFIG_SHA256 line below is "
            "filled, and the last line is this file's SHA-256 (Sec. 7); post-stamp criterion "
            "changes are prohibited.")


def self_hash_ok(text: str) -> bool:
    m = _FOOTER.findall(text)
    if len(m) != 1 or not re.fullmatch(r"[0-9a-f]{64}", m[0]):
        return False
    blank = _FOOTER.sub(FOOTER_OPEN, text, count=1)
    return hashlib.sha256(blank.encode("utf-8")).hexdigest() == m[0]


def stamp(prereg: Path, config: Path, date: str) -> dict:
    from fejepa.analysis.holdout import load_spec
    from fejepa.report import (PREREG_PLACEHOLDER, config_sha256, read_prereg_entries,
                               stamp_prereg)

    if not _DATE.match(date):
        raise SystemExit(f"--date {date!r}: write it as e.g. '18 October 2026'")
    try:
        spec = load_spec(config)
    except ValueError as exc:
        raise SystemExit(f"{config}: {exc}") from exc
    if json.loads(config.read_text()) != spec:
        raise SystemExit(f"{config}: does not load as written")
    text = prereg.read_text(encoding="utf-8")
    entries = read_prereg_entries(prereg)
    if [lab for lab, _ in entries] != [LABEL]:
        raise SystemExit(f"{prereg}: the CONFIG_SHA256 lines are {[lab for lab, _ in entries]}, "
                         f"not [{LABEL!r}]")
    if entries[0][1] != PREREG_PLACEHOLDER:
        raise SystemExit(f"{prereg}: already stamped -- nothing changed")
    if text.count(DRAFT_STATUS) != 1:
        raise SystemExit(f"{prereg}: the status line is not the approved r4 draft's")
    if len(_FOOTER.findall(text)) != 1 or not text.rstrip("\n").endswith(FOOTER_OPEN):
        raise SystemExit(f"{prereg}: the last line is not `{FOOTER_OPEN}`")
    h = stamp_prereg(prereg, spec, label=LABEL)
    text = prereg.read_text(encoding="utf-8").replace(DRAFT_STATUS, stamped_status(date))
    own = hashlib.sha256(text.encode("utf-8")).hexdigest()
    text = _FOOTER.sub(f"PREREG_HOLDOUT_SHA256 = {own}", text, count=1)
    prereg.write_text(text, encoding="utf-8")
    back = dict(read_prereg_entries(prereg))
    if back.get(LABEL) != config_sha256(spec) or not \
            self_hash_ok(prereg.read_text(encoding="utf-8")):
        raise SystemExit(f"{prereg}: the stamped file does not verify")
    return {"config_sha256": {LABEL: h}, "prereg_sha256": own, "date": date}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="the stamp date, e.g. '18 October 2026'")
    ap.add_argument("--prereg", default=str(ROOT / "PREREG_HOLDOUT.md"))
    ap.add_argument("--config", default=str(ROOT / "configs" / f"{LABEL}.json"))
    a = ap.parse_args()
    res = stamp(Path(a.prereg), Path(a.config), a.date)
    print(json.dumps(res, indent=1))
    print(f"next: commit, tag {TAG}, push the branch and the tag")


if __name__ == "__main__":
    main()
