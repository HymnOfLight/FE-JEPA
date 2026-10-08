#!/usr/bin/env python3
"""cmame-paper: stamp PREREG_CM2D.md (its Sec. 6) in one step, refusing a
file whose status line, CONFIG_SHA256 line or last line is not the approved
draft's, and a configuration that is not the generator's:

  1. configs/cm2d_v1.json is the generator's (scripts/make_cm2d_config.py);
  2. the file is the draft: its one CONFIG_SHA256 line (cm2d_v1) open, the
     footer open, the status line the draft's;
  3. the CONFIG_SHA256 line is filled with the configuration's canonical
     SHA-256 (fejepa.report.stamp_prereg, as `fejepa prereg --stamp` does);
  4. the status line records the stamp date;
  5. the footer records the file's SHA-256, computed over the file with the
     footer reading `PREREG_CM2D_SHA256 = <record after commit>`;
  6. the result is read back and checked.

Then: commit, tag `prereg-cm2d`, push the branch and the tag
(RUNBOOK_CMAME Sec. B fetches both).

    python scripts/stamp_prereg_cm2d.py --date "8 October 2026"
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

LABEL = "cm2d_v1"
DRAFT_STATUS = ("**Status:** r3 DRAFT (8 October 2026), not stamped; the CONFIG_SHA256 line "
                "below is open. Stamping fills it before the run (Sec. 6); post-stamp "
                "criterion changes are prohibited.")
FOOTER_OPEN = "PREREG_CM2D_SHA256 = <record after commit>"
_FOOTER = re.compile(r"^PREREG_CM2D_SHA256 = (.+)$", re.M)
_DATE = re.compile(r"^[1-9]\d? (January|February|March|April|May|June|July|August|September|"
                   r"October|November|December) 20\d\d$")


def stamped_status(date: str) -> str:
    return (f"**Status:** r3, stamped {date} before the run: the CONFIG_SHA256 line below is "
            "filled, and the last line is this file's SHA-256 (Sec. 6); post-stamp criterion "
            "changes are prohibited.")


def _generator():
    spec = importlib.util.spec_from_file_location("make_cm2d_config",
                                                  ROOT / "scripts" / "make_cm2d_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def self_hash_ok(text: str) -> bool:
    m = _FOOTER.findall(text)
    if len(m) != 1 or not re.fullmatch(r"[0-9a-f]{64}", m[0]):
        return False
    blank = _FOOTER.sub(FOOTER_OPEN, text, count=1)
    return hashlib.sha256(blank.encode("utf-8")).hexdigest() == m[0]


def stamp(prereg: Path, config: Path, date: str, check_generator: bool = True) -> dict:
    from fejepa.report import (PREREG_PLACEHOLDER, config_sha256, read_prereg_entries,
                               stamp_prereg)

    if not _DATE.match(date):
        raise SystemExit(f"--date {date!r}: write it as e.g. '8 October 2026'")
    text_cfg = config.read_text()
    if check_generator:
        mk = _generator()
        if text_cfg != mk.render(mk.cm2d_config(mk.load_base())):
            raise SystemExit(f"{config}: not the generator's output (scripts/make_cm2d_config.py)")
    cfg = json.loads(text_cfg)
    text = prereg.read_text(encoding="utf-8")
    entries = read_prereg_entries(prereg)
    if [lab for lab, _ in entries] != [LABEL]:
        raise SystemExit(f"{prereg}: the CONFIG_SHA256 lines are {[lab for lab, _ in entries]}, "
                         f"not [{LABEL!r}]")
    if entries[0][1] != PREREG_PLACEHOLDER:
        raise SystemExit(f"{prereg}: already stamped -- nothing changed")
    if text.count(DRAFT_STATUS) != 1:
        raise SystemExit(f"{prereg}: the status line is not the approved r3 draft's")
    if len(_FOOTER.findall(text)) != 1 or not text.rstrip("\n").endswith(FOOTER_OPEN):
        raise SystemExit(f"{prereg}: the last line is not `{FOOTER_OPEN}`")
    h = stamp_prereg(prereg, cfg, label=LABEL)
    text = prereg.read_text(encoding="utf-8").replace(DRAFT_STATUS, stamped_status(date))
    own = hashlib.sha256(text.encode("utf-8")).hexdigest()
    text = _FOOTER.sub(f"PREREG_CM2D_SHA256 = {own}", text, count=1)
    prereg.write_text(text, encoding="utf-8")
    back = dict(read_prereg_entries(prereg))
    if back.get(LABEL) != config_sha256(cfg) or not \
            self_hash_ok(prereg.read_text(encoding="utf-8")):
        raise SystemExit(f"{prereg}: the stamped file does not verify")
    return {"config_sha256": {LABEL: h}, "prereg_sha256": own, "date": date}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="the stamp date, e.g. '8 October 2026'")
    ap.add_argument("--prereg", default=str(ROOT / "PREREG_CM2D.md"))
    ap.add_argument("--config", default=str(ROOT / "configs" / f"{LABEL}.json"))
    a = ap.parse_args()
    res = stamp(Path(a.prereg), Path(a.config), a.date)
    print(json.dumps(res, indent=1))
    print("next: commit, tag prereg-cm2d, push the branch and the tag")


if __name__ == "__main__":
    main()
