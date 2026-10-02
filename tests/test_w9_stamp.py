"""wp9 Stage 0c: scripts/stamp_prereg_w9.py stamps PREREG_W9.md exactly as
its Sec. 8 says -- before the stamp, on a copy of the draft; after it, the
committed file must be what the script makes from the draft it came from."""

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("stamp_prereg_w9",
                                                  ROOT / "scripts" / "stamp_prereg_w9.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _draft(text: str, mod) -> tuple:
    """(the r3 draft, the stamp date or None) of PREREG_W9.md's text."""
    from fejepa.report import PREREG_PLACEHOLDER

    m = re.search(r"\*\*Status:\*\* r3, stamped (.+?) before session 1: .*?prohibited\.", text)
    if not m:
        return text, None
    text = text.replace(m.group(0), mod.DRAFT_STATUS)
    text = re.sub(r"^(CONFIG_SHA256\[[^\]]+\] = )[0-9a-f]{64}$", rf"\g<1>{PREREG_PLACEHOLDER}",
                  text, flags=re.M)
    text = re.sub(r"^PREREG_W9_SHA256 = [0-9a-f]{64}$", mod.FOOTER_OPEN, text, flags=re.M)
    return text, m.group(1)


def test_stamping_the_draft(tmp_path):
    from fejepa.report import config_sha256, read_prereg_entries

    mod = _mod()
    committed = (ROOT / "PREREG_W9.md").read_text(encoding="utf-8")
    draft, date = _draft(committed, mod)
    pf = tmp_path / "PREREG_W9.md"
    pf.write_text(draft, encoding="utf-8")
    res = mod.stamp(pf, ROOT / "configs", date or "9 October 2026")
    text = pf.read_text(encoding="utf-8")
    for label, value in read_prereg_entries(pf):
        cfg = json.loads((ROOT / "configs" / f"{label}.json").read_text())
        assert value == config_sha256(cfg) == res["config_sha256"][label]
    assert mod.self_hash_ok(text) and text.rstrip("\n").endswith(res["prereg_sha256"])
    assert mod.stamped_status(date or "9 October 2026") in text and mod.DRAFT_STATUS not in text
    assert len(text.splitlines()) == len(draft.splitlines())
    if date is not None:                                  # stamped: byte for byte
        assert text == committed
    with pytest.raises(SystemExit, match="already"):
        mod.stamp(pf, ROOT / "configs", "9 October 2026")


def test_stamping_refuses_anything_but_the_approved_draft(tmp_path):
    mod = _mod()
    draft, _ = _draft((ROOT / "PREREG_W9.md").read_text(encoding="utf-8"), mod)
    pf = tmp_path / "PREREG_W9.md"
    pf.write_text(draft, encoding="utf-8")
    with pytest.raises(SystemExit, match="--date"):
        mod.stamp(pf, ROOT / "configs", "2026-10-09")
    cfgs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", cfgs)
    c = json.loads((cfgs / "w9_c1_n4096.json").read_text())
    c["pretrain"]["lr"] = 5e-4
    (cfgs / "w9_c1_n4096.json").write_text(json.dumps(c, indent=1) + "\n")
    with pytest.raises(SystemExit, match="not the generator's output"):
        mod.stamp(pf, cfgs, "9 October 2026")
    pf.write_text(draft.replace("r3 DRAFT (2 October 2026)", "r4 DRAFT (9 October 2026)"),
                  encoding="utf-8")
    with pytest.raises(SystemExit, match="not the approved r3 draft"):
        mod.stamp(pf, ROOT / "configs", "9 October 2026")
    assert pf.read_text(encoding="utf-8").count("<fill before tagging>") == 6   # untouched
