"""The repository's licences (decided 10 October 2026): LICENSE is the PolyForm
Noncommercial License 1.0.0, verbatim, after the licensor's Required Notice line;
LICENSE-CC-BY-4.0 is the legal code of the Creative Commons Attribution 4.0
International licence, verbatim; the README states which files each licence covers,
and every path it names exists; the manuscript's Data availability statement names
both licences."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# SHA-256 of the official texts: PolyForm-Noncommercial-1.0.0.md of
# github.com/polyformproject/polyform-licenses (commit 76a278c) and text/CC-BY-4.0.txt of
# github.com/spdx/license-list-data (commit 31ba1a5), both fetched on 10 October 2026
POLYFORM_NC_SHA256 = "c0ea4a896d2c8c394b29f9427589996db826cd501c512279ff0ed3ef48fabbe5"
CC_BY_SHA256 = "d557539df68e771cc1eedcc91d13f70fca930e508d11eedcafa4b15db49e3744"
NOTICE = b"Required Notice: Copyright 2026 Ruifeng Cao\n\n"


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_licence_is_polyform_noncommercial_verbatim():
    data = (ROOT / "LICENSE").read_bytes()
    assert data.startswith(NOTICE)
    body = data[len(NOTICE):]
    assert body.startswith(b"# PolyForm Noncommercial License 1.0.0\n")
    assert hashlib.sha256(body).hexdigest() == POLYFORM_NC_SHA256


def test_cc_by_legal_code_is_verbatim():
    data = (ROOT / "LICENSE-CC-BY-4.0").read_bytes()
    assert data.startswith(b"Creative Commons Attribution 4.0 International")
    assert hashlib.sha256(data).hexdigest() == CC_BY_SHA256


def test_readme_states_what_each_licence_covers():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = _flat(readme.split("\n## Licence\n", 1)[1].split("\n## ", 1)[0])
    assert "[PolyForm Noncommercial License 1.0.0](LICENSE)" in section
    assert "Commercial use needs a separate licence" in section
    assert "(LICENSE-CC-BY-4.0)" in section
    for path in ("records/", "wp2_e2_console.txt", "DEVIATIONS_PHASE2.md",
                 "PROVENANCE_NOTE.md", "paper/"):
        assert f"`{path}`" in section, path
        assert (ROOT / path).exists(), path
    assert "`PREREG*.md`" in section and list(ROOT.glob("PREREG*.md"))
    assert "including earlier commits, branches and tags" in section


def test_data_availability_names_both_licences():
    tex = (ROOT / "paper" / "cmame" / "sections" / "declarations.tex").read_text(
        encoding="utf-8")
    statement = _flat(tex.split(r"\section*{Data availability}", 1)[1].split(r"\section*", 1)[0])
    assert "The code is licensed under the PolyForm Noncommercial License 1.0.0" in statement
    assert "licences for commercial use are available from the first author" in statement
    assert "The records and the pre-registration documents are licensed under CC BY 4.0." in (
        statement)
