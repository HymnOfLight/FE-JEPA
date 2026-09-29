"""wp8-lejepa Stage 1.34: the E-series stamp rests on committed records.

PREREG_E1 and PREREG_E2 were stamped from the returned lambda pilot and E2
bench (`records/wp8/`). These tests keep the stamp, the records and the
configurations one fact: the pilot record is the file PREREG_E1's parameter
line names and still yields that lambda / width under the generator's checks;
the bench records are the files PREREG_E2 names and pass the E2 adjudicator's
bench checks at both M; every CONFIG_SHA256 line is the committed
configuration's hash; and each file's footer SHA-256 matches its text, so an
edit after the stamp turns the suite red (a new revision re-stamps)."""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "records" / "wp8"
ARMS = {"PREREG_E1.md": ("e1_2d_base", "e1_2d_shaped", "e1_2d_raw_s0"),
        "PREREG_E2.md": ("e2_m512", "e2_m1024")}


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_pilot_record_is_the_one_prereg_e1_names_and_yields_its_parameters():
    from scripts.make_e_series_configs import PREREG_E1_PARAMS, read_pilot

    p1 = ROOT / "configs" / "phase1_rec8_v2.json"
    lam, width, sha = read_pilot(str(REC / "e1_pilot.json"), json.loads(p1.read_text()), _sha(p1))
    m = PREREG_E1_PARAMS.search((ROOT / "PREREG_E1.md").read_text(encoding="utf-8"))
    assert m is not None
    assert (float(m["lam"]), int(m["width"]), m["sha"]) == (lam, width, sha)
    assert sha == _sha(REC / "e1_pilot.json")
    sh = json.loads((ROOT / "configs" / "e1_2d_shaped.json").read_text())["pretrain"]["loss_spec"]
    assert (sh["lambda_reg"], sh["sigreg_head_width"]) == (lam, width)


def test_bench_records_are_the_ones_prereg_e2_names_and_pass_the_adjudicator():
    from fejepa.analysis.adjudicate import PREREG_E2_DECODE_K, bench_fine_step_s

    text = (ROOT / "PREREG_E2.md").read_text(encoding="utf-8")
    for m in (512, 1024):
        f = REC / f"bench_e2_m{m}.json"
        assert _sha(f) in text, f.name
        bench = json.loads(f.read_text())
        s, phase = bench_fine_step_s(bench, m, PREREG_E2_DECODE_K)
        assert s == phase["ms_per_step"] / 1000.0 > 0.0
        inband = bench["phases"][f"bottleneck{m}_inband_0"]
        assert inband["valid"] is True and inband["decode_k"] == PREREG_E2_DECODE_K


def test_every_stamp_line_is_the_committed_configuration():
    from fejepa.report import config_sha256, read_prereg_entries

    for name, arms in ARMS.items():
        ents = read_prereg_entries(ROOT / name)
        assert sorted(lab for lab, _ in ents) == sorted(arms), name
        for lab, val in ents:
            cfg = json.loads((ROOT / "configs" / f"{lab}.json").read_text())
            assert val == config_sha256(cfg), (name, lab)


@pytest.mark.parametrize("name,key", [("PREREG_E1.md", "PREREG_E1_SHA256"),
                                      ("PREREG_E2.md", "PREREG_E2_SHA256")])
def test_footer_sha256_matches_the_stamped_text(name, key):
    """The footer records the file's SHA-256 with the footer line reading
    `<record after commit>` (PREREG Sec. 5/6: a file cannot hold its own hash)."""
    text = (ROOT / name).read_text(encoding="utf-8")
    hits = re.findall(rf"^{key} = (\S+)$", text, flags=re.M)
    assert len(hits) == 1 and re.fullmatch(r"[0-9a-f]{64}", hits[0]), hits
    blank = text.replace(f"{key} = {hits[0]}", f"{key} = <record after commit>")
    assert hashlib.sha256(blank.encode("utf-8")).hexdigest() == hits[0]
