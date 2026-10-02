#!/usr/bin/env python3
"""CLI: the PREREG_W9 verdicts (H1, H2) and secondary readings, from the two
box returns (see fejepa.analysis.adjudicate_w9).

    python scripts/adjudicate_w9.py --session1 <session-1 return> \
        --session2 <session-2 return> --out records/wp9/w9_verdict.json

The session-1 return holds the readings, `decisions.json`, `ood2d.json` and
`ood2d/<family>/manifest.json` (RUNBOOK_W9 Sec. 1f); a reading the session
could not produce is simply absent (its rule was undecided). The session-2
return must hold `plan.json`, `status.txt`, `provenance.txt` and one directory
per arm with its `report.json` (Sec. 2d): a report the provenance file lists
must be there with the SHA-256 listed, and an arm the status file shows
finished must have its report. E1's base report is the records copy, refused
unless its configuration is PREREG_E1's stamped e1_2d_base. The decisions are
recomputed with the frozen rules (`scripts/w9_session1_decisions.py`, whose
SHA-256 they must record); every report and the decisions must have run on
`prereg-w9`; each report's verified hash must be PREREG_W9's line for its arm.
The verdict file records every input's SHA-256 and the adjudicating code's."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

E1_BASE_CONFIG_SHA256 = "4dfdea42e22c0cc45c113f11408723953e29e114ec8f94668b447c2a1740a52d"
ROLE_DIRS = {"n1024": ("c1_n1024",), "nmax": ("c1_n25600", "c1_n12800"),
             "n4096": ("c1_n4096",), "b1024": ("b_n1024",), "s": ("s_n1024",)}
SESSION1_FILES = {"amp2d": "c0_amp2d.json", "timing": "profile_2d_w9.json",
                  "memory": "c0_memory.json", "trainval": "c0_trainval.json",
                  "val": "c0_val.json"}
SESSION2_REQUIRED = ("plan.json", "status.txt", "provenance.txt")


def _rules():
    path = ROOT / "scripts" / "w9_session1_decisions.py"
    spec = importlib.util.spec_from_file_location("w9_rules", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decide, path


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unavailable"
    except Exception:                                     # noqa: BLE001
        return "unavailable"


def find_reports(s2: Path) -> dict:
    out = {}
    for role, names in ROLE_DIRS.items():
        found = [s2 / n / "report.json" for n in names if (s2 / n / "report.json").is_file()]
        if len(found) > 1:
            raise SystemExit(f"{s2}: more than one {role} report: {found}")
        if found:
            out[role] = found[0]
    return out


def session1_readings(s1: Path, file_sha256) -> tuple:
    """(inputs by argument name, SHA-256 by file name) as the decision script
    reads them: a missing file is absent from the hashes and None as input; a
    malformed one is hashed and None."""
    inputs, sha = {}, {}
    for k, f in SESSION1_FILES.items():
        p = s1 / f
        inputs[k] = None
        if p.is_file():
            sha[f] = file_sha256(p)
            try:
                inputs[k] = json.loads(p.read_text())
            except ValueError:
                pass
    return inputs, sha


def provenance_reports(text: str) -> dict:
    """{arm directory: SHA-256} of the reports RUNBOOK 2d's provenance lists."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+(?:\S*/)?runs/w9/([^/\s]+)/report\.json$", line.strip())
        if m:
            out[m.group(2)] = m.group(1)
    return out


def main() -> None:
    from fejepa.analysis.adjudicate_w9 import adjudicate_w9, file_sha256, load_json
    from fejepa.report import config_sha256, read_prereg_entries

    ap = argparse.ArgumentParser()
    ap.add_argument("--session1", required=True, help="the session-1 return directory")
    ap.add_argument("--session2", required=True, help="the session-2 return directory")
    ap.add_argument("--e1-report", default=str(ROOT / "records/wp8/e1/e1_2d_base/report.json"))
    ap.add_argument("--prereg", default=str(ROOT / "PREREG_W9.md"))
    ap.add_argument("--expected-git", default="prereg-w9",
                    help="the git describe every run must record")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s1, s2 = Path(a.session1), Path(a.session2)
    e1 = load_json(a.e1_report)
    if config_sha256(e1["config"]) != E1_BASE_CONFIG_SHA256:
        raise SystemExit(f"{a.e1_report}: not E1's stamped base run")
    entries = dict(read_prereg_entries(a.prereg))
    if any(v.startswith("<") for v in entries.values()):
        raise SystemExit(f"{a.prereg}: not stamped")
    missing = [f for f in SESSION2_REQUIRED if not (s2 / f).is_file()]
    if missing:
        raise SystemExit(f"{s2}: the session-2 return lacks {missing} (RUNBOOK_W9 Sec. 2d)")
    decide, rules_path = _rules()
    decisions = load_json(s1 / "decisions.json")
    decisions["_sha256"] = file_sha256(s1 / "decisions.json")
    inputs, input_sha = session1_readings(s1, file_sha256)
    report_paths = find_reports(s2)
    listed = provenance_reports((s2 / "provenance.txt").read_text())
    for arm, sha in sorted(listed.items()):
        p = s2 / arm / "report.json"
        if not p.is_file():
            raise SystemExit(f"{s2}: the provenance file lists {arm}'s report, the return lacks it")
        if file_sha256(p) != sha:
            raise SystemExit(f"{p}: not the report the box hashed (provenance.txt)")
    reports = {r: load_json(p) for r, p in report_paths.items()}
    r_manifest = s1 / "ood2d" / "R" / "manifest.json"
    try:
        res = adjudicate_w9(
            reports, e1, decisions, load_json(s1 / "ood2d.json"),
            r_manifest=load_json(r_manifest), r_manifest_sha256=file_sha256(r_manifest),
            prereg_entries=entries, expected_git=a.expected_git,
            e1_report_sha256=file_sha256(a.e1_report), session1_inputs=inputs,
            session1_sha256=input_sha, decide=decide, rules_sha256=file_sha256(rules_path),
            plan=load_json(s2 / "plan.json"), status_text=(s2 / "status.txt").read_text(),
            returned_arms={p.parent.name for p in report_paths.values()})
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    res["inputs"] = {**{r: {"path": str(p), "sha256": file_sha256(p)}
                        for r, p in report_paths.items()},
                     "decisions": {"path": str(s1 / "decisions.json"),
                                   "sha256": decisions["_sha256"]},
                     "session1_readings": input_sha,
                     "ood2d_record": {"path": str(s1 / "ood2d.json"),
                                      "sha256": file_sha256(s1 / "ood2d.json")},
                     "r_manifest": {"path": str(r_manifest), "sha256": file_sha256(r_manifest)},
                     **{f: {"path": str(s2 / f), "sha256": file_sha256(s2 / f)}
                        for f in SESSION2_REQUIRED},
                     "e1_report": {"path": a.e1_report, "sha256": file_sha256(a.e1_report)},
                     "prereg": {"path": a.prereg, "sha256": file_sha256(a.prereg)}}
    res["adjudicator"] = {
        "git": _git("describe", "--always", "--dirty", "--tags"),
        "sha256": {p: file_sha256(ROOT / p) for p in (
            "src/fejepa/analysis/adjudicate_w9.py", "src/fejepa/analysis/adjudicate.py",
            "scripts/adjudicate_w9.py", "scripts/w9_session1_decisions.py")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"H1": res["H1"]["verdict"], "H1_rel_change": res["H1"].get("rel_change"),
                      "H1_threshold": res["H1"].get("threshold"), "H2": res["H2"]["verdict"],
                      "refused_reports": res["refused_reports"],
                      "deviations": res["deviations"]}, indent=1))


if __name__ == "__main__":
    main()
