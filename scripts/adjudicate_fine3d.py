#!/usr/bin/env python3
"""fine3d: adjudicate PREREG_FINE3D (Sec. 8) from the box's return.

R2a reads its two reports and, for the mechanism readings, the post-hoc
amplitude records; R3 reads its two reports (the energy arm and the labelled
arm, its reference). The Phase-2b report and E2's
M = 1,024 report are the records copies, each checked by SHA-256; the stamped
CONFIG_SHA256 lines are read from PREREG_FINE3D.md, and every report must have
run on the tag. A part whose report is not given, not in the return or not
readable is NOT EVALUATED; the other is still adjudicated (an unreadable
amplitude record leaves the mechanism readings unevaluated). The verdict file
records the inputs' SHA-256, the session's attempts (with `--status`), and the
adjudicating code: its `git describe`, its files' SHA-256 and whether they and
the pre-registration read are the tag's. A difference, a tag that cannot be
read, a footer that does not verify, a status file not given or not in the
return, repeated or missing steps are deviations; "deviations" holds them with R2a's and R3's own.

    python scripts/adjudicate_fine3d.py \
        --r2a-b <return>/r2a_b_m1024/report.json --r2a-s <return>/r2a_s_m1024/report.json \
        --amp-b <return>/posthoc/amp_r2a_b_m1024.json --amp-s <return>/posthoc/amp_r2a_s_m1024.json \
        --r3 <return>/r3_eft_b16/report.json --r3-labelled <return>/r3_lft_b16/report.json \
        --status <return>/session/status.txt \
        --out records/wp9/fine3d/fine3d_verdict.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PHASE2B = "records/wp8/e2/baseline/report_phase2b.json"
E2_M1024 = "records/wp8/e2/e2_m1024/report.json"
E2_M1024_SHA256 = "3bcd56f14c74ac2a27dfa37e34992b552249da7f6b5e4d4d9d6c6a945f6d3e4e"
"""The records copy of E2's M = 1,024 report (wp8 Stage 1.35, from the E2
return package; tests/test_e_series_verdicts.py ties it to its stamp line)."""
ADJ_FILES = ("src/fejepa/analysis/adjudicate_fine3d.py", "src/fejepa/analysis/adjudicate_w9.py",
             "src/fejepa/analysis/adjudicate.py", "src/fejepa/metrics.py",
             "src/fejepa/report.py", "scripts/adjudicate_fine3d.py",
             "scripts/stamp_prereg_fine3d.py", "configs/r2a_s_m1024.json")
"""The adjudicating code and what it reads here (S's factor): compared with
the tag's copies; the pre-registration read is compared with the tag's
PREREG_FINE3D.md."""
STEPS = ("r2a_b_m1024", "r2a_s_m1024", "amp_r2a_b_m1024", "amp_r2a_s_m1024", "r3_eft_b16",
         "r3_lft_b16")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stamp_mod():
    spec = importlib.util.spec_from_file_location("stamp_prereg_fine3d",
                                                  ROOT / "scripts" / "stamp_prereg_fine3d.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _git_show(root: Path, tag: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(root), "show", f"{tag}:{path}"], capture_output=True,
                       timeout=60)
    return r.stdout if r.returncode == 0 else None


def adjudicator_record(prereg: Path, tag: str, root: Path = ROOT) -> dict:
    """The adjudicating files and the pre-registration read, against the
    tag's copies in the repository at `root`."""
    from fejepa.report import _git_describe

    files = {f: _sha256(root / f) for f in ADJ_FILES}
    tagged = {f: _git_show(root, tag, f) for f in (*ADJ_FILES, "PREREG_FINE3D.md")}
    available = all(v is not None for v in tagged.values())
    text = Path(prereg).read_text(encoding="utf-8")
    return {"git": _git_describe(), "files_sha256": files, "tag": tag,
            "tag_available": available,
            "differs_from_tag": ([f for f in ADJ_FILES
                                  if hashlib.sha256(tagged[f]).hexdigest() != files[f]]
                                 if available else None),
            "prereg_file": str(prereg), "prereg_sha256": _sha256(prereg),
            "prereg_is_tags": (hashlib.sha256(tagged["PREREG_FINE3D.md"]).hexdigest()
                               == _sha256(prereg)) if available else None,
            "prereg_is_repository_file":
                Path(prereg).resolve() == (root / "PREREG_FINE3D.md").resolve(),
            "prereg_self_hash_ok": bool(_stamp_mod().self_hash_ok(text))}


def code_deviations(rec: dict, prereg) -> list:
    """The deviations the adjudicator's own record shows."""
    tag = rec["tag"]
    if not rec["tag_available"]:
        dev = [f"the adjudicating code and the pre-registration read were not compared with "
               f"{tag} (the tag cannot be read here)"]
    else:
        dev = ([f"the adjudicating files differ from {tag}: {rec['differs_from_tag']}"]
               if rec["differs_from_tag"] else [])
        if not rec["prereg_is_tags"]:
            dev.append(f"{prereg}: the pre-registration read is not {tag}'s PREREG_FINE3D.md")
    if not rec["prereg_self_hash_ok"]:
        dev.append(f"{prereg}: its last line does not verify (PREREG_FINE3D_SHA256)")
    return dev


def strict(o):
    """The verdict file as strict JSON: a non-finite float becomes null (the
    verdicts and readings that read it say so in words)."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: strict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [strict(v) for v in o]
    return o


def attempts(status_text: str) -> dict:
    """{step: number of attempts} from the session's status file."""
    out = {}
    for line in status_text.splitlines():
        m = re.match(r"(\S+)\.log start ", line.strip())
        if m:
            out[m.group(1)] = out.get(m.group(1), 0) + 1
    return out


def main() -> None:
    from fejepa.analysis.adjudicate_fine3d import (PHASE2B_REPORT_SHA256, TAG,
                                                   adjudicate_fine3d, load_json)
    from fejepa.report import PREREG_PLACEHOLDER, read_prereg_entries

    ap = argparse.ArgumentParser()
    ap.add_argument("--r2a-b", default=None)
    ap.add_argument("--r2a-s", default=None)
    ap.add_argument("--amp-b", default=None)
    ap.add_argument("--amp-s", default=None)
    ap.add_argument("--r3", default=None, help="R3's energy arm (r3_eft_b16)")
    ap.add_argument("--r3-labelled", default=None, help="R3's labelled arm (r3_lft_b16)")
    ap.add_argument("--status", default=None, help="the session's status.txt (attempts)")
    ap.add_argument("--phase2b", default=str(ROOT / PHASE2B))
    ap.add_argument("--e2", default=str(ROOT / E2_M1024))
    ap.add_argument("--prereg", default=str(ROOT / "PREREG_FINE3D.md"))
    ap.add_argument("--expected-git", default=TAG,
                    help="the git describe every report must carry ('' = not checked)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if _sha256(a.phase2b) != PHASE2B_REPORT_SHA256:
        raise SystemExit(f"{a.phase2b}: not the Phase-2b report")
    if _sha256(a.e2) != E2_M1024_SHA256:
        raise SystemExit(f"{a.e2}: not E2's M = 1,024 report")
    entries = dict(read_prereg_entries(a.prereg))
    if any(v == PREREG_PLACEHOLDER for v in entries.values()):
        raise SystemExit(f"{a.prereg}: not stamped")
    s_factor = json.loads((ROOT / "configs" / "r2a_s_m1024.json").read_text())["model"][
        "decode_scale_factor"]

    def present(p):
        return p if p and Path(p).is_file() else None

    def read(p):
        """(the parsed file, None), or (None, why it cannot be read)."""
        try:
            return load_json(p), None
        except Exception as exc:                              # noqa: BLE001
            return None, f"{p} is not readable ({type(exc).__name__}: {exc})"

    def amp(p):
        if not present(p):
            return None
        rec, why = read(p)
        return rec if why is None else {"unreadable": why}

    def absent(*given):
        """What is missing, from (flag, path, present) triples."""
        return ", ".join(f"{p} is not in the return" if p else f"{flag} was not given"
                         for flag, p, ok in given if not ok)

    missing = {}
    r2a = None
    b, s = present(a.r2a_b), present(a.r2a_s)
    if b and s:
        (rb, wb), (rs, ws) = read(b), read(s)
        if wb or ws:
            missing["R2a"] = "; ".join(w for w in (wb, ws) if w)
        else:
            r2a = {"b": rb, "s": rs, "b_sha256": _sha256(b), "s_sha256": _sha256(s),
                   "amp_b": amp(a.amp_b), "amp_s": amp(a.amp_s)}
    else:
        missing["R2a"] = "R2a needs both reports: " + absent(("--r2a-b", a.r2a_b, b),
                                                              ("--r2a-s", a.r2a_s, s))
    r3 = None
    e, lab = present(a.r3), present(a.r3_labelled)
    if e and lab:
        (re_, we), (rl, wl) = read(e), read(lab)
        if we or wl:
            missing["R3"] = "; ".join(w for w in (we, wl) if w)
        else:
            r3 = {"energy": re_, "labelled": rl}
    else:
        missing["R3"] = "R3 needs both arms' reports: " + absent(
            ("--r3", a.r3, e), ("--r3-labelled", a.r3_labelled, lab))
    res = adjudicate_fine3d(r2a, r3, load_json(a.phase2b), load_json(a.e2), entries,
                            a.expected_git or None, s_factor, missing)
    rec = adjudicator_record(Path(a.prereg), TAG)
    res["adjudicator"] = rec
    dev = code_deviations(rec, a.prereg)
    if a.status:
        if present(a.status):
            att = attempts(Path(a.status).read_text())
            res["attempts"] = att
            dev += [f"{k}: {v} attempts" for k, v in att.items() if v > 1]
            dev += [f"{k}: not started" for k in STEPS if k not in att]
        else:
            res["attempts"] = None
            dev.append(f"the session's status file is not in the return ({a.status})")
    else:
        res["attempts"] = None
        dev.append("the session's status file was not given (--status)")
    res["session_and_code_deviations"] = dev
    res["deviations"] = (dev + list(res["R2a"].get("deviations") or [])
                         + list(res["R3"].get("deviations") or []))
    res["notes"] = list(res["R3"].get("notes") or [])
    res["inputs"] = {str(p): _sha256(p) for p in (a.r2a_b, a.r2a_s, a.amp_b, a.amp_s, a.r3,
                                                  a.r3_labelled,
                                                  a.status, a.phase2b, a.e2, a.prereg)
                     if present(p)}
    text = json.dumps(strict(res), indent=1, allow_nan=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    summary = {"R2a": res["R2a"].get("verdict"), "R2b": res["R2a"].get("r2b"),
               "R3": {k: v.get("verdict") for k, v in res["R3"]["verdicts"].items()},
               "deviations": res["deviations"], "notes": res["notes"]}
    print(json.dumps(strict(summary), indent=1, allow_nan=False))


if __name__ == "__main__":
    main()
