#!/usr/bin/env python3
"""CLI: independent audit of a Phase-2 report (see fejepa.analysis.audit).

    python scripts/audit_phase2_report.py runs/phase2/report_phase2.json \
        --expect-config-sha e3bdd1e8... --expect-ledger 1280 \
        [--expect-dataset-sha ffaa... --expect-dataset-sha aae3...] \
        [--ar-sha-file ar_states.sha256]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--expect-config-sha", default=None)
    ap.add_argument("--expect-ledger", type=int, default=None)
    ap.add_argument("--expect-git-prefix", default="prereg-phase2")
    ap.add_argument("--expect-dataset-sha", action="append", default=[])
    ap.add_argument("--ar-sha-file", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from fejepa.analysis.audit import AuditExpectations, audit
    from fejepa.analysis.common import write_json

    res = audit(json.loads(Path(a.report).read_text()),
                AuditExpectations(config_sha=a.expect_config_sha, git_prefix=a.expect_git_prefix,
                                  dataset_shas=a.expect_dataset_sha, ledger_total=a.expect_ledger,
                                  ar_sha_file=a.ar_sha_file))
    for c in res["checks"]:
        print(("PASS " if c["ok"] else "FAIL "), c["check"],
              ("-- " + c["detail"]) if c["detail"] and not c["ok"] else "")
    print("derived:", json.dumps(res["derived"]))
    print("ALL OK" if res["all_ok"] else "DISCREPANCIES PRESENT")
    if a.out:
        write_json(a.out, res)


if __name__ == "__main__":
    main()
