#!/usr/bin/env python3
"""cmame-paper Stage 11 (PREREG_HOLDOUT Sec. 2): how deep into each corpus's
pool every recorded run reached, read from the reports committed on every
branch of this repository, and how deep the recorded tools outside the runs
read.

A run reads its corpus through the seeded split (validation first, then the
pool in permutation order) and touches a prefix of the pool only; the
deepest prefix it touches is `fejepa.experiments.runner._pool_need` of its
configuration, and every report records the depth it used
(`data_economy.unlabeled_pool_depth_used`, which the runner sets from
`_pool_need`: the two agree by construction, and are listed side by side
as a check of the scan, not as two lines of evidence). This script lists,
for every report at the head of every branch (`git ls-tree` of the remote
branches, read from git objects, not from a working tree), its corpus,
split, recorded depth and the depth its configuration implies, and per
corpus the deepest (every JSON file with a configuration and results is a
report); it scans every JSON file reachable from any remote branch or tag
as well (`git rev-list --objects` of each, older commits included; the refs
are recorded) and records whether that finds other reports. It also reads
the depths that the recorded tools outside the runs reached
(`OTHER_READS`). The holdout set of a corpus starts after the deepest of
these (configs/holdout_v1.json `used_prefix`).

    python scripts/holdout_pool_depths.py --out records/cmame/holdout/pool_depths.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _precheck_prefix(d: dict) -> int:
    import re

    return int(re.search(r"pool prefix (\d+)", d["checks"]["labels"]["detail"]).group(1))


OTHER_READS = (
    # branch, record, the tool and what it reads, corpus, the depth from the record
    ("cmame-paper", "records/wp8/e1_pilot.json",
     "E1's lambda pilot (scripts/e1_lambda_pilot.py): trains on pool[:n_train] and selects "
     "on pool[n_train:n_train + n_val]", "runs/data2d", lambda d: d["n_train"] + d["n_val"]),
    ("cmame-paper", "records/wp8/posthoc/profile_2d_head.json",
     "2D step timing (scripts/posthoc_profile_2d.py): pool[:n_inst]", "runs/data2d",
     lambda d: d["n_inst"]),
    ("cmame-paper", "records/wp8/posthoc/profile_2d_v215.json",
     "2D step timing (scripts/posthoc_profile_2d.py): pool[:n_inst]", "runs/data2d",
     lambda d: d["n_inst"]),
    ("wp9-pool", "records/wp9/session1/profile_2d_w9.json",
     "2D step timing (scripts/posthoc_profile_2d.py): pool[:n_inst]", "runs/data2d",
     lambda d: d["n_inst"]),
    ("wp9-pool", "records/wp9/session1/c0_memory.json",
     "wp9 C0.6 memory audit (scripts/w9_c0.py): loads pool[:n] at each mark, no training",
     "runs/data2d", lambda d: max(int(m["n"]) for m in d["marks"])),
    ("wp9-pool", "records/wp9/session1/c0_trainval.json",
     "wp9 C0.4 training-set error (scripts/w9_c0.py): E1's states on pool[:n]",
     "runs/data2d", lambda d: int(d["train"]["n_instances"])),
    ("cmame-paper", "records/cmame/cm2d/return/precheck.json.used",
     "CM2D's precheck (scripts/cm2d_precheck.py): checks the labels of pool[:n], two "
     "training steps on pool[:2]", "runs/data2d", _precheck_prefix),
)
"""Records of tools that read a corpus's pool outside the runs' reports
(PREREG_HOLDOUT Sec. 2). The three-dimensional resident-prefix bench
(scripts/bench_phase2_preconditions.py --resident-prefix 1024) trained one
balanced-anchor epoch on pool[:1024] before the restart of D10
(DEVIATIONS_PHASE2.md, D10 (B)); its record, bench_d10.json, is in the
Phase-2 return the pre-registration names, not in this repository."""


def _git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=True).stdout


def branches() -> list:
    out = []
    for line in _git("for-each-ref", "--format=%(refname:short)", "refs/remotes/origin").split():
        if line not in ("origin", "origin/HEAD"):
            out.append(line)
    return sorted(out)


def report_depths(refs=None) -> dict:
    from fejepa.experiments.runner import _pool_need

    seen, rows, scanned = {}, [], []
    for ref in refs or branches():
        commit = _git("rev-parse", ref).strip()
        scanned.append({"branch": ref, "commit": commit})
        for path in _git("ls-tree", "-r", "--name-only", ref).split("\n"):
            if not path.endswith(".json"):
                continue
            blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:{path}"],
                                  capture_output=True, check=True).stdout
            try:
                rep = json.loads(blob)
            except ValueError:
                continue
            if not isinstance(rep, dict) or not isinstance(rep.get("config"), dict) \
                    or "results" not in rep:
                continue
            sha = hashlib.sha256(blob).hexdigest()
            if sha in seen:
                seen[sha]["branches"].append(ref)
                continue
            cfg = rep["config"]
            data = cfg.get("data") or {}
            de = rep.get("data_economy") or {}
            row = {"path": path, "branches": [ref], "commit": commit, "sha256": sha,
                   "corpus": data.get("dir"), "split": cfg.get("split"),
                   "recorded_depth": de.get("unlabeled_pool_depth_used"),
                   "config_depth": int(_pool_need(cfg.get("experiments") or {}))}
            seen[sha] = row
            rows.append(row)
    deepest = {}
    for r in rows:
        if r["corpus"] is None:
            continue
        d = max(x for x in (r["recorded_depth"], r["config_depth"]) if x is not None)
        deepest[r["corpus"]] = max(deepest.get(r["corpus"], 0), d)
    return {"what": "PREREG_HOLDOUT Sec. 2: the deepest pool prefix of each corpus that any "
                    "recorded run used, from every report committed on every branch",
            "scanned": scanned,
            "reports": sorted(rows, key=lambda r: (str(r["corpus"]), r["path"])),
            "deepest": deepest}


def history_refs() -> list:
    """Every remote branch and tag of this checkout, with the object it
    names, in name order."""
    out = []
    for line in _git("for-each-ref", "--format=%(refname) %(objectname)",
                     "refs/remotes/origin", "refs/tags").splitlines():
        ref, oid = line.split(" ")
        if ref != "refs/remotes/origin/HEAD":
            out.append({"ref": ref, "object": oid})
    return out


def history_reports(objects: list) -> dict:
    """The SHA-256 of every report among the JSON files reachable from the
    given commits or tags (`git rev-list --objects`: every older commit of
    them included)."""
    ids = sorted({line.split(" ", 1)[0] for line in
                  _git("rev-list", "--objects", *objects).splitlines()
                  if line.endswith(".json") and " " in line})
    p = subprocess.run(["git", "-C", str(ROOT), "cat-file", "--batch"],
                       input="".join(f"{i}\n" for i in ids).encode(), capture_output=True,
                       check=True)
    data, pos, shas, n = p.stdout, 0, set(), 0
    while pos < len(data):                    # "<id> <type> <size>\n<content>\n" per object
        eol = data.index(b"\n", pos)
        _oid, kind, size = data[pos:eol].decode().split(" ")
        blob = data[eol + 1:eol + 1 + int(size)]
        pos = eol + 1 + int(size) + 1
        if kind != "blob":
            continue
        n += 1
        try:
            rep = json.loads(blob)
        except ValueError:
            continue
        if isinstance(rep, dict) and isinstance(rep.get("config"), dict) and "results" in rep:
            shas.add(hashlib.sha256(blob).hexdigest())
    return {"json_objects": n, "reports": sorted(shas)}


def other_reads(commits: dict) -> list:
    """`OTHER_READS` read from git objects at the given {branch: commit}."""
    out = []
    for branch, path, what, corpus, depth in OTHER_READS:
        commit = commits[f"origin/{branch}"]
        blob = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
                              capture_output=True, check=True).stdout
        out.append({"branch": f"origin/{branch}", "commit": commit, "path": path,
                    "sha256": hashlib.sha256(blob).hexdigest(), "what": what,
                    "corpus": corpus, "depth": int(depth(json.loads(blob)))})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = report_depths()
    refs = history_refs()
    hist = history_reports([r["object"] for r in refs])
    res["history"] = {"refs": refs, "json_objects": hist["json_objects"],
                      "reports": len(hist["reports"]),
                      "same_as_heads": set(hist["reports"]) == {r["sha256"]
                                                                for r in res["reports"]}}
    res["other_reads"] = other_reads({s["branch"]: s["commit"] for s in res["scanned"]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["deepest"], indent=1))


if __name__ == "__main__":
    main()
