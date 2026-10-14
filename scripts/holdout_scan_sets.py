#!/usr/bin/env python3
"""cmame-paper Stage 11 (PREREG_HOLDOUT Sec. 2 and 7): the sets that the
corpora's generators made from the corpora's seed under the given
directories, against the bounds of `scripts/holdout_overlap.py` (`BOUNDS`).
The sandbox was scanned with it for PREREG_HOLDOUT Sec. 2, and is scanned
again before the stamp, which is not made unless it exits 0.

    python scripts/holdout_scan_sets.py /home/claude /tmp

Every `manifest.json` under the directories is read (directories of virtual
environments, site-packages, caches and git objects are skipped). The sets
of the corpora's generators (backend `gmsh` in two dimensions, `gmsh3d` in
three) and seed 0 are kept, per kind: three dimensions with a band of mesh
sizes (`lc_range`), three dimensions at one size, two dimensions. A
manifest of a corpus's own size (2,000 instances with the corpus's band,
30,000 in two dimensions) is a copy of that corpus's manifest: it is listed
apart, with the number of instance files beside it. Directories holding
instance files (`instance_NNNNN.npz`) and no readable manifest are listed
for inspection, with their index range and whether every file is empty.
The output is JSON. Exit status 0 iff, apart from the corpora's own
manifests, no set exceeds its kind's bound and no copy of a corpus manifest
has instance files beside it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SKIP = {".git", "__pycache__", "site-packages", "node_modules", ".pytest_cache",
        ".ruff_cache", ".cache"}
INSTANCE = re.compile(r"instance_(\d{5})\.npz")


def _overlap():
    spec = importlib.util.spec_from_file_location(
        "holdout_overlap", Path(__file__).resolve().parent / "holdout_overlap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skipped(d: str) -> bool:
    return d in SKIP or d.startswith("venv") or d.startswith("py3")


def scan(dirs) -> dict:
    ov = _overlap()
    band = list(ov.BAND)
    kinds = {"3d_band": [], "3d_one_size": [], "2d": []}
    copies, stray, seen = [], [], set()
    for base in dirs:
        for root, sub, files in os.walk(base):
            sub[:] = sorted(d for d in sub if not _skipped(d))
            real = os.path.realpath(root)
            if real in seen:
                continue
            seen.add(real)
            inst = sorted(int(m.group(1)) for f in files if (m := INSTANCE.fullmatch(f)))
            man = None
            if "manifest.json" in files:
                try:
                    man = json.loads((Path(root) / "manifest.json").read_text())
                except (OSError, ValueError):
                    man = None
            if not isinstance(man, dict):
                if inst:
                    stray.append({"dir": root, "files": len(inst), "first": inst[0],
                                  "last": inst[-1],
                                  "all_empty": all(os.path.getsize(Path(root) / f) == 0
                                                   for f in files if INSTANCE.fullmatch(f))})
                continue
            backend, seed = man.get("backend"), man.get("seed")
            if backend not in ("gmsh", "gmsh3d") or str(seed) != "0":
                continue
            n = len(man.get("instances") or [])
            rec = {"dir": root, "n": n, "files": len(inst),
                   "mesh_size": man.get("lc_range", man.get("lc"))}
            if backend == "gmsh3d" and isinstance(man.get("lc_range"), list):
                if n == 2000 and [float(x) for x in man["lc_range"]] == band:
                    copies.append(rec)
                else:
                    kinds["3d_band"].append(rec)
            elif backend == "gmsh3d":
                kinds["3d_one_size"].append(rec)
            elif n == 30000:
                copies.append(rec)
            else:
                kinds["2d"].append(rec)
    out = {"what": "PREREG_HOLDOUT Sec. 2 and 7: the sets of the corpora's generators and "
                   "seed under the scanned directories (scripts/holdout_scan_sets.py)",
           "scanned": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "dirs": [str(d) for d in dirs], "bounds": dict(ov.BOUNDS), "kinds": {},
           "corpus_manifest_copies": sorted(copies, key=lambda r: r["dir"]),
           "stray_instance_files": sorted(stray, key=lambda r: r["dir"])}
    ok = all(c["files"] == 0 for c in copies)
    for k, rows in kinds.items():
        longest = max((r["n"] for r in rows), default=0)
        beyond = sorted((r for r in rows if r["n"] > ov.BOUNDS[k]), key=lambda r: r["dir"])
        out["kinds"][k] = {"sets": len(rows), "longest": longest, "beyond": beyond}
        ok = ok and not beyond
    out["within_bounds"] = bool(ok)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = scan(a.dirs)
    text = json.dumps(res, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(json.dumps({"scanned": res["scanned"], "within_bounds": res["within_bounds"],
                      **{k: {"sets": v["sets"], "longest": v["longest"],
                             "beyond": len(v["beyond"])} for k, v in res["kinds"].items()},
                      "corpus_manifest_copies": len(res["corpus_manifest_copies"]),
                      "stray_instance_files": len(res["stray_instance_files"])}, indent=1))
    raise SystemExit(0 if res["within_bounds"] else 1)


if __name__ == "__main__":
    main()
