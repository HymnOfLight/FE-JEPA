"""WP7 3D-G1: SimJEB schema audit (first deliverable of the SimJEB ingestion
line; RUN_PLAN 2026-08-05 and Manual sec 8's G1'-gated register).

Offline by design: point it at a locally downloaded SimJEB (or DeepJEB) root
and it inventories the tree so the ingestion loader can be written against
facts rather than folklore -- file formats present, counts and sizes, mesh
headline numbers from cheap probes (.obj/.stl vertex and face counts), tabular
headers, candidate official-split files, and JSON key schemas. Nothing is
downloaded and nothing is modified; the report is JSON on stdout or --out.

Usage:
  python3 scripts/simjeb_schema_audit.py /path/to/simjeb --out audit.json

Licence note (fire-plan R1 discipline): per-model licence audit remains a
separate, mandatory step before any training use.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path


def _probe_obj(p: Path) -> dict:
    v = f = 0
    with open(p, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                v += 1
            elif line.startswith("f "):
                f += 1
    return {"vertices": v, "faces": f}


def _probe_stl(p: Path) -> dict:
    with open(p, "rb") as fh:
        head = fh.read(84)
    if head[:5] == b"solid" and b"\n" in head:
        return {"format": "ascii-stl"}
    n_tri = struct.unpack("<I", head[80:84])[0] if len(head) >= 84 else None
    return {"format": "binary-stl", "triangles": n_tri}


def _probe_csv(p: Path) -> dict:
    with open(p, "r", errors="ignore") as fh:
        header = fh.readline().strip()
    return {"header": header.split(",")}


def _probe_json(p: Path) -> dict:
    try:
        obj = json.load(open(p))
    except Exception as e:                                # noqa: BLE001
        return {"error": str(e)[:80]}
    if isinstance(obj, dict):
        return {"type": "dict", "keys": sorted(obj)[:20]}
    return {"type": type(obj).__name__, "len": len(obj) if hasattr(obj, "__len__") else None}


_PROBES = {".obj": _probe_obj, ".stl": _probe_stl, ".csv": _probe_csv,
           ".json": _probe_json}
_SPLIT_HINTS = ("split", "train", "test", "val", "fold")


def audit_tree(root: str | Path) -> dict:
    root = Path(root)
    counts, sizes, samples, splits = Counter(), Counter(), {}, []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        counts[ext] += 1
        sizes[ext] += p.stat().st_size
        if ext in _PROBES and ext not in samples:
            samples[ext] = {"example": str(p.relative_to(root)),
                            **_PROBES[ext](p)}
        if any(h in p.name.lower() for h in _SPLIT_HINTS):
            splits.append(str(p.relative_to(root)))
    return {"root": str(root),
            "counts_by_ext": dict(counts),
            "bytes_by_ext": dict(sizes),
            "samples": samples,
            "split_files": splits[:50]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rep = audit_tree(a.root)
    text = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(text)
        print(f"[simjeb-audit] -> {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
