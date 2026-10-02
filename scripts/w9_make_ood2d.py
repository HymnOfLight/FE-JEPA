#!/usr/bin/env python3
"""wp9 Stage 0a: generate and label the OOD-2D v1 evaluation families
(F1-F5), the 2D remesh set (R) and (Stage 0c, PREREG_W9 r3) the in-band
holdout IB -- on the box, once; the manifests (with per-file SHA-256) then
pin them. See fejepa.fe.ood2d for the definitions.

F5 and R come first: session 1's rule 1 reads them. A family that fails is
recorded and the others still run; the script then exits non-zero.

A family directory that already holds a manifest is not regenerated: the
manifest must describe the family asked for (name, seed, size; for R the
mesh sizes) and every file must match it, else the script refuses that
family (move the directory away and re-run). A directory without a readable
manifest is a partial generation and is regenerated.

    python scripts/w9_make_ood2d.py --out runs/w9/ood2d --record runs/w9/session1/ood2d.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ORDER = ["F5", "R", "IB", "F1", "F2", "F3", "F4"]


def _expected(fam: str, a) -> dict:
    from fejepa.fe.ood2d import DEFAULT_SEEDS, REMESH_H

    if fam == "R":
        return {"family": "R", "seed": DEFAULT_SEEDS["R"],
                "n_instances": a.n_remesh * len(REMESH_H), "hs": [float(h) for h in REMESH_H]}
    if fam == "IB":
        return {"family": "IB", "seed": DEFAULT_SEEDS["IB"], "n_instances": a.n_inband}
    return {"family": fam, "seed": DEFAULT_SEEDS[fam], "n_instances": a.n}


def _existing(d: Path, want: dict):
    """The manifest of a finished family directory, checked against `want`;
    None if there is no readable manifest (a partial generation)."""
    from fejepa.data.archive import load_manifest
    from fejepa.fe.ood2d import verify_manifest_files

    try:
        m = load_manifest(d)
    except (OSError, ValueError):
        return None
    diff = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    if diff:
        raise SystemExit(f"{d}: its manifest is not the family asked for (have, want): {diff}; "
                         "move the directory away and re-run")
    bad = verify_manifest_files(d)
    if bad:
        raise SystemExit(f"{d}: {len(bad)} files missing or differing from their manifest "
                         f"(first: {bad[0]}); move the directory away and re-run")
    return m


def main() -> None:
    from fejepa.fe.ood2d import FAMILY_N, IB_N, REMESH_GEOMETRIES

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/w9/ood2d")
    ap.add_argument("--families", nargs="+", default=ORDER)
    ap.add_argument("--n", type=int, default=FAMILY_N,
                    help="instances per family F1-F5 (PREREG_W9: ood2d.FAMILY_N; tests only "
                         "change it)")
    ap.add_argument("--n-remesh", type=int, default=REMESH_GEOMETRIES,
                    help="geometries of the remesh set R (PREREG_W9: ood2d.REMESH_GEOMETRIES)")
    ap.add_argument("--n-inband", type=int, default=IB_N,
                    help="instances of the in-band holdout IB (PREREG_W9: ood2d.IB_N)")
    ap.add_argument("--record", required=True, help="summary JSON (manifest SHA-256 per family)")
    a = ap.parse_args()

    from fejepa.analysis.common import write_json
    from fejepa.data.archive import load_manifest, manifest_sha256
    from fejepa.fe.ood2d import DEFAULT_SEEDS, FAMILIES, generate_family, generate_inband, \
        generate_remesh
    from fejepa.report import _git_describe

    for fam in a.families:
        if fam not in FAMILIES and fam not in ("R", "IB"):
            raise SystemExit(f"unknown family {fam!r}")
    out = Path(a.out)
    res = {"what": "wp9 OOD-2D v1 families and remesh set (evaluation only)",
           "git": _git_describe(), "families": {}, "failed": []}
    for fam in a.families:
        d = out / fam
        t0 = time.time()
        try:
            if _existing(d, _expected(fam, a)) is not None:
                status = "verified existing"
            else:
                if d.exists():                   # a partial directory
                    shutil.rmtree(d)
                if fam == "R":
                    generate_remesh(d, a.n_remesh, DEFAULT_SEEDS["R"])
                elif fam == "IB":
                    generate_inband(d, a.n_inband, DEFAULT_SEEDS["IB"])
                else:
                    generate_family(d, fam, a.n, DEFAULT_SEEDS[fam])
                _existing(d, _expected(fam, a))          # the new family checks out
                status = "generated"
        except SystemExit as exc:
            res["failed"].append(fam)
            res["families"][fam] = {"dir": str(d), "status": "refused", "error": str(exc)}
            print(f"[ood2d] {fam} REFUSED: {exc}", flush=True)
            write_json(a.record, res)
            continue
        except Exception as exc:                          # noqa: BLE001 -- keep the others
            res["failed"].append(fam)
            res["families"][fam] = {"dir": str(d), "status": "failed",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()[-2000:]}
            print(f"[ood2d] {fam} FAILED: {type(exc).__name__}: {exc}", flush=True)
            write_json(a.record, res)
            continue
        m = load_manifest(d)
        res["families"][fam] = {
            "dir": str(d), "status": status, "manifest_sha256": manifest_sha256(d),
            "n_instances": m["n_instances"], "seed": m.get("seed"),
            "definition": m.get("definition"), "gmsh_version": m.get("gmsh_version"),
            "ledger": m.get("ledger"),
            "n_nodes_min_median_max": _mmm([r["n_nodes"] for r in m["instances"]]),
            "seconds": round(time.time() - t0, 1)}
        print(json.dumps({fam: res["families"][fam]}), flush=True)
        write_json(a.record, res)
    write_json(a.record, res)
    if res["failed"]:
        raise SystemExit(f"[ood2d] not complete: {res['failed']} (see {a.record})")


def _mmm(v):
    import statistics

    return [min(v), int(statistics.median(v)), max(v)] if v else None


if __name__ == "__main__":
    main()
