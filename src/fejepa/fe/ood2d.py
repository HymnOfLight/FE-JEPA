"""wp9 Stage 0a: out-of-distribution 2D evaluation families (OOD-2D v1) and
the 2D remesh set.

Each family changes ONE attribute of the training family
(:func:`fejepa.fe.generator.sample_params`) and draws every other attribute
from the training ranges:

  F1  4-6 holes                                       (training: 0-3)
  F2  slender plates: width 3.0-4.5, height 0.6-0.8   (training: 1.5-3.0, 0.8-1.5)
  F3  1-3 holes, all large (radius 0.16-0.22 x min(w, h)) or all small
      (0.03-0.06 x min(w, h)), one size class per instance, each with
      probability 1/2                                 (training: 0.06-0.16)
  F4  Poisson ratio 0.40-0.45                         (training: 0.25-0.38)
  F5  mesh size target_h = 0.025                      (training: 0.05-0.12)
  R   remesh set: training-family geometries and loads, each meshed at
      several target_h (only the mesh changes)
  IB  in-band holdout (PREREG_W9 r3): the training family itself, drawn
      afresh by the training sampler -- with E1's validation split, H1's
      in-band set

Instances are labelled at generation (direct solve, counted in the
manifest's ledger) and serve evaluation only: no training and no selection
reads them. Each manifest record carries the file's SHA-256, so the
manifest's own SHA-256 pins the meshes (gmsh output depends on its
version; the families are generated once, on the box).

Hole placement follows the training rule (centres in the middle 60% of each
side, pairwise clearance 0.05). The training sampler drops a hole whose 50
placement attempts fail; F1 and F3, whose attribute IS the hole set, instead
redraw the whole set until every hole is placed (and, for F3's large holes,
keep a clearance of 0.02 x min(w, h) to the plate edge, which the training
radii satisfy automatically).

Secondary differences that come with the changed attribute (reported, not
corrected): F1's whole-set redraw favours smaller radii slightly (mean
radius / min(w, h) about 0.106 against the training family's 0.109); F3 has
no hole-free plates, and its large holes may leave an edge ligament down to
0.02 x min(w, h) (training radii leave at least 0.04 x min(w, h)).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..data.archive import save_instance, write_manifest
from .generator import build_instance, sample_params
from .solve import SolveLedger, solve_fe_displacement

FAMILIES = {
    "F1": "4-6 holes (training 0-3)",
    "F2": "slender plates: width 3.0-4.5, height 0.6-0.8 (training 1.5-3.0, 0.8-1.5)",
    "F3": "1-3 holes, all large (0.16-0.22) or all small (0.03-0.06) x min(w, h), "
          "one class per instance with probability 1/2 (training 0.06-0.16)",
    "F4": "Poisson ratio 0.40-0.45 (training 0.25-0.38)",
    "F5": "mesh size target_h = 0.025 (training 0.05-0.12)",
}
DEFAULT_SEEDS = {"F1": 91001, "F2": 91002, "F3": 91003, "F4": 91004, "F5": 91005, "R": 91006,
                 "IB": 91007}
IB_N = 2048
IB_DEFINITION = ("training family (fejepa.fe.generator.sample_params), drawn afresh: an "
                 "in-band holdout")
F5_TARGET_H = 0.025
REMESH_H = (0.12, 0.085, 0.05, 0.035, 0.025)
MAX_REDRAWS = 2000
FAMILY_N = 256
REMESH_GEOMETRIES = 16
SET_SIZES = {**{f: FAMILY_N for f in FAMILIES}, "R": REMESH_GEOMETRIES * len(REMESH_H),
             "IB": IB_N}
"""PREREG_W9's evaluation sets (Sec. 3): instances per set, each drawn from its
DEFAULT_SEEDS seed (scripts/w9_make_ood2d.py's defaults). The session-2 plan
and the adjudication check session 1's record against both."""


def _place(rng, width, height, n_holes, rmin, rmax, strict, edge_clear=0.0):
    """Training placement rule; `strict` redraws the whole set until all
    `n_holes` are placed (None if MAX_REDRAWS sets fail)."""
    for _ in range(MAX_REDRAWS if strict else 1):
        holes = []
        for _h in range(n_holes):
            for _attempt in range(50):
                r = float(rng.uniform(rmin, rmax) * min(width, height))
                cx = float(rng.uniform(0.2 * width, 0.8 * width))
                cy = float(rng.uniform(0.2 * height, 0.8 * height))
                inside = min(cx, width - cx, cy, height - cy) - r >= edge_clear
                if inside and all((cx - hx) ** 2 + (cy - hy) ** 2 > (r + hr + 0.05) ** 2
                                  for hx, hy, hr in holes):
                    holes.append([cx, cy, r])
                    break
        if not strict or len(holes) == n_holes:
            return holes
    return None


def sample_family_params(rng: np.random.Generator, family: str) -> dict:
    """Parameters of one instance of `family` (the training sampler's dict,
    plus the family tag and the F3 size class)."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    for _redraw in range(MAX_REDRAWS):
        if family == "F2":
            width, height = float(rng.uniform(3.0, 4.5)), float(rng.uniform(0.6, 0.8))
        else:
            width, height = float(rng.uniform(1.5, 3.0)), float(rng.uniform(0.8, 1.5))
        nu = float(rng.uniform(0.40, 0.45) if family == "F4" else rng.uniform(0.25, 0.38))
        extra = {}
        if family == "F1":
            holes = _place(rng, width, height, int(rng.integers(4, 7)), 0.06, 0.16, strict=True)
        elif family == "F3":
            large = bool(rng.random() < 0.5)
            rmin, rmax = (0.16, 0.22) if large else (0.03, 0.06)
            holes = _place(rng, width, height, int(rng.integers(1, 4)), rmin, rmax, strict=True,
                           edge_clear=0.02 * min(width, height))
            extra["hole_class"] = "large" if large else "small"
        else:
            holes = _place(rng, width, height, int(rng.integers(0, 4)), 0.06, 0.16, strict=False)
        if holes is None:                      # infeasible set for this plate: redraw it
            continue
        target_h = F5_TARGET_H if family == "F5" else float(rng.uniform(0.05, 0.12))
        traction_scales = 0.05 * rng.uniform(0.5, 1.5, size=4)
        return dict(width=width, height=height, nu=nu, holes=holes, target_h=target_h,
                    traction_scales=traction_scales.tolist(), family=family, **extra)
    raise RuntimeError(f"{family}: no feasible instance in {MAX_REDRAWS} redraws")


def labelled_instance(params: dict, ledger: SolveLedger | None = None):
    """Build, label (direct solve) and tag one instance."""
    arch = build_instance({k: params[k] for k in ("width", "height", "nu", "holes",
                                                    "target_h", "traction_scales")})
    arch.U_star, _ = solve_fe_displacement(arch.K, arch.F, arch.free_mask, method="direct",
                                           ledger=ledger, stage="evaluation-labels")
    for k in ("family", "hole_class", "geometry", "remesh_h"):
        if k in params:
            arch.meta["extra"][k] = params[k]
    return arch


def _record(path: Path, arch, params: dict) -> dict:
    ex = arch.meta["extra"]
    rec = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "n_nodes": arch.n_nodes, "n_holes": int(ex["n_holes"]),
           "target_h": float(ex["target_h"]), "width": float(ex["width"]),
           "height": float(ex["height"]), "nu": float(arch.meta["material"]["nu"]),
           "labelled": True}
    for k in ("hole_class", "geometry"):
        if k in params:
            rec[k] = params[k]
    return rec


def _ledger(ledger: SolveLedger) -> dict:
    """The solve counts without the wall clock: a manifest holds no volatile
    field, so regenerating a family with the same gmsh reproduces its SHA-256."""
    d = ledger.as_dict()
    return {"per_stage": d["per_stage"], "total": d["total"]}


def _gmsh_version() -> str:
    try:
        import gmsh

        return str(getattr(gmsh, "__version__", "unknown"))
    except Exception:                                     # noqa: BLE001
        return "unavailable"


def generate_family(out, family: str, n: int, seed: int | None = None) -> Path:
    """`n` labelled instances of `family` in `out` (instance i from
    SeedSequence(seed).spawn(n)[i], serial, manifest order = index order)."""
    seed = DEFAULT_SEEDS[family] if seed is None else int(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ledger, records = SolveLedger(), []
    for i, child in enumerate(np.random.SeedSequence(seed).spawn(n)):
        params = sample_family_params(np.random.default_rng(child), family)
        arch = labelled_instance(params, ledger)
        path = out / f"instance_{i:05d}.npz"
        save_instance(arch, path)
        records.append(_record(path, arch, params))
    write_manifest(out, records, {
        "backend": "gmsh", "gmsh_version": _gmsh_version(), "family": family,
        "definition": FAMILIES[family], "seed": seed, "labelled_policy": "all",
        "purpose": "evaluation only (wp9 OOD-2D v1)", "ledger": _ledger(ledger)})
    return out


def generate_inband(out, n: int = IB_N, seed: int | None = None) -> Path:
    """`n` labelled instances of the training family itself (instance i from
    SeedSequence(seed).spawn(n)[i], drawn by the training sampler as the
    corpus is; serial, manifest order = index order): the in-band holdout IB."""
    seed = DEFAULT_SEEDS["IB"] if seed is None else int(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ledger, records = SolveLedger(), []
    for i, child in enumerate(np.random.SeedSequence(seed).spawn(n)):
        params = dict(sample_params(np.random.default_rng(child)), family="IB")
        arch = labelled_instance(params, ledger)
        path = out / f"instance_{i:05d}.npz"
        save_instance(arch, path)
        records.append(_record(path, arch, params))
    write_manifest(out, records, {
        "backend": "gmsh", "gmsh_version": _gmsh_version(), "family": "IB",
        "definition": IB_DEFINITION, "seed": seed, "labelled_policy": "all",
        "purpose": "evaluation only (wp9: H1's in-band set with E1's validation split)",
        "ledger": _ledger(ledger)})
    return out


def generate_remesh(out, n_geometries: int, seed: int | None = None,
                    hs=REMESH_H) -> Path:
    """Training-family geometry g (from SeedSequence(seed).spawn(n)[g], drawn
    by the training sampler) meshed at every h in `hs`: identical geometry,
    material and loads, only the mesh changes; labelled."""
    seed = DEFAULT_SEEDS["R"] if seed is None else int(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ledger, records = SolveLedger(), []
    for g, child in enumerate(np.random.SeedSequence(seed).spawn(n_geometries)):
        base = sample_params(np.random.default_rng(child))
        for h in hs:
            params = dict(base, target_h=float(h), geometry=g, remesh_h=float(h))
            arch = labelled_instance(params, ledger)
            path = out / f"g{g:03d}_h{h:.4f}.npz"
            save_instance(arch, path)
            records.append(_record(path, arch, params))
    write_manifest(out, records, {
        "backend": "gmsh", "gmsh_version": _gmsh_version(), "family": "R",
        "definition": "training-family geometries, each meshed at every target_h in "
                      f"{list(hs)}", "seed": seed, "hs": [float(h) for h in hs],
        "labelled_policy": "all", "purpose": "evaluation only (wp9 C0.7-2D remesh)",
        "ledger": _ledger(ledger)})
    return out


def verify_manifest_files(data_dir) -> list:
    """Files that are missing or whose bytes differ from the SHA-256 their
    manifest records."""
    from ..data.archive import load_manifest

    d = Path(data_dir)
    return [r["file"] for r in load_manifest(d)["instances"]
            if not (d / r["file"]).is_file()
            or hashlib.sha256((d / r["file"]).read_bytes()).hexdigest() != r.get("sha256")]
