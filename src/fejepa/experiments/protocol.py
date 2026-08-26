"""Shared experimental protocol: splits, seeds, pipelines, result/kill records.

Plan v2.0 mapping:
  - Sec.6 (bottom): the two pipelines are *named and never conflated* --
    P-A (anchor as supervised auxiliary; E1') and P-B (pretrain -> fine-tune; E2, gate c).
  - Sec.5 item 3 (statistics floor): seeds are explicit lists; per-seed values are the
    caller's responsibility to persist (helpers here standardize the record shape).
  - Audit V4: splits are deterministic permutations of the manifest ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.archive import (MANIFEST_NAME, instance_files, load_instance,
                            load_manifest)

PIPELINE_PA = "P-A: anchor as supervised auxiliary"


def seeded_factory(factory, seed: int):
    """Seed torch BEFORE construction so per-seed runs differ in initialization,
    not only in data order (plan Sec.5 item 3: honest seed variation)."""
    try:
        import torch

        torch.manual_seed(int(seed))
    except Exception:
        pass
    return factory()
PIPELINE_PB = "P-B: pretrain -> fine-tune"


@dataclass
class Split:
    val_files: list
    pool_files: list


def load_split(data_dir, n_val: int, seed: int) -> Split:
    files = instance_files(data_dir)
    if len(files) <= n_val:
        raise ValueError(f"dataset too small: {len(files)} <= n_val={n_val}")
    perm = np.random.default_rng(seed).permutation(len(files))
    return Split(val_files=[files[i] for i in perm[:n_val]],
                 pool_files=[files[i] for i in perm[n_val:]])


def require_asis_corpus(dcfg: dict) -> None:
    """Fail fast when ``labelled_policy='asis'`` points at a missing corpus.

    v2.1.5 guard for the 2026-07-14 deciding-run failure: with 'asis' and no
    manifest on disk, `_ensure_dataset` used to regenerate the corpus -- which
    the v2 generators write *unlabelled* by design (WP5) -- so hours of gmsh
    generation were followed by a certain rejection at the labelling stage.
    'asis' means "a labelled corpus already exists"; it must never trigger
    generation.
    """
    if dcfg.get("labelled_policy") != "asis":
        return
    ddir = Path(dcfg["dir"])
    if (ddir / MANIFEST_NAME).exists():
        return
    raise ValueError(
        f"labelled_policy='asis' requires a pre-existing corpus at {ddir}, but "
        f"no {MANIFEST_NAME} was found there. Transfer the corpus, or create "
        f"and pre-label one:\n"
        f"  fejepa generate {ddir} --n <N> --seed <seed> --backend gmsh\n"
        f"  fejepa label {ddir} --n-val <split.n_val> --split-seed "
        f"<split.seed> --pool-prefix <label_need> --workers <W>\n"
        f"(or switch labelled_policy to 'economy'; under prereg_guard that "
        f"edit changes the config SHA-256 and requires re-stamping).")


def asis_missing_labels(files, data_dir, max_report: int = 8) -> list[str]:
    """Names among ``files`` that carry no reference labels (asis verification).

    The manifest's per-record ``labelled`` flags answer for most files in O(1);
    any file the manifest does not vouch for is opened and its ``U_star``
    checked directly, so corpora predating the flag (e.g. Phase-1 exports)
    still verify. Collection stops at ``max_report`` names -- enough to prove
    and report a violation without scanning a large unlabelled corpus.
    """
    flags: dict[str, bool] = {}
    try:
        for r in load_manifest(data_dir).get("instances", []):
            flags[r["file"]] = bool(r.get("labelled", False))
    except FileNotFoundError:
        pass
    missing: list[str] = []
    for f in files:
        name = Path(f).name
        if flags.get(name, False):
            continue
        if load_instance(f).labelled:   # manifest predates the flag: trust file
            continue
        missing.append(name)
        if len(missing) >= max_report:
            break
    return missing


def load_archs(files) -> list:
    return [load_instance(f) for f in files]


def seeds_list(n_seeds: int) -> list[int]:
    return list(range(int(n_seeds)))


def mean_std(xs) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {"mean": float(a.mean()), "std": float(a.std()), "per_seed": a.tolist()}


def t_stat(a: dict, b: dict, n_seeds: int) -> float:
    """Welch-style t on seed means: (a-b) / (sqrt(sa^2+sb^2)/sqrt(n))."""
    se = float(np.sqrt(a["std"] ** 2 + b["std"] ** 2) / np.sqrt(max(1, n_seeds)))
    return float((a["mean"] - b["mean"]) / se) if se > 0 else float("inf")


def kill(condition: str, triggered: bool, note: str = "") -> dict:
    return {"condition": condition, "triggered": bool(triggered), "note": note}


DIVERGENCE_DISP_LIMIT = 10.0


def divergence_flags(seed_evals: list, key: str = "disp_rel_l2",
                     limit: float = DIVERGENCE_DISP_LIMIT) -> list[bool]:
    """PREREG_PHASE2 r8 Sec.5 divergence rule: a run is flagged when its loss is
    non-finite or its relative L2 displacement error exceeds ``limit``. Flags are
    reported per seed; seed means ALWAYS include flagged runs (no exclusion)."""
    import numpy as np

    out = []
    for e in seed_evals:
        v = e.get(key) if isinstance(e, dict) else e
        out.append(bool(v is None or not np.isfinite(v) or v > limit))
    return out


def result(exp_id: str, plan_ref: str, protocol: dict, metrics: dict,
           kills: list[dict]) -> dict:
    return {"id": exp_id, "plan_ref": plan_ref, "protocol": protocol,
            "metrics": metrics, "kills": kills}
