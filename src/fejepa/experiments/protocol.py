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

import numpy as np

from ..data.archive import instance_files, load_instance

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


def result(exp_id: str, plan_ref: str, protocol: dict, metrics: dict,
           kills: list[dict]) -> dict:
    return {"id": exp_id, "plan_ref": plan_ref, "protocol": protocol,
            "metrics": metrics, "kills": kills}
