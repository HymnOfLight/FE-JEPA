"""Unit-level parallelism for the experiment grids (plan Sec.9: "wall-clock" /
rational use of the box).

Why this is the right lever on a single RTX 5090: one dim-256, N~400-node, batch-1
training uses a small fraction of the SMs, and the heavy experiments are grids of
*independent* trainings (E1' has 84 at rec8 scale, E8 ~33). Running `workers` units
concurrently -- each in its own spawned process with its own CUDA context -- multiplies
throughput until the GPU saturates, without touching any numerics.

Reliability contract (tested):
  - every unit is fully determined by its payload (seed, files, config); scheduling
    order cannot change results, and `workers=1` executes the *same* unit functions
    inline, so serial and parallel runs are identical (see
    tests/test_experiments_smoke.py::test_e1_parallel_matches_serial);
  - units receive archive *paths* and load from disk (payloads stay small; the OS page
    cache makes reloads cheap); in-memory-only archives cannot be dispatched;
  - workers are print-quiet (trainer milestones off); the parent shows one
    :class:`~fejepa.progress.Task` line per completed unit with ETA.

CPU hygiene: each worker limits torch threads to ~cpu_count/workers so 25 vCPUs are
shared instead of oversubscribed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

from ..progress import Task

# --------------------------------------------------------------- bootstrap ----

def _bootstrap(threads: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass


def map_units(func, payloads: list[dict], workers: int, label: str) -> list:
    """Run `func(payload)` over all payloads; returns results in payload order.

    workers <= 1: inline (trainer milestones stay on). workers > 1: spawn pool,
    payloads get ``quiet=True`` (unit-level progress only).
    """
    if not payloads:
        return []
    task = Task(label, total=len(payloads))
    if workers <= 1:
        out = []
        for p in payloads:
            out.append(func(p))
            task.step(p.get("tag", ""))
        task.done()
        return out

    for p in payloads:
        p["quiet"] = True
    threads = int(os.environ.get("FEJEPA_WORKER_THREADS",
                                 max(1, (os.cpu_count() or 8) // workers)))
    ctx = mp.get_context("spawn")                    # fork would break CUDA
    results: dict[int, object] = {}
    with ctx.Pool(processes=workers, initializer=_bootstrap,
                  initargs=(threads,)) as pool:
        jobs = [(i, p) for i, p in enumerate(payloads)]
        for i, res in pool.imap_unordered(_indexed(func), jobs):
            results[i] = res
            task.step(payloads[i].get("tag", ""))
    task.done()
    return [results[i] for i in range(len(payloads))]


class _indexed:
    """Picklable wrapper carrying the unit function; returns (index, result)."""

    def __init__(self, func):
        self.func = func

    def __call__(self, job):
        i, payload = job
        return i, self.func(payload)


# ------------------------------------------------------------ unit builders ----

def _load(files):
    from ..data.archive import load_instance

    return [load_instance(Path(f)) for f in files]


def _apply_runtime(payload) -> None:
    """Spawned workers are fresh interpreters: re-apply the TF32/device policy that
    the parent set (otherwise parallel units silently run without TF32 and diverge
    from the serial numerics path on CUDA)."""
    from ..runtime import setup_torch

    device = (payload.get("sup") or payload.get("pre") or {}).get("device", "cpu")
    setup_torch(device, tf32=bool(payload.get("tf32", True)))


def _build_model(payload):
    from ..models.features import FeatureSpec
    from .protocol import seeded_factory

    seed = int(payload["seed"])
    mcfg = payload["model"]
    if payload.get("kind", "fejepa") == "mgn":
        from ..models.gnn import build_mesh_gnn

        return seeded_factory(
            lambda: build_mesh_gnn(dim=int(mcfg.get("mgn_dim", 128)),
                                   depth=int(mcfg.get("mgn_depth", 8)),
                                   features=FeatureSpec.from_dict(
                                       mcfg.get("features")),
                                   scale_decode=bool(
                                       mcfg.get("scale_decode", True))), seed)
    from ..models.fejepa import FEJEPAConfig, build_fejepa

    return seeded_factory(lambda: build_fejepa(FEJEPAConfig.from_dict(mcfg)), seed)


def supervised_unit(payload: dict) -> dict:
    """One supervised training (any anchor_mode, optional pretrained state on disk).

    payload: kind, model, seed, train_files, val_files, sup{...SupervisedConfig kw},
             pretrained_path?, tag?, quiet?
    """
    from ..train.supervised import SupervisedConfig, train_supervised

    _apply_runtime(payload)
    model = _build_model(payload)
    sup = dict(payload["sup"])
    sup["seed"] = int(payload["seed"])
    if payload.get("quiet"):
        sup["log_every"] = -1
    state = None
    if payload.get("pretrained_path"):
        import torch

        state = torch.load(payload["pretrained_path"], map_location="cpu",
                           weights_only=True)
    res = train_supervised(model, _load(payload["train_files"]),
                           _load(payload["val_files"]),
                           SupervisedConfig(**sup), pretrained_state=state)
    return {k: res[k] for k in ("val", "pretrained_tensors_loaded")} | (
        {"balance_scale_mean": res["balance_scale_mean"]}
        if "balance_scale_mean" in res else {})


def pretrain_unit(payload: dict) -> dict:
    """One label-free pretraining (AR or JEPA); saves the state to payload['state_path']
    and optionally evaluates on val_files.

    payload: kind='fejepa', model, seed, files, loss='ar'|'jepa',
             pre{...PretrainConfig kw}, state_path, eval_val_files?, tag?, quiet?
    """
    import torch

    from ..metrics import evaluate_model, torch_predictor
    from ..train.losses import AR_CONFIG, JEPA_CONFIG
    from ..train.pretrain import PretrainConfig, pretrain

    _apply_runtime(payload)
    model = _build_model(payload)
    pre = dict(payload["pre"])
    pre["seed"] = int(payload["seed"])
    if payload.get("quiet"):
        pre["log_every"] = -1
    loss = AR_CONFIG if payload.get("loss", "ar") == "ar" else JEPA_CONFIG
    pretrain(model, _load(payload["files"]), PretrainConfig(loss=loss, **pre))

    sp = Path(payload["state_path"])
    sp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), sp)

    out = {"state_path": str(sp)}
    if payload.get("eval_val_files"):
        out["val"] = evaluate_model(
            torch_predictor(model, pre.get("device", "cpu")),
            _load(payload["eval_val_files"]))
    return out
