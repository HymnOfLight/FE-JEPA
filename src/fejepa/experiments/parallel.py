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
    if payload.get("kind", "fejepa") == "bottleneck":
        from ..models.bottleneck import BottleneckConfig, build_bottleneck

        return seeded_factory(
            lambda: build_bottleneck(BottleneckConfig.from_dict(payload["model"])),
            payload["seed"])
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


def _maybe_compile(model, payload):
    if payload.get("compile"):
        import torch

        return torch.compile(model)
    return model


def _state_dict(model):
    """State dict of the underlying module (torch.compile wraps in _orig_mod;
    saving the wrapper would poison the shared-checkpoint contract)."""
    return getattr(model, "_orig_mod", model).state_dict()


def _cache_path(payload: dict):
    cd = payload.get("cache_dir")
    if not cd:
        return None
    import re

    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(payload.get("tag", "unit")))
    return Path(cd) / f"{key}.pkl"


def cached_supervised_unit(payload: dict) -> dict:
    """D9: supervised_unit with an on-disk result cache keyed by the unit tag.
    A hit returns the stored result (marked from_cache=True); a miss trains,
    stores, returns. Without payload['cache_dir'] it is supervised_unit verbatim."""
    import pickle

    cp = _cache_path(payload)
    if cp is not None and cp.exists() and payload.get("reuse_existing"):
        try:                                                  # R9a fallback
            with cp.open("rb") as fh:
                res = pickle.load(fh)
            res["from_cache"] = True
            return res
        except Exception as exc:                              # noqa: BLE001
            print(f"[d9] {cp}: unusable cache ({type(exc).__name__}); removed, "
                  f"retraining", flush=True)
            cp.unlink(missing_ok=True)
    if cp is not None:                                        # R9b in-unit ckpt
        payload = dict(payload, ckpt_path=str(cp.with_suffix(".ckpt")))
    res = supervised_unit(payload)
    if cp is not None:
        from ..train.checkpoint import atomic_pickle_dump

        atomic_pickle_dump(res, cp)
        Path(payload["ckpt_path"]).unlink(missing_ok=True)      # unit complete
    return res


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
    sup.setdefault("precision", payload.get("precision", "fp32"))
    if payload.get("ckpt_path"):                              # R9b
        sup.setdefault("ckpt_path", payload["ckpt_path"])
        sup.setdefault("resume", bool(payload.get("reuse_existing")))
    if payload.get("quiet"):
        sup["log_every"] = -1
    state = None
    if payload.get("pretrained_path"):
        import torch

        state = torch.load(payload["pretrained_path"], map_location="cpu",
                           weights_only=True)
    model = _maybe_compile(model, payload)
    res = train_supervised(model, _load(payload["train_files"]),
                           _load(payload["val_files"]),
                           SupervisedConfig(**sup), pretrained_state=state)
    if payload.get("state_path"):
        import torch

        from ..train.checkpoint import atomic_torch_save

        atomic_torch_save(_state_dict(model), Path(payload["state_path"]))
    extra = ({"resumed_from_epoch": res["resumed_from_epoch"]}
             if "resumed_from_epoch" in res else {})
    return extra | {k: res[k] for k in ("val", "pretrained_tensors_loaded")} | (
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
    pre.setdefault("precision", payload.get("precision", "fp32"))
    if payload.get("quiet"):
        pre["log_every"] = -1
    spec = payload.get("loss", "ar")
    if isinstance(spec, dict):                      # wp8 E1: dict overrides on AR
        from dataclasses import replace as _replace

        loss = _replace(AR_CONFIG, **spec)
    else:
        loss = AR_CONFIG if spec == "ar" else JEPA_CONFIG
    sp = Path(payload["state_path"])
    reused = False
    resumed_from = None
    if payload.get("reuse_existing") and sp.exists():
        # D9: consume a state produced by an earlier attempt of the SAME stamped
        # configuration (identical configurations are trained once); the file's
        # SHA-256 is returned so the report can chain attempt-1 -> attempt-2.
        # R9a: a corrupt (e.g. truncated) file is removed and retrained.
        try:
            sd = torch.load(str(sp), map_location="cpu", weights_only=True)
            model.load_state_dict(sd, strict=True)
            model.to(pre.get("device", "cpu"))
            reused = True
        except Exception as exc:                              # noqa: BLE001
            print(f"[d9] {sp}: unusable state ({type(exc).__name__}); removed, "
                  f"retraining", flush=True)
            sp.unlink(missing_ok=True)
    if not reused:
        from ..train.checkpoint import atomic_torch_save

        model = _maybe_compile(model, payload)
        pre.setdefault("ckpt_path", str(sp.with_suffix(".ckpt")))   # R9b
        pre.setdefault("resume", bool(payload.get("reuse_existing")))
        hist = pretrain(model, _load(payload["files"]), PretrainConfig(loss=loss, **pre))
        # the SIGReg head is training scaffolding: the deliverable state is the
        # encoder/decoder only (strict-loadable into a fresh model)
        atomic_torch_save({k: v for k, v in _state_dict(model).items()
                           if not k.startswith("sigreg_head.")}, sp)
        Path(pre["ckpt_path"]).unlink(missing_ok=True)          # unit complete
        resumed_from = hist.get("resumed_from_epoch")

    import hashlib

    out = {"state_path": str(sp), "reused_state": reused,
           "state_sha256": hashlib.sha256(sp.read_bytes()).hexdigest()}
    if not reused and resumed_from is not None:
        out["resumed_from_epoch"] = int(resumed_from)
    if payload.get("eval_val_files"):
        out["val"] = evaluate_model(
            torch_predictor(model, pre.get("device", "cpu")),
            _load(payload["eval_val_files"]))
    return out
