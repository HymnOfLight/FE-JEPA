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

from ..data.archive import LazyArchives
from ..progress import Task

# --------------------------------------------------------------- bootstrap ----

def _bootstrap(threads: int, parent_pid: int | None = None) -> None:
    if parent_pid is not None:
        _watch_parent(parent_pid)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass


PARENT_WATCH_S = 5.0


def _watch_parent(parent_pid: int, period: float = PARENT_WATCH_S) -> None:
    """wp9 Stage 0b: a worker whose parent process has died (killed from
    outside) exits within `period` seconds instead of training its unit on as
    an orphan that holds the GPU. A thread polls the parent's pid, so it does
    not depend on which of the parent's threads started the worker.
    Scheduling only."""
    import threading
    import time

    def watch():
        while True:
            if os.getppid() != parent_pid:
                os._exit(1)
            time.sleep(period)

    threading.Thread(target=watch, name="fejepa-parent-watch", daemon=True).start()


def map_units(func, payloads: list[dict], workers: int, label: str) -> list:
    """Run `func(payload)` over all payloads; returns results in payload order.

    workers <= 1: inline (trainer milestones stay on). workers > 1: a spawn
    process pool, payloads get ``quiet=True`` (unit-level progress only).

    wp9 Stage 0b: the pool is a `concurrent.futures.ProcessPoolExecutor`,
    watched every WORKER_POLL_S seconds. A worker that dies -- killed from
    outside, e.g. by the system's out-of-memory killer, while running a unit
    or while waiting for one -- ends the map with RuntimeError instead of
    letting it wait forever (multiprocessing.Pool waited for the lost unit,
    and its shutdown could block on a queue lock the dead worker held). A
    unit that raises ends the map at once too. Either way the other workers
    are terminated, not waited for. Scheduling only -- every unit computes
    what it did.
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

    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    for p in payloads:
        p["quiet"] = True
    threads = int(os.environ.get("FEJEPA_WORKER_THREADS",
                                 max(1, (os.cpu_count() or 8) // workers)))
    ctx = mp.get_context("spawn")                    # fork would break CUDA
    results: dict[int, object] = {}
    ex = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_bootstrap,
                             initargs=(threads, os.getpid()))
    procs = []
    try:
        futs = {ex.submit(func, p): i for i, p in enumerate(payloads)}
        procs = list((getattr(ex, "_processes", None) or {}).values())   # never replaced
        pending = set(futs)
        while pending:
            done, pending = wait(pending, timeout=WORKER_POLL_S, return_when=FIRST_COMPLETED)
            for fut in done:
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except BrokenProcessPool:
                    raise _WorkerDied() from None
                task.step(payloads[i].get("tag", ""))
            if pending and not done and any(p.exitcode is not None for p in procs):
                raise _WorkerDied()
    except BaseException as exc:
        _stop_workers(ex, procs)
        if isinstance(exc, _WorkerDied):        # the exit codes are known once all are stopped
            raise _died(label, procs) from None
        raise
    ex.shutdown(wait=True)
    task.done()
    return [results[i] for i in range(len(payloads))]


WORKER_POLL_S = 30.0


class _WorkerDied(Exception):
    """A worker of the pool died (internal: becomes `_died`'s error once the
    workers are stopped)."""


def _died(label: str, procs: list) -> RuntimeError:
    """wp9 Stage 0c: the error of a map whose worker died names the workers'
    exit codes, read after every worker was stopped and joined (-9: killed by
    signal 9, the system's out-of-memory killer's signal; -11: a crash; -15:
    a worker stopped after the death; a positive code: the worker exited,
    e.g. failed at start-up). The executor's own thread may still be reaping
    a worker when the map stops: a code not yet known is waited for up to 2 s
    (None if it stays unknown). Scheduling only."""
    import time

    codes, deadline = [p.exitcode for p in procs], time.monotonic() + 2.0
    while any(c is None for c in codes) and time.monotonic() < deadline:
        time.sleep(0.05)
        codes = [p.exitcode for p in procs]
    return RuntimeError(f"{label}: a worker process died (worker exit codes {codes}; -9 is a "
                        "kill by signal 9, e.g. by the system's out-of-memory killer; -15 a "
                        "worker stopped after the death; see also the lines above) and its unit "
                        "cannot finish; nothing of it was saved beyond its epoch checkpoint -- "
                        "restart the run with --reuse-states")


def _stop_workers(ex, procs: list) -> None:
    """End a failed map at once: cancel what has not started and terminate the
    workers (a running unit is not waited for). A worker that died half-way
    through sending its result leaves the executor's manager thread blocked
    on the rest of the message, and the interpreter would join that thread
    for ever at exit: once every worker is gone, closing this process's end
    of the result pipe ends the read."""
    procs = procs or list((getattr(ex, "_processes", None) or {}).values())
    results = getattr(ex, "_result_queue", None)
    ex.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        try:
            p.terminate()
        except Exception:                                  # noqa: BLE001
            pass
    for p in procs:
        try:
            p.join(timeout=10)
        except Exception:                                  # noqa: BLE001
            pass
    if results is not None and all(p.exitcode is not None for p in procs):
        try:
            results._writer.close()
        except Exception:                                  # noqa: BLE001
            pass


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
    payload = dict(payload, model={k: v for k, v in payload["model"].items()
                                   if k != "kind"})   # wp8: kind is routing, not a field
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


def deliverable_state(sd):
    """The state a unit saves: `sd` without the SIGReg head (training
    scaffolding). Stage 1.36: the result stays the OrderedDict that
    `state_dict()` returns, `_metadata` included (minus the head's entries),
    so that without a head the saved file is byte-identical to saving the
    state dict itself -- the wp7-3d main line's bytes (a plain dict dropped
    `_metadata`: identical tensors, different file SHA-256)."""
    from collections import OrderedDict

    out = OrderedDict((k, v) for k, v in sd.items() if not k.startswith("sigreg_head."))
    meta = getattr(sd, "_metadata", None)
    if meta is not None:
        out._metadata = OrderedDict((k, v) for k, v in meta.items()
                                    if k != "sigreg_head" and not k.startswith("sigreg_head."))
    return out


def _file_sha256(path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_state(sd: dict) -> dict:
    """Normalise a loaded state dict: strip a torch.compile wrapper prefix
    (`_orig_mod.`) so states saved by other tooling still load strictly."""
    pre = "_orig_mod."
    return {(k[len(pre):] if k.startswith(pre) else k): v for k, v in sd.items()}


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

    # R22 (Phase-2b): a unit initialised from a pretrained state records that
    # state's SHA-256; a cache hit is honoured only if the state on disk still
    # has that hash. A forgotten cache surgery therefore cannot serve results
    # built on superseded AR states -- they are retrained, and it is logged.
    lineage = _file_sha256(payload["pretrained_path"]) if payload.get("pretrained_path") else None
    cp = _cache_path(payload)
    if cp is not None and cp.exists() and payload.get("reuse_existing"):
        try:                                                  # R9a fallback
            with cp.open("rb") as fh:
                res = pickle.load(fh)
            if lineage is not None and res.get("pretrained_sha256") != lineage:
                print(f"[r22] {cp.name}: cached unit was initialised from a different "
                      f"pretrained state (lineage mismatch); retraining", flush=True)
            else:
                res["from_cache"] = True
                return res
        except Exception as exc:                              # noqa: BLE001
            print(f"[d9] {cp}: unusable cache ({type(exc).__name__}); removed, "
                  f"retraining", flush=True)
            cp.unlink(missing_ok=True)
    if cp is not None:                                        # R9b in-unit ckpt
        payload = dict(payload, ckpt_path=str(cp.with_suffix(".ckpt")))
    res = supervised_unit(payload)
    if lineage is not None:
        res["pretrained_sha256"] = lineage
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
                           LazyArchives(payload["val_files"]),     # D12: val is iterated once
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
    ev = payload.get("eval_only_state")
    if ev:
        # wp9 (e8.reuse_from): no training -- the state another run trained for
        # this seed and pool, refused unless its bytes are the ones recorded
        sp = Path(ev["path"])
        got = _file_sha256(sp)
        if got != ev["sha256"]:
            raise ValueError(f"{sp}: SHA-256 {got[:12]}... is not the recorded "
                             f"{ev['sha256'][:12]}...")
        sd = clean_state(torch.load(str(sp), map_location="cpu", weights_only=True))
        model.load_state_dict(sd, strict=True)
        model.to(pre.get("device", "cpu"))
        reused = True
    elif payload.get("reuse_existing") and sp.exists():
        # D9: consume a state produced by an earlier attempt of the SAME stamped
        # configuration (identical configurations are trained once); the file's
        # SHA-256 is returned so the report can chain attempt-1 -> attempt-2.
        # R9a: a corrupt (e.g. truncated) file is removed and retrained.
        try:
            sd = clean_state(torch.load(str(sp), map_location="cpu", weights_only=True))
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
        atomic_torch_save(deliverable_state(_state_dict(model)), sp)
        Path(pre["ckpt_path"]).unlink(missing_ok=True)          # unit complete
        resumed_from = hist.get("resumed_from_epoch")

    import hashlib

    out = {"state_path": str(sp), "reused_state": reused,
           "state_sha256": hashlib.sha256(sp.read_bytes()).hexdigest()}
    if ev:
        out["eval_only"] = True
    if not reused and resumed_from is not None:
        out["resumed_from_epoch"] = int(resumed_from)
    if payload.get("amplitude") or payload.get("eval_sets"):
        # wp9 (configuration block `evaluation`): the same metric suite from the
        # same predictions, with amplitude readings, on val and the holdouts
        from .w9_eval import evaluate_model_w9

        pred = torch_predictor(model, pre.get("device", "cpu"))
        amp = bool(payload.get("amplitude"))
        if payload.get("eval_val_files"):
            out["val"] = evaluate_model_w9(pred, LazyArchives(payload["eval_val_files"]), amp)
        out["holdouts"] = {name: evaluate_model_w9(pred, LazyArchives(files), amp)
                           for name, files in (payload.get("eval_sets") or {}).items()}
    elif payload.get("eval_val_files"):
        out["val"] = evaluate_model(
            torch_predictor(model, pre.get("device", "cpu")),
            LazyArchives(payload["eval_val_files"]))       # D12: iterated once
    return out
