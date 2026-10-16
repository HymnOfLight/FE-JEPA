"""R9 (Phase-2 D9 hardening): atomic file writes and epoch-boundary training
checkpoints with exact resume.

Atomic writes: every durable artefact (states, unit caches, checkpoints,
instance archives) is written to a sibling temp file and moved into place
with os.replace, so a power cut leaves either the old file or the new one --
never a truncated one.

Epoch checkpoints capture EVERY state that evolves across an epoch boundary:
parameters, optimiser moments, scheduler counter, the numpy generator that
draws the per-epoch order and loss-side randomness, the torch RNG (CPU and
all CUDA devices), the step counter, and loop accumulators. Restoring all of
them makes the continued trajectory identical to the uninterrupted one on
deterministic backends (tests assert bitwise equality on CPU).
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_torch_save(obj, path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_pickle_dump(obj, path) -> None:
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


def _rng_states():
    import torch

    out = {"torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def _restore_rng_states(saved) -> None:
    import torch

    torch.set_rng_state(saved["torch_cpu"])
    if "torch_cuda" in saved and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["torch_cuda"])


def save_epoch_checkpoint(path, *, epochs_done: int, step: int, model, opt,
                          sched, rng, extra: dict | None = None) -> None:
    """Write the full loop state after `epochs_done` completed epochs."""
    from ..experiments.parallel import _state_dict

    payload = {"epochs_done": int(epochs_done), "step": int(step),
               "model": _state_dict(model), "opt": opt.state_dict(),
               "sched": sched.state_dict() if sched is not None else None,
               "np_rng": rng.bit_generator.state, "rng": _rng_states(),
               "extra": extra or {}}
    atomic_torch_save(payload, path)


def load_epoch_checkpoint(path, *, model, opt, sched, rng, device):
    """Restore the loop state in place. Returns (epochs_done, step, extra), or
    None when no checkpoint exists or its file cannot be read: such a file is
    moved aside (<name>.unusable.<time>) and reported, nothing has been
    restored, and training starts from scratch. A file that reads but cannot be
    restored (a state of another model, or an error on the device, e.g. a
    failing GPU) raises: the run stops and the checkpoint stays for the next
    attempt (fine3d r5: restoring part of it and then training "from scratch"
    would train from a mixed state, and a device error would discard hours of
    training)."""
    import time

    import torch

    path = Path(path)
    if not path.exists():
        return None
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        state = (int(ck["epochs_done"]), int(ck["step"]), ck["model"], ck["opt"],
                 ck.get("sched"), ck["np_rng"], ck["rng"], dict(ck.get("extra") or {}))
    except Exception as exc:                                  # noqa: BLE001
        aside = path.with_name(f"{path.name}.unusable.{time.strftime('%Y%m%d-%H%M%S')}")
        try:
            os.replace(path, aside)
            where = f"moved to {aside.name}"
        except OSError:
            where = "left in place (overwritten at the next checkpoint)"
        print(f"[ckpt] {path}: unusable ({type(exc).__name__}: {exc}); {where}, "
              f"training restarts from scratch", flush=True)
        return None
    epochs_done, step, model_sd, opt_sd, sched_sd, np_rng, rng_states, extra = state
    model.load_state_dict(model_sd, strict=True)
    model.to(device)
    opt.load_state_dict(opt_sd)
    if sched is not None and sched_sd is not None:
        sched.load_state_dict(sched_sd)
    rng.bit_generator.state = np_rng
    _restore_rng_states(rng_states)
    return epochs_done, step, extra
