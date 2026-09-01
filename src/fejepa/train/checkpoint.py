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
    """Restore the loop state in place. Returns (epochs_done, step, extra) or
    None when no usable checkpoint exists (a corrupt file is removed and
    reported, and training starts from scratch)."""
    import torch

    path = Path(path)
    if not path.exists():
        return None
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        model.to(device)
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched") is not None:
            sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]
        _restore_rng_states(ck["rng"])
        return int(ck["epochs_done"]), int(ck["step"]), dict(ck.get("extra") or {})
    except Exception as exc:                                  # noqa: BLE001
        print(f"[ckpt] {path}: unusable ({type(exc).__name__}: {exc}); "
              f"removed, training restarts from scratch", flush=True)
        try:
            path.unlink()
        except OSError:
            pass
        return None
