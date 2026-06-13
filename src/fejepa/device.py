"""Device selection: a single ``device`` parameter for CPU / GPU / auto.

All training and experiment entry points accept a ``device`` string:

* ``"auto"`` (default) -- use CUDA if available, otherwise CPU,
* ``"cpu"``            -- force CPU,
* ``"cuda"`` / ``"cuda:N"`` -- force a specific GPU (falls back to CPU with a
  warning if CUDA is unavailable, so configs are portable across machines).
"""

from __future__ import annotations

import warnings

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def resolve_device(device: str | None = "auto") -> str:
    """Resolve a device string to a concrete ``"cpu"`` or ``"cuda[:N]"`` value."""

    if device is None or device == "auto":
        return "cuda" if cuda_available() else "cpu"
    if str(device).startswith("cuda") and not cuda_available():
        warnings.warn(
            f"device={device!r} requested but CUDA is unavailable; falling back to CPU.",
            stacklevel=2,
        )
        return "cpu"
    return str(device)


def describe_device(device: str) -> str:
    """Human-readable description of a resolved device for logging."""

    if device.startswith("cuda") and cuda_available():
        idx = 0
        if ":" in device:
            try:
                idx = int(device.split(":", 1)[1])
            except ValueError:
                idx = 0
        try:
            name = torch.cuda.get_device_name(idx)
            return f"{device} ({name})"
        except Exception:  # noqa: BLE001
            return device
    return device
