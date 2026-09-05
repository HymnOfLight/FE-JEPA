"""Shared plumbing for the analysis tools: model construction from a config
and a state, instance iteration, kind-aware encoding, JSON output."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model_from_config(model_cfg: dict, state_path: str | None = None, seed: int = 0,
                            mode: str = "eval", device: str = "cpu"):
    """Build the configured model kind (`model_cfg["kind"]`, default fejepa)
    and, if given, strictly load a saved state.

    `mode="eval"` (the default) is for the measurement tools; a caller that
    trains next passes `mode="train"` -- the training loops also force
    train mode at entry, so this is about intent, not correctness."""
    import torch

    from ..experiments.parallel import _build_model

    model = _build_model({"kind": model_cfg.get("kind", "fejepa"), "model": model_cfg,
                          "seed": seed})
    if state_path:
        sd = torch.load(str(state_path), map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
    model.train(mode == "train")
    return model.to(resolve_device(device))


def instance_files(data_dir: str, n: int | None = None, split: dict | None = None,
                   subset: str = "pool") -> list:
    """Instance files of a corpus directory.

    With `split` (the run config's {"n_val": ..., "seed": ...}) and
    subset="val", returns the run's OWN held-out validation instances -- the
    set the E-series pre-registrations measure on. Without a split, returns
    the pool in manifest order. `n` truncates either list."""
    from ..experiments.protocol import load_split

    if split and subset == "val":
        sp = load_split(str(data_dir), int(split["n_val"]), seed=int(split.get("seed", 1)))
        files = sp.val_files
    elif split:
        sp = load_split(str(data_dir), int(split["n_val"]), seed=int(split.get("seed", 1)))
        files = sp.pool_files
    else:
        files = load_split(str(data_dir), 0, 1).pool_files
    return files if n is None else files[:n]


def encode_instance(model, arch, device: str | None = None):
    """Kind-aware encoding on the model's device: returns (tokens (rows, d), pack)."""
    import torch

    if device is None:
        device = str(next(model.parameters()).device)
    pack = model.prepare_instance(arch, device)
    with torch.no_grad():
        if getattr(model, "needs_pack", False):
            z = model.encode(pack["feats"], pack)
        else:
            z = model.encode(pack["feats"])
    return z.reshape(-1, z.shape[-1]), pack


def subsample_rows(x, max_rows: int, seed: int = 0):
    import torch

    if x.shape[0] <= max_rows:
        return x
    idx = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))[:max_rows]
    return x[idx]


def write_json(path: str | Path, obj: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
