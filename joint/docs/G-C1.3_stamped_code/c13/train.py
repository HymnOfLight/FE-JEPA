"""Train one arm and seed of gate G-C1.3, with checkpoints and resume, then predict the held-out and tested
instances. The conventions follow the repository's pretraining loop (fejepa.train.pretrain): AdamW, gradient
clipping, the cosine schedule with warm-up (fejepa.train.schedule.make_scheduler), one instance per step, a fresh
random order every epoch, epoch-boundary checkpoints.

A run that stops on a non-finite loss, or whose predictions are not finite, is final: its history.json records the
stop and a later call returns it without training. A finished run with its predictions returns at once. Every call
appends its start, any resume and its end to attempts.log in the run folder."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .data import load_export, load_label, manifest
from .model import features, build, field

MATMUL_PRECISION = "highest"          # float32 matmuls without TF32 (the configuration's precision)


def verify_repository(cfg: dict) -> str:
    """The repository files the run imports must have the SHA-256 the configuration pins. Checked before any of them
    is imported. Returns the repository root."""
    import importlib.util

    spec = importlib.util.find_spec("fejepa")
    if spec is None or spec.origin is None:
        raise RuntimeError("the repository package fejepa is not on the path (--repo)")
    root = Path(spec.origin).resolve().parents[2]
    bad = [f for f, h in cfg["repository_code"].items()
           if not (root / f).is_file() or hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
    if bad:
        raise RuntimeError(f"repository files differ from the configuration: {bad}")
    return str(root)


def _note(run: Path, text: str) -> None:
    with open(run / "attempts.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} {text}\n")
        fh.flush()


def _write_history(hp: Path, hist: dict) -> None:
    tmp = hp.with_suffix(".tmp")
    tmp.write_text(json.dumps(hist, indent=1))
    tmp.replace(hp)


def _check_manifest(cfg: dict, data_dir) -> dict:
    from .config import sha256

    man = manifest(data_dir)
    if man["config_sha256"] != sha256(cfg):
        raise RuntimeError("the data were generated under another configuration")
    return man


def _load_state(path):
    """A checkpoint, loaded on the CPU whatever the training device: the RNG states must stay CPU tensors, and
    load_state_dict moves the model's and the optimizer's tensors to their parameters' device."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _prepare(cfg, data_dir, ids, device, with_labels):
    import torch
    from fejepa.anchor.energy import EnergyAnchor

    out = []
    for iid in ids:
        ex, q = load_export(Path(data_dir) / "inst" / f"{iid}.npz")
        it = {"id": iid, "feats": torch.as_tensor(features(ex, q), dtype=torch.float32, device=device).unsqueeze(0),
              "anchor": EnergyAnchor(ex["K"], ex["F"], ex["dirichlet"], device=device),
              "nonneg": torch.as_tensor(ex["nonneg"], device=device),
              "free": torch.as_tensor(~ex["dirichlet"], dtype=torch.float32, device=device),
              "scale": float(ex["u_scale"]),
              "e_scale": 0.5 * (0.5 * cfg["data"]["P_full_N"]) * float(ex["u_scale"])}
        if with_labels:
            it["u_ref"] = torch.as_tensor(load_label(data_dir, iid)["u"], dtype=torch.float32, device=device)
        out.append(it)
    return out


def train(cfg: dict, arm: str, seed: int, data_dir, out_dir, device: str = "cuda", ckpt_every_epochs: int = 5,
          stop_after_epoch: int | None = None) -> dict:
    """Train (or resume) and write out_dir/<arm>_s<seed>/{state.pt, history.json, preds.npz}."""
    from .config import sha256

    a = cfg["arms"][arm]
    assert seed in a["seeds"], "seed not in the configuration"
    man = _check_manifest(cfg, data_dir)
    cfg_sha = sha256(cfg)
    run = Path(out_dir) / f"{arm}_s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    _note(run, "start")
    print(f"start {arm}_s{seed} {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)
    hp = run / "history.json"
    if hp.exists():
        old = json.loads(hp.read_text())
        if old.get("config_sha256") != cfg_sha:
            raise RuntimeError(f"{run} holds a run of another configuration; use a fresh runs directory")
        if str(old.get("status", "")).startswith("non-finite"):
            _note(run, "returned: " + old["status"])
            return old                                       # final: never retried
        if old.get("status") == "finished" and (run / "preds.npz").exists():
            _note(run, "returned: finished")
            return old
    verify_repository(cfg)
    import torch
    from fejepa.train.schedule import make_scheduler

    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    train_ids = [r["id"] for r in man["instances"] if r["split"] == "train"]
    assert len(train_ids) == cfg["data"]["n_train"], "the manifest does not hold the configured training set"
    sup = a["loss"] == "supervised"
    ids = train_ids[:a["labels"]] if sup else train_ids
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    items = _prepare(cfg, data_dir, ids, dev, with_labels=sup)
    enc, dec = build(cfg, items[0]["feats"].shape[-1])
    enc.to(dev); dec.to(dev)
    params = list(enc.parameters()) + list(dec.parameters())
    tr = cfg["training"]
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    total = a["epochs"] * len(items)
    sched = make_scheduler(opt, total)
    T = cfg["map"]["T"]
    hist = {"config_sha256": cfg_sha, "arm": arm, "seed": seed, "torch": torch.__version__, "device": str(dev),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "matmul_precision": torch.get_float32_matmul_precision(), "total_steps": total,
            "loss": [], "epoch_seconds": [], "resumes": [], "resumed_from_epoch": None}
    start_epoch, step = 0, 0
    ck = run / "state.pt"
    if ck.exists():
        s = _load_state(ck)
        if s["history"].get("config_sha256") != cfg_sha:
            raise RuntimeError(f"{ck} belongs to another configuration; use a fresh runs directory")
        enc.load_state_dict(s["enc"]); dec.load_state_dict(s["dec"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); rng.bit_generator.state = s["rng"]; torch.set_rng_state(s["torch_rng"])
        start_epoch, step, hist = s["epoch"], s["step"], s["history"]
        hist["resumed_from_epoch"] = start_epoch
        hist.setdefault("resumes", []).append(start_epoch)
        _note(run, f"resumed from epoch {start_epoch}")
        if start_epoch >= a["epochs"]:                       # finished before; complete what may be missing
            hist.setdefault("status", "finished"); hist.setdefault("steps", step)
            if not hp.exists():
                _write_history(hp, hist)
            if not (run / "preds.npz").exists() and not predict(cfg, enc, dec, data_dir, run / "preds.npz", dev):
                hist["status"] = "non-finite predictions"
                _write_history(hp, hist)
            _note(run, "ended: " + hist["status"])
            return hist
    enc.train(); dec.train()
    status = "running"
    for epoch in range(start_epoch, a["epochs"]):
        t0 = time.perf_counter()
        order = rng.permutation(len(items))
        loss_sum = 0.0
        for i in order:
            it = items[i]
            u = field(enc, dec, it["feats"], it["nonneg"], it["free"], it["scale"], T)
            if sup:
                loss = (((u - it["u_ref"]) / it["scale"]) ** 2).mean()
            else:
                loss = it["anchor"](u.unsqueeze(0)) / it["e_scale"]
            if not torch.isfinite(loss):
                status = f"non-finite loss at step {step}"
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tr["clip"])
            opt.step()
            sched.step()
            step += 1
            loss_sum += float(loss.detach())
        if status != "running":
            break
        if dev.type == "cuda":
            torch.cuda.synchronize()
        hist["loss"].append(loss_sum / len(items))
        hist["epoch_seconds"].append(round(time.perf_counter() - t0, 2))
        done = epoch + 1
        if done % ckpt_every_epochs == 0 or done == a["epochs"]:
            tmp = ck.with_suffix(".tmp")
            torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                        "epoch": done, "step": step, "history": hist}, tmp)
            tmp.replace(ck)
        if stop_after_epoch is not None and done >= stop_after_epoch:
            return hist
    hist["status"] = "finished" if status == "running" else status
    hist["steps"] = step
    if dev.type == "cuda":
        hist["peak_memory_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
    _write_history(hp, hist)
    if hist["status"] == "finished" and not predict(cfg, enc, dec, data_dir, run / "preds.npz", dev):
        hist["status"] = "non-finite predictions"
        _write_history(hp, hist)
    _note(run, "ended: " + hist["status"])
    return hist


def predict(cfg, enc, dec, data_dir, out_path, dev) -> bool:
    """Fields of the held-out and tested instances, nodal vectors keyed by instance id. Stored as float32, the
    network's own precision, so nothing is lost; the evaluation works in float64. If any value is not finite, nothing
    is written and False is returned."""
    import torch

    man = manifest(data_dir)
    ids = [r["id"] for r in man["instances"] if r["split"] in cfg["evaluation"]["sets"]]
    enc.eval(); dec.eval()
    preds = {}
    with torch.no_grad():
        for it in _prepare(cfg, data_dir, ids, dev, with_labels=False):
            u = field(enc, dec, it["feats"], it["nonneg"], it["free"], it["scale"], cfg["map"]["T"])
            preds[it["id"]] = u.float().cpu().numpy()
    if not all(np.isfinite(v).all() for v in preds.values()):
        return False
    tmp = Path(str(out_path) + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **preds)
    tmp.replace(out_path)
    return True


def bench(cfg: dict, data_dir, device: str = "cuda", instances: int = 8, warmup: int = 5, steps: int = 40) -> dict:
    """Seconds per label-free training step of the configured model on the first training instances (no labels,
    no change to any run). The repository's runner takes one such step per instance visit."""
    import platform

    man = _check_manifest(cfg, data_dir)
    verify_repository(cfg)
    import torch
    from fejepa.train.schedule import make_scheduler

    torch.set_float32_matmul_precision(MATMUL_PRECISION)
    ids = [r["id"] for r in man["instances"] if r["split"] == "train"][:instances]
    dev = torch.device(device)
    torch.manual_seed(0)
    items = _prepare(cfg, data_dir, ids, dev, with_labels=False)
    enc, dec = build(cfg, items[0]["feats"].shape[-1])
    enc.to(dev); dec.to(dev); enc.train(); dec.train()
    params = list(enc.parameters()) + list(dec.parameters())
    tr = cfg["training"]
    opt = torch.optim.AdamW(params, lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = make_scheduler(opt, warmup + steps)
    T = cfg["map"]["T"]

    def one(it):
        u = field(enc, dec, it["feats"], it["nonneg"], it["free"], it["scale"], T)
        loss = it["anchor"](u.unsqueeze(0)) / it["e_scale"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, tr["clip"])
        opt.step(); sched.step()
        return float(loss.detach())

    losses = [one(items[i % len(items)]) for i in range(warmup)]
    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        losses.append(one(items[i % len(items)]))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
            "torch": torch.__version__, "python": platform.python_version(),
            "matmul_precision": torch.get_float32_matmul_precision(), "threads": torch.get_num_threads(),
            "instances": ids, "nodes": [int(it["feats"].shape[1]) for it in items],
            "seconds_per_step_mean": float(np.mean(times)), "seconds_per_step_median": float(np.median(times)),
            "seconds_per_step_max": float(np.max(times)), "losses": losses,
            "losses_finite": bool(np.isfinite(losses).all()),
            "peak_memory_GiB": (torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == "cuda" else None}
