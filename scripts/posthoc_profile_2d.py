#!/usr/bin/env python3
"""wp8 Stage 1.36 post-hoc (item 5): why is a 2D training step ~5x slower now?

On the current box E1's AR steps ran at ~52 ms per step per process (base arm,
three units concurrently) and ~58 ms (the raw-ablation seed alone, AR + SIGReg
on the raw tokens); the August WP2 AR arm (code v2.1.5, plain AR, in-process,
the earlier container and torch 2.8) took 39 min for 204,800 steps (~11 ms).
This script times the step on the same box, set-up-free (differential: median
over pairs of runs of e1 and e2 epochs on n instances), for the code it is
imported from -- run it once with this checkout and once with a v2.1.5
worktree (`--src`) to separate code from machine (the torch stack stays the
box's: a torch regression would be booked as "machine"; the environment is
recorded) -- and, for the current code, under variants that undo one later
change each:

  default      plain AR, fresh model, run's TF32 policy, in-process (as the
               August reference ran)
  threads_w3   torch threads limited as an E1 worker (available CPUs // 3)
  no_ckpt      per-block activation checkpointing off (R17 / D13)
  resident     energy anchors resident on the GPU instead of streamed (R11 / D10)
  raw_sigreg   AR + SIGReg on the raw tokens (lambda 1, 256 projections), the
               raw ablation's loss

plus a torch.profiler table of the default AR step. Supervised (labels-only)
steps are timed too: they share the encoder but neither the anchors nor the
AR loss, so a supervised step that slowed by the same factor puts the cause in
the shared forward/backward path. Reported only.

    python scripts/posthoc_profile_2d.py --out runs/wp8/posthoc/profile_2d_head.json
    python scripts/posthoc_profile_2d.py --src ../FE-JEPA-v215/src --variants default \
        --out runs/wp8/posthoc/profile_2d_v215.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path


def _setup_path(src: str | None) -> str:
    path = str(Path(src).resolve()) if src else str(Path(__file__).resolve().parents[1] / "src")
    sys.path.insert(0, path)
    return path


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def _available_cpus() -> dict:
    """CPUs this process may use: affinity and the cgroup quota (a container's
    os.cpu_count() can report the host)."""
    out = {"os_cpu_count": os.cpu_count()}
    try:
        out["affinity"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        out["affinity"] = None
    try:
        q, per = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        out["cgroup_cpu_max"] = None if q == "max" else round(int(q) / int(per), 2)
    except (OSError, ValueError):
        out["cgroup_cpu_max"] = None
    cands = [v for v in (out["affinity"], out["cgroup_cpu_max"], out["os_cpu_count"]) if v]
    out["usable"] = max(1, int(min(cands))) if cands else 1
    return out


def _git(path: str, *args) -> str:
    import subprocess

    try:
        return subprocess.run(["git", "-C", path, *args], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unavailable"
    except Exception:                                     # noqa: BLE001
        return "unavailable"


def _driver() -> str:
    import subprocess

    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                     # noqa: BLE001
        return "unavailable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="import fejepa from this src dir (e.g. a "
                                                  "v2.1.5 worktree); default: this checkout")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1]
                                            / "configs" / "e1_2d_base.json"))
    ap.add_argument("--data", default="runs/data2d")
    ap.add_argument("--n-inst", type=int, default=32)
    ap.add_argument("--e1", type=int, default=2)
    ap.add_argument("--e2", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--variants", nargs="+",
                    default=["default", "threads_w3", "no_ckpt", "resident", "raw_sigreg"])
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-sup", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = _setup_path(a.src)

    import torch

    import fejepa
    import fejepa.train.pretrain as pt
    from fejepa.data.archive import load_instance
    from fejepa.experiments.parallel import _build_model
    from fejepa.experiments.protocol import load_split
    from fejepa.runtime import setup_torch
    from fejepa.train.losses import AR_CONFIG
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    cfg = json.loads(Path(a.config).read_text())
    split = cfg.get("split", {"n_val": 256, "seed": 1})
    sp = load_split(a.data, int(split["n_val"]), seed=int(split.get("seed", 1)))
    archs = [load_instance(f) for f in sp.pool_files[:a.n_inst]]
    val1 = [load_instance(sp.val_files[0])]
    policy = setup_torch(a.device, tf32=bool(cfg.get("tf32", True)))
    base_threads = torch.get_num_threads()
    lr = float(((cfg.get("experiments") or {}).get("e8") or {}).get("ar_lr", 1e-3))

    def sync():
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()

    def fresh(variant):
        m = _build_model({"kind": cfg["model"].get("kind", "fejepa"), "model": cfg["model"],
                          "seed": 0})
        if variant == "no_ckpt":
            import inspect
            enc = getattr(m, "encoder", None)
            if enc is None or "use_checkpoint" not in inspect.getsource(type(enc).forward):
                return None                      # this code has no activation checkpointing
            enc.use_checkpoint = False
        return m

    loss_of = {"current": AR_CONFIG}

    def run_ar(model, epochs):
        sync()
        t0 = time.perf_counter()
        pt.pretrain(model, archs, pt.PretrainConfig(loss=loss_of["current"], epochs=epochs,
                                                    lr=lr, device=a.device, log_every=-1, seed=0))
        sync()
        return time.perf_counter() - t0

    def run_sup(model, epochs):
        sync()
        t0 = time.perf_counter()
        train_supervised(model, archs, val1, SupervisedConfig(epochs=epochs, device=a.device,
                                                              log_every=-1, seed=0))
        sync()
        return time.perf_counter() - t0

    def differential(fn, variant):
        est = []
        for _ in range(a.pairs):
            m = fresh(variant)
            if m is None:
                return None
            fn(m, 1)                                            # warm-up
            t1 = fn(fresh(variant), a.e1)
            t2 = fn(fresh(variant), a.e2)
            est.append((t2 - t1) / ((a.e2 - a.e1) * len(archs)) * 1000.0)
        return {"ms_per_step": round(statistics.median(est), 3),
                "estimates_ms": [round(x, 3) for x in est]}

    cpus = _available_cpus()
    res = {"what": "2D step timing (wp8 Stage 1.36 item 5); reported only",
           "src": src, "fejepa_file": fejepa.__file__,
           "src_git": _git(str(Path(src).parent), "describe", "--tags", "--always", "--dirty"),
           "src_head": _git(str(Path(src).parent), "rev-parse", "HEAD"), "torch": torch.__version__,
           "cuda": torch.version.cuda, "driver": _driver() if a.device.startswith("cuda") else None,
           "numeric_policy": policy, "cpu": _cpu_model(), "cpus": cpus,
           "threads_default": base_threads,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
           "n_inst": len(archs), "median_nodes": int(statistics.median(x.n_nodes for x in archs)),
           "epochs": [a.e1, a.e2], "pairs": a.pairs, "variants": {}}
    orig_cache = pt.AnchorCache
    for v in a.variants:
        torch.set_num_threads(max(1, cpus["usable"] // 3) if v == "threads_w3"
                              else base_threads)
        loss_of["current"] = AR_CONFIG
        if v == "raw_sigreg":
            try:
                from fejepa.train.losses import ar_sigreg_config
            except ImportError:
                res["variants"][v] = "not applicable (this code has no SIGReg loss)"
                continue
            loss_of["current"] = ar_sigreg_config(1.0, head=False, n_proj=256)
        if v == "resident":
            import inspect
            if "resident" not in inspect.signature(orig_cache.__init__).parameters:
                res["variants"][v] = "not applicable (this code keeps anchors resident)"
                continue
            pt.AnchorCache = lambda device="cpu", dtype=None: orig_cache(  # noqa: E731
                device=device, dtype=dtype, resident="device")
        row = {"ar": differential(run_ar, v)}
        if not a.no_sup:
            row["sup"] = (differential(run_sup, v) if v in ("default", "threads_w3", "no_ckpt")
                          else "not timed (the supervised loss uses neither anchors nor SIGReg)")
        pt.AnchorCache = orig_cache
        loss_of["current"] = AR_CONFIG
        res["variants"][v] = row if row["ar"] is not None else "not applicable (no such switch here)"
        print(json.dumps({v: res["variants"][v]}), flush=True)
    torch.set_num_threads(base_threads)
    if not a.no_profile and a.device.startswith("cuda"):
        from torch.profiler import ProfilerActivity, profile

        m = fresh("default")
        run_ar(m, 1)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_ar(fresh("default"), 3)        # 3 epochs: the set-up epoch is a third of it
        ka = prof.key_averages()
        res["profile_self_cpu"] = ka.table(sort_by="self_cpu_time_total", row_limit=20)
        res["profile_self_cuda"] = ka.table(sort_by="self_cuda_time_total", row_limit=20)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("profile")}))


if __name__ == "__main__":
    main()
