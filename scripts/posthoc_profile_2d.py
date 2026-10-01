#!/usr/bin/env python3
"""wp8 post-hoc (wrap-up item 5; Stage 1.36, worker variants Stage 1.37): why
is a 2D training step ~5x slower now?

On the current box E1's AR steps ran at ~52 ms per step per unit (base arm:
three units at once, each in a spawned worker) and ~58 ms (the raw-ablation
seed alone in a worker, AR + SIGReg on the raw tokens); the August WP2 AR arm
(code v2.1.5, plain AR, IN-PROCESS, the earlier container, same model
configuration and corpus) took 39 min for 204,800 steps (~11 ms). This script
times the step on the same box, set-up-free (differential: median over pairs
of runs of e1 and e2 epochs on n instances), for the code it is imported from
-- run it once with this checkout and once with a v2.1.5 worktree (`--src`;
the script refuses to run if `fejepa` is not imported from that directory)
to separate code from machine (the torch stack stays the box's: a torch
regression would be booked as "machine"; the environment is recorded) --
under variants:

  default        plain AR, fresh model, run's TF32 policy, in-process (as the
                 August reference ran)
  threads_w3     in-process, torch threads = usable CPUs // 3 (affinity and
                 cgroup quota)
  threads_e1     in-process, torch threads = os.cpu_count() // 3 -- the
                 formula E1's workers used (map_units); in a container
                 os.cpu_count() can report the host
  worker         one AR unit through map_units(pretrain_unit, workers=3):
                 a spawned worker with E1's thread setting (OMP_NUM_THREADS
                 and torch threads), E1's unit code path (in-unit checkpoints
                 included) -- the raw arm's situation with the AR loss
  worker3        three AR units at once through map_units (E1's base arm)
  worker3_quota  as worker3 with FEJEPA_WORKER_THREADS = usable // 3 (the
                 candidate remedy)
  no_ckpt        in-process, per-block activation checkpointing off (R17)
  resident       in-process, energy anchors resident on the GPU (R11 / D10)
  raw_sigreg     in-process, AR + SIGReg on the raw tokens (lambda 1, 256
                 projections), the raw ablation's loss

plus a torch.profiler table of the default AR step. Supervised (labels-only)
steps are timed for default / threads_w3 / no_ckpt: they share the encoder
but neither the anchors nor the AR loss. The output is rewritten after every
variant. Reported only.

    python scripts/posthoc_profile_2d.py --out runs/wp8/posthoc/profile_2d_head.json
    python scripts/posthoc_profile_2d.py --src ../FE-JEPA-v215/src \
        --variants default threads_w3 threads_e1 worker3 --out runs/wp8/posthoc/profile_2d_v215.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

VARIANTS = ("default", "threads_w3", "threads_e1", "worker", "worker3", "worker3_quota",
            "no_ckpt", "resident", "raw_sigreg")
WORKER_VARIANTS = {"worker": (1, False), "worker3": (3, False), "worker3_quota": (3, True)}
SUP_VARIANTS = ("default", "threads_w3", "no_ckpt")


def _setup_path(src: str | None) -> str:
    path = str(Path(src).resolve()) if src else str(Path(__file__).resolve().parents[1] / "src")
    if not Path(path, "fejepa", "__init__.py").exists():
        raise SystemExit(f"--src {src}: no fejepa package under {path} (did `git worktree "
                         "add` fail?)")
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
    out["e1_worker_threads"] = int(os.environ.get("FEJEPA_WORKER_THREADS",
                                                  max(1, (os.cpu_count() or 8) // 3)))
    out["env_OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS")
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


def _write(path: str, res: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, p)


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
    ap.add_argument("--worker-e1", type=int, default=2,
                    help="worker variants: epochs of the short run (spawn jitter is seconds)")
    ap.add_argument("--worker-e2", type=int, default=22)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS,
                    default=["default", "threads_w3", "threads_e1", "worker", "worker3",
                             "worker3_quota", "no_ckpt", "resident", "raw_sigreg"])
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-sup", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = _setup_path(a.src)

    import torch

    import fejepa
    if not Path(fejepa.__file__).resolve().is_relative_to(Path(src)):
        raise SystemExit(f"fejepa was imported from {fejepa.__file__}, not from {src}: "
                         "refusing to time the wrong code")
    import fejepa.train.pretrain as pt
    from fejepa.data.archive import load_instance
    from fejepa.experiments.parallel import _build_model, map_units, pretrain_unit
    from fejepa.experiments.protocol import load_split
    from fejepa.runtime import setup_torch
    from fejepa.train.losses import AR_CONFIG
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    cfg = json.loads(Path(a.config).read_text())
    split = cfg.get("split", {"n_val": 256, "seed": 1})
    sp = load_split(a.data, int(split["n_val"]), seed=int(split.get("seed", 1)))
    files = list(sp.pool_files[:a.n_inst])
    archs = [load_instance(f) for f in files]
    val1 = [load_instance(sp.val_files[0])]
    tf32 = bool(cfg.get("tf32", True))
    policy = setup_torch(a.device, tf32=tf32)
    base_threads = torch.get_num_threads()
    lr = float(((cfg.get("experiments") or {}).get("e8") or {}).get("ar_lr", 1e-3))
    kind = cfg["model"].get("kind", "fejepa")
    tmpdir = Path(tempfile.mkdtemp(prefix="profile2d_"))

    def sync():
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()

    def fresh(variant):
        m = _build_model({"kind": kind, "model": cfg["model"], "seed": 0})
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
                "estimates_ms": [round(x, 3) for x in est], "valid": all(x > 0 for x in est)}

    counter = {"n": 0}

    def run_workers(units, epochs, quota):
        """Wall time of `units` AR units run at once through map_units(workers=3),
        each on the same instances (E1's unit code path and worker set-up)."""
        payloads = []
        for k in range(units):
            counter["n"] += 1
            payloads.append({"kind": kind, "model": cfg["model"], "seed": k, "tf32": tf32,
                             "files": [str(f) for f in files], "loss": "ar",
                             "pre": {"epochs": epochs, "lr": lr, "device": a.device,
                                     "desc": f"profile worker {k}"},
                             "state_path": str(tmpdir / f"w{counter['n']}_{k}.pt"),
                             "tag": f"profile {k}"})
        old = os.environ.get("FEJEPA_WORKER_THREADS")
        if quota:
            os.environ["FEJEPA_WORKER_THREADS"] = str(max(1, cpus["usable"] // 3))
        try:
            t0 = time.perf_counter()
            map_units(pretrain_unit, payloads, 3, "profile workers")
            return time.perf_counter() - t0
        finally:
            if quota:
                if old is None:
                    os.environ.pop("FEJEPA_WORKER_THREADS", None)
                else:
                    os.environ["FEJEPA_WORKER_THREADS"] = old
            for pl in payloads:                    # states are ~25 MB each: do not pile up
                sp_ = Path(pl["state_path"])
                sp_.unlink(missing_ok=True)
                sp_.with_suffix(".ckpt").unlink(missing_ok=True)

    def differential_workers(units, quota):
        est = []
        for _ in range(a.pairs):
            t1 = run_workers(units, a.worker_e1, quota)
            t2 = run_workers(units, a.worker_e2, quota)
            est.append((t2 - t1) / ((a.worker_e2 - a.worker_e1) * len(archs)) * 1000.0)
        return {"ms_per_step_per_unit": round(statistics.median(est), 3),
                "estimates_ms": [round(x, 3) for x in est], "valid": all(x > 0 for x in est),
                "units": units,
                "threads_per_worker": (max(1, cpus["usable"] // 3) if quota
                                       else cpus["e1_worker_threads"])}

    cpus = _available_cpus()
    res = {"what": "2D step timing (wp8 wrap-up item 5); reported only",
           "src": src, "fejepa_file": fejepa.__file__,
           "src_git": _git(str(Path(src).parent), "describe", "--tags", "--always", "--dirty"),
           "src_head": _git(str(Path(src).parent), "rev-parse", "HEAD"), "torch": torch.__version__,
           "cuda": torch.version.cuda, "driver": _driver() if a.device.startswith("cuda") else None,
           "numeric_policy": policy, "cpu": _cpu_model(), "cpus": cpus,
           "threads_default": base_threads,
           "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") else None,
           "n_inst": len(archs), "median_nodes": int(statistics.median(x.n_nodes for x in archs)),
           "epochs": [a.e1, a.e2], "worker_epochs": [a.worker_e1, a.worker_e2],
           "pairs": a.pairs, "variants": {}}
    orig_cache = pt.AnchorCache
    for v in a.variants:
        try:
            if v in WORKER_VARIANTS:
                units, quota = WORKER_VARIANTS[v]
                res["variants"][v] = {"ar": differential_workers(units, quota)}
                print(json.dumps({v: res["variants"][v]}), flush=True)
                _write(a.out, res)
                continue
            threads = {"threads_w3": max(1, cpus["usable"] // 3),
                       "threads_e1": cpus["e1_worker_threads"]}.get(v, base_threads)
            torch.set_num_threads(threads)
            loss_of["current"] = AR_CONFIG
            if v == "raw_sigreg":
                try:
                    from fejepa.train.losses import ar_sigreg_config
                except ImportError:
                    res["variants"][v] = "not applicable (this code has no SIGReg loss)"
                    _write(a.out, res)
                    continue
                loss_of["current"] = ar_sigreg_config(1.0, head=False, n_proj=256)
            if v == "resident":
                import inspect
                if "resident" not in inspect.signature(orig_cache.__init__).parameters:
                    res["variants"][v] = "not applicable (this code keeps anchors resident)"
                    _write(a.out, res)
                    continue
                pt.AnchorCache = lambda device="cpu", dtype=None: orig_cache(  # noqa: E731
                    device=device, dtype=dtype, resident="device")
            row = {"ar": differential(run_ar, v), "torch_threads": threads}
            if not a.no_sup:
                row["sup"] = (differential(run_sup, v) if v in SUP_VARIANTS
                              else "not timed (the supervised loss uses neither anchors nor SIGReg)")
            res["variants"][v] = row if row["ar"] is not None else "not applicable (no such switch here)"
        except Exception as exc:                          # noqa: BLE001 -- keep the others
            res["variants"][v] = {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            pt.AnchorCache = orig_cache
            loss_of["current"] = AR_CONFIG
            torch.set_num_threads(base_threads)
        print(json.dumps({v: res["variants"][v]}), flush=True)
        _write(a.out, res)
    if not a.no_profile and a.device.startswith("cuda"):
        try:
            from torch.profiler import ProfilerActivity, profile

            m = fresh("default")
            run_ar(m, 1)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                run_ar(fresh("default"), 3)    # 3 epochs: the set-up epoch is a third of it
            ka = prof.key_averages()
            res["profile_self_cpu"] = ka.table(sort_by="self_cpu_time_total", row_limit=20)
            res["profile_self_cuda"] = ka.table(sort_by="self_cuda_time_total", row_limit=20)
        except Exception as exc:                          # noqa: BLE001
            res["profile_error"] = f"{type(exc).__name__}: {exc}"
    _write(a.out, res)
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("profile")}))


if __name__ == "__main__":
    main()
