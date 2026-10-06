#!/usr/bin/env python3
"""cmame-paper Stage 2: the checks before CM2D's run (RUNBOOK_CMAME Sec. B;
PREREG_CM2D Sec. 6). Prints one line per check and last `GO` or
`STOP: <checks>`; writes the same to --out as JSON; exits 0 on GO, 3 on STOP.
It trains nothing that is kept and writes nothing but --out.

  stack        nothing else on the GPU (under 1,000 MiB in use) and no other
               run (no fejepa process, worker or script of this repository); the
               package `python -m fejepa.cli` imports is this checkout's; E1's
               torch version with CUDA; 7 GB free on the disk of runs/ (5 GB
               before a restart); checked first, before this process touches CUDA;
  checkout     HEAD is the tag `prereg-cm2d` -- `git describe --tags --match
               prereg-cm2d` prints exactly that, and so does `git describe
               --always --dirty --tags`, the string the report records -- with
               no local changes to tracked files;
  config       configs/cm2d_v1.json is the generator's (scripts/make_cm2d_config.py);
  dry run      `run-config --dry-run` verifies the stamp on PREREG_CM2D.md and
               E1's three states (SHA-256 against E1's report) and plans the
               grid with a labelled pool prefix of 1,024;
  corpus       the configuration's corpus is E1's (manifest SHA-256 from E1's
               report), and every validation instance and the first 1,024 pool
               instances carry labels;
  reproduction E1's three states, evaluated on the validation split by the
               run's own unit function, reproduce E1's per-instance relative
               energy gaps and displacement errors (largest relative deviation
               at most 1e-4);
  smoke        one epoch of stiffness-norm training on two pool instances on
               the run's device gives finite values (nothing is kept);
               (reproduction and smoke use the GPU: not run when it is busy;)
  fresh        no earlier attempt under the run's directory (no report,
               e8_states/, run.log or status.txt); with --restart: no report.

`--pre-stamp`: the readiness check for a visit before the stamp -- the tag,
the describe string and the stamp are reported, not required.
`--restart`: before a restart with --reuse-states (RUNBOOK_CMAME Sec. B3).

    python scripts/cm2d_precheck.py --out runs/cm2d/precheck.json
    python scripts/cm2d_precheck.py --restart --out runs/cm2d/precheck_restart.json
    python scripts/cm2d_precheck.py --pre-stamp --out runs/cmame/cm2d_ready.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TAG = "prereg-cm2d"
CONFIG = "configs/cm2d_v1.json"
E1_REPORT = "records/wp8/e1/e1_2d_base/report.json"
FREE_GB = 7.0
FREE_GB_RESTART = 5.0
OTHER_RUNS = r"fejepa|spawn_main|scripts/[A-Za-z0-9_]+\.py"
GPU_IDLE_MIB = 1000
REPRO_METRICS = ("energy_gap_rel", "disp_rel_l2")
REPRO_MAX = 1e-4


def _run(args, cwd=ROOT, timeout=60, env=None) -> tuple:
    try:
        p = subprocess.run(args, capture_output=True, text=True, cwd=cwd, timeout=timeout,
                           env=env)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return None, "", f"{type(exc).__name__}: {exc}"


def _generator():
    spec = importlib.util.spec_from_file_location("make_cm2d_config",
                                                  ROOT / "scripts" / "make_cm2d_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                     # noqa: BLE001
        return "cpu"


def stack_checks(e1_report: Path, free_gb: float = FREE_GB) -> dict:
    out = {}
    rc, used, err = _run(["nvidia-smi", "--query-gpu=memory.used,name",
                          "--format=csv,noheader,nounits"])
    try:
        mib = float(used.splitlines()[0].split(",")[0])
        out["gpu_idle"] = {"ok": rc == 0 and mib < GPU_IDLE_MIB, "detail": used}
    except (ValueError, IndexError):
        out["gpu_idle"] = {"ok": False, "detail": used or err or "nvidia-smi gave nothing"}
    rc, procs, err = _run(["pgrep", "-af", OTHER_RUNS])
    mine = str(os.getpid())
    others = [p for p in procs.splitlines() if p and p.split()[0] != mine]
    out["no_other_run"] = {"ok": rc in (0, 1) and not others,           # 1: nothing found
                           "detail": ("; ".join(others)[:400] or "none") if rc in (0, 1)
                           else f"pgrep failed: {err or rc}"}
    e1 = json.loads(e1_report.read_text())
    want = (e1["provenance"].get("versions") or {}).get("torch")
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        out["torch"] = {"ok": torch.__version__ == want and cuda,
                        "detail": f"{torch.__version__} cuda {cuda} (E1: {want})"}
    except Exception as exc:                              # noqa: BLE001
        out["torch"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    free = shutil.disk_usage(ROOT / "runs" if (ROOT / "runs").is_dir() else ROOT).free / 2 ** 30
    out["disk"] = {"ok": free >= free_gb, "detail": f"{free:.1f} GB free (need {free_gb:g})"}
    rc, path, err = _run([sys.executable, "-c", "import fejepa; print(fejepa.__file__)"],
                         env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    want = ROOT / "src" / "fejepa" / "__init__.py"
    out["imports"] = {"ok": rc == 0 and Path(path).resolve() == want.resolve(),
                      "detail": f"{path or err} (this checkout: {want})"}
    return out


def git_checks(pre_stamp: bool, root: Path = ROOT) -> dict:
    out = {}
    rc, desc, err = _run(["git", "describe", "--tags", "--match", TAG], cwd=root)
    out["tag"] = {"ok": None if pre_stamp else (rc == 0 and desc == TAG),
                  "detail": desc or err}
    rc, plain, err = _run(["git", "describe", "--always", "--dirty", "--tags"], cwd=root)
    out["report_git"] = {"ok": None if pre_stamp else (rc == 0 and plain == TAG),
                         "detail": plain or err}
    rc, st, err = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root)
    out["clean"] = {"ok": rc == 0 and st == "", "detail": st[:400] or err or "clean"}
    _, head, _ = _run(["git", "rev-parse", "HEAD"], cwd=root)
    _, tree, _ = _run(["git", "rev-parse", "HEAD^{tree}"], cwd=root)
    out["head"] = {"ok": None, "detail": f"HEAD {head} tree {tree}"}
    return out


def config_check(path: Path) -> dict:
    mk = _generator()
    want = mk.render(mk.cm2d_config(mk.load_base()))
    ok = path.is_file() and path.read_text() == want
    return {"ok": ok, "detail": "the generator's" if ok else "differs from the generator's"}


def dry_run_checks(path: Path, pre_stamp: bool) -> dict:
    from fejepa.experiments.runner import run_config

    try:
        s = run_config(str(path), dry_run=True)
    except (Exception, SystemExit) as exc:                # noqa: BLE001
        return {"dry_run": {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}}
    cfg = json.loads(path.read_text())
    need = max(int(b) for b in cfg["experiments"]["e8"]["budgets"])
    w9 = s.get("w9") or {}
    return {"stamp": {"ok": None if pre_stamp else s.get("prereg_status") == "verified",
                      "detail": s.get("prereg_status")},
            "e1_states": {"ok": w9.get("reuse_from") == "verified",
                          "detail": w9.get("reuse_from") or w9.get("error")},
            "plan": {"ok": s.get("label_need_pool_prefix") == need and not s.get("ar_only"),
                     "detail": f"steps {s.get('plan_steps')}, labelled pool prefix "
                               f"{s.get('label_need_pool_prefix')}"}}


def _data_dir(cfg: dict) -> Path:
    d = Path(cfg["data"]["dir"])
    return d if d.is_absolute() else Path.cwd() / d


def _split(cfg: dict):
    from fejepa.experiments.protocol import load_split

    return load_split(_data_dir(cfg), int(cfg["split"]["n_val"]), int(cfg["split"]["seed"]))


def corpus_checks(path: Path, e1_report: Path) -> dict:
    from fejepa.data.archive import load_manifest, manifest_sha256
    from fejepa.experiments.protocol import asis_missing_labels

    cfg = json.loads(path.read_text())
    ddir = _data_dir(cfg)
    e1 = json.loads(e1_report.read_text())
    want = [d["manifest_sha256"] for d in e1["provenance"]["datasets"]]
    try:
        got = manifest_sha256(ddir)
        n = int(load_manifest(ddir).get("n_instances") or 0)
    except (OSError, ValueError) as exc:
        return {"corpus": {"ok": False, "detail": f"{ddir}: {type(exc).__name__}: {exc}"}}
    out = {"corpus": {"ok": [got] == want, "detail": f"{ddir} manifest {got[:12]}..., "
                                                       f"{n} instances (E1: {want[0][:12]}...)"}}
    split = _split(cfg)
    need = max(int(b) for b in cfg["experiments"]["e8"]["budgets"])
    required = list(split.val_files) + list(split.pool_files[:need])
    missing = asis_missing_labels(required, ddir, max_report=len(required))
    out["labels"] = {"ok": not missing,
                     "detail": f"{len(required) - len(missing)} of {len(required)} labelled "
                               f"(validation {len(split.val_files)} + pool prefix {need})"
                               + (f"; first missing {missing[0]}" if missing else "")}
    return out


def reproduction_check(path: Path, e1_report: Path) -> dict:
    """E1's states evaluated as the run's phase A evaluates them (the same unit
    function and payload fields), against E1's per-instance values."""
    from fejepa.experiments.parallel import pretrain_unit
    from fejepa.experiments.w9_eval import verify_reuse

    cfg = json.loads(path.read_text())
    e8 = cfg["experiments"]["e8"]
    e1 = json.loads(e1_report.read_text())
    try:
        states = verify_reuse(cfg, e8["reuse_from"])["states"]
    except (OSError, ValueError, KeyError) as exc:
        return {"reproduction": {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}}
    pool = int(e8["pool_sizes"][0])
    ref = e1["results"]["e8"]["metrics"]["cells"]["ar"]
    ref = ref.get(str(pool), ref.get(pool))
    split = _split(cfg)
    device = _device()
    dev = 0.0
    for k, s in enumerate(sorted(states)):
        spath, sha = states[s][pool]
        out = pretrain_unit({
            "kind": str(cfg["model"].get("kind", "fejepa")), "model": cfg["model"], "seed": s,
            "tf32": bool(cfg.get("tf32", True)), "compile": e8.get("compile", False),
            "precision": e8.get("precision", "fp32"), "files": [], "loss": "ar",
            "pre": {"epochs": int(e8.get("ar_epochs", 100)), "lr": float(e8.get("ar_lr", 1e-3)),
                    "device": device, "desc": f"precheck s{s}"},
            "state_path": spath, "eval_val_files": [str(f) for f in split.val_files],
            "tag": f"precheck s{s}", "eval_only_state": {"path": spath, "sha256": sha}})
        got = out["val"]["per_instance"]
        want = ref["per_seed_eval"][k]["per_instance"]
        for m in REPRO_METRICS:
            a, b = np.asarray(got[m], float), np.asarray(want[m], float)
            if a.shape != b.shape or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
                dev = math.inf
                continue
            dev = max(dev, float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))))
    return {"reproduction": {"ok": dev <= REPRO_MAX,
                             "detail": f"largest relative deviation {dev:.3g} over "
                                       f"{len(states)} states on {device} "
                                       f"(at most {REPRO_MAX:g})"}}


def smoke_check(path: Path) -> dict:
    """One epoch of stiffness-norm training on two pool instances (nothing kept)."""
    from fejepa.data.archive import load_instance
    from fejepa.experiments.parallel import _build_model
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    cfg = json.loads(path.read_text())
    split = _split(cfg)
    try:
        archs = [load_instance(f) for f in split.pool_files[:2]]
        model = _build_model({"kind": "fejepa", "model": cfg["model"], "seed": 0})
        res = train_supervised(model, archs, archs[:1],
                               SupervisedConfig(epochs=1, lr=1e-4, loss="knorm",
                                                device=_device(), log_every=-1))
        g = float(res["val"]["energy_gap_rel"])
        return {"smoke": {"ok": math.isfinite(g),
                          "detail": f"relative energy gap {g:.4g} after one epoch on "
                                    f"{_device()}"}}
    except Exception as exc:                              # noqa: BLE001
        return {"smoke": {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}}


ATTEMPT_TRACES = ("report.json", "e8_states", "run.log", "status.txt")
"""What an earlier attempt leaves under the run's directory."""


def fresh_check(path: Path, restart: bool) -> dict:
    out_dir = Path(json.loads(path.read_text())["out"]).parent
    if restart:
        report = (out_dir / "report.json").exists()
        return {"fresh": {"ok": not report,
                          "detail": f"{out_dir}/report.json exists: the run finished; "
                                    "do not restart" if report else "no report yet"}}
    found = [n for n in ATTEMPT_TRACES if (out_dir / n).exists()]
    return {"fresh": {"ok": not found,
                      "detail": (f"{out_dir} holds an earlier attempt ({', '.join(found)}): "
                                 "a restart is RUNBOOK_CMAME B3") if found
                      else "no earlier attempt"}}


def precheck(config: Path, e1_report: Path, pre_stamp: bool = False, restart: bool = False,
             git: bool = True, stack: bool = True, generator: bool = True) -> dict:
    """{"go", "stop": [failed checks], "checks": {name: {"ok", "detail"}}};
    ok None = reported only. `git`, `stack`, `generator`: False only in tests."""
    checks = {}
    if stack:                                    # first: before this process touches CUDA
        checks.update(stack_checks(e1_report, FREE_GB_RESTART if restart else FREE_GB))
    if git:
        checks.update(git_checks(pre_stamp))
    if generator:
        checks["config"] = config_check(config)
    checks.update(fresh_check(config, restart))
    checks.update(dry_run_checks(config, pre_stamp))
    checks.update(corpus_checks(config, e1_report))
    busy = stack and not (checks["gpu_idle"]["ok"] and checks["no_other_run"]["ok"])
    ready = all(checks.get(k, {}).get("ok") for k in ("e1_states", "corpus", "labels"))
    if busy or not ready:
        why = "the GPU is busy" if busy else "E1's states, corpus or labels"
        checks["reproduction"] = {"ok": False, "detail": f"not run ({why})"}
    else:
        try:
            checks.update(reproduction_check(config, e1_report))
        except Exception as exc:                          # noqa: BLE001
            checks["reproduction"] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    if busy or not checks.get("labels", {}).get("ok"):
        why = "the GPU is busy" if busy else "labels"
        checks["smoke"] = {"ok": False, "detail": f"not run ({why})"}
    else:
        checks.update(smoke_check(config))
    stop = [k for k, v in checks.items() if v["ok"] is False]
    return {"go": not stop, "stop": stop, "pre_stamp": pre_stamp, "restart": restart,
            "checks": checks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--e1-report", default=E1_REPORT)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--pre-stamp", action="store_true",
                   help="readiness before the stamp: the tag and the stamp are reported only")
    g.add_argument("--restart", action="store_true",
                   help="before a restart with --reuse-states: an earlier attempt is expected")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = precheck(Path(a.config), Path(a.e1_report), pre_stamp=a.pre_stamp, restart=a.restart)
    for k, v in res["checks"].items():
        flag = {True: "ok  ", False: "FAIL", None: "info"}[v["ok"]]
        print(f"[precheck] {flag} {k}: {v['detail']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print("GO" if res["go"] else f"STOP: {', '.join(res['stop'])}", flush=True)
    sys.exit(0 if res["go"] else 3)


if __name__ == "__main__":
    main()
