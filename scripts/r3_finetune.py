#!/usr/bin/env python3
"""fine3d (PREREG_FINE3D Sec. 3, R3): fine-tuning at the fine mesh, one arm
per configuration -- the energy arm (configs/r3_eft_b16.json: the AR objective,
label-free) or the labelled arm (configs/r3_lft_b16.json: the displacement
labels, P3's labelled few-shot fine-tuning run again in the same session).

For each seed of the configuration, in order:
  1. the Phase-2b AR state of that seed, checked by SHA-256 against the
     Phase-2b report's record (the report itself checked by SHA-256), loaded
     strictly into the report's own model (its embedded configuration, which
     must hash to the report's recorded config SHA-256), under the report's
     TF32 policy;
  2. a content check: the state's predictions on the first `n_fine` fine
     evaluation instances and the first `n_inband` in-band validation
     instances reproduce the report's per-instance displacement errors and
     energy gaps (P3's zero-shot AR cell and E8's AR cell) to `max_rel_dev`;
     otherwise it stops before training;
  3. training on the first `n` instances of the fine few-shot prefix -- the
     instances on which P3's labelled few-shot fine-tuning trained at that
     budget -- with that fine-tuning's schedule (its epochs and learning rate;
     AdamW, weight decay 1e-4, gradient clip 1.0, 5% linear warm-up then
     cosine: SupervisedConfig's defaults, checked) and its seed, hence the same
     instance order in every epoch. The energy arm: `pretrain()` with
     AR_CONFIG (energy only); the training instances' labels are dropped when
     they are loaded (U_star = None), so the objective cannot read them. The
     labelled arm: `train_supervised()` with anchor_mode "none" (the
     displacement loss), as P3's unit runs it; every training instance must
     carry its labels;
  4. the fine-tuned state is saved (its SHA-256 recorded) and evaluated with
     the frozen metric suite and the amplitude readings on the fine
     evaluation set and the in-band validation split (a progress line every
     32 instances).
Each seed's result records the arm (its loss and trainer), the seed it
trained with, the epochs it ran and the steps (counted from the trainer's
history), the SHA-256 of the instance
orders that seed determines (`epoch_orders`; tests/test_fine3d_r3.py checks
that the trainer visits them), the code (`git describe`), the
configuration's SHA-256, the TF32 setting the training ran with, and the
machines (host and GPU driver) the seed ran on: where it started (kept in its
checkpoint's sidecar across a resumption), any it resumed on, and where it
ended. Both arms train in fp32 without torch.compile, as P3's labelled units
did: the source report's `runtime` block must say so (P3 read both from it).
The code's `git describe` is read before the first seed, and the report and
every seed record that reading; a run that cannot read it stops. A content
check fails on any non-finite prediction or deviation. A seed's result file
is written before its checkpoint is removed. A seed whose
result file is readable, of this seed,
this code and this configuration, with a passed content check and its state
on disk at the recorded SHA-256, is not run again (marked `from_disk` in the
report); any other result file is moved aside and the seed runs again. An
interrupted training resumes from its epoch checkpoint (written at most every
5 minutes) if this code and configuration wrote it; any other checkpoint is
moved aside and the seed trains from the start. The report is written when
every seed has a result.

    python scripts/r3_finetune.py configs/r3_eft_b16.json
    python scripts/r3_finetune.py configs/r3_lft_b16.json
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHECK_KEYS = ("disp_rel_l2", "energy_gap_rel")
PROGRESS_EVERY = 32
ARMS = {"ar": {"name": "energy", "prefix": "eft", "trainer": "pretrain"},
        "labels": {"name": "labelled", "prefix": "lft", "trainer": "train_supervised"}}
"""The two arms by `train.loss`: their name, file prefix and trainer."""


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def verify(cfg: dict, label: str) -> dict:
    """The guard, the source report and the trainer's schedule; returns what
    the report records about them."""
    from fejepa.report import config_sha256, verify_prereg
    from fejepa.train.schedule import make_scheduler
    from fejepa.train.supervised import SupervisedConfig

    out = {"config_sha256": config_sha256(cfg), "prereg": None}
    if cfg.get("prereg_guard"):
        pf = _rel(cfg["prereg_file"])
        out["prereg"] = {"file": cfg["prereg_file"],
                         "config_sha256": verify_prereg(cfg, pf, label=label)}
    src = cfg["source"]
    got = _sha256(_rel(src["report"]))
    if got != src["report_sha256"]:
        raise SystemExit(f"{src['report']}: SHA-256 {got[:12]}... is not the source report "
                         f"({src['report_sha256'][:12]}...)")
    tr = cfg["train"]
    if tr.get("loss") not in ARMS or tr.get("set") != "fine_fewshot_prefix":
        raise SystemExit(f"train: loss {tr.get('loss')!r} / set {tr.get('set')!r} -- this "
                         "script trains the AR objective ('ar') or the displacement labels "
                         "('labels') on the fine few-shot prefix only")
    wf = inspect.signature(make_scheduler).parameters["warmup_frac"].default
    if float(tr["warmup_frac"]) != float(wf):
        raise SystemExit(f"train.warmup_frac {tr['warmup_frac']} is not the trainer's ({wf})")
    sd = {f.name: f.default for f in dataclasses.fields(SupervisedConfig)}
    for k in ("weight_decay", "clip"):
        if float(tr[k]) != float(sd[k]):
            raise SystemExit(f"train.{k} {tr[k]} is not the labelled fine-tuning's "
                             f"(SupervisedConfig's {sd[k]})")
    return out


def check_runtime(report: dict) -> None:
    """The source report's runtime must be fp32 without compile: P3's labelled
    units took both from it (runner: `runtime.precision`, `runtime.compile`,
    defaults fp32 and False), and R3 trains with SupervisedConfig's and
    PretrainConfig's defaults (fp32) and never compiles."""
    rt = report["config"].get("runtime") or {}
    if rt.get("precision", "fp32") != "fp32" or bool(rt.get("compile", False)):
        raise SystemExit(f"the source report's runtime is {rt}: R3 trains in fp32 without "
                         "compile, so its labelled arm would not repeat P3's labelled units")


def machine(tries: int = 2) -> dict:
    """The host and the GPU driver (the first GPU's; None where there is none,
    after `tries` attempts)."""
    import platform
    import subprocess

    drv = None
    for i in range(tries):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                "--format=csv,noheader"], capture_output=True, text=True,
                               timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                drv = r.stdout.strip().splitlines()[0].strip()
                break
        except FileNotFoundError:                         # no tool: no GPU driver to read
            break
        except Exception:                                 # noqa: BLE001 (a failed query)
            pass
        if i + 1 < tries:
            time.sleep(5)
    return {"host": platform.node(), "gpu_driver": drv}


def instance_sets(report: dict, cfg: dict, fine_dir=None, inband_dir=None) -> dict:
    """The fine evaluation set, the training instances and the in-band
    validation split, each from a corpus whose manifest is the report's."""
    from fejepa.analysis.posthoc import run_files, verified_dir
    from fejepa.data.archive import instance_files

    rc = report["config"]
    dt = rc["data_transfer"]
    n_eval = int((dt.get("split") or {}).get("n_eval", 256))
    n_pref = int((dt.get("split") or {}).get("n_fewshot_prefix", 64))
    fdir = verified_dir(report, fine_dir or dt["dir"])
    ffiles = instance_files(Path(fdir))
    n_train = int(cfg["train"]["n"])
    if n_train > n_pref or len(ffiles) < n_eval + n_pref:
        raise SystemExit(f"the fine corpus has {len(ffiles)} instances; R3 needs the "
                         f"evaluation set ({n_eval}) and {n_train} of the prefix ({n_pref})")
    fine_eval = ffiles[:n_eval]                        # P3's pinned partition (manifest order)
    train = ffiles[n_eval:n_eval + n_pref][:n_train]   # P3's fine_prefix_files[:b]
    val = run_files(report, "val", None, inband_dir)
    want = cfg["evaluation"]
    if len(fine_eval) < int(want["fine"]) or len(val) < int(want["inband"]):
        raise SystemExit("the evaluation sets are smaller than the configuration's")
    return {"fine": fine_eval[:int(want["fine"])], "train": train,
            "inband": val[:int(want["inband"])], "fine_dir": fdir,
            "inband_dir": str(Path(val[0]).parent)}


def _ref_arrays(report: dict, which: str, seed_index: int) -> dict:
    m = report["results"]
    if which == "fine":
        cell = m["p3_transfer"]["metrics"]["ar"]["fine"]
    else:
        cells = m["e8"]["metrics"]["cells"]["ar"]
        cell = cells[max(cells, key=int)]
    return cell["per_seed_eval"][seed_index]["per_instance"]


def content_check(model, report: dict, seed_index: int, sets: dict, cfg: dict,
                  device: str) -> dict:
    """The loaded state's per-instance values against the report's own."""
    from fejepa.data.archive import LazyArchives
    from fejepa.metrics import evaluate_fields, torch_predictor

    cc = cfg["content_check"]
    pred = torch_predictor(model, device)
    dev = {}
    for which, n in (("fine", int(cc["n_fine"])), ("inband", int(cc["n_inband"]))):
        ref = _ref_arrays(report, which, seed_index)
        worst = 0.0
        for i, a in enumerate(LazyArchives(sets[which][:n])):
            vals = evaluate_fields(pred(a), a)
            for k in CHECK_KEYS:
                r = float(ref[k][i])
                d = abs(float(vals[k]) - r) / max(abs(r), 1e-30)
                worst = max(worst, d if math.isfinite(d) else math.inf)   # NaN fails
        dev[which] = worst
    ok = all(v <= float(cc["max_rel_dev"]) for v in dev.values())
    return {"max_rel_dev": dev, "tolerance": float(cc["max_rel_dev"]), "passed": bool(ok),
            "n_fine": int(cc["n_fine"]), "n_inband": int(cc["n_inband"])}


def epoch_orders(seed: int, n: int, epochs: int) -> list:
    """The instance order of every epoch as both trainers draw it -- one
    `np.random.default_rng(seed)` and one permutation per epoch, nothing else
    drawn from it on the AR path (train_supervised and pretrain alike;
    tests/test_fine3d_r3.py checks both trainers against this)."""
    import numpy as np

    rng = np.random.default_rng(int(seed))
    return [rng.permutation(int(n)).tolist() for _ in range(int(epochs))]


def _progress(archs, what: str, every: int = PROGRESS_EVERY):
    """Yield the instances, printing a progress line every `every`."""
    n = len(archs)
    for i, a in enumerate(archs):
        if i % every == 0:
            print(f"[r3] evaluating {what}: {i}/{n}", flush=True)
        yield a


def load_training(files: list, labels: bool) -> list:
    """The training instances: for the energy arm with their labels dropped
    (the objective is label-free), for the labelled arm with their labels,
    which every one must carry."""
    from fejepa.data.archive import load_instance

    archs = []
    for f in files:
        a = load_instance(Path(f))
        if not labels:
            a.U_star = None
        archs.append(a)
    if not labels and any(a.labelled for a in archs):
        raise SystemExit("a training instance still carries labels")
    if labels and not all(a.labelled for a in archs):
        raise SystemExit("a training instance of the labelled arm carries no labels")
    return archs


def _sidecar(ckpt: Path) -> Path:
    return ckpt.with_name(ckpt.name + ".json")


def own_checkpoint(ckpt: Path, mine: dict, here: dict) -> list:
    """An epoch checkpoint is resumed only by the code and configuration that
    wrote it (its sidecar file records both, and the machines the seed has run
    on); any other is moved aside and the seed trains from the start. Then the
    sidecar names this run. Returns the machines of this seed's training: the
    first is where it started, the last (`here`) where it runs now."""
    machines = []
    if ckpt.exists():
        try:
            side = json.loads(_sidecar(ckpt).read_text())
            ok = {k: side.get(k) for k in mine} == mine
        except (OSError, ValueError, AttributeError):
            side, ok = {}, False
        if ok:
            machines = [m for m in side.get("machines") or () if isinstance(m, dict)]
        else:
            aside = ckpt.with_name(f"{ckpt.name}.unusable.{time.strftime('%Y%m%d-%H%M%S')}")
            ckpt.rename(aside)
            print(f"[r3] {ckpt.name} was written by another code or configuration (moved to "
                  f"{aside.name}); training from the start", flush=True)
    machines.append(here)
    _sidecar(ckpt).write_text(json.dumps(dict(mine, machines=machines)))
    return machines


def ckpt_path(out_dir: Path, cfg: dict, seed: int) -> Path:
    """A seed's epoch checkpoint (its sidecar: the same name + .json)."""
    tr = cfg["train"]
    return out_dir / f"{ARMS[tr['loss']]['prefix']}_b{tr['n']}_s{seed}.ckpt"


def train_runtime() -> dict:
    """The TF32 setting the training runs with (set by run_model from the
    source report's policy)."""
    import torch

    return {"tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def run_seed(seed: int, seed_index: int, report: dict, cfg: dict, sets: dict, out_dir: Path,
             device: str, state_path: Path, git: str) -> dict:
    """One seed: load, check, train, save, evaluate. The caller writes the
    result file and then removes the seed's checkpoint."""
    from fejepa.analysis.posthoc import run_model
    from fejepa.data.archive import LazyArchives
    from fejepa.experiments.w9_eval import evaluate_model_w9
    from fejepa.metrics import torch_predictor
    from fejepa.report import config_sha256
    from fejepa.train.checkpoint import atomic_torch_save
    from fejepa.train.losses import AR_CONFIG
    from fejepa.train.pretrain import PretrainConfig, pretrain
    from fejepa.train.supervised import SupervisedConfig, train_supervised

    t0 = time.time()
    here = machine()
    model = run_model(report, state_path, seed, device)       # verified config, strict load
    check = content_check(model, report, seed_index, sets, cfg, device)
    print(json.dumps({"seed": seed, "content_check": check}), flush=True)
    if not check["passed"]:
        raise SystemExit(f"seed {seed}: the state's predictions do not reproduce the report's "
                         f"(max relative deviation {check['max_rel_dev']}); stopped before "
                         "training")
    t1 = time.time()
    tr = cfg["train"]
    arm = ARMS[tr["loss"]]
    archs = load_training(sets["train"], labels=tr["loss"] == "labels")
    ckpt = ckpt_path(out_dir, cfg, seed)
    machines = own_checkpoint(ckpt, {"git": git, "config_sha256": config_sha256(cfg)}, here)
    runtime = train_runtime()
    common = dict(epochs=int(tr["epochs"]), lr=float(tr["lr"]),
                  weight_decay=float(tr["weight_decay"]), clip=float(tr["clip"]),
                  seed=int(seed), device=device, desc=f"R3 {arm['name']} FT b{tr['n']} s{seed}",
                  ckpt_path=str(ckpt), resume=True)
    if tr["loss"] == "ar":
        hist = pretrain(model, archs, PretrainConfig(loss=AR_CONFIG, **common))
    else:                         # P3's labelled unit: the displacement loss, no anchor
        out = train_supervised(model, archs, archs[:1],
                               SupervisedConfig(anchor_mode="none", **common))
        hist = dict(out["history"], resumed_from_epoch=out.get("resumed_from_epoch"))
    sp = out_dir / f"{arm['prefix']}_b{tr['n']}_s{seed}.pt"
    atomic_torch_save(model.state_dict(), sp)
    n_train = len(archs)
    del archs
    t2 = time.time()
    pred = torch_predictor(model, device)
    amp = bool(cfg["evaluation"].get("amplitude"))
    losses = list(hist.get("loss") or [])
    orders = epoch_orders(int(seed), n_train, int(tr["epochs"]))
    res = {"seed": seed, "git": git, "config_sha256": config_sha256(cfg),
           "machine": {"start": machines[0], "resumed_on": machines[1:], "end": machine()},
           "source_state": {"path": str(state_path), "sha256": _sha256(state_path)},
           "content_check": check,
           "train": {"loss": tr["loss"], "trainer": arm["trainer"],
                     "files": [Path(f).name for f in sets["train"]], "seed": int(seed),
                     "epochs": int(tr["epochs"]), "epochs_run": len(losses),
                     "steps": len(losses) * n_train,
                     "order_sha256": hashlib.sha256(json.dumps(orders).encode()).hexdigest(),
                     "first_epoch_order": orders[0] if orders else [],
                     "loss_by_epoch": losses, "runtime": runtime,
                     "resumed_from_epoch": hist.get("resumed_from_epoch")},
           "state": {"path": str(sp), "sha256": _sha256(sp)},
           "fine": evaluate_model_w9(pred, _progress(LazyArchives(sets["fine"]), "fine"), amp),
           "inband": evaluate_model_w9(pred, _progress(LazyArchives(sets["inband"]), "in band"),
                                       amp)}
    res["seconds"] = {"load_and_check": round(t1 - t0, 1), "train": round(t2 - t1, 1),
                      "evaluate": round(time.time() - t2, 1)}
    return res


def finished(rf: Path, seed: int, git: str, config_sha: str) -> dict | None:
    """A seed's result file, if it is usable: readable, of this seed, of this
    code (`git describe`) and configuration, its content check passed, its
    state on disk at the recorded SHA-256."""
    try:
        prev = json.loads(rf.read_text())
        sp = Path(prev["state"]["path"])
        ok = (int(prev.get("seed", -1)) == int(seed) and prev.get("git") == git
              and prev.get("config_sha256") == config_sha
              and prev["content_check"]["passed"] is True
              and sp.is_file() and _sha256(sp) == prev["state"]["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return prev if ok else None


def run(config_path, device_override: str | None = None, fine_dir=None,
        inband_dir=None, dry_run: bool = False) -> dict:
    from fejepa.analysis.common import resolve_device, write_json
    from fejepa.analysis.posthoc import verified_states
    from fejepa.report import _git_describe, provenance

    cp = Path(config_path)
    cfg = json.loads(cp.read_text())
    ver = verify(cfg, cp.stem)
    report = json.loads(_rel(cfg["source"]["report"]).read_text())
    check_runtime(report)
    device = resolve_device(device_override or cfg.get("device", "auto"))
    seeds = [int(s) for s in cfg["seeds"]]
    rep_seeds = [int(s) for s in report["provenance"]["seeds"]]
    if any(s not in rep_seeds for s in seeds):
        raise SystemExit(f"seeds {seeds}: the source report trained {rep_seeds}")
    states = verified_states(report, _rel(cfg["source"]["states_dir"]), seeds)
    sets = instance_sets(report, cfg, fine_dir, inband_dir)
    out = _rel(cfg["out"])
    out_dir = out.parent
    plan = {"r3": str(cp), "arm": ARMS[cfg["train"]["loss"]]["name"], "device": device,
            "seeds": seeds,
            "train": [Path(f).name for f in sets["train"]],
            "n_fine": len(sets["fine"]), "n_inband": len(sets["inband"]),
            "states_verified": len(states), "prereg": ver["prereg"]}
    print(json.dumps(plan), flush=True)
    if dry_run:
        return {"dry_run": True, **plan}
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed, git = {}, _git_describe()
    if git == "unavailable":
        raise SystemExit("git describe could not be read: the code this run uses cannot be "
                         "recorded; stopped before training")
    for s in seeds:
        rf = out_dir / f"seed_{s}.json"
        if rf.is_file():
            prev = finished(rf, s, git, ver["config_sha256"])
            if prev is not None:
                print(f"[r3] seed {s}: result on disk ({rf.name}); not run again", flush=True)
                per_seed[f"s{s}"] = dict(prev, from_disk=True)
                continue
            aside = rf.with_name(f"{rf.name}.unusable.{time.strftime('%Y%m%d-%H%M%S')}")
            rf.rename(aside)
            print(f"[r3] seed {s}: {rf.name} unusable (moved to {aside.name}); run again",
                  flush=True)
        res = run_seed(s, rep_seeds.index(s), report, cfg, sets, out_dir, device, states[s], git)
        write_json(rf, res)                       # the result first, then the checkpoint goes
        ck = ckpt_path(out_dir, cfg, s)
        ck.unlink(missing_ok=True)
        _sidecar(ck).unlink(missing_ok=True)
        per_seed[f"s{s}"] = res
        f, b = res["fine"], res["inband"]
        print(json.dumps({"seed": s, "fine": {k: f[k] for k in ("disp_rel_l2", "energy_gap_rel",
                                                                  "vm_rel_l2")},
                          "inband": {k: b[k] for k in ("disp_rel_l2", "energy_gap_rel",
                                                       "vm_rel_l2")},
                          "seconds": res["seconds"]}), flush=True)
    from fejepa.runtime import setup_torch

    policy = setup_torch(device, tf32=bool(report["config"].get("tf32", True)))
    rep = {"what": f"fine3d R3: fine-tuning at the fine mesh, the {ARMS[cfg['train']['loss']]['name']} "
                   "arm (PREREG_FINE3D)",
           "config": cfg, "prereg": ver["prereg"],
           "provenance": dict(provenance(cfg, [sets["fine_dir"], sets["inband_dir"]], seeds),
                              git=git,            # the reading before the first seed
                              source_report=cfg["source"]["report"],
                              source_report_sha256=_sha256(_rel(cfg["source"]["report"])),
                              source_config_sha256=report["provenance"]["config_sha256"]),
           "runtime_policy": policy, "device": device, "per_seed": per_seed}
    write_json(out, rep)
    print(f"[r3] report -> {out}", flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fine-dir", default=None, help="override the report's transfer dir")
    ap.add_argument("--inband-dir", default=None, help="override the report's data dir")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the guard, the report, the states and the corpora; train nothing")
    a = ap.parse_args()
    run(a.config, a.device, a.fine_dir, a.inband_dir, dry_run=a.dry_run)


if __name__ == "__main__":
    main()
