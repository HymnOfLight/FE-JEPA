#!/usr/bin/env python3
"""wp9 Stage 0a: the zero-training readings of session 1 (C0), on the E1 base
states (the N = 1,024 arm of C1). Inference and memory only; no training.

    val       C0.1 ensemble decomposition and C0.3 CG energy-drop estimates
              on E1's 256 validation instances, with the reproduction of
              E1's own per-instance arrays
    trainval  C0.4 energy gap on the 1,024 training instances against the
              256 validation instances (a prediction for C1, recorded before
              C1 runs; it changes no C1 quantity)
    amp2d     C0.7-2D energy-optimal amplitude on the fine-mesh family F5
              and on the remesh set R, with the in-band reference
    memory    C0.6 host and device memory of one training unit's residency
              (instances, packs, energy anchors) as the pool grows, and on a
              GPU the extra device memory of AR steps on the largest held
              instance with activation checkpointing on and off

The states are refused unless their SHA-256 is the one E1's report records,
the corpus unless its manifest is the report's, and the evaluation families
unless every file matches its manifest. Reported only; the session-1
decision rules read these outputs (scripts/w9_session1_decisions.py).

    python scripts/w9_c0.py val --report records/wp8/e1/e1_2d_base/report.json \
        --states-dir runs/e1_2d_base/e8_states --out runs/w9/session1/c0_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

KS = (5, 10, 20, 40)


def _setup(a, need_states=True):
    from fejepa.analysis.common import resolve_device
    from fejepa.analysis.posthoc import run_model, verified_states

    report = json.loads(Path(a.report).read_text())
    dev = resolve_device(a.device)
    models = {}
    if need_states:
        states = verified_states(report, a.states_dir)
        models = {s: run_model(report, p, s, dev) for s, p in states.items()}
    return report, dev, models


def _head(a, report, what):
    from fejepa.analysis.common import sha256_of
    from fejepa.report import _git_describe

    return {"what": what, "git": _git_describe(), "report": a.report,
            "report_sha256": sha256_of(a.report), "states_dir": getattr(a, "states_dir", None),
            "device": None, "seeds": [int(s) for s in report["provenance"]["seeds"]]}


def _ar_cell(report):
    cells = report["results"]["e8"]["metrics"]["cells"]["ar"]
    return cells[max(cells, key=int)]


def _reproduction(report, rows_by_seed, rk):
    """Max relative deviation of the recomputed per-instance values from the
    report's own arrays (E1 is FE-JEPA: 0 is expected)."""
    per = _ar_cell(report)["per_seed_eval"]
    seeds = [int(s) for s in report["provenance"]["seeds"]]
    out = {}
    for s, vals in rows_by_seed.items():
        ref = np.asarray(per[seeds.index(s)]["per_instance"][rk][:len(vals)], float)
        got = np.asarray(vals, float)
        out[f"s{s}"] = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)))
    return out


# ------------------------------------------------------------------ val --
def cmd_val(a):
    from fejepa.analysis.common import write_json
    from fejepa.analysis.posthoc import run_files
    from fejepa.analysis.w9 import ensemble_row, gap_estimate_row, predictions, spearman
    from fejepa.data.archive import LazyArchives
    from fejepa.metrics import displacement_errors

    report, dev, models = _setup(a)
    files = run_files(report, "val", a.n)
    res = _head(a, report, "wp9 C0.1 ensemble decomposition and C0.3 CG energy drops "
                           "(E1 base, validation split); reported only")
    res.update(device=dev, ks=list(a.ks), n_instances=len(files), rows=[])
    disp = {s: [] for s in models}
    gap = {s: [] for s in models}
    t0 = time.time()
    for i, arch in enumerate(LazyArchives(files)):
        P = predictions(models, arch, dev)
        row = {"file": Path(str(arch.path)).name, "ensemble": ensemble_row(arch, P), "seeds": {}}
        for s, U in P.items():
            g = gap_estimate_row(arch, U, a.ks)
            g["disp"] = float(displacement_errors(U, arch).mean())
            row["seeds"][f"s{s}"] = g
            disp[s].append(g["disp"])
            gap[s].append(g["gap_rel"])
        res["rows"].append(row)
        if (i + 1) % 64 == 0:
            print(f"[c0 val] {i + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    rows = res["rows"]
    ens = {k: float(np.mean([r["ensemble"][k] for r in rows]))
           for k in ("gap_seed_mean", "gap_ensemble", "D", "D_labelfree")}
    ens["identity_residual_max"] = float(max(r["ensemble"]["identity_residual"] for r in rows))
    gsm = [r["ensemble"]["gap_seed_mean"] for r in rows]
    ens["spearman_D_vs_gap"] = spearman([r["ensemble"]["D"] for r in rows], gsm)
    ens["spearman_D_labelfree_vs_gap"] = spearman([r["ensemble"]["D_labelfree"] for r in rows], gsm)
    est = {}
    for k in a.ks:
        per_seed = [spearman([r["seeds"][f"s{s}"]["est_rel"][str(k)] for r in rows],
                             [r["seeds"][f"s{s}"]["gap_rel"] for r in rows]) for s in models]
        cap = [r["seeds"][f"s{s}"]["captured"][str(k)] for r in rows for s in models]
        est[str(k)] = {"spearman_per_seed": per_seed, "spearman_seed_mean": float(np.mean(per_seed)),
                       "captured_median": float(np.median(cap))}
    res["summary"] = {"ensemble": ens, "cg_estimates": est,
                      "reproduction": {"disp": _reproduction(report, disp, "disp_rel_l2"),
                                       "egap": _reproduction(report, gap, "energy_gap_rel")},
                      "seconds": round(time.time() - t0, 1)}
    write_json(a.out, res)
    print(json.dumps(res["summary"]), flush=True)


# ------------------------------------------------------------- trainval --
def cmd_trainval(a):
    from fejepa.analysis.common import write_json
    from fejepa.analysis.posthoc import run_files
    from fejepa.analysis.w9 import predictions
    from fejepa.data.archive import LazyArchives
    from fejepa.metrics import displacement_errors, energy_gap_rel

    report, dev, models = _setup(a)
    res = _head(a, report, "wp9 C0.4 training-set against validation-set error of the "
                           "E1 base states; a prediction for C1, reported only")
    res["device"] = dev
    t0 = time.time()
    for name in ("train", "val"):
        files = run_files(report, name)
        per = {f"s{s}": {"disp": [], "egap": []} for s in models}
        for arch in LazyArchives(files):
            if arch.U_star is None:
                raise SystemExit(f"{arch.path}: no labels (the E1 corpus labels the "
                                 "validation split and the 1,024-instance prefix)")
            for s, U in predictions(models, arch, dev).items():
                per[f"s{s}"]["disp"].append(float(displacement_errors(U, arch).mean()))
                per[f"s{s}"]["egap"].append(float(energy_gap_rel(U, arch).mean()))
        res[name] = {"n_instances": len(files), "per_seed": per,
                     "seed_means": {k: float(np.mean([np.mean(v[k]) for v in per.values()]))
                                    for k in ("disp", "egap")}}
        print(f"[c0 trainval] {name} done {time.time() - t0:.0f}s", flush=True)
    tr, va = res["train"]["seed_means"], res["val"]["seed_means"]
    res["summary"] = {"egap_train_over_val": tr["egap"] / va["egap"],
                      "disp_train_over_val": tr["disp"] / va["disp"],
                      "seconds": round(time.time() - t0, 1)}
    write_json(a.out, res)
    print(json.dumps(res["summary"]), flush=True)


# ---------------------------------------------------------------- amp2d --
def _family(path, expect: str):
    from fejepa.data.archive import instance_files, load_manifest, manifest_sha256
    from fejepa.fe.ood2d import verify_manifest_files

    d = Path(path)
    m = load_manifest(d)
    if m.get("family") != expect:
        raise SystemExit(f"{d}: holds family {m.get('family')!r}, not {expect!r}")
    bad = verify_manifest_files(d)
    if bad:
        raise SystemExit(f"{d}: {len(bad)} files differ from their manifest (first {bad[0]})")
    return instance_files(d), {"dir": str(d), "manifest_sha256": manifest_sha256(d),
                               "family": m.get("family"), "seed": m.get("seed"),
                               "n_instances": m["n_instances"]}


def cmd_amp2d(a):
    from fejepa.analysis.common import write_json
    from fejepa.analysis.posthoc import run_files
    from fejepa.analysis.w9 import amplitude_row, amplitude_summary, predictions
    from fejepa.data.archive import LazyArchives, load_manifest

    report, dev, models = _setup(a)
    res = _head(a, report, "wp9 C0.7-2D energy-optimal amplitude of the E1 base states on "
                           "the in-band validation split, the fine-mesh family F5 and the "
                           "remesh set R; reported only")
    res["device"] = dev
    f5_files, res["F5"] = _family(a.family_dir, "F5")
    r_files, res["R"] = _family(a.remesh_dir, "R")
    t0 = time.time()
    sets = {"inband": run_files(report, "val", a.n), "F5": f5_files}
    res["sets"] = {}
    for name, files in sets.items():
        rows = {s: [] for s in models}
        for arch in LazyArchives(files):
            for s, U in predictions(models, arch, dev).items():
                rows[s].append(dict(amplitude_row(arch, U), file=Path(str(arch.path)).name))
        res["sets"][name] = {f"s{s}": {"summary": amplitude_summary(v), "rows": v}
                             for s, v in rows.items()}
        print(f"[c0 amp2d] {name} done {time.time() - t0:.0f}s", flush=True)
    # remesh: rows grouped by geometry, ordered by h as the manifest lists them
    man = {r["file"]: r for r in load_manifest(Path(a.remesh_dir))["instances"]}
    rem = {s: [] for s in models}
    for arch in LazyArchives(r_files):
        rec = man[Path(str(arch.path)).name]
        for s, U in predictions(models, arch, dev).items():
            rem[s].append(dict(amplitude_row(arch, U), geometry=int(rec["geometry"]),
                               file=rec["file"]))
    res["remesh"] = {f"s{s}": {"rows": v, "by_h": _by_h(v)} for s, v in rem.items()}
    seeds = [f"s{s}" for s in models]

    def sm(st, key):
        return float(np.mean([res["sets"][st][s]["summary"][key] for s in seeds]))

    res["summary"] = {
        "disp_inband": sm("inband", "disp"), "disp_F5": sm("F5", "disp"),
        "disp_c_inband": sm("inband", "disp_c"), "disp_c_F5": sm("F5", "disp_c"),
        "egap_inband": sm("inband", "egap"), "egap_F5": sm("F5", "egap"),
        "c_star_median_inband": sm("inband", "c_star_median"),
        "c_star_median_F5": sm("F5", "c_star_median"),
        "ratio_disp_F5_over_inband": sm("F5", "disp") / sm("inband", "disp"),
        "seconds": round(time.time() - t0, 1)}
    write_json(a.out, res)
    print(json.dumps(res["summary"]), flush=True)


def _by_h(rows):
    """Per mesh size: median c_b, the median predicted energy norm relative to
    the same geometry's coarsest mesh (label-free) and the same ratio of the
    exact solutions' energy norms (the discretisation's own change), the
    median fscale ratio, and the median displacement error."""
    hs = sorted({r["target_h"] for r in rows}, reverse=True)
    base = {r["geometry"]: r for r in rows if r["target_h"] == hs[0]}

    def ratio(sel, key):
        v = [u / b for r in sel for u, b in zip(r[key], base[r["geometry"]][key], strict=True)
             if b > 0]
        return float(np.median(v))

    out = {}
    for h in hs:
        sel = [r for r in rows if r["target_h"] == h]
        out[repr(h)] = {"n": len(sel),
                        "c_battery_median": float(np.median([r["c_battery"] for r in sel])),
                        "u_norm_K_ratio_median": ratio(sel, "u_norm_K"),
                        "ustar_norm_K_ratio_median": (ratio(sel, "ustar_norm_K")
                                                      if "ustar_norm_K" in sel[0] else None),
                        "fscale_ratio_median": float(np.median(
                            [r["fscale"] / base[r["geometry"]]["fscale"] for r in sel])),
                        "disp_median": float(np.median([r["disp"] for r in sel])),
                        "n_nodes_median": int(np.median([r["n_nodes"] for r in sel]))}
    return out


# --------------------------------------------------------------- memory --
def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return -1


def _read_int(path):
    try:
        v = Path(path).read_text().strip()
        return None if v == "max" or int(v) >= 1 << 60 else int(v)
    except (OSError, ValueError):
        return None


def _memory_limit() -> dict:
    """The host memory the session's processes may hold: the cgroup limit
    (bounded by MemTotal); without a cgroup limit -- a container that sees the
    host's memory -- the MemAvailable read at the audit's start. The source is
    recorded."""
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            for key, name in (("MemTotal:", "meminfo_total"),
                              ("MemAvailable:", "meminfo_available")):
                if line.startswith(key):
                    out[name] = int(line.split()[1]) * 1024
    except OSError:
        pass
    for lim, cur in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                     ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                      "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        limit = _read_int(lim)
        if limit is not None:
            out["cgroup_limit"], out["cgroup_current"] = limit, _read_int(cur)
            break
    if out.get("cgroup_limit"):
        cands = {k: out[k] for k in ("cgroup_limit", "meminfo_total") if out.get(k)}
    else:
        cands = {k: out[k] for k in ("meminfo_available",) if out.get(k)}
    if cands:
        src = min(cands, key=cands.get)
        out["usable"], out["usable_source"] = cands[src], src
    else:
        out["usable"], out["usable_source"] = -1, None
    return out


def _step_overhead(model, held, anchors) -> dict:
    """Extra device memory of two AR steps (loss, backward, AdamW step) on the
    largest held instance, with activation checkpointing on and off; each
    mode from an emptied allocator cache and a fresh optimiser."""
    import gc

    import torch

    from fejepa.models.regularizers import PooledBuffer
    from fejepa.train.losses import AR_CONFIG, compute_loss

    arch, pack = max(held, key=lambda t: t[0].n_nodes)
    out = {"n_nodes": int(arch.n_nodes)}
    model.train()
    for mode, flag in (("ckpt_on", True), ("ckpt_off", False)):
        model.encoder.use_checkpoint = flag
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        r0, a0 = torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(0)
        for _ in range(2):
            loss, _parts = compute_loss(model, pack, anchors.get(arch), None, PooledBuffer(),
                                        rng, AR_CONFIG)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        torch.cuda.synchronize()
        out[mode] = {"reserved_overhead": int(torch.cuda.max_memory_reserved() - r0),
                     "allocated_overhead": int(torch.cuda.max_memory_allocated() - a0)}
        del opt, loss, _parts
    model.zero_grad(set_to_none=True)
    model.encoder.use_checkpoint = True
    return out


def cmd_memory(a):
    import torch

    from fejepa.analysis.common import build_model_from_config, instance_files, write_json
    from fejepa.analysis.posthoc import verified_dir
    from fejepa.anchor.energy import AnchorCache
    from fejepa.data.archive import load_instance
    from fejepa.runtime import setup_torch

    report, dev, _ = _setup(a, need_states=False)
    cfg = report["config"]
    setup_torch(dev, tf32=bool(cfg.get("tf32", True)))         # the run's numeric policy
    d = verified_dir(report, cfg["data"]["dir"])
    pool = instance_files(d, split=cfg["split"], subset="pool")
    marks = sorted(int(n) for n in a.checkpoints)
    if marks[-1] > len(pool):
        raise SystemExit(f"the pool has {len(pool)} instances, fewer than {marks[-1]}")
    lim = _memory_limit()
    res = _head(a, report, "wp9 C0.6 memory of one AR training unit's residency (instances, "
                           "packs, CPU-resident energy anchors) as the pool grows, and the "
                           "device memory of AR steps; reported only")
    res.update(device=dev, limits=lim, workers=a.workers, stop_fraction=a.stop_fraction,
               marks=[], stopped_at=None)
    model = build_model_from_config(cfg["model"], seed=0, mode="train", device=dev)
    gpu = dev.startswith("cuda")
    if gpu:
        # this process's CUDA context, estimated as what the device has in use
        # beyond this process's allocator (anything else on the GPU counts too:
        # the audit runs alone)
        free0, total0 = torch.cuda.mem_get_info()
        res["gpu"] = {"name": torch.cuda.get_device_name(0), "total": int(total0),
                      "device_used_start": int(total0 - free0),
                      "reserved_start": int(torch.cuda.memory_reserved()),
                      "context_estimate": int(total0 - free0 - torch.cuda.memory_reserved())}
    anchors = AnchorCache(device=dev)
    held = []
    rss0, t0 = _rss_bytes(), time.time()
    res["rss_before"] = rss0

    def mark(i):
        res["marks"].append({"n": i, "rss": _rss_bytes(),
                             "gpu_allocated": int(torch.cuda.memory_allocated()) if gpu else 0,
                             "gpu_reserved": int(torch.cuda.memory_reserved()) if gpu else 0,
                             "seconds": round(time.time() - t0, 1)})
        print(json.dumps(res["marks"][-1]), flush=True)
        write_json(a.out, res)

    for i, f in enumerate(pool[:marks[-1]], 1):
        arch = load_instance(f)
        pack = model.prepare_instance(arch, dev)
        anchors.get(arch)
        held.append((arch, pack))
        if i in marks:
            mark(i)
        over_host = lim["usable"] > 0 and _rss_bytes() > a.stop_fraction * lim["usable"]
        over_gpu = gpu and torch.cuda.memory_reserved() > a.stop_fraction * res["gpu"]["total"]
        if over_host or over_gpu:
            res["stopped_at"] = i
            if i not in marks:
                mark(i)
            print(f"[c0 memory] stopped at {i}: {'RSS' if over_host else 'GPU reserve'} above "
                  f"{a.stop_fraction} of the limit", flush=True)
            break
    if gpu and held:
        # the steps also load CUDA modules and handles, and may grow the host side:
        # both are read again afterwards (the decisions take the larger values)
        try:
            res["gpu"]["step"] = _step_overhead(model, held, anchors)
        except Exception as exc:                          # noqa: BLE001 -- record it
            res["gpu"]["step_error"] = f"{type(exc).__name__}: {exc}"
        free1, total1 = torch.cuda.mem_get_info()
        res["gpu"]["context_after_step"] = int(total1 - free1 - torch.cuda.memory_reserved())
        res["rss_after_step"] = _rss_bytes()
        print(json.dumps({k: res["gpu"].get(k) for k in ("step", "step_error",
                                                         "context_after_step")}), flush=True)
    write_json(a.out, res)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("val", "trainval", "amp2d", "memory"):
        p = sub.add_parser(name)
        p.add_argument("--report", required=True, help="E1 base report (the records copy)")
        p.add_argument("--device", default="auto")
        p.add_argument("--out", required=True)
        if name != "memory":
            p.add_argument("--states-dir", required=True)
        if name in ("val", "amp2d"):
            p.add_argument("--n", type=int, default=None, help="validation instances (default all)")
        if name == "val":
            p.add_argument("--ks", type=int, nargs="+", default=list(KS))
        if name == "amp2d":
            p.add_argument("--family-dir", required=True)
            p.add_argument("--remesh-dir", required=True)
        if name == "memory":
            p.add_argument("--checkpoints", type=int, nargs="+",
                           default=[1024, 4096, 12800, 25600])
            p.add_argument("--workers", type=int, default=3)
            p.add_argument("--stop-fraction", type=float, default=0.6)
    a = ap.parse_args()
    {"val": cmd_val, "trainval": cmd_trainval, "amp2d": cmd_amp2d, "memory": cmd_memory}[a.cmd](a)


if __name__ == "__main__":
    main()
