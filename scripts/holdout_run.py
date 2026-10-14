#!/usr/bin/env python3
"""cmame-paper Stage 11 (PREREG_HOLDOUT Sec. 3-4): label a holdout set and
evaluate the stored states on it. Box only; no training.

Stages (`--stage`):

* `label`: the set's never-used instances (`fejepa.analysis.holdout`), each
  solved exactly once into `<out>/<set>/labels/`, with
  `<out>/<set>/manifest.json`. Resumable: a labels file solved from the same
  corpus file is kept; a labelling that keeps every label leaves the
  manifest as it was.
* `eval`: every state of the set, checked by SHA-256 against the
  configuration (refused otherwise, exit status 3, before any model is
  built and before anything is written) and by content on the run's
  validation split against the report's arrays (a state with a non-finite
  value or whose medians deviate by more than `content_tol` is recorded as
  a content mismatch: it is still evaluated, and the script ends with exit
  status 5 after writing everything); then every state on every holdout
  instance, with the frozen metric suite. Outputs `<out>/<set>/eval.json`
  (inputs and their hashes, the checks, summaries, timings; complete when it
  records `arrays_sha256`) and `<out>/<set>/eval_arrays.npz` (per row, seed
  and instance: the five metrics; per load case: the relative energy gap and
  Pi_h(u); the same for the validation split). Every file is written to a
  temporary name, flushed to the disk and then moved into place.
* `all`: `label`, then `eval`.

    python scripts/holdout_run.py --config configs/holdout_v1.json --set h3d --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _torch_info(dev: str) -> dict:
    import torch

    out = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": dev,
           "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
           "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if str(dev).startswith("cuda"):
        out["gpu"] = torch.cuda.get_device_name(0)
    return out


def run_label(spec: dict, name: str, out: Path) -> dict:
    """One summary line: this attempt's counts (`reused`, `solves`; `kept`
    when the manifest already there was kept as it was) and the manifest's
    instances (`n`, `residual_max`, `source_labelled`)."""
    from fejepa.analysis.holdout import label_set

    man, att = label_set(name, spec["sets"][name], out, log=lambda m: print(m, flush=True))
    print(json.dumps({"set": name, "n": man["n"], "reused": att["reused"],
                      "solves": att["solves"], "kept": att["kept"],
                      "residual_max": max(r["residual"] for r in man["instances"]),
                      "source_labelled": sum(bool(r["source_labelled"])
                                             for r in man["instances"])}), flush=True)
    return man


def run_eval(spec: dict, spec_path: str, name: str, out: Path, dev: str,
             n_val: int | None = None, n_holdout: int | None = None) -> int:
    from fejepa.analysis.common import sha256_of
    from fejepa.analysis.holdout import (METRICS, check_labelled_set, check_states,
                                         content_check, holdout_instances, instance_record,
                                         predict, report_arrays, row_model, save_npz_synced,
                                         spec_sha256, worse_than_zero, write_json_synced)
    from fejepa.analysis.posthoc import run_files
    from fejepa.anchor.energy import pi_h
    from fejepa.data.archive import LazyArchives
    from fejepa.report import _git_describe

    sdef = spec["sets"][name]
    seeds = [int(s) for s in spec["seeds"]]
    rows = list(sdef["rows"])
    rp = Path(sdef["report"])
    if sha256_of(rp) != sdef["report_sha256"]:
        print(f"{rp}: not the report the configuration names", file=sys.stderr)
        return 3
    report = json.loads(rp.read_text())
    try:
        states = check_states(sdef, seeds)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    man = check_labelled_set(name, sdef, out)
    t0 = time.time()
    models = {k: row_model(report, sdef, k[0], p, k[1], dev) for k, p in states.items()}
    res = {"what": f"PREREG_HOLDOUT: the stored states on holdout set {name}",
           "git": _git_describe(), "torch": _torch_info(dev), "set": name,
           "config": spec_path, "config_sha256": spec_sha256(spec),
           "report": str(rp), "report_sha256": sha256_of(rp),
           "labels_manifest_sha256": sha256_of(out / "manifest.json"),
           "corpus_manifest_sha256": sdef["manifest_sha256"], "rows": rows, "seeds": seeds,
           "metrics": list(METRICS), "content_tol": spec["content_tol"],
           "states": {f"{r}_s{s}": {"file": str(p), "sha256": sdef["rows"][r]["sha256"][str(s)]}
                      for (r, s), p in states.items()},
           "excluded": list(sdef["excluded"]), "n_val_limit": n_val,
           "n_holdout_limit": n_holdout}
    write_json_synced(out / "eval.json", res)

    # ---- content: the run's validation split against the report's arrays --
    val_files = run_files(report, "val")
    if n_val:
        val_files = val_files[:n_val]
    nv = len(val_files)
    V = np.full((len(rows), len(seeds), nv, len(METRICS)), np.nan)
    VG = VP = VPS = None
    for i, arch in enumerate(LazyArchives(val_files)):
        if VG is None:
            VG = np.full((len(rows), len(seeds), nv, arch.n_loads), np.nan)
            VP = np.full(VG.shape, np.nan)
            VPS = np.full((nv, arch.n_loads), np.nan)
        VPS[i] = pi_h(arch.U_star, arch.K, arch.F)
        for (r, s), m in models.items():
            rec = instance_record(predict(m, arch, dev), arch)
            ri, si = rows.index(r), seeds.index(s)
            V[ri, si, i] = rec["metrics"]
            VG[ri, si, i] = rec["gap_load"]
            VP[ri, si, i] = rec["pi_load"]
        if (i + 1) % 32 == 0 or i + 1 == nv:
            print(f"[holdout] {name} val {i + 1}/{nv} | {time.time() - t0:.0f} s", flush=True)
    checks, mismatch = {}, []
    for (r, s) in models:
        got = {k: V[rows.index(r), seeds.index(s), :, j].tolist() for j, k in enumerate(METRICS)}
        ref = report_arrays(report, sdef, r, seeds.index(s))
        ref = {k: list(ref[k])[:nv] for k in METRICS}
        chk = content_check(got, ref, float(spec["content_tol"]))
        checks[f"{r}_s{s}"] = chk
        if not chk["ok"]:
            mismatch.append(f"{r}_s{s}")
    res.update(val={"n": nv, "files": [Path(f).name for f in val_files]},
               content_checks=checks, content_mismatch=mismatch,
               val_seconds=round(time.time() - t0, 1))
    write_json_synced(out / "eval.json", res)

    # ---- the holdout set ---------------------------------------------------
    n = len(man["instances"]) if not n_holdout else min(n_holdout, len(man["instances"]))
    L = None
    M = np.full((len(rows), len(seeds), n, len(METRICS)), np.nan)
    G = P = PS = None
    files, t1 = [], time.time()
    t_inf = {f"{r}_s{s}": 0.0 for (r, s) in models}
    for i, arch in enumerate(holdout_instances(name, sdef, out)):
        if i >= n:
            break
        if L is None:
            L = arch.n_loads
            G = np.full((len(rows), len(seeds), n, L), np.nan)
            P = np.full((len(rows), len(seeds), n, L), np.nan)
            PS = np.full((n, L), np.nan)
        files.append(Path(arch.path).name)
        PS[i] = pi_h(arch.U_star, arch.K, arch.F)
        for (r, s), m in models.items():
            ta = time.perf_counter()
            U = predict(m, arch, dev)
            t_inf[f"{r}_s{s}"] += time.perf_counter() - ta
            rec = instance_record(U, arch)
            ri, si = rows.index(r), seeds.index(s)
            M[ri, si, i] = rec["metrics"]
            G[ri, si, i] = rec["gap_load"]
            P[ri, si, i] = rec["pi_load"]
        if (i + 1) % 32 == 0 or i + 1 == n:
            print(f"[holdout] {name} holdout {i + 1}/{n} | {time.time() - t1:.0f} s", flush=True)
    summary, worse = {}, {}
    for r in rows:
        for s in seeds:
            ri, si = rows.index(r), seeds.index(s)
            summary[f"{r}_s{s}"] = {k: float(np.mean(M[ri, si, :, j]))
                                    for j, k in enumerate(METRICS)}
            worse[f"{r}_s{s}"] = worse_than_zero(G[ri, si], P[ri, si])
    save_npz_synced(out / "eval_arrays.npz", rows=np.array(rows), seeds=np.array(seeds),
                    metrics=np.array(METRICS), files=np.array(files), values=M,
                    gap_load=G, pi_load=P, pi_star=PS, val_values=V, val_gap_load=VG,
                    val_pi_load=VP, val_pi_star=VPS,
                    val_files=np.array([Path(f).name for f in val_files]))
    res.update(holdout={"n": n, "files_sha256": hashlib.sha256(
                   "\n".join(files).encode()).hexdigest()},
               summary=summary, worse_than_zero=worse, inference_seconds=t_inf,
               holdout_seconds=round(time.time() - t1, 1),
               seconds=round(time.time() - t0, 1),
               arrays_sha256=sha256_of(out / "eval_arrays.npz"))
    write_json_synced(out / "eval.json", res)
    print(json.dumps({"set": name, "n_val": nv, "n_holdout": n,
                      "content_median_rel_dev": {k: [c["disp_rel_l2_median_rel_dev"],
                                                     c["energy_gap_rel_median_rel_dev"]]
                                                 for k, c in checks.items()},
                      "content_mismatch": mismatch, "summary": summary,
                      "worse_than_zero": worse}), flush=True)
    return 5 if mismatch else 0


def check_stamp(spec: dict) -> None:
    """Refuse to run unless the pre-registration is stamped for this
    configuration (its CONFIG_SHA256[holdout_v1] line) and its last line is
    its own SHA-256."""
    import importlib.util

    from fejepa.report import verify_prereg

    p = Path(spec["prereg_file"])
    try:
        verify_prereg(spec, p, label="holdout_v1")
    except ValueError as exc:
        raise SystemExit(f"{p}: {exc}") from exc
    sp = importlib.util.spec_from_file_location(
        "stamp_prereg_holdout", Path(__file__).resolve().parent / "stamp_prereg_holdout.py")
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    if not mod.self_hash_ok(p.read_text(encoding="utf-8")):
        raise SystemExit(f"{p}: its last line is not its own SHA-256 (not stamped)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/holdout_v1.json")
    ap.add_argument("--set", required=True, choices=("h3d", "h2d"))
    ap.add_argument("--stage", required=True, choices=("label", "eval", "all"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-val", type=int, default=None,
                    help="validation instances for the content check (all; tests only)")
    ap.add_argument("--n-holdout", type=int, default=None,
                    help="holdout instances to evaluate (all; tests only)")
    a = ap.parse_args()
    from fejepa.analysis.common import resolve_device
    from fejepa.analysis.holdout import load_spec

    spec = load_spec(a.config)
    check_stamp(spec)
    out = Path(spec["out"]) / a.set
    code = 0
    if a.stage in ("label", "all"):
        run_label(spec, a.set, out)
    if a.stage in ("eval", "all"):
        code = run_eval(spec, a.config, a.set, out, resolve_device(a.device),
                        n_val=a.n_val, n_holdout=a.n_holdout)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
