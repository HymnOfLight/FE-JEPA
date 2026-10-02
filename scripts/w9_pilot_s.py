#!/usr/bin/env python3
"""wp9 Stage 0c: a CPU pilot of the S switch at toy scale, run before
PREREG_W9 is stamped. Exploratory -- not evidence for H2, which is decided on
the box at E1's scale, seeds and evaluation sets.

Pre-declared (2 October 2026, before the pilot ran):
  questions  (a) does an S model train stably for thousands of steps; (b) does
             S's mechanism show: on meshes finer than training, does the
             max-scaled baseline's amplitude deficit (c* above 1, growing as h
             shrinks) shrink under S?
  red flags  S's readings non-finite, or S's in-band energy gap more than
             twice the baseline's (seed means): reported to the PI before the
             stamp. No other decision is taken on these numbers; the S factor
             (1/64) stays fixed by PREREG_W9 Sec. 2's data-free argument.
  data       its own draws only: a gmsh training-family corpus (seed 920), an
             F5-like family (32 instances at h = 0.025, seed 92005) and an
             R-like remesh set (8 geometries at the five R mesh sizes, seed
             92006); none of PREREG_W9's sets (seeds 91001-91006) is generated
             or read.
  model      dim 32, depth 2, heads 4 (E1: 256 / 8 / 8); AR on a pool of 256 x
             12 epochs (3,072 steps per seed; E1: 204,800); seeds 0-1 for both
             arms; CPU, two workers.
  as run     the first attempt (two workers) trained both baseline seeds and
             saved their states; one worker was then killed by the container's
             memory limit (5.8 GiB) while evaluating: on CPU, PyTorch's
             inference path of nn.MultiheadAttention materializes the attention
             matrices (4 load cases x 4 heads x N^2 floats, ~3.9 GB at the
             largest F5-like mesh, N = 7,760; CUDA evaluates without them). The
             pilot was rerun with --workers 1 --reuse-states: scheduling only,
             the saved states were evaluated, and S was trained the same way.
Also a reading that needs no training: on the R-like set, the battery's
largest nodal force and its summed |F| per mesh size relative to each
geometry's coarsest mesh (the scale lemma: Theta(h) and mesh-free in 2D).

    python scripts/w9_pilot_s.py --work runs/w9_pilot --out records/wp9/pilot_s_cpu/summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PILOT = {"corpus_seed": 920, "n": 288, "n_val": 32, "pool": 256, "epochs": 12, "seeds": 2,
         "dim": 32, "depth": 2, "heads": 4, "f5_seed": 92005, "f5_n": 32,
         "r_seed": 92006, "r_geometries": 8, "workers": 2}
S_KEYS = {"decode_scale": "l1", "decode_scale_factor": 0.015625}
RED_FLAG_RATIO = 2.0


def make_sets(work: Path) -> dict:
    from fejepa.fe.ood2d import generate_family, generate_remesh

    sets = {"F5": work / "F5p", "R": work / "Rp"}
    if not (sets["F5"] / "manifest.json").is_file():
        generate_family(sets["F5"], "F5", PILOT["f5_n"], seed=PILOT["f5_seed"])
    if not (sets["R"] / "manifest.json").is_file():
        generate_remesh(sets["R"], PILOT["r_geometries"], seed=PILOT["r_seed"])
    return sets


def config(work: Path, arm: str, sets: dict, workers: int = PILOT["workers"]) -> dict:
    cfg = json.loads((ROOT / "configs" / "e1_2d_base.json").read_text())
    cfg["data"] = {"dir": str(work / "data2d"), "n": PILOT["n"], "seed": PILOT["corpus_seed"],
                   "backend": "gmsh"}
    cfg["split"] = {"n_val": PILOT["n_val"], "seed": 1}
    cfg["model"].update(dim=PILOT["dim"], depth=PILOT["depth"], heads=PILOT["heads"])
    if arm == "s":
        cfg["model"].update(S_KEYS)
        cfg["model"]["features"] = dict(cfg["model"]["features"], load_density=True)
    cfg["experiments"]["e8"].update(pool_sizes=[PILOT["pool"]], ar_epochs=PILOT["epochs"],
                                   seeds=PILOT["seeds"])
    cfg["evaluation"] = {"holdouts": {k: {"dir": str(v), "family": k} for k, v in sets.items()},
                         "amplitude": True}
    cfg.update(device="cpu", workers=workers, tf32=False,
               out=str(work / arm / "report.json"), prereg_guard=False)
    cfg.pop("prereg_file", None)
    cfg["_comment"] = f"wp9 Stage 0c CPU pilot, arm {arm} (scripts/w9_pilot_s.py); not a stamped run"
    return cfg


def scale_readings(r_dir: Path) -> dict:
    """Per mesh size: the median over geometries of max|F| and sum|F|
    relative to the geometry's coarsest mesh (no training involved)."""
    from fejepa.data.archive import load_instance, load_manifest
    from fejepa.models.features import battery_fscale, battery_l1

    recs = load_manifest(r_dir)["instances"]
    by = {}
    for r in recs:
        a = load_instance(r_dir / r["file"])
        by.setdefault(r["geometry"], {})[float(r["target_h"])] = (battery_fscale(a.F),
                                                                  battery_l1(a.F))
    hs = sorted({h for g in by.values() for h in g}, reverse=True)
    out = {}
    for h in hs:
        mx = [g[h][0] / g[hs[0]][0] for g in by.values()]
        l1 = [g[h][1] / g[hs[0]][1] for g in by.values()]
        out[repr(h)] = {"max_ratio_median": statistics.median(mx),
                        "l1_ratio_median": statistics.median(l1),
                        "h_ratio": h / hs[0]}
    return out


def summarize(reports: dict, r_manifest: dict) -> dict:
    """The pilot's readings on the sets it has (in band, F5-like, R-like),
    with the adjudicator's own accessors."""
    from fejepa.analysis.adjudicate_w9 import (cell, remesh_readings, seed_amp, seed_means,
                                               seed_ratios)

    out = {}
    for arm, rep in reports.items():
        per = {s: {m: seed_means(cell(rep, s), m) for m in ("energy_gap_rel", "disp_rel_l2")}
               for s in ("val", "F5", "R")}
        amp = {k: seed_amp(cell(rep, "F5"), k) for k in ("c_star_median", "disp_c", "egap_c")}
        out[arm] = {
            "val_energy_gap": statistics.fmean(per["val"]["energy_gap_rel"]),
            "val_disp": statistics.fmean(per["val"]["disp_rel_l2"]),
            "F5_energy_gap": statistics.fmean(per["F5"]["energy_gap_rel"]),
            "F5_disp": statistics.fmean(per["F5"]["disp_rel_l2"]),
            "F5_over_val_disp": statistics.fmean(seed_ratios(rep)),
            "F5_c_star_median": statistics.fmean(amp["c_star_median"]),
            "F5_disp_after_c_star": statistics.fmean(amp["disp_c"]),
            "per_seed": {**per, "F5_amplitude": amp, "F5_over_val_disp": seed_ratios(rep)},
            "remesh": {h: {k: statistics.fmean(v) for k, v in row.items()}
                       for h, row in remesh_readings(rep, r_manifest).items()},
        }
    base, s = out.get("base"), out.get("s")
    flags = []
    if s is not None:
        vals = [s["val_energy_gap"], s["val_disp"], s["F5_disp"], s["F5_c_star_median"]]
        if any(v is None or not math.isfinite(v) for v in vals):
            flags.append("S has a non-finite or missing reading")
        if base and s["val_energy_gap"] and base["val_energy_gap"] and \
                s["val_energy_gap"] > RED_FLAG_RATIO * base["val_energy_gap"]:
            flags.append(f"S's in-band energy gap is more than {RED_FLAG_RATIO}x the baseline's")
    return {"arms": out, "red_flags": flags}


def main() -> None:
    from fejepa.analysis.adjudicate_w9 import load_json
    from fejepa.experiments.runner import run_config

    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="runs/w9_pilot")
    ap.add_argument("--out", default=None, help="summary JSON (default: <work>/summary.json)")
    ap.add_argument("--arms", nargs="+", default=["base", "s"])
    ap.add_argument("--workers", type=int, default=PILOT["workers"])
    ap.add_argument("--reuse-states", action="store_true",
                    help="consume the states an earlier attempt saved (scheduling only)")
    a = ap.parse_args()
    work = Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sets = make_sets(work)
    timing = {"sets_s": round(time.time() - t0, 1)}
    reports = {}
    for arm in a.arms:
        rep = work / arm / "report.json"
        if not rep.is_file():
            p = work / f"{arm}.json"
            p.write_text(json.dumps(config(work, arm, sets, a.workers), indent=1) + "\n")
            t1 = time.time()
            run_config(str(p), reuse_states=a.reuse_states)
            timing[f"{arm}_s"] = round(time.time() - t1, 1)
        reports[arm] = load_json(rep)
    res = {"what": "wp9 Stage 0c CPU pilot of S (exploratory; scripts/w9_pilot_s.py)",
           "design": PILOT, "s_keys": S_KEYS, "timing": timing,
           "as_run": {"workers": a.workers, "reuse_states": bool(a.reuse_states)},
           "scale": scale_readings(sets["R"]),
           **summarize(reports, load_json(sets["R"] / "manifest.json"))}
    out = Path(a.out) if a.out else work / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("scale", "red_flags")}, indent=1))
    for arm, r in res["arms"].items():
        print(arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                    if k not in ("per_seed", "remesh")})
        print("  remesh", {h: {k: round(v, 3) for k, v in row.items()}
                           for h, row in r["remesh"].items()})


if __name__ == "__main__":
    main()
