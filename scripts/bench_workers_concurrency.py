#!/usr/bin/env python3
"""Single-GPU concurrency micro-bench (O2; ~10 minutes on the box).

Question: do concurrent training processes on one GPU raise AGGREGATE
throughput at in-band sizes? (The TF32 null result showed steps are not
GEMM-throughput-bound; whether they are launch-bound -- concurrency helps --
or bandwidth-bound -- it does not -- is decided here, not assumed.)

Method: W identical AR trainings (the production pretrain_unit path, spawn
processes via map_units) on the SAME mid-band bench instance, for
W in {1, 2, 3}; report aggregate steps/sec and speedup vs W=1.

Adoption rule (pre-declared): in-band workers=2 if speedup(2) >= 1.4;
workers=3 if additionally speedup(3) >= 1.9. The fine stage stays workers=1
regardless (23.5 GiB peak per process).

Usage on the box (bench_data must exist from the preconditions bench):
    python scripts/bench_workers_concurrency.py \
        [--instance runs/phase2/bench_data/one_inband_1] [--repeats 30]
        [--out runs/phase2/bench_workers.json]
Smoke anywhere: --smoke (synthetic tiny corpus, CPU-safe).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default="runs/phase2/bench_data/one_inband_1")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/phase2/bench_workers.json")
    a = ap.parse_args()

    import torch

    from fejepa.experiments.parallel import map_units, pretrain_unit
    from fejepa.experiments.protocol import load_split
    from fejepa.runtime import setup_torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy = setup_torch(dev, tf32=True)

    if a.smoke:
        import tempfile

        from fejepa.fe.synthetic import generate_synthetic_dataset

        ddir = generate_synthetic_dataset(
            Path(tempfile.mkdtemp()) / "wsmoke", n=2, seed=1)
        mcfg = {"dim": 16, "depth": 1, "heads": 2,
                "features": {"load_summary": True, "geometry": True}}
        a.repeats = 2
    else:
        ddir = a.instance
        mcfg = {"dim": 256, "depth": 8, "heads": 8, "mask_frac": 0.2,
                "scale_decode": True,
                "features": {"load_summary": True, "geometry": True,
                             "spatial_dim": 3}}
    files = [str(f) for f in load_split(str(ddir), 0, 1).pool_files[:1]]
    import tempfile

    state_scratch = Path(tempfile.mkdtemp(prefix="conc_states_"))

    res = {"device": dev, "numeric_policy": policy, "repeats": a.repeats,
           "instance": str(ddir), "rows": {}}
    base = None
    for w in a.workers:
        payloads = [{"kind": "fejepa", "model": mcfg, "seed": 100 + i,
                     "tf32": True, "precision": "fp32", "files": files,
                     "loss": "ar",
                     "pre": {"epochs": a.repeats, "lr": 1e-3, "device": dev,
                             "log_every": -1},
                     "state_path": str(state_scratch / f"w{w}_u{i}.pt"),
                     "tag": f"conc w{w} u{i}", "quiet": True}
                    for i in range(w)]
        t0 = time.perf_counter()
        map_units(pretrain_unit, payloads, w, f"concurrency W={w}")
        wall = time.perf_counter() - t0
        agg = w * a.repeats / wall
        row = {"wall_s": round(wall, 1),
               "aggregate_steps_per_s": round(agg, 3)}
        if base is None:
            base = agg
        row["speedup_vs_w1"] = round(agg / base, 3)
        res["rows"][str(w)] = row
        print(f"W={w}: wall {wall:.1f}s  aggregate {agg:.3f} steps/s  "
              f"speedup {row['speedup_vs_w1']}x", flush=True)

    s2 = res["rows"].get("2", {}).get("speedup_vs_w1", 0)
    s3 = res["rows"].get("3", {}).get("speedup_vs_w1", 0)
    res["adoption"] = {"rule": "w2 if s2>=1.4; w3 if also s3>=1.9",
                       "inband_workers": 3 if (s2 >= 1.4 and s3 >= 1.9)
                       else (2 if s2 >= 1.4 else 1),
                       "fine_workers": 1}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["adoption"], indent=1))


if __name__ == "__main__":
    main()
