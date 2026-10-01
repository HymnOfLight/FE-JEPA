#!/usr/bin/env python3
"""wp8 Stage 1.36 post-hoc (item 4): does E1's separation instrument read the
geometry-descriptor INPUT rather than learned structure?

On the run's own validation split (E1: the 256 instances its verdict used),
the separation readings (`measure_separation`: S, PC1, linear probe R^2,
1-NN, SIGReg monitors) of UNTRAINED models built from the run's model
configuration -- with the geometry descriptor as an input (as trained) and
without it -- for seeds 0/1/2. Compare with the trained readings already on
record (records/wp8/e1/sep_*.json). Reported only; no verdict.

    python scripts/posthoc_probe_random_init.py --report records/wp8/e1/e1_2d_base/report.json \
        --out runs/wp8/posthoc/probe_random_init.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="E1 base report (the records copy)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--data", default=None, help="override the report's data dir")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from fejepa.analysis.common import (build_model_from_config, resolve_device, sha256_of,
                                        write_json)
    from fejepa.analysis.posthoc import run_files
    from fejepa.analysis.separation import measure_separation
    from fejepa.data.archive import load_instance
    from fejepa.report import _git_describe
    from fejepa.runtime import setup_torch

    report = json.loads(Path(a.report).read_text())
    cfg = report["config"]
    dev = resolve_device(a.device)
    # the recorded separation files (records/wp8/e1/sep_*.json) were measured in
    # plain fp32 (latent_separation.py sets no TF32 policy): match them
    setup_torch(dev, tf32=False)
    files = run_files(report, "val", a.n, a.data)
    archs = [load_instance(f) for f in files]
    res = {"what": "untrained-model separation readings (wp8 Stage 1.36 item 4); reported only",
           "git": _git_describe(), "report": a.report, "report_sha256": sha256_of(a.report),
           "n_instances": len(archs), "device": dev, "readings": {}}
    keys = ("S_silhouette", "probe_r2_geometry", "loo_1nn_bin_accuracy", "pc1_variance_share",
            "sigreg_monitor_pooled", "sigreg_monitor_tokens")
    for geo in (True, False):
        mcfg = json.loads(json.dumps(cfg["model"]))
        mcfg.setdefault("features", {})["geometry"] = geo
        for s in a.seeds:
            m = build_model_from_config(mcfg, seed=s, device=dev)
            r = measure_separation(m, archs)
            row = {k: r[k] for k in keys}
            res["readings"][f"geometry_input_{str(geo).lower()}_s{s}"] = row
            print(json.dumps({"geometry_input": geo, "seed": s, **row}), flush=True)
    write_json(a.out, res)


if __name__ == "__main__":
    main()
