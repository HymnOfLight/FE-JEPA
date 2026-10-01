#!/usr/bin/env python3
"""wp8 post-hoc (wrap-up item 4; Stage 1.36, paired arm Stage 1.37): does
E1's separation instrument read the geometry-descriptor INPUT rather than
learned structure?

On the run's own validation split (E1: the 256 instances its verdict used),
the separation readings (`measure_separation`: S with its bootstrap interval,
PC1, linear probe R^2, 1-NN, SIGReg monitors) of UNTRAINED models built from
the run's model configuration, for seeds 0/1/2, in three arms:

  geometry_input_true    the descriptor is an input (as trained) -- exactly
                         E1's seed-s initialisation
  geometry_input_zeroed  the same initialisation with the descriptor's input
                         weights set to zero (the encoder's first layer reads
                         the descriptor from the last GEOMETRY_DIM feature
                         columns): the paired control, identical otherwise
  geometry_input_false   a model built without the descriptor input (its
                         first layer has another shape, so its random
                         initialisation differs throughout)

Compare with the trained readings on record (records/wp8/e1/sep_*.json),
measured with the same function on the same instances in plain fp32. The PC1
variance share depends on the instances' descriptors only, so it must equal
the recorded value: the output checks that (`pc1_matches_record`).
Reported only; no verdict.

    python scripts/posthoc_probe_random_init.py --report records/wp8/e1/e1_2d_base/report.json \
        --out runs/wp8/posthoc/probe_random_init.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ARMS = ("geometry_input_true", "geometry_input_zeroed", "geometry_input_false")


def build_arm(cfg_model: dict, arm: str, seed: int, device: str):
    import torch

    from fejepa.analysis.common import build_model_from_config
    from fejepa.models.features import GEOMETRY_DIM

    mcfg = json.loads(json.dumps(cfg_model))
    mcfg.setdefault("features", {})["geometry"] = arm != "geometry_input_false"
    m = build_model_from_config(mcfg, seed=seed, device=device)
    if arm == "geometry_input_zeroed":
        inp = m.encoder.inp
        if inp.weight.shape[1] < GEOMETRY_DIM:
            raise SystemExit("the encoder's input layer is narrower than the descriptor")
        with torch.no_grad():
            inp.weight[:, -GEOMETRY_DIM:] = 0.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="E1 base report (the records copy)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--data", default=None, help="override the report's data dir")
    ap.add_argument("--record", default=None,
                    help="a recorded separation reading on the same instances (default: "
                         "sep_base_s0.json next to the report's directory)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from fejepa.analysis.common import resolve_device, sha256_of, write_json
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
    rec_path = Path(a.record) if a.record else Path(a.report).resolve().parents[1] / "sep_base_s0.json"
    record = json.loads(rec_path.read_text()) if rec_path.exists() else None
    res = {"what": "untrained-model separation readings (wp8 wrap-up item 4); reported only",
           "git": _git_describe(), "report": a.report, "report_sha256": sha256_of(a.report),
           "n_instances": len(archs), "device": dev, "arms": list(ARMS),
           "record": str(rec_path) if record else None, "readings": {}}
    keys = ("S_silhouette", "S_bootstrap_ci95", "probe_r2_geometry", "loo_1nn_bin_accuracy",
            "pc1_variance_share", "sigreg_monitor_pooled", "sigreg_monitor_tokens", "bins")
    for arm in ARMS:
        for s in a.seeds:
            m = build_arm(cfg["model"], arm, s, dev)
            r = measure_separation(m, archs)
            row = {k: r[k] for k in keys if k in r}
            res["readings"][f"{arm}_s{s}"] = row
            print(json.dumps({"arm": arm, "seed": s, **{k: v for k, v in row.items()
                                                       if k != "bins"}}), flush=True)
            if record is not None and "pc1_matches_record" not in res:
                got, want = float(r["pc1_variance_share"]), float(record["pc1_variance_share"])
                res["pc1_record"] = want
                res["pc1_matches_record"] = abs(got - want) <= 1e-9 * max(abs(want), 1.0)
                if record.get("n_instances") != len(archs):
                    res["pc1_matches_record"] = False
                print(json.dumps({"pc1_matches_record": res["pc1_matches_record"],
                                  "pc1": got, "pc1_record": want}), flush=True)
            write_json(a.out, res)                         # incremental
    write_json(a.out, res)


if __name__ == "__main__":
    main()
