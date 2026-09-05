#!/usr/bin/env python3
"""CLI: cross-geometry latent separation (see fejepa.analysis.separation).

    python scripts/latent_separation.py --config <cfg.json> --state <state.pt> \
        --data <corpus dir> --n-instances 256 --out runs/wp8/sep_<arm>.json
    python scripts/latent_separation.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2_v1.json")
    ap.add_argument("--state", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-instances", type=int, default=256)
    ap.add_argument("--subset", choices=("val", "pool"), default="val",
                    help="which instances to measure on; 'val' = the run's held-out "
                         "validation split from the config's `split` block (the "
                         "pre-registered choice)")
    ap.add_argument("--device", default="auto", help="cuda if available (default), or cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/wp8/latent_separation.json")
    a = ap.parse_args()

    from fejepa.analysis.common import build_model_from_config, instance_files, write_json
    from fejepa.analysis.separation import measure_separation
    from fejepa.data.archive import load_instance

    if a.smoke:
        import tempfile

        from fejepa.fe.synthetic import generate_synthetic_dataset

        ddir = generate_synthetic_dataset(Path(tempfile.mkdtemp()) / "sep", n=16, seed=5)
        mcfg = {"dim": 16, "depth": 1, "heads": 2,
                "features": {"load_summary": True, "geometry": True}}
        files, model = instance_files(ddir), build_model_from_config(mcfg)
    else:
        cfg = json.loads(Path(a.config).read_text())
        mcfg = cfg["model"]
        files = instance_files(a.data, a.n_instances, split=cfg.get("split"), subset=a.subset)
        model = build_model_from_config(mcfg, a.state, device=a.device)
    res = measure_separation(model, [load_instance(f) for f in files])
    res.update({"subset": a.subset, "state": a.state, "kind": mcfg.get("kind", "fejepa"), "smoke": a.smoke})
    write_json(a.out, res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
