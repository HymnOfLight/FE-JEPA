#!/usr/bin/env python3
"""CLI: intrinsic dimension of latent tokens (see fejepa.analysis.intrinsic_dim).

    python scripts/intrinsic_dimension.py --config <cfg.json> --state <state.pt> \
        --data <corpus dir> --n-instances 32 --out runs/wp8/intrinsic_dim.json
    python scripts/intrinsic_dimension.py --smoke
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
    ap.add_argument("--n-instances", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--subset", choices=("val", "pool"), default="val",
                    help="which instances to measure on; 'val' = the run's held-out "
                         "validation split from the config's `split` block (the "
                         "pre-registered choice)")
    ap.add_argument("--device", default="auto", help="cuda if available (default), or cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/wp8/intrinsic_dim.json")
    a = ap.parse_args()

    from fejepa.analysis.common import build_model_from_config, instance_files, write_json
    from fejepa.analysis.intrinsic_dim import measure_intrinsic_dimension
    from fejepa.data.archive import load_instance

    if a.smoke:
        import tempfile

        from fejepa.fe.synthetic import generate_synthetic_dataset

        ddir = generate_synthetic_dataset(Path(tempfile.mkdtemp()) / "id", n=6, seed=2)
        mcfg = {"dim": 16, "depth": 1, "heads": 2,
                "features": {"load_summary": True, "geometry": True}}
        files, model = instance_files(ddir, 4), build_model_from_config(mcfg)
    else:
        cfg = json.loads(Path(a.config).read_text())
        mcfg = cfg["model"]
        files = instance_files(a.data, a.n_instances, split=cfg.get("split"), subset=a.subset)
        model = build_model_from_config(mcfg, a.state, device=a.device)
    res = measure_intrinsic_dimension(model, [load_instance(f) for f in files], a.max_rows)
    res.update({"subset": a.subset, "state": a.state, "smoke": a.smoke})
    write_json(a.out, res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
