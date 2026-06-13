"""Config-driven experiment runner for full-scale (GPU) FE-JEPA runs.

Reads a JSON config describing the dataset and which experiments to run, then
generates data (if needed) and executes the regime comparison, the falsification
battery, and/or the label-efficiency sweep, writing one JSON report per stage.

This is the entry point intended for the headline Phase-1/2 runs:

    fejepa run-config configs/phase1_2d.json

On a machine with a GPU, set ``"device": "cuda"`` in the config.  All numeric
results land in the JSON reports under the configured ``out`` paths; copy them
into ``RESULTS.md`` (which ships with blank tables) to record the run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fejepa.models.fejepa import FEJEPAConfig


def _maybe_generate(ds: dict) -> Path:
    from fejepa.data.archive import read_manifest
    from fejepa.fe.generator import GeneratorConfig, generate_dataset

    out = Path(ds["out"])
    exists = (out / "manifest.json").exists()
    if exists and not ds.get("regenerate", False):
        try:
            n_have = read_manifest(out)["n_instances"]
            if n_have >= ds["n"]:
                print(f"[run-config] reusing dataset at {out} ({n_have} instances)")
                return out
        except Exception:  # noqa: BLE001 - regenerate on any manifest problem
            pass
    print(f"[run-config] generating {ds['n']} instances into {out} ...")
    cfg = GeneratorConfig(max_holes=ds.get("max_holes", 3))
    generate_dataset(out, n_instances=ds["n"], seed=ds.get("seed", 0),
                     labelled=ds.get("labelled", True), cfg=cfg, verbose=True)
    return out


def run_config(config_path: str | Path) -> dict:
    from fejepa.experiments.falsification import BatteryConfig, load_split, run_battery
    from fejepa.experiments.regimes import compare_training_regimes
    from fejepa.train.pretrain import PretrainConfig
    from fejepa.train.supervised import SupervisedConfig, label_efficiency_sweep

    cfg = json.loads(Path(config_path).read_text())
    device = cfg.get("device", "cpu")
    model_cfg = FEJEPAConfig(**cfg.get("model", {}))
    data_dir = _maybe_generate(cfg["dataset"])

    summary: dict = {"config": cfg, "data_dir": str(data_dir), "reports": {}}

    if cfg.get("regimes", {}).get("enabled"):
        r = cfg["regimes"]
        print("[run-config] regime comparison ...")
        pool, val = load_split(data_dir, r.get("n_val", 64), r.get("seed", 0))
        sup = SupervisedConfig(epochs=r["epochs"], lr=r.get("lr", 1.5e-3), model=model_cfg, device=device)
        pre = PretrainConfig(epochs=r["epochs"], lr=r.get("lr", 1.5e-3), model=model_cfg, device=device)
        t = time.time()
        rep = compare_training_regimes(
            pool, val, n_train=r["n_train"], sup_cfg=sup, pre_cfg=pre,
            out_report=r.get("out"),
        )
        rep["elapsed_sec"] = time.time() - t
        summary["reports"]["regimes"] = r.get("out") or rep

    if cfg.get("battery", {}).get("enabled"):
        b = cfg["battery"]
        print("[run-config] falsification battery ...")
        bcfg = BatteryConfig(
            budgets=b["budgets"], n_val=b.get("n_val", 64), seed=b.get("seed", 0),
            decision_budget=b.get("decision_budget", 64), lambda_phys=b.get("lambda_phys", 1.0),
            device=device,
            sup=SupervisedConfig(epochs=b["epochs"], lr=b.get("lr", 1.5e-3), model=model_cfg),
        )
        run_battery(data_dir, cfg=bcfg, experiments=b.get("experiments"), out_report=b.get("out"))
        summary["reports"]["battery"] = b.get("out")

    if cfg.get("label_efficiency", {}).get("enabled"):
        le = cfg["label_efficiency"]
        print("[run-config] label-efficiency sweep ...")
        sup = SupervisedConfig(epochs=le["epochs"], lr=le.get("lr", 1.5e-3), model=model_cfg, device=device)
        scratch = label_efficiency_sweep(
            data_dir, budgets=le["budgets"], n_val=le.get("n_val", 64), cfg=sup,
            init_ckpt=le.get("init_ckpt"), seed=le.get("seed", 0),
        )
        out = le.get("out")
        if out:
            Path(out).write_text(json.dumps({"label_efficiency": scratch}, indent=2))
        summary["reports"]["label_efficiency"] = out or scratch

    print("[run-config] done.")
    return summary
