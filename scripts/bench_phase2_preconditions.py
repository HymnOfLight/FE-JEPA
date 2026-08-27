#!/usr/bin/env python3
"""Phase-2 preconditions bench (PREREG_PHASE2 r8, Sec. 9 step 1).

Three mandatory measurements before stamping:
  (i)   transfer-scale feasibility: one lc = fine instance through the full
        model -- forward, backward, one optimiser step (executed as real
        pretrain steps) -- with peak GPU memory. Failure => the
        attention-memory fix lands and is committed FIRST.
  (ii)  torch.compile behaviour across >= 3 distinct in-band sizes plus the
        fine instance: recompilation counters per size; pathological growth
        => compile OFF by pre-stamp config edit.
  (iii) wall-clock projection for the full battery from MEASURED ms/step
        (D8: measured, not manual), for the provenance note.

Usage on the box, in-checkout:
    python scripts/bench_phase2_preconditions.py --config configs/phase2_v1.json \
        [--repeats 20] [--out runs/phase2/bench_preconditions.json]
Modes: --plan  (no torch/gmsh; prints the training-count plan)
       --smoke (synthetic backend, tiny model; validates the wiring anywhere)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def load_cfg(path: str) -> dict:
    return json.loads(Path(path).read_text())


def training_plan(cfg: dict) -> dict:
    """Per-arm honest step counts (r10): supervised arms train on b instances,
    not the corpus; mgn on its budget subset; no E1' battery (P2 subsumed)."""
    e8 = cfg["experiments"]["e8"]
    p3 = cfg["experiments"]["p3_transfer"]
    seeds = len(cfg.get("seeds", [0, 1, 2]))
    ep_pre = int(cfg.get("pretrain", {}).get("epochs", 200))
    ep_sup = int(cfg.get("sup", {}).get("epochs", 200))
    ep_fs = int(p3.get("fewshot_epochs", 200))
    budgets = [int(b) for b in e8["budgets"]]
    mgnb = [int(b) for b in e8.get("mgn_budgets", budgets)]
    n_ar = int(cfg["data"]["n"]) - int(cfg.get("split", {}).get("n_val", 256))
    steps = {
        "ar (pretrain, shared with P3)": seeds * ep_pre * n_ar,
        "labels (P1)": seeds * ep_sup * sum(budgets),
        "labels_anchor policy (P1; KP3 source)": seeds * ep_sup * sum(budgets),
        "mgn (subset %s)" % mgnb: seeds * ep_sup * sum(mgnb),
    }
    fine_steps = {
        "P3 few-shot fine-tune": seeds * ep_fs * sum(p3["fewshot_budgets"]),
        "P3 scratch-at-fine": seeds * ep_fs * sum(p3["fewshot_budgets"]),
    }
    return {"inband_steps": steps, "fine_steps": fine_steps,
            "inband_total": sum(steps.values()),
            "fine_total": sum(fine_steps.values()),
            "note": "zero-shot/naive evaluations add forward-only cost, "
                    "reported separately by the run itself"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config_pos", nargs="?", help="config path (positional)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="runs/phase2/bench_preconditions.json")
    a = ap.parse_args()
    cfg_path = a.config or a.config_pos
    if not cfg_path:
        ap.error("config path required (positional or --config)")
    cfg = load_cfg(cfg_path)
    if a.plan:
        print(json.dumps(training_plan(cfg), indent=1))
        return

    import torch

    from fejepa.data.archive import load_instance
    from fejepa.experiments.parallel import _build_model
    from fejepa.experiments.protocol import load_split
    from fejepa.train.losses import AR_CONFIG
    from fejepa.train.pretrain import PretrainConfig, pretrain

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    work = Path(a.out).parent / "bench_data"
    work.mkdir(parents=True, exist_ok=True)

    if a.smoke:
        from fejepa.fe.synthetic import generate_synthetic_dataset

        sizes = {}
        for tag in ("inband_0", "inband_1", "inband_2", "fine"):
            d = generate_synthetic_dataset(work / f"smoke_{tag}", n=1,
                                           seed=hash(tag) % 1000)
            sizes[tag] = load_instance(load_split(d, 0, 1).pool_files[0])
        mcfg = {"dim": 16, "depth": 1, "heads": 2,
                "features": {"load_summary": True, "geometry": True}}
    else:
        from fejepa.fe.gmsh3d import generate_gmsh3d_dataset

        lo, hi = cfg["data"]["lc_range"]
        lcs = {"inband_0": lo, "inband_1": (lo + hi) / 2, "inband_2": hi,
               "fine": cfg["data_transfer"]["lc"]}
        sizes = {}
        for tag, lc in lcs.items():
            d = work / f"one_{tag}"
            if not (d / "manifest.json").exists():
                generate_gmsh3d_dataset(d, 1, 999, labelled="none", lc=lc)
            sizes[tag] = load_instance(load_split(d, 0, 1).pool_files[0])
        mcfg = cfg["model"]

    model = _build_model({"kind": "fejepa", "model": mcfg, "seed": 0})
    if cfg.get("runtime", {}).get("compile", False) and not a.smoke:
        model = torch.compile(model)

    res = {"device": dev, "torch": torch.__version__, "smoke": a.smoke,
           "phases": {}, "compile": {"enabled": bool(
               cfg.get("runtime", {}).get("compile", False))}}

    def counters_snapshot():
        try:
            import torch._dynamo as dyn

            return {k: dict(v) for k, v in dyn.utils.counters.items()}
        except Exception:                                     # noqa: BLE001
            return {}

    for tag, arch in sizes.items():
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        before = counters_snapshot()
        t0 = time.perf_counter()
        pretrain(model, [arch],
                 PretrainConfig(loss=AR_CONFIG, epochs=a.repeats,
                                lr=float(cfg.get("pretrain", {})
                                         .get("lr", 1e-3)),
                                device=dev, log_every=-1, seed=0))
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000 / a.repeats
        res["phases"][tag] = {
            "n_nodes": int(getattr(arch, "n_nodes", 0) or
                           len(getattr(arch, "coords", []))),
            "ms_per_step": round(ms, 2),
            "peak_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                         if dev == "cuda" else None),
            "dynamo_counters_after": counters_snapshot(),
            "dynamo_counters_delta_note":
                "compare with previous phase; growth per new size = recompile",
        }
        res["compile"].setdefault("first_counters", before)

    plan = training_plan(cfg)
    ms_in = res["phases"]["inband_1"]["ms_per_step"]
    ms_fi = res["phases"]["fine"]["ms_per_step"]
    hours = (plan["inband_total"] * ms_in
             + plan["fine_total"] * ms_fi) / 3.6e6
    res["projection"] = {
        "per_arm": plan, "ms_in_band_mid": ms_in, "ms_fine": ms_fi,
        "precision": (cfg.get("runtime") or {}).get("precision", "fp32"),
        "hours_at_measured_precision": round(hours, 1),
        "note": ("per-arm honest projection (r10); if the bf16 pilot passes, "
                 "divide by its measured speedup for the stamped provenance "
                 "number (D8: measured, not manual)")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("device", "phases", "projection")},
                     indent=1))


if __name__ == "__main__":
    main()
