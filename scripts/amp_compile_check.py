"""WP7: AMP / torch.compile validation protocol (Manual sec 8's gated item).

Measures, on one small 3D training unit, the numeric deviation and step timing
of (a) plain fp32, (b) autocast mixed precision, and (c) torch.compile against
the fp32 reference -- same seeds, same data. This script GATHERS the numbers;
acceptance thresholds are frozen later in the Phase-2 pre-registration, not
here. Run on the target GPU box for the numbers that count; a CPU run
validates the harness only (bf16 autocast + compile both exercise on CPU).

Usage:
  PYTHONPATH=src python3 scripts/amp_compile_check.py \
      --steps 30 --out runs/amp_check/amp_compile.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _provenance() -> dict:
    """Best-effort provenance for measurement JSONs (git describe, versions)."""
    import subprocess
    prov = {}
    try:
        prov["git"] = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10).stdout.strip() or "unavailable"
    except Exception:                                     # noqa: BLE001
        prov["git"] = "unavailable"
    prov["numpy"] = np.__version__
    try:
        import gmsh
        prov["gmsh"] = ".".join(str(v) for v in gmsh.GMSH_API_VERSION.split("."))             if isinstance(getattr(gmsh, "GMSH_API_VERSION", None), str) else             str(getattr(gmsh, "GMSH_API_VERSION", "unknown"))
    except Exception:                                     # noqa: BLE001
        pass
    return prov


def _unit(device, steps, mode, seed=0):
    import torch

    from fejepa.fe.tet3d import tet_instance
    from fejepa.models.features import FeatureSpec
    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    archs = [tet_instance(rng, labelled=True) for _ in range(4)]
    cfg = FEJEPAConfig(dim=32, depth=2, heads=2,
                       features=FeatureSpec(load_summary=True, geometry=True,
                                            spatial_dim=3))
    model = build_fejepa(cfg).to(device)
    if mode == "compile":
        model = torch.compile(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    packs = [model.prepare_instance(a, device) if hasattr(model, "prepare_instance")
             else model._orig_mod.prepare_instance(a, device) for a in archs]
    tgts = [torch.as_tensor(a.U_star, dtype=torch.float32, device=device)
            for a in archs]
    amp_dtype = torch.bfloat16 if device == "cpu" else torch.float16
    use_amp = mode == "amp"
    losses, t0 = [], time.perf_counter()
    fwd = (model.forward_instance if hasattr(model, "forward_instance")
           else model._orig_mod.forward_instance)
    for s in range(steps):
        pack, tgt = packs[s % 4], tgts[s % 4]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype,
                            enabled=use_amp):
            u = fwd(pack)
            loss = torch.mean((u - tgt) ** 2)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    wall = time.perf_counter() - t0
    with torch.no_grad():
        u_final = fwd(packs[0]).float().cpu().numpy()
    return {"mode": mode, "losses": losses, "steps_per_s": round(steps / wall, 2),
            "final_pred": u_final}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/amp_check/amp_compile.json")
    a = ap.parse_args(argv)
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ref = _unit(device, a.steps, "fp32")
    rows = [dict(mode="fp32", steps_per_s=ref["steps_per_s"],
                 final_loss=ref["losses"][-1], max_pred_dev=0.0,
                 loss_dev_last=0.0)]
    scale = float(np.abs(ref["final_pred"]).max()) + 1e-12
    for mode in ("amp", "compile"):
        r = _unit(device, a.steps, mode)
        rows.append(dict(mode=mode, steps_per_s=r["steps_per_s"],
                         final_loss=r["losses"][-1],
                         max_pred_dev=float(np.abs(r["final_pred"]
                                                   - ref["final_pred"]).max() / scale),
                         loss_dev_last=abs(r["losses"][-1] - ref["losses"][-1])
                         / (abs(ref["losses"][-1]) + 1e-30)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"device": device, "steps": a.steps,
               "torch": torch.__version__, "provenance": _provenance(),
               "rows": rows,
               "note": "numbers gathered for the Phase-2 prereg; thresholds "
                       "are frozen there, not here"}
    out.write_text(json.dumps(payload, indent=1))
    for r in rows:
        print(f"[amp-check] {r['mode']:<8} {r['steps_per_s']:>7} steps/s | "
              f"final loss {r['final_loss']:.6e} | rel pred dev "
              f"{r['max_pred_dev']:.3e} | rel loss dev {r['loss_dev_last']:.3e}")
    print(f"[amp-check] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
