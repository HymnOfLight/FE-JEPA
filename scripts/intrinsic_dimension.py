#!/usr/bin/env python3
"""Intrinsic dimension of FE-JEPA latent tokens (wp8-lejepa, Stage 0, B1 gate).

LeWorldModel's recorded weakness: SIGReg pushes toward a d-dimensional
isotropic Gaussian and underperforms when the data's intrinsic dimension is
far below d (their Two-Room case). Before any latent-shaping arm is
pre-registered we therefore MEASURE, on a trained checkpoint:
  * TwoNN intrinsic dimension (Facco et al. 2017) of the node-token cloud,
  * PCA cumulative-variance dimensions at 90/95/99%,
  * the SIGReg monitor value of the raw latents (anisotropy baseline),
  * whether the encoder's last module is a LayerNorm (LeWM: a final LayerNorm
    blocks distribution shaping; a BatchNorm projector must follow it).

Usage (on a box with a corpus and a state):
    python scripts/intrinsic_dimension.py --config configs/phase2_v1.json \
        --state runs/phase2/e8_states/ar_p1024_s0.pt --data runs/data3d_phase2 \
        --n-instances 32 --out runs/wp8/intrinsic_dim.json
Anywhere: --smoke (synthetic corpus, random model; validates the wiring).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def twonn(x, discard_frac: float = 0.1) -> float:
    """TwoNN intrinsic dimension: fit -log(1-F(mu)) = d log(mu) through the
    origin on the ratios mu = r2/r1 of the two nearest-neighbour distances."""
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    # pairwise distances in chunks (memory-safe)
    r1 = np.empty(n)
    r2 = np.empty(n)
    chunk = 2048
    sq = (x * x).sum(1)                                          # (N,)
    for s in range(0, n, chunk):
        blk = x[s:s + chunk]
        # |a-b|^2 = |a|^2 + |b|^2 - 2 a.b : (chunk, N) only -- never (chunk, N, d)
        d = sq[s:s + chunk, None] + sq[None, :] - 2.0 * blk @ x.T
        np.maximum(d, 0.0, out=d)
        d[np.arange(blk.shape[0]), np.arange(s, s + blk.shape[0])] = np.inf
        part = np.partition(d, 1, axis=1)[:, :2]
        part.sort(axis=1)
        r1[s:s + chunk] = np.sqrt(part[:, 0])
        r2[s:s + chunk] = np.sqrt(part[:, 1])
    mu = r2 / np.maximum(r1, 1e-12)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1.0)])
    # Facco et al. 2017: the empirical CDF is taken over ALL points; the
    # top `discard_frac` of ratios is excluded from the FIT only (they are
    # the noisy tail), never re-normalised away.
    f = np.arange(1, mu.size + 1) / (mu.size + 1)
    keep = int((1.0 - discard_frac) * mu.size)
    xs, ys = np.log(mu[:keep]), -np.log(1.0 - f[:keep])
    return float((xs * ys).sum() / max((xs * xs).sum(), 1e-12))


def pca_dims(x, levels=(0.90, 0.95, 0.99)) -> dict:
    import numpy as np

    xc = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    ev = s ** 2 / (s ** 2).sum()
    cum = np.cumsum(ev)
    return {f"pca_{int(l * 100)}": int(np.searchsorted(cum, l) + 1) for l in levels}


def last_executed_module(model, feats, pack_for_hooks=None) -> str:
    """Name of the leaf module that fires LAST in `model.encode(feats)`
    (forward-hook order, not registration order)."""
    import torch

    fired = []
    hooks = []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: fired.append((n, type(mod).__name__))))
    try:
        with torch.no_grad():
            if getattr(model, "needs_pack", False):
                model.encode(feats, pack_for_hooks)
            else:
                model.encode(feats)
    finally:
        for h in hooks:
            h.remove()
    return fired[-1][1] if fired else "none"


def last_module_is_layernorm(model, feats, pack_for_hooks=None) -> bool:
    return last_executed_module(model, feats, pack_for_hooks) == "LayerNorm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2_v1.json")
    ap.add_argument("--state", default=None, help="fejepa state_dict (.pt)")
    ap.add_argument("--data", default=None, help="instance directory")
    ap.add_argument("--n-instances", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/wp8/intrinsic_dim.json")
    a = ap.parse_args()

    import torch

    from fejepa.data.archive import load_instance
    from fejepa.experiments.parallel import _build_model
    from fejepa.experiments.protocol import load_split
    from fejepa.train.sigreg import sigreg_monitor

    if a.smoke:
        import tempfile

        from fejepa.fe.synthetic import generate_synthetic_dataset

        ddir = generate_synthetic_dataset(Path(tempfile.mkdtemp()) / "id", n=6, seed=2)
        mcfg = {"dim": 16, "depth": 1, "heads": 2,
                "features": {"load_summary": True, "geometry": True}}
        files = load_split(str(ddir), 0, 1).pool_files[:4]
        model = _build_model({"kind": "fejepa", "model": mcfg, "seed": 0})
    else:
        cfg = json.loads(Path(a.config).read_text())
        mcfg = cfg["model"]
        files = load_split(a.data, 0, 1).pool_files[:a.n_instances]
        model = _build_model({"kind": "fejepa", "model": mcfg, "seed": 0})
        sd = torch.load(a.state, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
    model.eval()

    rows = []
    first_feats = None
    with torch.no_grad():
        for f in files:
            arch = load_instance(f)
            pack = model.prepare_instance(arch, "cpu")
            if first_feats is None:
                first_feats, first_pack = pack["feats"], pack
            z = (model.encode(pack["feats"], pack) if getattr(model, "needs_pack", False)
                 else model.encode(pack["feats"]))           # node (or bottleneck) tokens
            rows.append(z.reshape(-1, z.shape[-1]).cpu())
    z_all = torch.cat(rows, 0)
    if z_all.shape[0] > a.max_rows:
        idx = torch.randperm(z_all.shape[0], generator=torch.Generator().manual_seed(0))[:a.max_rows]
        z_all = z_all[idx]
    x = z_all.numpy()

    res = {"n_rows": int(x.shape[0]), "latent_dim": int(x.shape[1]),
           "twonn_id": twonn(x), **pca_dims(x),
           "sigreg_monitor_raw": sigreg_monitor(z_all, n_proj=256),
           "encoder_ends_with_layernorm": last_module_is_layernorm(model, first_feats, first_pack),
           "last_executed_module": last_executed_module(model, first_feats, first_pack),
           "state": a.state, "smoke": a.smoke}
    res["b1_reading"] = (
        "intrinsic dimension << latent dim: shape the latent on a projector "
        "head sized near the intrinsic dimension (LeWM Two-Room caveat)"
        if res["twonn_id"] < 0.25 * res["latent_dim"] else
        "intrinsic dimension comparable to latent dim: full-width SIGReg admissible")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
