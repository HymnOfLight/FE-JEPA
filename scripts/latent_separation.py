#!/usr/bin/env python3
"""Cross-geometry latent separation (wp8-lejepa, E1's adjudication instrument).

E1 asks whether latent shaping improves separation ACROSS GEOMETRIES -- the
gap Proposition 1 leaves open. The statistic is fixed here, identically for
every arm:

  1. pooled latent per instance: mean of the encoder output over load cases
     and nodes (or bottleneck tokens) -- (d,) per instance;
  2. geometry bins: the first principal component of the per-instance
     6-D `geometry_descriptor(arch.meta)` across the evaluation set, cut at
     its quartiles into 4 bins;
  3. S = mean silhouette score of the bins in latent space (Euclidean).

Secondary readings (reported, not adjudicated): a leave-one-out 1-NN bin
accuracy, and the SIGReg monitor of the pooled and of the token latents.

Usage:
    python scripts/latent_separation.py --config <cfg.json> --state <state.pt> \
        --data <instance dir> --n-instances 256 --out runs/wp8/sep_<arm>.json
Anywhere: --smoke.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def quartile_bins(scalar: np.ndarray) -> np.ndarray:
    q = np.quantile(scalar, [0.25, 0.5, 0.75])
    return np.searchsorted(q, scalar, side="right")            # 0..3


def silhouette(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over samples whose bin has >= 2 members."""
    n = x.shape[0]
    sq = (x * x).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * x @ x.T, 0.0))
    vals = []
    for i in range(n):
        own = labels == labels[i]
        if own.sum() < 2:
            continue
        a = d[i, own].sum() / (own.sum() - 1)
        b = min(d[i, labels == l].mean() for l in np.unique(labels) if l != labels[i])
        vals.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(vals)) if vals else float("nan")


def loo_1nn_accuracy(x: np.ndarray, labels: np.ndarray) -> float:
    sq = (x * x).sum(1)
    d = sq[:, None] + sq[None, :] - 2.0 * x @ x.T
    np.fill_diagonal(d, np.inf)
    return float((labels[np.argmin(d, axis=1)] == labels).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2_v1.json")
    ap.add_argument("--state", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--n-instances", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="runs/wp8/latent_separation.json")
    a = ap.parse_args()

    import torch

    from fejepa.data.archive import load_instance
    from fejepa.experiments.parallel import _build_model
    from fejepa.experiments.protocol import load_split
    from fejepa.models.features import geometry_descriptor
    from fejepa.train.sigreg import sigreg_monitor

    if a.smoke:
        import tempfile

        from fejepa.fe.synthetic import generate_synthetic_dataset

        ddir = generate_synthetic_dataset(Path(tempfile.mkdtemp()) / "sep", n=16, seed=5)
        mcfg = {"dim": 16, "depth": 1, "heads": 2,
                "features": {"load_summary": True, "geometry": True}}
        files = load_split(str(ddir), 0, 1).pool_files
        model = _build_model({"kind": "fejepa", "model": mcfg, "seed": 0})
    else:
        cfg = json.loads(Path(a.config).read_text())
        mcfg = cfg["model"]
        files = load_split(a.data, 0, 1).pool_files[:a.n_instances]
        model = _build_model({"kind": mcfg.get("kind", "fejepa"), "model": mcfg, "seed": 0})
        sd = torch.load(a.state, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
    model.eval()
    needs_pack = bool(getattr(model, "needs_pack", False))

    pooled, descs, tokens = [], [], []
    with torch.no_grad():
        for f in files:
            arch = load_instance(f)
            pack = model.prepare_instance(arch, "cpu")
            z = model.encode(pack["feats"], pack) if needs_pack else model.encode(pack["feats"])
            z2 = z.reshape(-1, z.shape[-1])
            pooled.append(z2.mean(0).cpu().numpy())
            tokens.append(z2.cpu())
            descs.append(np.asarray(geometry_descriptor(arch.meta), dtype=np.float64))
    X = np.stack(pooled)
    G = np.stack(descs)
    Gc = G - G.mean(0)
    pc1 = Gc @ np.linalg.svd(Gc, full_matrices=False)[2][0]
    bins = quartile_bins(pc1)
    tok = torch.cat(tokens, 0)
    res = {"n_instances": int(X.shape[0]), "latent_dim": int(X.shape[1]),
           "bins": np.bincount(bins, minlength=4).tolist(),
           "S_silhouette": silhouette(X, bins),
           "loo_1nn_bin_accuracy": loo_1nn_accuracy(X, bins),
           "sigreg_monitor_pooled": sigreg_monitor(torch.as_tensor(X, dtype=torch.float32),
                                                   n_proj=256),
           "sigreg_monitor_tokens": sigreg_monitor(tok[:20000], n_proj=256),
           "state": a.state, "kind": mcfg.get("kind", "fejepa"), "smoke": a.smoke}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
