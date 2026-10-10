"""Cost probe for stage C1.3: the time and memory of one label-free training step of the project's transformer on
joint instances of the C1.3 family (exploratory; not a gate).

One step is what the repository's runner does per instance visit: encode the node features, decode the field,
keep the contact unknowns non-negative, evaluate the energy with the repository's EnergyAnchor, back-propagate,
and take an Adam step. The encoder is the repository's build_encoder (with its activation checkpointing); the
decoder copies the repository's FieldDecoder. The instances come from fejoint.export (the condensed nodal problem
of the idealised joint model), on the two meshes considered for C1.3.

On the GPU host, from the root of this folder (the joint code) with the repository checked out at REPO:
    python -I probes/c13_cost/cost_probe.py --repo REPO --device cuda --out c13_cost_gpu.json
Functional check on a CPU:
    python -I probes/c13_cost/cost_probe.py --repo REPO --device cpu --small --out c13_cost_cpu_small.json
"""
import argparse, json, math, platform, sys, time, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fejoint.family import sample, geometry, RANGES        # noqa: E402
from fejoint.export import export                          # noqa: E402

MESHES = {"h10_one_layer": dict(h=10.0, n_tp=1, n_tf=1), "h7.5_two_layers": dict(h=7.5, n_tp=2, n_tf=2)}
FULL = dict(dim=256, depth=8, heads=8)                     # Phase-2b's model (configs/phase2b_v1.json)
SMALL = dict(dim=64, depth=2, heads=4)


def features(ex, q):
    """Per-node features: centred and RMS-normalised coordinates (3), Dirichlet flags (3), the load over its
    largest component (3), the node flags of the export (6), the family parameters over their range maxima (8)."""
    nodes = ex["nodes"]
    c = nodes - nodes.mean(axis=0, keepdims=True)
    c = c / (np.sqrt((c ** 2).sum(axis=1).mean()) + 1e-12)
    n = nodes.shape[0]
    dmask = ex["dirichlet"].reshape(n, 3).astype(float)
    f = ex["F"].reshape(n, 3) / (np.abs(ex["F"]).max() + 1e-12)
    flags = ex["node_flags"].astype(float)
    desc = np.array([getattr(q, k) / RANGES[k][1] for k in RANGES])
    return np.concatenate([c, dmask, f, flags, np.broadcast_to(desc, (n, desc.size))], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="checkout of the FE-JEPA repository (wp9-pool or later)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--small", action="store_true", help="a small model, for a functional check on a CPU")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(a.repo) / "src"))
    import torch
    from torch import nn
    from fejepa.models.fejepa import build_encoder
    from fejepa.anchor.energy import EnergyAnchor
    cfg = SMALL if a.small else FULL
    dev = torch.device(a.device)
    torch.manual_seed(0)
    params, _ = sample(a.instances, seed=20261010)
    out = {"model": cfg, "device": str(dev), "torch": torch.__version__, "python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "threads": torch.get_num_threads(), "steps_timed": a.steps, "warmup": a.warmup, "meshes": {}}
    for mname, mp in MESHES.items():
        insts = []
        for q in params:
            ex = export(geometry(q), **mp)
            feats = torch.as_tensor(features(ex, q), dtype=torch.float32, device=dev).unsqueeze(0)
            anchor = EnergyAnchor(ex["K"], ex["F"], ex["dirichlet"], device=dev)
            nonneg = torch.as_tensor(ex["nonneg"], device=dev)
            fscale = float(np.abs(ex["F"]).max())
            insts.append({"feats": feats, "anchor": anchor, "nonneg": nonneg, "fscale": fscale,
                          "N": int(ex["nodes"].shape[0]), "nnz": int(ex["K"].nnz)})
        in_dim = insts[0]["feats"].shape[-1]
        enc = build_encoder(in_dim, cfg["dim"], cfg["depth"], cfg["heads"]).to(dev)
        enc.use_checkpoint = True
        dec = nn.Sequential(nn.Linear(2 * cfg["dim"], 2 * cfg["dim"]), nn.GELU(),
                            nn.Linear(2 * cfg["dim"], cfg["dim"]), nn.GELU(), nn.Linear(cfg["dim"], 3)).to(dev)
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
        T = 1e-3                                             # softplus temperature, in the decoded field's units

        def step(inst):
            z = enc(inst["feats"])                           # (1, N, dim)
            pooled = z.mean(dim=-2, keepdim=True).expand_as(z)
            u = dec(torch.cat([z, pooled], dim=-1)).reshape(-1)
            u = torch.where(inst["nonneg"], T * nn.functional.softplus(u / T), u) * inst["fscale"]
            loss = inst["anchor"](u.unsqueeze(0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return float(loss.detach())

        def sync():
            if dev.type == "cuda":
                torch.cuda.synchronize()

        enc.train(); dec.train()
        for i in range(a.warmup):
            step(insts[i % len(insts)])
        sync()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(a.steps):
            t0 = time.perf_counter()
            loss = step(insts[i % len(insts)])
            sync()
            times.append(time.perf_counter() - t0)
            if not math.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {i}")
        out["meshes"][mname] = {
            "nodes": [x["N"] for x in insts], "nnz": [x["nnz"] for x in insts], "in_dim": int(in_dim),
            "seconds_per_step_mean": float(np.mean(times)), "seconds_per_step_median": float(np.median(times)),
            "seconds_per_step_max": float(np.max(times)),
            "peak_memory_GiB": (torch.cuda.max_memory_allocated() / 2 ** 30) if dev.type == "cuda" else None}
        print(mname, json.dumps(out["meshes"][mname]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
