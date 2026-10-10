"""End-to-end smoke test of the C1.3 pipeline on one joint (exploratory; not a gate).

Export one family member (seed 999, kept out of any later sample) on the 10 mm mesh, train the repository's
encoder with a copy of its decoder on that one instance by the energy alone (softplus on the contact unknowns),
and compare with the exact solution of that instance: energy gap, stiffness S_paper, contact on the column face.
A small model on a CPU: this checks that the pieces fit and the energy goes down, not what C1.3 can reach.
    python -I probes/c13_cost/smoke_train.py --repo REPO --steps 300 --out smoke.json
"""
import argparse, json, sys, time, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fejoint.family import sample, geometry, asdict          # noqa: E402
from fejoint.export import export, solve_exported, energy, stiffness   # noqa: E402
from cost_probe import features, SMALL                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--zero-init", action="store_true", help="start from the zero field (last layer zeroed)")
    ap.add_argument("--const-lr", action="store_true", help="constant learning rate instead of one cycle")
    ap.add_argument("--segment", type=float, default=None, help="beam segment length (mm); default the full beam")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(a.repo) / "src"))
    import torch
    from torch import nn
    from fejepa.models.fejepa import build_encoder
    from fejepa.anchor.energy import EnergyAnchor
    torch.manual_seed(0)
    q = sample(1, seed=999)[0][0]
    ex = export(geometry(q), h=10.0, n_tp=1, n_tf=1, beam_segment=a.segment)
    ref = solve_exported(ex)
    feats = torch.as_tensor(features(ex, q), dtype=torch.float32).unsqueeze(0)
    anchor = EnergyAnchor(ex["K"], ex["F"], ex["dirichlet"])
    nonneg = torch.as_tensor(ex["nonneg"])
    scale = float(ex["u_scale"])                                  # label-free displacement scale
    T = 1e-3
    enc = build_encoder(feats.shape[-1], SMALL["dim"], SMALL["depth"], SMALL["heads"])
    enc.use_checkpoint = True
    d = SMALL["dim"]
    dec = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))
    if a.zero_init:
        with torch.no_grad():
            dec[-1].weight.zero_(); dec[-1].bias.zero_()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    sched = (torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 1.0) if a.const_lr
             else torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1))
    e_scale = 0.5 * (0.5 * ex["meta"]["model"]["P_full_N"]) * scale   # label-free energy scale: (P/2) u_scale / 2

    def field():
        z = enc(feats)
        pooled = z.mean(dim=-2, keepdim=True).expand_as(z)
        u = dec(torch.cat([z, pooled], dim=-1)).reshape(-1)
        return torch.where(nonneg, T * nn.functional.softplus(u / T), u) * scale

    hist = []
    t0 = time.perf_counter()
    for it in range(1, a.steps + 1):
        loss = anchor(field().unsqueeze(0)) / e_scale
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if it % 25 == 0 or it == 1:
            with torch.no_grad():
                u = field().double().numpy()
            u[ex["dirichlet"]] = 0.0
            gap = (energy(ex, u) - ref["Pi"]) / abs(ref["Pi"])
            hist.append({"step": it, "loss": float(loss), "gap": gap, "S_pred_over_ref": stiffness(ex, u) / ref["S_paper"]})
            print(json.dumps(hist[-1]), flush=True)
    with torch.no_grad():
        u = field().double().numpy()
    u[ex["dirichlet"]] = 0.0
    col = 3 * np.nonzero(ex["node_flags"][:, 1])[0]
    tol = 1e-3 * scale
    pc, tc = u[col] <= tol, ref["u"][col] <= 1e-10 * np.abs(ref["u"]).max()
    out = {"params": asdict(q), "nodes": int(ex["nodes"].shape[0]), "model": SMALL, "steps": a.steps, "lr": a.lr,
           "zero_init": a.zero_init, "const_lr": a.const_lr, "beam_segment": a.segment,
           "seconds": round(time.perf_counter() - t0, 1), "history": hist,
           "final": {"gap": (energy(ex, u) - ref["Pi"]) / abs(ref["Pi"]), "S_pred": stiffness(ex, u),
                     "S_ref": ref["S_paper"], "column_contact_iou": float((pc & tc).sum() / max((pc | tc).sum(), 1)),
                     "column_contact_share_ref": float(tc.mean()), "column_contact_share_pred": float(pc.mean()),
                     "min_nonneg_over_scale": float(u[ex["nonneg"]].min() / scale)}}
    print(json.dumps(out["final"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
