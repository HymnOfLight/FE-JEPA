"""CPU pilot of stage C1.3 over a small family (exploratory; before the pre-specification; not a gate).

Pilot seeds (training 101, held-out 102) are kept out of any later sample. Family members on the 10 mm mesh with
the 300 mm beam segment; the repository's encoder (small configuration for the CPU) with a copy of its decoder;
softplus on the contact unknowns. Arms:
    energy       label-free: each instance's energy over its label-free energy scale
    supervised   the exact solutions of the training members, mean squared error in units of the scale
Held-out metrics: energy gap, stiffness S_paper against the exact one, contact overlap on the column face (the
same tolerance for prediction and reference, 1e-3 of the displacement scale).
    python -I probes/c13_cost/pilot_family.py --repo REPO --arm energy --out pilot_energy.json
"""
import argparse, json, sys, time, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fejoint.family import sample, geometry, asdict           # noqa: E402
from fejoint.export import export, solve_exported, energy, stiffness   # noqa: E402
from cost_probe import features                              # noqa: E402

TOL = 1e-3


def contact_sets(ex, u):
    col = 3 * np.nonzero(ex["node_flags"][:, 1])[0]
    return u[col] <= TOL * ex["u_scale"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--arm", choices=["energy", "supervised"], required=True)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(pathlib.Path(a.repo) / "src"))
    import torch
    from torch import nn
    from fejepa.models.fejepa import build_encoder
    from fejepa.anchor.energy import EnergyAnchor
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()

    def make(qs, label):
        out = []
        for q in qs:
            ex = export(geometry(q), h=10.0, n_tp=1, n_tf=1, beam_segment=300.0)
            item = {"q": q, "ex": ex, "feats": torch.as_tensor(features(ex, q), dtype=torch.float32).unsqueeze(0),
                    "anchor": EnergyAnchor(ex["K"], ex["F"], ex["dirichlet"]),
                    "nonneg": torch.as_tensor(ex["nonneg"]), "scale": float(ex["u_scale"]),
                    "free": torch.as_tensor(~ex["dirichlet"]).float()}
            item["e_scale"] = 0.5 * (0.5 * ex["meta"]["model"]["P_full_N"]) * item["scale"]
            if label:
                ref = solve_exported(ex)
                item["ref"] = ref
                item["u_ref"] = torch.as_tensor(ref["u"], dtype=torch.float32)
            out.append(item)
        return out

    train_q, _ = sample(a.n_train, seed=101)
    eval_q, _ = sample(a.n_eval, seed=102)
    train = make(train_q, label=(a.arm == "supervised"))
    held = make(eval_q, label=True)
    t_data = time.perf_counter() - t0
    in_dim = train[0]["feats"].shape[-1]
    enc = build_encoder(in_dim, a.dim, a.depth, a.heads)
    enc.use_checkpoint = True
    d = a.dim
    dec = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))
    with torch.no_grad():
        dec[-1].weight.zero_(); dec[-1].bias.zero_()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
    T = 1e-3

    def field(it):
        z = enc(it["feats"])
        pooled = z.mean(dim=-2, keepdim=True).expand_as(z)
        u = dec(torch.cat([z, pooled], dim=-1)).reshape(-1)
        return torch.where(it["nonneg"], T * nn.functional.softplus(u / T), u) * it["scale"] * it["free"]

    def evaluate(items):
        enc.eval(); dec.eval()
        rows = []
        with torch.no_grad():
            for it in items:
                u = field(it).double().numpy()
                ex, ref = it["ex"], it["ref"]
                gap = (energy(ex, u) - ref["Pi"]) / abs(ref["Pi"])
                s_err = stiffness(ex, u) / ref["S_paper"] - 1
                pc, tc = contact_sets(ex, u), contact_sets(ex, ref["u"])
                iou = float((pc & tc).sum() / max((pc | tc).sum(), 1))
                e = u - ref["u"]
                ek = float(np.sqrt(max(e @ (ex["K"] @ e), 0.0) / max(ref["u"] @ (ex["K"] @ ref["u"]), 1e-300)))
                rows.append({"gap": gap, "S_err": s_err, "iou": iou, "err_K": ek,
                             "min_nonneg_over_scale": float(u[ex["nonneg"]].min() / ex["u_scale"])})
        enc.train(); dec.train()
        agg = {k: {"median": float(np.median([r[k] for r in rows])), "max": float(np.max([r[k] for r in rows])),
                   "min": float(np.min([r[k] for r in rows]))} for k in rows[0]}
        agg["abs_S_err_median"] = float(np.median([abs(r["S_err"]) for r in rows]))
        return agg, rows

    rng = np.random.default_rng(a.seed)
    hist = []
    t1 = time.perf_counter()
    for step in range(1, a.steps + 1):
        it = train[int(rng.integers(len(train)))]
        u = field(it)
        if a.arm == "energy":
            loss = it["anchor"](u.unsqueeze(0)) / it["e_scale"]
        else:
            loss = (((u - it["u_ref"]) / it["scale"]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % (a.steps // 4) == 0:
            agg, _ = evaluate(held)
            hist.append({"step": step, "seconds": round(time.perf_counter() - t1, 1), **{k: v for k, v in agg.items()}})
            print(json.dumps({"step": step, "gap_med": agg["gap"]["median"], "abs_S_err_med": agg["abs_S_err_median"],
                              "iou_med": agg["iou"]["median"], "s": round(time.perf_counter() - t1)}), flush=True)
    agg, rows = evaluate(held)
    agg_tr, _ = evaluate([it for it in train[:16] if "ref" in it]) if a.arm == "supervised" else (None, None)
    out = {"arm": a.arm, "args": vars(a), "nodes": [int(it["ex"]["nodes"].shape[0]) for it in held],
           "seconds_data": round(t_data, 1), "seconds_train": round(time.perf_counter() - t1, 1), "history": hist,
           "held_out": agg, "held_out_rows": rows, "train_subset": agg_tr,
           "eval_params": [asdict(q) for q in eval_q]}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
