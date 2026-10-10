"""Exploratory network probe for stage C1.3 (not pre-registered, not a gate).

Question: when a network is trained without solutions, on the total potential energy alone, over a family of
two-dimensional welded T-stubs with unilateral contact (tstub2d.py), does the way the contact unknowns are kept
non-negative make the training stall? A direct optimisation of single fields (probes/param_probe.py) found ReLU
stuck, softplus good only with a small learning rate, the square good, and no constraint penetrating. This probe
asks the same of a network that serves the whole family.

The network is a small transformer over the nodes (permutation equivariant; the node features carry the
coordinates, the boundary labels and the geometry). Its two outputs per node are scaled by a label-free
displacement scale (tstub2d.u_scale). The vertical unknown of each node on the contact face is passed through
phi, the variant under test:
    identity   no constraint (penetration possible)
    relu       max(z, 0)
    softplus3  T log(1 + exp(z / T)), T = 1e-3
    softplus2  the same with T = 1e-2
    square     z^2
Label-free loss: the potential energy of each geometry over the label-free energy scale (F / 2) u_scale / 4.
Control: the square variant trained on the exact solutions of the same training geometries (mean squared error).
Usage: python -I net_probe.py VARIANT SEED [STEPS] [LR] [TAG]   (writes results_VARIANT_sSEED[_TAG].json here)
"""
import sys, json, time, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tstub2d as T
import jax
import jax.numpy as jnp
import optax

N_TRAIN, N_EVAL, SEED_TRAIN, SEED_EVAL = 512, 128, 20261010, 20261011
D_MODEL, HEADS, LAYERS, D_MLP = 64, 4, 3, 128
BATCH, LR, WARMUP = 32, 1e-3, 200


def dataset(gs, with_solution):
    nodes0, els, sets = T.mesh(gs[0])
    N = nodes0.shape[0]
    X, KE, F, KB, US, SOL, PI = [], [], [], [], [], [], []
    for g in gs:
        nodes, _, _ = T.mesh(g)
        x_b = g.t_w / 2 + g.m; x_t = x_b + g.n
        flags = np.zeros((N, 4))
        flags[sets["symmetry"], 0] = 1; flags[sets["contact"], 1] = 1; flags[sets["bolt"], 2] = 1
        flags[sets["load"], 3] = 1
        desc = np.array([g.t_f / 20, g.m / 45, g.n / 40, np.log10(g.k_b) - 4.0])
        feat = np.c_[nodes / 100.0, (nodes[:, :1] - x_b) / 50.0, (nodes[:, :1] - x_t) / 50.0, flags,
                     np.tile(desc, (N, 1))]
        X.append(feat); KE.append(T.element_matrices(g, nodes, els)); F.append(T.load_vector(g, nodes, sets))
        KB.append(g.k_b); US.append(T.u_scale(g))
        if with_solution:
            o = T.solve(g); SOL.append(o["u"]); PI.append(o["Pi"])
    d = {"X": np.array(X), "Ke": np.array(KE), "f": np.array(F), "kb": np.array(KB), "us": np.array(US),
         "els": els, "sets": sets, "N": N}
    if with_solution:
        d["u"] = np.array(SOL); d["Pi"] = np.array(PI)
    return d


def init_params(key, d_in):
    def dense(k, a, b):
        return {"w": jax.random.normal(k, (a, b)) * np.sqrt(1.0 / a), "b": jnp.zeros(b)}
    ks = jax.random.split(key, 2 + 4 * LAYERS)
    p = {"emb": dense(ks[0], d_in, D_MODEL), "out": dense(ks[1], D_MODEL, 2), "layers": []}
    p["out"]["w"] = p["out"]["w"] * 0.1
    for i in range(LAYERS):
        k = ks[2 + 4 * i: 6 + 4 * i]
        p["layers"].append({"ln1": {"g": jnp.ones(D_MODEL), "b": jnp.zeros(D_MODEL)},
                            "qkv": dense(k[0], D_MODEL, 3 * D_MODEL), "o": dense(k[1], D_MODEL, D_MODEL),
                            "ln2": {"g": jnp.ones(D_MODEL), "b": jnp.zeros(D_MODEL)},
                            "m1": dense(k[2], D_MODEL, D_MLP), "m2": dense(k[3], D_MLP, D_MODEL)})
    p["ln_f"] = {"g": jnp.ones(D_MODEL), "b": jnp.zeros(D_MODEL)}
    return p


def layer_norm(x, p):
    mu = x.mean(-1, keepdims=True); var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-6) * p["g"] + p["b"]


def network(p, X):
    h = X @ p["emb"]["w"] + p["emb"]["b"]                                   # B x N x d
    B, N, _ = h.shape
    dh = D_MODEL // HEADS
    for L in p["layers"]:
        a = layer_norm(h, L["ln1"])
        qkv = a @ L["qkv"]["w"] + L["qkv"]["b"]
        q, k, v = jnp.split(qkv.reshape(B, N, 3, HEADS, dh), 3, axis=2)
        q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]                          # B x N x H x dh
        att = jax.nn.softmax(jnp.einsum("bnhd,bmhd->bhnm", q, k) / np.sqrt(dh), axis=-1)
        o = jnp.einsum("bhnm,bmhd->bnhd", att, v).reshape(B, N, D_MODEL)
        h = h + o @ L["o"]["w"] + L["o"]["b"]
        a = layer_norm(h, L["ln2"])
        h = h + jax.nn.gelu(a @ L["m1"]["w"] + L["m1"]["b"]) @ L["m2"]["w"] + L["m2"]["b"]
    return layer_norm(h, p["ln_f"]) @ p["out"]["w"] + p["out"]["b"]         # B x N x 2


PHI = {"identity": lambda z: z, "relu": jax.nn.relu,
       "softplus3": lambda z: 1e-3 * jax.nn.softplus(z / 1e-3),
       "softplus2": lambda z: 1e-2 * jax.nn.softplus(z / 1e-2), "square": lambda z: z ** 2}


def make_fns(data, variant):
    phi = PHI[variant]
    N = data["N"]; sets = data["sets"]
    sym = np.zeros(N); sym[sets["symmetry"]] = 1
    con = np.zeros(N); con[sets["contact"]] = 1
    mask_x = jnp.asarray(1.0 - sym); con = jnp.asarray(con)
    dofs = np.stack([2 * data["els"][:, k] + c for k in range(4) for c in (0, 1)], axis=1)   # E x 8
    dofs = jnp.asarray(dofs); bolt = 2 * sets["bolt"] + 1

    def displacement(p, X, us):
        z = network(p, X)
        ux = z[..., 0] * mask_x
        uy = jnp.where(con > 0, phi(z[..., 1]), z[..., 1])
        return jnp.stack([ux, uy], -1).reshape(X.shape[0], 2 * N) * us[:, None]

    def energy(u, Ke, f, kb):
        ue = u[:, dofs]                                                     # B x E x 8
        return (0.5 * jnp.einsum("bei,beij,bej->b", ue, Ke, ue) + 0.5 * kb * u[:, bolt] ** 2
                - jnp.einsum("bi,bi->b", f, u))

    def loss_energy(p, X, Ke, f, kb, us):
        u = displacement(p, X, us)
        return jnp.mean(energy(u, Ke, f, kb) / (0.5 * 0.5 * f.sum(-1) * us))   # f.sum = F / 2

    def loss_sup(p, X, us, u_true):
        u = displacement(p, X, us)
        return jnp.mean(((u - u_true) / us[:, None]) ** 2)

    return displacement, energy, loss_energy, loss_sup


def evaluate(data, u_pred):
    """Metrics against the exact solutions (numpy, double precision)."""
    sets = data["sets"]; c = 2 * sets["contact"] + 1; ctr = 2 * sets["centre"] + 1
    out = {k: [] for k in ("gap", "gap_rescaled", "err_K_norm", "stiffness_error", "iou", "false_contact",
                           "missed_contact", "min_uy_over_scale")}
    for i in range(u_pred.shape[0]):
        nodes = None
        g_Ke, f, kb, us = data["Ke"][i], data["f"][i], data["kb"][i], data["us"][i]
        n = f.size
        dofs = np.stack([2 * data["els"][:, k] + cc for k in range(4) for cc in (0, 1)], axis=1)
        r = np.repeat(dofs, 8, axis=1).ravel(); cc_ = np.tile(dofs, (1, 8)).ravel()
        import scipy.sparse as sp
        K = sp.csr_matrix((g_Ke.reshape(-1), (r, cc_)), shape=(n, n)) + sp.csr_matrix(
            ([kb], ([2 * sets["bolt"] + 1], [2 * sets["bolt"] + 1])), shape=(n, n))
        u, us_, Pi = u_pred[i].astype(float), data["u"][i], data["Pi"][i]
        e = u - us_
        out["gap"].append((T.energy(K, f, u) - Pi) / abs(Pi))
        a = (f @ u) / (u @ (K @ u)) if (f @ u) > 0 else 0.0
        out["gap_rescaled"].append((T.energy(K, f, a * u) - Pi) / abs(Pi))
        out["err_K_norm"].append(np.sqrt((e @ (K @ e)) / (us_ @ (K @ us_))))
        out["stiffness_error"].append(us_[ctr] / u[ctr] - 1 if u[ctr] > 0 else float("inf"))
        tol = 1e-3 * us
        # note: the exact set uses a tighter tolerance than the predicted one, so the exact solution itself scores
        # an exact contact set in 123 of the 128 evaluation geometries (independent check, 10 October 2026)
        pc = u[c] <= tol; tc = us_[c] <= 1e-10 * np.abs(us_).max()
        union = (pc | tc).sum()
        out["iou"].append(1.0 if union == 0 else float((pc & tc).sum() / union))
        out["false_contact"].append(float((pc & ~tc).sum() / max((~tc).sum(), 1)))
        out["missed_contact"].append(float((~pc & tc).sum() / max(tc.sum(), 1)))
        out["min_uy_over_scale"].append(float(u[c].min() / us))
    s = {}
    for k, v in out.items():
        v = np.array(v, float)
        s[k] = {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "min": float(v.min()),
                "max": float(v.max()), "mean": float(v.mean())}
    s["iou_all_exact"] = float(np.mean(np.array(out["iou"]) == 1.0))
    s["share_penetrating"] = float(np.mean(np.array(out["min_uy_over_scale"]) < -1e-3))
    return s


def main():
    variant, seed = sys.argv[1], int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else LR
    tag = ("_" + sys.argv[5]) if len(sys.argv) > 5 else ""
    supervised = variant == "supervised"
    phi_name = "square" if supervised else variant
    t0 = time.time()
    cache = HERE / "data_cache.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        tr, ev = z["tr"].item(), z["ev"].item()
    else:
        tr = dataset(T.sample(N_TRAIN, SEED_TRAIN), with_solution=True)
        ev = dataset(T.sample(N_EVAL, SEED_EVAL), with_solution=True)
        np.savez(cache, tr=np.array(tr, dtype=object), ev=np.array(ev, dtype=object))
    t_data = time.time() - t0
    displacement, energy, loss_energy, loss_sup = make_fns(tr, phi_name)
    key = jax.random.PRNGKey(seed)
    p = init_params(key, tr["X"].shape[-1])
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, WARMUP, steps, lr * 1e-2)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    st = opt.init(p)
    j = {k: jnp.asarray(tr[k], jnp.float32) for k in ("X", "Ke", "f", "kb", "us")}
    j_u = jnp.asarray(tr["u"], jnp.float32)

    @jax.jit
    def step(p, st, idx):
        if supervised:
            l, g = jax.value_and_grad(loss_sup)(p, j["X"][idx], j["us"][idx], j_u[idx])
        else:
            l, g = jax.value_and_grad(loss_energy)(p, j["X"][idx], j["Ke"][idx], j["f"][idx], j["kb"][idx],
                                                   j["us"][idx])
        upd, st = opt.update(g, st, p)
        return optax.apply_updates(p, upd), st, l

    pred = jax.jit(lambda p, X, us: displacement(p, X, us))
    rng = np.random.default_rng(seed)
    hist = []
    t1 = time.time()
    for it in range(1, steps + 1):
        idx = jnp.asarray(rng.choice(N_TRAIN, BATCH, replace=False))
        p, st, l = step(p, st, idx)
        if it % 200 == 0 or it == 1:
            hist.append((it, float(l)))
        if it in (steps // 4, steps // 2, steps):
            u_ev = np.asarray(pred(p, jnp.asarray(ev["X"], jnp.float32), jnp.asarray(ev["us"], jnp.float32)))
            m = evaluate(ev, u_ev)
            hist.append((it, "eval", m["gap"]["median"], m["iou_all_exact"], m["false_contact"]["mean"]))
            print(f"{variant} s{seed} step {it}: loss {float(l):.5f}  gap median {m['gap']['median']:.2e}  "
                  f"stiffness err median {m['stiffness_error']['median']:+.3%}  IoU=1 share {m['iou_all_exact']:.2f}  "
                  f"false contact {m['false_contact']['mean']:.3f}  {time.time()-t1:.0f}s", flush=True)
    u_tr = np.asarray(pred(p, j["X"], j["us"]))
    res = {"variant": variant, "seed": seed, "steps": steps, "batch": BATCH, "lr": lr, "tag": tag,
           "arch": {"d_model": D_MODEL, "heads": HEADS, "layers": LAYERS, "d_mlp": D_MLP},
           "n_train": N_TRAIN, "n_eval": N_EVAL, "nodes": int(tr["N"]), "seconds_data": round(t_data, 1),
           "seconds_train": round(time.time() - t1, 1), "history": hist,
           "eval": evaluate(ev, np.asarray(pred(p, jnp.asarray(ev["X"], jnp.float32),
                                               jnp.asarray(ev["us"], jnp.float32)))),
           "train": evaluate(tr, u_tr)}
    json.dump(res, open(HERE / f"results_{variant}_s{seed}{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
