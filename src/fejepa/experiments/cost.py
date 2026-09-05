"""Compute cost model + bench (plan Sec.9: "wall-clock projected from the *measured*
cost model (`fejepa bench`)").

:func:`count_steps` mirrors the runner's arithmetic exactly (tested against a hand
count); :func:`bench` measures ms/step for one supervised and one pretrain step on a
synthetic instance (torch required) so projections are measured, not guessed.
"""

from __future__ import annotations

import time


def _e(cfg: dict, name: str) -> dict | None:
    e = (cfg.get("experiments") or {}).get(name) or {}
    return e if e.get("enabled") else None


def count_steps(cfg: dict) -> dict:
    """Optimizer steps per experiment (1 step per instance visit, batch-1)."""
    out = {}
    sup_ep = int(cfg.get("sup", {}).get("epochs", 200))
    pre_ep = int(cfg.get("pretrain", {}).get("epochs", 100))

    e1 = _e(cfg, "e1")
    if e1:
        arms = 3 + len(e1.get("grid", []))
        out["e1"] = sum(arms * int(e1.get("seeds", 3)) * int(e1.get("epochs", sup_ep)) * b
                        for b in e1.get("budgets", [16, 64, 256, 1024]))
    e2 = _e(cfg, "e2")
    if e2:
        s = int(e2.get("seeds", 3))
        pre = 2 * int(e2.get("pool_size", 1024)) * int(e2.get("pre_epochs", pre_ep))
        ft = sum(3 * int(e2.get("ft_epochs", sup_ep)) * b
                 for b in e2.get("budgets", [16, 64, 256]))
        out["e2"] = s * (pre + ft)
    e3 = _e(cfg, "e3")
    if e3:
        conds = len(e3.get("geometry_conditions", [False, True]))
        runs = 1 + len(e3.get("modes", ["sigreg", "sigreg_pooled", "vicreg_pooled"]))
        out["e3"] = conds * runs * int(e3.get("steps", 2000))
    e4 = _e(cfg, "e4")
    if e4:
        out["e4"] = len(e4.get("coarsens", [1.8, 2.5])) * 2 \
            * int(e4.get("epochs", pre_ep)) * int(e4.get("n_train", 512))
    e5 = _e(cfg, "e5")
    if e5:
        # fallback trains seed-0 balanced per budget; its epochs default is the
        # literal 200 in e5_sanity._train_anchored, mirrored here exactly
        out["e5"] = 0 if _e(cfg, "e1") else sum(
            int(e5.get("epochs", 200)) * b
            for b in e5.get("budgets", [16, 64, 256, 1024]))
    e6 = _e(cfg, "e6")
    if e6:
        out["e6"] = int(e6.get("pool_size", 256)) * int(e6.get("pre_epochs", 50))
    e7 = _e(cfg, "e7")
    if e7:
        out["e7"] = int(e7.get("pool_size", 256)) * int(e7.get("pre_epochs", 50))
    wp2 = _e(cfg, "wp2")
    if wp2:
        out["wp2"] = len(wp2.get("ratios", [0.2, 0.4, 0.6])) \
            * int(wp2.get("steps", 600))
    e8 = _e(cfg, "e8")
    if e8:
        s = int(e8.get("seeds", 3))
        ar = sum(p * int(e8.get("ar_epochs", pre_ep))
                 for p in e8.get("pool_sizes", [1024]))
        if e8.get("ar_only"):                              # wp8 E-series: no supervised grid
            supd = 0
        else:
            n_sup = 3 + (1 if e8.get("include_mgn") else 0)   # labels, anchored, ar_ft(+mgn)
            supd = sum(n_sup * int(e8.get("sup_epochs", sup_ep)) * b
                       for b in e8.get("budgets", [16, 64, 256, 1024]))
        out["e8"] = s * (ar + supd)
    out["total"] = sum(out.values())
    return out


def bench(model_cfg: dict | None = None, n_steps: int = 10, device: str = "cpu",
          tf32: bool = True) -> dict:
    """Measured ms/step on a synthetic instance (needs the torch extra).

    Applies the same runtime policy (TF32 flag) as `run-config`, and -- when a config
    is given -- the config's own model block, so projections are measured on the
    numerics path and model size the real run will use."""
    import numpy as np
    import torch

    from ..runtime import setup_torch

    policy = setup_torch(device, tf32=tf32)

    from ..anchor.energy import AnchorCache
    from ..fe.synthetic import synthetic_instance
    from ..models.fejepa import FEJEPAConfig, build_fejepa, mesh_adjacency
    from ..models.regularizers import PooledBuffer
    from ..train.losses import JEPA_CONFIG, compute_loss
    from ..train.supervised import _disp_loss

    rng = np.random.default_rng(0)
    arch = synthetic_instance(rng, nx=16, ny=12, labelled=True)
    model = build_fejepa(FEJEPAConfig.from_dict(model_cfg or {})).to(device)
    pack = model.prepare_instance(arch, device)
    pack["u_star"] = torch.as_tensor(arch.U_star, dtype=pack["free"].dtype, device=device)
    anchors = AnchorCache(device=device)
    adj = mesh_adjacency(arch.elements, arch.n_nodes)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def timed(fn):
        fn()                                              # warmup
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            fn()
        if device != "cpu":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_steps * 1e3

    def sup_step():
        u = model.forward_instance(pack)
        loss = _disp_loss(u, pack["u_star"], pack["free"]) \
            + anchors.get(arch).energies(u).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    buffer = PooledBuffer()

    def pre_step():
        loss, _ = compute_loss(model, pack, anchors.get(arch), adj, buffer,
                               rng, JEPA_CONFIG)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    return {"device": device, "tf32": policy["tf32"], "n_nodes": arch.n_nodes,
            "ms_per_supervised_step": round(timed(sup_step), 3),
            "ms_per_jepa_pretrain_step": round(timed(pre_step), 3)}
