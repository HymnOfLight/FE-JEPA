#!/usr/bin/env python3
"""cmame-paper Stage 3: a functional check of the stiffness-norm loss, run
before PREREG_CM2D's stamp and disclosed in its Sec. 4. It is not evidence for
any of PREREG_CM2D's hypotheses: another corpus, a far smaller network and
training set, two seeds.

Instances of the 2D training family are drawn by the gmsh generator with
their own seed (777; not E1's corpus) and labelled. A transformer of width 32
and depth 2 (load-summary and geometry channels) is trained on the first
`--n-train` of them, one instance per step, for `--epochs` epochs, per seed:
with the displacement loss L_D and the stiffness-norm loss L_K (learning rate
1.5e-3, the supervised rows' trainer and settings otherwise) and with the
label-free objective (1e-3, the label-free row's); each network starts from
the seed's initial weights. Recorded per loss and seed: the frozen metric
suite on the other `--n-val` instances (and three of its metrics on the
training instances), the gradient norms before clipping and the share of
steps on which clipping at 1 acted, and the seconds taken; and the largest
relative deviation of the L_K value the trainer computes from the mean square
root of the per-load relative energy gap of the prediction it was given
(Lemma 1), over the first 200 steps of the first seed's L_K training. CPU,
one thread.

    python scripts/cm2d_funccheck.py --out records/cmame/cm2d_funccheck.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL = {"dim": 32, "depth": 2, "heads": 4,
         "features": {"load_summary": True, "geometry": True}}
LOSSES = ("disp", "knorm", "ar")
"""L_D, L_K and the label-free objective, by the code's names."""
LR = {"disp": 1.5e-3, "knorm": 1.5e-3, "ar": 1e-3}
DATA_SEED = 777
CHECKED_STEPS = 200


def _git(*args) -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                           timeout=30)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _src_tree() -> str | None:
    """The tree of the `src` this script imports: HEAD's, when `src` has no
    local changes; None otherwise (the code that ran is then not named)."""
    return _git("rev-parse", "HEAD:src") if _git("status", "--porcelain", "--", "src") == "" \
        else None


def _summary(norms: list) -> dict:
    a = np.asarray(norms, dtype=np.float64)
    return {"steps": int(a.size),
            "p10_p50_p90": [float(x) for x in np.percentile(a, [10, 50, 90])],
            "share_clipped": float(np.mean(a > 1.0))}


def funccheck(data: Path, n_train: int, n_val: int, epochs: int, seeds: list) -> dict:
    import torch

    import fejepa.train.supervised as sup
    from fejepa.data.archive import instance_files, load_instance, manifest_sha256
    from fejepa.experiments.parallel import _build_model
    from fejepa.fe.generator import generate_dataset
    from fejepa.metrics import energy_gap_rel, evaluate_model, torch_predictor
    from fejepa.train.losses import AR_CONFIG
    from fejepa.train.pretrain import PretrainConfig, pretrain

    torch.set_num_threads(1)
    t0 = time.time()
    generate_dataset(data, n=n_train + n_val, seed=DATA_SEED, labelled="all", jobs=1)
    gen_s = time.time() - t0
    archs = [load_instance(f) for f in instance_files(data)]
    train, val = archs[:n_train], archs[n_train:n_train + n_val]

    norms, calls = [], []
    clip = torch.nn.utils.clip_grad_norm_
    knorm = sup._knorm_loss

    def clip_spy(params, max_norm, *a, **k):
        n = clip(params, max_norm, *a, **k)
        norms.append(float(n))
        return n

    def knorm_spy(anchor, u, u_star, ustar_k):
        v = knorm(anchor, u, u_star, ustar_k)
        if record_calls and len(calls) < CHECKED_STEPS:
            calls.append((u.detach().double().cpu().numpy(),
                          u_star.double().cpu().numpy(), float(v)))
        return v

    torch.nn.utils.clip_grad_norm_ = clip_spy
    sup._knorm_loss = knorm_spy
    runs = {k: {} for k in LOSSES}
    try:
        for seed in seeds:
            for kind in LOSSES:
                record_calls = kind == "knorm" and seed == seeds[0]
                norms.clear()
                model = _build_model({"kind": "fejepa", "model": MODEL, "seed": seed})
                t = time.time()
                if kind == "ar":
                    pretrain(model, train, PretrainConfig(epochs=epochs, lr=LR[kind], seed=seed,
                                                          loss=AR_CONFIG, log_every=-1))
                    ev = evaluate_model(torch_predictor(model, "cpu"), val)
                else:
                    ev = train_supervised_eval(sup, model, train, val, epochs, seed, kind)
                tr = evaluate_model(torch_predictor(model, "cpu"), train)
                runs[kind][str(seed)] = {
                    "val": {k: v for k, v in ev.items() if k != "per_instance"},
                    "train": {k: tr[k] for k in ("disp_rel_l2", "energy_gap_rel", "vm_rel_l2")},
                    "grad_norm_before_clip": _summary(norms),
                    "seconds": round(time.time() - t, 1)}
                print(f"[funccheck] {kind} seed {seed}: "
                      f"{json.dumps(runs[kind][str(seed)]['val'])}", flush=True)
    finally:
        torch.nn.utils.clip_grad_norm_ = clip
        sup._knorm_loss = knorm

    dev = []
    for U, Us, loss in calls:
        a = next(x for x in train
                 if x.U_star.shape == Us.shape and np.allclose(x.U_star, Us, rtol=1e-6,
                                                               atol=1e-12))
        want = float(np.mean(np.sqrt(energy_gap_rel(U, a))))
        dev.append(abs(loss - want) / want)
    nodes = [int(a.n_nodes) for a in archs]
    return {"corpus": {"generator": "gmsh, 2D training family", "seed": DATA_SEED,
                       "n": len(archs), "n_train": n_train, "n_val": n_val,
                       "manifest_sha256": manifest_sha256(data),
                       "nodes_median": int(np.median(nodes)), "nodes_max": max(nodes),
                       "generation_seconds": round(gen_s, 1)},
            "runs": runs,
            "knorm_loss_against_lemma_1": {"steps": len(dev),
                                           "max_rel_dev": float(max(dev)),
                                           "median_rel_dev": float(np.median(dev))}}


def train_supervised_eval(sup, model, train, val, epochs: int, seed: int, kind: str) -> dict:
    res = sup.train_supervised(model, train, val,
                               sup.SupervisedConfig(epochs=epochs, lr=LR[kind], seed=seed,
                                                    loss=kind, log_every=-1))
    return res["val"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-val", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    import scipy
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        res = funccheck(Path(tmp) / "data", a.n_train, a.n_val, a.epochs, a.seeds)
    out = {"what": "cmame-paper Stage 3: functional check of the stiffness-norm loss before "
                   "PREREG_CM2D's stamp (disclosed in its Sec. 4); not evidence for any of "
                   "its hypotheses",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "src_tree": _src_tree(),
           "settings": {"model": MODEL, "epochs": a.epochs, "seeds": a.seeds, "lr": LR,
                        "steps_per_epoch": a.n_train, "device": "cpu", "threads": 1},
           "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "torch": torch.__version__},
           **res}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"[funccheck] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
