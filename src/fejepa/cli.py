"""Command-line interface for FE-JEPA Phase-0 workflows.

Subcommands::

    fejepa generate   # build a dataset of FE instance archives
    fejepa gate-g0    # run the Gate G0 neural-solver sanity check
    fejepa pretrain   # label-free FE-JEPA pretraining over a dataset
    fejepa info       # summarise a dataset directory
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _cmd_generate(args: argparse.Namespace) -> int:
    from fejepa.fe.generator import GeneratorConfig, generate_dataset

    cfg = GeneratorConfig(max_holes=args.max_holes)
    generate_dataset(
        args.out,
        n_instances=args.n,
        seed=args.seed,
        labelled=not args.unlabelled,
        cfg=cfg,
    )
    print(f"Wrote {args.n} instances to {args.out}")
    return 0


def _cmd_gate_g0(args: argparse.Namespace) -> int:
    from fejepa.data.archive import load_problem, save_problem
    from fejepa.fe.generator import GeneratorConfig, sample_instance
    from fejepa.train.g0 import run_gate_g0

    if args.instance:
        arch = load_problem(args.instance)
    else:
        cfg = GeneratorConfig(mesh_size_frac=(0.16, 0.22), max_holes=args.max_holes)
        prob = sample_instance(np.random.default_rng(args.seed), cfg, labelled=True)
        tmp = Path(args.out or "g0_instance.npz")
        save_problem(tmp, prob)
        arch = load_problem(tmp)
        print(f"Sampled instance with {arch.n_nodes} nodes -> {tmp}")

    res = run_gate_g0(
        arch, steps=args.steps, lr=args.lr, log_every=args.log_every, device=args.device
    )
    print(res)
    return 0 if res.passed else 1


def _cmd_pretrain(args: argparse.Namespace) -> int:
    from fejepa.train.pretrain import PretrainConfig, pretrain

    cfg = PretrainConfig(
        epochs=args.epochs, lr=args.lr, max_instances=args.max_instances, device=args.device
    )
    result = pretrain(args.data, out_ckpt=args.ckpt, cfg=cfg)
    print(
        f"Pretraining done: {result['steps']} steps in "
        f"{result['elapsed_sec']:.1f}s"
        + (f", checkpoint -> {result['checkpoint']}" if "checkpoint" in result else "")
    )
    return 0


def _cmd_run_config(args: argparse.Namespace) -> int:
    from fejepa.experiments.runner import run_config

    run_config(args.config)
    return 0


def _cmd_regimes(args: argparse.Namespace) -> int:
    from fejepa.experiments.falsification import load_split
    from fejepa.experiments.regimes import compare_training_regimes
    from fejepa.models.fejepa import FEJEPAConfig
    from fejepa.train.pretrain import PretrainConfig
    from fejepa.train.supervised import SupervisedConfig

    model_cfg = FEJEPAConfig(dim=args.dim, depth=args.depth)
    pool_files, val_archs = load_split(args.data, args.n_val, args.seed)
    sup = SupervisedConfig(epochs=args.epochs, lr=args.lr, model=model_cfg, device=args.device)
    pre = PretrainConfig(epochs=args.epochs, lr=args.lr, model=model_cfg, device=args.device)
    compare_training_regimes(
        pool_files, val_archs, n_train=args.n_train, sup_cfg=sup, pre_cfg=pre, out_report=args.out
    )
    return 0


def _cmd_battery(args: argparse.Namespace) -> int:
    from fejepa.experiments.falsification import BatteryConfig, run_battery
    from fejepa.models.fejepa import FEJEPAConfig
    from fejepa.train.supervised import SupervisedConfig

    cfg = BatteryConfig(
        budgets=[int(b) for b in args.budgets.split(",")],
        n_val=args.n_val,
        seed=args.seed,
        decision_budget=args.decision_budget,
        lambda_phys=args.lambda_phys,
        lambda_grid=[float(x) for x in args.lambda_grid.split(",")] if args.lambda_grid else None,
        n_seeds=args.n_seeds,
        device=args.device,
        sup=SupervisedConfig(
            epochs=args.epochs, lr=args.lr, model=FEJEPAConfig(dim=args.dim, depth=args.depth)
        ),
    )
    exps = [e.strip() for e in args.experiments.split(",")] if args.experiments else None
    run_battery(args.data, cfg=cfg, experiments=exps, out_report=args.out)
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from fejepa.data.archive import read_manifest

    manifest = read_manifest(args.data)
    insts = manifest["instances"]
    nodes = [r["n_nodes"] for r in insts]
    print(f"Dataset: {args.data}")
    print(f"  instances : {len(insts)}")
    print(f"  labelled  : {manifest.get('labelled')}")
    print(f"  load names: {manifest.get('load_names')}")
    print(f"  nodes     : min={min(nodes)} max={max(nodes)} mean={np.mean(nodes):.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fejepa", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a dataset of FE instances")
    g.add_argument("--out", required=True)
    g.add_argument("-n", type=int, default=100)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--max-holes", type=int, default=3)
    g.add_argument("--unlabelled", action="store_true", help="skip the U* solve")
    g.set_defaults(func=_cmd_generate)

    g0 = sub.add_parser("gate-g0", help="run the Gate G0 sanity check")
    g0.add_argument("--instance", default=None, help="path to a labelled .npz instance")
    g0.add_argument("--out", default=None, help="where to save a sampled instance")
    g0.add_argument("--steps", type=int, default=2500)
    g0.add_argument("--lr", type=float, default=5e-3)
    g0.add_argument("--max-holes", type=int, default=0)
    g0.add_argument("--seed", type=int, default=0)
    g0.add_argument("--log-every", type=int, default=250)
    g0.add_argument("--device", default="auto", help="cpu | cuda | cuda:N | auto")
    g0.set_defaults(func=_cmd_gate_g0)

    pt = sub.add_parser("pretrain", help="label-free pretraining over a dataset")
    pt.add_argument("--data", required=True)
    pt.add_argument("--ckpt", default=None)
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--max-instances", type=int, default=None)
    pt.add_argument("--device", default="auto", help="cpu | cuda | cuda:N | auto")
    pt.set_defaults(func=_cmd_pretrain)

    bat = sub.add_parser("battery", help="run the Phase-1 falsification battery + Gate G1")
    bat.add_argument("--data", required=True)
    bat.add_argument("--out", default=None, help="path to write the JSON report")
    bat.add_argument("--budgets", default="16,64,256")
    bat.add_argument("--experiments", default="E1,E3,E5", help="comma list of E1..E5")
    bat.add_argument("--n-val", type=int, default=16)
    bat.add_argument("--seed", type=int, default=0)
    bat.add_argument("--decision-budget", type=int, default=64)
    bat.add_argument("--lambda-phys", type=float, default=1.0)
    bat.add_argument("--lambda-grid", default=None, help="comma list, e.g. 0.1,0.3,1.0 (E1 sweep)")
    bat.add_argument("--n-seeds", type=int, default=1)
    bat.add_argument("--epochs", type=int, default=40)
    bat.add_argument("--lr", type=float, default=3e-3)
    bat.add_argument("--dim", type=int, default=96)
    bat.add_argument("--depth", type=int, default=4)
    bat.add_argument("--device", default="auto", help="cpu | cuda | cuda:N | auto")
    bat.set_defaults(func=_cmd_battery)

    rc = sub.add_parser("run-config", help="run a full experiment pipeline from a JSON config")
    rc.add_argument("config")
    rc.set_defaults(func=_cmd_run_config)

    reg = sub.add_parser("regimes", help="compare labels / labels+anchor / anchor-only")
    reg.add_argument("--data", required=True)
    reg.add_argument("--out", default=None)
    reg.add_argument("--n-train", type=int, default=48)
    reg.add_argument("--n-val", type=int, default=16)
    reg.add_argument("--seed", type=int, default=0)
    reg.add_argument("--epochs", type=int, default=60)
    reg.add_argument("--lr", type=float, default=1.5e-3)
    reg.add_argument("--dim", type=int, default=96)
    reg.add_argument("--depth", type=int, default=4)
    reg.add_argument("--device", default="auto", help="cpu | cuda | cuda:N | auto")
    reg.set_defaults(func=_cmd_regimes)

    info = sub.add_parser("info", help="summarise a dataset directory")
    info.add_argument("--data", required=True)
    info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
