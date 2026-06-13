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

    res = run_gate_g0(arch, steps=args.steps, lr=args.lr, log_every=args.log_every)
    print(res)
    return 0 if res.passed else 1


def _cmd_pretrain(args: argparse.Namespace) -> int:
    from fejepa.train.pretrain import PretrainConfig, pretrain

    cfg = PretrainConfig(epochs=args.epochs, lr=args.lr, max_instances=args.max_instances)
    result = pretrain(args.data, out_ckpt=args.ckpt, cfg=cfg)
    print(
        f"Pretraining done: {result['steps']} steps in "
        f"{result['elapsed_sec']:.1f}s"
        + (f", checkpoint -> {result['checkpoint']}" if "checkpoint" in result else "")
    )
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
    g0.set_defaults(func=_cmd_gate_g0)

    pt = sub.add_parser("pretrain", help="label-free pretraining over a dataset")
    pt.add_argument("--data", required=True)
    pt.add_argument("--ckpt", default=None)
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--max-instances", type=int, default=None)
    pt.set_defaults(func=_cmd_pretrain)

    info = sub.add_parser("info", help="summarise a dataset directory")
    info.add_argument("--data", required=True)
    info.set_defaults(func=_cmd_info)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
