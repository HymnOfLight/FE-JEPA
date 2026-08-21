"""Command-line interface: generate | label | run-config | bench | info."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fejepa",
                                description="FE-JEPA v2.0 (plan-v2.0 one-to-one)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a dataset (economy: unlabelled)")
    g.add_argument("out")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--backend", choices=["gmsh", "synthetic", "tet3d", "gmsh3d"],
                   default="gmsh")
    g.add_argument("--labelled", choices=["none", "all"], default="none")
    g.add_argument("--jobs", type=int, default=0)
    g.add_argument("--lc", type=float, default=0.30,
                   help="gmsh3d characteristic length (WP7 3D-G1)")
    g.add_argument("--multires-coarsen", type=float, default=None)

    l = sub.add_parser("label", help="solve+attach U* for val + a pool prefix")
    l.add_argument("data_dir")
    l.add_argument("--n-val", type=int, required=True)
    l.add_argument("--split-seed", type=int, default=1)
    l.add_argument("--pool-prefix", type=int, default=0)
    l.add_argument("--workers", type=int, default=1,
                   help="parallel CPU labelling processes")

    r = sub.add_parser("run-config", help="execute a config end-to-end")
    r.add_argument("config")
    r.add_argument("--device", default=None, choices=["cpu", "cuda", "auto"],
                   help="override the config's device field")
    r.add_argument("--workers", type=int, default=None,
                   help="override the config's workers field (concurrent "
                        "training units on one GPU)")

    b = sub.add_parser("bench", help="measured ms/step (plan Sec.9)")
    b.add_argument("--device", default="cpu")
    b.add_argument("--config", default=None, help="config JSON to project steps for")

    i = sub.add_parser("info", help="dataset manifest summary")
    i.add_argument("data_dir")

    th = sub.add_parser("theory", help="WP6 numeric falsification pass "
                                        "(GPU-free; conditioning / Chebyshev / "
                                        "Prop.1 scoping)")
    th.add_argument("--data", default=None, help="labelled dataset dir")
    th.add_argument("--n-val", type=int, default=16)
    th.add_argument("--synthetic", type=int, default=0,
                    help="run on N in-memory synthetic instances instead")
    th.add_argument("--seed", type=int, default=0)

    rs = sub.add_parser("results", help="render RESULTS.md (and Figure-1) "
                                        "from a report JSON (plan WP1 acceptance)")
    rs.add_argument("report")
    rs.add_argument("--out", default=None)
    rs.add_argument("--figures", action="store_true")

    pr = sub.add_parser("prereg", help="compute / stamp / check the frozen config "
                                       "hash (plan Sec.5 item 7)")
    pr.add_argument("config")
    pr.add_argument("--stamp", action="store_true",
                    help="write the hash into the prereg file's CONFIG_SHA256 line")
    pr.add_argument("--prereg-file", default="PREREG.md")

    a = p.parse_args(argv)

    if a.cmd == "generate":
        from .fe.solve import SolveLedger

        ledger = SolveLedger()
        if a.backend == "tet3d":                            # WP7 3D-S1
            if a.multires_coarsen:
                raise SystemExit("3D multires is the E4-3D wiring item")
            from .fe.tet3d import generate_tet3d_dataset

            generate_tet3d_dataset(a.out, a.n, a.seed, labelled=a.labelled,
                                   ledger=ledger)
        elif a.backend == "gmsh3d":                         # WP7 3D-G1
            if a.multires_coarsen:
                raise SystemExit("3D multires is the E4-3D wiring item")
            from .fe.gmsh3d import generate_gmsh3d_dataset

            generate_gmsh3d_dataset(a.out, a.n, a.seed, labelled=a.labelled,
                                    lc=a.lc, ledger=ledger)
        elif a.backend == "synthetic":
            from .fe.synthetic import (generate_synthetic_dataset,
                                       generate_synthetic_multires)

            if a.multires_coarsen:
                generate_synthetic_multires(a.out, a.n, a.seed, a.multires_coarsen,
                                            labelled=a.labelled, ledger=ledger)
            else:
                generate_synthetic_dataset(a.out, a.n, a.seed,
                                           labelled=a.labelled, ledger=ledger)
        else:
            from .fe.generator import generate_dataset, generate_multires_dataset

            if a.multires_coarsen:
                generate_multires_dataset(a.out, a.n, a.seed, a.multires_coarsen,
                                          labelled=a.labelled, ledger=ledger)
            else:
                generate_dataset(a.out, a.n, a.seed, labelled=a.labelled,
                                 jobs=a.jobs, ledger=ledger)
        print(f"[fejepa] generated {a.n} -> {a.out}; ledger={ledger.as_dict()}")

    elif a.cmd == "label":
        from .experiments.protocol import load_split
        from .experiments.runner import _label_files
        from .fe.solve import SolveLedger

        ledger = SolveLedger()
        split = load_split(a.data_dir, a.n_val, a.split_seed)
        n1 = _label_files(split.val_files, ledger, "labelling-val",
                          workers=a.workers)
        n2 = _label_files(split.pool_files[:a.pool_prefix], ledger,
                          "labelling-pool-prefix", workers=a.workers)
        print(f"[fejepa] labelled val={n1} pool={n2}; ledger={ledger.as_dict()}")

    elif a.cmd == "run-config":
        from .experiments.runner import run_config

        run_config(a.config, device_override=a.device,
                   workers_override=a.workers)

    elif a.cmd == "bench":
        from .experiments.cost import bench, count_steps

        cfgd = json.loads(Path(a.config).read_text()) if a.config else {}
        out = bench(device=a.device, tf32=bool(cfgd.get("tf32", True)),
                    model_cfg=cfgd.get("model"))
        if a.config:
            cfg = cfgd
            steps = count_steps(cfg)
            ms = out["ms_per_supervised_step"]
            w = max(1, int(cfg.get("workers", 1)))
            out["projected"] = {"steps": steps, "workers": w,
                                "hours_single_stream":
                                    round(steps["total"] * ms / 3.6e6, 2),
                                "hours_with_workers":
                                    round(steps["total"] * ms / 3.6e6 / w, 2)}
        print(json.dumps(out, indent=1))

    elif a.cmd == "theory":
        import numpy as np

        from .theory import run_theory_checks

        if a.synthetic:
            from .fe.synthetic import synthetic_instance

            rng = np.random.default_rng(a.seed)
            archs = [synthetic_instance(rng, labelled=True)
                     for _ in range(a.synthetic)]
        elif a.data:
            from .data.archive import load_instance
            from .experiments.protocol import load_split

            split = load_split(a.data, a.n_val, seed=1)
            archs = [load_instance(f) for f in split.val_files]
            if not all(x.labelled for x in archs):
                raise SystemExit(
                    "theory: the val split is unlabelled; label it first:\n"
                    f"  fejepa label --data {a.data} --n-val {a.n_val}")
        else:
            raise SystemExit("theory: give --data DIR or --synthetic N")
        res = run_theory_checks(archs, {"n_check": len(archs), "seed": a.seed})
        print(json.dumps({"metrics": res["metrics"], "kills": res["kills"]},
                         indent=1))

    elif a.cmd == "results":
        from .results import write_figures, write_results

        rp = Path(a.report)
        out = Path(a.out) if a.out else rp.parent / "RESULTS.md"
        print(f"RESULTS.md -> {write_results(rp, out)}")
        if a.figures:
            for f in write_figures(rp, rp.parent):
                print(f"figure -> {f}")

    elif a.cmd == "prereg":
        from .report import config_sha256, read_prereg_hash, stamp_prereg

        cfg = json.loads(Path(a.config).read_text())
        if a.stamp:
            h = stamp_prereg(a.prereg_file, cfg)
            print(f"stamped {a.prereg_file}: CONFIG_SHA256 = {h}")
            print("now: git add + commit + `git tag prereg-v2.0`")
        else:
            h = config_sha256(cfg)
            rec = read_prereg_hash(a.prereg_file) if Path(a.prereg_file).exists() \
                else None
            state = ("MATCH" if rec == h else
                     "unstamped placeholder" if rec and "<" in rec else
                     f"MISMATCH (recorded {rec[:12]}...)" if rec else
                     "no CONFIG_SHA256 line / file missing")
            print(f"config hash: {h}")
            print(f"{a.prereg_file}: {state}")

    elif a.cmd == "info":
        from .data.archive import load_manifest, manifest_sha256

        m = load_manifest(a.data_dir)
        n_lab = sum(1 for r in m.get("instances", []) if r.get("labelled"))
        print(json.dumps({"backend": m.get("backend"),
                          "n_instances": m.get("n_instances"),
                          "n_labelled": n_lab,
                          "labelled_policy": m.get("labelled_policy"),
                          "coarsen": m.get("coarsen"),
                          "manifest_sha256": manifest_sha256(a.data_dir)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
