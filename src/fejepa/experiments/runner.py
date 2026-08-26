"""Config-driven pipeline: dataset -> labelling (economy) -> experiments -> gate -> report.

Plan v2.0 mapping:
  - WP1: one config file (`configs/phase1_rec8_v2.json`) is the deciding run; the runner
    executes it and writes a single report whose provenance block satisfies B1.
  - WP5: the labelling stage buys labels for exactly val + the maximal pool prefix any
    enabled experiment needs, through :class:`SolveLedger` -- the data-asymmetry table
    falls out of the report. ``labelled_policy``:
      * "economy": generate unlabelled, label what is needed (the plan's default);
      * "all":     legacy fully-labelled generation (economy not demonstrated);
      * "asis":    pre-existing corpus (e.g. the Phase-1 30k); labels are verified.
  - Experiment order: E1' -> E5' (anchored errors injected from E1') -> E8 -> E2 ->
    E3' -> E4' -> E6 -> E7 -> Gate G1'(E5', E8, E1').
  - ``data.backend``: "gmsh" (real corpus) or "synthetic" (smoke/tests/bench; no
    meshing stack required).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import numpy as np
from pathlib import Path

from ..data.archive import add_labels, load_instance, load_manifest, mark_labelled, instance_files
from ..fe.solve import SolveLedger, solve_fe_displacement
from ..models.features import FeatureSpec
from ..models.fejepa import FEJEPAConfig, build_fejepa
from ..progress import stage
from ..runtime import setup_torch
from ..report import provenance, write_report
from .protocol import asis_missing_labels, load_split, require_asis_corpus
from . import gate as gate_mod
from .cost import count_steps


# ------------------------------------------------------------- factories ----

def make_model_factory(model_cfg: dict):
    base = FEJEPAConfig.from_dict(model_cfg)

    def factory(features: FeatureSpec | None = None):
        cfg = base if features is None else replace(base, features=features)
        return build_fejepa(cfg)

    return factory, base


# ------------------------------------------------------------ data stages ----

def _ensure_dataset(dcfg: dict, ledger: SolveLedger) -> Path:
    ddir = Path(dcfg["dir"])
    if (ddir / "manifest.json").exists():
        return ddir
    require_asis_corpus(dcfg)   # asis must never trigger generation (v2.1.5)
    n, seed = int(dcfg.get("n", 100)), int(dcfg.get("seed", 0))
    labelled = "all" if dcfg.get("labelled_policy") == "all" else "none"
    backend = dcfg.get("backend", "gmsh")
    if backend == "synthetic":
        from ..fe.synthetic import generate_synthetic_dataset

        generate_synthetic_dataset(ddir, n, seed, labelled=labelled, ledger=ledger)
    elif backend == "tet3d":                               # WP7 3D-S1
        from ..fe.tet3d import generate_tet3d_dataset

        generate_tet3d_dataset(ddir, n, seed, labelled=labelled, ledger=ledger,
                               nx=int(dcfg.get("nx", 4)), ny=int(dcfg.get("ny", 3)),
                               nz=int(dcfg.get("nz", 3)))
    elif backend == "gmsh3d":                              # WP7 3D-G1
        from ..fe.gmsh3d import generate_gmsh3d_dataset

        generate_gmsh3d_dataset(ddir, n, seed, labelled=labelled, ledger=ledger,
                                lc=float(dcfg.get("lc", 0.30)),
                                lc_range=dcfg.get("lc_range"),
                                solve_method=str(dcfg.get("solve_method", "cg")))
    else:
        from ..fe.generator import generate_dataset

        generate_dataset(ddir, n, seed, labelled=labelled,
                         jobs=int(dcfg.get("jobs", 0)), ledger=ledger)
    return ddir


def _ensure_multires(dcfg: dict, coarsen: float, n: int, ledger: SolveLedger) -> Path:
    ddir = Path(str(dcfg["dir"]) + "_c" + str(coarsen).replace(".", "p"))
    if (ddir / "manifest.json").exists():
        return ddir
    seed = int(dcfg.get("seed", 0)) + 7919
    if dcfg.get("backend") == "tet3d":                     # WP7 E4-3D
        from ..fe.tet3d import generate_tet3d_multires

        generate_tet3d_multires(ddir, n, seed, coarsen, ledger=ledger,
                                nx=int(dcfg.get("nx", 8)),
                                ny=int(dcfg.get("ny", 6)),
                                nz=int(dcfg.get("nz", 6)))
        return ddir
    if dcfg.get("backend") == "gmsh3d":                    # WP7 E4-3D
        from ..fe.gmsh3d import generate_gmsh3d_multires

        generate_gmsh3d_multires(ddir, n, seed, coarsen, ledger=ledger,
                                 lc=float(dcfg.get("lc", 0.30)),
                                 solve_method=str(dcfg.get("solve_method", "cg")))
        return ddir
    if dcfg.get("backend", "gmsh") == "synthetic":
        from ..fe.synthetic import generate_synthetic_multires

        generate_synthetic_multires(ddir, n, seed, coarsen, ledger=ledger)
    else:
        from ..fe.generator import generate_multires_dataset

        generate_multires_dataset(ddir, n, seed, coarsen, ledger=ledger)
    return ddir


def _label_one(fstr: str):
    """Top-level (picklable) labelling worker: solve+attach U* for one archive."""
    import time

    p = Path(fstr)
    arch = load_instance(p)
    if arch.labelled:
        return (fstr, 0, 0.0)
    t0 = time.perf_counter()
    U, _ = solve_fe_displacement(arch.K, arch.F, arch.free_mask, method="direct")
    add_labels(p, U)
    return (fstr, arch.n_loads, time.perf_counter() - t0)


def _label_files(files, ledger: SolveLedger, stage_name: str,
                 workers: int = 1) -> int:
    """Solve+attach U* for any unlabelled archive in `files`; returns #labelled now.

    workers > 1 fans the independent direct solves over CPU processes (this box has
    25 vCPUs; splu is single-threaded per instance). Ledger accounting is aggregated
    in the parent, so counts are identical to the serial path."""
    files = [str(f) for f in files]
    every = max(1, len(files) // 10)
    done: dict = {}
    if workers > 1 and len(files) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(processes=workers) as pool:
            it = pool.imap_unordered(_label_one, files, chunksize=4)
            for i, (fstr, n, dt) in enumerate(it, 1):
                if n:
                    ledger.add(stage_name, n=n, seconds=dt)
                    done.setdefault(Path(fstr).parent, set()).add(Path(fstr).name)
                if i % every == 0 or i == len(files):
                    print(f"[{stage_name}] {i}/{len(files)}", flush=True)
    else:
        for i, fstr in enumerate(files, 1):
            fstr, n, dt = _label_one(fstr)
            if n:
                ledger.add(stage_name, n=n, seconds=dt)
                done.setdefault(Path(fstr).parent, set()).add(Path(fstr).name)
            if i % every == 0 or i == len(files):
                print(f"[{stage_name}] {i}/{len(files)}", flush=True)
    for ddir, names in done.items():
        try:
            mark_labelled(ddir, names)
        except Exception:
            pass                                          # multires manifests have no records
    return sum(len(v) for v in done.values())


def _label_need(exps: dict) -> int:
    """Largest labelled pool prefix any enabled experiment consumes (plan WP5).
    Defaults here MUST mirror each experiment's own defaults. Includes the P3
    strongest-form naive budget (r8: fine naives are built from the labelled
    in-band prefix)."""
    need = 0
    for name in ("e1", "e2", "e8"):
        e = exps.get(name) or {}
        if e.get("enabled"):
            need = max(need, max(e.get("budgets", [1024]) or [0]))
    e5 = exps.get("e5") or {}
    if e5.get("enabled"):
        b5 = e5.get("budgets", [16, 64, 256, 1024]) or [0]
        need = max(need, max(b5), int(e5.get("fit_budget", max(b5))))
    e7 = exps.get("e7") or {}
    if e7.get("enabled"):
        need = max(need, int(e7.get("fit_budget", 256)))
    p3 = exps.get("p3_transfer") or {}
    if p3.get("enabled"):
        need = max(need, int(p3.get("naive_budget", 1024)))
    return need


def _pool_need(exps: dict) -> int:
    """Deepest pool prefix (labelled or not) any enabled experiment touches.
    Defaults mirror the experiments' own defaults (silent shortfalls are bugs)."""
    need = _label_need(exps)
    for name, key, dflt in (("e2", "pool_size", 1024), ("e6", "pool_size", 256),
                            ("e7", "pool_size", 256)):
        e = exps.get(name) or {}
        if e.get("enabled"):
            need = max(need, int(e.get(key, dflt)))
    e3 = exps.get("e3") or {}
    if e3.get("enabled"):
        need = max(need, int(e3.get("n_probe", 64)), int(e3.get("n_train", 32)))
    e8 = exps.get("e8") or {}
    if e8.get("enabled"):
        need = max(need, max(e8.get("pool_sizes", [1024]) or [0]))
    wp2 = exps.get("wp2") or {}
    if wp2.get("enabled"):
        need = max(need, int(wp2.get("n_train", 32)) + int(wp2.get("n_holdout", 16)))
    return need


# ----------------------------------------------------------------- runner ----

def data_economy_summary(ledger: SolveLedger, n_val: int, labelled_prefix: int,
                         unlabeled_pool_depth: int, n_loads: int) -> dict:
    """Plan WP5: 'the solve ledger then writes the data-asymmetry table'.

    Labelled instances = val + the budget prefix; every one costs ``n_loads``
    reference solves. The unlabeled pool depth is what AR/JEPA consume for free."""
    labelled = n_val + labelled_prefix
    return {
        "labelled_instances": labelled,
        "labelled_val": n_val,
        "labelled_pool_prefix": labelled_prefix,
        "solves_per_labelled_instance": n_loads,
        "reference_solves_total": ledger.total,
        "unlabeled_pool_depth_used": unlabeled_pool_depth,
        "unlabeled_over_labelled_pool":
            unlabeled_pool_depth / max(1, labelled_prefix),
        "ledger": ledger.as_dict(),
    }


def _pool_need_in_memory(exps: dict) -> int:
    """Pool prefix preloaded as archives -- only for the arch-based experiments
    (E2/E3'/E6/E7); E1'/E8 stream archive paths to their units instead."""
    need = 0
    for name, key, dflt in (("e2", "pool_size", 1024), ("e6", "pool_size", 256),
                            ("e7", "pool_size", 256)):
        e = exps.get(name) or {}
        if e.get("enabled"):
            need = max(need, int(e.get(key, dflt)), int(e.get("fit_budget", 0)))
    e3 = exps.get("e3") or {}
    if e3.get("enabled"):
        need = max(need, int(e3.get("n_probe", 64)), int(e3.get("n_train", 32)))
    e5 = exps.get("e5") or {}
    if e5.get("enabled"):
        b5 = e5.get("budgets", [16, 64, 256, 1024]) or [0]
        need = max(need, max(b5), int(e5.get("fit_budget", max(b5))))
    return need


def run_config(path, device_override: str | None = None,
               workers_override: int | None = None) -> dict:
    cfg = json.loads(Path(path).read_text())
    prereg = None
    if cfg.get("prereg_guard"):
        from ..report import verify_prereg

        pf = cfg.get("prereg_file", "PREREG.md")
        prereg = {"file": str(pf), "config_sha256": verify_prereg(cfg, pf)}
        print(f"[prereg] verified against {pf}: {prereg['config_sha256'][:12]}...",
              flush=True)
    device = device_override or cfg.get("device", "auto")
    workers = int(workers_override or cfg.get("workers", 1))
    label_workers = int(cfg.get("label_workers", min(8, os.cpu_count() or 8)))
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    policy = setup_torch(device, tf32=bool(cfg.get("tf32", True)))
    ledger = SolveLedger()
    exps = cfg.get("experiments", {})
    stage(f"fejepa v2 run: {path} | device={device} | workers={workers} "
          f"| tf32={policy['tf32']}")
    print(f"[plan] steps by experiment: {count_steps(cfg)}", flush=True)
    stage("dataset")
    ddir = _ensure_dataset(cfg["data"], ledger)
    split = load_split(ddir, int(cfg["split"]["n_val"]), int(cfg["split"]["seed"]))
    print(f"[dataset] {ddir} | n_val={len(split.val_files)} "
          f"pool={len(split.pool_files)}", flush=True)

    # ---- labelling stage (WP5 economy) -----------------------------------
    stage("labelling (WP5 economy)")
    need = _label_need(exps)
    lpolicy = cfg["data"].get("labelled_policy", "economy")
    print(f"[labelling] policy={cfg['data'].get('labelled_policy', 'economy')} "
          f"label_need(pool prefix)={need} label_workers={label_workers}",
          flush=True)
    if lpolicy == "asis":
        # Verify labels on everything the run will consume as labelled data:
        # the full val set plus the maximal pool prefix any enabled experiment
        # needs (E1'/E8 consume deterministic prefixes; see _label_need). The
        # old single-sample check accepted partially labelled corpora and let
        # runs die mid-experiment (v2.1.5).
        required = list(split.val_files) + list(split.pool_files[:need])
        missing = asis_missing_labels(required, ddir)
        if missing:
            qual = "at least " if len(missing) >= 8 else ""
            raise ValueError(
                f"labelled_policy='asis' but {qual}{len(missing)} of the "
                f"{len(required)} required instances (val "
                f"{len(split.val_files)} + pool prefix {need}) carry no "
                f"labels, e.g. {missing[:3]}. Pre-label them (config "
                f"unchanged, prereg hash preserved) with:\n"
                f"  fejepa label {ddir} --n-val {len(split.val_files)} "
                f"--split-seed {int(cfg['split']['seed'])} "
                f"--pool-prefix {need} --workers {label_workers}")
        print(f"[labelling] asis: verified labels on val "
              f"{len(split.val_files)} + pool prefix {need}", flush=True)
        ledger.add("asis-preexisting-corpus", n=0)
    else:
        _label_files(split.val_files, ledger, "labelling-val",
                     workers=label_workers)
        _label_files(split.pool_files[:need], ledger, "labelling-pool-prefix",
                     workers=label_workers)

    if _pool_need(exps) > len(split.pool_files):
        raise ValueError(f"pool needs {_pool_need(exps)} instances; dataset "
                         f"provides {len(split.pool_files)} after the val split")
    pool_hi = _pool_need_in_memory(exps)
    stage(f"loading archives (in-memory prefix {pool_hi} + val "
          f"{len(split.val_files)}; E1'/E8 stream paths)")
    pool_archs = [load_instance(f) for f in split.pool_files[:pool_hi]]
    val_archs = [load_instance(f) for f in split.val_files]

    factory, _ = make_model_factory(cfg.get("model", {}))
    model_cfg = cfg.get("model", {})
    dev = {"device": device}
    run_opts = {"device": device, "workers": workers, "tf32": policy["tf32"],
                "compile": bool(cfg.get("runtime", {}).get("compile", False))}
    results: dict = {}

    if (exps.get("e1") or {}).get("enabled"):
        from .e1_anchor import run_e1
        stage("E1'")

        results["e1"] = run_e1(model_cfg, split.pool_files, split.val_files,
                               {**exps["e1"], **run_opts})

    if (exps.get("e5") or {}).get("enabled"):
        from .e5_sanity import run_e5
        stage("E5'")

        anchored = None
        if "e1" in results:
            anchored = {}
            for entry in results["e1"]["metrics"]["per_budget"]:
                arm = entry["arms"]["balanced"]
                anchored[entry["budget"]] = {
                    "mean": arm["disp"]["mean"],
                    "per_instance": arm["per_instance_by_seed"][0]["disp_rel_l2"],
                }
        results["e5"] = run_e5(pool_archs, val_archs, {**exps["e5"], **dev},
                               anchored=anchored, model_factory=factory)

    if (exps.get("e8") or {}).get("enabled"):
        from .e8_regimes import run_e8
        stage("E8")

        results["e8"] = run_e8(model_cfg, split.pool_files, split.val_files,
                               {**exps["e8"], **run_opts,
                                "state_dir": str(Path(cfg.get("out",
                                                 "runs/report_v2.json")).parent
                                                 / "e8_states")})

    if (exps.get("wp2") or {}).get("enabled"):
        from .wp2_masking import run_wp2

        stage("WP2 mask-ratio sweep")
        results["wp2"] = run_wp2(model_cfg, pool_archs, {**exps["wp2"], **dev})

    if (exps.get("e2") or {}).get("enabled"):
        from .e2_jepa import run_e2
        stage("E2")

        results["e2"] = run_e2(factory, pool_archs, val_archs, {**exps["e2"], **dev})

    if (exps.get("e3") or {}).get("enabled"):
        from .e3_collapse import run_e3
        stage("E3'")

        results["e3"] = run_e3(factory, pool_archs, {**exps["e3"], **dev})

    extra_prov_dirs: list = []
    mr_dirs = {}
    if (exps.get("e4") or {}).get("enabled"):
        from .e4_meshviews import run_e4
        stage("E4'")

        e4c = exps["e4"]
        for c in e4c.get("coarsens", [1.8, 2.5]):
            mr_dirs[float(c)] = _ensure_multires(cfg["data"], float(c),
                                                 int(e4c.get("n", 500)), ledger)

        def load_pair(ddir_, rec):
            fine = load_instance(Path(ddir_) / rec["fine"])
            coarse = load_instance(Path(ddir_) / rec["coarse"])
            return fine, coarse

        # label the val pairs only (economy): E4 trains label-free
        n_val = int(e4c.get("n_val", 128))
        seed = int(e4c.get("seed", 0))
        for c, d in mr_dirs.items():
            pairs = load_manifest(d)["pairs"]
            perm = np.random.default_rng(seed).permutation(len(pairs))
            val_recs = [pairs[i] for i in perm[:n_val]]
            files = [Path(d) / r[k] for r in val_recs for k in ("fine", "coarse")]
            _label_files(files, ledger, f"labelling-multires-val-c{c}")
        results["e4"] = run_e4(factory, mr_dirs, load_pair, {**e4c, **dev})

    if (exps.get("e6") or {}).get("enabled"):
        from .e6_alignment import run_e6
        stage("E6")

        n_probe = min(int(exps["e6"].get("n_probe", 64)), len(val_archs))
        results["e6"] = run_e6(factory, pool_archs, val_archs[:n_probe],
                               {**exps["e6"], **dev})

    if (exps.get("e7") or {}).get("enabled"):
        from .e7_polish import run_e7
        stage("E7")

        results["e7"] = run_e7(factory, pool_archs, val_archs, {**exps["e7"], **dev})

    if (exps.get("p3_transfer") or {}).get("enabled"):
        from .p3_transfer import run_p3
        stage("P3 (resolution transfer)")

        dt = dict(cfg["data_transfer"])
        dt_split = dt.pop("split", {})
        fdir = _ensure_dataset(dt, ledger)
        extra_prov_dirs.append(fdir)
        ffiles = instance_files(Path(fdir))
        n_eval = int(dt_split.get("n_eval", 256))
        n_pref = int(dt_split.get("n_fewshot_prefix", 64))
        if len(ffiles) < n_eval + n_pref:
            raise ValueError(f"P3 transfer set has {len(ffiles)} instances; "
                             f"needs n_eval {n_eval} + prefix {n_pref}")
        # Pinned partition (r8 Sec.7): manifest order -- first n_eval are the
        # evaluation set, the next n_fewshot_prefix the few-shot prefix.
        fine_eval_files = ffiles[:n_eval]
        fine_prefix_files = ffiles[n_eval:n_eval + n_pref]
        _label_files(fine_eval_files, ledger, "labelling-fine-val",
                     workers=label_workers)
        _label_files(fine_prefix_files, ledger, "labelling-fine-prefix",
                     workers=label_workers)
        fine_eval_archs = [load_instance(f) for f in fine_eval_files]
        e8c = exps.get("e8") or {}
        results["p3_transfer"] = run_p3(
            model_cfg, split.pool_files, val_archs, fine_eval_archs,
            fine_eval_files, fine_prefix_files,
            {**exps["p3_transfer"], **run_opts,
             "state_dir": str(Path(cfg.get("out", "runs/report_v2.json")).parent
                              / "e8_states"),
             "pool_size": (e8c.get("pool_sizes") or [1024])[0],
             "bmax": max(e8c.get("budgets", [1024]))})

    if (exps.get("wp6") or {}).get("enabled"):
        from ..theory import run_theory_checks

        stage("WP6 theory falsification pass (GPU-free)")
        results["wp6"] = run_theory_checks(val_archs, exps["wp6"])

    if cfg.get("gate_g2"):
        from .gate_g2 import gate_g2

        stage("gate G2")
        gate = gate_g2(results.get("e8"), results.get("e1"),
                       results.get("p3_transfer"), results.get("e6"),
                       results.get("wp6"), gate_cfg=cfg.get("gate_g2"),
                       kill_cfg=cfg.get("kills"))
        gate_key = "gate_g2"
    else:
        stage("gate G1'")
        gate = gate_mod.g1_prime(results.get("e5"), results.get("e8"),
                                 results.get("e1"),
                                 decision_budget=int(cfg.get("gate", {})
                                                     .get("decision_budget", 64)),
                                 thresholds=cfg.get("gate", {}).get("thresholds"))
        gate_key = "gate_g1_prime"

    seeds = sorted({int(s) for e in exps.values() if isinstance(e, dict)
                    for s in range(int(e.get("seeds", 0) or 0))}) or [0]
    n_loads = val_archs[0].n_loads if val_archs else 4
    payload = {
        "config": cfg,
        "prereg": prereg,
        "data_economy": data_economy_summary(ledger, len(split.val_files), need,
                                             _pool_need(exps), n_loads),
        "runtime_policy": policy,
        "planned_steps": count_steps(cfg),
        "results": results,
        gate_key: gate,
        "solve_ledger": ledger.as_dict(),
        "provenance": provenance(cfg, [ddir, *mr_dirs.values(),
                                       *extra_prov_dirs], seeds),
    }
    out = cfg.get("out", "runs/report_v2.json")
    write_report(out, payload)
    print(f"[fejepa] report -> {out}")
    from ..results import write_figures, write_results

    # The report JSON above is already on disk; a rendering bug must never
    # cost a finished (possibly multi-day) run. Fail loud, not fatal.
    try:
        md = write_results(payload, Path(out).parent / "RESULTS.md")
        print(f"[fejepa] RESULTS.md -> {md}")
    except Exception as e:                                    # noqa: BLE001
        print(f"[fejepa] WARNING: RESULTS.md rendering failed "
              f"({type(e).__name__}: {e}); re-render later with "
              f"`fejepa results {out}`", flush=True)
    if results.get("e8"):
        try:
            for f in write_figures(payload, Path(out).parent):
                print(f"[fejepa] figure -> {f}")
        except ImportError as e:
            print(f"[fejepa] figures skipped: {e}")
        except Exception as e:                                # noqa: BLE001
            print(f"[fejepa] WARNING: figure rendering failed "
                  f"({type(e).__name__}: {e}); re-render later with "
                  f"`fejepa results {out} --figures`", flush=True)
    print(f"[fejepa] gate G1' passed={gate['passed']} conditions={gate['conditions']}")
    print(f"[fejepa] solve ledger: {ledger.as_dict()}")
    return payload
