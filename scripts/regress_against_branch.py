#!/usr/bin/env python3
"""Cross-branch regression (wp8 Stage 1.36, extended at Stage 1.37): does this
checkout reproduce another checkout bitwise on the main-line (Phase-2b) code
path?

Runs a MINIATURE of a stamped configuration (default `configs/phase2b_v1.json`:
same structure -- E8 with labels, labels_anchor at a fixed-lambda budget (4)
and at the gradient-balanced decision budget (64), AR and MGN; P3 zero-shot
and few-shot; WP6 and E6; gate G2 with its reference form -- at toy sizes,
CPU, one thread, guard off), each side in its own process importing `fejepa`
from its own source tree (refused if the import resolves elsewhere), twice:
fresh, then again in restart mode (`reuse_states`: AR states reloaded,
supervised units from the cache). It compares every number of the two
reports (all top-level blocks except the configuration and provenance, whose
paths and git strings differ; the corpus manifests are compared), for both
passes -- bitwise, except WP6's ARPACK values (round-off, see _TOLERANT) --
and the bytes of every state file. Exit status 0 iff all identical. The
summary names each side by its source tree's git id (`src_tree`: what `git
rev-parse <commit>:src` prints for a commit of those files), not by a path.

    git worktree add ../FE-JEPA-wp7 origin/wp7-3d
    python scripts/regress_against_branch.py --other ../FE-JEPA-wp7/src \
        --work runs/regress_wp7 --out runs/regress_wp7/summary.json

wp9 Stage 0b runs it on E1's 2D base configuration against wp8-lejepa (the
reuse of E1's states for C1's 1,024-instance arm rests on it):

    python scripts/regress_against_branch.py --other ../FE-JEPA-wp8/src \
        --config configs/e1_2d_base.json --work runs/regress_wp8_e1 \
        --out runs/regress_wp8_e1/summary.json

Bitwise identity is a same-machine, same-torch statement: run both sides here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_IGNORED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints", "runs"}
_IGNORED_SUFFIXES = (".pyc", ".pt")
"""What the repository's .gitignore leaves out (plus *.egg-info directories)."""


def src_tree(path) -> str:
    """The git tree id of a source directory as it is on disk: what `git
    rev-parse <commit>:src` prints for a commit of exactly these files. Regular
    files are blobs (mode 100755 if the owner may execute them, else 100644),
    directories trees; what .gitignore leaves out and empty directories are
    skipped; a symbolic link is refused."""
    def tree(d: Path):
        entries = []
        for p in d.iterdir():
            if p.is_symlink():
                raise ValueError(f"{p}: a symbolic link (src_tree does not hash links)")
            if p.is_dir():
                if p.name in _IGNORED_DIRS or p.name.endswith(".egg-info"):
                    continue
                h = tree(p)
                if h is not None:
                    entries.append((p.name + "/", b"40000 " + p.name.encode(), h))
            elif not p.name.endswith(_IGNORED_SUFFIXES):
                data = p.read_bytes()
                mode = b"100755" if p.stat().st_mode & 0o100 else b"100644"
                entries.append((p.name, mode + b" " + p.name.encode(),
                                hashlib.sha1(b"blob %d\0" % len(data) + data).digest()))
        if not entries:
            return None
        body = b"".join(head + b"\0" + h for _, head, h in sorted(entries))
        return hashlib.sha1(b"tree %d\0" % len(body) + body).digest()

    h = tree(Path(path))
    return h.hex() if h is not None else hashlib.sha1(b"tree 0\0").hexdigest()


def _rel(path) -> str:
    """A path relative to this checkout if it lies inside, else its name."""
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name


def miniature(cfg: dict, work: Path, n: int = 80, n_fine: int = 7) -> dict:
    """The configuration at toy sizes; everything else as stamped. Budget 64 is
    kept so that labels_anchor runs its gradient-balanced policy
    (e8_regimes.POLICY_BALANCED_FROM) as well as the fixed one."""
    m = copy.deepcopy(cfg)
    m["data"].update(dir=str(work / "d3"), n=n, lc_range=[0.30, 0.36])
    if m["data"].get("labelled_policy") == "asis":
        # wp9: a stamped 2D configuration reads a pre-existing corpus; the
        # miniature generates its own (unlabelled, then labelled as the run needs)
        m["data"]["labelled_policy"] = "economy"
    if m.get("data_transfer"):
        m["data_transfer"].update(dir=str(work / "d3f"), n=n_fine, lc=0.24,
                                  split={"n_eval": 3, "n_fewshot_prefix": 4})
    m["split"] = {"n_val": 3, "seed": 1}
    if m.get("labels"):
        m["labels"].update(inband_prefix=64, fine_prefix=4)
    m["model"].update(dim=16, depth=1, heads=2, mgn_dim=16, mgn_depth=1)
    m.setdefault("sup", {})["epochs"] = 2
    m.setdefault("pretrain", {})["epochs"] = 2
    exps = m["experiments"]
    exps["e8"].update(budgets=[4, 64], seeds=2, ar_epochs=2, sup_epochs=2, pool_sizes=[64],
                      mgn_budgets=[64])
    if "p3_transfer" in exps:
        exps["p3_transfer"].update(fewshot_budgets=[2, 4], fewshot_epochs=2, naive_budget=64)
    if "wp6" in exps:
        exps["wp6"].update(enabled=True, n_check=2)
    if "e6" in exps:
        exps["e6"].update(enabled=True, pool_size=8, pre_epochs=2)
    if m.get("gate_g2"):
        m["gate_g2"].update(decision_budget=64, sanity_min_budget=64)
    m["prereg_guard"] = False
    m["device"], m["workers"] = "cpu", 1
    m["out"] = str(work / "report.json")
    return m


_CHILD = r'''
import json, shutil, sys
from pathlib import Path
src, cfg_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, src)

def main():
    import torch
    torch.set_num_threads(1)
    import fejepa
    if not Path(fejepa.__file__).resolve().is_relative_to(Path(src).resolve()):
        raise SystemExit(f"fejepa imported from {fejepa.__file__}, not from {src}")
    from fejepa.experiments.runner import run_config
    out = Path(json.loads(Path(cfg_path).read_text())["out"])
    run_config(cfg_path, device_override="cpu")
    shutil.copyfile(out, out.with_name("report_fresh.json"))
    run_config(cfg_path, device_override="cpu", reuse_states=True)
    shutil.copyfile(out, out.with_name("report_restart.json"))
    print("[regress] fejepa from", fejepa.__file__, flush=True)

if __name__ == "__main__":
    main()
'''


def run_side(src: str, cfg: dict, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    cp = work / "mini.json"
    cp.write_text(json.dumps(miniature(cfg, work), indent=1))
    child = work / "child.py"
    child.write_text(_CHILD)
    env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(child), str(Path(src).resolve()), str(cp)],
                       cwd=str(work), capture_output=True, text=True, env=env)
    (work / "run.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"{src}: miniature run failed (see {work / 'run.log'}):\n"
                         + (r.stdout + r.stderr)[-2000:])
    reports = {}
    for name in ("fresh", "restart"):
        rep = json.loads((work / f"report_{name}.json").read_text())
        rep["corpus_manifests"] = [(d.get("manifest_sha256"), d.get("n_instances"))
                                   for d in (rep.get("provenance") or {}).get("datasets", [])]
        reports[name] = {k: v for k, v in rep.items() if k not in ("config", "provenance")}
    states = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((work / "e8_states").glob("*.pt"))}
    return {"reports": reports, "states": states}


_SKIP = {"wall_clock_s", "seconds", "elapsed_s", "time_s", "timing", "state_path",
         "state_dir", "cache_dir"}
_TOLERANT = ("/results/wp6/",)
"""WP6's eigenvalue checks use ARPACK with a random start vector: they differ
at round-off from run to run on ONE branch (fresh vs restart pass), so they
are compared to a relative 1e-12 plus an absolute 1e-14 (some of them, e.g.
the modewise-contraction error, are round-off quantities themselves; the
observed differences are below 6e-15 relative and 1e-15 absolute);
everything else must be bitwise equal."""
_RTOL, _ATOL = 1e-12, 1e-14


def compare(a, b, path="", out=None, n=None):
    """Paths where two JSON trees differ (floats: exact, NaN equals NaN; under
    _TOLERANT prefixes: relative 1e-12 plus absolute 1e-14)."""
    out = [] if out is None else out
    n = [0] if n is None else n
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in _SKIP:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present on one side only")
            else:
                compare(a[k], b[k], f"{path}/{k}", out, n)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (u, v) in enumerate(zip(a, b, strict=True)):
                compare(u, v, f"{path}[{i}]", out, n)
    elif isinstance(a, float) and isinstance(b, float):
        n[0] += 1
        if math.isnan(a) and math.isnan(b):
            return out, n[0]
        tol = (_RTOL * max(abs(a), abs(b)) + _ATOL) if any(t in path for t in _TOLERANT) else 0.0
        if not abs(a - b) <= tol:
            out.append(f"{path}: {a!r} vs {b!r}")
    elif a != b:
        out.append(f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}")
    return out, n[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True, help="the other checkout's src directory")
    ap.add_argument("--src", default=str(ROOT / "src"), help="this side (default: here)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2b_v1.json"))
    ap.add_argument("--work", default="runs/regress")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    for side in (a.src, a.other):
        if not Path(side, "fejepa", "__init__.py").exists():
            raise SystemExit(f"{side}: no fejepa package there (pass a checkout's src directory)")
    cfg = json.loads(Path(a.config).read_text())
    work = Path(a.work)
    this = run_side(a.src, cfg, work / "this")
    other = run_side(a.other, cfg, work / "other")
    diffs, n_float = [], 0
    for name in ("fresh", "restart"):
        d, n = compare(this["reports"][name], other["reports"][name], f"/{name}")
        diffs += d
        n_float += n
    names = sorted(set(this["states"]) | set(other["states"]))
    states = {nm: this["states"].get(nm) == other["states"].get(nm) for nm in names}
    summary = {"this_src_tree": src_tree(a.src), "other_src_tree": src_tree(a.other),
               "config": _rel(a.config), "floats_compared": n_float, "differences": diffs[:50],
               "n_differences": len(diffs), "state_files": len(names),
               "state_files_identical": states,
               "identical": not diffs and all(states.values()) and bool(names)}
    text = json.dumps(summary, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)
    sys.exit(0 if summary["identical"] else 1)


if __name__ == "__main__":
    main()
