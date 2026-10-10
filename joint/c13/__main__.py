"""Command line of gate G-C1.3. Run from the folder that holds fejoint and c13, with the repository at REPO
(python -I -m does not search the current folder, so the script is run by its path):

    python -I c13/__main__.py write-config --out c13_config.json --repo REPO   (prints the configuration hash, the
                                                                    code hashes, and checks the repository files)
    python -I c13/__main__.py generate --data DATA                   (exports, no solve)
    python -I c13/__main__.py label --data DATA --split eval         (exact solutions; also tested, train:N)
    python -I c13/__main__.py check-labels --data DATA               (files against manifest and ledger; residuals)
    python -I c13/__main__.py train --data DATA --runs RUNS --arm ARM --seed S --repo REPO --config-sha SHA
    python -I c13/__main__.py evaluate --data DATA --runs RUNS --config-sha SHA
    python -I c13/__main__.py bench --data DATA --repo REPO --out bench.json   (time per training step; no labels)

train and evaluate refuse to run unless --config-sha equals the hash of the configuration in c13/config.py; train,
bench and write-config --repo refuse to run unless the repository files have the SHA-256 the configuration pins.
"""
import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE = ["c13/__init__.py", "c13/__main__.py", "c13/config.py", "c13/data.py", "c13/model.py", "c13/train.py",
        "c13/evaluate.py", "fejoint/__init__.py", "fejoint/family.py", "fejoint/export.py", "fejoint/joint.py",
        "fejoint/specimens.py", "fejoint/hex8i.py", "fejoint/contact.py", "fejoint/linsolve.py"]


def code_hashes() -> dict:
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE}


def main(argv=None):
    here = pathlib.Path(__file__).resolve().parent              # the script's own folder must not shadow modules
    sys.path[:] = [p for p in sys.path if not p or pathlib.Path(p).resolve() != here]
    sys.path.insert(0, str(ROOT))
    from c13.config import config, sha256
    ap = argparse.ArgumentParser(prog="c13")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("write-config"); p.add_argument("--out", required=True); p.add_argument("--repo")
    p = sub.add_parser("generate"); p.add_argument("--data", required=True); p.add_argument("--limit", type=int)
    p = sub.add_parser("label"); p.add_argument("--data", required=True)
    p.add_argument("--split", required=True, help="eval, tested or train:N (the first N training members)")
    p = sub.add_parser("check-labels"); p.add_argument("--data", required=True)
    p = sub.add_parser("train")
    for k in ("--data", "--runs", "--arm", "--repo", "--config-sha"):
        p.add_argument(k, required=True)
    p.add_argument("--seed", type=int, required=True); p.add_argument("--device", default="cuda")
    p = sub.add_parser("evaluate")
    for k in ("--data", "--runs", "--config-sha"):
        p.add_argument(k, required=True)
    p = sub.add_parser("bench")
    for k in ("--data", "--repo", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--instances", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5); p.add_argument("--steps", type=int, default=40)
    a = ap.parse_args(argv)
    cfg = config()
    sha = sha256(cfg)
    if a.cmd == "write-config":
        pathlib.Path(a.out).write_text(json.dumps(cfg, indent=1, sort_keys=True))
        print("config sha256", sha)
        for f, h in code_hashes().items():
            print(f, h)
        if a.repo:
            sys.path.insert(0, str(pathlib.Path(a.repo) / "src"))
            from c13.train import verify_repository
            verify_repository(cfg)
            for f, h in cfg["repository_code"].items():
                print("repository", f, h, "checked")
        return
    if a.cmd in ("train", "evaluate") and a.config_sha != sha:
        raise SystemExit(f"configuration hash {sha} differs from the stamped {a.config_sha}; refusing to run")
    from c13 import data
    if a.cmd == "generate":
        print(data.generate(cfg, a.data, limit=a.limit))
    elif a.cmd == "label":
        man = data.manifest(a.data)
        if a.split.startswith("train:"):
            n = int(a.split.split(":")[1])
            ids = [r["id"] for r in man["instances"] if r["split"] == "train"][:n]
        else:
            ids = [r["id"] for r in man["instances"] if r["split"] == a.split]
        data.label(a.data, ids)
        print("labelled", len(ids))
    elif a.cmd == "check-labels":
        r = data.check_labels(cfg, a.data)
        print(json.dumps(r))
        if not r["ok"]:
            raise SystemExit("labels fail the checks; stop and report")
    elif a.cmd == "train":
        sys.path.insert(0, str(pathlib.Path(a.repo) / "src"))
        from c13.train import train
        hist = train(cfg, a.arm, a.seed, a.data, a.runs, device=a.device)
        print(json.dumps({k: hist[k] for k in ("status", "steps") if k in hist}))
    elif a.cmd == "evaluate":
        from c13.evaluate import evaluate
        v = evaluate(cfg, a.data, a.runs)
        print(json.dumps({k: v[k] for k in ("verdict", "C1_value", "C2_value", "missing", "non_finite") if k in v}))
    elif a.cmd == "bench":
        sys.path.insert(0, str(pathlib.Path(a.repo) / "src"))
        from c13.train import bench
        out = bench(cfg, a.data, device=a.device, instances=a.instances, warmup=a.warmup, steps=a.steps)
        out["config_sha256"] = sha
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out))


if __name__ == "__main__":
    main()
