#!/usr/bin/env python3
"""PREREG_HOLDOUT: the claims T1-T3, H1, H2a, H2b, H2c and D2, the reading H3
and the secondary readings, from the box return of RUNBOOK_CMAME Sec. E.

    python scripts/adjudicate_holdout.py --return records/cmame/holdout/return \\
        --out records/cmame/holdout/verdict.json

The return holds `status.txt`, `provenance.txt` (the commit, the tree and the
SHA-256 of every other returned file), the logs and machine records, and per
set that ran `<set>/eval.json`, `<set>/eval_arrays.npz` and
`<set>/manifest.json` (the labels manifest; the labels themselves stay on the
box). PREREG_HOLDOUT.md must be stamped for configs/holdout_v1.json (its
CONFIG_SHA256 line, its last line its own SHA-256).

It refuses to adjudicate at all (PREREG_HOLDOUT Sec. 7) when: the
pre-registration is not stamped for this configuration; the provenance file's
commit or tree is not `prereg-holdout`'s in this checkout (a checkout without
the tag records a deviation instead); the provenance file does not list
`status.txt`, lists a file that is missing or differs, or the return holds a
file it does not list; an evaluation records another set, configuration,
git describe than `prereg-holdout`, report, states, rows, seeds, metrics,
content tolerance or excluded instances, a limit on the instances, other
instance counts than its arrays', or arrays other than those returned; the
labels manifest does not describe the set, names other
instances than the set's in pool order, or holds a label above the residual
bound; the arrays name other rows, seeds, metrics, holdout or validation
instances, or have other shapes or load cases; the evaluation's summaries,
zero-field counts or content checks are not those of its arrays (recomputed
here); a report of the configuration is not the one it names in this
checkout; a set's evaluation records another labels manifest than the one
returned while `status.txt` records no labelling of the set after the start
of the evaluation that wrote the returned file (its last attempt that
exited 0 or 5). A state that fails its content check (recomputed here from
the returned validation arrays against the report's) leaves the claims that
read it NOT EVALUATED. It records as deviations, reading `status.txt` by
attempt (a start line and the exit line that follows it): a stage started
more than once, an attempt without an exit line, a last attempt cut off or
ending with a non-zero exit, a stage missing, and a stage outside the
runbook; stages started on more than one host; a set that did not return,
that is, whose directory lacks the evaluation, its arrays or its labels
manifest, or whose evaluation did not complete, and a set whose evaluation
records another labels manifest than the one returned after such a
labelling (not adjudicable; every other check of the set still applies)
(the claims of either NOT EVALUATED); a resumed labelling; a holdout instance that the corpus manifest
or the corpus file records as labelled; a torch version or GPU other than
the original runs'; and adjudicating code that differs from its
`prereg-holdout` version. The verdict file records every input's SHA-256 and
the adjudicating code's.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LABEL = "holdout_v1"
TAG = "prereg-holdout"
PREREG_NAME = "PREREG_HOLDOUT.md"
RETURN_FILES = ("status.txt", "provenance.txt")
SET_FILES = ("eval.json", "eval_arrays.npz", "manifest.json")
STAGES = ("label_h3d.log", "eval_h3d.log", "label_h2d.log", "eval_h2d.log")
ORIGINAL_STACK = {"torch": "2.12.1+cu130", "gpu": "NVIDIA GeForce RTX 5090"}
"""The software and GPU of the runs whose states are evaluated (Phase-2b, E1
and CM2D record these)."""
ADJUDICATING_FILES = ("scripts/adjudicate_holdout.py", "scripts/stamp_prereg_holdout.py",
                      "src/fejepa/analysis/adjudicate_holdout.py",
                      "src/fejepa/analysis/holdout.py", "src/fejepa/analysis/adjudicate_w9.py",
                      "src/fejepa/analysis/adjudicate.py", "src/fejepa/report.py")
"""The files whose code decides a verdict or a refusal."""
ARRAY_KEYS = ("rows", "seeds", "metrics", "files", "values", "gap_load", "pi_load", "pi_star",
              "val_values", "val_gap_load", "val_pi_load", "val_pi_star", "val_files")


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonable(x):
    """The verdict as strict JSON: non-finite floats become strings."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(float(x)) else str(float(x))
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ------------------------------------------------------------ provenance --

def provenance_commit(text: str) -> dict:
    out = {}
    for key in ("HEAD", "tree"):
        m = re.search(rf"^{key} ([0-9a-f]{{40}})\s*$", text, re.M)
        out[key] = m.group(1) if m else None
    return out


def provenance_hashes(text: str) -> dict:
    """{path relative to the return, without a leading './': SHA-256} of the
    `sha256sum` lines."""
    out = {}
    for line in text.splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line.strip())
        if m:
            p = m.group(2)
            out[p[2:] if p.startswith("./") else p] = m.group(1)
    return out


def check_listing(ret: Path, listed: dict) -> None:
    """Every file of the return but the provenance file is listed with its
    SHA-256, and the listing names no other file (ValueError otherwise)."""
    present = {p.relative_to(ret).as_posix() for p in ret.rglob("*")
               if p.is_file() and p.relative_to(ret).as_posix() != "provenance.txt"}
    if "status.txt" not in listed:
        raise ValueError("provenance.txt does not list status.txt")
    unlisted = sorted(present - set(listed))
    if unlisted:
        raise ValueError(f"the return holds files provenance.txt does not list: {unlisted[:5]}")
    missing = sorted(set(listed) - present)
    if missing:
        raise ValueError(f"provenance.txt lists files the return lacks: {missing[:5]}")
    for p in sorted(present):
        if file_sha256(ret / p) != listed[p]:
            raise ValueError(f"{ret / p}: not the file the provenance file lists")


def _git(*args) -> str | None:
    try:
        p = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                           timeout=30)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def check_commit(prov_text: str, tag: str) -> str | None:
    rec = provenance_commit(prov_text)
    want = {"HEAD": _git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"),
            "tree": _git("rev-parse", "--verify", "--quiet", f"{tag}^{{tree}}")}
    if want["HEAD"] is None:
        return f"the run's commit not compared with {tag} (no such tag in this checkout)"
    if rec != want:
        raise ValueError(f"the return records HEAD {rec['HEAD']} and tree {rec['tree']}, "
                         f"{tag} is {want['HEAD']} with tree {want['tree']}")
    return None


def code_against(ref: str) -> dict:
    out = {"ref": ref, "compared": False, "differs": []}
    try:
        ok = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "--quiet",
                             f"{ref}^{{commit}}"], capture_output=True, timeout=10).returncode == 0
        if not ok:
            out["note"] = f"no {ref} in this checkout"
            return out
        for p in ADJUDICATING_FILES:
            rc = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", ref, "--", p],
                                capture_output=True, timeout=30).returncode
            if rc == 1:
                out["differs"].append(p)
            elif rc != 0:
                raise RuntimeError(f"git diff exited {rc} on {p}")
        out["compared"] = True
    except Exception as exc:                              # noqa: BLE001
        out["differs"] = []
        out["note"] = f"comparison failed ({type(exc).__name__}: {exc})"
    return out


def status_attempts(text: str) -> dict:
    """Per stage, its attempts in the order of `status.txt`: each a start line
    (`<log> start <UTC time> <host>`) with the exit line that follows it
    (`<log> exit=<code>`, None when cut off); an exit line without a start
    line before it is an attempt of its own. `line` is the attempt's first
    line."""
    out = {}
    for i, line in enumerate(text.splitlines()):
        s = line.strip()
        m = re.fullmatch(r"(\S+) exit=(-?\d+)", s)
        if m:
            att = out.setdefault(m.group(1), [])
            if att and att[-1]["start"] is not None and att[-1]["exit"] is None:
                att[-1]["exit"] = int(m.group(2))
            else:
                att.append({"start": None, "host": None, "exit": int(m.group(2)), "line": i})
            continue
        m = re.fullmatch(r"(\S+) start (\S+)(?: (\S+))?", s)
        if m:
            out.setdefault(m.group(1), []).append(
                {"start": m.group(2), "host": m.group(3), "exit": None, "line": i})
    return out


def labelled_after_evaluation(text: str, name: str) -> bool:
    """Whether `status.txt` records a start of the set's labelling after the
    start of the evaluation that wrote the returned evaluation (its last
    attempt that exited 0 or 5; the first attempt when none did): a
    labelling run again after it, which may rewrite the labels manifest."""
    att = status_attempts(text)
    ev = [a for a in att.get(f"eval_{name}.log", []) if a["start"] is not None]
    if not ev:
        return False
    done = [a for a in ev if a["exit"] in (0, 5)]
    since = (done[-1] if done else ev[0])["line"]
    return any(a["start"] is not None and a["line"] > since
               for a in att.get(f"label_{name}.log", []))


def status_deviations(text: str) -> list:
    """From the status lines, per stage read by attempt: stages started more
    than once, attempts without an exit line, a last attempt cut off or
    ending with an exit other than 0, a stage missing, stages started on
    more than one host, and lines of stages outside the runbook."""
    att = status_attempts(text)
    out = []
    hosts = {a["host"] for v in att.values() for a in v if a["host"]}
    for st in STAGES:
        a = att.get(st, [])
        codes = [x["exit"] for x in a if x["exit"] is not None]
        n_start = sum(1 for x in a if x["start"] is not None)
        cut = sum(1 for x in a if x["exit"] is None)
        if not codes:
            out.append(f"{st}: no exit line in status.txt"
                       + (f" ({n_start} start line(s))" if n_start else ""))
        if len(a) > 1:
            out.append(f"{st}: {len(a)} attempts (exits {codes})")
        if cut and codes:
            out.append(f"{st}: {cut} attempt(s) without an exit line")
        if codes and a[-1]["exit"] is None:
            out.append(f"{st}: last attempt cut off (no exit line)")
        elif codes and a[-1]["exit"] != 0:
            out.append(f"{st}: last exit {a[-1]['exit']}")
    for st in sorted(att):
        if st not in STAGES:
            out.append(f"{st}: a stage outside the runbook")
    if len(hosts) > 1:
        out.append(f"stages started on more than one host: {sorted(hosts)}")
    return out


# ------------------------------------------------------------------ sets --

def check_set(name: str, spec: dict, ev: dict, npz: dict, man: dict, shas: dict,
              expected_git: str) -> tuple:
    """Refuse (ValueError) a set's return that is not this configuration's run;
    returns ({(row, seed): used}, content checks recomputed, notes)."""
    from fejepa.analysis.holdout import (METRICS, content_check, holdout_names,
                                         report_arrays, spec_sha256, worse_than_zero)

    sdef = spec["sets"][name]
    seeds = [int(s) for s in spec["seeds"]]
    rows = list(sdef["rows"])
    sp = sdef["split"]
    want = {"set": name, "config_sha256": spec_sha256(spec), "git": expected_git,
            "report_sha256": sdef["report_sha256"], "n_val_limit": None,
            "n_holdout_limit": None,
            "arrays_sha256": shas["eval_arrays.npz"], "rows": rows, "seeds": seeds,
            "metrics": list(METRICS), "content_tol": spec["content_tol"],
            "excluded": list(sdef["excluded"])}
    for k, v in want.items():
        if ev.get(k) != v:
            raise ValueError(f"{name}/eval.json: {k} is {ev.get(k)!r}, expected {v!r}")
    for r in rows:
        for s in seeds:
            rec = (ev.get("states") or {}).get(f"{r}_s{s}") or {}
            if rec.get("sha256") != sdef["rows"][r]["sha256"][str(s)]:
                raise ValueError(f"{name}/eval.json: state {r}_s{s} is not the configuration's")
    # the labels manifest
    lw = {"set": name, "corpus_manifest_sha256": sdef["manifest_sha256"],
          "n_corpus": int(sdef["n_corpus"]),
          "split": {"n_val": int(sp["n_val"]), "seed": int(sp["seed"])},
          "used_prefix": int(sdef["used_prefix"]), "excluded": list(sdef["excluded"]),
          "method": "direct", "n": int(sdef["n_holdout"])}
    for k, v in lw.items():
        if man.get(k) != v:
            raise ValueError(f"{name}/manifest.json: {k} is {man.get(k)!r}, expected {v!r}")
    names = holdout_names(sdef)
    if [r["file"] for r in man["instances"]] != names:
        raise ValueError(f"{name}/manifest.json: not the split's holdout set in pool order")
    bound = float(spec["labels"]["residual_max"])
    if any(not r["residual"] <= bound for r in man["instances"]):
        raise ValueError(f"{name}/manifest.json: a label above the residual bound")
    loads = {int(r["n_loads"]) for r in man["instances"]}
    if len(loads) != 1:
        raise ValueError(f"{name}/manifest.json: instances with {sorted(loads)} load cases")
    L = loads.pop()
    # the arrays
    missing = [k for k in ARRAY_KEYS if k not in npz]
    if missing:
        raise ValueError(f"{name}/eval_arrays.npz: lacks {missing}")
    if [str(x) for x in npz["rows"]] != rows or [int(x) for x in npz["seeds"]] != seeds \
            or [str(x) for x in npz["metrics"]] != list(METRICS):
        raise ValueError(f"{name}/eval_arrays.npz: rows, seeds or metrics differ")
    if [str(x) for x in npz["files"]] != names:
        raise ValueError(f"{name}/eval_arrays.npz: not the holdout instances in pool order")
    perm = np.random.default_rng(int(sp["seed"])).permutation(int(sdef["n_corpus"]))
    val_names = [f"instance_{int(i):05d}.npz" for i in perm[:int(sp["n_val"])]]
    if [str(x) for x in npz["val_files"]] != val_names:
        raise ValueError(f"{name}/eval_arrays.npz: not the run's validation split")
    R, S, N, Nv = len(rows), len(seeds), len(names), len(val_names)
    shapes = {"values": (R, S, N, len(METRICS)), "gap_load": (R, S, N, L),
              "pi_load": (R, S, N, L), "pi_star": (N, L),
              "val_values": (R, S, Nv, len(METRICS)), "val_gap_load": (R, S, Nv, L),
              "val_pi_load": (R, S, Nv, L), "val_pi_star": (Nv, L)}
    for k, shape in shapes.items():
        if tuple(npz[k].shape) != shape:
            raise ValueError(f"{name}/eval_arrays.npz: {k} has shape {tuple(npz[k].shape)}, "
                             f"expected {shape}")
    if (ev.get("holdout") or {}).get("n") != N or (ev.get("val") or {}).get("n") != Nv:
        raise ValueError(f"{name}/eval.json: instance counts differ from the arrays'")
    for r in rows:
        for s in seeds:
            ri, si = rows.index(r), seeds.index(s)
            got = npz["values"][ri, si].mean(axis=0)
            rec = ev["summary"][f"{r}_s{s}"]
            if not np.allclose(got, [rec[k] for k in METRICS], rtol=1e-12, atol=0,
                               equal_nan=True):
                raise ValueError(f"{name}/eval.json: summary {r}_s{s} is not its arrays' mean")
            w = worse_than_zero(npz["gap_load"][ri, si], npz["pi_load"][ri, si])
            if (ev.get("worse_than_zero") or {}).get(f"{r}_s{s}") != w:
                raise ValueError(f"{name}/eval.json: worse_than_zero {r}_s{s} is not its "
                                 "arrays'")
    # content, recomputed
    rp = ROOT / sdef["report"]
    if file_sha256(rp) != sdef["report_sha256"]:
        raise ValueError(f"{rp}: not the report the configuration names")
    report = json.loads(rp.read_text())
    used, checks = {}, {}
    for r in rows:
        for s in seeds:
            got = {k: npz["val_values"][rows.index(r), seeds.index(s), :, j].tolist()
                   for j, k in enumerate(METRICS)}
            ref = report_arrays(report, sdef, r, seeds.index(s))
            chk = content_check(got, {k: list(ref[k]) for k in METRICS},
                                float(spec["content_tol"]))
            box = (ev.get("content_checks") or {}).get(f"{r}_s{s}") or {}
            if box.get("ok") != chk["ok"]:
                raise ValueError(f"{name}/eval.json: content check {r}_s{s} reads "
                                 f"{box.get('ok')}, its arrays read {chk['ok']}")
            used[(r, s)] = chk["ok"]
            checks[f"{r}_s{s}"] = chk
    notes = []
    if man.get("reused"):
        notes.append(f"{name}: labelling resumed ({man['reused']} labels kept from an "
                     "earlier attempt)")
    flagged = [r["file"] for r in man["instances"]
               if r.get("corpus_labelled_flag") or r.get("source_labelled")]
    if flagged:
        notes.append(f"{name}: {len(flagged)} holdout instance(s) recorded as labelled by the "
                     f"corpus manifest or file (first {flagged[0]})")
    tor = ev.get("torch") or {}
    if tor.get("torch") != ORIGINAL_STACK["torch"] or tor.get("gpu") != ORIGINAL_STACK["gpu"]:
        notes.append(f"{name}: evaluated with torch {tor.get('torch')} on {tor.get('gpu')}, "
                     f"the original runs used {ORIGINAL_STACK['torch']} on "
                     f"{ORIGINAL_STACK['gpu']}")
    bad = [k for k, ok in used.items() if not ok]
    if bad:
        notes.append(f"{name}: content mismatch {sorted(f'{r}_s{s}' for r, s in bad)}")
    return used, checks, notes


def set_returned(ret: Path, name: str) -> str | None:
    """None when the set returned complete; otherwise why it did not (the
    set's claims are then NOT EVALUATED)."""
    d = ret / name
    if not d.is_dir():
        return f"{name}: no return"
    present = sorted(p.name for p in d.iterdir() if p.is_file())
    lack = [f for f in SET_FILES if f not in present]
    if lack:
        return f"{name}: not returned (lacks {lack}; present {present})"
    try:
        ev = json.loads((d / "eval.json").read_text())
    except ValueError:
        return f"{name}: not returned (eval.json does not parse)"
    if not isinstance(ev, dict) or "arrays_sha256" not in ev:
        return f"{name}: not returned (the evaluation did not complete: eval.json has no " \
               "arrays_sha256)"
    return None


def load_set(ret: Path, name: str) -> tuple:
    d = ret / name
    shas = {f: file_sha256(d / f) for f in SET_FILES}
    ev = json.loads((d / "eval.json").read_text())
    with np.load(d / "eval_arrays.npz", allow_pickle=False) as z:
        npz = {k: z[k] for k in z.files}
    man = json.loads((d / "manifest.json").read_text())
    return ev, npz, man, shas


def adjudicate(ret: Path, spec: dict, prereg: Path, expected_git: str) -> dict:
    from fejepa.analysis.adjudicate_holdout import Arrays, adjudicate_sets
    from fejepa.report import verify_prereg

    try:
        verify_prereg(spec, prereg, label=LABEL)
    except ValueError as exc:
        raise ValueError(f"{prereg}: {exc}") from exc
    if not _script("stamp_prereg_holdout").self_hash_ok(prereg.read_text(encoding="utf-8")):
        raise ValueError(f"{prereg}: its last line is not its own SHA-256")
    for f in RETURN_FILES:
        if not (ret / f).is_file():
            raise ValueError(f"{ret}: the return lacks {f} (RUNBOOK_CMAME Sec. E)")
    prov = (ret / "provenance.txt").read_text()
    listed = provenance_hashes(prov)
    check_listing(ret, listed)
    status = (ret / "status.txt").read_text()
    deviations = status_deviations(status)
    arrays, val_arrays, checks, inputs, why_not = {}, {}, {}, {}, {}
    for name in ("h3d", "h2d"):
        why = set_returned(ret, name)
        if why:
            deviations.append(why)
            arrays[name] = val_arrays[name] = None
            why_not[name] = why
            continue
        ev, npz, man, shas = load_set(ret, name)
        used, chk, notes = check_set(name, spec, ev, npz, man, shas, expected_git)
        deviations += notes
        inputs[name] = {f: {"path": str(ret / name / f), "sha256": shas[f]} for f in SET_FILES}
        if ev.get("labels_manifest_sha256") != shas["manifest.json"]:
            said = (f"its evaluation records the labels manifest "
                    f"{str(ev.get('labels_manifest_sha256'))[:12]}..., the return holds "
                    f"{shas['manifest.json'][:12]}...")
            if not labelled_after_evaluation(status, name):
                raise ValueError(f"{name}: {said}, and status.txt records no labelling "
                                 "after the evaluation")
            why = f"{name}: not adjudicable: {said} (a labelling after the evaluation)"
            deviations.append(why)
            arrays[name] = val_arrays[name] = None
            why_not[name] = why
            continue
        arrays[name] = Arrays(npz, used)
        val_arrays[name] = Arrays(npz, used, prefix="val_")
        checks[name] = chk
    res = adjudicate_sets(arrays["h3d"], arrays["h2d"], spec["criteria"],
                          h3d_val=val_arrays["h3d"], h2d_val=val_arrays["h2d"], why=why_not)
    res.update(content_checks=checks, deviations=deviations,
               inputs={**inputs, **{f: {"path": str(ret / f), "sha256": file_sha256(ret / f)}
                                    for f in RETURN_FILES},
                       "listed": listed, "recorded_commit": provenance_commit(prov),
                       "prereg": {"path": str(prereg), "sha256": file_sha256(prereg)}})
    return res


def main() -> None:
    from fejepa.analysis.holdout import load_spec, spec_sha256

    ap = argparse.ArgumentParser()
    ap.add_argument("--return", dest="ret", required=True, help="the return directory")
    ap.add_argument("--config", default=str(ROOT / "configs" / f"{LABEL}.json"))
    ap.add_argument("--prereg", default=str(ROOT / PREREG_NAME))
    ap.add_argument("--expected-git", default=TAG, help="the git describe the run must record")
    ap.add_argument("--code-ref", default=TAG,
                    help="the commit the return and the adjudicating code are compared with")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = load_spec(a.config)
    ret = Path(a.ret)
    try:
        commit_note = check_commit((ret / "provenance.txt").read_text(), a.code_ref) \
            if (ret / "provenance.txt").is_file() else None
        res = adjudicate(ret, spec, Path(a.prereg), a.expected_git)
    except ValueError as exc:
        raise SystemExit(f"refused: {exc}") from exc
    if commit_note:
        res["deviations"].append(commit_note)
    code = code_against(a.code_ref)
    res["config"] = {"path": a.config, "sha256": spec_sha256(spec)}
    res["adjudicator"] = {"sha256": {p: file_sha256(ROOT / p) for p in ADJUDICATING_FILES},
                          "against_stamp": code}
    if not code["compared"]:
        res["deviations"].append(f"adjudicating code not compared with {a.code_ref} "
                                 f"({code.get('note')})")
    elif code["differs"]:
        res["deviations"].append(f"adjudicating code differs from {a.code_ref} in "
                                 f"{code['differs']} (its SHA-256 under adjudicator)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(jsonable(res), indent=1, allow_nan=False) + "\n")
    print(json.dumps({c: v["verdict"] for c, v in res["claims"].items()}
                     | {"H3": res["readings"]["H3"]["reading"], "tally": res["tally"],
                        "deviations": res["deviations"]}, indent=1))


if __name__ == "__main__":
    main()
