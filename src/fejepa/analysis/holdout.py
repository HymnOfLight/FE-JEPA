"""cmame-paper Stage 11 (PREREG_HOLDOUT): the manuscript's comparisons on
instances that no network trained on and no decision read.

Sets. A run splits its corpus with a seeded permutation of the manifest
order (`fejepa.experiments.protocol.load_split`): the first `n_val` entries
are the validation instances, the rest the pool, and every recorded run
trained on, labelled or evaluated a network on a prefix of that pool only.
The tail of a corpus is the rest of its pool, `pool[used_prefix:]`, where
`used_prefix` is the deepest prefix any recorded run used; the holdout set
is the tail without the instances that another recorded set drew again,
that a recorded tool rebuilt, or whose own file is held outside the box (an
archive of a whole corpus kept as a backup does not count; `excluded`, from
`scripts/holdout_overlap.py`):

  h3d  runs/data3d_phase2 (2,000 instances, split 256 / seed 1): pool[1024:],
       720 instances, never trained on by any arm of the runs on that corpus
       (DEVIATIONS_PHASE2 D16; every 3D report records a pool depth of
       1,024), without the 34 that other sets of the same seed drew again
       (instances 0-79, by the cross-branch regression's miniature, and 106
       and 294, by the fine-mesh set): 686 instances;
  h2d  runs/data2d (30,000 instances, split 256 / seed 1): pool[25600:],
       4,144 instances (the deepest 2D pool, wp9's C1 arm, used 25,600),
       without the 32 of instances 0-255 that wp9's load-scale measurement
       rebuilt and the one other instance whose own file is held outside
       the box: 4,111 instances.

The corpus files are named by manifest index (`instance_{i:05d}.npz`), so
the names of a tail follow from the corpus size and the split alone
(`expected_names`), and those of a holdout set from them and the exclusions
(`holdout_names`); the adjudication recomputes them.

Labels. Each holdout instance is labelled once, by the sparse direct solve
of the free block that labelled the corpora (`solve_fe_displacement`, method
"direct": one factorisation, four load cases; a labels file cut off before
it was complete is solved again), into a labels file of its own under the
set's output directory, beside a manifest that records, per instance, the
SHA-256 of the corpus file it was solved from and of its labels file, and
the solve's relative residual. The corpus is read and never written: the corpus files
keep their bytes, so the corpus manifests that the earlier reports pin stay
valid.

Evaluation. Each stored state is checked by SHA-256 against the record that
the configuration copies from the run that trained it, and by content: on
the run's own validation split, its five per-instance metrics, recomputed
here with the frozen metric suite (`fejepa.metrics.evaluate_fields`),
against the report's arrays. Then every state predicts every holdout
instance, and the same suite gives the five metrics per instance, beside the
per-load relative energy gaps and energies Pi_h(u) (a load case is worse
than the zero field exactly when its gap exceeds 1, exactly when
Pi_h(u) > 0: the gap is computed from the energies, so the two agree by
construction). Nothing here trains, and nothing reads a verdict.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..metrics import FIELD_KEYS

SETS = ("h3d", "h2d")
ROWS = {"h3d": ("ar", "labels", "mgn"), "h2d": ("ar", "labels", "labels_knorm", "mgn")}
"""The stored states of each set, by the row names of the run that trained them."""
METRICS = FIELD_KEYS
"""The frozen suite's five per-instance metrics, in its order."""
HIGHER_BETTER = frozenset({"crit_recall"})
RESIDUAL_MAX = 1e-8
"""Largest relative residual ||K_ff u_f - F_f|| / ||F_f|| a holdout label may
have (a direct solve of these systems reaches about 1e-12 to 1e-10)."""
SPEC_KEYS = ("_comment", "prereg_file", "seeds", "out", "labels", "content_tol", "sets",
             "criteria")
SET_KEYS = ("corpus", "manifest_sha256", "n_corpus", "split", "used_prefix", "excluded",
            "n_holdout", "report", "report_sha256", "rows")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------- the spec --

def load_spec(path) -> dict:
    """The configuration (`configs/holdout_v1.json`), with its structure checked."""
    spec = json.loads(Path(path).read_text())
    if sorted(spec) != sorted(SPEC_KEYS):
        raise ValueError(f"{path}: top-level keys {sorted(spec)}, expected {sorted(SPEC_KEYS)}")
    if sorted(spec["sets"]) != sorted(SETS):
        raise ValueError(f"{path}: sets {sorted(spec['sets'])}, expected {sorted(SETS)}")
    for name, s in spec["sets"].items():
        if sorted(s) != sorted(SET_KEYS):
            raise ValueError(f"{path}: set {name} has keys {sorted(s)}")
        if list(s["rows"]) != list(ROWS[name]):
            raise ValueError(f"{path}: set {name} has rows {list(s['rows'])}, "
                             f"expected {list(ROWS[name])}")
        for row, r in s["rows"].items():
            if sorted(r["sha256"]) != sorted(str(x) for x in spec["seeds"]):
                raise ValueError(f"{path}: {name}.{row} records states for seeds "
                                 f"{sorted(r['sha256'])}, expected {spec['seeds']}")
        ex = s["excluded"]
        if not isinstance(ex, list) or not all(isinstance(x, str) for x in ex) \
                or ex != sorted(set(ex)):
            raise ValueError(f"{path}: set {name}: excluded must be a sorted list of distinct "
                             "file names")
        sp = s["split"]
        tail = set(expected_names(s["n_corpus"], sp["n_val"], sp["seed"], s["used_prefix"]))
        if not set(ex) <= tail:
            raise ValueError(f"{path}: set {name}: excluded names outside the tail "
                             f"{sorted(set(ex) - tail)[:3]}")
        if s["n_corpus"] - sp["n_val"] - s["used_prefix"] - len(ex) != s["n_holdout"]:
            raise ValueError(f"{path}: set {name}: n_corpus - n_val - used_prefix - "
                             "len(excluded) != n_holdout")
    lab = spec["labels"]
    if lab.get("method") != "direct" or float(lab.get("residual_max", -1)) != RESIDUAL_MAX:
        raise ValueError(f"{path}: labels must be direct solves with residual_max "
                         f"{RESIDUAL_MAX:g} (the code's), not {lab}")
    return spec


def spec_sha256(spec: dict) -> str:
    """The canonical SHA-256 of a configuration (as `fejepa prereg --stamp`)."""
    from ..report import config_sha256

    return config_sha256(spec)


# ------------------------------------------------------------- the sets --

def expected_names(n_corpus: int, n_val: int, seed: int, used_prefix: int) -> list:
    """The tail's file names, pool[used_prefix:] in pool order, from the
    corpus size and the split alone (corpus file i is `instance_{i:05d}.npz`
    and the split is `numpy.random.default_rng(seed).permutation(n_corpus)`)."""
    perm = np.random.default_rng(int(seed)).permutation(int(n_corpus))
    return [f"instance_{int(i):05d}.npz" for i in perm[int(n_val) + int(used_prefix):]]


def holdout_names(sdef: dict) -> list:
    """The holdout set's file names in pool order: the tail without the
    configuration's `excluded` names."""
    sp = sdef["split"]
    ex = set(sdef["excluded"])
    return [n for n in expected_names(sdef["n_corpus"], sp["n_val"], sp["seed"],
                                      sdef["used_prefix"]) if n not in ex]


def holdout_files(sdef: dict) -> list:
    """The holdout set's corpus files, in pool order: the corpus manifest's
    SHA-256 and size are checked, the tail's names against `expected_names`,
    and the excluded names are left out."""
    from ..data.archive import load_manifest, manifest_sha256
    from ..experiments.protocol import load_split

    d = Path(sdef["corpus"])
    got = manifest_sha256(d)
    if got != sdef["manifest_sha256"]:
        raise ValueError(f"{d}: manifest SHA-256 {got[:12]}..., the configuration names "
                         f"{sdef['manifest_sha256'][:12]}...")
    n = len(load_manifest(d)["instances"])
    if n != int(sdef["n_corpus"]):
        raise ValueError(f"{d}: {n} instances, the configuration names {sdef['n_corpus']}")
    sp = sdef["split"]
    files = load_split(str(d), int(sp["n_val"]), seed=int(sp["seed"])).pool_files
    files = files[int(sdef["used_prefix"]):]
    names = [Path(f).name for f in files]
    if names != expected_names(sdef["n_corpus"], sp["n_val"], sp["seed"], sdef["used_prefix"]):
        raise ValueError(f"{d}: the holdout files are not the split's (manifest order is not "
                         "index order)")
    ex = set(sdef["excluded"])
    files = [Path(f) for f in files if Path(f).name not in ex]
    if len(files) != int(sdef["n_holdout"]):
        raise ValueError(f"{d}: {len(files)} holdout instances, expected {sdef['n_holdout']}")
    return files


def corpus_flags(sdef: dict) -> dict:
    """{file name: the corpus manifest's `labelled` flag}."""
    from ..data.archive import load_manifest

    return {r["file"]: bool(r.get("labelled", False))
            for r in load_manifest(Path(sdef["corpus"]))["instances"]}


# ----------------------------------------------------------- labelling --

def relative_residual(arch, U: np.ndarray) -> float:
    """max over load cases of ||K_ff u_f - F_f|| / ||F_f||."""
    import scipy.sparse as sp

    free = np.asarray(arch.free_mask, dtype=bool).ravel()
    Kf = sp.csr_matrix(arch.K)[free][:, free]
    U2 = np.atleast_2d(np.asarray(U, dtype=np.float64))[:, free]
    F2 = np.atleast_2d(np.asarray(arch.F, dtype=np.float64))[:, free]
    R = (Kf @ U2.T).T - F2
    den = np.linalg.norm(F2, axis=1)
    den[den == 0] = 1.0
    return float(np.max(np.linalg.norm(R, axis=1) / den))


def _synced_replace(path: Path, write) -> None:
    """Write through `write(fh)` to a temporary file, flush it to the disk,
    and move it over `path`: a machine that stops mid-write leaves the
    previous complete file (or none), never a truncated one."""
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:                                       # the rename itself, where the system allows
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_json_synced(path, obj: dict) -> None:
    _synced_replace(Path(path), lambda fh: fh.write(json.dumps(obj, indent=1).encode()))


def save_npz_synced(path, **arrays) -> None:
    _synced_replace(Path(path), lambda fh: np.savez_compressed(fh, **arrays))


def _save_labels(path: Path, U: np.ndarray, source: str, source_sha: str) -> None:
    save_npz_synced(path, U_star=np.asarray(U, dtype=np.float64), source_file=np.array(source),
                    source_sha256=np.array(source_sha))


def _read_labels(path: Path) -> tuple:
    with np.load(path, allow_pickle=False) as d:
        return (np.asarray(d["U_star"], dtype=np.float64), str(d["source_file"]),
                str(d["source_sha256"]))


MANIFEST_VOLATILE = ("reused", "seconds", "ledger")
"""Labels-manifest fields that a labelling run that reuses every label
changes without changing a label."""


def label_set(name: str, sdef: dict, out_dir, log=print) -> dict:
    """Label the holdout set `name` into `out_dir` (`labels/` and
    `manifest.json`); a labels file already there is reused when it was
    solved from the same corpus file (same SHA-256) and its residual still
    passes (anything unreadable is solved again). A labelling that reuses
    every label and finds every record as the manifest already there holds
    it keeps that manifest unchanged, so that an evaluation already made
    from it stays valid. Returns the manifest and this attempt's own counts
    (`reused`, `solves`, whether the manifest was `kept`): a kept manifest
    records the attempt that wrote it."""
    from ..data.archive import load_instance
    from ..fe.solve import SolveLedger, solve_fe_displacement

    out = Path(out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    files = holdout_files(sdef)
    flags = corpus_flags(sdef)
    ledger, records, reused, t0 = SolveLedger(), [], 0, time.time()
    for i, f in enumerate(files):
        src_sha = sha256(f)
        arch = load_instance(f)
        lp = out / "labels" / f.name
        U = None
        if lp.exists():
            try:
                U0, src0, sha0 = _read_labels(lp)
                keep = (src0 == f.name and sha0 == src_sha and U0.shape == arch.F.shape
                        and relative_residual(arch, U0) <= RESIDUAL_MAX)
            except Exception:                           # noqa: BLE001 -- truncated: solve again
                keep = False
            if keep:
                U, reused = U0, reused + 1
        if U is None:
            U, _ = solve_fe_displacement(arch.K, arch.F, np.asarray(arch.free_mask, bool),
                                         method="direct", ledger=ledger,
                                         stage="holdout-labels")
            _save_labels(lp, U, f.name, src_sha)
        res = relative_residual(arch, U)
        if not res <= RESIDUAL_MAX:
            raise ValueError(f"{f.name}: relative residual {res:.3e} of its label exceeds "
                             f"{RESIDUAL_MAX:g}")
        rec = {"file": f.name, "source_sha256": src_sha, "labels_sha256": sha256(lp),
               "n_nodes": int(arch.n_nodes), "ndof": int(arch.ndof),
               "n_loads": int(arch.n_loads), "residual": res,
               "corpus_labelled_flag": flags.get(f.name), "source_labelled": arch.labelled}
        if arch.labelled:                    # a corpus file that carries labels already
            S = np.atleast_2d(np.asarray(arch.U_star, dtype=np.float64))
            rec["source_label_max_rel_dev"] = float(
                np.max(np.linalg.norm(S - U, axis=1)
                       / np.maximum(np.linalg.norm(U, axis=1), 1e-300)))
        records.append(rec)
        if (i + 1) % 64 == 0 or i + 1 == len(files):
            log(f"[holdout] {name} labels {i + 1}/{len(files)} | {time.time() - t0:.0f} s")
    sp = sdef["split"]
    man = {"what": f"PREREG_HOLDOUT: the labels of holdout set {name}", "set": name,
           "corpus": str(sdef["corpus"]), "corpus_manifest_sha256": sdef["manifest_sha256"],
           "n_corpus": int(sdef["n_corpus"]), "split": {"n_val": int(sp["n_val"]),
                                                       "seed": int(sp["seed"])},
           "used_prefix": int(sdef["used_prefix"]), "excluded": list(sdef["excluded"]),
           "n": len(records), "method": "direct",
           "residual_max": RESIDUAL_MAX, "reused": reused,
           "ledger": {"per_stage": ledger.as_dict()["per_stage"], "total": ledger.total},
           "seconds": round(time.time() - t0, 1), "instances": records}
    mp = out / "manifest.json"
    if reused == len(records) and mp.is_file():
        try:
            prev = json.loads(mp.read_text())
        except Exception:                               # noqa: BLE001 -- rewritten below
            prev = None
        if isinstance(prev, dict) and {k: v for k, v in prev.items()
                                       if k not in MANIFEST_VOLATILE} == \
                json.loads(json.dumps({k: v for k, v in man.items()
                                       if k not in MANIFEST_VOLATILE})):
            log(f"[holdout] {name} labels unchanged: {mp} kept as it was")
            return prev, {"reused": reused, "solves": int(ledger.total), "kept": True}
    write_json_synced(mp, man)
    return man, {"reused": reused, "solves": int(ledger.total), "kept": False}


def check_labelled_set(name: str, sdef: dict, set_dir) -> dict:
    """The set's labels manifest, refused unless it describes this set and
    names the expected instances in pool order."""
    man = json.loads((Path(set_dir) / "manifest.json").read_text())
    sp = sdef["split"]
    want = {"set": name, "corpus_manifest_sha256": sdef["manifest_sha256"],
            "n_corpus": int(sdef["n_corpus"]), "split": {"n_val": int(sp["n_val"]),
                                                         "seed": int(sp["seed"])},
            "used_prefix": int(sdef["used_prefix"]), "excluded": list(sdef["excluded"]),
            "method": "direct", "n": int(sdef["n_holdout"])}
    for k, v in want.items():
        if man.get(k) != v:
            raise ValueError(f"{set_dir}: the labels manifest's {k} is {man.get(k)!r}, "
                             f"expected {v!r}")
    names = [r["file"] for r in man["instances"]]
    if names != holdout_names(sdef):
        raise ValueError(f"{set_dir}: the labels manifest does not list the holdout set in "
                         "pool order")
    bad = [r["file"] for r in man["instances"] if not r["residual"] <= RESIDUAL_MAX]
    if bad:
        raise ValueError(f"{set_dir}: {len(bad)} labels above the residual bound "
                         f"(first {bad[0]})")
    return man


def holdout_instances(name: str, sdef: dict, set_dir):
    """The labelled holdout instances, in pool order, one at a time: each
    corpus file and labels file is checked against the labels manifest's
    SHA-256 before use."""
    from ..data.archive import load_instance

    man = check_labelled_set(name, sdef, set_dir)
    corpus = Path(sdef["corpus"])
    for r in man["instances"]:
        f, lp = corpus / r["file"], Path(set_dir) / "labels" / r["file"]
        if sha256(f) != r["source_sha256"]:
            raise ValueError(f"{f}: its bytes differ from those the label was solved from")
        if sha256(lp) != r["labels_sha256"]:
            raise ValueError(f"{lp}: its bytes differ from the labels manifest's record")
        U, src, src_sha = _read_labels(lp)
        if src != r["file"] or src_sha != r["source_sha256"]:
            raise ValueError(f"{lp}: names another corpus file")
        arch = load_instance(f)
        arch.U_star = U
        yield arch


# ---------------------------------------------------------------- states --

def state_path(sdef: dict, row: str, seed: int) -> Path:
    return Path(sdef["rows"][row]["state"].format(seed=int(seed)))


def check_states(sdef: dict, seeds) -> dict:
    """{(row, seed): path}, refused (ValueError) unless every state exists and
    has the SHA-256 the configuration records for it."""
    out, bad = {}, []
    for row, r in sdef["rows"].items():
        for s in seeds:
            p = state_path(sdef, row, s)
            want = r["sha256"][str(int(s))]
            if not p.is_file():
                bad.append(f"{p}: missing")
                continue
            got = sha256(p)
            if got != want:
                bad.append(f"{p}: SHA-256 {got[:12]}..., recorded {want[:12]}...")
            out[(row, int(s))] = p
    if bad:
        raise ValueError("states refused: " + "; ".join(bad))
    return out


def row_model(report: dict, sdef: dict, row: str, state, seed: int, dev: str):
    """The report's model of `row`, strictly loaded, in eval mode, under the
    run's TF32 policy (the graph network is the run's own `kind: mgn` build,
    as scripts/export_fields.py and scripts/cm2d_spectra.py build it)."""
    from ..report import config_sha256
    from ..runtime import setup_torch
    from .common import build_model_from_config
    from .posthoc import run_model

    if sdef["rows"][row].get("kind", "fejepa") != "mgn":
        return run_model(report, state, seed, dev)
    cfg = report["config"]
    if config_sha256(cfg) != report["provenance"]["config_sha256"]:
        raise ValueError("the report's embedded configuration does not hash to its recorded "
                         "config_sha256")
    setup_torch(dev, tf32=bool(cfg.get("tf32", True)))
    return build_model_from_config(dict(cfg["model"], kind="mgn"), state_path=str(state),
                                   seed=int(seed), device=dev)


def predict(model, arch, dev: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        u = model.forward_instance(model.prepare_instance(arch, dev))
    return u.detach().cpu().numpy()


def report_arrays(report: dict, sdef: dict, row: str, seed_index: int) -> dict:
    """The report's per-instance validation arrays of one row and seed."""
    cell = sdef["rows"][row]["cell"]
    cells = report["results"]["e8"]["metrics"]["cells"]
    return cells[row][str(cell)]["per_seed_eval"][seed_index]["per_instance"]


def content_check(got: dict, ref: dict, tol: float) -> dict:
    """Recomputed per-instance metrics against the report's: the median and
    largest relative deviation of each; the state passes when every
    recomputed value is finite and the medians of the displacement error and
    the relative energy gap deviate by at most `tol` (the rule of
    scripts/export_fields.py and scripts/cm2d_spectra.py, with finiteness
    added)."""
    out = {"n": len(got[METRICS[0]])}
    finite = True
    for k in METRICS:
        g = np.asarray(got[k], dtype=float)
        r = np.asarray(ref[k], dtype=float)
        finite = finite and bool(g.size) and bool(np.all(np.isfinite(g)))
        if g.shape != r.shape:
            out[f"{k}_median_rel_dev"] = out[f"{k}_max_rel_dev"] = float("inf")
            finite = False
            continue
        dev = np.abs(g - r) / np.maximum(np.abs(r), 1e-30)
        dev[~(np.isfinite(g) & np.isfinite(r))] = np.inf
        out[f"{k}_median_rel_dev"] = float(np.median(dev))
        out[f"{k}_max_rel_dev"] = float(np.max(dev))
    out["finite"] = bool(finite)
    out["ok"] = bool(finite and all(out[f"{k}_median_rel_dev"] <= tol
                                    for k in ("disp_rel_l2", "energy_gap_rel")))
    return out


# ------------------------------------------------------------ evaluation --

def instance_record(U: np.ndarray, arch) -> dict:
    """One prediction of one instance: the frozen suite's five values (means
    over the load cases), and per load case the relative energy gap and
    Pi_h(u)."""
    from ..anchor.energy import pi_h
    from ..metrics import energy_gap_rel, evaluate_fields

    vals = evaluate_fields(U, arch)
    return {"metrics": [float(vals[k]) for k in METRICS],
            "gap_load": np.asarray(energy_gap_rel(U, arch), dtype=float),
            "pi_load": np.asarray(pi_h(U, arch.K, arch.F), dtype=float)}


def worse_than_zero(gap_load: np.ndarray, pi_load: np.ndarray, tol: float = 1e-9) -> dict:
    """Load cases worse than the zero field (relative gap above 1) and the
    check that exactly these have Pi_h(u) > 0 (the manuscript's Corollary
    "Zero-field test", Sec. 3.5); load cases within `tol` of a gap of 1 are left
    out of the check.
    The gap is computed from the energies (`fejepa.metrics.energy_gap_rel`:
    g = (Pi_h(u) - Pi_h(U*)) / |Pi_h(U*)|, so g - 1 = Pi_h(u) / |Pi_h(U*)|),
    and the check holds by construction: it guards the bookkeeping, and
    the count is the information."""
    g = np.asarray(gap_load, dtype=float)
    p = np.asarray(pi_load, dtype=float)
    fin = np.isfinite(g) & np.isfinite(p)
    away = fin & (np.abs(g - 1.0) > tol)
    return {"load_cases": int(g.size), "finite": int(fin.sum()),
            "worse": int(np.sum(fin & (g > 1.0))),
            "sign_mismatch": int(np.sum(away & ((g > 1.0) != (p > 0.0))))}
