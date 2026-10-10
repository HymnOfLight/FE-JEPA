"""Instances of gate G-C1.3: generate the exports (no solve), label them (solves, after the stamp), load them.

Layout of a data directory:
    manifest.json             one record per instance: id, split, family parameters, file SHA-256, sizes
    inst/<id>.npz             the export of fejoint.export (the condensed nodal problem), nothing solved; it also
                              stores the family parameters and the configuration hash it was made under
    labels/<id>.npz           the exact solution of that problem (written by `label` only)
    labels/ledger.json        every solve: id, time, iterations, optimality residuals, the SHA-256 of the instance
                              file solved and of the label file written
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fejoint.family import sample, geometry, tested_details, FamilyParams
from fejoint.export import export, solve_exported


def members(cfg: dict) -> list[tuple[str, str, FamilyParams]]:
    """(id, split, parameters) for every instance, in a fixed order."""
    d = cfg["data"]
    for s in (d["train_seed"], d["eval_seed"]):
        assert s not in d["seeds_seen_before_stamp"], "a seed used before the stamp"
    tr, _ = sample(d["n_train"], seed=d["train_seed"])
    ev, _ = sample(d["n_eval"], seed=d["eval_seed"])
    out = [(f"train_{i:04d}", "train", q) for i, q in enumerate(tr)]
    out += [(f"eval_{i:04d}", "eval", q) for i, q in enumerate(ev)]
    td = tested_details()
    out += [(f"tested_{n}", "tested", td[n]) for n in d["tested"]]
    return out


def export_member(cfg: dict, q: FamilyParams) -> dict:
    d = cfg["data"]
    return export(geometry(q), P_full=d["P_full_N"], k_rot_factor=d["k_rot_factor"], reg=d["reg"],
                  beam_segment=d["beam_segment_mm"], **d["mesh"])


def _pack(ex: dict, q: FamilyParams, cfg_sha: str) -> dict:
    K = ex["K"].tocsr()
    pat = ex["patches"]
    r = ex["reading"]
    out = dict(K_data=K.data, K_indices=K.indices.astype(np.int64), K_indptr=K.indptr.astype(np.int64),
               K_shape=np.asarray(K.shape, np.int64), F=ex["F"], dirichlet=ex["dirichlet"], nonneg=ex["nonneg"],
               node_flags=ex["node_flags"], nodes=ex["nodes"], hexes=ex["hexes"].astype(np.int64),
               part=ex["part"].astype(np.int64), heads_from_x=ex["heads_from_x"],
               patch_nodes=np.concatenate(pat).astype(np.int64),
               patch_offsets=np.cumsum([0] + [len(p_) for p_ in pat]).astype(np.int64),
               const=np.float64(ex["const"]), u_scale=np.float64(ex["u_scale"]),
               reading_kind=np.bytes_(r["kind"]),
               reading_consts=np.array([r["a1"], r["L1"], r["d_eb1"], r["M_face"]]),
               meta_json=np.frombuffer(json.dumps(ex["meta"]).encode(), np.uint8),
               params_json=np.frombuffer(json.dumps(asdict(q)).encode(), np.uint8),
               config_sha256=np.bytes_(cfg_sha))
    if r["kind"] == "segment":
        m = ex["masters"]
        out.update(reading_segment=np.array([r["sd"], r["beyond"]]), m0=m["m0"], m_from_x=m["m_from_x"],
                   f_m=m["f_m"], y_ref=np.float64(m["y_ref"]), section=m["section"].astype(np.int64))
    else:
        out.update(dt1_nodes=r["dt1_nodes"].astype(np.int64), dt1_weights=r["dt1_weights"])
    return out


def load_export(path) -> tuple[dict, FamilyParams]:
    """The export dict of fejoint.export (the keys solve_exported, energy, stiffness and physical_displacement use)
    and the family parameters."""
    with np.load(path, allow_pickle=False) as z:
        K = sp.csr_matrix((z["K_data"], z["K_indices"], z["K_indptr"]), shape=tuple(int(s) for s in z["K_shape"]))
        offs = z["patch_offsets"]
        pn = z["patch_nodes"]
        a1, L1, d_eb1, M_face = (float(v) for v in z["reading_consts"])
        kind = bytes(z["reading_kind"]).decode()
        reading = {"kind": kind, "a1": a1, "L1": L1, "d_eb1": d_eb1, "M_face": M_face}
        masters = None
        if kind == "segment":
            sd, beyond = (float(v) for v in z["reading_segment"])
            reading.update(sd=sd, beyond=beyond)
            masters = {"m0": z["m0"], "m_from_x": z["m_from_x"], "f_m": z["f_m"], "y_ref": float(z["y_ref"]),
                       "section": z["section"]}
        else:
            reading.update(dt1_nodes=z["dt1_nodes"], dt1_weights=z["dt1_weights"])
        ex = {"K": K, "F": z["F"], "dirichlet": z["dirichlet"], "nonneg": z["nonneg"], "node_flags": z["node_flags"],
              "nodes": z["nodes"], "hexes": z["hexes"], "part": z["part"], "heads_from_x": z["heads_from_x"],
              "patches": [pn[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)], "const": float(z["const"]),
              "u_scale": float(z["u_scale"]), "reading": reading, "masters": masters,
              "meta": json.loads(bytes(z["meta_json"].tobytes()).decode())}
        q = FamilyParams(**json.loads(bytes(z["params_json"].tobytes()).decode()))
    return ex, q


def stored_identity(path) -> tuple[dict, str]:
    """The family parameters and the configuration hash an instance file was made under."""
    with np.load(path, allow_pickle=False) as z:
        return (json.loads(bytes(z["params_json"].tobytes()).decode()),
                bytes(z["config_sha256"]).decode() if "config_sha256" in z.files else "")


def _write_json_atomic(p: Path, obj) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(cfg: dict, data_dir, splits=("train", "eval", "tested"), limit: int | None = None) -> Path:
    """Export every member of the given splits and write the manifest. Resumes: a file already written is kept if
    it was made for the same member under the same configuration, and refused otherwise."""
    from .config import sha256

    cfg_sha = sha256(cfg)
    data_dir = Path(data_dir)
    (data_dir / "inst").mkdir(parents=True, exist_ok=True)
    recs = []
    todo = [m for m in members(cfg) if m[1] in splits]
    if limit is not None:
        per = {}
        todo = [m for m in todo if per.setdefault(m[1], []).append(m) or len(per[m[1]]) <= limit]
    for iid, split, q in todo:
        p = data_dir / "inst" / f"{iid}.npz"
        if p.exists():
            params, made_under = stored_identity(p)
            if params != asdict(q) or made_under != cfg_sha:
                raise RuntimeError(f"{p} was made for another member or configuration; use a fresh directory")
        else:
            ex = export_member(cfg, q)
            tmp = p.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, **_pack(ex, q, cfg_sha))
            tmp.replace(p)
        with np.load(p) as z:
            n_nodes, nnz = int(z["nodes"].shape[0]), int(z["K_data"].size)
        recs.append({"id": iid, "split": split, "params": asdict(q), "file": f"inst/{iid}.npz",
                     "sha256": _sha256_file(p), "nodes": n_nodes, "nnz": nnz})
    _write_json_atomic(data_dir / "manifest.json", {"config_sha256": cfg_sha, "n": len(recs), "instances": recs})
    return data_dir / "manifest.json"


def label(data_dir, ids) -> None:
    """Solve the listed instances (resumes) and record each solve in the ledger, with the SHA-256 of the instance
    file solved (checked against the manifest first) and of the label file written."""
    data_dir = Path(data_dir)
    (data_dir / "labels").mkdir(exist_ok=True)
    rec = {r["id"]: r for r in manifest(data_dir)["instances"]}
    led_p = data_dir / "labels" / "ledger.json"
    ledger = json.loads(led_p.read_text()) if led_p.exists() else []
    done = {e["id"] for e in ledger}
    for iid in ids:
        out = data_dir / "labels" / f"{iid}.npz"
        if iid in done and out.exists():
            continue
        inst = data_dir / "inst" / f"{iid}.npz"
        inst_sha = _sha256_file(inst)
        if inst_sha != rec[iid]["sha256"]:
            raise RuntimeError(f"{inst} differs from the manifest")
        ex, _ = load_export(inst)
        t0 = time.perf_counter()
        s = solve_exported(ex)
        tmp = out.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, u=s["u"], Pi=np.float64(s["Pi"]), S_paper=np.float64(s["S_paper"]))
        tmp.replace(out)
        ledger = [e for e in ledger if e["id"] != iid]
        ledger.append({"id": iid, "seconds": round(time.perf_counter() - t0, 2), "iterations": s["iterations"],
                       "kkt": s["kkt"], "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "solver": _solver(),
                       "inst_sha256": inst_sha, "label_sha256": _sha256_file(out)})
        _write_json_atomic(led_p, ledger)


def _solver() -> str:
    """The sparse solver of the label solves: PARDISO when pypardiso loads, otherwise scipy (fejoint.linsolve)."""
    from fejoint import linsolve
    return "pardiso" if linsolve._PARDISO is not None else "scipy"


def check_files(data_dir, ids) -> None:
    """Every listed instance file matches the manifest and every label file matches the ledger."""
    data_dir = Path(data_dir)
    rec = {r["id"]: r for r in manifest(data_dir)["instances"]}
    led = {e["id"]: e for e in json.loads((data_dir / "labels" / "ledger.json").read_text())}
    for iid in ids:
        if _sha256_file(data_dir / "inst" / f"{iid}.npz") != rec[iid]["sha256"]:
            raise RuntimeError(f"instance {iid} differs from the manifest")
        e = led[iid]
        lab = _sha256_file(data_dir / "labels" / f"{iid}.npz")
        if e["inst_sha256"] != rec[iid]["sha256"] or lab != e["label_sha256"]:
            raise RuntimeError(f"label {iid} differs from the ledger")


def load_label(data_dir, iid) -> dict:
    with np.load(Path(data_dir) / "labels" / f"{iid}.npz") as z:
        return {"u": z["u"], "Pi": float(z["Pi"]), "S_paper": float(z["S_paper"])}


def manifest(data_dir) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


KKT_LIMITS = {"min_gap": (">=", -1e-9), "min_lambda": (">=", -1e-9), "max_complementarity": ("<=", 1e-9),
              "max_lambda_inactive": ("<=", 1e-6), "max_residual_unconstrained": ("<=", 1e-6)}


def check_labels(cfg: dict, data_dir) -> dict:
    """Every member of the manifest labelled once, its instance and label files matching the manifest and the ledger,
    and the worst optimality residual of each kind within KKT_LIMITS (relative to the largest load and displacement
    components, as fejoint.contact reports them)."""
    from .config import sha256

    man = manifest(data_dir)
    ids = [r["id"] for r in man["instances"]]
    led = json.loads((Path(data_dir) / "labels" / "ledger.json").read_text())
    check_files(data_dir, ids)
    worst = {}
    for k, (op, lim) in KKT_LIMITS.items():
        v = [e["kkt"][k] for e in led]
        worst[k] = min(v) if op == ">=" else max(v)
    ok_kkt = {k: (worst[k] >= lim if op == ">=" else worst[k] <= lim) for k, (op, lim) in KKT_LIMITS.items()}
    ok = (man["config_sha256"] == sha256(cfg) and len(led) == len(ids) == man["n"]
          and sorted(e["id"] for e in led) == sorted(ids) and all(ok_kkt.values()))
    return {"labels": len(led), "instances": len(ids), "worst": worst, "within_limits": ok_kkt, "ok": bool(ok)}
