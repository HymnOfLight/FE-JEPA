"""Audit of gate G-C1.3 from its return package (after the run; not under the stamp).

    python -I audit/c13_recompute.py RECORDS SCRATCH OUT.json [BLAS_THREADS]

RECORDS is the return package's records/ folder and SCRATCH an empty folder for the regenerated instances. Run it
from the joint/ folder, with c13/ and fejoint/ as stamped (docs/G-C1.3_code.sha256). BLAS_THREADS (default 25, as on the GPU host) sets the BLAS threads of numpy
and scipy for the whole audit: the thread count changes the last bits of a few entries of the condensed stiffness,
so the files reproduce the host's bytes only with a thread count like the host's. The script
  1. regenerates the held-out and tested instances with the stamped code and compares their parameters, sizes and
     file hashes with the returned manifest;
  2. checks the returned labels against the ledger, and the ledger against the manifest;
  3. re-solves seven of the instances and compares the solutions with the returned labels. The host had no
     pypardiso, so its labels were solved with scipy's SuperLU; FEJOINT_SOLVER=legacy makes the re-solves do the
     same where pypardiso loads;
  4. recomputes every metric of the three label-free runs from the returned predictions and labels on the
     regenerated instances, compares each with the returned verdict, and recomputes C1, C2 and the verdict.
"""
import hashlib
import json
import pathlib
import platform
import sys
from dataclasses import asdict

import numpy as np
import scipy
import threadpoolctl
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from c13.config import config, sha256            # noqa: E402
from c13 import data as D                         # noqa: E402
from c13.evaluate import metrics, _summary        # noqa: E402
from fejoint.export import solve_exported         # noqa: E402

STAMPED_CONFIG = "727d9bde73e5129a3ea9d26b3d6f3ac93d1f7f19505cd9605000c601152bb11e"
RESOLVE = ["eval_0000", "eval_0031", "eval_0064", "eval_0097", "eval_0127", "tested_FS1", "tested_FS4"]


def main(rec: pathlib.Path, scratch: pathlib.Path, out: pathlib.Path, blas_threads: int = 25) -> None:
    with threadpool_limits(limits=blas_threads, user_api="blas"):
        pools = [(x.get("internal_api"), x.get("num_threads")) for x in threadpool_info() if x.get("user_api") == "blas"]
        audit(rec, scratch, out, blas_threads, pools)


def audit(rec: pathlib.Path, scratch: pathlib.Path, out: pathlib.Path, blas_threads: int, pools: list) -> None:
    cfg = config()
    assert sha256(cfg) == STAMPED_CONFIG, "not the stamped configuration"
    box_man = json.loads((rec / "manifest.json").read_text())
    assert box_man["config_sha256"] == STAMPED_CONFIG
    box = {r["id"]: r for r in box_man["instances"]}
    res = {"config_sha256": STAMPED_CONFIG,
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "threadpoolctl": threadpoolctl.__version__, "resolve_solver": D._solver(),
                           "blas_threads": blas_threads, "blas_pools": pools}}

    # 1. regeneration
    D.generate(cfg, scratch, splits=("eval", "tested"))
    mine = {r["id"]: r for r in D.manifest(scratch)["instances"]}
    draws = {iid: asdict(q) for iid, _, q in D.members(cfg)}
    tr = {json.dumps(box[i]["params"], sort_keys=True) for i in box if box[i]["split"] == "train"}
    ev = {json.dumps(box[i]["params"], sort_keys=True) for i in box if box[i]["split"] == "eval"}
    res["regeneration"] = {
        "blas_threads": blas_threads, "blas_pools": pools,
        "instances": len(mine),
        "splits_returned": {s: sum(1 for r in box.values() if r["split"] == s) for s in ("train", "eval", "tested")},
        "every_returned_member_is_the_configured_draw": all(box[i]["params"] == draws[i] for i in box),
        "train_eval_overlap": len(tr & ev),
        "parameters_equal": all(mine[i]["params"] == box[i]["params"] for i in mine),
        "nodes_and_nnz_equal": all(mine[i]["nodes"] == box[i]["nodes"] and mine[i]["nnz"] == box[i]["nnz"] for i in mine),
        "file_sha256_equal": sum(mine[i]["sha256"] == box[i]["sha256"] for i in mine),
    }

    # 2. labels against ledger and manifest
    led = {e["id"]: e for e in json.loads((rec / "ledger.json").read_text())}
    res["labels"] = {
        "ledger_entries": len(led),
        "returned_label_files": len(list((rec / "labels").glob("*.npz"))),
        "returned_labels_match_ledger": all(
            hashlib.sha256((rec / "labels" / f"{i}.npz").read_bytes()).hexdigest() == led[i]["label_sha256"] for i in mine),
        "ledger_instances_match_manifest": all(led[i]["inst_sha256"] == box[i]["sha256"] for i in box),
        "solvers": sorted({str(e.get("solver")) for e in led.values()}),
    }

    def label(i):
        with np.load(rec / "labels" / f"{i}.npz") as z:
            return {"u": z["u"], "Pi": float(z["Pi"]), "S_paper": float(z["S_paper"])}

    exs = {i: D.load_export(scratch / "inst" / f"{i}.npz")[0] for i in mine}

    # 3. re-solves
    rs = {}
    for i in RESOLVE:
        s, lab = solve_exported(exs[i]), label(i)
        rs[i] = {"bit_identical": bool(np.array_equal(s["u"], lab["u"]) and s["Pi"] == lab["Pi"]
                                       and s["S_paper"] == lab["S_paper"]),
                 "iterations": s["iterations"], "ledger_iterations": led[i]["iterations"],
                 "max_du_over_max_u": float(np.abs(s["u"] - lab["u"]).max() / np.abs(lab["u"]).max()),
                 "rel_dPi": abs(s["Pi"] / lab["Pi"] - 1), "rel_dS": abs(s["S_paper"] / lab["S_paper"] - 1)}
    res["resolves"] = rs

    # 4. metrics of the label-free arm
    verdict = json.loads((rec / "verdict.json").read_text())
    worst, seeds = {}, {}
    for s in cfg["arms"]["label_free"]["seeds"]:
        s = str(s)
        with np.load(rec / "runs" / f"label_free_s{s}" / "preds.npz") as z:
            keys = sorted(z.files)
            assert keys == sorted(mine), "prediction keys differ from the held-out and tested members"
            preds = {i: z[i] for i in keys}
        rows = {i: metrics(exs[i], preds[i], label(i)) for i in keys}
        theirs = dict(verdict["arms"]["label_free"][s]["eval_rows"])
        theirs.update(verdict["arms"]["label_free"][s]["tested"])
        for i, m in rows.items():
            for k, x in m.items():
                y = float(theirs[i][k])
                worst[k] = max(worst.get(k, 0.0), abs(float(x) - y) / max(abs(y), 1e-300))
        summ = _summary([rows[i] for i in keys if mine[i]["split"] == "eval"])
        nn = np.concatenate([np.asarray(preds[i], dtype=float)[exs[i]["nonneg"]] for i in keys])
        seeds[s] = {"abs_S_err_median": summ["abs_S_err"]["median"], "iou_median": summ["iou"]["median"],
                    "min_nonneg": float(nn.min()), "nonneg_exactly_zero_share": float((nn == 0).mean()),
                    "all_finite": bool(all(np.isfinite(preds[i]).all() for i in keys)),
                    "dtype": str(preds[keys[0]].dtype)}
    C1 = float(np.median([v["abs_S_err_median"] for v in seeds.values()]))
    C2 = float(np.median([v["iou_median"] for v in seeds.values()]))
    res["label_free"] = {"per_seed": seeds, "largest_relative_difference_to_returned_metrics": worst,
                         "C1_value": C1, "C2_value": C2, "C1": C1 <= 0.05, "C2": C2 >= 0.9,
                         "verdict": "PASS" if (C1 <= 0.05 and C2 >= 0.9) else "FAIL",
                         "returned": {k: verdict[k] for k in ("verdict", "C1_value", "C2_value")}}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("environment", "regeneration", "labels", "resolves")}, indent=1))
    print(json.dumps(res["label_free"], indent=1))


if __name__ == "__main__":
    main(*map(pathlib.Path, sys.argv[1:4]), *(int(x) for x in sys.argv[4:5]))
