#!/usr/bin/env python3
"""cmame-paper Stage 11 (PREREG_HOLDOUT Sec. 2): the instances of the two
corpora's tails that another recorded set drew again, that a recorded tool
rebuilt, or whose own file is held outside the box (an archive of a whole
corpus kept as a backup does not count): the holdout sets leave them out
(configs/holdout_v1.json `excluded`).

Three dimensions. The gmsh generator draws every instance of a set from one
sequential stream, `numpy.random.default_rng(seed)`: per instance the mesh
size when the set has a band of them, then the geometry and material
(`fejepa.fe.gmsh3d.sample_params3d`), then the four load scales; meshing
draws nothing (`generate_gmsh3d_dataset`, `gmsh3d_instance`). Two sets of one
seed that both draw a mesh size per instance therefore hold the same
instances, index by index, whatever their bands (the mesh-size draw is one
double either way); a set of that seed without the draw meets them now and
then: where an instance of one starts at the stream position at which an
instance of the other starts, the two are the same geometry, material and
loads, meshed at their own sizes. Three recorded sets share the in-band
corpus's seed 0: the fine-mesh set of the Phase-2 runs (one mesh size), and
the cross-branch regression's miniature of `configs/phase2b_v1.json`
(`scripts/regress_against_branch.py`, `miniature`: 80 instances with a band,
in-band instances 0 to 79 meshed coarsely, and 7 at one mesh size, fine
instances 0 to 6), which runs in this project's sandbox. The sandbox's other
seed-0 sets (`SANDBOX`, from `scripts/holdout_scan_sets.py`) are shorter
prefixes of the same two streams. This
script replays the draws of every recorded three-dimensional set, without
meshing; checks the replay against the instances the records hold (the
Poisson ratio, the box and the four load scales of the field export's three
figure instances, two in-band and one fine); and lists every instance of one
set equal to an instance of another (geometry, material and load scales, bit
for bit), and the pairs that share only the box and the Poisson ratio
(different problems, listed for disclosure). The lc calibration's five
geometries (`scripts/g1_lc_calibration.py`: children of
`SeedSequence(17)`, parameters only) are compared by their geometry.

Two dimensions. The 2D generator draws instance i of a set from child i of
`numpy.random.SeedSequence(seed).spawn(n)`, so two sets of one seed share
instances by index, and sets of different seeds share none (the July
cross-resolution sets drew from seed 7919, `runner._ensure_multires`). Other
uses of seed 0, the seed of E1's corpus runs/data2d: `scripts/w9_scale_factor.py`
rebuilt instances 0 to 255 to measure load scales (no label, no network), and
the regression's miniature of `configs/e1_2d_base.json` drew instances 0 to
79 (labelled 3, trained label-free networks on 64; sandbox); the sandbox's
other seed-0 sets are shorter. Copies of 21 corpus files are held outside
the box (`COPIES_2D`). The tail instances among all of these are left out of
h2d. Archives of the whole corpus kept as backups (`BACKUPS_2D`), from which
no file is recorded as extracted outside the box, leave nothing out.

Bounds (`BOUNDS`). A set of either seed of at most 80 instances with a band
of mesh sizes, 320 at one size or 256 in two dimensions shares with the
tails only instances left out here: the script checks this, so that sets
drawn again within the bounds (the regression tool draws its miniatures
whenever it runs) leave the holdout sets as they are.

    python scripts/holdout_overlap.py --config configs/holdout_v1.json \\
        --out records/cmame/holdout/overlap.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BAND = (0.0579, 0.0906)
CORPUS_3D, FINE_3D = "runs/data3d_phase2", "runs/data3d_phase2_fine"
MINI_3D, MINI_FINE_3D = "regression miniature (data)", "regression miniature (data_transfer)"
SETS_3D = (
    # name, seed, n, mesh-size band (None: one size, no draw), where the generation is recorded
    (CORPUS_3D, 0, 2000, BAND, "configs/e2_m1024.json: data"),
    (FINE_3D, 0, 320, None, "configs/e2_m1024.json: data_transfer"),
    (MINI_3D, 0, 80, (0.30, 0.36),
     "scripts/regress_against_branch.py miniature() of configs/phase2b_v1.json (sandbox; "
     "records/cmame/README.md, records/wp9/README.md)"),
    (MINI_FINE_3D, 0, 7, None, "scripts/regress_against_branch.py miniature(): data_transfer"),
    ("runs/phase2/pilot_bf16_data", 123, 128, BAND, "scripts/pilot_bf16_transparency.py"),
    ("runs/phase2/bench_data/one_*", 999, 1, None, "scripts/bench_phase2_preconditions.py"),
    ("runs/data3d_g1pilot", 13, 24, None, "configs/wp7_e4_3d_pilot.json: data"),
    ("runs/data3d_g1pilot_c2p0", 13 + 7919, 8, None,
     "configs/wp7_e4_3d_pilot.json: e4 (runner._ensure_multires)"),
) + tuple((f"posthoc_amplitude geometry {g}", 90000 + g, 1, None,
           "scripts/posthoc_amplitude.py --remesh 16 (RUNBOOK_E_SERIES 4e)") for g in range(16))
CALIBRATION_3D = ("lc calibration geometries", 17, 5,
                  "scripts/g1_lc_calibration.py (WP7 3D-G1.1; seed 17, 5 samples)")
SANDBOX = {"scanned": "14 October 2026, 05:37 UTC",
           "tool": "scripts/holdout_scan_sets.py /home/claude /tmp /mnt/user-data",
           "3d_band": {"seed": 0, "sets": 45, "longest": 80,
                       "what": "the regression miniatures (n 16 at wp8 Stage 1.36, 80 since), "
                               "rehearsals of this project's box procedures and reviewers' "
                               "reproductions (n 6 to 40): prefixes of the in-band stream; two "
                               "reproductions at the corpus's own band of mesh sizes (in-band "
                               "instances 0 to 5, 29 September 2026; 0 to 9, 14 October 2026, "
                               "deleted before the scan) are copies of those instances"},
           "3d_one_size": {"seed": 0, "sets": 44, "longest": 20,
                           "what": "the miniatures' and rehearsals' fine sets (n 4 to 20): "
                                   "prefixes of the fine-mesh stream"},
           "2d": {"seed": 0, "sets": 29, "longest": 80,
                  "what": "the regression miniatures (n 80) and rehearsals of the box "
                          "procedures (n 12 to 40; some labelled, networks evaluated on "
                          "some): instances 0 to 79 or fewer"},
           "copies": "two copies of the corpus manifest of runs/data2d, from the authors' "
                     "archives, without instance files; its files 0 to 10, uploaded, and a "
                     "copy of them (COPIES_2D)"}
"""The sets of the corpora's generators (gmsh, gmsh3d) and seed in this
project's sandbox (`scripts/holdout_scan_sets.py` over the directories named,
at the time given): every one a prefix within the bounds below."""
BOUNDS = {"3d_band": 80, "3d_one_size": 320, "2d": 256}
"""The longest seed-0 sets, per kind, within which every tail instance a set
shares is left out (checked by `overlap`)."""
FIGURES = (("fig5", CORPUS_3D), ("fig6", CORPUS_3D), ("fig7", FINE_3D))
COPIES_2D = {"archive": "outputs_fejepa.zip", "made": "15-17 July 2026",
             "sha256": "4e2150dedd3e51d47112cc61410c8f287fe9fcfdb3143cad7f6d0bef0c50e33c",
             "corpus_manifest_sha256":
                 "3553396d183115b843512a35cba1a1e8448b9e83e10c1d1a5b02b235a7c79d26",
             "instances": list(range(0, 11)) + list(range(29990, 30000))}
"""Files of runs/data2d held outside the box: the authors' archive of the July
2026 outputs holds the corpus manifest and these 21 instance files (and the
July cross-resolution pairs, of seed 7919); copies of instances 0 to 10 have
been read outside the box since."""
BACKUPS_2D = (
    {"archive": "archive_data2d_full.tar.gz",
     "sha256": "5080d9d7ca50573507458cce65e46253379bcb622fb9184720182084b710a78b",
     "holds": "the whole corpus", "where": "on the box's data disk, and an off-box copy",
     "record": "records/wp8/restore_data2d.log; PROJECT_STATUS of 6 August 2026"},
    {"archive": "the labelled subset", "sha256": "b06d4320",
     "holds": "the 1,280 labelled instances: validation and pool[:1024]", "where": "off-box",
     "record": "PROJECT_STATUS of 6 August 2026"})
"""Archives of runs/data2d kept as backups: no file is recorded as extracted
from them outside the box, and the off-box subset holds no tail instance."""
REGRESS_2D = {"script": "scripts/regress_against_branch.py --config configs/e1_2d_base.json",
              "n": 80, "split": {"n_val": 3, "seed": 1}, "pool": 64,
              "use": "the miniature's 3 validation instances labelled; label-free networks "
                     "trained on its pool[:64] (sandbox)"}


def replay_3d(seed: int, n: int, band=None) -> list:
    """The draws of `generate_gmsh3d_dataset(out, n, seed, lc_range=band)`
    (or a fixed `lc` when `band` is None), without meshing: per instance the
    mesh size drawn (or None), the parameters and the four load scales."""
    from fejepa.fe.gmsh3d import sample_params3d

    rng = np.random.default_rng(int(seed))
    out = []
    for _ in range(int(n)):
        lc = float(rng.uniform(*band)) if band else None
        p = sample_params3d(rng)
        ts = 0.05 * rng.uniform(0.5, 1.5, size=4)
        out.append({"lc": lc, "params": p, "load_scales": [float(x) for x in ts]})
    return out


def replay_children(seed: int, n: int) -> list:
    """The lc calibration's geometries: `sample_params3d` of child i of
    `SeedSequence(seed).spawn(n)` (no load scales)."""
    from fejepa.fe.gmsh3d import sample_params3d

    return [{"lc": None, "params": sample_params3d(np.random.default_rng(c)),
             "load_scales": None} for c in np.random.SeedSequence(int(seed)).spawn(int(n))]


def key(inst: dict) -> tuple:
    """Geometry, material and load scales, bit for bit."""
    return geometry(inst) + (tuple(inst["load_scales"] or ()),)


def geometry(inst: dict) -> tuple:
    p = inst["params"]
    return (p["width"], p["height"], p["depth"], p["nu"],
            tuple(tuple(h) for h in p["holes"]))


def recorded_scales(z) -> dict:
    """The box, Poisson ratio and four load scales of a recorded instance
    (F: tractions on the faces x = w (two) and y = h, gravity over the mesh)."""
    nodes, F = np.asarray(z["nodes"]), np.asarray(z["F"])
    meta = json.loads(str(z["meta"]))
    w, h, d = (float(x) for x in nodes.max(axis=0))
    vol = float(np.asarray(z["vol"]).sum())
    ts = [-F[0][1::3].sum() / (h * d), F[1][0::3].sum() / (h * d),
          F[2][0::3].sum() / (w * d), -F[3][1::3].sum() / vol]
    return {"file": meta["file"], "box": [w, h, d], "nu": float(meta["material"]["nu"]),
            "load_scales": [float(x) for x in ts]}


def roles_3d(sdef: dict, name: str, n: int) -> list:
    """Each instance's role: in the in-band corpus by the split (validation,
    the pool prefix used, the tail), in the fine set by the pinned partition
    (`runner`: the first 256 evaluate, the next 64 few-shot), in the
    regression's miniature by its own split (n_val 3, seed 1; labels on its
    pool[:64], `miniature`)."""
    if name == sdef["corpus"]:
        sp = sdef["split"]
        perm = np.random.default_rng(int(sp["seed"])).permutation(int(sdef["n_corpus"]))
        role = [None] * n
        nv, used = int(sp["n_val"]), int(sdef["used_prefix"])
        for r, i in enumerate(perm):
            role[int(i)] = ("validation" if r < nv else
                            "training pool" if r < nv + used else "tail")
        return role
    if name == FINE_3D:
        return ["fine evaluation" if i < 256 else "fine few-shot" for i in range(n)]
    if name == MINI_3D:
        perm = np.random.default_rng(1).permutation(n)
        role = [None] * n
        for r, i in enumerate(perm):
            role[int(i)] = ("miniature validation, labelled" if r < 3 else
                            "miniature pool, labelled and trained on" if r < 3 + 64 else
                            "miniature pool, drawn only")
        return role
    if name == MINI_FINE_3D:
        return ["miniature fine evaluation" if i < 3 else "miniature fine few-shot"
                for i in range(n)]
    return ["-"] * n


def _pairs(index: dict, sets: dict, what) -> list:
    out = []
    for where in index.values():
        if len(where) < 2:
            continue
        for a in range(len(where)):
            for b in range(a + 1, len(where)):
                (sa, i), (sb, j) = where[a], where[b]
                if what(sets[sa]["instances"][i], sets[sb]["instances"][j]):
                    out.append({"a": sa, "i": i, "a_role": sets[sa]["roles"][i],
                                "b": sb, "j": j, "b_role": sets[sb]["roles"][j],
                                "lc_a": sets[sa]["instances"][i]["lc"],
                                "lc_b": sets[sb]["instances"][j]["lc"]})
    return sorted(out, key=lambda r: (r["a"], r["i"], r["b"], r["j"]))


def overlap(spec: dict) -> dict:
    t0 = time.time()
    h3 = spec["sets"]["h3d"]
    if h3["corpus"] != CORPUS_3D:
        raise SystemExit(f"h3d's corpus is {h3['corpus']}, the replay's {CORPUS_3D}")
    sets = {}
    for name, seed, n, band, where in SETS_3D:
        sets[name] = {"seed": seed, "n": n, "mesh_size": list(band) if band else "one size",
                      "recorded": where, "instances": replay_3d(seed, n, band),
                      "roles": roles_3d(h3, name, n)}
    name, seed, n, where = CALIBRATION_3D
    sets[name] = {"seed": f"SeedSequence({seed}) children", "n": n, "mesh_size": "geometry only",
                  "recorded": where, "instances": replay_children(seed, n), "roles": ["-"] * n}
    by_key, by_geo, by_box = {}, {}, {}
    for name, s in sets.items():
        for i, x in enumerate(s["instances"]):
            if x["load_scales"] is not None:
                by_key.setdefault(key(x), []).append((name, i))
            by_geo.setdefault(geometry(x), []).append((name, i))
            by_box.setdefault(geometry(x)[:4], []).append((name, i))
    checks = []
    for fig, name in FIGURES:
        with np.load(ROOT / "records/cmame/timing/fields" / f"{fig}.npz") as z:
            rec = recorded_scales(z)
        i = int(rec["file"][len("instance_"):-len(".npz")])
        x = sets[name]["instances"][i]
        p = x["params"]
        ok = (p["nu"] == rec["nu"]
              and np.allclose([p["width"], p["height"], p["depth"]], rec["box"], rtol=0,
                              atol=1e-9)
              and np.allclose(x["load_scales"], rec["load_scales"], rtol=1e-9, atol=0))
        checks.append({"figure": fig, "set": name, "index": i, "recorded": rec,
                       "replayed": {"box": [p["width"], p["height"], p["depth"]],
                                    "nu": p["nu"], "load_scales": x["load_scales"],
                                    "lc": x["lc"]},
                       "ok": bool(ok)})
    if not all(c["ok"] for c in checks):
        raise SystemExit("the replay does not reproduce the recorded instances: "
                         + json.dumps(checks, indent=1))
    full = _pairs(by_key, sets, lambda x, y: True)
    same_geometry = _pairs(by_geo, sets, lambda x, y: x["load_scales"] is None
                           or y["load_scales"] is None or key(x) != key(y))
    same_box = _pairs(by_box, sets, lambda x, y: geometry(x) != geometry(y))
    hold = sorted({r["i"] for r in full + same_geometry
                   if r["a"] == CORPUS_3D and r["a_role"] == "tail"}
                  | {r["j"] for r in full + same_geometry
                     if r["b"] == CORPUS_3D and r["b_role"] == "tail"})
    # two dimensions
    h2 = spec["sets"]["h2d"]
    sf = json.loads((ROOT / "records/wp9/scale_factor.json").read_text())
    e1 = json.loads((ROOT / "configs/e1_2d_base.json").read_text())["data"]
    if int(sf["corpus_seed"]) != int(e1["seed"]) or e1["dir"] != h2["corpus"]:
        raise SystemExit("records/wp9/scale_factor.json is not of h2d's corpus seed")
    if COPIES_2D["corpus_manifest_sha256"] != h2["manifest_sha256"]:
        raise SystemExit("COPIES_2D is not of h2d's corpus")
    sp2 = h2["split"]
    perm2 = np.random.default_rng(int(sp2["seed"])).permutation(int(h2["n_corpus"]))
    in_h2 = set(int(i) for i in perm2[int(sp2["n_val"]) + int(h2["used_prefix"]):])
    rebuilt = [int(r["instance"]) for r in sf["instances"]]
    regress = list(range(REGRESS_2D["n"]))
    rperm = np.random.default_rng(int(REGRESS_2D["split"]["seed"])).permutation(REGRESS_2D["n"])
    copies = [i for i in COPIES_2D["instances"] if i in in_h2]
    excluded_2d = sorted({i for i in rebuilt + regress if i in in_h2} | set(copies))
    # the bounds: a band set of seed 0 of n <= 80 is in-band instances 0 to n - 1, a
    # one-size set of n <= 320 a prefix of the fine-mesh set replayed above, a 2D set
    # of n <= 256 instances 0 to n - 1: each shares with the tails only instances left out
    sp3 = h3["split"]
    perm3 = np.random.default_rng(int(sp3["seed"])).permutation(int(h3["n_corpus"]))
    in_h3 = set(int(i) for i in perm3[int(sp3["n_val"]) + int(h3["used_prefix"]):])
    fine_n = next(n for k, _, n, _, _ in SETS_3D if k == FINE_3D)
    bounds_ok = (all(i in hold for i in in_h3 if i < BOUNDS["3d_band"])
                 and BOUNDS["3d_one_size"] <= fine_n
                 and all(i in excluded_2d for i in in_h2 if i < BOUNDS["2d"])
                 and SANDBOX["3d_band"]["longest"] <= BOUNDS["3d_band"]
                 and SANDBOX["3d_one_size"]["longest"] <= BOUNDS["3d_one_size"]
                 and SANDBOX["2d"]["longest"] <= BOUNDS["2d"])
    if not bounds_ok:
        raise SystemExit("a set within the bounds would share a tail instance that is not "
                         "left out")
    name = lambda i: f"instance_{i:05d}.npz"                                  # noqa: E731
    return {"what": "PREREG_HOLDOUT Sec. 2: instances of the holdout sets' tails that another "
                    "recorded set drew again, that a recorded tool rebuilt, or whose own file "
                    "is held outside the box (scripts/holdout_overlap.py)",
            "sets_3d": [{"name": k, **{f: v[f] for f in ("seed", "n", "mesh_size", "recorded")}}
                        for k, v in sets.items()],
            "sandbox": SANDBOX,
            "bounds": {**BOUNDS, "within_bounds_left_out": bounds_ok},
            "checks": checks,
            "coincidences_3d": full,
            "same_geometry_other_loads_3d": same_geometry,
            "same_box_and_nu_only_3d": same_box,
            "h3d_drawn_again": [name(i) for i in hold],
            "rebuilt_2d": {"script": "scripts/w9_scale_factor.py",
                           "record": "records/wp9/scale_factor.json",
                           "corpus_seed": int(sf["corpus_seed"]),
                           "instances": [min(rebuilt), max(rebuilt)], "n": len(rebuilt),
                           "remeshed": int(sf["remesh_geometries"]),
                           "in_h2d": [name(i) for i in rebuilt if i in in_h2],
                           "remeshed_in_h2d": [name(i) for i in
                                               rebuilt[:int(sf["remesh_geometries"])]
                                               if i in in_h2],
                           "use": "load scales only: no label, no network"},
            "regress_2d": {**REGRESS_2D, "instances": [0, REGRESS_2D["n"] - 1],
                           "labelled": sorted(name(int(i)) for i in rperm[:3]),
                           "trained_on": sorted(name(int(i)) for i in rperm[3:3 + 64]),
                           "in_h2d": [name(i) for i in regress if i in in_h2]},
            "copies_2d": {**{k: v for k, v in COPIES_2D.items() if k != "instances"},
                          "instances": [name(i) for i in COPIES_2D["instances"]],
                          "in_h2d": [name(i) for i in copies]},
            "backups_2d": list(BACKUPS_2D),
            "h2d_excluded": [name(i) for i in excluded_2d],
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    from fejepa.analysis.holdout import load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "holdout_v1.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = overlap(load_spec(a.config))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    tail = lambda rows: [(r["a"], r["i"], r["b"], r["j"]) for r in rows  # noqa: E731
                         if "tail" in (r["a_role"], r["b_role"])]
    print(json.dumps({"checks": [c["ok"] for c in res["checks"]],
                      "coincidences_3d": len(res["coincidences_3d"]),
                      "same_geometry_other_loads_3d": len(res["same_geometry_other_loads_3d"]),
                      "same_box_and_nu_only_3d": len(res["same_box_and_nu_only_3d"]),
                      "same_box_and_nu_only_3d_in_tail": tail(res["same_box_and_nu_only_3d"]),
                      "h3d_drawn_again": len(res["h3d_drawn_again"]),
                      "h2d_excluded": len(res["h2d_excluded"]),
                      "copies_in_h2d": res["copies_2d"]["in_h2d"]}, indent=1))


if __name__ == "__main__":
    main()
