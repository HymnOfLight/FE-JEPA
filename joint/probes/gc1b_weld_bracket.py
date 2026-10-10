"""Post-hoc check, prompted by the G-C1a2 audit (not pre-specified; it does not change the G-C1b verdict).

On the meshes of gate G-C1b the stair-stepped fillet welds keep some of the cells whose centroids lie exactly on the
weld face and drop others, by round-off. On the fine (primary) mesh the 5 mm welds reach 6.43 mm up the flange
instead of 7.07 mm (probes/tstub_weld_check.json). This script reruns G-C1b's model with the weld drawn consistently,
both ways, and applies G-C1b's criteria to each:
    kept     cells on the weld face kept: the welds contain the exact fillet;
    dropped  cells on the weld face dropped: the welds lie inside the exact fillet.
The model is the stamped fejoint/tstub_hf.py with one line changed in memory (asserted below); the file itself is
not modified. Usage:
    python -I probes/gc1b_weld_bracket.py run        (resumes)
    python -I probes/gc1b_weld_bracket.py evaluate
"""
import sys, json, math, time, types, hashlib, pathlib, resource
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fejoint
from fejoint import tstub_hf
from fejoint.delft_tstubs import SERIES, hifi

OUT = ROOT / "probes" / "gc1b_weld_bracket.json"
SRC = ROOT / "fejoint" / "tstub_hf.py"
STAMPED = "ed7f8b8f966a0ae2b7908c20e414799484e93c8ae290d312d1a8ff17c52c1c4f"
LINE = "weld = (cx > g.t_f) & (cz > g.t_w / 2) & ((cx - g.t_f) + (cz - g.t_w / 2) <= g.leg)"
TOL = {"kept": "g.leg + 1e-6", "dropped": "g.leg - 1e-6"}


def variant(name):
    src = SRC.read_text()
    assert hashlib.sha256(src.encode()).hexdigest() == STAMPED, "fejoint/tstub_hf.py is not the stamped version"
    assert src.count(LINE) == 1
    mod = types.ModuleType(f"fejoint.tstub_hf_{name}")
    mod.__package__ = "fejoint"
    sys.modules[mod.__name__] = mod                      # dataclasses look the module up while building classes
    exec(compile(src.replace(LINE, LINE.replace("<= g.leg", "<= " + TOL[name])), f"tstub_hf[{name}]", "exec"),
         mod.__dict__)
    return mod


def weld_geometry(mod, g, mesh):
    msh = mod.build(g, **{k: v for k, v in tstub_hf.MESHES[mesh].items()})
    nd, hx, mat = msh["nodes"], msh["hexes"], msh["mat"]
    c = nd[hx].mean(axis=1); lo, hi = nd[hx].min(axis=1), nd[hx].max(axis=1)
    w = (mat == 1) & (c[:, 0] > g.t_f) & (c[:, 2] > g.t_w / 2)
    Ly = float(nd[:, 1].max() - nd[:, 1].min())
    vol = float(np.prod(hi[w] - lo[w], axis=1).sum())
    f = w & np.isclose(lo[:, 0], g.t_f); s = w & np.isclose(lo[:, 2], g.t_w / 2)
    return {"volume_over_exact": vol / (0.5 * g.leg ** 2 * Ly),
            "reach_on_flange_mm": float(hi[f, 2].max() - g.t_w / 2), "reach_on_web_mm": float(hi[s, 0].max() - g.t_f)}


def run():
    res = json.load(open(OUT)) if OUT.exists() else {"runs": [], "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    done = {(r["variant"], r["series"], r["mesh_level"]) for r in res["runs"]}
    mods = {v: variant(v) for v in TOL}
    for mesh in ("medium", "fine"):
        for v in ("kept", "dropped"):
            for n in SERIES:
                if (v, n, mesh) in done:
                    continue
                g = hifi(n)
                t0 = time.perf_counter()
                o = mods[v].analyse(g, mesh, F=10000.0, reg=1e-6)
                rec = {"variant": v, "series": n, "mesh_level": mesh, "K_kN_per_mm": o["K_kN_per_mm"],
                       "weld": weld_geometry(mods[v], g, mesh), "pdas_iterations": o.get("pdas_iterations"),
                       "equilibrium_rel": o.get("equilibrium_rel"), "seconds": round(time.perf_counter() - t0, 1),
                       "maxrss_MB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}
                res["runs"].append(rec)
                json.dump(res, open(OUT, "w"), indent=1, default=float)
                print(f"{v:8s} {n:9s} {mesh:6s} K {rec['K_kN_per_mm']:8.2f}  {rec['seconds']:6.1f} s  "
                      f"weld {rec['weld']['volume_over_exact']:.3f} {rec['weld']['reach_on_flange_mm']:.2f}", flush=True)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    json.dump(res, open(OUT, "w"), indent=1, default=float)


def evaluate():
    sys.path.insert(0, str(ROOT / "checks"))
    res = json.load(open(OUT))
    stamped = json.load(open(ROOT / "checks" / "gc1b_verdict.json"))
    # gc1b_evaluate.py runs on import and rewrites checks/gc1b_verdict.json; take its judge() by source instead
    src = (ROOT / "checks" / "gc1b_evaluate.py").read_text()
    head = src.split("\nout = ")[0]                      # imports, data, pick, K, gm, judge
    assert "def judge(" in head and "json.dump" not in head
    ns = {"__file__": str(ROOT / "checks" / "gc1b_evaluate.py")}
    exec(compile(head, "gc1b_evaluate[head]", "exec"), ns)
    judge = ns["judge"]
    out = {"note": "post hoc; G-C1b criteria applied to the weld variants; the stamped verdict stands",
           "stamped": {k: stamped["primary"][k] for k in ("gm_R", "n_inside", "outside", "C1", "C2", "verdict")}}
    for mesh in ("medium", "fine"):
        for v in ("kept", "dropped"):
            Ks = {r["series"]: r["K_kN_per_mm"] for r in res["runs"] if r["variant"] == v and r["mesh_level"] == mesh}
            if len(Ks) < len(SERIES):
                continue
            j = judge(Ks)
            ref = stamped["K_hifi_" + mesh]
            out[f"{v} {mesh}"] = {**{k: j[k] for k in ("gm_R", "n_inside", "outside", "R_min", "R_max", "C1a", "C1b",
                                                       "C2a", "C2b", "verdict")},
                                  "R": j["R"], "K": Ks, "change_from_stamped": {n: Ks[n] / ref[n] - 1 for n in Ks}}
    for v in TOL:
        if f"{v} medium" in out and f"{v} fine" in out:
            out[f"{v} mesh_change_fine_vs_medium"] = {n: out[f"{v} fine"]["K"][n] / out[f"{v} medium"]["K"][n] - 1
                                                      for n in SERIES}
    json.dump(out, open(ROOT / "probes" / "gc1b_weld_bracket_eval.json", "w"), indent=1)
    for k, o in out.items():
        if isinstance(o, dict) and "gm_R" in o:
            print(k, {x: o[x] for x in ("gm_R", "n_inside", "outside", "verdict") if x in o})


if __name__ == "__main__":
    {"run": run, "evaluate": evaluate}[sys.argv[1]]()
