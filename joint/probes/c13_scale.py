"""Exploratory scale check for stage C1.3 (not a gate): how small can the joint model's mesh be while its initial
stiffness stays near the fine-mesh value? The idealised model of G-C1a (fejoint.joint, the options of
checks/gc1a_run.py) is solved on meshes coarser than G-C1a's medium and compared with G-C1a's fine-mesh values.
The higher-fidelity model (fejoint.joint_hf) is only counted, not solved: its structured grid carries the fine
spacing of the welds and bearing rings across the whole model. Writes probes/c13_scale.json.
    python -I probes/c13_scale.py
"""
import sys, json, time, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fejoint.joint import analyse
from fejoint.specimens import specimen
from fejoint import joint_hf
from fejoint.tstub_hf import BOLTS

BOLT = dict(washer="head_contact", k_rot_factor=4.0, reg=1e-6)          # as checks/gc1a_run.py
FINE = json.load(open(ROOT / "checks" / "gc1a_verdict.json"))["S_FE_fine"]
MESHES = {"h10_tp1": dict(h=10.0, n_tp=1, n_tf=1), "h10_tp2": dict(h=10.0, n_tp=2, n_tf=2),
          "h7.5_tp2": dict(h=7.5, n_tp=2, n_tf=2), "h5_tp2 (G-C1a medium)": dict(h=5.0, n_tp=2, n_tf=2)}
out = {"idealised": {}, "higher_fidelity_counts": {}}
for name in ("FS1", "FS2", "FS3", "FS4"):
    for lab, mp in MESHES.items():
        t0 = time.perf_counter()
        o, _ = analyse(specimen(name), support="contact", **BOLT, **mp)
        r = {"S_paper": o["S_paper_kNm_per_mrad"], "over_fine": o["S_paper_kNm_per_mrad"] / FINE[name],
             "mesh": o["mesh"], "seconds": round(time.perf_counter() - t0, 1)}
        out["idealised"][f"{name} {lab}"] = r
        print(f"{name} {lab:22s} S {r['S_paper']:7.3f}  /fine {r['over_fine']:.4f}  {r['mesh']}  {r['seconds']} s",
              flush=True)
    g = specimen(name, washer_r=BOLTS[20].dw / 2, hole_d=22.0)
    for lab, mp in joint_hf.MESHES.items():
        m = joint_hf.build(g, a_flange=5.75, a_web=3.75, **mp)
        out["higher_fidelity_counts"][f"{name} {lab}"] = {"nodes": int(m["nodes"].shape[0]),
                                                          "hexes": int(m["hexes"].shape[0])}
    m = joint_hf.build(g, a_flange=5.75, a_web=3.75, h=10.0, n_b=1.0, n_tp=1, n_tf=1, n_head=1)
    out["higher_fidelity_counts"][f"{name} coarsest tried (h 10, n_b 1, one layer in plate, flange and head)"] = {
        "nodes": int(m["nodes"].shape[0]), "hexes": int(m["hexes"].shape[0])}
json.dump(out, open(ROOT / "probes" / "c13_scale.json", "w"), indent=1, default=float)
