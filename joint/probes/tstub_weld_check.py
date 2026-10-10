"""Post-hoc check, prompted by the G-C1a2 audit: on the actual meshes of gate G-C1b, how much of each fillet weld does
the stair-step rule of fejoint.tstub_hf.build keep, and how far up the flange does the weld reach?
Mesh building only; no solve. Writes probes/tstub_weld_check.json.
    python -I probes/tstub_weld_check.py
"""
import sys, json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fejoint.tstub_hf import build, MESHES
from fejoint.delft_tstubs import SERIES, hifi

out = {}
for name in SERIES:
    g = hifi(name)
    for lvl in ("medium", "fine"):
        mp = {k: v for k, v in MESHES[lvl].items()}
        msh = build(g, **mp)
        nd, hx, mat = msh["nodes"], msh["hexes"], msh["mat"]
        c = nd[hx].mean(axis=1)
        lo, hi = nd[hx].min(axis=1), nd[hx].max(axis=1)
        vol = np.prod(hi - lo, axis=1)                      # the cells are boxes
        w = (mat == 1) & (c[:, 0] > g.t_f) & (c[:, 2] > g.t_w / 2)
        Ly = float(nd[:, 1].max() - nd[:, 1].min())
        exact = 0.5 * g.leg ** 2 * Ly
        first = w & np.isclose(lo[:, 0], g.t_f)            # weld cells sitting on the flange
        reach = float(hi[first, 2].max() - g.t_w / 2)       # weld toe on the flange, from the web face
        on_web = w & np.isclose(lo[:, 2], g.t_w / 2)        # weld cells against the web
        up = float(hi[on_web, 0].max() - g.t_f)             # weld toe on the web, from the flange face
        xs = msh["grids"]["xs"]; zs = msh["grids"]["zs"]
        hw_x = np.diff(xs[(xs >= g.t_f - 1e-9) & (xs <= g.t_f + g.leg + 1e-9)])
        hw_z = np.diff(zs[(zs >= g.t_w / 2 - 1e-9) & (zs <= g.t_w / 2 + g.leg + 1e-9)])
        out[f"{name} {lvl}"] = {"a_w": g.a_w, "leg": g.leg, "volume_over_exact": float(vol[w].sum() / exact),
                                "reach_on_flange_mm": reach, "reach_on_web_mm": up,
                                "cells_across_leg_x": len(hw_x), "cells_across_leg_z": len(hw_z),
                                "spacing_x_mm": float(hw_x.mean()), "spacing_z_mm": float(hw_z.mean())}
        r = out[f"{name} {lvl}"]
        print(f"{name:9s} {lvl:6s} a_w {g.a_w:.0f} leg {g.leg:5.2f}  vol/exact {r['volume_over_exact']:.3f}  "
              f"reach flange {reach:5.2f}  web {up:5.2f}  cells {len(hw_x)}x{len(hw_z)}", flush=True)
json.dump(out, open(ROOT / "probes" / "tstub_weld_check.json", "w"), indent=1)
