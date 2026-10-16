"""Sensitivity of S to the radius of the bolt-head bearing patch (exploratory, after gate G-C1.3; not a gate).

The patch is the set of plate nodes within washer_r of the bolt's axis, so S can jump when a node crosses the
circle. On FS1's 10 mm mesh two of the three patches each have a node just outside washer_r = 14.1 mm, at 14.102
and 14.110 mm: 14.1 and 14.12 mm are the same radius physically but not on the mesh. Element H8I (the current
one); one JSON line per run.

    python -I probes/elements/washer_radius.py OUT.jsonl
"""
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fejoint import linsolve                                   # noqa: E402
from fejoint.specimens import specimen                         # noqa: E402
from joint_generic import analyse_generic                      # noqa: E402

RADII = (12.5, 13.0, 14.1, 14.12, 15.0)
MESHES = ((10.0, 1, 1, 1), (5.0, 2, 2, 1))


def main(out):
    out = pathlib.Path(out)
    with out.open("w") as fh:
        for name in ("FS1", "FS3"):
            for (h, a, b, c) in MESHES:
                for wr in RADII:
                    t0 = time.time()
                    r = analyse_generic(specimen(name, washer_r=wr), "H8I", h, a, b, c)
                    rec = {"specimen": name, "kind": "H8I", "mesh": {"h": h, "n_tp": a, "n_tf": b, "n_tw": c},
                           "washer_r": wr, "patch_nodes": r["mesh"]["patch_nodes"],
                           "S_paper_kNm_per_mrad": r["S_paper_kNm_per_mrad"], "seconds": round(time.time() - t0, 1),
                           "solver": "PARDISO" if linsolve._PARDISO is not None else "legacy"}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    print(f"{name} h={h:<4} washer_r={wr:<5} patches={rec['patch_nodes']} "
                          f"S={rec['S_paper_kNm_per_mrad']:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
