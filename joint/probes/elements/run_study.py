"""Element and mesh study of the joint model (exploratory, after gate G-C1.3; not a gate).

The C1.3 reference physics (frictionless contact with the rigid column flange, bolt heads bearing unilaterally with
tilt springs, a 300 mm beam segment with a rigid loaded end section), the specimens' measured moduli, and the
paper's rotation; only the element kind and the mesh change, and with them the node sets of the contact face, the
bolt patches and the rigid end section (see joint_generic.py). Records one JSON line per run in OUT.jsonl, skipping
runs already recorded, so the study can be resumed.

    python -I probes/elements/run_study.py OUT.jsonl [PHASE ...]
"""
import json
import pathlib
import platform
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np                                            # noqa: E402
import scipy                                                  # noqa: E402
from fejoint import linsolve                                   # noqa: E402
from fejoint.specimens import specimen, SJ_INI                 # noqa: E402
from joint_generic import analyse_generic                      # noqa: E402

L0, L1, L2, L3 = (20.0, 1, 1, 1), (10.0, 1, 1, 1), (5.0, 2, 2, 1), (2.5, 4, 4, 2)
LIN, QUAD, QUAD_HEAVY = ("H8", "H8I", "T4"), ("H20", "H20R"), ("H27", "T10")
PHASES = {
    "coarse": [(k, L) for L in (L1,) for k in LIN + QUAD + QUAD_HEAVY] + [(k, L0) for k in QUAD + QUAD_HEAVY],
    "medium": [(k, L2) for k in LIN + QUAD],
    "layers": [(k, (5.0, n, 2, 1)) for k in LIN for n in (1, 4, 8)] + [(k, (5.0, n, 2, 1)) for k in ("H20",) for n in (1, 4)],
    "fine": [(k, L3) for k in LIN],
    "heavy": [(k, L2) for k in QUAD_HEAVY],
    "extra": [(k, (2.5, 8, 4, 2)) for k in LIN],
}
SPECIMENS = ("FS1", "FS3")


def key(rec):
    m = rec["mesh"]
    return (rec["specimen"], rec["kind"], m["h"], m["n_tp"], m["n_tf"], m["n_tw"])


def main(out, phases):
    out = pathlib.Path(out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    done.add(key(r))
    env = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "pardiso": linsolve._PARDISO is not None}
    for ph in phases:
        for name in SPECIMENS:
            g = specimen(name)
            for kind, (h, ntp, ntf, ntw) in PHASES[ph]:
                k = (name, kind, h, ntp, ntf, ntw)
                if k in done:
                    continue
                t0 = time.time()
                try:
                    r = analyse_generic(g, kind, h, ntp, ntf, ntw)
                    tests = [v for t, v in SJ_INI.items() if t.startswith(name)]
                    r.update({"specimen": name, "phase": ph, "R_tests": [r["S_paper_kNm_per_mrad"] / v for v in tests],
                              "env": env, "seconds_total": round(time.time() - t0, 1),
                              "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z")})
                except Exception as e:                       # recorded, then the study goes on
                    r = {"specimen": name, "kind": kind, "phase": ph,
                         "mesh": {"h": h, "n_tp": ntp, "n_tf": ntf, "n_tw": ntw}, "error": repr(e),
                         "trace": traceback.format_exc()[-2000:]}
                with out.open("a") as fh:
                    fh.write(json.dumps(r) + "\n")
                s = r.get("S_paper_kNm_per_mrad")
                print(f"{ph:7s} {name} {kind:5s} h={h:<4} tp={ntp} tf={ntf} tw={ntw} "
                      + (f"S={s:.4f} dof={r['mesh']['dof_reduced']} it={r['pdas_iterations']} t={r['seconds_total']}s"
                         if s is not None else f"ERROR {r['error'][:200]}"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or list(PHASES))
