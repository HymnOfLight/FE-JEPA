"""The generic joint analysis with kind H8I against fejoint.joint.analyse (must agree to round-off).

    python -I probes/elements/check_generic.py
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fejoint import joint as J                              # noqa: E402
from fejoint.specimens import specimen                       # noqa: E402
from joint_generic import analyse_generic                    # noqa: E402

for name in ("FS1", "FS3"):
    g = specimen(name)
    for (h, ntp, ntf, ntw) in ((10.0, 1, 1, 1), (5.0, 2, 2, 1)):
        ref, _ = J.analyse(g, h=h, n_tp=ntp, n_tf=ntf, n_tw=ntw, support="contact", washer="head_contact",
                           k_rot_factor=4.0, reg=1e-6, beam_segment=300.0)
        gen = analyse_generic(g, "H8I", h, ntp, ntf, ntw)
        a, b = ref["S_paper_kNm_per_mrad"], gen["S_paper_kNm_per_mrad"]
        print(name, h, ntp, ntf, ntw, "joint.analyse %.12f" % a, "generic %.12f" % b, "rel diff %.1e" % abs(b / a - 1),
              "dof", gen["mesh"]["dof_reduced"], "it", gen["pdas_iterations"])
