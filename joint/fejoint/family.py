"""Parameter family of extended end plate joints for stage C1.3 (draft ranges; the pre-specification fixes them).

The family varies the end plate and the bolt layout around the four tested details of Girao Coelho, Bijlaard
and Simoes da Silva (2004, Table 2) and keeps the beam, the column flange thickness, the bolts and the load arm
fixed at the values of FS1. All parts take E = 210,000 MPa and nu = 0.3; the measured moduli are not used, so
that a sampled geometry and a test detail differ only in geometry.

Sampled, uniformly and independently, then kept only if the layout is valid (rejection sampling):
    tp      end plate thickness
    eX      top edge of the plate to bolt row 1
    p       row 1 to row 2 (across the tension flange)
    p23     row 2 to row 3 (row 3 sits in the compression zone)
    w       gauge
    o       overhang of the plate beyond the beam's flanges on each side; the plate width is bp = bb + 2 o
    LX      top edge of the plate to the beam's top face
    c_bot   extension of the plate below the beam's bottom face; the plate height is hp = LX + hb + c_bot
Validity: every bearing patch (radius washer_r) lies inside the plate, clears the beam's footprint by at least
`clear`, and clears the plate's edges by at least `clear`; the rows are in order.
An overhang below SNAP is set to zero: the structured mesh puts a grid line on the plate's edge and one on the
flange's, and two lines a fraction of a millimetre apart would leave a sliver of cells through the whole model.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np

from .joint import JointGeometry
from .specimens import TABLE2, BEAM, T_FC, X_DT1

RANGES = {"tp": (8.0, 22.0), "eX": (25.0, 40.0), "p": (80.0, 100.0), "p23": (190.0, 220.0),
          "w": (80.0, 110.0), "o": (0.0, 15.0), "LX": (55.0, 85.0), "c_bot": (20.0, 45.0)}
CLEAR = 2.0          # mm, minimum clearance of a bearing patch from the beam's footprint and the plate's edges
SNAP = 2.0           # mm, overhangs below this are set to zero
WASHER_R = 14.1      # mm, bearing face radius of an M20 head (as fejoint.specimens)


@dataclass(frozen=True)
class FamilyParams:
    tp: float; eX: float; p: float; p23: float; w: float; o: float; LX: float; c_bot: float


def geometry(q: FamilyParams) -> JointGeometry:
    """The JointGeometry of a family member: the beam, bolts, column flange and load arm of FS1, nominal moduli."""
    bm = BEAM["FS1"]
    hp = q.LX + bm["hb"] + q.c_bot
    bp = bm["bb"] + 2.0 * (q.o if q.o >= SNAP else 0.0)
    return JointGeometry(tp=q.tp, hp=hp, bp=bp, eX=q.eX, p=q.p, p23=q.p23, w=q.w, LX=q.LX,
                         hb=bm["hb"], bb=bm["bb"], tfb=bm["tfb"], twb=bm["twb"],
                         L_beam=bm["L_load"] - q.tp + 50.0, L_load=bm["L_load"] - q.tp, x_dt1=X_DT1 - q.tp,
                         t_fc=T_FC, d=20.0, As=245.0, head=12.5, nut=18.0, washer_r=WASHER_R, n_washers=0,
                         E=210000.0, nu=0.3)


def violations(g: JointGeometry, clear: float = CLEAR) -> list[str]:
    """Reasons a layout is invalid (empty if valid)."""
    out = []
    r1, r2, r3 = g.rows
    R = g.washer_r + clear
    if not r1 > r2 > r3 > 0:
        out.append("rows out of order")
    if g.hp - r1 < R:
        out.append("row 1 too close to the top edge")
    if r3 < R:
        out.append("row 3 too close to the bottom edge")
    if r1 - g.y_top < R:
        out.append("row 1 patch reaches the top flange")
    if (g.y_top - g.tfb) - r2 < R:
        out.append("row 2 patch reaches the top flange")
    if r3 - (g.y_bot + g.tfb) < R:
        out.append("row 3 patch reaches the bottom flange")
    if g.w / 2 - g.twb / 2 < R:
        out.append("patches reach the web")
    if g.bp / 2 - g.w / 2 < R:
        out.append("patches reach the plate's side edges")
    if g.bp < g.bb:
        out.append("plate narrower than the beam's flanges")
    if g.y_bot < 0:
        out.append("beam below the plate")
    return out


def sample(n: int, seed: int, ranges: dict = RANGES, clear: float = CLEAR, max_tries: int = 100000):
    """n valid family members, deterministic in seed. Returns (list of FamilyParams, number of draws)."""
    rng = np.random.default_rng(seed)
    keys = list(FamilyParams.__dataclass_fields__)
    out, draws = [], 0
    while len(out) < n:
        draws += 1
        if draws > max_tries:
            raise RuntimeError("too many rejections: the ranges admit too few valid layouts")
        q = FamilyParams(**{k: float(rng.uniform(*ranges[k])) for k in keys})
        if not violations(geometry(q), clear):
            out.append(q)
    return out, draws


def tested_details():
    """The four tested details as family members (Table 2 plate and bolt layout; FS1 beam, nominal moduli).
    c_bot is taken from each detail's own plate height; their plates are within 1 mm of the flanges' width, so
    their overhang snaps to zero."""
    out = {}
    bb = BEAM["FS1"]["bb"]
    for name, t in TABLE2.items():
        c_bot = t["hp"] - t["LX"] - BEAM["FS1"]["hb"]
        out[name] = FamilyParams(tp=t["tp"], eX=t["eX"], p=t["p"], p23=t["p23"], w=t["w"], o=(t["bp"] - bb) / 2,
                                 LX=t["LX"], c_bot=c_bot)
    return out


__all__ = ["RANGES", "CLEAR", "SNAP", "FamilyParams", "geometry", "violations", "sample", "tested_details", "asdict"]
