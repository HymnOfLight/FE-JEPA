"""The four joint details tested by Girao Coelho, Bijlaard and Simoes da Silva, Engineering Structures 26(9)
(2004) 1185-1206, doi:10.1016/j.engstruct.2000.09.001, as JointGeometry objects.

Every number below is from the published paper (read from the typeset PDF on 10 October 2026), except
where marked ASSUMED.

  end plate, bolt layout     Table 2 (measured, mm)
  beam section               Table 2 (measured hb, bb, tfb, twb); root radii ignored (MODEL)
  column flange              Table 2, tfc = 40.21 mm; enters only the bolt grip (the column is rigid, Sec. 2.1)
  load arm                   Table 2, L_load, "distance between the load application point and the face of
                             the end plate"; taken from the plate's contact face (ASSUMED reading of "face")
  DT1                        900 mm from the face of the end plate (Fig. 5, eq. 5); same reading of "face"
  Young's moduli             Table 5 (end plate by series; beam flange 209496, web 208332 MPa);
                             Table 6 (bolts: batch 1 223166, batch 2 222982 MPa); Poisson ratio 0.3 (ASSUMED)
  bolts                      M20 grade 8.8, full thread, 22 mm holes, hand-tightened then a 45 degree turn
                             with an ordinary spanner (Sec. 2.5); no preload modelled (MODEL)
  washers                    none: not mentioned in the text, none visible in Figs 9 and 22
  head and nut               ISO 4017 / ISO 4032 nominal M20: head height 12.5 mm, nut height 18 mm, bearing
                             face diameter about 28.2 mm (ASSUMED: types not stated in the paper)
  bolt holes                 not modelled: solid plate (MODEL)
Positions in JointGeometry are measured from the plate's beam-side face x = tp, so the paper's distances
from the contact face are converted by subtracting tp.
"""
from __future__ import annotations

from dataclasses import replace

from .joint import JointGeometry

# Table 2: end plate and connection geometry (mm)
TABLE2 = {
    "FS1": dict(hp=401.04, bp=149.84, tp=10.40, e=30.01, w=89.91, eX=29.90, LX=69.35, p=90.03, p23=205.90, ecomp=76.45),
    "FS2": dict(hp=400.84, bp=149.41, tp=15.01, e=29.76, w=89.89, eX=30.10, LX=69.30, p=89.98, p23=205.04, ecomp=75.44),
    "FS3": dict(hp=401.40, bp=150.47, tp=20.02, e=30.27, w=89.93, eX=29.74, LX=68.90, p=90.14, p23=204.84, ecomp=76.82),
    "FS4": dict(hp=401.69, bp=149.76, tp=10.06, e=29.94, w=89.88, eX=29.83, LX=69.86, p=89.95, p23=205.28, ecomp=76.13),
}
# Table 2: beam profile (mm) and load arm
BEAM = {
    "FS1": dict(hb=300.45, bb=150.50, tfb=10.76, twb=7.20, L_beam=1200.00, L_load=1002.50),
    "FS2": dict(hb=301.40, bb=149.60, tfb=10.67, twb=7.01, L_beam=1200.38, L_load=1000.25),
    "FS3": dict(hb=301.46, bb=149.75, tfb=10.57, twb=7.03, L_beam=1191.50, L_load=992.63),
    "FS4": dict(hb=300.66, bb=149.54, tfb=11.86, twb=7.03, L_beam=1218.75, L_load=991.88),
}
T_FC = 40.21                                                   # Table 2, column flange
E_PLATE = {"FS1": 209856.0, "FS2": 208538.0, "FS3": 208622.0, "FS4": 204462.0}   # Table 5
E_FLANGE, E_WEB = 209496.0, 208332.0                           # Table 5, beam
E_BOLT = {"FS1": 223166.0, "FS2": 223166.0, "FS3": 223166.0, "FS4": 222982.0}    # Table 6 (FS3b used batch 2: 222982)
X_DT1 = 900.0                                                  # Fig. 5, eq. (5)
STEEL_GRADE = {"FS1": "S355", "FS2": "S355", "FS3": "S355", "FS4": "S690"}

# Table 8: measured initial stiffness (kNm/mrad), regression on the unloading branch after 2/3 Mj.Rd
SJ_INI = {"FS1a": 18.19, "FS1b": 16.84, "FS2a": 23.39, "FS2b": 22.00, "FS3a": 23.23, "FS3b": 21.56,
          "FS4a": 16.18, "FS4b": 17.15}


def specimen(name: str, **overrides) -> JointGeometry:
    t, bm = TABLE2[name], BEAM[name]
    tp = t["tp"]
    g = JointGeometry(tp=tp, hp=t["hp"], bp=t["bp"], eX=t["eX"], p=t["p"], p23=t["p23"], w=t["w"], LX=t["LX"],
                      hb=bm["hb"], bb=bm["bb"], tfb=bm["tfb"], twb=bm["twb"],
                      L_beam=bm["L_load"] - tp + 50.0, L_load=bm["L_load"] - tp, x_dt1=X_DT1 - tp, t_fc=T_FC,
                      d=20.0, As=245.0, head=12.5, nut=18.0, washer_r=14.1, n_washers=0,
                      E=210000.0, nu=0.3, E_plate=E_PLATE[name], E_flange=E_FLANGE, E_web=E_WEB, E_bolt=E_BOLT[name])
    return replace(g, **overrides) if overrides else g


def agerskov_length(g: JointGeometry) -> float:
    """Effective bolt length after Agerskov, as quoted by Bursi and Jaspart (1997), eqs (1) and (2), for a fully
    threaded bolt without washers: E A_b / (K1 + 2 K4) = E A_s / L_eff, K1 = l_s + 1.43 l_t + 0.71 l_n,
    K4 = 0.1 l_n + 0.2 l_w, with l_s = 0, l_t = grip, l_w = 0 (sensitivity only)."""
    import math
    A_b = math.pi * g.d ** 2 / 4.0
    l_t, l_n, l_w = g.tp + g.t_fc, g.nut, 0.0
    K1 = 1.43 * l_t + 0.71 * l_n
    K4 = 0.1 * l_n + 0.2 * l_w
    return g.As * (K1 + 2 * K4) / A_b
