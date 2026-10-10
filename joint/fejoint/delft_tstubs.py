"""The welded T-stub tests of Girao Coelho, Bijlaard, Gresnigt and Simoes da Silva, Experimental assessment
of the behaviour of bolted T-stub connections made up of welded plates, J. Constr. Steel Res. 60(2) (2004)
269-311, doi:10.1016/j.jcsr.2003.08.008 (read from the typeset PDF on 10 October 2026).

Every number is from that paper unless marked ASSUMED.
  geometry            Table 1, averaged actual values (b, p, e, w, n; mm); weld throat a_w nominal (Table 1);
                      WT1f was welded with a_w = 8 mm by mistake (Sec. 2.4)
  plates              t_f = t_w = 10.0 mm, "two plates of 10.0 mm thickness" (Sec. 2.1); the actual thickness
                      is not reported (ASSUMED nominal)
  Young's moduli      Table 4 (S355 flange 209856, web 209211; S690 flange 204462, web 208895 MPa);
                      bolts Table 2 (M12: 8.8 FT 216942, 8.8 ST 221886, 10.9 FT 217060, 10.9 ST 217824 MPa);
                      the M16 and M20 bolts were not tested (Sec. 2.2.1): 8.8 FT value used (ASSUMED);
                      Poisson's ratio 0.3 (ASSUMED)
  bolts               snug tight (hand tight, then a 45 degree turn with an ordinary spanner; Sec. 2.3);
                      washers neither mentioned nor drawn (Fig. 2): none (ASSUMED); head, nut, bearing face
                      ISO 4014 / 4017 / 4032 and holes with normal clearance (ASSUMED, fejoint.tstub_hf.BOLTS)
  rows                two-row specimens: e, p, e along b; one-row specimens (WT7, WT57): the row at e from
                      one end, the end of LVDT HP1, "the shorter edge side" (Sec. 3.2.4)
  measured stiffness  Table 9, k_e.0, "computed by means of a regression analysis of the unloading portion of
                      the F-Delta curve" after a first loading to 2/3 F_Rd (Secs 2.3 and 3.2.1), from the
                      average of the two LVDTs, which measure the gap between the flanges at the web centre
                      line at opposite ends of the specimen (Sec. 2.3, Fig. 10)
  EC3 prediction      Table 8 (k_e.0, kN/mm), quoted for context only
"""
from __future__ import annotations

from .tstub_hf import WeldedTStub, BOLTS
from .tstub import TStubPair

E_PLATE = {"S355": {"flange": 209856.0, "web": 209211.0}, "S690": {"flange": 204462.0, "web": 208895.0}}
E_BOLT = {("8.8", "FT"): 216942.0, ("8.8", "ST"): 221886.0, ("10.9", "FT"): 217060.0, ("10.9", "ST"): 217824.0}

# Table 1, actual averaged geometry (mm), nominal weld throat, bolt, plate grade, bolt grade and type
SERIES = {
    "WT1":      dict(b=90.1, p=49.8, e=20.1, w=89.7, n=30.0, a_w=5.0, bolt=12, grade="S355", bg="8.8", bt="ST"),
    "WT1f":     dict(b=90.1, p=49.8, e=20.1, w=89.7, n=30.0, a_w=8.0, bolt=12, grade="S355", bg="8.8", bt="ST"),
    "WT2A":     dict(b=90.0, p=49.9, e=20.0, w=89.9, n=29.9, a_w=3.0, bolt=12, grade="S355", bg="8.8", bt="ST"),
    "WT2B":     dict(b=89.9, p=49.9, e=20.0, w=89.9, n=29.9, a_w=7.0, bolt=12, grade="S355", bg="8.8", bt="ST"),
    "WT4A":     dict(b=149.7, p=89.6, e=30.1, w=89.7, n=30.0, a_w=5.0, bolt=12, grade="S355", bg="8.8", bt="ST"),
    "WT51":     dict(b=90.0, p=50.7, e=19.6, w=90.1, n=30.2, a_w=5.0, bolt=12, grade="S690", bg="8.8", bt="ST"),
    "WT53C":    dict(b=90.1, p=50.0, e=20.0, w=90.1, n=30.0, a_w=5.0, bolt=12, grade="S690", bg="8.8", bt="FT"),
    "WT53D":    dict(b=90.0, p=49.9, e=20.0, w=90.0, n=30.0, a_w=5.0, bolt=12, grade="S690", bg="10.9", bt="ST"),
    "WT53E":    dict(b=89.3, p=49.2, e=20.0, w=90.0, n=30.1, a_w=5.0, bolt=12, grade="S690", bg="10.9", bt="FT"),
    "WT7_M12":  dict(b=75.6, p=None, e=30.0, w=89.9, n=29.9, a_w=5.0, bolt=12, grade="S355", bg="8.8", bt="ST"),
    "WT7_M16":  dict(b=74.9, p=None, e=30.0, w=89.9, n=29.8, a_w=5.0, bolt=16, grade="S355", bg="8.8", bt="FT"),
    "WT7_M20":  dict(b=75.2, p=None, e=29.9, w=89.8, n=29.7, a_w=5.0, bolt=20, grade="S355", bg="8.8", bt="FT"),
    "WT57_M12": dict(b=75.0, p=None, e=30.0, w=89.7, n=30.2, a_w=5.0, bolt=12, grade="S690", bg="8.8", bt="FT"),
    "WT57_M16": dict(b=75.3, p=None, e=30.0, w=90.0, n=30.1, a_w=5.0, bolt=16, grade="S690", bg="8.8", bt="FT"),
    "WT57_M20": dict(b=75.1, p=None, e=30.0, w=90.0, n=30.2, a_w=5.0, bolt=20, grade="S690", bg="8.8", bt="FT"),
}

# Table 9: measured initial stiffness k_e.0 (kN/mm) of the unstiffened specimens with parallel T-stubs,
# and the series (row of SERIES) each test belongs to
KE0 = {
    "WT1a": ("WT1", 96.28), "WT1b": ("WT1", 109.88), "WT1c": ("WT1", 128.63), "WT1d": ("WT1", 120.42),
    "WT1e": ("WT1", 134.25), "WT1f": ("WT1f", 118.46), "WT1g": ("WT1", 137.16), "WT1h": ("WT1", 147.17),
    "WT2Aa": ("WT2A", 128.63), "WT2Ab": ("WT2A", 123.65), "WT2Ba": ("WT2B", 127.15), "WT2Bb": ("WT2B", 159.49),
    "WT4Aa": ("WT4A", 150.15), "WT4Ab": ("WT4A", 173.91), "WT51a": ("WT51", 119.24), "WT51b": ("WT51", 123.67),
    "WT53C": ("WT53C", 128.46), "WT53D": ("WT53D", 105.79), "WT53E": ("WT53E", 129.63),
    "WT7_M12": ("WT7_M12", 91.18), "WT7_M16": ("WT7_M16", 116.09), "WT7_M20": ("WT7_M20", 137.70),
    "WT57_M12": ("WT57_M12", 85.78), "WT57_M16": ("WT57_M16", 110.43), "WT57_M20": ("WT57_M20", 150.96),
}

# Table 8: Eurocode 3 predictions of k_e.0 (kN/mm), context only
EC3_KE0 = {"WT1": 109.48, "WT1f": 109.48, "WT2A": 88.68, "WT2B": 129.21, "WT4A": 181.98, "WT51": 92.93,
           "WT53C": 96.57, "WT53D": 98.54, "WT53E": 96.40, "WT7_M12": 86.60, "WT7_M16": 89.09,
           "WT7_M20": 91.63, "WT57_M12": 78.83, "WT57_M16": 82.97, "WT57_M20": 83.85}


def rows_of(s: dict):
    return (s["e"], s["e"] + s["p"]) if s["p"] is not None else (s["e"],)


def hifi(series: str, **overrides) -> WeldedTStub:
    s = SERIES[series]
    kw = dict(name=series, b=s["b"], w=s["w"], n=s["n"], a_w=s["a_w"], rows=rows_of(s), bolt=s["bolt"],
              E_f=E_PLATE[s["grade"]]["flange"], E_w=E_PLATE[s["grade"]]["web"], E_b=E_BOLT[(s["bg"], s["bt"])])
    kw.update(overrides)
    return WeldedTStub(**kw)


def baseline(series: str, t: float = 10.0, H: float = 80.0) -> TStubPair:
    """The G-C1a model's machinery (fejoint.tstub) for the same specimen: no hole, rigid tilting head plane
    on the bearing face r <= d_w / 2, bolt springs with the EN 1993-1-8 bolt length, shear springs."""
    s = SERIES[series]
    bs = BOLTS[s["bolt"]]
    two = s["p"] is not None
    y_rows = ((s["b"] - s["p"]) / 2,) if two else (s["e"],)
    return TStubPair(t_f=t, t_w=t, b=s["w"] + 2 * s["n"], L=s["b"], w=s["w"], bolt_y=y_rows, H=H, corner="weld",
                     r=s["a_w"] * 2 ** 0.5, d=bs.d, As=bs.As, head=bs.k, nut=bs.m, bearing_r=bs.dw / 2,
                     n_washers=0, washer_t=0.0, E=E_PLATE[s["grade"]]["flange"], nu=0.3, x_meas=t + 30.0,
                     E_f=E_PLATE[s["grade"]]["flange"], E_w=E_PLATE[s["grade"]]["web"],
                     E_b=E_BOLT[(s["bg"], s["bt"])], full_length=not two, measure="gap")
