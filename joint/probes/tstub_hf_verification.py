"""Verification of the higher-fidelity T-stub model before gate G-C1b is stamped: mesh convergence and the
effect of the changes from the baseline, on two synthetic T-stubs (VT1, one off-centre M20 row; VT2, two M16
rows). Neither is a specimen of the gate; no test of Girao Coelho et al. (2004) is solved here."""
import sys, json, time, pathlib, resource
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataclasses import replace
from fejoint.tstub_hf import WeldedTStub, analyse, BOLTS
from fejoint.tstub import TStubPair, analyse as base_analyse

VT2 = WeldedTStub(name="VT2", b=100.0, w=100.0, n=35.0, a_w=4.0, rows=(22.5, 77.5), bolt=16,
                  E_f=205000.0, E_w=211000.0, E_b=214000.0, t_f=12.0, t_w=8.0)
VT1 = WeldedTStub(name="VT1", b=80.0, w=100.0, n=35.0, a_w=4.0, rows=(32.0,), bolt=20,
                  E_f=205000.0, E_w=211000.0, E_b=214000.0, t_f=12.0, t_w=8.0)


def base_of(g):
    bs = BOLTS[g.bolt]
    two = len(g.rows) == 2
    yr = ((g.b - (g.rows[1] - g.rows[0])) / 2,) if two else (g.rows[0],)
    return TStubPair(t_f=g.t_f, t_w=g.t_w, b=g.w + 2 * g.n, L=g.b, w=g.w, bolt_y=yr, H=g.H, corner="weld",
                     r=g.leg, d=bs.d, As=bs.As, head=bs.k, nut=bs.m, bearing_r=bs.dw / 2, x_meas=g.t_f + 30.0,
                     E=g.E_f, E_f=g.E_f, E_w=g.E_w, E_b=g.E_b, full_length=not two, measure="gap")


rows = []
out = pathlib.Path(__file__).with_suffix(".json")
for g in (VT2, VT1):
    for mesh in ("coarse", "medium", "fine"):
        t0 = time.perf_counter()
        o = analyse(g, mesh)
        o["seconds"] = round(time.perf_counter() - t0, 1)
        o["model"] = "hifi"; o["maxrss_MB"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        rows.append(o); json.dump(rows, open(out, "w"), indent=1)
        print(g.name, "hifi", mesh, o["mesh"]["dof"], round(o["K_kN_per_mm"], 3), o["seconds"], "s", flush=True)
    for h, ntf in ((5.0, 2), (2.5, 4), (1.25, 6)):
        t0 = time.perf_counter()
        o = base_analyse(base_of(g), h=h, n_tf=ntf)
        o.update({"name": g.name, "model": "baseline", "seconds": round(time.perf_counter() - t0, 1)})
        rows.append(o); json.dump(rows, open(out, "w"), indent=1)
        print(g.name, "baseline", h, o["mesh"]["dof"], round(o["K_kN_per_mm"], 3), o["seconds"], "s", flush=True)
    # one change at a time, medium mesh: Agerskov -> EN 1993-1-8 bolt length
    o = analyse(replace(g, bolt_length="ec3"), "medium"); o.update({"model": "hifi_ec3_length"})
    rows.append(o); json.dump(rows, open(out, "w"), indent=1)
    print(g.name, "hifi ec3 length medium", round(o["K_kN_per_mm"], 3), flush=True)
print("done", flush=True)
