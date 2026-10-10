"""Exploratory (post hoc to G-C1a): the non-preloaded T-stub pairs T1 and T2 of Bursi and Jaspart (1997),
Fig. 1, modelled with the joint model's machinery. Reported against initial slopes read by eye from the
test curves of their Figs 14 and 18 (reading uncertainty large, about +-25%)."""
import sys, json, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fejoint.tstub import TStubPair, analyse

T1 = TStubPair(t_f=10.7, t_w=7.1, b=150.0, L=80.0, w=90.0, bolt_y=(20.0,), H=220.0, corner="root", r=15.0,
               d=12.0, As=84.3, head=7.5, nut=10.8, bearing_r=12.0, n_washers=2, washer_t=2.5, x_meas=150.0)
# T2: gauge base d = 70 mm (Bursi and Jaspart Fig. 1(b)), so the measuring points are 35 mm from the contact plane;
# washers only under the nuts, so the head side bears on the head (bearing radius about 8.3 mm): both radii are run.
T2 = TStubPair(t_f=16.0, t_w=9.5, b=150.0, L=80.0, w=90.0, bolt_y=(20.0,), H=220.0, corner="root", r=18.0,
               d=12.0, As=84.3, head=7.5, nut=10.8, bearing_r=8.3, n_washers=1, washer_t=2.5, x_meas=35.0)
from dataclasses import replace
T2w = replace(T2, bearing_r=12.0)
rows = []
for name, g in (("T1", T1), ("T2 (r 8.3)", T2), ("T2 (r 12)", T2w)):
    for mesh in (dict(h=5.0, n_tf=2), dict(h=2.5, n_tf=4)):
        t0 = time.perf_counter()
        o = analyse(g, **mesh)
        o.update({"name": name, "seconds": round(time.perf_counter() - t0, 1)})
        rows.append(o)
        print(f"{name} h {mesh['h']} dof {o['mesh']['dof']} K {o['K_kN_per_mm']:.1f} kN/mm  bolt force/F {o['bolt_force_total_N']/o['F_N']:.3f}"
              f"  contact {o['contact_fraction']:.2f}  it {o['pdas_iterations']}  kkt {max(abs(v) for v in o['kkt'].values()):.1e}  {o['seconds']} s", flush=True)
json.dump(rows, open(pathlib.Path(__file__).with_suffix(".json"), "w"), indent=1)
