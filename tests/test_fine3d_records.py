"""fine3d: the records behind PREREG_FINE3D -- the 3D scale-factor measurement
(its summary follows from its own rows, its sample is fresh and of the
training family, its remesh medians are the box's for the same geometries)
and the rules' simulations (re-run here, byte for byte)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "records" / "wp9" / "fine3d"


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scale_factor_record_follows_from_its_rows():
    sf = _mod("fine3d_scale_factor")
    rec = json.loads((REC / "scale_factor_3d.json").read_text())
    assert rec["sample_seed"] == sf.SAMPLE_SEED == 93000
    e2 = json.loads((ROOT / "configs" / "e2_m1024.json").read_text())
    assert rec["lc_range"] == e2["data"]["lc_range"] == [0.0579, 0.0906]
    assert rec["remesh_lc"] == list(sf.REMESH_LC)
    rows = rec["instances"]
    assert len(rows) == rec["sample"]["n"] == 256
    assert all(0.0579 <= r["lc"] <= 0.0906 for r in rows)
    assert all(abs(r["ratio"] - r["max_abs_F"] / r["sum_abs_F"]) <= 1e-15 * r["ratio"]
               for r in rows)
    per_geo = [{float(k): tuple(v) for k, v in g.items()} for g in rec["remesh_per_geometry"]]
    again = sf.summarise(rows, per_geo)
    assert again["sample"] == rec["sample"] and again["remesh"] == rec["remesh"]
    assert rec["remesh_geometries"] == len(per_geo) == 16


def test_the_sample_is_fresh():
    """Seed 93000 is no corpus's: the sample's first instance is not E2's
    corpus's first instance (seed 0) -- their lc already differ."""
    import numpy as np

    rec = json.loads((REC / "scale_factor_3d.json").read_text())
    rng0 = np.random.default_rng(0)
    assert abs(float(rng0.uniform(0.0579, 0.0906)) - rec["instances"][0]["lc"]) > 1e-6
    rng = np.random.default_rng(93000)
    assert float(rng.uniform(0.0579, 0.0906)) == rec["instances"][0]["lc"]


def test_remesh_medians_are_the_boxs_for_the_same_geometries():
    rec = json.loads((REC / "scale_factor_3d.json").read_text())
    for name in ("amp_e2_m1024", "amp_e2_m512", "amp_phase2b"):
        amp = json.loads((ROOT / "records" / "wp8" / "posthoc" / f"{name}.json").read_text())
        for s in amp["seeds"].values():
            box = s["remesh"]["fscale_ratio_median_by_lc"]
            assert {float(k): v for k, v in box.items()} == \
                {float(k): v["max_abs_F_ratio_median"] for k, v in rec["remesh"].items()}, name


def test_simulations_record_is_the_scripts_output(tmp_path):
    out = tmp_path / "sims.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fine3d_sims.py"), "--out",
                        str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.read_text() == (REC / "simulations.json").read_text()
