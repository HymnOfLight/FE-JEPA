"""WP7 E4-3D wiring tests -- multires pairs for the 3D backends.

The contract inherited from 2D: each pair is the SAME sampled BVP meshed at
two resolutions (`_fine`/`_coarse` archives; manifest carries `pairs`). The
rng-state reset must make geometry parameters AND traction scales identical
across the pair; on unstructured meshes the per-node loads differ but the
total applied force per load case is resolution-invariant (exact quadrature),
which is the strongest cheap witness of pair identity.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fejepa.data.archive import load_instance
from fejepa.fe.tet3d import generate_tet3d_multires


def _pair_checks(fine, coarse, geom_keys):
    ef, ec = fine.meta["extra"], coarse.meta["extra"]
    for k in geom_keys:
        assert ef[k] == ec[k], k                    # identical sampled geometry
    assert fine.meta["material"] == coarse.meta["material"]
    assert fine.n_nodes > coarse.n_nodes            # resolutions actually differ
    # traction totals are resolution-invariant exactly (plane faces, exact
    # quadrature); gravity totals equal rho*g times each mesh's own discrete
    # volume, so they agree only up to the cavity-facet volume difference.
    tf = fine.F.reshape(fine.n_loads, -1, 3).sum(axis=1)
    tc = coarse.F.reshape(coarse.n_loads, -1, 3).sum(axis=1)
    assert np.allclose(tf[:3], tc[:3], rtol=1e-8, atol=1e-12)
    from fejepa.fe.tet3d import _tet_geometry
    vol_f = _tet_geometry(fine.nodes, fine.elements)[0].sum()
    vol_c = _tet_geometry(coarse.nodes, coarse.elements)[0].sum()
    # rho*g identical across the pair <=> gravity total / own mesh volume equal
    assert np.isclose(abs(tf[3, 1]) / vol_f, abs(tc[3, 1]) / vol_c, rtol=1e-9)
    assert np.allclose(tf[3], tc[3], rtol=2e-2)   # discrete volumes are close


def test_tet3d_multires_pairs(tmp_path):
    out = generate_tet3d_multires(tmp_path / "m", n=3, seed=5, coarsen=2.0,
                                  nx=6, ny=4, nz=4)
    m = json.load(open(out / "manifest.json"))
    assert m["backend"] == "tet3d" and m["coarsen"] == 2.0 and len(m["pairs"]) == 3
    for rec in m["pairs"]:
        fine = load_instance(out / rec["fine"])
        coarse = load_instance(out / rec["coarse"])
        _pair_checks(fine, coarse, ("width", "height", "depth"))


def test_gmsh3d_multires_pairs(tmp_path):
    pytest.importorskip("gmsh")
    from fejepa.fe.gmsh3d import generate_gmsh3d_multires

    out = generate_gmsh3d_multires(tmp_path / "g", n=2, seed=9, coarsen=2.0,
                                   lc=0.32)
    m = json.load(open(out / "manifest.json"))
    assert m["backend"] == "gmsh3d" and m["lc"] == 0.32 and len(m["pairs"]) == 2
    for rec in m["pairs"]:
        fine = load_instance(out / rec["fine"])
        coarse = load_instance(out / rec["coarse"])
        _pair_checks(fine, coarse, ("width", "height", "depth", "holes"))
        assert coarse.meta["extra"]["lc"] == pytest.approx(0.64)


def test_cli_multires_routes_3d(tmp_path):
    from fejepa.cli import main

    out = tmp_path / "cli_m"
    main(["generate", str(out), "--n", "2", "--seed", "4", "--backend", "tet3d",
          "--multires-coarsen", "2.0"])
    m = json.load(open(out / "manifest.json"))
    assert len(m["pairs"]) == 2 and m["backend"] == "tet3d"
