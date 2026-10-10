"""Checks of the C1.3 joint family and of the nodal export (fejoint/family.py, fejoint/export.py)."""
import numpy as np
import pytest

from fejoint import family
from fejoint.family import sample, geometry, violations, RANGES, FamilyParams
from fejoint.export import export, solve_exported, energy, stiffness
from fejoint.joint import analyse
from fejoint.specimens import specimen


def test_sampler_deterministic_valid_and_in_range():
    a, na = sample(64, seed=7)
    b, nb = sample(64, seed=7)
    assert a == b and na == nb
    c, _ = sample(64, seed=8)
    assert a != c
    for q in a:
        assert not violations(geometry(q))
        for k, (lo, hi) in RANGES.items():
            assert lo <= getattr(q, k) <= hi


def test_tested_details_are_valid_members():
    for name, q in family.tested_details().items():
        g = geometry(q)
        assert not violations(g), name
        assert g.bp == g.bb                                   # overhang below the snap: plate as wide as the flanges


def test_invalid_layout_is_rejected():
    q = FamilyParams(tp=10.0, eX=30.0, p=90.0, p23=205.0, w=30.0, o=0.0, LX=70.0, c_bot=31.0)
    assert any("web" in v for v in violations(geometry(q)))


@pytest.mark.parametrize("which,seg", [("FS1", None), ("family", None), ("family", 300.0)])
def test_export_reproduces_the_joint_model(which, seg):
    """The condensed nodal problem has the solution of fejoint.joint.analyse (contact, head_contact), for a tested
    layout and for a family member, with the full beam and with the 300 mm segment."""
    g = specimen("FS1") if which == "FS1" else geometry(sample(1, seed=3)[0][0])
    ex = export(g, h=10.0, n_tp=1, n_tf=1, beam_segment=seg)
    s = solve_exported(ex)
    o, d = analyse(g, h=10.0, n_tp=1, n_tf=1, support="contact", washer="head_contact", k_rot_factor=4.0, reg=1e-6,
                   beam_segment=seg)
    assert abs(s["S_paper"] / o["S_paper_kNm_per_mrad"] - 1) < 1e-9
    own = ~ex["dirichlet"][1::3]                                  # nodes whose u_y is an unknown of its own
    uy = d["u"][1::3]
    assert np.abs(s["u"][1::3][own] - uy[own]).max() < 1e-9 * np.abs(uy).max()
    if seg is None:                                               # Clapeyron for the cone: Pi* = -1/2 f^T u*
        assert np.isclose(s["Pi"], -0.5 * ex["F"] @ s["u"], rtol=1e-9)


def test_masks_and_energy_identity():
    """Pi(v) - Pi* = 1/2 ||v - u*||_S^2 + lambda*^T v on the constrained unknowns, for any v (Dirichlet zeroed)."""
    ex = export(specimen("FS3"), h=10.0, n_tp=1, n_tf=1)
    assert not (ex["nonneg"] & ex["dirichlet"]).any()
    assert ex["nonneg"][0::3].sum() == ex["nonneg"].sum()        # only x slots are constrained
    s = solve_exported(ex)
    u = s["u"]
    assert u[ex["nonneg"]].min() >= -1e-12 * np.abs(u).max()
    lam = ex["K"] @ u - ex["F"]
    free = ~ex["dirichlet"]
    assert np.abs(lam[free & ~ex["nonneg"]]).max() < 1e-8 * np.abs(ex["F"]).max()
    rng = np.random.default_rng(0)
    v = u + 1e-2 * np.abs(u).max() * rng.standard_normal(u.size)
    v[ex["dirichlet"]] = 0.0
    e = v - u
    lhs = energy(ex, v) - s["Pi"]
    rhs = 0.5 * e @ (ex["K"] @ e) + lam[ex["nonneg"]] @ v[ex["nonneg"]]
    assert np.isclose(lhs, rhs, rtol=1e-8, atol=1e-10 * abs(s["Pi"]))
    # the reading is linear in the deflection at small loads: scaling the field scales phi
    assert np.isclose(stiffness(ex, u), s["S_paper"])


def test_export_with_beam_segment_reproduces_the_joint_model():
    """The segment variant (rigid, loaded end section condensed) has the solution of analyse(beam_segment=...)."""
    g = specimen("FS1")
    ex = export(g, h=10.0, n_tp=1, n_tf=1, beam_segment=300.0)
    s = solve_exported(ex)
    o, d = analyse(g, h=10.0, n_tp=1, n_tf=1, support="contact", washer="head_contact", k_rot_factor=4.0, reg=1e-6,
                   beam_segment=300.0)
    assert abs(s["S_paper"] / o["S_paper_kNm_per_mrad"] - 1) < 1e-9
    own = ~ex["dirichlet"][1::3]                                  # nodes whose u_y is an unknown of its own
    uy = d["u"][1::3]
    assert np.abs(s["u"][1::3][own] - uy[own]).max() < 1e-9 * np.abs(uy).max()
    m = ex["masters"]["m0"] + ex["masters"]["m_from_x"] @ np.where(ex["dirichlet"], 0.0, s["u"])
    assert np.isclose(s["Pi"], -0.5 * ex["masters"]["f_m"] @ m, rtol=1e-9)    # Clapeyron with the section's load


@pytest.mark.parametrize("seg", [None, 300.0])
def test_physical_displacement_matches_the_joint_model(seg):
    """Every displacement component, including the patches' u_x and a segment's end section, as analyse gives it."""
    from fejoint.export import physical_displacement
    g = specimen("FS2")
    ex = export(g, h=10.0, n_tp=1, n_tf=1, beam_segment=seg)
    s = solve_exported(ex)
    o, d = analyse(g, h=10.0, n_tp=1, n_tf=1, support="contact", washer="head_contact", k_rot_factor=4.0, reg=1e-6,
                   beam_segment=seg)
    u = physical_displacement(ex, s["u"])
    assert np.abs(u - d["u"]).max() < 1e-9 * np.abs(d["u"]).max()
