"""The CMAME manuscript (paper/cmame): its tables, figures and in-text numbers
are generated from the committed records only and reproduce byte for byte; the
numbers agree with the records when recomputed independently; the text carries
no hand-typed result; the design constants it states match the configurations
and the code; the statements of its Section 3 hold on assembled instances; and
the manuscript compiles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "cmame"
OUT = PAPER / "generated"
TEXT = ("table_2d.tex", "table_3d.tex", "table_label_efficiency.tex", "table_transfer.tex",
        "table_criteria.tex", "table_amplitude.tex", "table_remesh.tex", "table_e2.tex",
        "table_e1.tex", "numbers.tex", "hashes.tex", "material.md", "sources.json")
FIGS = ("fig_disp_vs_stress.pdf", "fig_energy_gap.pdf", "fig_label_efficiency.pdf",
        "fig_e2_cost_accuracy.pdf")
R2D = ROOT / "records" / "phase1" / "report_rec8_v2.json"
RDIAG = ROOT / "records" / "phase1" / "report_diag.json"
RWP2 = ROOT / "records" / "phase1" / "report_wp2_e2.json"
R3D = ROOT / "records" / "wp8" / "e2" / "baseline" / "report_phase2b.json"


def _mod():
    spec = importlib.util.spec_from_file_location(
        "make_cmame_material", ROOT / "scripts" / "make_cmame_material.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built():
    return _mod().build()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _f(s: str) -> float:
    """A macro's numeric value (thousands separators, %, typographic minus removed)."""
    return float(s.replace(",", "").replace("%", "").replace("\u2212", "-"))


# ------------------------------------------------------------------ records
def test_phase1_records_match_their_provenance():
    readme = (ROOT / "records" / "phase1" / "README.md").read_text(encoding="utf-8")
    for p in (R2D, RDIAG, RWP2):
        assert f"`{p.name}` | `{_sha(p)}`" in readme, p.name
    # the deciding run's and the WP2/E2 run's reports are the ones the provenance note
    # attests (Runs 1 and 3)
    prov = re.sub(r"\s+", "", (ROOT / "PROVENANCE_NOTE.md").read_text(encoding="utf-8"))
    assert _sha(R2D) in prov and _sha(RWP2) in prov
    for p, cfg in ((R2D, "62b26ad868d424ef5527c8cb7d826c818aa1ba5cebbc76c7bfe665062781f0ce"),
                   (RDIAG, "1a85eeabf2646568e8ac17f3119abe980f56852cf44b05b684b72fa3d1619b78"),
                   (RWP2, "60088135b1fecb2768d882ad6fee37add0fc3463160decee98a34b511159cb2c")):
        assert json.loads(p.read_bytes())["provenance"]["config_sha256"] == cfg


# ------------------------------------------------------------------ regeneration
def test_material_regenerates_byte_for_byte(built):
    files = built["files"]
    assert set(files) == set(TEXT)
    for name, text in files.items():
        assert (OUT / name).read_text(encoding="utf-8") == text, name
    for name in FIGS:
        assert (OUT / name).stat().st_size > 1000, name


def test_sources_name_every_record_read(built):
    src = json.loads(built["files"]["sources.json"])["inputs_sha256"]
    for p in (R2D, RDIAG, RWP2, R3D, ROOT / "records" / "wp8" / "posthoc" / "amp_phase2b.json",
              ROOT / "DEVIATIONS_PHASE2.md"):
        rel = str(p.relative_to(ROOT))
        assert src[rel] == _sha(p), rel


# ------------------------------------------------------------------ numbers
def _cells(r):
    return r["results"]["e8"]["metrics"]["cells"]


def _seeds(r, arm, metric, b="1024"):
    c = _cells(r)[arm]
    return c[b if b in c else max(c, key=int)][metric]["per_seed"]


def _inst(r, arm, metric, b="1024"):
    c = _cells(r)[arm]
    return np.array([e["per_instance"][metric]
                     for e in c[b if b in c else max(c, key=int)]["per_seed_eval"]])


def test_numbers_agree_with_the_records(built):
    N = built["numbers"]
    r2, r3 = json.loads(R2D.read_bytes()), json.loads(R3D.read_bytes())
    for dim, r in (("Two", r2), ("Three", r3)):
        for arm, tag in (("ar", "Free"), ("labels", "Lab"), ("labels_anchor", "Anc"),
                         ("mgn", "Mgn")):
            for m, mt in (("disp_rel_l2", "Disp"), ("energy_gap_rel", "Gap"),
                          ("vm_rel_l2", "Vm"), ("peak_vm_rel_err", "Peak")):
                assert _f(N[f"num{dim}{tag}{mt}"]) == pytest.approx(
                    statistics.fmean(_seeds(r, arm, m)), rel=5e-3), (dim, arm, m)
            g = _inst(r, arm, "energy_gap_rel")
            assert _f(N[f"num{dim}{tag}Worse"]) == int((g > 1).sum())
    # the headline ratios and per-instance comparisons of Section 6
    vm = statistics.fmean(_seeds(r3, "mgn", "vm_rel_l2")) / statistics.fmean(
        _seeds(r3, "ar", "vm_rel_l2"))
    assert _f(N["numThreeMgnOverFreeVm"]) == pytest.approx(vm, abs=0.05)
    assert _f(N["numMgnVmHigher"]) == int((_inst(r3, "mgn", "vm_rel_l2")
                                          > _inst(r3, "ar", "vm_rel_l2")).sum())
    lab = _inst(r3, "labels", "energy_gap_rel")
    assert int(N["numWorstIndex"]) == int(np.argmax(lab[0])) == int(np.argmax(lab[1]))
    assert _f(N["numThreeLabWorst"]) == pytest.approx(lab.max(), abs=0.5)
    # the 2D deciding run's kill records
    e7 = r2["results"]["e7"]["metrics"]["iterations_to_tol_mean"]
    assert _f(N["numKfiveZero"]) == pytest.approx(e7["zero"], abs=0.05)
    assert _f(N["numKfiveFree"]) == pytest.approx(e7["learned"], abs=0.05)


def test_highlights_match_the_numbers(built):
    lines = (PAPER / "highlights.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 and all(0 < len(x) <= 85 for x in lines)
    assert f"{built['numbers']['numThreeMgnOverFreeVm']} times the label-free stress error" \
        in lines[2]
    assert "zero labels beat 1,024 labels on all five metrics" in lines[1]


def test_abstract_has_no_numbers_and_at_most_250_words():
    text = (PAPER / "sections" / "abstract.tex").read_text(encoding="utf-8")
    assert not re.search(r"\d", text)
    assert len(text.split()) <= 250


def _prose(tex: str) -> str:
    """The text of a section without comments, macro names, labels, references,
    graphics paths and TikZ options."""
    tex = "\n".join(line.split("%")[0] if not line.lstrip().startswith("%") else ""
                    for line in tex.replace("\\%", "PCT").splitlines())
    tex = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", "", tex, flags=re.S)
    tex = re.sub(r"\\(ref|cref|Cref|label|eqref|cite|input|includegraphics|path)"
                 r"(\[[^\]]*\])?\{[^}]*\}", "", tex)
    return re.sub(r"\\[A-Za-z]+", " ", tex)


# decimals and percentages the text may state: design constants of the problems,
# networks and protocol, each checked against the configurations and code below
ALLOWED_DECIMALS = {"0.05", "0.5", "1.5", "0.25", "0.38", "0.8", "0.06", "0.16", "0.12", "0.6",
                    "1.2", "0.08", "0.0579", "0.0906", "0.0374", "1.8", "2.5"}
ALLOWED_PERCENT = {"5", "10", "20", "30", "40"}


def test_the_text_carries_no_hand_typed_result():
    bad = []
    for f in sorted((PAPER / "sections").glob("*.tex")):
        prose = _prose(f.read_text(encoding="utf-8"))
        for m in re.finditer(r"(?<![\w.])(\d+\.\d+)(?![\w.])", prose):
            if m.group(1) not in ALLOWED_DECIMALS:
                bad.append(f"{f.name}: {m.group(1)}")
        for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)PCT", prose):
            if m.group(1) not in ALLOWED_PERCENT:
                bad.append(f"{f.name}: {m.group(1)}%")
    assert not bad, "numbers typed by hand (use generated/numbers.tex):\n" + "\n".join(bad)


def test_design_constants_match_the_configurations_and_code():
    c2 = json.loads((ROOT / "configs" / "phase1_rec8_v2.json").read_text())
    c3 = json.loads((ROOT / "configs" / "phase2b_v1.json").read_text())
    assert (c2["data"]["n"], c2["split"]["n_val"]) == (30000, 256)
    assert (c3["data"]["n"], c3["split"]["n_val"]) == (2000, 256)
    assert c3["data"]["lc_range"] == [0.0579, 0.0906] and c3["data_transfer"]["lc"] == 0.0374
    assert c3["data_transfer"]["n"] == 320 and c3["data_transfer"]["split"] == {
        "n_eval": 256, "n_fewshot_prefix": 64}
    # every label is a sparse direct solve: the corpora are generated without labels
    # and labelled by the runner's direct path; no code reads the labels' cg_tol key
    assert c3["data"]["labelled_policy"] == c3["data_transfer"]["labelled_policy"] == "economy"
    run = (ROOT / "src" / "fejepa" / "experiments" / "runner.py").read_text(encoding="utf-8")
    assert 'labelled = "all" if dcfg.get("labelled_policy") == "all" else "none"' in run
    assert 'solve_fe_displacement(arch.K, arch.F, arch.free_mask, method="direct")' in run
    assert not [q for q in (ROOT / "src").rglob("*.py") if "cg_tol" in q.read_text("utf-8")]
    gen2 = (ROOT / "src" / "fejepa" / "fe" / "generator.py").read_text(encoding="utf-8")
    assert 'method="direct", ledger=ledger,' in gen2
    for c in (c2, c3):
        m = c["model"]
        assert (m["dim"], m["depth"], m["heads"], m["mgn_dim"], m["mgn_depth"]) == (256, 8, 8,
                                                                                   256, 8)
        e8 = c["experiments"]["e8"]
        assert e8["budgets"] == [16, 64, 256, 1024] and e8["pool_sizes"] == [1024]
        assert (e8["seeds"] if "seeds" in e8 else len(c["seeds"])) == 3
        assert (e8["ar_epochs"], e8["sup_epochs"]) == (200, 200)
        assert (c["pretrain"]["lr"], c["sup"]["lr"]) == (1e-3, 1.5e-3)
    assert c3["experiments"]["e8"]["mgn_budgets"] == [64, 1024]
    p3 = c3["experiments"]["p3_transfer"]
    assert (p3["fewshot_epochs"], p3["fewshot_lr"], p3["fewshot_budgets"]) == (50, 1.5e-3,
                                                                              [16, 64])
    e7, e4 = c2["experiments"]["e7"], c2["experiments"]["e4"]
    assert (e7["pre_epochs"], e7["n_eval"], e7["tol"]) == (100, 64, 1e-6)
    assert (e4["coarsens"], e4["n_train"], e4["n_val"]) == ([1.8, 2.5], 512, 128)
    # the settings of the warm-start and cross-resolution tests (Section 5), as run
    r2 = json.loads(R2D.read_bytes())["results"]
    pre = r2["e4"]["protocol"]["pretrain"]
    assert (pre["epochs"], pre["lr"], pre["seed"]) == (100, 1e-3, 0)
    assert all((row["n_train"], row["n_val"]) == (512, 128)
               for row in r2["e4"]["metrics"]["per_coarsen"])
    assert "seed" not in e7 and "seed" not in e4
    e7src = (ROOT / "src" / "fejepa" / "experiments" / "e7_polish.py").read_text("utf-8")
    assert 'seeded_factory(model_factory, int(cfg.get("seed", 0)))' in e7src
    assert (r2["e7"]["protocol"]["tol"], r2["e7"]["protocol"]["n_eval"]) == (1e-6, 64)
    g = c3["gate_g2"]
    assert (g["parity_band"], g["egap_adv_min"], g["sanity_x"]) == (0.1, 0.4, 3.0)
    pilot = json.loads((ROOT / "configs" / "phase2b_pilot.json").read_text())
    assert pilot["experiments"]["e8"]["ar_epochs"] == 20
    # the code behind the methods section
    src = {n: (ROOT / "src" / "fejepa" / n).read_text(encoding="utf-8") for n in (
        "fe/generator.py", "fe/gmsh3d.py", "train/schedule.py", "train/pretrain.py",
        "train/supervised.py", "experiments/e8_regimes.py")}
    for lit in ("rng.uniform(1.5, 3.0)", "rng.uniform(0.8, 1.5)", "rng.uniform(0.25, 0.38)",
                "rng.integers(0, 4)", "rng.uniform(0.06, 0.16)", "rng.uniform(0.05, 0.12)",
                "0.05 * rng.uniform(0.5, 1.5, size=4)"):
        assert lit in src["fe/generator.py"], lit
    for lit in ("rng.uniform(0.6, 1.2)", "rng.uniform(0.08, 0.16)",
                "0.05 * rng.uniform(0.5, 1.5, size=4)"):
        assert lit in src["fe/gmsh3d.py"], lit
    assert "warmup_frac: float = 0.05" in src["train/schedule.py"]
    for n in ("train/pretrain.py", "train/supervised.py"):
        assert "weight_decay: float = 1e-4" in src[n] and "clip: float = 1.0" in src[n]
    assert "POLICY_BALANCED_FROM = 64" in src["experiments/e8_regimes.py"]
    # stated thresholds of the pre-registered kills
    kills = {k["condition"] for e in r_kills() for k in e}
    assert any("> 30% worse" in k for k in kills) and any("< 40%" in k for k in kills)
    assert any(k.startswith("K5") and "< 20%" in k for k in kills)
    led = (ROOT / "DEVIATIONS_PHASE2.md").read_text(encoding="utf-8")
    assert "1,744" in led and "720 in-band instances" in led
    pre = (ROOT / "PREREG_PHASE2.md").read_text(encoding="utf-8")
    assert "10k-30k dof" in pre and "100,182" in pre


def r_kills():
    r2 = json.loads(R2D.read_bytes())
    return [v["kills"] for v in r2["results"].values()]


def test_network_sizes_match_the_modules(built):
    """The parameter counts of Section 4 (by formula in the generator) are those of
    the torch modules the runs built, in both dimensions."""
    pytest.importorskip("torch")
    from fejepa.models.fejepa import FEJEPAConfig, build_fejepa
    from fejepa.models.gnn import build_mesh_gnn

    gen, N = _mod(), built["numbers"]
    n = lambda mod: sum(q.numel() for q in mod.parameters())                  # noqa: E731
    for cfg_name in ("phase1_rec8_v2", "phase2b_v1"):
        c = json.loads((ROOT / "configs" / f"{cfg_name}.json").read_text())["model"]
        cfg = FEJEPAConfig.from_dict(c)
        f, sd = cfg.features.dim, int(cfg.features.spatial_dim)
        m = build_fejepa(cfg)
        used, aux = gen._params_transformer(f=f, sd=sd)
        assert (n(m.encoder) + n(m.decoder), n(m.predictor) + n(m.proj)) == (used, aux)
        assert n(m) == used + aux
        g = build_mesh_gnn(dim=c["mgn_dim"], depth=c["mgn_depth"], features=cfg.features)
        assert n(g) == gen._params_graph(f=f, sd=sd)
        # the text states one value for both dimensions
        assert (f"{used / 1e6:.1f}", f"{aux / 1e6:.1f}", f"{n(g) / 1e6:.1f}") == (
            N["numParamsTransformer"], N["numParamsAux"], N["numParamsGraph"])


# ------------------------------------------------------------------ theory
def _instances():
    from fejepa.fe.synthetic import synthetic_instance
    from fejepa.fe.tet3d import tet_instance

    rng = np.random.default_rng(7)
    return [synthetic_instance(rng, labelled=True), tet_instance(rng, labelled=True)]


def _stresses(a, u):
    m = a.meta["material"]
    if a.nodes.shape[1] == 3:
        from fejepa.fe.tet3d import _tet_geometry, tet_stresses, tet_von_mises

        vol, _ = _tet_geometry(a.nodes, a.elements)
        s = tet_stresses(a.nodes, a.elements, u, m)
        return vol, tet_von_mises(a.nodes, a.elements, u, m), s[:, :3].sum(axis=1) / 3
    from fejepa.fe.stress import _geometry, element_stresses, element_von_mises

    area, _, _ = _geometry(a.nodes, a.elements)
    s = element_stresses(a.nodes, a.elements, u, m)
    return area, element_von_mises(a.nodes, a.elements, u, m), (s[:, 0] + s[:, 1]) / 3


def test_proposition_1_energy_gap_and_stress_error():
    rng = np.random.default_rng(1)
    for a in _instances():
        E, nu = a.meta["material"]["E"], a.meta["material"]["nu"]
        G, B = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
        free = ~a.dirichlet_mask
        for j in range(a.n_loads):
            us = a.U_star[j]
            for scale in (1e-3, 0.1, 1.0):
                v = rng.normal(size=us.size) * free * np.abs(us).max() * scale
                meas, vm_v, p_v = _stresses(a, v)
                # (eq:stressnorm): v'Kv = int vm^2/(3G) + p^2/B
                assert v @ (a.K @ v) == pytest.approx(
                    float(np.sum(meas * (vm_v ** 2 / (3 * G) + p_v ** 2 / B))), rel=1e-10)
                u = us + v
                _, vm_u, _ = _stresses(a, u)
                _, vm_s, p_s = _stresses(a, us)
                lhs = float(np.sum(meas * (vm_u - vm_s) ** 2))
                assert lhs <= 3 * G * (v @ (a.K @ v)) * (1 + 1e-12)          # (eq:vmbound)
                gamma = 3 * G * float(np.sum(meas * p_s ** 2)) / (B * float(np.sum(meas * vm_s ** 2)))
                g = (v @ (a.K @ v)) / (us @ (a.K @ us))
                assert lhs / float(np.sum(meas * vm_s ** 2)) <= (1 + gamma) * g * (1 + 1e-12)


def test_proposition_2_and_corollary_3_amplitude_and_zero_field():
    rng = np.random.default_rng(2)
    for a in _instances():
        free = ~a.dirichlet_mask
        for j in range(a.n_loads):
            us, F = a.U_star[j], a.F[j] * free
            pi = lambda w: 0.5 * w @ (a.K @ w) - F @ w                        # noqa: E731
            nus = us @ (a.K @ us)
            for scale in (0.05, 0.7, 1.3, 3.0):
                u = (us * rng.uniform(0.2, 2.0)
                     + scale * rng.normal(size=us.size) * free * np.abs(us).max())
                ku = u @ (a.K @ u)
                c = (F @ u) / ku
                assert pi(c * u) == pytest.approx(-(F @ u) ** 2 / (2 * ku), rel=1e-9)
                assert pi(c * u) <= min(pi(u), 0.0) + 1e-15 * abs(pi(us))
                cos = (u @ (a.K @ us)) / np.sqrt(ku * nus)
                g_c = ((c * u - us) @ (a.K @ (c * u - us))) / nus
                assert g_c == pytest.approx(1 - cos ** 2, abs=1e-10)
                g = ((u - us) @ (a.K @ (u - us))) / nus
                assert (pi(u) > 0) == (g > 1)                                 # Corollary 3
                w = us + rng.normal(size=us.size) * free * np.abs(us).max() * 0.1
                g_w = ((w - us) @ (a.K @ (w - us))) / nus
                assert (pi(u) - pi(w)) == pytest.approx(0.5 * nus * (g - g_w), rel=1e-8)


def test_corollary_4_warm_start_iteration_bound():
    """Conjugate gradients from a warm start with relative energy gap g0 reach the
    target g_bar within the iterations Corollary 4 states, the Chebyshev bound holds
    along the way, and the bound's saving is the fraction of eq:warm."""
    rng = np.random.default_rng(4)
    g_bar, tested = 1e-4, 0
    for a in _instances():
        free = ~a.dirichlet_mask
        K = a.K[free][:, free].toarray()
        lam = np.linalg.eigvalsh(K)
        rho = (np.sqrt(lam[-1] / lam[0]) - 1) / (np.sqrt(lam[-1] / lam[0]) + 1)
        for j in range(a.n_loads):
            us = a.U_star[j][free]
            F = K @ us
            nus = us @ F
            for scale in (1e-3, 1e-2, 3e-2):
                x = us + scale * rng.normal(size=us.size) * np.abs(us).max()
                g0 = ((x - us) @ K @ (x - us)) / nus
                if not g_bar < g0 <= 1:
                    continue
                k_warm = np.log(4 * g0 / g_bar) / (2 * np.log(1 / rho))
                k_zero = np.log(4 / g_bar) / (2 * np.log(1 / rho))
                assert (k_zero - k_warm) / k_zero == pytest.approx(
                    np.log(1 / g0) / np.log(4 / g_bar), rel=1e-12)            # eq:warm
                r = F - K @ x
                d, rr = r.copy(), r @ r
                for k in range(1, int(np.ceil(k_warm)) + 1):
                    if np.sqrt(rr) < 1e-14 * np.linalg.norm(F):
                        break
                    Kd = K @ d
                    alpha = rr / (d @ Kd)
                    x, r = x + alpha * d, r - alpha * Kd
                    rr, rr_old = r @ r, rr
                    d = r + (rr / rr_old) * d
                    g = ((x - us) @ K @ (x - us)) / nus
                    assert g <= 4 * g0 * rho ** (2 * k) * (1 + 1e-6) + 1e-14    # eq:cheb
                assert ((x - us) @ K @ (x - us)) / nus <= g_bar * (1 + 1e-6)
                tested += 1
    assert tested >= 4


def test_remarks_2_and_3_rank_and_residual_bounds():
    rng = np.random.default_rng(3)
    for a in _instances():
        free = ~a.dirichlet_mask
        K = a.K[free][:, free].toarray()
        lam = np.linalg.eigvalsh(K)
        kappa = lam[-1] / lam[0]
        for _ in range(20):
            ea, eb = rng.normal(size=(2, K.shape[0])) * rng.uniform(0.01, 1, size=(2, 1))
            ratio = (np.sqrt(ea @ K @ ea / (eb @ K @ eb))
                     / (np.linalg.norm(ea) / np.linalg.norm(eb)))
            assert kappa ** -0.5 * (1 - 1e-12) <= ratio <= kappa ** 0.5 * (1 + 1e-12)
        for j in range(a.n_loads):
            us = a.U_star[j][free]
            F = K @ us
            e0 = rng.normal(size=us.size) * np.abs(us).max() * 0.1
            g0 = (e0 @ K @ e0) / (us @ K @ us)
            r0 = K @ e0
            assert np.linalg.norm(r0) / np.linalg.norm(F) <= np.sqrt(kappa * g0) * (1 + 1e-12)


# ------------------------------------------------------------------ compile
@pytest.mark.skipif(shutil.which("latexmk") is None or shutil.which("pdflatex") is None,
                    reason="no LaTeX toolchain")
def test_manuscript_compiles(tmp_path):
    kpse = shutil.which("kpsewhich")
    if kpse is None or not subprocess.run([kpse, "elsarticle.cls"], capture_output=True,
                                          text=True).stdout.strip():
        pytest.skip("elsarticle not installed")
    work = tmp_path / "cmame"
    shutil.copytree(PAPER, work, ignore=shutil.ignore_patterns(
        "*.aux", "*.log", "*.fls", "*.fdb_latexmk", "*.out", "*.bbl", "*.blg", "*.pdf.tmp"))
    for f in work.glob("main.pdf"):
        f.unlink()
    res = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                          "main.tex"], cwd=work, capture_output=True, text=True, timeout=600)
    log = (work / "main.log").read_text(encoding="latin-1")
    assert res.returncode == 0, log[-3000:]
    assert (work / "main.pdf").stat().st_size > 100_000
    assert not re.search(r"(Reference|Citation) .* undefined", log)
    assert "Float too large" not in log
