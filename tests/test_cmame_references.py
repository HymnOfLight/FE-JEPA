"""cmame-paper Stage 10f: the references checked on 11 October 2026, and the archive cited as
a data reference.

The six references checked against the PDFs of their published versions carry a comment
saying so: the chapter of LeCun et al. is cited as published (2007), with its six editors and
its pages, instead of the authors' version, and Saad (2003) has its DOI. The Zenodo archive of
the version described is a data reference, rendered with "[dataset]" before it as the
journal's guide asks, with the title, author and DOI of CITATION.cff and the tag that its
identifier names, and cited in the Data availability statement. Where the text gives the
iteration bound of a warm start for conjugate gradients, it credits the estimate of Sun et al.
in the relative residual and the bound of Wang et al. for fixed-point iterations; it states their solvers, their observation on
Newton residuals, the polycrystal comparison of Monteiro Fernandes et al. and the convexity
of incremental potentials as the published texts report them."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "cmame"
SECTIONS = PAPER / "sections"
BIB = PAPER / "references.bib"
DOI = "10.5281/zenodo.23298799"
TAG = "cmame-submission"
# the entries checked on 11 October 2026 against the PDFs of their published versions, and
# the preprint of Sun et al.
CHECKED = {"OrtizStainier1999": "article", "Taylor2023DFR": "article",
           "MonteiroFernandes2025": "article", "Wang2026PretrainFEM": "article",
           "Saad2003": "book", "LeCun2007EBM": "incollection", "Sun2026FEVessel": "misc"}
WANG = r"Wang et~al.~\cite{Wang2026PretrainFEM}"
SUN = r"Sun et~al.~\cite{Sun2026FEVessel}"


def _flat(text: str) -> str:
    return " ".join(text.split())


def _entries(bib: str) -> dict[str, tuple[str, str, str]]:
    """key -> (entry type, the comment lines just above the entry, the entry's fields)."""
    return {m.group(3): (m.group(2), m.group(1), m.group(4))
            for m in re.finditer(r"((?:^%[^\n]*\n)*)^@(\w+)\{([^,\n]+),\n(.*?)^\}", bib,
                                 re.M | re.S)}


def _field(body: str, name: str) -> str:
    m = re.search(rf"^\s*{name}\s*=\s*\{{(.*)\}},?\s*$", body, re.M)
    assert m, name
    return m.group(1)


def _sections() -> dict[str, str]:
    return {f.name: _flat(f.read_text(encoding="utf-8")) for f in SECTIONS.glob("*.tex")}


def test_the_archive_is_a_data_reference_with_the_metadata_of_the_citation_file():
    kind, _, body = _entries(BIB.read_text(encoding="utf-8"))["Cao2026Zenodo"]
    assert kind == "misc"
    assert _field(body, "author") == r"Cao, {\relax [dataset] R}uifeng"
    assert (_field(body, "title"), _field(body, "howpublished"), _field(body, "year"),
            _field(body, "doi")) == ("label-free-fe-surrogates", f"Zenodo, version {TAG}",
                                     "2026", DOI)
    # the record to be published under the reserved DOI carries what CITATION.cff says
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for s in ('title: "label-free-fe-surrogates"\n', f"doi: {DOI}\n",
              "authors:\n  - family-names: Cao\n    given-names: Ruifeng\n",
              f'description: "Zenodo archive of the version tagged {TAG}"\n'):
        assert s in cff, s
    statement = _flat(_sections()["declarations.tex"]
                      .split(r"\section*{Data availability}", 1)[1].split(r"\section*", 1)[0])
    assert (rf"tagged \texttt{{{TAG}}} and archived at Zenodo \cite{{Cao2026Zenodo}}, "
            rf"\url{{https://doi.org/{DOI}}}.") in statement


def test_the_bibliography_renders_the_data_reference_and_the_chapter(tmp_path):
    bibtex, kpse = shutil.which("bibtex"), shutil.which("kpsewhich")
    if bibtex is None or kpse is None or not subprocess.run(
            [kpse, "elsarticle-num.bst"], capture_output=True, text=True).stdout.strip():
        pytest.skip("bibtex or elsarticle-num.bst not installed")
    shutil.copy(BIB, tmp_path / "references.bib")
    (tmp_path / "t.aux").write_text(
        "\\citation{Cao2026Zenodo}\n\\citation{LeCun2007EBM}\n\\citation{Saad2003}\n"
        "\\bibstyle{elsarticle-num}\n\\bibdata{references}\n", encoding="ascii")
    res = subprocess.run([bibtex, "t"], cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "Warning--" not in res.stdout, res.stdout     # bibtex exits 0 on warnings too
    bbl = _flat((tmp_path / "t.bbl").read_text(encoding="utf-8"))
    for s in (r"\bibitem{Cao2026Zenodo} {\relax [dataset] R}.~Cao, label-free-fe-surrogates, "
              rf"Zenodo, version {TAG} (2026). \newblock \href {{https://doi.org/{DOI}}} "
              rf"{{\path{{doi:{DOI}}}}}.",
              r"\bibitem{LeCun2007EBM} Y.~LeCun, S.~Chopra, R.~Hadsell, M.~Ranzato, F.~J. Huang, "
              r"Energy-based models, in: G.~Bak{\i}r, T.~Hofmann, B.~Sch{\"o}lkopf, A.~J. "
              r"Smola, B.~Taskar, S.~V.~N. Vishwanathan (Eds.), Predicting Structured Data, "
              r"MIT Press, Cambridge, MA, 2007, pp. 191--246. \newblock \href "
              r"{https://doi.org/10.7551/mitpress/7443.003.0014}",
              r"\bibitem{Saad2003} Y.~Saad, Iterative Methods for Sparse Linear Systems, 2nd "
              r"Edition, SIAM, Philadelphia, 2003. \newblock \href "
              r"{https://doi.org/10.1137/1.9780898718003}"):
        assert s in bbl, s


def test_the_references_checked_on_11_october_say_so():
    bib = BIB.read_text(encoding="utf-8")
    entries = _entries(bib)
    for key, kind in CHECKED.items():
        assert entries[key][0] == kind, key
        assert entries[key][1].startswith("% verified 11 October 2026 against "), key
    header = _flat(re.sub(r"^% ?", "", bib.split("\n\n", 1)[0], flags=re.M))
    for s in ("the entries without a comment of their own are among these",
              "On 11 October 2026 OrtizStainier1999, Taylor2023DFR, MonteiroFernandes2025, "
              "Wang2026PretrainFEM, Saad2003 and LeCun2007EBM were checked against PDFs from "
              "their publishers' sites", "Sun2026FEVessel against the PDF of its preprint"):
        assert s in header, s
    for s in ("still to be", "to be confirmed", "lecun2006ebm"):
        assert s not in bib.lower(), s
    # the same-network comparison of Dalton et al., read in the accepted manuscript
    assert "on 11 October 2026 its comparison of the same graph network" in _flat(
        entries["Dalton2023"][1].replace("% ", ""))
    # the authors' version of the chapter is no longer an entry
    assert not any("tutorial on energy-based" in body for _, _, body in entries.values())
    # the entries without a comment of their own are those of the batch of 10 October
    batch = {"EYu2018", "Pfaff2021", "Lu2021DeepONet", "Li2021FNO", "Vaswani2017",
             "Loshchilov2019", "HestenesStiefel1952", "Ciarlet1978"}
    assert {k for k, (_, comment, _) in entries.items() if not comment} == batch
    assert all(re.match(r"% (verified|checked)\b", comment)
               for k, (_, comment, _) in entries.items()
               if comment and k != "Cao2026Zenodo")
    lecun = entries["LeCun2007EBM"][2]
    assert (_field(lecun, "title"), _field(lecun, "booktitle"), _field(lecun, "publisher"),
            _field(lecun, "address"), _field(lecun, "year"), _field(lecun, "pages"),
            _field(lecun, "doi")) == (
        "Energy-based models", "Predicting Structured Data", "MIT Press", "Cambridge, MA",
        "2007", "191--246", "10.7551/mitpress/7443.003.0014")    # the chapter's own DOI
    editors = _field(lecun, "editor").split(" and ")
    assert len(editors) == 6 and editors[-1] == "Vishwanathan, S. V. N."
    saad = entries["Saad2003"][2]
    assert (_field(saad, "isbn"), _field(saad, "doi")) == ("978-0-89871-534-7",
                                                           "10.1137/1.9780898718003")
    # each is still cited
    text = " ".join(_sections().values())
    for key in CHECKED:
        assert re.search(rf"\\cite(\[[^\]]*\])?\{{([^}}]*,)?{key}[,}}]", text), key


def test_the_text_states_the_published_versions_and_credits_the_bound():
    s = _sections()
    for name, quote in (
            ("theory.tex", r"the other statements are added here, except \cref{cor:warm}, "
                           r"which states in the energy norm an estimate that " + SUN + r" give "
                           r"in the relative residual, after a bound of " + WANG + r" for "
                           r"fixed-point iterations."),
            ("theory.tex", r"the classical bound \cite[Section~6.11.3]{Saad2003} is"),
            ("theory.tex", SUN + r" draw the same conclusion for conjugate gradients in the "
                           r"relative residual, where the constant of the bound and the conversion "
                           r"between residual and error add a bounded offset to the saving, and "
                           + WANG + r" from an asymptotic "
                           r"bound of the same form for fixed-point iterations; \cref{cor:warm} "
                           r"states it in the energy norm, without that offset, and "
                           r"\cref{rem:residual} bounds the initial residual by the energy gap."),
            ("theory.tex", WANG + r" observed the same for Newton's method: the initial "
                           r"residual of their predictions could exceed that of the zero "
                           r"initial guess although their errors were much smaller."),
            ("introduction.tex", r"(\cref{cor:warm}); " + SUN + r" give this estimate in the "
                                 r"relative residual, after a bound of " + WANG + r" for "
                                 r"fixed-point iterations."),
            ("related.tex", r"Two of these studies explain why the saving shrinks as the "
                            r"tolerance tightens: " + WANG + r" with an asymptotic iteration "
                            r"bound for fixed-point iterations, and " + SUN + r" with its "
                            r"extension to conjugate gradients, in the relative residual, and a "
                            r"model of the solver's set-up cost. \Cref{cor:warm} states the "
                            r"bound for conjugate gradients in the energy norm"),
            ("related.tex", r"the supervised one was more accurate in displacement and in the "
                            r"$H^1$ seminorm \cite{Sun2026FEVessel}"),
            ("related.tex", r"in polycrystals the label-free network was more accurate in "
                            r"stress than the supervised one, which was trained on the "
                            r"reference stresses \cite{MonteiroFernandes2025}"),
            ("discussion.tex", WANG + r" and " + SUN + r" explain why the benefit shrinks as "
                               r"the tolerance tightens: a warm start lowers an iteration bound by a "
                               r"fixed number of iterations, as Wang et~al.\ show for fixed-point "
                               r"iterations and Sun et~al., in the relative residual, also for "
                               r"conjugate gradients. The bound is not the count: with conjugate "
                               r"gradients, the warm start of Sun et~al.\ saved 580 iterations at "
                               r"$10^{-3}$ but 69 at $10^{-4}$ (their Table 3). \Cref{cor:warm} "
                               r"states the bound for conjugate gradients in the energy norm"),
            ("discussion.tex", r"In the pressure vessels of " + SUN + r", the same point-cloud "
                               r"transformer trained on the relative error of the displacement "
                               r"labels instead of the energy, with every other setting "
                               r"unchanged, was more accurate in displacement and in the $H^1$ "
                               r"seminorm (their Table 2)"),
            ("discussion.tex", SUN + r" report iteration counts and wall-clock speed-ups at three "
                               r"relative-residual tolerances, the speed-ups computed from times "
                               r"that include the network's forward pass and, for algebraic "
                               r"multigrid, its set-up."),
            ("related.tex", r"Three train the same network on the same data with and without "
                            r"reference solutions, in single runs. In soft-tissue beams the two "
                            r"had comparable displacement errors and the label-free one smaller "
                            r"errors in energy and deformation gradient, a gap that adding the "
                            r"deformation gradient to the supervised loss narrowed but did not "
                            r"close \cite{Dalton2023}"),
            ("introduction.tex", r"have given opposite outcomes "
                                 r"\cite{Dalton2023,MonteiroFernandes2025,Sun2026FEVessel}"),
            ("methods.tex", r"the extents, each divided by the upper end of its sampling range, "
                            r"so that the absolute size is kept"),
            ("methods.tex", r"a supervised network with $n$ labelled instances took $200n$ "
                            r"steps, 3,200 with 16, and the comparisons at the smaller budgets "
                            r"differ in the number of steps as well as in the labels."),
            ("discussion.tex", WANG + r" report, for their linear problems, iteration counts "
                               r"to a relative residual of $10^{-3}$, with unpreconditioned "
                               r"conjugate gradients in two dimensions and with unpreconditioned "
                               r"and preconditioned conjugate gradients in three, and Sun et~"
                               r"al.~\cite{Sun2026FEVessel} report"),
            ("discussion.tex", r"the minimisation of an incremental potential "
                               r"\cite{OrtizStainier1999}, and variational integrators derive "
                               r"each step of a time integration from a discrete variational "
                               r"principle \cite{MarsdenWest2001}."),
            ("discussion.tex", r"at finite strains the incremental potential is in general not "
                               r"convex \cite{OrtizStainier1999}, so that its minimiser need "
                               r"not be unique."),
            ("discussion.tex", r"the discretisation invariance reported for point-cloud "
                               r"operators trained on the energy"),
            ("discussion.tex", r"compatible with $X$ at all \cite{LeCun2007EBM}."),
            ("transfer.tex", r"was tested by varying the density of the input point cloud, "
                             r"not by refining the mesh as here.")):
        assert quote in s[name], (name, quote)
    # the statements corrected, gone from where they were
    for name, gone in (
            ("related.tex", "slightly more accurate"),
            ("related.tex", "bounds it for conjugate gradients"),
            ("discussion.tex", "a preconditioned method in three"),
            ("discussion.tex", "both note"),
            ("discussion.tex", "a warm start saves a fixed number"),
            ("related.tex", "Two train the same network"),
            ("discussion.tex", "$H^1$ norm"),
            ("related.tex", "$H^1$ norm"),
            ("related.tex", "Two of these studies note"),
            ("theory.tex", r"\cref{cor:warm} after a bound"),
            ("discussion.tex", "observed that the benefit shrinks"),
            ("discussion.tex", "bounds this effect"),
            ("discussion.tex", "resolution invariance"),
            ("discussion.tex", r"variational integrators \cite{MarsdenWest2001} write each"),
            ("transfer.tex", "on the training mesh")):
        assert gone.lower() not in s[name].lower(), (name, gone)
