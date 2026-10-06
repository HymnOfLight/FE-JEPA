# BRANCH NOTES -- cmame-paper

Branched 3 Oct 2026 from `wp9-pool` at `414a372` (Stage 0e). Purpose: the
CMAME manuscript (`paper/cmame/`) and the box sessions it needs. `wp9-pool`
stays at its pre-stamp head for PREREG_W9; nothing here touches PREREG_W9,
its configurations or any other stamped configuration.

## Governance

- The manuscript's numbers are generated from committed records only
  (`scripts/make_cmame_material.py`, checked by `tests/test_cmame_material.py`).
- Box sessions follow `RUNBOOK_CMAME.md`. A session's return is committed
  under `records/cmame/` before the manuscript reads it.
- Code changes leave every default training and evaluation path unchanged;
  new behaviour is opt-in, with tests.

## Stage 0 (3 Oct 2026) -- the manuscript draft

`5d089b7`: `paper/cmame/` (sections, generated tables, figures and numbers,
references), the generator, its tests, and the three two-dimensional reports
of July 2026 under `records/phase1/` with their hashes. Two independent
reviews addressed. Suite 453.

## Stage 1 (6 Oct 2026) -- diagnostics in the field export; standalone timing

- `scripts/export_fields.py` records, per model, seed, validation instance and
  load case: the squared Euclidean and stiffness norms of the error (their
  ratio, the error's Rayleigh quotient, is its error-weighted mean eigenvalue
  of K, a single number for its weight in stiff modes; normalised by the
  solution's own quotient it equals g over the squared relative displacement
  error); the relative von Mises error over the element values (the
  evaluation's) and weighted by element volume (the L^2 norm of Proposition
  1); the stress-energy integral of the error, equal to its squared stiffness
  norm by Proposition 1; and the largest absolute prediction on a constrained
  dof (0, since K is stored unconstrained). Per instance and load case of U*:
  its two norms and gamma*. `fields.json` summarises them per model (medians,
  the largest ratio to Proposition 1's bound, the two identities' deviations,
  and how often a supervised model's error has the larger Rayleigh quotient
  than the label-free model's on the same instance, load case and seed). The
  figure files carry the element volumes. The manuscript's two statements
  that these quantities were not measured (the errors' weight in stiff modes;
  the bound against measured errors) can then be replaced by measurements.
- `RUNBOOK_CMAME.md` Sec. A: the timing and field export of `RUNBOOK_W9.md`
  Sec. 4, run on their own before PREREG_W9's stamp (the 2D timing on E1's
  validation instances only, since F5 is made by wp9's session 1); outputs
  under `runs/cmame/timing/`.
- Tests: `tests/test_w9_fields.py` checks the new arrays against direct
  computations (Lemma 1 and Proposition 1's identity to 1e-8, gamma* from
  the stress tensors, the stiffness norm against the free block, the bound on
  every prediction, the Rayleigh quotients inside the stiffness matrix's
  spectrum, the element-count error against the evaluation's, the
  volume-weighted error against the figure's stresses) and the summaries
  against the arrays. Timing on a mesh of 17,425 nodes and 76,800 elements:
  1.5 s per instance for the nine models (the largest validation mesh has
  17,692 nodes), a few minutes over the validation split.
  `tests/test_cmame_runbook.py` checks that the runbook names only existing
  scripts and options, writes only under `runs/cmame/`, and states the
  suite's current count (the two LaTeX builds deselected on the box), and that
  its commands are RUNBOOK_W9 Sec. 4's apart from the F5 timing. Suite 457.
