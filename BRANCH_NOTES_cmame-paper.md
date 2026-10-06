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

## Stage 2 (6 Oct 2026) -- CM2D: the 2D supervised networks retrained, with a stiffness-norm control

The manuscript's 2D comparison reads the run of 16 July 2026 (the July code);
the label-free transformer was trained again with the current code (E1), the
supervised networks were not. And its explanation of the label-free
network's lower stress errors -- the norm of the objective -- was never
tested with labels. CM2D (PREREG_CM2D.md, r1 draft, not stamped) retrains the
supervised rows on E1's corpus, split and seeds with the current code, beside
E1's states (evaluated, not trained), and adds the same transformer trained
on the relative stiffness-norm error: H1 (label-free against labels-only,
energy gap), H2a and H2b (stiffness norm against labels-only: energy gap, and
the von Mises error that neither loss contains), H3 (a two-sided reading,
stiffness norm against label-free).

- Code (opt-in; every default path unchanged):
  `EnergyAnchor.quad` (v^T K v with the constrained dofs masked, analytic
  gradient); `SupervisedConfig.loss = "knorm"`, the mean over load cases of
  ||u - U*||_K / ||U*||_K (the displacement loss with the norm replaced;
  recorded in the protocol only when not the default); E8's
  `include_anchor` (default true) and `include_knorm` (default false) rows,
  the stiffness-norm row's largest-budget states kept; `e8.reuse_from` with
  `supervised_grid: true` evaluates the reused label-free states beside a
  trained supervised grid (refused with a fine-tuning row; the configuration
  may differ from the source run only in the keys that shape the supervised
  grid); the step count exact for configurations that name `include_knorm`;
  the results page and figure show the new row.
- `records/cmame/`: the default path against `wp8-lejepa` again on both
  miniatures, identical (the final Stage 2 `src`); `tests/test_w9_records.py`
  now requires the committed `src` to be the tree named by the newest
  regression record. `cm2d_sims.json`: the decision rule's operating
  characteristics (equal and unequal seed spreads, July's spread, per-instance
  blow-ups) and the run's wall time by a simulation of its unit schedule.
- `scripts/make_cm2d_config.py` -> `configs/cm2d_v1.json` (from E1's stamped
  base); PREREG_CM2D.md; `scripts/stamp_prereg_cm2d.py`;
  `scripts/cm2d_precheck.py` (GO/STOP before the run: the GPU idle, the
  checkout on the tag with the describe string the report records, the
  configuration, the dry run, E1's corpus and labels, E1's states
  reproducing E1's validation values when evaluated as the run evaluates
  them, a short stiffness-norm training on the GPU, no earlier attempt;
  `--restart` before a restart, refused once a report exists; `--pre-stamp`
  for a readiness check before the stamp); `scripts/adjudicate_cm2d.py`;
  RUNBOOK_CMAME Sec. B (the box: readiness, checks, the run started only on
  GO, health check, restart, return) and C (the adjudication).
- Three independent reviews (code, pre-registration, operations) before the
  draft went to the PI. Addressed: H2 split into H2a (energy gap; close to a
  check that the loss took effect) and H2b (von Mises error; the test of the
  manuscript's explanation); L_K uses each reference solution only through
  one number per load case and its gradient is the label-free one weighted
  per load case, so H3 compares two weightings of the same residual; the
  label-free and supervised transformers of a seed share their initial
  weights and instance order (tested); the operating characteristics extended
  to July's spread, a noisy new arm and per-instance blow-ups; the
  reproduction of E1's values moved before the run; a finished report can no
  longer be overwritten by a restart, and the run starts only on the
  precheck's GO; the adjudicator checks E1's report as a stamped run, the
  recorded commit and tree against the tag, every seed value's presence,
  attempts by start and exit lines and by run logs, and writes strict JSON;
  secondary readings never block a verdict; mutation checks of the new tests.
  Second round: the background states what July's labels+energy row did
  and did not test; the constrained dofs and the relation of the von Mises
  error to Proposition 1 stated; the graph-network reading corrected
  against the earlier runs; H3's reading with the roles exchanged reported
  beside it; the precheck leaves the GPU alone when it is busy, turns a
  failed reproduction into a FAIL line, sees any script of the repository
  as another run, checks that the run imports this checkout's code, and
  treats any trace of an earlier attempt as one; the run and the restart
  use up their GO, so a re-pasted block starts nothing; B0 fetches its own
  commit; the restart rules no longer contradict the health check.
- Tests: `tests/test_cm2d_training.py` (the loss and its gradient against
  direct computations, what the trainer minimises, the reuse run on a
  miniature with its refusals and exact step count, the units' payloads, the
  supervised-grid keys against the label-free units, every other key
  compared, shared initial weights and order), `tests/test_cm2d_prereg.py`
  (configuration, stamp, simulation record, the quoted numbers),
  `tests/test_cm2d_adjudicate.py` (verdicts, refusals, reuse checks,
  deviations, the command line on a made-up return around E1's report, the
  pre-run checks on a miniature laid out like the box, the git and stack
  checks), `tests/test_cmame_runbook.py` per section. Suite 501.
