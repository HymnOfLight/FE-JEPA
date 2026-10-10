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

## Stage 3 (7 Oct 2026) -- a full check of Stages 1-2; RUNBOOK_CMAME Sec. B rehearsed end to end; PREREG_CM2D r2

- Checked: the code of Stages 1-2 read against PREREG_CM2D and RUNBOOK_CMAME;
  every number PREREG_CM2D quotes from E1's, July's and Phase-2b's records
  recomputed (all as quoted). No defect in `src`, which is unchanged.
- A functional check of the stiffness-norm loss (`scripts/cm2d_funccheck.py`,
  `records/cmame/cm2d_funccheck.json`: another corpus drawn by the 2D
  generator, a transformer of width 32 and depth 2 trained for 25 epochs on
  48 instances with each loss and the label-free objective, two seeds): L_K's
  validation energy gap and von Mises error far below L_D's, the L_K value
  the trainer computes equal to the mean square root of the per-load relative
  energy gap up to float32 round-off (relative deviation 7e-6 in the median,
  1.1e-4 at most: the stiffness spread of these meshes amplifies it, as the
  same formula in float32 shows), gradient clipping at 1 acting about as
  often for both losses. Its result bears on H2a and H2b (and on H1's and
  H3's comparisons), so PREREG_CM2D r2 discloses it, with the rehearsal's
  adjudication of its miniature (Sec. 1, 4, 7 and 9; r1 said that no
  supervised network had been trained with the stiffness norm and that the
  effects of L_K had never been measured); no question, arm, measurement or
  criterion changed, and the stamp script accepts only the r2 draft.
- Rehearsed (sandbox, CPU): Sec. B0-B4 and C, their blocks run verbatim on a
  miniature laid out like the box -- a local origin, an E1-like run on a tag
  `prereg-e1`, the generated configuration, the stamp and the tag
  `prereg-cm2d`; B0's and B2's checks gave GO; the run started on its GO and
  was killed with its workers as by a box restart; the re-pasted block
  started nothing; B3's checks gave GO and the restart took the finished
  units from the unit cache and resumed the interrupted one from its epoch
  checkpoint; B2' ran during it; B4's return was adjudicated by Sec. C with
  the reuse checks passed and the restart, the cached and resumed units, the
  two attempts and the two run logs recorded as deviations (on that
  miniature -- three supervised epochs beside label-free states of one --
  H1 NOT SUPPORTED, H2a and H2b SUPPORTED, H3 the stiffness-norm
  transformer lower: mechanics, not evidence). What a CPU
  sandbox cannot provide was stood in for in the sandbox copy only (an idle
  `nvidia-smi`, the CPU accepted on the precheck's torch line, E1's hash
  constants following the miniature).
- Fixed: one of the precheck's tests compared the precheck's reproduction of
  E1-like values with reference values computed on the CPU and required an
  exact match, while the precheck evaluates on the GPU whenever there is one:
  it passes on a machine without a GPU and would fail on the box. The
  precheck's tests now run it on the CPU; on the box the precheck itself
  still uses the GPU, as the run does. The zero-field count of the
  adjudication covers every row, the naive rows included (PREREG_CM2D
  Sec. 5: "per row and budget"). The test of the loss the trainer computes
  allows a relative deviation of 1e-4 (3e-5 observed), not 2e-3. And the
  suite could end without its summary line: `gmsh.initialize()` restores
  SIGPIPE's default action in the process that meshes, gmsh stays
  initialised in the test process after the first test that meshes in it,
  and a failed worker pool that still writes to its stopped workers' queue
  (tests/test_w9_parallel_kill.py, after tests/test_w9_ood2d.py) then ends
  the whole process (once in five full runs in the sandbox; reproduced in
  isolation). `tests/conftest.py` starts every test with SIGPIPE ignored
  again, as Python has it. The runs themselves are not affected: neither
  CM2D's run nor Sec. A's scripts mesh in their main process. (A run that
  generates a gmsh corpus in its main process and then loses a worker would
  end by SIGPIPE instead of its RuntimeError: for wp9, not changed here.)
- New tests: a restart beside the reused states, after an attempt that ended
  before its report and before one unit's result, evaluates the states
  again, takes the other units from the cache, ends with the first attempt's
  values everywhere and is adjudicated to the same verdicts with the restart
  recorded; a stiffness-norm training resumed from its epoch checkpoint ends
  bit for bit as the uninterrupted one; the functional check's record is the
  script's output on the Stage 2 code, PREREG_CM2D quotes its numbers, and
  the script runs; every test starts with SIGPIPE ignored, also after a test
  that meshed in the test process. An independent review of this stage
  addressed. Suite 507.

## Stage 4 (8 Oct 2026) -- the three roles of the energy, named in the manuscript

- Introduction: before the list of contributions, two sentences name the
  three roles of the energy (training objective, evaluation metric, failure
  detector) and point to the theory section, which derives each from the
  exactness lemma, and to the first three subsections of the
  three-dimensional results, which take them in turn.
- Discussion: a paragraph "The energy as an energy-based model" after "The
  energy in deployment". An energy-based model in machine learning is used,
  among other things, to find the output most compatible with an input, to
  rank two candidates and to detect whether one is compatible at all; its
  energy is learned and in arbitrary units, and the detection threshold is
  generally unknown when the model is built. The discrete potential energy
  is assembled and, with compatibility measured in the energy norm, answers
  the three exactly: its minimiser is the solution, and the ranking and the
  sign test of the preceding paragraph answer the other two, a compatible
  prediction being one no further from the solution than the zero field and
  the threshold, zero, needing no tuning. The picture of a training loss as
  an energy landscape is literal here as well; the loss is in general not
  convex in the parameters. The paragraph does not mention joint-embedding
  architectures. "The energy in deployment" now scopes its exact ranking to
  one load case, as the corollary "Zero-field test and ranking" states (it
  said one instance).
- References: LeCun, Chopra, Hadsell, Ranzato and Huang, "A tutorial on
  energy-based learning", cited in the authors' version (v1.0, 19 August
  2006, with the URL of the PDF read). The book (Predicting Structured Data,
  MIT Press, 2007) contains the chapter "Energy-Based Models" by LeCun and
  others; its author list and pages are to be confirmed before the entry is
  switched to it (comment in `references.bib`, whose header now says how an
  entry checked against another source is marked).
- Data and code availability (pending boxes): the repository is to be moved
  and renamed. The boxes ask for its address, its licence and the tag of the
  version described; note that the reviewers need the present address if
  the move comes after submission; and ask that the move be a transfer and
  rename on the hosting service, because git records the dates given to
  commits and annotated tags but not when they were pushed, which only the
  service records. A dated archive taken before the move, with an export of
  the service's record, is a further safeguard.
- No code, configuration, record or generated file changed. Suite 507.

## Stage 4b (8 Oct 2026) -- the return of RUNBOOK_CMAME Sec. A and B0, recorded

- `records/cmame/timing/`: Sec. A as it ran on the box on 8 October 2026 on
  commit `61f018f` (Stage 3): inference timing against exact solves (2D:
  E1's label-free state of seed 0 on 32 validation instances; 3D: Phase-2b's
  on 32 validation and the first 8 fine-set instances) and the field export
  of Phase-2b's nine states of the 1,024-instance budget, with the box's
  suite (505 passed, the two LaTeX builds deselected), the machine and the
  check before the timing. Fourteen files, copied byte for byte from the
  operator's tarball, whose SHA-256 matched on receipt;
  `records/cmame/README.md` lists their hashes and what each holds.
- `records/cmame/cm2d_ready.log`: B0 on the same commit (after Sec. A, by
  the operator's account): GO, with PREREG_CM2D unstamped, as it must be
  until the stamp.
- `tests/test_cmame_timing_records.py` (8 tests): the files and their
  hashes; every step exited 0 on the named commit and tree, with the GPU
  idle before the timing; the reports read are the committed records and the
  states the runs' own; every solver solved every instance, no CG fell back
  to the direct solve, the solutions agree with the labels, every matching
  CG met its target, and the timed model reproduces its report on every
  timed instance; the field export is complete and within the runbook's
  ranges; its counts agree with the manuscript's numbers; on every load case
  of `energies_val.npz`, the exactness lemma, the zero-field test and
  ranking, and the stress bound hold between quantities computed
  independently (the energies against the error's stiffness norm, the von
  Mises error against that norm), and the energy-optimal amplitude holds on
  the stored gaps; the per-instance means reproduce the report's energy
  gaps, displacement errors and von Mises errors; the figure rules,
  applied to the report's arrays, select the exported instances; B0 said GO
  on the same commit with PREREG_CM2D unstamped. Corruptions of a scratch
  copy of the records, with the README's hashes updated so that the hash
  test could not mask them, each made the test reading that record fail
  (an energy's sign, a timed row, the stamp line, the suite's summary line,
  and fourteen more in the review).
- `RUNBOOK_CMAME.md`: the suite count the box should now see is 513 passed,
  2 deselected (A0c, B1c); Sec. A says that it has returned, and where; B0's
  note on its commit now reads that the commit must contain the precheck
  script (Stage 1's, on which Sec. A was first planned, does not; both ran on
  Stage 3's).
- What the records show, for the next stage. The manuscript does not read
  these records yet; what it already states from other records (the
  label-free per-load count, the 2D warm-start finding) agrees with them.
  - Cost, medians over instances. Surrogate: first call on the GPU,
    preparation and copy back included. Solvers on the CPU, in a 16-CPU
    allotment with OMP and MKL set to 16 threads (the threads the direct
    solve and the mat-vec actually used are not recorded; to be settled
    before the manuscript states any): SuperLU on
    the free block, CG unpreconditioned to a relative residual of 1e-10, and
    the matching CG, stopped by the labels and so a lower bound. 2D
    validation (about 1,100 free dofs): surrogate 2.6 ms, direct 3.1 ms, CG
    24 ms, matching CG 7.2 ms. 3D validation (about 14,000): 49 ms, 0.91 s,
    1.2 s, 0.22 s. 3D fine set (about 112,000): 2.2 s, 105 s, 26 s, 4.3 s;
    there the surrogate's displacement error is 0.27 (median over the 8
    timed instances; 0.20 over all 256 of the fine set; seed 0) against
    0.031 on the 32 validation ones, and on 3 of the 32 fine load cases it
    is worse than the zero field, so that the matching CG took no step
    there. Started from the prediction, CG to 1e-10 took about as many
    iterations as from zero (per-instance ratios 0.98 to 1.04; on the fine
    set 1.6% to 3.5% more).
  - Per load case on the 3D validation split (3,072 per network): labels
    only 126 and graph network 106 worse than the zero field, label-free 1;
    none after the rescaling by c*.
  - The errors' Rayleigh quotient over the solution's, median: label-free
    12, labels only 132, graph network 1,026; a supervised error has the
    larger quotient on 89% (labels only) and 99.9% (graph network) of the
    instance, load case and seed triples. The volume-weighted von Mises
    error is close to the unweighted element-wise one (medians 0.047 and
    0.051 label-free, 0.25 and 0.26 labels only, 0.21 and 0.23 graph
    network), and Proposition 1's bound held on every prediction (largest
    ratio 0.85).
- The manuscript is unchanged. The next stage is to write the cost table,
  the field figures, the supervised networks' per-load counts and the
  spectral readings from these records, and to revise what they bear on: the
  discussion's "We have not measured the spectral content of the errors",
  the limitations' list of what is pending, and the introduction's "sparse
  direct and iterative solvers are fast at the sizes studied here" (in 3D,
  per instance, the direct solve took a median 20 times the surrogate's
  time on the validation set and 51 times on the fine set, and the matching
  CG, a lower bound, 3.9 times on the validation set).
- No source, script, configuration or generated file changed; one test file
  added. Suite 515.

## Stage 5 (8 Oct 2026) -- a last check of the code; PREREG_CM2D r3, stamped

- A last check before the stamp, by three independent reviews (the training
  and evaluation path the run executes; the adjudicator against the
  pre-registration and against the report the runner writes; the stamp, the
  tag, the precheck and the runbook, simulated on a single-branch clone),
  found no defect in the code. `src` is unchanged since Stage 2.
- PREREG_CM2D r3, approved by the principal investigator on 8 October 2026,
  four corrections of text; no question, arm, measurement or criterion
  changed:
  - Sec. 4 quoted the guard's operating characteristics "at July's L_D
    spread (8.9% in the relative energy gap; 7.1% in the von Mises error)"
    from a simulation at 9% only. It now quotes the 9% simulation for the
    energy gap (H1's and H2a's reference) and a new one at 7.1% for the von
    Mises error (H2b's): false support 1.3 / 3.4 / 8.8 / 11.8 / 16.5% and
    detection of a ratio of 0.8 in 97 / 89 / 57 / 45 / 38% at new-arm
    spreads of 5 / 9 / 20 / 30 / 50%; the summary "8-16% at new-arm spreads
    of 20-50%" holds for both.
  - Sec. 7 says the cost profile (52 ms per step) ran on the host the
    instance had then (Xeon Platinum 8470Q, 25 CPUs), and that on 8 October
    the instance ran on another (Xeon Gold 6459C, 16 CPUs; same GPU model,
    driver and torch) on which no training step has been timed; that the
    graph network takes the four load cases one after another through its
    checkpointed layers, where the transformer encodes them as one batch, so
    its step may take twice the transformer's or longer; the schedule is
    simulated at two and a half and three times (19.8 and 22.0 h); and the
    balance is raised from about 20 h to about 30 h.
  - Sec. 4 discloses the runs on the box before the stamp: the readiness
    check B0 (E1's states reproduced; the run's transformer trained with L_K
    for one epoch of two steps on two pool instances, relative energy gap
    0.995, nothing kept) and Sec. A's 2D timing of E1's label-free state of
    seed 0 on 32 validation instances (E1's values recomputed there, plus the
    rescaling by c* and conjugate gradients' iterations and times, none of
    which the hypotheses read).
  - Sec. 6 names the r3 draft as the one the stamp script accepts (it named
    r2).
- Stamped on 8 October 2026 with `scripts/stamp_prereg_cm2d.py` (which now
  accepts the r3 draft only): CONFIG_SHA256[cm2d_v1] =
  `bf1f1143b397331b95954229010bdce7bd9340d1b6e2abeb139337b117784a03`
  (configuration unchanged), footer
  `6eaf835fc9cab109d51160e50aec9da8e19ee4c2211c11977ac4b21d76846b7c`; the
  run's own guard (`verify_prereg`, label `cm2d_v1`) accepts it.
- `scripts/cm2d_sims.py`: the 7.1% cells, drawn after all the others so that
  every earlier value of `records/cmame/cm2d_sims.json` is unchanged, and
  graph-network step factors of 2.5 and 3; the record regenerated.
- `RUNBOOK_CMAME.md`: the box fetches directly (A0b, B0, B1b): on 8 October,
  by the operator's report, AutoDL's network route answered HTTP 503 and the
  direct fetch worked; the tag's refspec is forced, so that a stale local tag is
  replaced. The run's time is 11-22 h and the balance at least 30 h; the
  graph network's units at 1,024 labels may take 6-9 h. Before the operator
  instruction is sent, the repo side checks on the remote that only
  `prereg-cm2d` points at the stamped commit. B3: after a worker's
  death the run's own message suggests `--reuse-states`; the runbook's
  no-restart rule takes precedence. Sec. C fetches the tag by a forced
  refspec, in a clone of its own, checks that only `prereg-cm2d` points at
  the stamped commit, and checks out that commit before adjudicating. The
  box's suite count is now 514 passed, 2 deselected.
- Tests: the r3 quotes are pinned to their records (the 7.1% cells and the
  8-16% summary, the schedule's hours and the 30 h balance, B0's log and the
  precheck's smoke code, Sec. A's 2D timing, the two hosts); the stamp
  script's r3 markers. Suite 516.
- Review findings left as they are, for the record: the run records its
  `git describe` only when it writes the report (5 s timeout; a timeout would
  make the adjudication refuse; the operator runs no git command during the
  run, and `src` is not changed before the stamp for it); "no other tag on
  that commit" (PREREG Sec. 6) is checked on the remote before the run and
  at the adjudication (Sec. C), not by the precheck; the runbook fetches
  directly only, as the principal investigator decided on 8 October, with no
  fall-back to AutoDL's network route; a unit that hangs has
  no watchdog (the health check B2' shows it); each worker sets its torch
  threads from the host's CPU count (42 on this host's 128 CPUs, 16 of them
  allotted), as in E1; E8's built-in summaries use population standard
  deviations, the verdicts sample ones; the adjudicator flags a medians'
  disagreement by comparing the guard's outcome on medians with that on
  means, inside its `robustness` block.
- An incident in the sandbox, nothing pushed: one review made its scratch
  copy of this worktree with `cp -r`, which shares the worktree's git
  directory, and its simulated stamp commit and tag landed in the local
  repository. On the principal investigator's approval the branch was set
  back to `16973d3` and the tag deleted; the remote was never touched.
  Scratch copies are made with `git clone`.

## Stage 6 (9 Oct 2026) -- CM2D's return and verdict, recorded

- `records/cmame/cm2d/return/`: RUNBOOK_CMAME Sec. B as it ran on the box on
  the stamped commit `094c804` (tag `prereg-cm2d`): started
  2026-10-08T15:24:06Z, one attempt, exit 0, the supervised grid 12.8 h;
  nine files copied byte for byte from the operator's tarball, whose SHA-256
  matched on receipt (`records/cmame/README.md` lists them).
- `records/cmame/cm2d/verdict.json`: `scripts/adjudicate_cm2d.py` at the
  stamped commit, in a clone of its own with the tag fetched by a forced
  refspec (RUNBOOK_CMAME Sec. C), on 9 October; run again on the same inputs
  it writes the same file byte for byte. The label-free row passed its reuse
  checks (largest relative deviation from E1's values 0), and the adjudicator
  recorded no deviation. At 1,024 labels, seed means:
  - H1 SUPPORTED: relative energy gap 0.0383 (label-free) against 0.1334
    (labels only, L_D): rel -71.3%, threshold 12.4%.
  - H2a SUPPORTED: 0.0290 (stiffness norm, L_K) against 0.1334: rel -78.2%,
    threshold 16.4%.
  - H2b SUPPORTED: relative L2 von Mises error 0.106 (L_K) against 0.235
    (L_D): rel -54.9%, threshold 12.3%.
  - H3, no difference shown: 0.0290 (L_K) against 0.0383 (label-free): rel
    -24.2%, threshold 38.1%; L_K's seed values 0.0370, 0.0146 and 0.0355.
- Secondary readings (exploratory, PREREG_CM2D Sec. 5): the graph network
  has the lowest seed-mean displacement error at 1,024 labels (0.0499,
  against 0.0505 for L_D) and a relative energy gap of 0.825, above both
  L_D's and the label-free row's. The label-free row has the lower energy gap
  and the lower von Mises error than L_D on 767 of 768 instance-seed pairs,
  and so has L_K. L_K is worse than the label-free row beyond the guard at
  16 labels in every metric and at 64 in four of five (critical-region recall
  within), and within the guard at 256 and 1,024 in every metric. Values
  worse than the zero field: L_D 768, 532, 53 and 0 of 768 at 16, 64, 256 and
  1,024 labels; L_K 11, 1, 0 and 0; the graph network 766 and 187 at 64 and
  1,024; the label-free row none.
- Cost on this host (16 CPUs allotted, three workers), from the completion
  times in `run.log` (each unit's time includes its setup and evaluation):
  36-41 ms per transformer step (L_K about 5% slower than L_D overall) and
  73-78 ms per graph-network step, about twice; the grid took 12.8 h (PREREG_CM2D Sec. 7: 11.4 h at 52 ms with the graph network as fast
  as the transformer, 17.9 h at twice).
- Departures from the runbook's operations (`records/cmame/README.md`), none
  touching a file the adjudicator reads or a criterion: the direct fetch
  timed out and the operator fetched by one of the two routes of the
  supplement to his instruction of 8 October (which one is not recorded);
  after the run, B2's line that sets an earlier precheck log aside ran again
  at the box's terminal and renamed the precheck log (arrow keys pressed
  during the run may have recalled it from the shell history; how it came to
  run is not recorded).
- `tests/test_cm2d_records.py` (9 tests): the files and their hashes; one
  attempt on the stamped commit and tree; the report is the stamped
  configuration's on E1's corpus and seeds, with no restart and a solve
  ledger of 0; the verdict was made from these files; the verdicts, the 80
  comparisons, the label efficiency and the secondary counts follow from the
  report's per-instance arrays by the rule of PREREG_CM2D Sec. 4, recomputed
  in the test; and everything in the verdict that does not depend on git is
  what the adjudicator computes in process from the committed files. An
  independent review forged the hashes and confirmed that 21 corruptions of
  the return or the verdict each made a test fail.
- `RUNBOOK_CMAME.md`: Sec. B says that it has returned, and where; the box's
  suite count is now 523 passed, 2 deselected. The branch is no longer
  frozen.
- The manuscript is unchanged. The next stage is to bring this run into it:
  the two-dimensional comparison with the current code beside July's, and
  the statements the verdict bears on (the discussion's "we did not train
  one" and its reading of the stiffness norm, the limitation on the
  supervised baselines).
- No source, script, configuration or generated file changed. Suite 525.

## Stage 7 (9 Oct 2026) -- the error spectra of CM2D's models: script and runbook (post hoc)

The manuscript explains the label-free network's accurate stresses by the
norm of its objective (Corollary "Modewise contraction": the energy norm
weights the error in each eigenmode of K by its eigenvalue, the Euclidean
displacement norm weights all modes equally), and its discussion says that
the spectral content of the errors has not been measured. CM2D's H2b changed
only the norm of the supervised loss and the von Mises error halved; this
stage measures what the change did to the error itself. Specified after
CM2D's verdict: post hoc, reported only; nothing in PREREG_CM2D reads it.

- `scripts/cm2d_spectra.py` (new): for CM2D's kept states at 1,024 labels
  (L_D, L_K and the graph network, seeds 0-2) and E1's three label-free
  states, on the run's 256 validation instances, per load case:
  - the error's squared Euclidean and stiffness norms; normalised by the
    solution's, their ratio is the error's Rayleigh quotient over the
    solution's, as in Stage 1's 3D export;
  - the error's spectrum over the eigenmodes of the free stiffness block,
    diagonalised once per instance (at most about 3,600 free dofs), in both
    norms and in two binnings: by each mode's eigenvalue over the solution's
    own Rayleigh quotient RQ*, on fixed 1/8-decade edges from 10^-2 to
    10^6.5 with a bin below and one above (the axis on which the two losses
    differ: the Euclidean-weighted mean of lambda / RQ* is the normalised
    Rayleigh quotient; CM2D's meshes span about 10^-1.62 to 10^5.44 RQ*),
    and by mode rank in 40 bins of nearly equal count (for completeness:
    nearly all of a smooth field's Euclidean norm falls in the first rank
    bin, so the rank spectra cannot separate the rows); the solution's own
    spectra;
  - the von Mises errors over the element values (the evaluation's) and
    area-weighted; the energies and c*;
  - the rounding: every model also predicts every instance with TF32 off
    (`fejepa.runtime.setup_torch(tf32=False)`, the attention by the math
    backend), the run's policy restored after each instance; the error of
    that prediction (norms, log spectra) and the difference u - u_ieee
    (norms, stiffness log spectrum) are recorded beside the run's own
    predictions, so that the stiff end of a spectrum can be told from TF32's
    rounding, which is rough and falls there (the content check reads the
    run's own predictions only);
  - the checks: the spectra of each binning sum to the two norms; Lemma 1
    (||e||_K^2 = 2 (Pi_h(u) - Pi_h(U*))); the plane-stress form of
    Proposition "Energy gap and stress error" (the stress-energy identity
    with the three-dimensional bulk modulus, the bound
    vm^2 <= (1 + gamma*) g on every prediction, and gamma* itself through
    (1 + gamma*) ||s_vm(U*)||^2 / (3G) = ||U*||_K^2).
  Per row the summary gives the quotient (overall, per seed and per load
  case: the solution's own quotient varies between load cases by up to a
  decade relative to the smallest eigenvalue), the shares above 10^k RQ*,
  the IEEE predictions' readings and the rounding's share of the error's
  stiffness norm, overall and above 10^2, 10^3 and 10^4 RQ*. Per pair of
  rows it gives the ratios (first / second) of the means of the energy gap
  and of the von Mises error (the verdict's kind of comparison: for L_D
  against L_K they are H2a's and H2b's) and, since per load case the
  relative gap is the normalised Rayleigh quotient times the squared
  relative displacement error, the geometric-mean ratio of the gaps factored
  exactly into the ratios of the two (how much of a difference in energy is
  a difference in the error's spectral content and how much one in its
  size; the factorisation holds for the geometric means, not for the ratios
  of means), overall and per seed, with the shares of opposite rankings
  (Remark "Opposite rankings"). At a given seed the three transformer rows
  share their initial weights and instance order (PREREG_CM2D Sec. 2), so
  their pairing is a control; for the graph network's pairs only the ratios
  of means, the pooled geometric means and the share on the seed geometric
  means do not depend on the pairing. One figure instance by a rule fixed in
  the script: the median of L_D's relative energy gap, seed 0 (validation
  index 221, `instance_27356.npz`), with the mesh, the fields (also the IEEE
  predictions), the stresses, RQ*, gamma* and every mode's squared
  coefficient of the reference and of each row's seed-0 error. Before any
  model runs, the report's SHA-256 and every state's are checked against
  CM2D's committed provenance file (the label-free ones also against the
  report's d9_restart record), and a mismatch refuses; every model is also
  checked against the report's per-instance arrays (all five metrics
  recorded, the displacement error and the relative energy gap gated at a
  median relative deviation of 1e-3), and a mismatch ends with exit status
  5 after everything is written.
- `tests/test_cm2d_spectra.py` (10 tests): end to end on a CM2D-shaped CPU
  miniature on coarse gmsh plates (unequal element areas, holes; states,
  report, provenance file), with every identity on every array, the
  per-load arrays averaging to the report's per-instance values, the
  eigenvalues recomputed, every summary (whole tail curves included)
  recomputed from the arrays, the stored log spectra rebinned from the
  figure file's per-mode data, what the script prints, and strict JSON; the
  binning (the rank edges; an eigenmode's spectra one in its own bin of each
  binning; a mode exactly at 10^k RQ* counted at and above 10^k; the bins
  below and above); gamma*, the area-weighted von Mises error, the identity
  and the bound against independent computations on a structured mesh and
  on a gmsh mesh; a prediction that is not zero on a constrained dof;
  constructed smooth and rough errors and a rough rounding, which the
  summary orders the right way round (a wrong gamma* or a wrong spectrum
  moves the summary's checks of them); the refusals (a supervised state, the
  report, a label-free state against the d9_restart record and against the
  provenance file); exit status 5, with one outlying instance not failing a
  model and the figure instance beyond `--n-val`; the second pass's switch
  (TF32 off and the math attention backend inside, the run's policy and the
  default backends after, also when the pass fails); and, in process, with
  every prediction perturbed only inside that switch, the perturbation
  recorded as u - u_ieee (the figure's too), the content check reading the
  run's own predictions. Every one of 15 mutants of the second pass's code
  is killed by these tests.
- `RUNBOOK_CMAME.md` Sec. D (new): fetch (directly, as A0b ran on 8
  October, or from a git bundle of the branch the operator instruction
  sends, made with `git bundle create cmame-paper.bundle 094c804..cmame-paper`;
  a fetch that prints nothing for 10 min is abandoned for the bundle),
  checkout, CUDA and the suite, the export (`--device cuda`; about 10 min;
  nothing else on the GPU), what to expect, the return after any exit, and
  in the repo an end-to-end check against the verdict (H2a's and H2b's
  ratios of means). The header no longer calls the sections independent
  (Sec. D reads Sec. B's states and Sec. C's records); Sec. C says when it
  ran. The box's suite count, in A0c, B1c and D0c, is now 535 passed, 2
  deselected: so the suite reads under the box's Python 3.12.3, numpy 2.4.6,
  scipy 1.18.0 and torch 2.12.1+cu130, in a fresh clone (CPU); there, a
  rehearsal of the export at CM2D's model sizes on gmsh plates ended with
  exit 0, every model reproducing its arrays exactly and the TF32 policy
  restored. `tests/test_cmame_runbook.py` (two new tests): Sec. D names the
  script with options it accepts, writes only under `runs/cmame/spectra/`
  and its own return (every `tee`, redirection, `mkdir` and `OUT`), uses
  Sec. A's `run` helper, A0b's fetch and B1c's process check, pins its
  bundle route, checkout, rotations and the return line by line, exports
  from CM2D's committed return, and names the figure instance that the
  script's rule selects from the committed report and the summary line's
  keys; the suite count and both `--deselect` options are checked in each
  of the three sections.
- Three independent reviews, each reviewer in a clone of its own:
  - Code and science (three rounds, the third re-running every mutant on
    the candidate): no blocking defect; the identities, the log binning,
    the pair statistics and the gamma* check verified on the real
    validation meshes, rebuilt from the generator; the binning by mode rank
    alone could not have separated the rows and was supplemented, before
    any box run, by the binning by lambda / RQ*, whose edges were then
    widened to 10^-2 (seven instances put one mode of the axial load case
    just below 10^-1.5 RQ*). Of the reviewer's mutants of the script, all
    are killed by the tests above except two that are equivalent in
    practice (`>=` for `>` in a share; a padded bin count).
  - Operations (three rounds): no blocking defect; the runbook's blocks
    were run as pasted in a simulated box (both fetch routes, exit 0, exit
    5, a refusal, a crash, second pastes); the bundle block made pasteable,
    the return named once per paste, a CUDA check added; all of the
    reviewer's 31 mutants of Sec. D are killed by the tests above.
  - A fresh review before the push, of the run on the box itself: no
    blocking defect; one important point, answered by the second pass with
    TF32 off: TF32's rounding is rough and the stiffness norm weights it
    heavily, so on the most accurate rows it could account for part of the
    stiff end of a spectrum, and the states exist only on the box. Also from
    it: a guard against a non-positive eigenvalue, the pairing-free share,
    the per-load quotients, the runbook's rules for a fetch that hangs and
    for going on to the return after any exit, and a test assertion on
    element areas made robust to the gmsh version.
- Stage 6's record completed (`records/cmame/README.md`): the operator
  fetched the branch and the tag from the git bundle, one of the two routes
  of the supplement to his instruction of 8 October (his report, relayed on
  9 October).
- No source, configuration, pre-registration or manuscript change. Suite 537.

## Stage 8 (9 Oct 2026) -- the error spectra of CM2D's models, returned and recorded

- `records/cmame/spectra/`: RUNBOOK_CMAME Sec. D as it ran on the box on
  Stage 7's commit `8b5d443` on 9 October (16:23 UTC at the return): the
  suite 535 passed, 2 deselected; the export exited 0, its timed part 70 s
  (the eigendecompositions 31 s, the second pass with TF32 off 14 s); seven
  files copied byte for byte from the operator's tarball, whose SHA-256
  matched on receipt (`records/cmame/README.md` lists them). Every state
  matched CM2D's provenance file; every transformer reproduced the report's
  per-instance arrays exactly, the graph network within 1.4e-4 in median on
  the displacement error and the energy gap, the two the export gates
  (5.2e-4 on the peak von Mises error; its CUDA scatter reductions are not
  bitwise reproducible); the TF32 policy was in force after the second pass
  as before it.
- End to end (RUNBOOK_CMAME D3): the export's seed means of the relative
  energy gap and of the von Mises error are the verdict's, seed by seed, for
  H1, H2a, H2b and H3 (largest relative deviation 2.2e-13); L_D's over L_K's
  ratios of means are 4.596 (energy gap) and 2.215 (von Mises error), H2a's
  and H2b's.
- Readings (post hoc, reported only; medians over a row's 3,072
  seed-instance-load-case triples, pair readings on the same seed, instance
  and load case; the solution's own Rayleigh quotient is a median 1.27 times
  the smallest eigenvalue):
  - the error's normalised Rayleigh quotient (its Rayleigh quotient over the
    solution's): label-free 10.3, L_K 11.3, L_D 115, graph network 553;
  - L_D against L_K: L_D's quotient the larger on 94.9% of the triples
    (98.8% on the seed geometric means, which do not depend on the pairing),
    geometric-mean ratio 7.80 (per seed 8.26, 9.46 and 6.07); the squared
    relative displacement error's geometric-mean ratio 1.26 (per seed 0.96,
    1.65 and 1.26; L_D's the smaller on 50.0% of the triples); so the
    relative energy gap's ratio is 9.85 (= 7.800 x 1.262). The change of
    norm changed the error's spectral content, and its Euclidean size much
    less: on the log scale about 90% of the gap ratio is spectral (per seed
    102%, 82% and 88%);
  - L_K against the label-free row: quotient ratio 1.04 (L_K's the larger on
    49.1%), squared displacement ratio 0.64 (per seed 0.81, 0.39 and 0.81):
    in the geometric mean the same spectral content and a smaller error;
    H3, on the seed means of the gap, showed no difference (L_K's seeds
    vary widely);
  - the graph network against L_D: quotient ratio 4.33, squared displacement
    ratio 1.00;
  - the share of the error's squared stiffness norm in the modes at or above
    100 RQ*: label-free 69%, L_K 73%, L_D 95%, graph network 99% (the
    solution's own 1.5%); of its squared Euclidean norm at or above 10 RQ*:
    7.2%, 6.6%, 24% and 49% (the solution's 0.2%);
  - the bound of Proposition "Energy gap and stress error" holds on every
    load case (largest ratio 0.90);
  - TF32's rounding (u - u_ieee): a median 2.7% (label-free), 2.9% (L_K),
    0.7% (L_D) and 0.05% (graph network) of the error's squared stiffness
    norm, and, in the modes at or above 10^4 RQ*, 41% (label-free) and 36%
    (L_K) of the error's squared stiffness norm there (on the 2,010 of 3,072
    load cases whose modes reach that far); with TF32 off the quotients are
    9.1, 9.4, 114 and 558, and L_D's over L_K's geometric-mean quotient
    ratio 9.30 and gap ratio 11.9. The rounding makes the two smooth rows
    look rougher, so the readings as run understate the difference.
- `tests/test_cm2d_spectra_records.py` (11 tests): the files and their
  hashes; one export on Stage 7's commit and tree, exit 0 after a green
  suite, the TF32 policy kept, the machine, texts and timings quoted, and
  the log's summary line that of the JSON; the inputs (CM2D's committed
  report and provenance file, the stamped configuration, the twelve states,
  the run's validation split); every model's agreement with the report, also
  recomputed from the per-load arrays (the graph network's recorded
  deviations too); Lemma "Exactness", the zero-field test and ranking, the
  energy-optimal amplitude (the gap of c* u from the stored energies), the
  plane-stress stress identity and bound (gamma* from the stored integrals),
  the spectra's sums and the Rayleigh quotients' range on every array; bin
  by bin, every spectrum's stiffness-weighted over Euclidean mass within the
  bin's eigenvalue range and the outer bins empty, and the rounding of
  TF32's size on every prediction (so the second pass was not the first
  again); the summaries, counts and eigenvalue ranges recomputed with the
  script's functions, and the readings the manuscript may quote written out
  again from their definitions (overall, per seed and with TF32 off); the
  verdict's seed means for every hypothesis and H2a's and H2b's ratios; and
  the figure file, the rule's instance, with its stiffness matrix
  reassembled from its mesh by the generator's assembly (the eigenvalues, U*
  solving the stored loads, every row's seed-0 coefficients, norms,
  energies, gaps, c*, spectra and stresses).
- An independent review, in a clone of its own: no blocking defect; one
  statement corrected (the graph network's median deviation, which is within
  1.4e-4 on the gated metrics only), two readings worded more cautiously (L_D
  against L_K's displacement errors; L_K against the label-free row, beside
  H3), and the checks above added where its mutation run found gaps. Of its
  113 corruptions of the records (each with the README's hash forged), 105
  now fail a test; the other 8 touch nothing the records claim or quote (a
  shift of the rounding's mass between two low bins, the largest eigenvalue
  or the dof count of an instance that is not an extreme, two graph-network
  deviations that the records cannot recompute, one model's inference time,
  the suite's warning count and its time).
- `RUNBOOK_CMAME.md`: Sec. D says that it has returned, and where; the box's
  suite count, in A0c, B1c and D0c, is now 546 passed, 2 deselected.
- The manuscript is unchanged: its discussion still says that the spectral
  content of the errors has not been measured. The next stage brings CM2D
  and these readings into it.
- No source, script, configuration, pre-registration or manuscript change.
  Suite 548.

## Stage 9 (9 Oct 2026) -- CM2D, the error spectra and the field export in the manuscript

The manuscript reads the records of Stages 4b, 6 and 8. Every new number is
generated from them by `scripts/make_cmame_material.py`, which checks each
statement the text makes about them and stops otherwise; the cost table
waits for the thread measurement of RUNBOOK_W9 Sec. 4.

- Section 5 (two dimensions) now reports CM2D, the run of 9 October 2026, as
  the comparison. Its opening says what was retrained (L_D at every budget,
  the graph network at 64 and 1,024) and what was reused (E1's label-free
  states). Table 1 (`table_2d.tex`) is CM2D at 1,024 labels (label-free,
  supervised L_D, supervised in the stiffness norm L_K, graph network, the
  naive rows). 5.1: accuracy, H1, the guard defined in the text ("no
  difference shown" is not equivalence), the readings beside the verdicts
  (Table A.2), the graph network against L_D (lowest displacement error, 1%
  below L_D and within the guard; lower gap or von Mises error on no pair).
  5.2: H2a and H2b (the area-weighted ratio of von Mises errors, 2.26 against
  2.22 element-wise); H3 on its own (within the guard; seed 1 makes most of
  the difference, seeds 0 and 2 alone 2.7%; L_K lower on 87.6% of the pairs;
  instance resampling 21.9% to 28.0% lower, the seed-level guard does not
  separate). 5.3: the spectra (post hoc; Figure 2 `fig_spectra.pdf`; Figure 3
  `fig_field2d.pdf`, the rule's instance, 221, seed 0); the 90% is the
  log-share of the geometric-mean ratio 9.85, not of H2a's 4.60; TF32's
  rounding (2,010 triples = 670 load cases x 3 seeds) understates the
  differences between the networks trained in the two norms. 5.4: label
  efficiency; the budget comparisons named as 80 exploratory secondary readings
  without multiplicity correction; at 256 labels L_K's seed means are within
  the guard of the label-free ones although the label-free network has the
  lower gap on 72.4% of the pairs. 5.5: the run of 16 July 2026 (its table
  `table_2djuly.tex`, the former Table 1; decisions; observations).
- Theory: Remark "Size and spectral position of an error" (g = rho delta^2,
  eq:factor); the conjugate-gradient factor renamed q (rho is the quotient);
  Sec. 3.3 gains the area- against element-weighted von Mises medians in 2D
  and the volume-weighted ones in 3D (three significant digits); the counts
  of single load cases and the exported networks named in Sec. 3.4 and 3.7.
- Methods: L_D and L_K as numbered equations, L_K's dependence on each
  solution through ||U*_j||_K^2 only and its gradient (eq:lkgrad); TF32
  defined; the code versions by run; CM2D's reuse of E1's states, shared
  initial weights and the same sequence of per-epoch permutations; the
  field-plot rules, with the 2D rule disclosed as written after CM2D's verdict.
- Section 6: Figures 6 and 7 (`fig_field_worst.pdf`, `fig_field_median.pdf`)
  and Section 7's Figure 9 (`fig_field_fine.pdf`): the three visible faces of
  the box in a right-handed oblique view (front face z = max; the candidate's
  view was a mirror image); on the worst instance's load case the label-free
  displacement has the reference's shape and too large an amplitude (delta
  0.22, 0.045 after c* = 0.78); the 3D quotients and per-load counts; the
  stale F_max attribution of the 2D/3D displacement difference replaced.
- Abstract (250 words, no digits), introduction and conclusion: the 2D
  label-free against supervised result restored in the abstract; "with the
  labels fixed, the norm of the loss decided the stress accuracy"; the
  spectral reading flagged as post hoc; the 3D failure sentences scoped.
  Discussion: H3 as "no difference shown, which is not equivalence", the
  spectral share stated as a log-share of the geometric-mean ratio,
  Limitations (what was known at registration; exploratory budget readings).
- Appendix A: CM2D's criteria rows with network names; the guard note without
  tau (tau is the CG tolerance); Table A.2 (`table_cm2d.tex`): per-seed
  values, rel, SE_rel, guard, the guard on medians, Welch's and the
  instance-resampling 95% intervals for H1-H3 and H3 with the roles exchanged;
  "Three things were known" (the functional check also showed H1's direction
  and H3's split between its two seeds); the operational departures named; the
  run dated 8-9 October. The provenance sentences of Sec. 4.5 and the body of
  Appendix A's July paragraph, whose correction awaits the principal
  investigator, are unchanged apart from the paragraph title. Appendix B:
  Proposition B.1's bound renamed beta (delta is the displacement error).
- Generator: CM2D (report and verdict), the spectra and the field export are
  read and hashed into `sources.json`, with `scripts/adjudicate_cm2d.py`
  (checked against the hash the verdict records; the reuse bound 10^-4 is read
  from it). Every spectral reading the text quotes is recomputed from the
  per-load arrays and checked against the export's summary (TF32 off, the
  stiff-mode rounding and its count, per seed, seed geometric means, the
  2,010/670 count, the log bins). New checks: opposite rankings in either
  order; one count for label-free and L_D per load case; the gap and von
  Mises shares on the same pairs; the label-free gap below the graph
  network's at 64; L_K's errors the smaller in every seed, by far in seed 1;
  the medians and intervals beside the verdicts; the fig5 and fig6 load-case
  statements; Pi_h(u) > 0 against the error norms (not against g > 1, which is
  computed from the same energies). The field plots' load case "max" is
  np.argmax; each panel's sign is the stored energy's. Legends use the tables'
  names. Five new figures; colours follow the entity (L_K green, marker X; the
  five series validate all-pairs in light mode); single-hue ramps for fields.
- Tests (`tests/test_cmame_material.py`, 13 new): CM2D's numbers against the
  verdict and the report's arrays; every quoted spectral and CM2D reading
  recomputed independently; the criteria table against the verdict and the
  adjudicator's bound; every cell of Table A.2 recomputed from the report alone
  (seeds, rel, SE_rel, guard, reading, the guard on medians, Welch's interval,
  an independent resampling of the instances, the exchanged roles); the
  spectral and 3D readings from the per-load arrays; each field plot's
  instance and load case by its rule; the figure files tied to the energies
  and to Phase-2b's P3 arrays; the faces of the oblique view (right-handed,
  owner tetrahedra, exact tiling); each panel's annotation and colour scale;
  Appendix A's description of the functional check against its record.
  RUNBOOK_CMAME's suite count (A0c, B1c, D0c): 562 passed, 2 deselected.
- Review: three independent reviews of the candidate (science, code,
  editorial). No number was wrong. Two blocking findings, both fixed: "could
  not be told apart" (now: seed means within the noise guard, which is not
  equivalence) and Figure 4's caption (the 2D panels are CM2D's; July trained
  the energy-term transformer in 2D and the graph network at every budget).
  The important and minor findings were applied as listed above; the
  reviewers' proposed tests were adapted and added. A second, fresh review of
  the fixed candidate (with the mesh fix) found nothing blocking; its findings
  were applied: Table A.2 pinned cell by cell, Appendix C's description of the
  tests scoped ("the main numbers"), "the later code" in Sec. 6.1, the mesh
  wording; the proof of Proposition 1's display is numbered. Correction to
  Stage 8's note: seed 2's spectral log-share is 88.5% (written there as 88%).
- The structured tetrahedral mesh (`src/fejepa/fe/tet3d.py`; the patch
  `tet3d_kuhn_fix.patch` from another session, with two corrections from the
  second review: the code comment, and the coverage test asserting exactly
  once): `structured_tet_mesh` split each hexahedral cell into five tetrahedra
  that all share one corner, a table that tiles a cell under no corner
  ordering; with the ordering used, they covered 5/6 of the cell (no overlaps,
  one void per cell), and neighbouring cells did not conform. It now uses the
  six-tetrahedron Kuhn split along each cell's c0-c6 diagonal, the same in
  every cell, which is conforming. `tests/test_tet3d_mesh.py` (3 tests): exact
  and conforming tiling, every point covered exactly once, the P1 energy of a
  linear field exact. No record and no run reported in the manuscript used
  this mesh: every configuration of the reported runs and of wp9 uses gmsh
  (2D) or gmsh3d (3D); only the WP7-S1 smoke configurations,
  `scripts/bench3d_scale.py`, `scripts/amp_compile_check.py` and tests did.
  The tests' three-dimensional instances, on which the manuscript's theory
  checks also run, are now proper meshes of their boxes; every check passes,
  as it did on the defective meshes (the identities and bounds hold for any
  assembled system). No experiment is re-run. Because `src` changed, the
  cross-branch regression against wp8-lejepa ran again in the sandbox (both
  miniatures identical: 162 and 2,590 numbers, every state file
  byte-identical; 43 s and 100 s), and `records/cmame/`'s two summaries and its
  README name the new `src` tree, `7cc8ac1b...`, as `tests/test_w9_records.py`
  requires. wp9-pool, which noted the split as an observation in its own Stage
  notes, is frozen until session 2 returns; the fix goes there afterwards, with
  its regression record and its runbook's suite count.
- One source file changed (the mesh fix above), and the two regression
  summaries with it; no configuration, pre-registration or run record. Suite
  564.

## Stage 10a (10 Oct 2026) -- the provenance of the July 2026 documents, corrected

The manuscript said that the pre-registration documents of the two-dimensional
runs of July 2026 were committed on 3 August 2026, after the runs. For the run
of 16 July 2026 that was inaccurate, and to the paper's disadvantage; and it
did not mention that the repository's history had been replaced. Checked on
10 October 2026 against the hosting service and the commits themselves:

- The repository was created on 14 July 2026. Its first commit, `b365af5`
  ("Add files via upload", 03:25 UTC), was made through GitHub's web
  interface, which created it (committer GitHub) and signed it (verified; the
  packet names GitHub's web-flow key B5690EEEBB952194 and 03:25:10 UTC). It
  held the code of v2.1.4, which differs from v2.1.5 only in the four files
  that FIX_NOTES_v2_1_5.md lists; `PREREG.md` as shipped with that release, a
  template whose hash line was blank and which otherwise equals the stamped
  document that the run of 16 July checked (SHA-256 `b9470e68...`, as
  PROVENANCE_NOTE.md records); and `configs/phase1_rec8_v2.json` byte for byte
  as now (canonical SHA-256 `62b26ad8...`, the stamped hash, recomputed). It
  did not hold `PREREG_WP2.md`. The service still serves `b365af5` by its hash.
- The activity record logs nothing on the repository between that commit and
  a force push on 2 August 2026 at 17:29 UTC, which replaced the history (that
  commit alone) with the one that begins at the root commit `a548825`
  ("v2.1.5", dated 2026-08-03 01:26 +0800, 17:26 UTC on 2 August). That commit
  holds both stamped documents and PROVENANCE_NOTE.md as compiled on 1 August,
  which records their hashes. The record logs no tag creations and does not
  show visibility.
- The owner account's security log (exported by the account holder on
  10 October 2026; 643 entries from 13 April) records that the present
  repository (id 1299960094) was created, public, on 14 July 2026 at
  03:23:30 UTC, and that a public repository of the same name (id
  1268189576), created on 13 June 2026, had been deleted 13 seconds before
  (03:23:17 UTC). No entry of the export has the action `repo.access` (a
  change of visibility).
- FIX_NOTES_v2_1_5.md records a first attempt of the run of 16 July, on
  14 July, which stopped before any training (the corpus it had generated
  carried no labels); PROVENANCE_NOTE.md records the offline labelling of
  15 July. The manuscript did not mention the attempt.

Changes:
- Methods (pre-registration): the run of 16 July (document and configuration
  uploaded on 14 July in a commit that the service created and signed; the
  document there the template, differing from the checked copy only in the
  hash line, which held the hash of the configuration uploaded then; a first
  attempt stopped before any training) is separated from the run of 31 July
  (document committed on 2 August, after the run, its content resting on the
  archives and the provenance note); the replaced history is stated, with a
  pointer to Appendix A.
- Discussion (limitations): the same correction in one sentence.
- Appendix A, "Provenance of the two-dimensional runs of July 2026": the facts
  above, times in UTC (the committer's time zone, UTC+8, named for the date of
  3 August), the first attempt and the offline labelling, the tag
  `provenance-2026-07-14` (created in October 2026) that keeps `b365af5`, the
  exported activity record, and the security log's record that the
  repository was public from its creation. The deleted namesake of 13 June,
  which held nothing that the paper relies on (the author), is stated in the
  records and in the provenance note's addendum, not in the text, at the
  author's decision.
- Methods: the repository named public since its creation on 14 July.
- Appendix C: the activity record, exported on 10 October 2026, is committed
  in `records/provenance/` with a README on how it was obtained
  (`gh api ... /activity --paginate`; pages joined and sorted, events as
  returned; 103 events) and on the two commits; beside it, the security
  log's 15 entries that name FE-JEPA, with user agents, request identifiers
  and request headers removed (no other entry of the export committed).
- PROVENANCE_NOTE.md: a dated addendum (10 October 2026) on the first history,
  the force push and the tag; its header names the addendum.
- `tests/test_provenance_records.py` (5 tests): the export is the committed one;
  its counts, its two events before 6 August and the absence of logged tag
  creations, and the times, commits and tag quoted by Appendix A and the
  addendum; `a548825` a root commit with that date, holding the two stamped
  documents and the note as compiled on 1 August, which records their hashes,
  unchanged since (skipped without the history); `b365af5`, where present (the
  tag fetched), a root commit made and signed by the service at that time,
  whose `PREREG.md` differs from the stamped one only in the blank hash line,
  whose configuration is the registered one by blob and canonical hash, without
  `PREREG_WP2.md`, and whose code differs from v2.1.5 only in the four files,
  with the tag on it if the tag is present (without the commit, the README's
  statements are checked); the security log's entries: the committed ones,
  the three creation and deletion times, public throughout, no visibility
  change, nothing private kept, and the text and the note agreeing.
- Review: an independent review verified every statement against the API and
  the commit objects; its findings were applied: the stale limitation
  sentence; "the hash entered later" (the box's copy was stamped by
  14 July, as the fix notes record; the uploaded document was the release's
  template); "pushed" for a web upload, with the service's signature; the
  public status on 14 July stated only from the owner account's security
  log; the tag's creation date; the first
  attempt disclosed; tests for the first commit and for the note as
  committed; "today" removed; the replaced history named as one commit.
  RUNBOOK_CMAME's suite count (A0c, B1c, D0c): 567 passed, 2 deselected.
- The tag `provenance-2026-07-14` on `b365af5` is created and pushed by the
  author from a terminal (an annotated tag on a commit that no branch holds),
  before this commit, which names it.
- No source file, configuration, pre-registration or run record changed.
  Suite 569.
