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
