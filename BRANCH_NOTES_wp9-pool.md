# BRANCH NOTES -- wp9-pool

Branched 1 Oct 2026 from `wp8-lejepa` at `5f8e2df` (Stage 1.38; wp8 is
read-only from that head on; main line A). Plan: the wp9 plan v0.1 of 1 Oct
2026 (outside the repository), approved by the PI as recommended.

Purpose: two questions, both in 2D, both label-free at training time.

- **Q1 (C1, unlabelled pool size).** At a fixed budget of 204,800 AR steps per
  seed, does training on more unlabelled instances -- 1,024 (200 epochs),
  4,096 (50), 25,600 (8) -- lower the energy gap in band, in the tail and on
  held-out families? Every deciding run so far trained on 1,024 instances.
  The pools are nested prefixes of E1's pool (E1's corpus, split and
  configuration otherwise). The 1,024 arm reuses E1's base states (their
  SHA-256 are in E1's report), provided the default training path is shown
  bitwise unchanged against `wp8-lejepa` (Stage 0b).
- **Q2 (S, mesh-independent scale).** Do a decode scale and a load input that
  do not change with the mesh (1/64 of the battery's L1 load instead of its
  largest nodal force; per-node load densities; load summaries free of the
  mesh) reduce the growth of the error on a finer mesh than training saw --
  the fine-mesh amplitude deficit of wp8's post-hoc reading 4e? S runs only if
  session 1's rule 1 admits it, against a baseline trained afresh on fresh
  seeds (since r3 the baseline runs in every session 2).

C2 (adaptive allocation of the training budget) and 3D are not in this round.

## Governance (carried from wp8)

- Nothing a stamped run depends on changes: the stamped configurations and
  PREREG files are untouched, and every code change leaves the default
  training and evaluation path bitwise unchanged -- new switches default to
  the old behaviour, with tests, and `scripts/regress_against_branch.py`
  checks the whole path against `wp8-lejepa` (Stage 0b, item 4).
- wp8's records and tests stay in the suite.
- The OOD-2D families are evaluation-only: no hyper-parameter, switch, rule or
  selection reads them (rule 1 below reads the base states' error on F5, a
  reading of the existing model, as pre-declared). Their manifests pin every
  file by SHA-256.
- Session 1 trains nothing and adjudicates nothing. Its readings that feed a
  decision are read by rules committed, as code, before it runs (next section).
  C1 and S run only after a stamped and tagged pre-registration (PREREG_W9.md,
  Stage 0b).
- The repository and the papers are English-only (`tests/test_english_only.py`).
- No claim of being the first.

## Pre-declared readings (session 1)

Committed before session 1 as `scripts/w9_session1_decisions.py` (tested in
`tests/test_w9_analysis.py`), applied to the session's outputs as they come,
and not tuned after they are seen.

1. **S enters** the pre-registration iff, for E1's base states, the F5 /
   in-band displacement-error ratio (seed means) is >= 1.5 (KP4's own line)
   AND the median c* on F5 (seed mean of per-seed medians) is >= 1.10.
   Otherwise the 2D baseline shows no deficit for S to remove; S waits for 3D.
2. **C1's largest pool** keeps the design before the schedule: 25,600 with
   three workers if they fit; else 25,600 with one worker at a time (seeds in
   sequence); else 12,800 (16 epochs) with three workers, else with one; else
   neither, and the pool design is revisited. k workers fit at pool size n iff
   k x one worker's host need <= 80% of the usable host memory and k x one
   worker's device need <= 80% of the device memory. Host need = the audit's
   residency at n (what one training unit holds -- instances, CPU-resident
   anchors, the process's baseline; the packs sit on the device) + the growth
   of the process during the AR step measurement. Device need = CUDA context (the larger of the readings before
   and after the steps) + device residency at n + the extra memory of two AR
   steps with activation checkpointing on, on the largest held instance.
   Usable host memory = the cgroup limit (bounded by MemTotal); without a
   cgroup limit, the MemAvailable read at the audit's start; the source is
   recorded. Residency at a pool size the audit did not reach is extrapolated
   linearly from its last two marks. Not measured: the runner's parent process
   (in an `ar_only` run it holds file lists, its torch import and a CUDA
   context, a few hundred MB on each side); the 20% margins cover it.
3. **Activation checkpointing off** in wp9's 2D configurations iff the AR step
   in E1's worker set-up is >= 1.3x faster without it (`worker3` against
   `worker3_no_ckpt`, ms per step per unit, both valid) AND rule 2's choice
   still fits with the extra memory of steps without it; otherwise on.
   Checkpointing is memory-only and exact (bitwise on CPU, tested); on CUDA
   the attention backward's atomic reductions make any two runs differ
   slightly, with or without it.
4. **C0.4 is a prediction only**: a training-set / validation-set energy-gap
   ratio < 0.8 reads "the 1,024-instance arm overfits: C1 likely improves in
   band"; otherwise "near parity: C1's in-band gain likely small". No C1
   quantity or criterion depends on it.
5. **For C2 later (recorded only)**: the CG energy-drop estimate is usable as
   a score iff its seed-mean Spearman correlation with the true gap at k = 10
   is >= 0.9 (undecided if there is no reading at k = 10 or the correlation
   is undefined). No session-2 effect.

Each rule is evaluated on its own: a missing or malformed input leaves the
rules that read it undecided (null, with the error; rule 3 also when rule 2
is undecided), the others are decided, and the script exits non-zero. Rules 2
and 3 refine the plan's wording (host memory only; "off if >= 1.3x faster"):
the device memory is checked too, and a faster step never costs the pool or
the parallel schedule.

## Stage 0a (1 Oct 2026) -- session-1 instruments, CPU-validated

1. **Checkpointing switch.** `model.activation_checkpointing` in
   `FEJEPAConfig`, default true (the stamped behaviour); it sets the encoder's
   existing `use_checkpoint` attribute (D13), a plain attribute, so state
   dicts are unchanged. Tests (`test_w9_ckpt_switch.py`): default and round
   trip; the switch reaches the encoder with identical state dicts; AR
   training through `pretrain_unit` with the switch on and off gives
   bitwise-identical states on CPU.
2. **OOD-2D v1** (`fejepa/fe/ood2d.py`, `scripts/w9_make_ood2d.py`). Each
   family changes one attribute of the training family: F1 4-6 holes
   (training 0-3); F2 slender plates, width 3.0-4.5 and height 0.6-0.8
   (1.5-3.0, 0.8-1.5); F3 1-3 holes, all large (0.16-0.22) or all small
   (0.03-0.06) x min(w, h), one class per instance with probability 1/2
   (0.06-0.16); F4 Poisson ratio 0.40-0.45 (0.25-0.38); F5 mesh size
   target_h = 0.025 (0.05-0.12). R: 16 training-family geometries, each
   meshed at h = 0.12, 0.085, 0.05, 0.035, 0.025. Seeds 91001-91006; labels
   by direct solve (ledger stage `evaluation-labels`); manifests with
   per-file SHA-256 and no volatile field. Secondary differences that come
   with the changed attribute (reported, not corrected): F1's whole-set redraw
   favours smaller radii slightly (mean radius / min(w, h) 0.106 against the
   training family's 0.109); F3 has no hole-free plates, and its large holes
   may leave an edge ligament down to 0.02 x min(w, h) (training: at least
   0.04). The script generates F5 and R first (rule 1 reads them), records a
   failing family and continues, and refuses an existing directory whose
   manifest is not the family asked for (name, seed, size; for R the mesh
   sizes) or whose files differ from it. Tests (`test_w9_ood2d.py`): each
   family keeps every other attribute in the training ranges (400 draws);
   determinism; generated labels solve their systems; regeneration reproduces
   the manifest; a tampered or missing file is detected; on R the edge
   tractions' resultants are mesh-independent to 1e-9 and gravity's to 1e-2
   (the polygonal holes' area changes with the mesh).
3. **C0 readings** (`fejepa/analysis/w9.py`, `scripts/w9_c0.py`), on E1's base
   states, all inference: `val` -- C0.1, the seed-mean relative gap split
   exactly into the ensemble's gap plus the seed disagreement D (parallel-axis
   identity in the K inner product; residual reported) with a label-free D;
   C0.3, the energy drop of k unpreconditioned CG steps (k = 5, 10, 20, 40), a
   label-free lower bound of the gap, and its Spearman correlation with the
   true gap; the reproduction of E1's own per-instance arrays. `trainval` --
   C0.4. `amp2d` -- C0.7-2D: c* and c_b in band, on F5, and per h on R, with
   the predicted and the exact energy norms relative to the coarsest mesh.
   `memory` -- C0.6: one process holding the pool prefix as a training unit
   does (instances, packs, CPU-resident anchors) to 1,024 / 4,096 / 12,800 /
   25,600, host and device residency per mark, the CUDA context before and
   after, the extra device memory of two AR steps on the largest held
   instance with checkpointing on and off, and the host residency after
   them. States, corpus and families (by name and per-file SHA-256) are
   verified before use. Tests (`test_w9_analysis.py`, `test_w9_c0_cli.py`):
   the identity, CG drops against scipy's iterates and as non-decreasing
   lower bounds, a pure amplitude error recovered (c* = 1.25), tie-aware
   Spearman, every decision rule at and around its thresholds (inclusive and
   exclusive sides), the memory terms, the rules failing one at a time; and
   every subcommand end to end on a miniature E1-like run (reproduction
   exact, swapped or tampered families and a foreign state refused, the step
   measurement's code path with the allocator queries stubbed, an unreadable
   input leaving only its rule undecided), and the generation script refusing
   a different family while finishing the others.
4. **2D timing** (`scripts/posthoc_profile_2d.py`): variants `worker3_no_ckpt`
   (E1's worker set-up, three units, checkpointing off through the units'
   configuration) and `threads_w3_no_ckpt`.
5. **Runbook**: RUNBOOK_W9.md, Sec. 0-1 (session 1, ~1-1.5 h, plus the optional
   torch-stack timing).

Rehearsed in the sandbox on a miniature E1-like run (gmsh corpus of 48
instances, model width 16, three seeds, CPU): every session-1 command ran with
sizes scaled down. Suite 355 (the pushed text said 350: five review tests were
added after the count was written; corrected at Stage 0b).

## Stage 0b (1-2 Oct 2026) -- session 2, CPU-validated; PREREG_W9 r2 drafted

The PI asked for every piece of code before the box runs anything (Song is
away for a few days), so session 2 is written now and PREREG_W9 can be stamped
BEFORE session 1, covering every arm the session-1 rules can select: the box
can then run both sessions in one visit, and no reading precedes the stamp.
Whether to stamp now is the PI's decision (the alternative, stamping after
session 1, needs no code change, but session 1's decision step must then run
again on the stamped commit: the plan and the adjudication require the
decisions to have run on `prereg-w9`). After the stamp the branch is frozen
until session 2 has returned (the box checks that HEAD is exactly
`prereg-w9`; the plan and the command script check it again).

1. **S switch** (`model.decode_scale`, `model.decode_scale_factor`,
   `model.features.load_density`; off by default). `decode_scale = "l1"`: the
   decoded field is multiplied by `decode_scale_factor` times the sum of the
   absolute values of the battery's nodal force components, sum |F|
   (`battery_l1`), instead of the largest of them -- for consistent nodal
   loads the former is the
   integral of the tractions and body forces and does not change with the
   mesh, the latter shrinks like the element size. wp9 uses the factor 1/64:
   over 256 training-family instances drawn in the sandbox, max|F| / sum|F|
   had median 0.0156 (10th-90th percentile 0.0098-0.0226) and grew with the
   mesh size, so at training mesh sizes S asks the network for outputs of
   E1's level on the median instance (without the factor, 40-110x smaller;
   the code review's design note). `load_density`: the per-node load
   columns become load densities -- nodal force over the boundary length
   (traction) or area (body load) the
   node carries, from the facets or cells whose vertices are all loaded;
   exact for uniform loads with P1 elements; a point load falls back to the
   node's own share -- over the battery's largest density, and the load
   summary becomes ratios of mesh-free totals (this case's sum |f| and
   resultant over the battery's sum |F|; the loaded fraction of the boundary
   or body). `load_densities` classifies a case as a body load iff it loads
   an interior node. Dimension-generic (triangles and tetrahedra). The
   factor must be positive and applies to "l1" only; "l1" needs the scaled
   decode; `run_config` refuses S keys for any model kind but fejepa, and S
   with MGN units. Tests (`test_w9_s_switch.py`, 16): the default features
   are wp8's bit for bit (2D and 3D, every channel combination, against
   wp8's function verbatim); default spec and decode scale unchanged;
   refusals; simplex geometry (incl. a conforming Kuhn 3D mesh); densities
   exact and uniform for uniform tractions and gravity; on structured
   remeshes (6 x 4 to 24 x 16 cells) the S load columns, summary and decode
   scale agree to 1e-9 while wp8's change; on gmsh remeshes with holes (h
   0.12 and 0.05) the densities equal the applied tractions and gravity to
   1e-9 and the summary agrees to 2%; equivariance; the point-load fallback;
   an S model prepares exactly the S scale and features (pack against
   `battery_l1` and `build_features_battery`); its AR loss scores exactly the
   field inference decodes (the D14 spy); it trains through the unit path.
   Observations, not changed: the tet3d smoke backend's hexahedron split
   (`_HEX_TO_TETS`) covers 5/6 of each cell and is not conforming (no
   deciding run used it; Phase-2 used gmsh3d); gravity on a mesh without
   interior nodes would be classed as a traction, and a case mixing traction
   and body force gets non-uniform densities (neither occurs in the wp9
   corpora: classification and exactness were checked on F1-F4 and R).
2. **Evaluation additions** (`experiments/w9_eval.py`, wired through
   `runner.run_config`, `e8_regimes.run_e8`, `parallel.pretrain_unit`), all
   opt-in: a configuration block `evaluation` with evaluation-only `holdouts`
   (verified before anything trains: manifest family, per-file SHA-256,
   labels; recorded in the report with each manifest's SHA-256) and
   `amplitude` (per instance: c* per load case, c_b, the c*-scaled errors,
   energy norms of prediction and solution); `e8.reuse_from` -- an
   evaluation-only E8 on another run's states, refused unless each file's
   SHA-256 is the one that run's report records (seeds from its d9 block, one
   pool size) and the configuration equals that report's outside the
   evaluation-only keys; `e8.seed_offset` (independent replicates; 0 = as
   before); and `run-config --activation-checkpointing on|off`, a run-time
   setting like `--workers`, recorded in the report as `runtime_overrides`.
   `run_config` refuses the runner keys inside `experiments.e8`, `reuse_from`
   without `ar_only` or with P3, and `seed_offset` with P3. The dry run
   verifies holdouts and reused states too. `analysis.w9.predictions`
   refuses models with different preparation settings (one shared pack
   would decode with the wrong scale). Tests (`test_w9_runner_eval.py`,
   14): a configuration without the blocks gains no key; with them,
   training is bitwise unchanged and the validation arrays identical;
   holdouts and amplitude recorded; the evaluator equals the frozen one plus
   the amplitude block; the amplitude readings are the predictions' own
   (recomputed from a reloaded state) and the summary by hand; the reused
   arm trains nothing and reproduces the source's arrays; foreign states,
   changed configurations, unlabelled holdouts, a source with several pools
   or a missing state refused; the override recorded and exact; holdouts
   refused before training; unsupported combinations refused; an S arm and
   a seed offset run through the runner.
3. **Process pool and reports, robust to a killed process** (scheduling
   only). `parallel.map_units` runs units in a
   `concurrent.futures.ProcessPoolExecutor` watched every 30 s: a worker
   killed from outside (the out-of-memory killer) -- busy, idle, or half-way
   through sending its result -- ends the map with an error that says to
   restart with `--reuse-states`; a unit that raises ends it at once; the
   other workers are terminated, not waited for, and the process exits
   (after a death mid-send, closing this process's end of the result pipe
   releases the executor's reader, which the interpreter would otherwise
   join for ever at exit). With `multiprocessing.Pool` (until now) a killed
   worker made the arm wait for ever with no output (the operations review),
   and Pool's shutdown can block on a queue lock a dead worker held. Each
   worker also exits within 5 s when its parent dies (no orphan unit keeps
   training). A first fix with the Linux parent-death signal was withdrawn:
   the signal is tied to the thread that started the worker, so Pool's
   replacement workers died at shutdown holding the queue lock and the parent
   hung -- found by the rehearsal below. The poll and the pipe release use
   the executor's private `_processes` and `_result_queue` (present in
   Python 3.11 and 3.12; the code review ran the kill tests on 3.11.15,
   3.12.0, 3.12.3 and 3.12.11). `write_report` writes atomically (temporary
   file and rename). Tests (`test_w9_parallel_kill.py`, 6): a busy worker,
   an idle one (two distinct workers forced) and one killed while sending a
   200 MB result (the process must exit; the test fails without the pipe
   release and hangs without the poll); a failing unit; a killed parent (its
   workers exit); and E8's AR units with holdouts and amplitude through the
   pool equal the inline run (states byte for byte, metrics identical;
   single-threaded torch, the E1 contract of `test_experiments_smoke.py`).
4. **Default path against wp8-lejepa** (`scripts/regress_against_branch.py`,
   CPU, both sides in their own processes, fresh and restart passes): a
   miniature of E1's base configuration (2 states, 162 numbers) and of the
   Phase-2b configuration (6 states, 2,590 numbers) -- identical, on the
   final Stage 0b code (`records/wp9/`, with a README naming the code). The
   miniatures run units inline; the pool path is covered by the
   parallel-equals-inline tests of item 3. The tool now lets a miniature of
   an `asis` configuration generate its own corpus. This is what licenses
   reusing E1's states for C1's 1,024 arm.
5. **Configurations** (`scripts/make_w9_configs.py`; `configs/w9_c1_n1024`,
   `_n4096`, `_n25600`, `_n12800`, `w9_b_n1024`, `w9_s_n1024`): E1's base
   (refused unless its canonical SHA-256 is PREREG_E1's stamped one) plus the
   evaluation block, output directory, guard on PREREG_W9, and per arm the
   pool size and epochs (204,800 steps per seed), `reuse_from` (1,024 arm),
   `seed_offset` 3 (the fresh baseline and S) or S's three keys. Tests
   (`test_w9_configs.py`, 5): byte-for-byte regeneration; differences from
   E1 only in PREREG_W9 Sec. 2's keys; PREREG_W9's labelled lines name
   exactly these configurations (and, once stamped, their hashes and the
   file's self-hash); every configuration validates under the dry run.
6. **PREREG_W9.md r2** (draft): the arms, the evaluation sets, the session-1
   selection, measurements; H1 -- the N_max arm's in-band energy gap lower
   than the 1,024 arm's beyond max(10%, 2 SE_rel) (the E-series guard, seed
   means; by simulation at E1's sample noise, 4.9%: about 0.4% false support,
   48% / 91% / 99.7% detection of true 10% / 15% / 20% reductions); H2 -- S
   against the fresh baseline (seeds 3-5): F5 displacement error lower AND
   the F5 / in-band displacement ratio lower, each beyond the guard, AND K1
   on both in-band metrics; secondary readings (exploratory); mechanics;
   preconditions with costs; deviations.
7. **Adjudication** (`analysis/adjudicate_w9.py`, `scripts/adjudicate_w9.py`,
   from the two box returns). Session-level faults refuse the adjudication:
   decisions that the frozen rules, recomputed from the returned readings,
   do not reproduce, or that other rules, other readings or another commit
   than `prereg-w9` wrote; an incomplete OOD record or a foreign R manifest;
   a plan made from another decision file; a session-2 return without its
   plan, status or provenance file; a report the provenance file lists that
   is missing or has another SHA-256; an arm the status file shows finished
   without its report (a verdict cannot be avoided by leaving a report out).
   A report that fails its own checks is refused alone -- left out, recorded
   with the reason: not a guard-verified run under its own CONFIG_SHA256
   line, another commit, a difference from E1's base beyond Sec. 2's keys,
   not the stamped arm for its role, another corpus or seeds, a cell without
   one evaluation per seed, holdout manifests other than session 1's, a
   1,024 arm that is not E1's states or does not reproduce E1's validation
   arrays (1e-4; a non-finite value never reproduces), an N_max or S arm the
   rules did not select. A hypothesis without its valid reports, or with a
   non-finite reference, is NOT EVALUATED; the other stands. Readings the
   session could not produce (a failed timing step: rule 3 undecided) are
   read as the decision script read them. Deviations: the run-time settings
   against the plan, restarts and repeated attempts (from the status file),
   resumed units, a non-zero solve ledger, a stack or GPU other than E1's.
   Secondary readings never block a verdict; they include S against the
   fresh baseline on every set and the fresh baseline against E1's states.
   The verdict file records every input's SHA-256 and the code's.
8. **Session-2 plan** (`scripts/w9_session2_plan.py`): the gate (the
   required session-1 steps exited 0; the decision file is the frozen rules'
   output on the readings beside it, recomputed, and was written on
   `prereg-w9`, which the checkout must also be; rules 1-2 decided and a
   pool found; the
   evaluation sets (six; seven with IB since r3) complete and the manifests
   on disk the record's; E1's
   validation arrays reproduced to 1e-4; a GPU and E1's torch; 5 GB free),
   the selection and order (1,024; N_max; the fresh baseline -- since r3 in
   every session 2; S if admitted; 4,096), workers (rule 2's for N_max) and checkpointing (rule 3;
   kept on for an arm that would not fit without it, or when rule 3 is
   undecided), a dry run of each selected configuration, a summary line, and
   the command script: it changes to the repository, refuses to run twice at
   once or without a GPU and E1's torch, checks before every arm that the
   code is still `prereg-w9`, skips an arm whose report is complete, moves an
   unreadable report aside, restarts an arm with states and no report under
   `--reuse-states`, keeps every attempt's log and time-stamps each attempt;
   on STOP no command script is left. Tests
   (`test_w9_session2.py`, 15, on a miniature laid out like the box -- an
   E1-like base stamped and run, its report copied to records/, the
   evaluation sets under runs/w9/ood2d/, the six arms from the real
   generator stamped against a PREREG_W9.md and run, session-1 readings
   written so that the real decision script decides as each test needs):
   the guard's semantics; every arm adjudicated; H1's verdicts on
   constructed values (reference, metric, divergence); H2's three conditions
   (a lower F5 error with the ratio unchanged is not support; K1 on each
   in-band metric), divergence and a void reference; every per-report
   refusal, each leaving only its own hypothesis not evaluated; every
   session-level refusal; a missing primary arm; the decisions' integrity;
   every deviation; the CLI from the two returns (and with the timing
   reading absent, a report missing or altered, the plan missing, an
   unstamped file); selection, order, flags and the real pre-flight;
   checkpointing kept where memory would not fit, and on when rule 3 is
   undecided; every gate stop (incl. the frozen rules' hash, the
   recomputation, the commit, the amplitude reading's manifests, a
   non-finite reproduction); the command script's first run, skip, restart
   after a truncated report, and its stop on another commit. Eight targeted
   mutations of these checks are each caught.
9. **Runbook** (RUNBOOK_W9.md): one visit in one tmux session -- Sec. 0
   (fetch in a subshell with an explicit FETCH-OK, the `prereg-w9` and tree
   checks, disk, nothing running, the GPU idle, the suite), session 1
   (earlier attempts' logs kept on a re-paste), its return (sent at once),
   the plan with GO/STOP, the out-of-memory baseline and the memory sanity
   check, the runs with their normal signs and durations, a health check by
   process counts and GPU use (inside a container, nvidia-smi may list no
   processes), the interruption and crash policy, the return (with the
   reports' SHA-256), the adjudication, and Sec. 4 (the optional torch-stack
   timing, formerly 1e, moved after session 2, with its own directory: pip
   in a subshell, no cache, temporary files on the data disk, the venv's
   torch checked before timing).

**Pre-run reviews (2 Oct).** Three independent reviews of the r1 state --
code and default path, pre-registration and adjudicator, operations -- found
no defect in the default path and these issues, all addressed above:
H2's reference was the very states rule 1 selected on F5 (winner's curse;
now a fresh baseline on fresh seeds), and H2 tested the F5 level rather than
its growth (now the ratio too, and K1 on both in-band metrics); S changed
the output level by 40-110x (now the factor); the decision file and each
report's stamped line and commit were not checked (now they are, and the
decisions are recomputed); a killed worker hung the arm silently; a restart
overwrote the failed attempt's log; a truncated report was skipped as
complete; nothing checked the GPU and torch before the arms; the one-visit
flow (tmux, the session-1 to session-2 step, 1e's network and disk use) and
the costs (the sequential branch) were underspecified; the simulation used
the population spread; exactness claims are bitwise on CPU only; several
paths were untested (now tested: the S pack, the amplitude readings, the
CLI, the refusals listed in items 2 and 7). Two further independent reviews
of the resulting state (code; documents and operations) found: the suite
would fail on the stamped head (a test read the repository's own describe;
fixed, and the suite is now run on a stamped scratch clone before stamping);
the adjudication crashed on a session-1 return without the timing reading;
a verdict could be avoided by leaving a report out of the return; a failing
secondary report blocked both verdicts; the plan did not recompute the
decisions or check the commit; a worker killed while sending its result
hung the process at exit; NaN slipped through the reproduction checks; the
health check relied on nvidia-smi listing processes inside a container;
several checks were untested (eight mutations survived). All are fixed above.

**Rehearsal** (sandbox, 2 Oct, final code; `rehearse.sh` outside the
repository): a box-like miniature (gmsh corpus of 48 instances, model width
16, three seeds, CPU; E1-like base stamped, run and copied to records/; the
six arms generated, PREREG_W9 stamped with its self-hash, committed and
tagged `prereg-w9`): Sec. 0's describe and clean-tree checks; session 1
(1a-1d) with the real commands at small sizes; the rules' inputs then edited
(marked) so that the frozen rules choose the branch that runs every arm (S
in, N_max 25,600 one seed at a time, checkpointing off) and the real
decision script run on them; 1f; the plan (GO: the decisions recomputed,
the commit checked, pre-flight passed); the command script with one worker
of the 4,096 arm killed mid-run (the arm stopped within a second with the
worker-death message, no worker left running); the re-run skipped the
finished arms and restarted that one from its states (earlier log kept);
2d (the reports hashed in the provenance file); the adjudication CLI on the
two returns (every run on `prereg-w9`, the decisions recomputed, the return
checked against its status and provenance files, no report refused, H1 and
H2 issued, the restart recorded). Its numbers mean nothing (16 steps).

Suite: 411 (Stage 0a 355 + 56) passed.

## Stage 0c (2 Oct 2026) -- pre-stamp review; PREREG_W9 r3; CPU pilot of S

PI decisions (2 Oct). Stage 0b's four recommendations approved: PREREG_W9
stamped before session 1 so that both sessions run in one visit; S's factor
1/64; H2 against a fresh baseline on seeds 3-5 with three conditions; the r2
text. The stamp waits until the PI calls the run. After the pre-stamp review
below, r3 approved as recommended (A1-A3, C1-C12; options B1, B2, B3, B5; not
B4, a scale-only ablation arm, nor B6, fallbacks for a failed gate); and the
in-band holdout IB in H1's in-band set (option C, chosen by the PI over A,
disclosure only, and B, capping each instance's value at 0.25).

1. **Pre-stamp review** of r2 (mine and an independent one). The main
   finding: the seed spread of H1's reference (E1's states, in band
   0.0374 / 0.0405 / 0.0371) comes from one validation instance (#233: 0.760
   in seed 1 against 0.011 in seeds 0 and 2; without it the coefficient of
   variation is 0.6%); one such blow-up moves a 256-instance arm mean by
   about 2.6%, so the 10% floor meant 7.9-13% of the reference's clean level
   depending on how many blow-ups N_max shows. r3 adds IB (2,048 fresh
   training-family instances, seed 91007) to H1's in-band set (2,304
   instances: a blow-up then moves an arm mean by 0.2-0.3%) and states the
   finding; it also says what a verdict licenses, corrects Q1's reading of
   the theory note (a statement about cost) and Q2's "remove", states H2's
   operating characteristics, its remaining selection on F5's instances, the
   bundle's scope, the two ratio definitions and the uninformative case,
   commits the 1/64 measurement and every simulation it quotes, gives the
   reuse chain from E1's commit (`cdac731`: post-hoc modules, an atomic JSON
   write and the saved-state container changed since, none on the training
   path), records the CPU pilot, and names the stamp script. Options: B1 the
   fresh baseline runs in every session 2 (a coarse on-box check of the
   training path behind the reuse, and N_max against it beside H1); B2
   per-seed medians, Welch and instance-resampling intervals beside H1 and
   each H2 condition; B3 a growth reading on R; B5 the adjudicating code
   compared with `prereg-w9`. No threshold changed.
2. **Code.** `fe/ood2d.generate_inband` and session 1's `w9_make_ood2d.py`
   (IB, seed 91007, 2,048); every configuration evaluates IB (all six
   regenerated; the configurations differ from E1's base only in Sec. 2's
   keys, as before); the plan's gate checks seven sets; the adjudicator
   decides H1 on the validation split and IB together (`pooled_seed_means`,
   each set weighted by its instance count; H2 and rule 1 read the
   validation split alone), reports the pooled in-band readings in every
   arm's table, the readings beside the verdicts (`robustness`,
   `welch_interval`, `instance_bootstrap`), R's growth (`remesh_growth`),
   the fresh-baseline flag and H2's uninformative note -- none of which can
   change or block a verdict -- and the ratio of seed means (rule 1's form);
   the plan and the adjudicator expect the fresh baseline in every session 2;
   the CLI compares the adjudicating files (incl. `report.py`) with
   `prereg-w9` (`code_against`) and records a difference as a deviation. A
   worker's death names the workers' exit codes (`parallel._died`; -9 is the
   out-of-memory killer's signal). New scripts: `stamp_prereg_w9.py` (the
   stamp of Sec. 8 in one step: refuses anything but the r3 draft, fills the
   six lines and the status, computes and verifies the self-hash),
   `w9_scale_factor.py` (`records/wp9/scale_factor.json`), `w9_sims.py`
   (`records/wp9/simulations.json`), `w9_pilot_s.py`
   (`records/wp9/pilot_s_cpu/`), `make_w9_paper_material.py` (tables from
   the verdict file; diverged values print as such; LaTeX-checked);
   `paper/wp9/scale_lemma.tex` (why max|F| shrinks like h^(d-1) and the
   summed |F| does not, with proof-status flags). `tests/test_w9_records.py`
   ties the regression record to the committed `src` tree.
3. **CPU pilot of S** (exploratory, toy scale, own seeds;
   `records/wp9/pilot_s_cpu/README.md`): no red flag; at that scale the
   baseline's median c* on an F5-like set was 2.4-2.7 and S's 0.94-0.98, and
   S's predicted energy norm followed the exact one across R-like mesh
   sizes. A side finding: on CPU, PyTorch's inference path of
   `nn.MultiheadAttention` materializes the attention matrices (about 3.9 GB
   at 7,760 nodes with 4 heads and 4 load cases), which killed a pilot
   worker in the sandbox; CUDA evaluates without them (Phase 2 evaluated
   about 4e4-node 3D meshes with this encoder on the box), and no box step
   evaluates on CPU.
4. **Independent review of Stage 0c** found thirteen issues, all addressed:
   the adjudication crashed on a non-finite fresh baseline (the flag's
   format; now flagged as diverged, tested); the paper tables crashed on a
   diverged arm (now printed as diverged, tested); statement (b) of the
   scale lemma needed load cases that are a traction or a body load alone;
   three numbers were double-rounded; the stamp comparison of B5 now uses
   `git diff` and covers `report.py`; H2's secondary readings moved out of
   its try block; "a check" of the reuse became "a coarse check"; LaTeX's
   "<"; and wording in r3.
5. **Regression** re-run on the final Stage 0c code (`src` tree
   `710ccb32...` in `records/wp9/README.md`): identical (162 and 2,590
   numbers, every state file byte for byte). The summaries now name the two
   sides by their `src` trees -- git tree ids computed from the files
   (`regress_against_branch.src_tree`, tested against `git write-tree`; the
   other side is `wp8-lejepa` at `5f8e2df` exactly) -- instead of the
   sandbox's paths, which Stage 0b's copies held (they remain in the
   history); `tests/test_w9_records.py` checks the summaries' trees against
   the README.
6. **Independent review of the IB change** (IB, the simulations, the
   workers' exit codes, the r3 text): nothing that could break the one-visit
   run or change a verdict; three medium and seven low findings, all
   addressed. A commit hash the remote does not have, in the pilot's README
   (removed). The sandbox paths in the regression summaries (item 5). No
   suite count for the box's step 0c to compare with (RUNBOOK and this
   section now give it). Nothing checked IB's size or seed:
   `w9_make_ood2d.py --n-inband` could have changed H1's set unnoticed (the
   plan's gate and the adjudication now refuse a session-1 record whose sets
   are not Sec. 3's sizes and seeds, `ood2d.SET_SIZES` and `DEFAULT_SEEDS`,
   which are also the script's defaults). The tests could not tell weighted
   from unweighted pooling (the miniature's IB now holds 12 instances against
   the validation split's 4). A report without per-instance arrays would
   have stopped the whole adjudication (`check_arm` now refuses it alone).
   r3 names the in-band set each reading uses (H1: the validation split and
   IB; H2, rule 1 and every F5 / in-band ratio: the validation split), states
   that a set that fails to generate stops session 2, and its revision line
   no longer overclaims. Stale text (session 2's 10-19 h, docstrings, the
   pilot README's seeds). Two simulation sections shared a random stream
   (`exploratory` has its own now: its quoted 4-6% unchanged; every other
   section byte-identical).
7. **The evaluation sets at full size** (`records/wp9/ood2d_sandbox.json`):
   step 1a as the runbook runs it, in the sandbox -- all seven sets
   generated and verified from their seeds, no failed draw (3,408
   instances, 5.5 min on one CPU, 0.83 GB; gmsh 4.15.2). The instance files
   are byte-reproducible on one machine, so session 1's manifests can be
   compared with these.

Suite: 427 (Stage 0b 411 + 16) passed. On the box: 427 passed, or 426
passed and 1 skipped where pdflatex is missing (the paper tables' LaTeX
build).
