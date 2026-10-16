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

## Stage 0d (2 Oct 2026) -- the paper's cost table, timed on the box

PI decisions (2 Oct, before any wp9 result exists): the CMAME manuscript is
written now from the record up to wp8 and submitted as soon as possible;
wp9's results are not part of it and are reported separately whatever they
are; the inference time is measured on the box in this visit; the manuscript
does not cite arXiv:2604.01349 (as PROVENANCE_NOTE.md requires; this
supersedes the prior-art list at the top of BRANCH_NOTES_wp8-lejepa.md on
that point).

1. **`scripts/time_inference_vs_solve.py`**: per instance, on the same
   instances and machine, the surrogate's inference (`prepare_instance`,
   `forward_instance` repeated and synchronised, the copy to the host; batch
   of one, every load case in one pass, the run's TF32 policy; cold, i.e.
   with the first pass, as the headline, and warm) and its accuracy there
   (the evaluation's own `evaluate_fields`), against three exact solves of
   the free system: `solve_fe_displacement(method="direct")`, the SuperLU
   call that bought every label (the labelling cost); unpreconditioned CG to
   a relative residual of 1e-10 from zero (the 3D generator labels with CG
   by default for that reason: a measured crossover at about 7k dof,
   `fe/gmsh3d.py`); and the same CG from the surrogate's prediction
   (iterations against CG from zero). A solve under
   one second is repeated three times; each solution is checked against the
   stored labels. Assembly is excluded from both sides (the archives hold K
   and F; the surrogate needs no K). One SHA-verified state is timed; its
   metrics on the first validation instances are compared with the report's
   own per-instance arrays. Sets: the run's validation split, its fine set
   (3D), and OOD-2D families by name (manifest and files verified; a family
   failing the check is skipped and recorded, exit 3; a family of the other
   dimension is refused before any timing). Each solver stops starting
   solves after a per-set budget; the surrogate is timed on every instance.
   One progress line per instance. Reported only.
2. **RUNBOOK Sec. 4** (new; the torch-stack timing is now Sec. 5): after the
   session-2 return, and also after a STOP in 2a, since it needs nothing
   from session 2; its own GPU and process checks first (recorded); E1's
   states on 32 validation and 32 F5 instances, Phase-2b's on 32 in-band and
   8 fine instances; about 1 h; read only; its own return.
3. **Tests** (`tests/test_w9_timing.py`): every timer and solver on a 3D run
   with a fine set (the direct solve reproducing the stored labels to
   round-off, fast solves repeated, CG iterations recorded, a CG cut short
   by its iteration cap recorded as a fallback); the check
   against the report's arrays built by the evaluation's own code path (0 for
   the matching seed, a planted difference found); a 2D run with an OOD
   family under a zero solve budget; a family other than the one named
   skipped with exit 3 while the rest is timed; an empty set; a family of
   the other dimension and a state other than the report's refused; the
   summary statistics.
4. **Independent review** of this stage before packaging: eleven findings,
   all addressed -- chiefly that SuperLU alone overstated the 3D speed-up
   against an exact solver (CG added, and CG from the prediction), that a
   STOP in 2a skipped the timing, that the cold time and the accuracy on the
   timed instances were missing, and that the checks before the runs left no
   record.

Suite: 431 (Stage 0c 427 + 4) passed. On the box: 431 passed, or 430 passed
and 1 skipped where pdflatex is missing (the paper tables' LaTeX build).

## Stage 0e (2 Oct 2026) -- the paper's field figures; CG at the surrogate's accuracy

PI decisions (2 Oct, before any wp9 result exists): the CMAME manuscript
shows field figures of Phase-2b's models (three instances, chosen by rules
fixed in the export script); the cost table adds the time an exact solver
needs to be as accurate as the surrogate.

1. **`scripts/export_fields.py`** (box, RUNBOOK Sec. 4): Phase-2b's
   label-free, labels-only and graph-network states (seeds 0-2, the largest
   budget, the run's own `kind: mgn` build for the last), on the run's own
   instance lists. The AR states are checked against the report's
   d9_restart record and refused otherwise; the supervised ones against
   `records/wp9/phase2_supervised_states.json`; every model also by
   content -- its per-instance displacement error and relative energy gap on
   the validation split, recomputed with the evaluation's own functions,
   against the report's arrays. A supervised state is used if its hash or
   its content matches (median relative deviations at most 1e-3, for the
   graph network's non-bitwise CUDA reductions); one failing both is left
   out, recorded, and the script ends with exit 4 after writing the rest;
   a state used on its hash whose content does not match is recorded as a
   content mismatch (exit 5). The instance lists and the three figure files
   (present, labelled) are resolved before any model runs; the validation
   figures reuse the energy pass's seed-0 predictions, so their energies are
   those of `energies_val.npz`; `fields.json` records torch, CUDA and the
   TF32 flags.
   Outputs: per model, seed, validation instance and load case, Pi_h(u),
   the relative energy gap, c* and the gap of c* u (`energies_val.npz`;
   the counts of load cases with Pi_h(u) > 0, i.e. worse than the zero
   field in the energy norm, and of instances whose load-averaged gap
   exceeds 1, in `fields.json`), and three figure files with the mesh, the
   reference, the seed-0 predictions, element von Mises stresses and the
   per-load energies. The rules (from the report's arrays): fig5, the
   validation instance with the labels-only arm's largest gap (seed 0; the
   other seeds' choices recorded -- in the record all three are validation
   index 107, `instance_00485.npz`); fig6, the label-free arm's median gap
   (seed 0); fig7, the fine-set instance with the label-free arm's median
   displacement error (seed 0). The fig5 rule was set after reading the
   per-instance records, before any field was looked at; the manuscript says
   so. Reported only.
2. **`records/wp9/phase2_supervised_states.json`**: the six supervised
   states' SHA-256 from the Phase-2 return package of 22 September 2026
   (no report records them; Phase-2b reused those units and read those
   files as its P3 shared checkpoints); README section added.
3. **`scripts/time_inference_vs_solve.py`**: two more kinds, on by default.
   `cg_match`: CG from zero, the `cg` solver's own CG, stopped on each load
   case at the first step whose relative energy error is at or below the
   surrogate's relative gap there (0 steps when the zero field already is);
   the steps are found by an untimed pass with an energy-recording callback,
   then that many steps are timed without it. `cg_match_cstar`: the same for
   the prediction rescaled by c* (its time `cstar_s`, one mat-vec with the
   full K, the median of three runs, is added to the surrogate's side; its
   accuracy recorded as `energy_gap_rel_cstar` and `disp_rel_l2_cstar`).
   One trace per load case serves both kinds and stops once the smaller
   target is met; it also records CG's displacement error at the matched
   step and how the trace ended; the timed run's step count is recorded and
   compared with the one found. The untimed pass counts toward the per-set
   budget. The stop is an oracle (the labels decide it), so `cg_match` is a
   lower bound on any CG run to the surrogate's accuracy, matched in the
   energy norm; the manuscript says so.
4. **RUNBOOK Sec. 4**: the export after the timing (about 10-20 min,
   estimated; Sec. 4 now about 1.5 h); the return copies directories
   (`cp -r`); expected outputs and exit codes; header and suite count.
5. **Tests**: `tests/test_w9_fields.py` (6: end to end on a Phase-2b-shaped
   run with all three model kinds -- the rules, the content checks at 0, the
   energy identities, c* against its formula, the stresses, the validation
   figures' energies equal to the energy pass's; a supervised state with
   other bytes but the same weights used, one with other weights left out
   with exit 4 and the rest exported; a hash-verified state whose arrays
   differ recorded with exit 5; a missing figure file refused before the
   long pass; the rules on Phase-2b's own report (107 in every seed, 154,
   77) and the hash record's keys; refusals of a changed AR state, a missing
   supervised state and a report without P3 arrays);
   `tests/test_w9_timing.py` (+2: the trace is CG's own energy error,
   monotone, and stops at a target; one trace serves both kinds; the match
   is the first step at or below the target and the timed CG makes exactly
   those steps and reaches it; unreachable and zero-step targets recorded;
   c* with the full K equals the free-dof value; an unlabelled instance
   skips the matching kinds; the end-to-end test checks the new kinds'
   steps against CG's and the rescaled targets against the raw ones).
6. **Independent review** before packaging: no blocker; the three findings
   to fix (a content mismatch on a hash-verified state was silent; the
   figure inputs were read only after the long pass; `cstar_s` included a
   free-block extraction c* does not need, from one sample) and the minor
   ones (a shared, early-stopped trace instead of two full ones; the timed
   step count checked; how a trace ended; the timed CG's tolerance tied to
   `--cg-tol`; the runbook's expected values and file names; torch and TF32
   recorded; `--n-val 0` and the default hash path; the oracle stop
   documented) are all addressed.

Suite: 439 (Stage 0d 431 + 8) passed. On the box: 439 passed, or 438 passed
and 1 skipped where pdflatex is missing.

## Stage 0f (9 Oct 2026) -- PREREG_W9 r3 stamped; the runbook brought up to date

Between Stage 0e and this stage wp9-pool did not move, and the box ran
cmame-paper's RUNBOOK_CMAME (branched from Stage 0e), all of it on a new host:
on 8 October its Sec. A, this runbook's Sec. 4 run on its own without the F5
family (the paper's timing and field export, on cmame-paper's `61f018f`),
then CM2D, and on 9 October CM2D's spectral export. E1 ran on a Xeon
Platinum 8470Q host with 25 CPUs allotted (`os.cpu_count()` 208: E1's worker
set-up gave each of three workers 69 torch threads); the instance now runs on
a Xeon Gold 6459C host with 16 CPUs allotted (`os.cpu_count()` 128: 42
threads per worker) and a 92 GiB memory limit; the GPU, its driver and torch
are E1's (the adjudication records a stack or GPU other than E1's as a
deviation; the returns' provenance files record the hostname, the CPUs and
the memory limit, and Sec. 4's machine.txt the CPU model). This stage stamps
PREREG_W9 r3 as reviewed on 2 October, with no change to src, scripts other
than one new read-only script, configurations or the pre-registration's
text, and brings the runbook up to date.

- `PREREG_W9.md`: r3 stamped on 9 October 2026 by
  `scripts/stamp_prereg_w9.py` (the six CONFIG_SHA256 lines, the status line
  and the file's own SHA-256; the configurations are the generator's output,
  unchanged since Stage 0c). CONFIG_SHA256: `w9_c1_n1024` `acccf57e...`,
  `w9_c1_n4096` `0bae42a4...`, `w9_c1_n25600` `91e6ecd7...`,
  `w9_c1_n12800` `15a14602...`, `w9_b_n1024` `49574ab0...`, `w9_s_n1024`
  `b3cbf33c...`; PREREG_W9_SHA256 `0f61e286...` (full values in the file).
- `RUNBOOK_W9.md`:
  - Sec. 0b: the direct fetch in one line, with explicit refspecs for the
    branch and the tag (on 8 October AutoDL's network route returned HTTP 503
    while the direct fetch worked), and a bundle route for the same refs
    (later that day the direct fetch timed out and CM2D's operator fetched
    from a bundle); a fetch that prints nothing for 10 min, or has not ended
    after 20, is abandoned for the bundle; the repo-side check of the remote
    branch and tag before the operator instruction is sent.
  - Sec. 0c: 5 GB free on / (the session-2 plan's tests check the disk of
    their temporary directory); the suite count.
  - Sec. 2a: the plan's memory reading X must be the container's limit
    (`/sys/fs/cgroup/memory.max`, now printed in 2a) over 1e9, with the
    source `cgroup_limit`: here `98.8 GB (cgroup_limit)` (98,784,247,808
    bytes, 92 GiB); otherwise STOP; another limit is reported as a change of
    host. The earlier rule (X larger than the instance's memory in the AutoDL
    console) could have stopped a healthy session 2 if the console gives the
    limit in GiB.
  - Sec. 4: the paper's timing and field export ran as RUNBOOK_CMAME Sec. A;
    Sec. 4 now times the paper's CPU solvers (direct and CG) in the box's
    environment and with one thread, in 2D (E1, 8 validation instances) and
    3D (Phase-2b, 8 validation and 2 fine instances; 3D once more in the
    environment after the one-thread run, so that drift can be told from a
    thread effect), and reads the thread pools of the BLAS libraries numpy
    and SciPy's SuperLU load (`scripts/blas_threads.py`, new): how many
    threads the cost table's solvers used was not recorded (threadpoolctl is
    not installed on the box). About 25 min, after 2d, read only; nothing in
    PREREG_W9 reads it.
  - Sec. 5 withdrawn (it needed AutoDL's network route and a 3-4 GB
    download, and nothing reads it).
  - The header: Stage 0f, Sec. 4's time, both hosts.
- `scripts/blas_threads.py` (new, read only): per loaded BLAS or OpenMP
  library, its thread count from the library's own query (as threadpoolctl
  reads it), with the environment's thread variables, the CPU affinity and
  torch's threads (read after the pools).
- `tests/conftest.py`: every test starts with SIGPIPE ignored (found on
  cmame-paper at its Stage 3: gmsh.initialize() restores the signal's
  default action in the test process, and a later test that stops a worker
  pool could end the whole run without a summary line; this suite also
  meshes with gmsh in process and stops pools in its tests).
- `tests/test_w9_runbook.py` (6 tests): every script the runbook names exists
  and accepts its options; the fetch and bundle blocks line by line; Sec. 4's
  seven steps and its precheck; the timing script with Sec. 4's 3D options
  on a Phase-2b-shaped CPU run, in the test's environment and with one
  thread (only the direct solve and CG, both complete, the threads
  recorded); the BLAS readout in both settings; the suite count in Sec. 0c.
- Checks: the suite (444 before the readout was added, 445 after) in a
  fresh clone, in the sandbox's environment and under the box's Python,
  numpy, scipy and torch versions, with the stamp committed and tagged in a
  scratch clone; Sec. 0b's bundle route and 0c's git checks run as pasted on
  a single-branch clone of cmame-paper at `8b5d443` (the box's checkout),
  from a bundle made as the runbook says: `FETCH-OK`, both lines at the
  commit, `describe` printing `prereg-w9`, a clean status.
- An independent review, in clones of its own: src, scripts (but the new
  one) and configurations unchanged since 414a372; the stamp reproduced byte
  for byte from r3's draft, every CONFIG_SHA256 and the self-hash verified;
  nothing in r3 contradicted by current facts. One blocking finding, the
  memory STOP rule of Sec. 2a above; the BLAS readout, the drift check, the
  host details, the fetch cap and the clarifications above; all addressed. A
  second round found nothing blocking and tied the memory rule to the
  container's own limit, so that another change of host does not stop a
  healthy session 2. A later note: cmame-paper's
  `tests/test_cmame_runbook.py` pins this runbook's former Sec. 4
  (`test_section_a_is_wp9s_section_4_without_f5`) and must change if the two
  branches are ever merged.
- Suite 445 (on the box 444 passed and 1 skipped where pdflatex is missing).

## Stage 1 (11 Oct 2026) -- the returns committed; PREREG_W9 adjudicated: H1 NOT SUPPORTED, H2 SUPPORTED

The box ran session 1 (returned on 10 October) and session 2 and Sec. 4
(10-11 October) at the stamped commit `92dfc93` (`prereg-w9`). This stage
commits the three returns, the adjudication and the tables generated from
it. No change to `src`, the configurations, the pre-registration or the
adjudicating code.

- **The returns** (`records/wp9/session1/`, `session2/`, `threads/`), as
  received, with their tarballs' SHA-256 in `records/wp9/README.md`. Session
  2 ran the five planned arms in the plan's order (the 1,024 arm, evaluation
  only; 25,600; the fresh baseline; S; 4,096), each once, with exit 0, no
  restart, solve ledgers of 0 and no out-of-memory kill.
- **The verdict** (`records/wp9/w9_verdict.json`): RUNBOOK_W9 Sec. 3 on the
  stamped code, in a clean checkout of `prereg-w9` with the returns in place
  (`adjudicator.git` = `prereg-w9`); the two optional paths are given
  relative to the repository, so that the file holds no machine path. The
  decisions recomputed from session 1's readings were identical; no report
  was refused; the adjudication recorded no deviation.
  - **H1 NOT SUPPORTED.** The 25,600 arm's in-band relative energy gap (the
    validation split and IB together) against E1's states: -2.5%, tau 10%.
    Session 1's rule 4 had read "near parity". Against the fresh baseline,
    trained on the same host and stack, the 25,600 arm reads +1.1%
    (validation split) and +0.8% (IB).
  - **H2 SUPPORTED.** S against the fresh baseline: (i) F5 displacement
    error -76.0%; (ii) the F5 / in-band ratio -74.1% (4.67 to 1.21); (iii)
    in band, energy gap -1.9% and displacement error -7.3%, neither worse;
    tau 10% throughout. The fresh baseline showed the deficit rule 1
    admitted S on; S's median c* on F5 was 0.986 against the baseline's
    1.28.
- **Exploratory readings** (Sec. 7, no verdict; the tables in
  `paper/wp9/tables.md`, the counts and the level and cosine readings below
  from the reports' per-instance arrays):
  - R: S's displacement error grew from the coarser mesh sizes to h 0.025
    by a factor of 1.10-1.35 (seed means, one per coarser size) against the
    baseline's 4.40-5.62.
  - F2 (slender plates) is the hardest set for every arm: each arm has a
    seed in which more than a quarter of the 256 instances have a relative
    energy gap above 1 (worse than the zero field), and S has the most
    (171, 220 and 194 by seed). S's mean energy gap and displacement error
    on F2 are +267.8% and +243.0% against the baseline, within the guard
    only because its tau is 407% and 486%; S's displacement error there is
    3.10 in seed 3 against 0.35 and 0.40 in the others.
  - Level and shape on F2. c* is the solution's energy norm over the
    prediction's times the cosine between the two in the energy inner
    product, so a small c* can mean a large level or a poor shape. S's
    predicted energy norm on F2 is 1.48, 3.25 and 1.74 times the solution's
    by seed (medians over the load cases), against the baseline's 1.04, 1.14
    and 0.99, and its cosine 0.53, 0.12 and 0.54 against 0.75, 0.68 and
    0.86; S's median c* there is 0.031-0.296 by seed against the baseline's
    0.592-0.847. S raises the predicted level on F2 as on F5, where it lifts
    the baseline's 0.74-0.77 of the solution's energy norm to 0.99-1.00; on
    slender plates it overshoots, and its shape is worse as well. Scaling by
    c* removes the level and leaves the shape: after it, S's errors on F2
    are worse than the baseline's beyond the guard (energy gap +33.2%,
    displacement error +29.1%), the only crossings against S, while the
    25,600 arm's are lower than the baseline's (-30.3%, -28.9%). The new
    table `table_w9_s_amp.tex` gives c* and the changes after it on every
    set.
  - Outliers: outside F2, every instance of the fresh baseline worse than
    the zero field is in its third seed, seed 5 (14; the largest relative
    energy gaps 1,907 on F1, 901 on F3, 454 on F4 and 886 on F5). They make
    the -91% to -99% changes of S's and the 25,600 arm's mean energy gaps on
    F1, F3, F4 and F5 against it. E1's states have 34 such instance-seed
    values outside F2 (25 on F3); S and the 4,096 and 25,600 arms have none.
    No value is non-finite and no seed is flagged as diverged.
- **`DEVIATIONS_W9.md`** (new): the departures the adjudication cannot see.
  W1: session 2 ran on a third host with NVIDIA driver 580.95.05 instead of
  E1's and session 1's 595.71.05 (the reports do not record the driver),
  without the operator instruction's report of the new host. E1's states
  gave the same values on both hosts; the fresh baseline is within the
  guard of E1's states on every in-band reading (2 of 43 comparisons on all
  sets cross it); on one stack, too, the larger pool shows no reduction. W2:
  the instance was shut down between the sessions (RUNBOOK_W9 1f says not
  to; PREREG_W9 allows two visits). W3: Sec. 4 ran with 25 threads instead
  of the cost table's 16.
- For later runbooks: the session-2 host check of RUNBOOK_W9 Sec. 2a reads
  the container's memory limit, which the new host shared, so it could not
  show this change; a check that is to catch one needs the driver, the CPU
  model or the host name. And a correction to Stage 0f: the returns'
  provenance files were to record the memory limit, but Sec. 2d's block
  prints the CPUs and the memory events, not the limit, and session 1's
  (1f) was written without a GPU; W1 says what the records do show.
- `scripts/make_w9_paper_material.py`: a table of S against the fresh
  baseline with c* (`table_w9_s_amp.tex`, whose note states what c*
  measures; `tests/test_w9_session2.py` expects it); the displacement
  table's note no longer speaks of the tail, which only the energy-gap
  table has; the Markdown names the hand-kept ledger beside the
  adjudication's deviations. Its outputs, generated from the verdict, are
  committed in `paper/wp9/` beside `scale_lemma.tex`.
- `RUNBOOK_W9.md`: where the sessions and Sec. 4 ran. Sec. 0c keeps the
  stamped commit's count, which the box ran; `tests/test_w9_runbook.py` now
  checks it against Stage 0f's count, and the notes' latest count against
  the suite.
- `records/wp9/README.md`: the returns, the verdict and how to re-check it,
  the evaluation sets on the box against the sandbox (the F2 manifest
  identical; the other six differ while every set's node-count range and
  median agree), and Sec. 4's readings.
- `tests/test_w9_returns.py` (10 tests): the returns whole and consistent
  with each other (status, provenance hashes, the plan's decision file and
  manifests); Sec. 4's return; the adjudication re-run on the committed
  returns equals the committed verdict but for the adjudicator's record of
  its own code, which is checked against the stamp's files; H1 and H2, and
  the c* readings behind the deficit check, recomputed from the reports
  without the adjudicator's code; E1's states' values on both hosts; the
  verdict words and the numbers that the ledger, the README and these notes
  quote, checked phrase by phrase with the labels they carry; the paper
  material byte for byte.
- Three independent reviews, two of the first package (the verdict and the
  records; the package's mechanics) and one of the revised package: nothing
  blocking. Addressed: the F2 readings (level, shape and the raw changes),
  the outliers and the c* table above; the training comparisons off the
  band, the sources and inferences of the host facts, and the shutdown
  between the sessions (W1, W2); the claims about the verdicts worded as
  what the records show; tests that tie each quoted fact and verdict word
  to its record, compare bytes, check lengths and exact equality, and
  survive later changes to the adjudicating code; the runbook's count.
  Not changed: the returns keep the box's file names and trailing
  whitespace, as received.
- Suite 455 (445 + 10).

## Stage 2 (15 October 2026) -- PREREG_FINE3D r1 drafted, not stamped: S on the token bottleneck (R2a) and label-free fine-tuning at the fine mesh (R3)

The PI chose on 14 October 2026 to run R1 (cmame-paper, PREREG_HOLDOUT), R2a
and R3 in one box visit, R3 at the budget of 16 only, and to stamp when the
run is about to start. This stage writes everything R2a and R3 need; nothing
is stamped and nothing has run on the box.

- **PREREG_FINE3D.md** (r1 draft; its three CONFIG_SHA256 lines open). R2a:
  E2's M = 1,024 bottleneck trained again on fresh seeds 3-5 (the reference)
  against the same with S; SUPPORTED iff the fine displacement error and the
  fine / in-band ratio are lower beyond the E-series guard and K1 in band
  holds (W9's H2, in 3D). The R2b rule, fixed before R2a runs: the
  transformer with S is recommended iff R2a is SUPPORTED and S's fine
  displacement error is at most half the fresh baseline's. R3: the Phase-2b
  AR states fine-tuned with the AR objective on the 16 fine instances on
  which P3's labelled fine-tuning trained, with its schedule and seeds; R3-E
  and R3-V: the fine energy gap and von Mises error lower than the labelled
  fine-tuning's beyond the guard. That reference is the recorded run (the
  Phase-2b report), not trained again: the rules' properties are given with
  the reference held at its record. Properties of all three rules by
  simulation (`records/wp9/fine3d/simulations.json`).
- **S on the bottleneck** (`src/fejepa/models/bottleneck.py`):
  `decode_scale`, `decode_scale_factor` and `features.load_density` with the
  transformer's validation and values, now shared in
  `fejepa.models.features` (`check_decode_scale`, `decode_scale_value`);
  `FEJEPAConfig` and the transformer's pack call the same two functions (same
  messages, same values). The runner accepts S for the kinds `fejepa` and
  `bottleneck` (`runner.S_MODEL_KINDS`); the MGN baseline still refuses it.
- **P3 takes E8's seed offset** (`p3_transfer.run_p3`, the runner passes it
  only when set): the zero-shot rows and few-shot units read
  `ar_p<pool>_s<offset + i>` and record those seeds. The refusal of an offset
  with P3 is gone; an offset written into P3's own block is refused. The
  zero-shot pass prints a line every 32 instances (`[p3] zero-shot ...`): in
  E2's run on the box it went about an hour without output, which the
  runbook's health check would have read as a stalled session. Prints only.
- **The 3D factor** (`scripts/fine3d_scale_factor.py`,
  `records/wp9/fine3d/scale_factor_3d.json`): 2^-10, the power of two nearest
  the median 0.00113 of max|F| / sum|F| over 256 fresh training-family
  instances (seed 93000, so that E2's corpus and R1's 720-instance holdout
  are not read). The remesh medians equal the box's in wp8's post-hoc
  records: the sandbox's gmsh meshes these geometries as the box's.
- **Configurations** (`scripts/make_fine3d_configs.py`): `r2a_b_m1024`,
  `r2a_s_m1024` from E2's stamped file (refused unless its SHA-256 is
  PREREG_E2's line), `r3_eft_b16` from the Phase-2b report (refused unless its
  SHA-256 is the one PREREG_E2 records).
- **R3's script** (`scripts/r3_energy_finetune.py`): verifies the guard, the
  source report, the states and both corpora; a content check of each state
  against the report's arrays (relative 1e-3) before it trains; drops the
  training instances' labels on load; trains with `pretrain()` and
  `AR_CONFIG`; evaluates both sets with the amplitude readings; records per
  seed the seed it trained with, the epochs run and the steps (counted from
  the trainer's history), the SHA-256 of the instance orders its seed
  determines, the code (`git describe`) and the configuration's SHA-256;
  skips a finished seed of the same code and configuration, moves any other
  result aside and runs that seed again, resumes an interrupted seed from a
  checkpoint the same code and configuration wrote (its sidecar file says
  which; any other is moved aside and the seed trains from the start);
  `--dry-run` checks and stops.
- **The session** (`scripts/fine3d_session.sh`, `RUNBOOK_FINE3D.md`): the two
  R2a arms, the post-hoc amplitude pass on each (`--remesh 16`), then R3; a
  finished step is skipped, an interrupted R2a arm restarts with
  `--reuse-states`; before every step the checkout must still be the tag
  alone and unmodified (`git describe --always --dirty --tags`, the string
  every report records, prints exactly the tag, and no other tag is on HEAD).
- **Stamping** (`scripts/stamp_prereg_fine3d.py`): refuses anything but the
  approved draft (its SHA-256 pinned in the script) and a `src` other than
  the tree the regression ran; fills the three CONFIG_SHA256 lines, the
  status line and the footer (the file's own SHA-256) in one step.
- **Adjudication** (`src/fejepa/analysis/adjudicate_fine3d.py`,
  `scripts/adjudicate_fine3d.py`): the verdicts of PREREG_FINE3D Sec. 6, the
  readings of Sec. 7, the refusals of Sec. 8; R2a and R3 independently (a
  part whose report is missing, unreadable or refused is NOT EVALUATED, the
  other is adjudicated). The verdict file records the inputs' SHA-256, the
  session's attempts, and the adjudicating code and the pre-registration it
  read against the tag's (a difference, or no tag to compare with, is a
  deviation); its "deviations" collect the session's, the code's, R2a's and
  R3's.
- **Two independent reviews before delivery** (code and statistics;
  pre-registration and operations). Changed after them: the R3 reference
  stated as a recorded run, with its rules' properties held at the record
  (the floor its own spread puts under tau, and a record above its
  expectation); R2a's sensitivity to larger or correlated seed spreads
  simulated; the R2b rule moved from the ratio to the fine error;
  condition (ii) reads P3's in-band pass and K1 both in-band metrics, as
  the text says; the amplitude record is read only when complete and of
  the same report; R3's report records the fields above and the
  adjudicator refuses a seed trained otherwise; the session's code check is
  the reports' string; the stamp pinned to the draft; the regression run
  again on the final `src`.
- **A third review, of the revision.** It found the suite failing on the
  stamped commit (a test still assumed the draft's open footer: the box's 0c
  would have stopped), the R2a arms' silent P3 pass against the health check
  and their time (about 8.5 h, not 8 h), and the adjudicator comparing the
  repository's pre-registration with the tag's instead of the one it read.
  Changed: that test; P3's progress lines; the times; the comparison, with the
  tag's absence a deviation and three more files compared; unreadable inputs
  leave only their part unevaluated; a missing status file is one deviation;
  R3 reuses a seed's result only from the same code and configuration; the
  order record described as what the seed determines; "about a 1.0- and
  1.6-standard-error event"; the runbook's tmux step after R1, FETCH-FAILED
  taking the bundle route and 0c's time, as the operator's Chinese text had
  them; tests for the rules it found unpinned.
- **A fourth review, of that revision** (no blocking finding). Changed after
  it: R3's reuse test writes the result it plants with the configuration it
  re-runs, with a control that must be reused (five of its six cases had
  passed on the configuration hash alone); a test that the run's own P3
  pass prints; R3 resumes only a checkpoint of its own code and
  configuration; the temporary repository's git configuration isolated;
  the verdict file's notes pinned; PREREG Sec. 4 cites the overlap record
  it rests on; the runbook's order of the log lines; the delivery checks
  below recorded.
- **Delivery checks** (sandbox, 15 October 2026). The package committed on
  2f35f2d: the full suite passes (pdflatex present: every test runs).
  Stamped as Sec. 8 says (`scripts/stamp_prereg_fine3d.py`, a commit, an
  annotated tag `prereg-fine3d`): `git describe --always --dirty --tags` and
  `git tag --points-at HEAD` print exactly the tag, the full suite passes on
  that commit and leaves the tree clean. A CPU rehearsal of the session (not
  committed; the box's layout with sizes scaled down): gmsh3d corpora of 24
  and 10 instances at the box's paths, a miniature Phase-2b run as R3's
  source, miniature configurations from the generator's functions with
  PREREG_FINE3D stamped over them, committed, tagged (annotated) and pushed
  to a bare repository; a single-branch clone at 92dfc93 as the box. From
  the stamped commit's runbook, verbatim: 0b (FETCH-OK, both lines the
  commit), 0c without the suite (describe and tags exact, status empty), 0d
  (both dry runs `verified`, R3's plan), 1' during the amplitude pass, 2
  (the return). The session script (`FINE3D_REHEARSAL=1`) was killed after
  `r2a_s_m1024`'s first seed state, run again (that arm restarted with
  `--reuse-states`), killed during R3's first seed, run to the end (that seed
  resumed from epoch 1) and run once more (every step skipped but R3, which
  took both seeds from disk: the miniature has two seeds, the skip rule asks
  for s0-s2). The return, unpacked, was adjudicated with the module's sizes
  set to the miniature's: every part evaluated; the attempts, the restart,
  the resumption and the seeds taken from disk recorded as deviations.
- `scripts/regress_against_branch.py`: a miniature of a non-FE-JEPA
  configuration keeps WP6 and E6 disabled (the runner refuses them for the
  bottleneck). The regression against wp8-lejepa, run again on the new src
  tree for E1's and Phase-2b's configurations and newly for E2's, is
  identical (`records/wp9/regression_wp8_*.json`, README).
- tet3d: wp9-pool keeps `structured_tet_mesh`'s five-tetrahedron split
  (cmame-paper Stage 9 and main replaced it with the Kuhn split); neither R2a
  nor R3 uses it (both read gmsh3d corpora), so the fix is not carried here.
- Tests changed on purpose: `test_w9_s_switch.py` (the bottleneck now accepts
  S; the shared validation is checked on both configs) and
  `test_w9_runner_eval.py` (the refusal now names the MGN kind; the P3-block
  offset refused); `test_w9_records.py` reads the third regression record.
- New tests (124): `test_fine3d_bottleneck_s.py`, `test_fine3d_p3_offset.py`,
  `test_fine3d_r3.py`, `test_fine3d_configs.py`, `test_fine3d_adjudication.py`,
  `test_fine3d_records.py`, `test_fine3d_prereg.py`, `test_fine3d_runbook.py`.
- Suite 579 (455 + 124).

## Stage 2, r2 (15 October 2026) -- PREREG_FINE3D r2 drafted, not stamped: R3 against its labelled fine-tuning run again in the same session; the R2b line at -40%

The PI decided on 15 October 2026 that R3's reference is to be run again in
the same session, and asked for the R2b line to be evaluated again. Nothing
is stamped and nothing has run on the box.

- **R3's labelled arm** (`configs/r3_lft_b16.json`): P3's labelled few-shot
  fine-tuning at 16 run again from the same states, on the same 16
  instances, with the same schedule, seeds and instance orders, in the same
  session as the energy arm and after it (the session starts it only once
  the energy arm's report is complete: R3 needs both). It replaces the
  Phase-2b record as the reference of R3-E and R3-V; the record is read
  beside it (every metric with the guard; a difference beyond the guard is a
  flag), and the energy arm likewise. The two configurations differ only in
  `_comment`, `out` and `train.loss`.
- **R3's script**, renamed `scripts/r3_finetune.py` (it was
  `scripts/r3_energy_finetune.py`): one arm per configuration, chosen by
  `train.loss` -- `ar`: `pretrain()` with `AR_CONFIG`, the labels dropped on
  load; `labels`: `train_supervised()` with `anchor_mode = "none"`, as P3's
  unit runs it, every training instance required to carry its labels. It
  also checks the weight decay and the clip against `SupervisedConfig`'s
  defaults and refuses a source report whose runtime is not fp32 without
  compilation (P3's units took both from it); each result records its loss,
  trainer and machine (host, GPU driver); the files are `eft_*` and `lft_*`.
  On the CPU miniature the labelled arm reproduces P3's labelled unit per
  instance, bit for bit, and its trainer configuration field by field.
- **`train_supervised`** records the last step's loss of each epoch
  (`history`, kept in its epoch checkpoint, so that a resumed run counts
  every epoch). Nothing else changes: the regression against wp8-lejepa,
  run again on the new `src` tree, is identical (records/wp9/README.md).
- **Adjudication**: R3 reads both reports (`--r3`, `--r3-labelled`); each is
  refused as before, and for the other arm's configuration, objective or
  trainer; the two are refused together if they ran on different code or
  recorded different epoch orders. Another torch, GPU or TF32 setting
  between them, or seeds on more than one host or GPU driver, is a
  deviation; another setting between the labelled arm and the record, a
  note. A non-finite value of the labelled arm voids the verdict whose
  metric it is (the other verdict, the readings and the deviations stand);
  one of the energy arm fails its verdict (diverged). A report not given or
  not in the return is named; a status file not given is a deviation.
- **The R2b line** at -40% (it was -50%). `scripts/fine3d_sims.py`'s new
  `r2b` block simulates R2b itself, R2a's rule on the Phase-2b transformer's
  spreads, with S's reduction on the transformer taken as 0.87 times the
  bottleneck's (the ratio of the two c* reductions; 0.6 and 1.0 as
  alternatives) and the bottleneck's true reduction uniform on 0-80%. An
  R2a observed near -40% is followed by a supported R2b in 60% of trials
  (50% near -35%, the point where the prediction passes one half; 79% near
  -50%); the recommendations at -40% are followed by a supported R2b in 83%
  of cases (70% on R2a SUPPORTED alone, 87% at -50%), and 17% of trials go
  unrecommended though R2b would be supported (9% and 25%).
  `r2b_sensitivity`: 60% with the prior on 0-60%, but 21% with the
  transformer's spreads doubled, so R2b's own pre-registration should weigh
  its number of seeds. `r3_paired` correlates the two R3 arms within a seed;
  `r3_spread_x2` doubles both arms' spreads.
- **PREREG_FINE3D r2**: Q-R3; Sec. 3 (two arms, why their order cannot
  favour either), 4, 5, 6 (R3's reference and its properties, unpaired,
  paired and with doubled spreads, and why the record is not the reference;
  the R2b rule with its reasons and its sensitivity), 7, 8, 9 (about 39 h)
  and the revision line; four CONFIG_SHA256 lines. The stamp script expects
  the r2 draft (its SHA-256 pinned again).
- **The session and RUNBOOK_FINE3D**: six steps, the two R3 arms last
  (energy, then labelled once the energy arm is complete); 0d dry-runs both
  R3 configurations; about 39 h, the balance about 50 h; the return carries
  `r3_lft_b16`; the adjudication command reads both arms.
- **Two independent reviews before delivery** (code and statistics;
  pre-registration and operations), no blocking finding. Changed after them:
  the R2b rationale's wording (the prediction passes one half near -35%, the
  line is the first grid point clearly above it; "if judged by R2a's rule";
  the spreads compared metric by metric; the window and the prior's bound
  stated) and its sensitivity; R3's properties with both arms' spreads
  doubled and a weak-evidence sentence; the standard-error convention of the
  record-held rates; a void verdict per metric, keeping the rest; the
  machine record and its deviation; the labelled arm only after a complete
  energy arm; the source report's runtime checked; Sec. 8's R3 refusals as
  the code has them ("recorded different epoch orders", another corpus);
  tests for the non-finite paths, the labelled arm's trainer configuration,
  its reuse, the session's step list as the adjudicator counts it, a status
  file not given, and the session's energy-first rule.
- **A third review, of the revision** (no blocking finding). Changed after
  it: the doubled-spread case of the R2b rationale backed by its gates (R2b
  supported after 49% of the recommendations at -40%, not "seldom"); each R3
  seed's machine recorded when it starts and when it ends, the driver query
  tried twice; a paired reading with a non-finite value is null and the
  verdict file is strict JSON (non-finite values written as null); R2a's
  void keeps its readings and deviations, as R3's does; the revision line
  lists the r2 changes; tests for an energy arm's report with two seeds and
  for the strict JSON.
- **Delivery checks** (sandbox, 15 October 2026). The package committed on
  1060d3c (the old script removed): the full suite passes (pdflatex present:
  every test runs). Stamped as Sec. 8 says (`scripts/stamp_prereg_fine3d.py`,
  a commit, an annotated tag `prereg-fine3d`): `git describe --always
  --dirty --tags` and `git tag --points-at HEAD` print exactly the tag, the
  full suite passes on that commit and leaves the tree clean. A CPU
  rehearsal of the session as in Stage 2 (the box's layout, sizes scaled
  down; a miniature Phase-2b run with three AR seeds as R3's source;
  miniature configurations from the generator's functions with
  PREREG_FINE3D stamped over them, committed, tagged and pushed to a bare
  repository; a single-branch clone at 92dfc93 as the box). From the stamped
  commit's runbook, verbatim: 0b, 0c without the suite, 0d (both R2a dry runs
  `verified`; R3's two plans, the energy arm's then the labelled arm's, each
  with its own line's hash), 1' during the amplitude pass, 2 (the return).
  The session script was killed after `r2a_s_m1024`'s first seed state,
  during the energy arm's first seed and during the labelled arm's first
  seed, then run to the end (that arm restarted with `--reuse-states`, each
  R3 seed resumed from epoch 1) and once more (every step skipped). The
  return, unpacked, was adjudicated with the module's sizes set to the
  miniature's: every part evaluated; the labelled arm, interrupted and
  resumed, equal to the miniature record of the same fine-tuning in every
  metric (rel 0); the attempts, the restart and both resumptions recorded as
  deviations, no note.
- Tests changed on purpose: `test_fine3d_r3.py` (both arms; the labelled arm
  against P3's labelled unit; the old script name gone),
  `test_fine3d_adjudication.py` (two R3 reports, their refusals and pair
  checks; the reference is the labelled arm, not the record; the R2b line at
  -40%), `test_fine3d_prereg.py` (r2's numbers, the R2b rationale),
  `test_fine3d_configs.py` (four configurations; the two arms differ only in
  their objective), `test_fine3d_runbook.py` (six steps; the energy-first
  rule).
- Suite 625 (579 + 46).

## Stage 2, r3 (15 October 2026) -- PREREG_FINE3D r3 drafted, not stamped: the code checked in full before the stamp; the R2b line at -45%

After r2 was pushed (7eaf4f6) the PI asked for the R2b line to be evaluated
again and for the whole code to be checked again. Nothing is stamped and
nothing has run on the box.

- **Three independent reviews of everything the session runs or the
  adjudication reads**, on the pushed commit: R2a's path (configurations,
  runner, bottleneck with S, P3, the post-hoc pass, the adjudicator's R2a
  half); R3's path (the script, the trainers, checkpoints, evaluation, the
  adjudicator's R3 half); the runbook, session script, stamp, adjudication
  script, simulations and tests (the stamp, a box-like single-branch flow and
  the suite simulated). Found and changed:
  - R3's content check let a NaN prediction through (`max(0.0, nan)` is 0.0
    in Python): any non-finite prediction or deviation now fails it.
  - A resumed R3 seed recorded the machine it resumed on as the one it
    started on: the start machine is kept in the checkpoint's sidecar, and the
    result records the start, every resumption and the end.
  - An amplitude record with an incomplete pass was read: a record is read
    only with every seed's in-band and fine passes complete and their
    reproduction checked, its remesh readings only with all 16 geometries and
    none failed; a reproduction beyond 5% is a deviation; the remesh field
    `fscale` is reported as what it is (max|F|, for S as for the baseline).
  - A P3 pass on a model built otherwise than E8's (without S, say) would
    have gone unnoticed: the adjudication refuses an R2a report whose P3
    in-band pass does not reproduce its E8 AR cell (seed means to 1e-3; the
    box's E2 runs: 3e-5); a test runs the exact R2a path (the generator's S
    configuration through run-config on gmsh3d miniature corpora) and the
    amplitude pass on its report; a test hooks the bottleneck's decoder and
    checks that S's scale is applied once.
  - Reports whose configuration does not hash to their recorded SHA-256 are
    refused (R2a and R3); R2a's restart deviation names the states reused.
  - R3 records each seed's TF32 setting during training (another than the
    source report's is a deviation), reads `git describe` once per run and
    stops if it cannot, and writes a seed's result before it removes the
    checkpoint.
  - The session runs the amplitude pass again when its record was measured on
    another report of the arm. RUNBOOK 0c checks that no fine3d run is on the
    box yet and names the checkout's and the suite's normal output in full
    (the operator would otherwise have stopped at "set up to track").
  - PREREG Sec. 2: 57 of the 64 remesh meshes have the box's node counts and
    the other 7 differ by 1 to 21 nodes (it said the sandbox meshed them as
    the box did); a test pins it.
  - Tests for the energy arm's trainer configuration field by field and its
    unlabelled instances, the R2a rule by hand with seed-specific values, the
    amplitude record rules, the per-seed TF32 setting and resumed machines,
    and the second-copy lock without a fixed wait; mutation checks confirm
    that each catches its fault.
- **Two independent reviews of that revision** (its code; its statistics and
  documents), found and changed:
  - The new P3 / E8 refusal turned a seed that diverged in training (NaN in
    E8's cell and in both P3 passes) into a refusal of the whole report,
    against Sec. 6's rules for divergence: a seed mean finite in neither is
    left to Sec. 6, one finite in one of the two only fails; tests diverge a
    seed in each arm, and test the rule's energy-gap half.
  - A failed remesh geometry discarded the whole amplitude record, which the
    session counts as complete and does not run again: the in-band and fine
    readings are read, the remesh readings are not evaluated.
  - R3's report read `git describe` again at the end of the arm (a timeout
    there would have refused the arm): it records the one read before the
    first seed.
  - Sec. 8 states the refusal of an R3 seed of another configuration than its
    report (r2's code); RUNBOOK 0c's new check ignores `runs/fine3d/session`,
    which 0c itself makes, so a repeated 0c passes; stale docstrings.
  - The reasons given for keeping the R2b line at -40% were selective (the
    R2b item below).
- **Two more independent reviews, of those fixes** (code; statistics and
  documents), found and changed: the refusal of an R3 seed of another
  configuration than its report had no test, nor the P3 / E8 rule's other
  direction (E8's value not finite) or a `session_old` entry in 0c's check;
  the R2b rule compares with a tolerance of 1e-12, so that S's fine error at
  exactly 55% of the baseline's (the float rel -0.44999999999999996) is
  recommended as Sec. 6 says, and prints rel to 0.01%; Sec. 6 names the
  refusal of a seed mean non-finite in only one of P3's in-band pass and E8's
  cell (Sec. 8); the R2b line's dependence on the model of the spreads'
  uncertainty is recorded and stated, and its test reads the decision's
  assumptions from the record (eleven phrases of the paragraph could change
  without failing it); wording.
- **The R2b line at -45% instead of -40%.** R2b's cost corrected: with a
  fresh baseline as in R2a it is six transformer runs, about a week of one
  GPU (the text said three seeds). The line is set as a decision: recommend
  where R2b's predicted support passes one half, the line of the largest
  expected value with a supported R2b worth twice its cost (`r2b_line`, new).
  With the transformer's spreads at their three-seed estimates it falls near
  -35%; three seeds estimate a spread poorly and the support depends strongly
  on it, and with each spread drawn in every trial from its uncertainty
  (sigma^2 = 2 s^2 / chi2_2, at most 4 times the estimate) it falls at -45%:
  from -44% to -46% uncapped or capped at 3 times, with the ratio of the
  effects uncertain around 0.87, the bottleneck's spreads drawn too or the
  other priors; at -47% with an in-band effect of S and -53% with a lower
  ratio; from -35% to -42% under milder models of the spreads' uncertainty
  (a cap at 2 times, one draw for the three metrics, a prior worth 2 or 4
  more degrees of freedom, the prior 1/sigma^2). Over these 16 three-seed
  cases -45% loses at most 8% of a case's best expected value, -40% up to
  25% and -50% up to 16%: the smallest largest loss of the lines simulated.
  The expected value is flat near its peak (in the main case -40% and -50%
  reach about 97% of it); with a supported R2b worth three times its cost the
  line would be -35%. Five seeds per arm raise the supported share of the
  recommendations but lower the expected value at every line, with a
  supported R2b worth 1.5 to 3 times the cost of six runs. r3's first draft
  kept -40% as lying between the line near -35% and the one with the ratio
  uniform on 0.5-1.0 (whose mean is 0.75): both held the spreads at their
  estimates.
- A final independent check of those changes found the uncapped case's line
  rounded twice (its crossing -45.45% kept as -45.5, quoted as -46%): the
  record keeps the crossings to 0.001 and the text says -45%. Sec. 6 states
  the R2b rule's tolerance of 1e-12; the test reads whole sentences of the
  R2b paragraph and checks Sec. 6's new R2a clause; an incomplete remesh pass
  is named with its row count.
- The simulation record: `r2b_line` new (18 cases with their parameters, and
  each line's largest loss over the three-seed cases); `r2b_sensitivity`
  gains the ratio drawn uniform on 0.5-1.0, the spreads 1.5 times, R2b on
  five seeds (with the spreads as recorded and 1.5 times) and two other
  priors; the blocks
  before r3 are unchanged but for the R2a blocks' `p_r2b_recommended` (the
  line) and the record's `what` and `r2b_fine_rel_max`.
- Tests changed on purpose: `test_fine3d_adjudication.py` (condition (ii)
  and K1 read through the verdict itself, since the adjudication now refuses
  a P3 in-band pass that differs from E8's cell; the amplitude helper writes
  complete records; the R2b line at -45%, its inclusive edge fed to the
  rule directly), `test_fine3d_p3_offset.py` (a record without its remesh
  pass is read without its remesh readings), `test_fine3d_prereg.py` (the
  R2b paragraph, its decision and its assumptions read from the record),
  `test_fine3d_runbook.py` (the stub records the report it measured).
- Suite 652 (625 + 27).

## Stage 2, r4 (16 October 2026) -- PREREG_FINE3D r4 drafted, not stamped: every requirement checked against the code; the new GPU paths measured at full size; R1 then fine3d on one checkout

After r3 was pushed (bffc66d) the PI asked for the code to be checked again
for whether everything it must do is implemented. Nothing is stamped and
nothing has run on the box.

- **Every requirement traced** (an independent audit): each statement of
  PREREG_FINE3D Sec. 2-9 and RUNBOOK_FINE3D mapped to the code that
  implements it and to the test or run that exercises it; on a stamped and
  tagged scratch copy, a return built from the records (R2a reports from
  E2's, amplitude records from wp8's box record measured again on them, R3
  reports as scripts/r3_finetune.py writes them) packed by RUNBOOK Sec. 2 and
  adjudicated by Sec. 3's command, word for word, with faults injected one at
  a time. Nothing was missing; changed:
  - An amplitude record whose remesh pass is incomplete lost its remesh
    readings silently: now a deviation. The remesh pass on the bottleneck
    with S is tested (one geometry, the miniature's S run).
  - R2a's report reads `git describe` when it is written, after about 8.5 h,
    with a 5 s timeout, and a report recorded as unavailable would be
    refused: `git describe` is tried three times, 30 s each and 2 s apart,
    for every report and record.
  - A seed that diverged gave its amplitude record a deviation of inf (NaN
    in both the record and the report): instances non-finite in both are not
    compared, and are counted.
  - R2a's uninformative note and its flags against E2's seeds 0-2 were in
    R2a's section of the verdict file but not in its notes: listed there,
    with R3's. An amplitude record given but missing from the return is named
    so.
  - A step whose last attempt did not exit 0, or has no exit line, is a
    deviation.
  - Tests for the readings compared, the script's check of the Phase-2b
    record, the post-hoc model's refusal of a configuration that does not
    hash to its record, and the instances each experiment opens: R2a none of
    the pool files after its pool (the tail R1's holdout is drawn from), R3
    only its training and evaluation instances.
- **The new GPU paths at full size** (an independent measurement on CPU, the
  real configurations, instances of 6,860 to 70,674 nodes from the project's
  generator): the bottleneck with S, R3's energy arm and the amplitude pass
  with S cost the same memory and step time as the paths the box already ran
  (the bottleneck without S, P3's labelled fine-tuning, the amplitude pass
  without S): ratios 1.00 in memory, 0.97-1.09 in time. Projected from wp8's
  bench of the same step on the box, R3's energy step needs about 8 GiB at
  the fine set's largest instance (71,749 nodes) on the 32 GB card, and 800
  steps take about 3.0 h per seed and arm (5-95%: 2.3-3.8 h for a draw of 16
  fine instances; PREREG Sec. 9: 3.3 h). S adds about 11-15 min of instance
  preparation to its arm and about 3 min to its amplitude pass. On the CPU
  the transformer's evaluation builds the full attention matrix (the GPU's
  fused kernel does not), so R3 cannot evaluate at full size on the CPU.
- **The session checks the GPU before every step it runs** (at most 300 s;
  after its skip decisions, before an R2a arm says it restarts): a GPU lost
  during the session would have let the next step start on the CPU (R2a for
  days, R3 out of memory at its first fine instance). The step is not
  started, `STOP before <step>: no GPU` is in the status file, and RUNBOOK
  Sec. 1c says what to do.
- **R1, then fine3d, on one checkout**: a clone of cmame-paper after R1's
  suite (683 passed, 3 deselected) and with outputs under runs/holdout ran
  RUNBOOK 0b and 0c of the stamped wp9-pool word for word (r3's code; r4
  changes neither block but for the suite's count): the checkout switched
  cleanly (no tracked or unignored file left by R1), `git describe` and the
  tag read prereg-fine3d, no fine3d run was found, and the suite passed
  there. R1 writes its labels under runs/holdout, not into the
  corpora, so the manifests R2a and R3 check are unchanged.
- **An independent review of these changes**, with mutation checks and the
  adjudication of built returns: nothing blocking; changed: the GPU check
  moved before an R2a arm's "restarting from its states" line, with a 300 s
  limit; Sec. 8 says that a step without the GPU is not started (the session
  goes on to check the next); `git describe`'s tries are 2 s apart; tests
  for a finished arm with the GPU lost, a failed attempt followed by one that
  exits 0, no amplitude record given and a `git describe` without output;
  docstrings and the runbook's wording.
- **Delivery checks of r3 and r4** (r3's entry did not record them): the
  package committed on the pushed base, the suite with and without
  pdflatex, the stamp, an annotated tag, the suite on the stamped commit, and
  the CPU rehearsal of the session (three interruptions, the restarts, the
  return, the adjudication). RUNBOOK's header names this entry.
- Suite 660 (652 + 8).

## Stage 2, r5 (16 October 2026) -- PREREG_FINE3D r5 drafted, not stamped: five more independent checks; Ctrl-C, the GPU check, the session's checks, the remesh pass and checkpoint restores made safe

After r4 was pushed (b7dcd6e) the PI asked for the code to be checked again.
Nothing is stamped and nothing has run on the box. Five independent checks,
each from an angle the earlier rounds had not taken, then an independent
review of the changes, after which three of the five checks were run again
on them.

- **The rules from the text alone** (review A): Sec. 6 implemented again from
  PREREG_FINE3D and the pre-registrations it cites before any code was read,
  frozen, then compared with the adjudication in 1,020,242 comparisons
  (116,380 R2a cases and 55,542 R3 cases; 1,000 of each through the real
  adjudication on reports built from the records). Every case these metrics
  can produce gives the text's verdicts; rel, SE_rel and tau agree to 1e-12.
  Changed:
  - A seed of S whose P3 in-band mean is infinite while its fine mean is
    finite gave a ratio of 0 (finite / inf), so condition (ii) was computed
    as if nothing had diverged. The verdict could not change (Sec. 8 makes
    E8's value non-finite as well, so (iii) fails), but the failing
    conditions named were wrong. The ratio is now NaN unless both values are
    finite.
  - A usable amplitude record whose median c* on the fine set is not finite
    read as "not below 1.10": the note's c* part is now not evaluated, as
    Sec. 7 says.
  - Not changed (unreachable with these metrics): the guard's denominators
    regularised at a reference mean of 0 or below; an overflow above about
    1e154 would leave a part not evaluated; the P3/E8 check against an E8
    mean of exactly 0; last-bit differences exactly at a threshold.
- **Failures only the GPU would show** (review B): every kernel the six steps
  use already ran on the box at these sizes (P3's labelled fine-tuning at 16
  covers R3's attention and memory; the energy objective adds about
  0.1 GiB); R3's content check and the P3/E8 rule run under the policy and
  code path of their references (the transformer reproduced bit for bit on
  the box, the bottleneck within 3.1e-5). Changed:
  - A CUDA error while a D9 state or an epoch checkpoint was restored (a
    failing GPU, or one taken by another process) was caught as a corrupt
    file: the file was deleted (up to an R2a unit's or an R3 seed's
    training) and the run went on from scratch. Now only a file that cannot
    be read is moved aside (`<name>.unusable.<time>`) and trained again; one
    that reads but cannot be restored stops the step and is kept. The
    session's GPU check also allocates, synchronises and needs 90% of the
    GPU's memory free (an idle card leaves about 98%), and says why it
    fails.
  - A CUDA error in the remesh pass was recorded as a failed geometry, every
    later geometry failed in turn, the pass exited 0 and the session skipped
    it from then on. An error while the states are evaluated now stops the
    pass; a geometry that cannot be meshed is still recorded and the pass
    goes on.
  - An nvidia-smi query that failed gave a driver of None, which the
    adjudication counted as another machine. A driver not read is no longer
    another machine; records without it beside records with it are a
    deviation of their own (none read anywhere, as on the CPU, is not).
  - R3's evaluations take about half an hour per seed and arm on the box
    (the metric suite runs on the CPU), not about 13 min: about 41 h in all.
  - Not changed: the amplitude record's 5% reproduction line (the concern
    that S's smaller energy gaps would trip it does not hold on E2's
    records: at in-band gaps of 0.006 to 0.05 the relative deviation is at
    most 0.5%); the steps' device stays "auto", the GPU check running
    seconds before each step.
- **The code no test executed** (review C): branch coverage with subprocesses
  over the 18 files the session and the adjudication run, 92.3% from the
  suite (pretrain.py 100%, the adjudication 98%); every unexecuted line the
  session can reach was read, and 35 tests were written for those that
  matter. pyflakes and shellcheck clean. Changed: a restore that failed
  after part of the state was loaded returned "from scratch" with the
  restored weights, optimiser or scheduler left in place (the fix above); a
  checkpoint record whose machines field is not a list stopped every rerun
  of its seed; the report encoder writes numpy booleans.
- **Interruptions at random** (review D): 420 trials of the session (1,652
  runs, 814 kills: SIGKILL, SIGTERM, SIGINT and SIGHUP to the session's
  group, the step alone, torn writes, injected crashes), then 30 trials of
  the real R3 script and 26 of run-config on miniatures, each against an
  uninterrupted run (bit for bit on CPU). The design held, except:
  - Ctrl-C did not reliably stop the session: the per-step GPU check ran
    under timeout, whose child is in its own process group and never had
    the SIGINT, so the next step started; and a check interrupted while
    Python started exits 1, which the session read as an incomplete output
    (a complete R2a report moved aside and the arm run again, a step run
    again, a false `STOP: no GPU`). The session now traps SIGINT (it stops
    once the command it interrupts has ended, with `session interrupted` in
    the status file; a minimal trap covers the lines before the lock) and runs the GPU
    check with `timeout --foreground`.
  - A kill of the session's group during that check left it holding the
    session's lock: the restart was refused for up to 300 s. The check now
    stays in the group, and the subshell it runs in closes the lock.
  - A completeness check that died on its own (the OOM killer, a failed
    fork, an I/O error) read as an incomplete output: a complete report
    could be moved aside and its arm run again, or the labelled arm left
    out as if the energy arm were incomplete. Each check now exits 0
    (complete) or 3 (not); any other exit stops that step, with
    `STOP before <step>: the check of ... failed (exit N)`, and moves or
    runs nothing.
  - R3's checkpoint record was written in place; a kill mid-write made the
    next attempt discard a valid checkpoint. It is now written atomically
    and names the seed (a checkpoint planted from another seed was
    accepted).
  - Not changed: attempts within one second share a log name (impossible on
    the box); a kill between a step's exit 0 and its exit line reads as
    interrupted (conservative); a checkpoint that cannot be read after its
    record was accepted lists a resumption with no resumed epoch (only a
    planted file does this).
- **The suite under the box's versions** (Python 3.12.3, torch 2.12.1+cu130,
  numpy 2.4.6, scipy 1.18.0): r4's suite passed, 659 passed and 1 skipped
  without LaTeX (two tests that read the free disk space failed while the
  sandbox's disk was short and passed once it was freed).
- **An independent review of these changes**, then reviews B, C and D run
  again on them: D's 240 trials (1,095 runs, 604 kills) and 32 + 10 trials
  on miniatures held every invariant, B's CUDA-error reproducers now stop
  with the files kept, and C's coverage of the changed lines is complete
  but for one unreachable branch. Changed after the review:
  - Two of the new tests needed a machine without a GPU to raise ("the
    device" cases): on the box they would have failed RUNBOOK 0c's suite.
    The device failure is now simulated on any machine.
  - The amplitude record with a geometry that could not be meshed was
    briefly treated as incomplete; a second pass would not mesh it either,
    and a rerun that failed could replace a usable record. A record with
    its 16 rows is complete again.
  - Ctrl-C can still be missed in the microseconds while a command
    substitution's process exits (a bash race), so the texts no longer say
    that nothing runs after it; the runbook says to press it again if a
    step starts instead. The instance's balance must cover about 53 h (one
    re-run of an R3 arm included). R3 resumes a checkpoint of its own code,
    configuration and seed (Sec. 8 and the script's docstring say so).
  - Tests made behavioural: the GPU check's process group and the lock
    (each fails with `--foreground` or the lock's closing removed), each
    completeness check killed on its own, the seed in R3's checkpoint
    record where the script writes it; a test that depended on the order of
    the R3 tests now asks for its fixture.
- **A review of those changes** (the same reviewer), and D's harness run
  again on the final script (180 random trials with 454 kills and 83
  targeted ones, 500 Ctrl-Cs with the real interpreter: no violation).
  Changed after it:
  - The completeness checks caught OSError with the parse errors, so an I/O
    error while reading a complete output still read as "not complete" (a
    complete report moved aside, a step run again). They now catch only
    what a missing key or a malformed file raises; an I/O error stops the
    step, as the texts say.
  - A remesh geometry was recorded as failed when its cache file could not
    be written, or when memory ran out, though a second pass could mesh it:
    a cache file that cannot be written is now skipped with a line, and
    running out of memory stops the pass; only what gmsh cannot mesh is a
    failed geometry (Sec. 8 says so).
  - The GPU check refuses to run outside `$(...)`, where closing the lock
    would release the session's own.
  - Tests for the trap before the lock (a slow stand-in for flock), the
    probe's own code against a stand-in torch (each message, and the
    refusal), an I/O error in each check, the lock message, a cache file
    that cannot be written and a memory error while meshing; each fails on
    the code before its change.
- The regression against wp8-lejepa run again on the r5 src tree
  (e120b37d): E1, Phase-2b and E2 identical. The src tree did not change
  after it (round 3 changed the scripts and tests only).
- Suite 705 (660 + 45).
