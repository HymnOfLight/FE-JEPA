# PHASE-2 DEVIATIONS LEDGER

Companion to `PREREG_PHASE2.md` (stamped 28 Aug 2026, config sha e3bdd1e8...,
tag `prereg-phase2`). The stamped file is never edited; every departure from
its Sec. 10 "execute once" protocol is recorded here with evidence, before the
corresponding action is taken. Criteria, thresholds and the configuration are
NOT touched by any entry below (the runner's guard refuses to start on any
configuration byte change).

## D9 -- attempt 1 aborted by GPU OOM in the MGN comparator (31 Aug 2026)

**Facts.** Attempt 1 started 28 Aug ~16:30 (+0800) at the tagged commit
`9bd529e`. Guard verified (`e3bdd1e8778d...`). Labelling completed: val 256
and pool-prefix 1024 instances (4 load cases each => 5,120 solves; the
per-stage progress lines are in `runs/phase2/deciding_run_attempt1.log`; the
process died before printing its ledger summary). E8 AR pretraining completed
for all three seeds (66 h 23 min; states `ar_p1024_s{0,1,2}.pt` on disk,
loaded strictly into the stamped-config model, no wrapper prefixes). Four
supervised units completed (labels/labels_anchor at b=16 and b=64, seed 0;
5 h 08 min). Unit 5 -- `mgn b64 s0`, the FIRST MGN forward of the run --
raised `torch.OutOfMemoryError` (30.34 GiB allocated by PyTorch of 31.36
GiB) inside the edge update `cat([e, h[src], h[dst]])`. No gate or kill
had been computed; no threshold or configuration was changed in response.

**Root cause.** The Sec. 9(1) preconditions bench measured the FE-JEPA
transformer only (23.5 GiB at the fine scale). The MGN comparator was never
benchmarked: on 3D tetrahedral meshes its per-edge activations
(`[E, 3*mgn_dim]` per layer, kept alive across mgn_depth=8 layers and four
load cases for the joint backward) exceed the GPU at in-band sizes. In 2D the
meshes were small enough for this never to surface. Instrument blind spot,
not a numerical fault.

**Disposition (engineering only; configuration untouched; guard still
passes).** (A) `models/gnn.py`: per-layer activation checkpointing in
training mode. Memory-only; the backward recomputes the identical operations
on the identical inputs, so values and gradients are unchanged -- asserted
bitwise on CPU by `tests/test_d9_restart.py::test_mgn_checkpoint_is_bitwise_exact`
(on CUDA the pre-existing `index_add_` atomics are the only nondeterminism,
present in both paths). MGN is a reported comparator: no gate condition and
no kill consumes it. (B) `run-config --reuse-states` (CLI flag, not part of
the stamped configuration): AR states from attempt 1 are consumed instead of
retrained -- the same stamped configuration, seeds and corpus produced them;
their SHA-256 values are embedded in the report
(`results.e8.metrics.d9_restart.ar_states`) to chain attempt 1 to attempt 2.
(C) Supervised and few-shot units now persist their results to a unit cache
(always written; read only under `--reuse-states`), so a further
infrastructure failure costs the current unit rather than the run. (D) The
preconditions bench gains an MGN train-step phase on the largest in-band
instance; attempt 2 starts only after it reports a green memory envelope
(D8: measured, not manual).

**Ledger accounting.** Attempt 1: 5,120 solves (val 1,024 + pool-prefix
4,096), evidenced by the preserved log. Attempt 2: those instances are already
labelled, so its ledger shows the fine stages only (fine-val 1,024 +
fine-prefix 256 = 1,280). Total across attempts = 6,400, exactly the Sec. 7
expectation; the report's economy story is read across the two ledgers.

**Wall clock (re-projected from OBSERVED step times, not the bench).**
The bench measured AR-type steps; on the real prefix instances supervised
steps cost 0.436 s (labels), 0.467 s (fixed anchor, b=16) and 0.774 s
(balanced anchor, two backward passes). Attempt 2 with AR reuse:
labels ~99 h, anchored ~175 h, MGN (to be measured by the new bench phase),
fine few-shot 53-81 h, evaluations ~5 h -- of order 16-20 days.

**Disclosure.** Partial results were observed before the abort: AR loss
curves and four supervised validation displacement values. They influenced
nothing: the configuration hash is unchanged (guard), the fixes are
memory/resilience only, and the observed values are recorded in the
preserved attempt-1 log. Attempt 2 is "the" deciding run of Sec. 10;
attempt 1 is an infrastructure failure that produced no verdict.

## D9 hardening (R9, 1 Sep 2026) -- power-loss resilience, no configuration change

Prompted by the question "can attempt 2 survive a power cut?". Before R9 the
answer was: at unit granularity only (the in-flight unit restarts from
zero -- up to ~44 h for a balanced b=1024 unit), with non-atomic writes that
could leave truncated states, caches or archives. R9 adds: (a) atomic writes
(temp file + os.replace) for every durable artefact -- states, unit caches,
epoch checkpoints, instance archives; (b) corrupt-artefact fallbacks (an
unusable state or cache is reported, removed and retrained rather than
crashing the unit); (c) epoch-boundary checkpoints inside every training
unit, capturing parameters, optimiser moments, scheduler counter, the numpy
order generator, the torch CPU/CUDA RNGs, the step counter and loop
accumulators -- always written, consumed only under `--reuse-states`,
removed when the unit completes. Resume reproduces the uninterrupted
trajectory bitwise on CPU (tests for AR pretraining, balanced supervised
training and a unit-level interruption). Worst-case loss after any
interruption is now one epoch (~7-13 min at b=1024) plus the uncached
recomputations (E6 ~1 h, P3 zero-shot evaluations ~2-3 h). Restart remains a
manual, logged action (a new tmux session, a new log name, a D-note); no
automatic retry loop is added, so a deterministic failure cannot burn cycles
unnoticed. JEPA-loss units (2D legacy) do not resume (their pooled buffer is
not checkpointed); the Phase-2 battery has none.

**Numerical invariance evidence (1 Sep 2026).** With the stamped tag checked
out in a separate worktree, the same seeds and data were trained in
separate processes under the tag's code and under the current head (R8 +
R9 + R10): FE-JEPA AR pretraining, FE-JEPA balanced supervised training and
MGN supervised training (checkpointed layers active) all reproduce the
tag's parameters BITWISE on CPU. The post-stamp engineering commits changed
memory, resilience and provenance only. R10 adds a save throttle for the
epoch checkpoints (first and last epoch always save; otherwise at most one
save per 300 s) and records `resumed_from_epoch` per resumed unit in the
report's `d9_restart` block.

## D10 -- attempt 2 aborted by GPU OOM in the first balanced b=1024 unit (2 Sep 2026)

**Facts.** Attempt 2 (restart mode, head `99d3674`) started 1 Sep; the D9
mechanisms worked: the three AR states were reused (6 min), `mgn b64 s0`
trained (44 min -- the checkpointed comparator fits), and eight supervised
units completed and were cached (44 h, including the 22 h `labels b1024 s0`).
Unit 9, `labels_anchor b1024 s0` (balanced anchor), raised
`torch.OutOfMemoryError` inside the FE-JEPA encoder forward within its first
10% of steps (no progress line yet; 26.82 GiB allocated by PyTorch). The
in-band labelling progress lines in the attempt-2 log are cosmetic: the
labeller prints progress for every file and counts only solves; no label was
re-bought (ledger unaffected).

**Root cause.** `AnchorCache` kept every instance's sparse stiffness matrix
resident on the GPU and never evicted: a 3D anchor is ~10-16 MiB, so the
cache of a b=1024 unit grows to ~10-16 GiB within its first epoch; balanced
mode adds a second backward pass (~2x activations). b=256 units (~3 GiB of
anchors) and the label-only b=1024 unit (no anchors) fit; the balanced
b=1024 unit did not. The preconditions bench measured single-instance step
memory (7.1 GiB at 12k nodes), never a unit's resident set -- the same
instrument blind spot as D9, one level up.

**Disposition (engineering only; configuration untouched; guard passes).**
(A) `anchor/energy.py`: anchors are constructed and kept on the CPU;
`energies()` streams K, F and the mask to the prediction's device per call
(~ms per step against 0.4-0.8 s steps); the cache's GPU footprint is zero for
any prefix. Values and gradients are unchanged (same fp32 numbers, same
kernels): the CPU path is bitwise-identical to before (tag-vs-head regression
equal; test asserts streaming == resident bitwise), and the CUDA path
executes the identical kernels on identical tensors. (B) The preconditions
bench gains `--corpus DIR --resident-prefix N`: one full balanced-anchor
epoch on the first N labelled instances of the real corpus, reporting the
peak -- the restart is gated on `resident_balanced_epoch_b1024` leaving
headroom on the 32 GiB card. (C) Restart via `--reuse-states` (attempt 3):
AR reused, the eight cached units served, and unit 9 RESUMED from its
epoch-1 checkpoint (`[ckpt] resumed E8 labels_anchor b1024 s0 at epoch
1/200`, 3 Sep) -- attempt 2 had completed one epoch of that unit before the
OOM, which also dates the failure to the cache-filled second epoch. The
first live use of the R9 resume path; the report's
`d9_restart.units_resumed_from_epoch` records it.

**Ledger accounting.** Attempt 2 bought no labels (all existed); attempt 3
buys only the fine stages (1,280) -- the cross-attempt total remains 6,400.

**Wall clock.** Remaining after the cache: three balanced b=1024 units
(~44 h each), the b=1024 MGN units, seeds 1-2 of the smaller cells, the fine
block and evaluations -- of order 14-15 days from restart.

**Disclosure.** Partial results observed before the abort (eight supervised
validation values and the AR reuse) influenced nothing: the configuration
hash is unchanged and the fix is memory-placement only.

## D11 -- attempt 3 main process died at the start of P3 (13 Sep 2026; cause pending)

**Facts.** Attempt 3 (restart mode, head `e672110`) completed all 30 E8
supervised units (239 h 25 min; every unit cached, AR states reused) and
E6, then entered P3. The log shows no P3 progress line: sixteen
`SpawnPoolWorker` tracebacks with `BrokenPipeError` on `put(...)` -- the
fine-set labelling workers found the result pipe closed -- followed only by
the resource tracker's shutdown warning. No Python traceback from the main
process exists in the log, and `tee` stayed alive (it recorded the workers'
tracebacks), so the main process alone was terminated by a signal while
waiting for pool results. The most plausible cause is the host OOM killer:
P3 spawns eight labelling workers (each importing torch and solving a
41k-node instance) while the main process still held the 256 in-band prefix
archives, the CUDA context and all E8 results. Confirmation requested from
the box (`dmesg`, `free -g`, `uptime`); this entry is updated when it
arrives.

**Attempt 4 (14 Sep, restarted per manual v6, i.e. without the fix) and
one further restart.** E8 served entirely from cache (30/30 in 0 s), E6
recomputed, P3 entered; the fine-val labelling advanced to 175/256 (labels
written atomically persist across attempts) before the main process died in
the same way; the next restart died again. Three reproductions at varying
points inside the fine labelling -- a memory-driven death that accumulates
(eight workers' resident sets grow across solves on top of the main
process's footprint), not a single-step failure. Dr Song stopped after the
second consecutive repeat, as the manual instructs.

**Root cause (quantified, 15 Sep).** Labelling uses the direct sparse LU
(`method="direct"`, splu) -- the label definition of record. On the in-band
instances (9k-37k dof) a factorisation costs well under 1 GiB, so eight
parallel workers were harmless (attempt 1 labelled 1,280 in-band instances
that way). The fine instances have ~124k dof: measured in the sandbox, a 3D
factorisation grows superlinearly -- 0.2 GiB / 11.5 s at 35k dof, 1.8 GiB /
81 s at 75k dof -- extrapolating to ~7 GiB and several minutes at 124k dof.
Eight workers factorising large fine instances concurrently reach tens of
GiB on top of the main process's footprint; the cgroup OOM killer then
removes the largest process, the main run. A size-driven peak, not a leak;
the death point advances across attempts because solved labels persist.

**Disposition (engineering only; configuration untouched; guard passes).**
(A) `run-config --label-workers N` (CLI, outside the stamped config) so a
restart can trade labelling speed for host-RAM headroom. (B) The in-band
prefix archives are released (`del` + `gc.collect()`) before P3, where they
are no longer used. (C) Labelling workers are bounded with
`maxtasksperchild=64`. None changes any value: labels are computed by the
same solver, the training paths are untouched (tag-vs-head bitwise
regression equal).

**Cost.** E8 is entirely cached; the restart recomputes E6 (~1 h) and runs
P3 (fine labelling, zero-shot, few-shot 53-81 h, naives), WP6 and the gate:
of order 3-4 days.

**Accounting across attempts (R16).** Attempts 3, 4 and 5 died before their
ledger summaries were printed, so the fine solves they bought are recorded
nowhere but in the archives themselves. The report now carries
`labels_present` -- the number of labelled instances in each set the run
relies on (in-band val and prefix, fine val and prefix) with the set sizes
and the load count -- so the cross-attempt total (320 fine instances x 4
loads = 1,280; 1,280 in-band x 4 = 5,120 from attempt 1) is auditable from
the report alone, while each attempt's own ledger covers only its tail.

## D12 -- host-RAM residency of the fine evaluation set (pre-emptive, 15 Sep 2026)

**Facts.** Walking the remaining timeline of attempt 6 after D11: once the
fine labelling completes, P3 loaded all 256 fine evaluation archives into
the main process (`fine_eval_archs` as a list) and every few-shot unit
loaded the same 256 again in-process as its validation set. A labelled 3D
archive is ~1.85 KiB per node (measured: 6.1 MiB at 3.6k nodes, 46.5 MiB at
25k nodes), so a 41k-node fine archive is ~76 MiB and the evaluation set
~15-19 GiB -- held for the whole of P3, and doubled inside each few-shot
unit -- on top of the training archives, anchors and CUDA context. No
attempt reached this point; it is the next cliff of the D11 class.

**Disposition (engineering only; configuration untouched; guard passes).**
Evaluation sets are iterated once per evaluation (metrics, naive
predictions, the supervised loop's final validation), so they are now
`LazyArchives` -- loaded on access, never resident: the runner's fine
evaluation set, and the validation/evaluation sets inside the supervised
and pretrain units. Training archives stay resident (<= 64 fine instances,
~5 GiB). Values are identical to the eager lists (same files, same order;
test asserts equal metrics). Cost: ~2 min of disk I/O per evaluation pass
(~20 passes in P3).

## D13 -- attempt 6: CUDA OOM at the first fine few-shot training step (16 Sep 2026)

**Facts.** Attempt 6 (restart mode, `--label-workers 1`, head `4856e90`) passed
everything D11 had blocked: E8 served from cache (30/30), E6 recomputed, the
fine labelling completed sequentially (val 256/256, prefix 64/64). It then
died at the first training step of the first P3 few-shot unit with
`torch.OutOfMemoryError`: 30.43 GiB in use on the 31.36 GiB card (26.70
allocated by PyTorch, 3.15 reserved but unallocated), 1.06 GiB requested
inside the encoder MLP at ~4e4 nodes x 4 load cases. The bench's fine phase
had measured 23.4 GiB as the allocated peak of one training step; the unit
adds the resident packs of its training instances, the CUDA context and the
allocator's fragmentation reserve, and the sum exceeds the card. Same class
as D9 (MGN), one architecture later.

**Disposition (engineering only; configuration untouched; guard passes).**
Per-block activation checkpointing in the FE-JEPA encoder in training mode,
mirroring D9's MGN treatment: the backward pass recomputes the identical
ops on the identical inputs, so values and gradients are unchanged. Tests
assert bitwise equality of trained parameters with and without it on CPU,
and the three legacy training paths remain bitwise-equal to the tag with
checkpointing on by default. Restart additionally sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
Gate before restart: the bench's `fine.peak_gib` must fall below 16 GiB
(from 23.4), leaving headroom for packs, context and reserve.

**Cost.** Recomputation adds roughly 30% to fine-step time: the few-shot
block grows from ~81 h to ~105 h. E8 remains cached; E6 (~1.3 h) and the
few-shot units restart from scratch (no unit had completed a step).

## D14 -- instrument defect in the label-free (AR) objective, found after the verdict (22 Sep 2026)

**Status of the verdict.** Attempt 8 completed the stamped protocol on 21
Sep 2026 (`prereg-phase2-10-g46cad81`, config `e3bdd1e8778d` verified).
Gate G2: NO-GO -- (a) False, (b) False, (c) False; kills KP1, KP2, KP4
triggered; KP3, KP5, KP6 not. That verdict is the outcome of the stamped
instrument and stands as recorded. Everything below is post-hoc analysis,
labelled as such, and it changes what the next pre-registration must be,
not what this one said.

**Facts.** The AR arm (0 labels, pool 1024) scored disp_rel_l2 0.99964 and
energy_gap_rel 0.99928 on the 256 held-out in-band instances, identical to
five decimals across the three seeds, and 0.99991 at fine. Both numbers are
consistent with a prediction that is the true field scaled by a constant:
with u_hat = alpha u*, disp = 1 - alpha and egap = (1 - alpha)^2; alpha =
3.6e-4 reproduces both to five decimals. In P3, fine-tuning from the AR
states beat training from scratch at every budget (b16: 0.380 vs 0.799;
b64: 0.061 vs 0.193), so the representation was informative while the
zero-shot output was ~1e-4 of the true magnitude.

**Root cause (code, stamped tag).** `forward_instance` applies the battery
scale in decode (`u = decoder(z) * free * fscale`, WP7 3D-P0.5
`scale_decode`); the AR loss in `train/losses.py::compute_loss` decoded
`u = decoder(z) * free` WITHOUT the scale and scored that field with the
anchor. Training therefore drove the unscaled decoder output toward u*;
inference returned it multiplied by fscale = max|F| of the battery (~3.6e-4
on this corpus), giving alpha = fscale. The Phase-1 (2D) code predates
`scale_decode`, so the same objective was consistent there (AR disp 0.166).
The tag-vs-head bitwise regressions could not see this: both sides carried
the defect. No test asserted that the loss and inference decode the same
field.

**What it invalidates.** Every quantity that depends on the AR objective:
the AR cells (in-band and fine), the E6 probe (trained with the same loss),
P3's AR zero-shot and the P3 fine-tune arm (initialised from AR states),
and therefore gate conditions (b) and (c) and kills KP1, KP2, KP4 as
evaluated. What stands: the supervised grid (30 units; labels,
labels_anchor, mgn -- trained and evaluated through `forward_instance`),
the naive baselines, the P3 scratch arm, WP6, the labels, the corpora and
their hashes, and the compute ledger. Gate condition (a) (supervised b=16
only 2.71x better than zero, threshold 3.0x) failed independently of the
defect and is NOT addressed by the fix.

**Fix (R18).** A single decode path: `decode_battery(z, pack)` on the model
(free mask and, when `scale_decode` is on, the battery scale), used by
`forward_instance` and by the AR loss. Test: the anchor inside the AR loss
receives bitwise the field `forward_instance` returns (fails on the stamped
code, passes on R18). Tag regression: the AR path changes by design; the
supervised and MGN paths remain bitwise-equal to the tag. On a CPU corpus
the corrected objective trains (disp 0.44, egap 0.20 at toy scale) under
both decode conventions.

**Disposition.** No re-run under the Phase-2 pre-registration. A
pre-registered amendment (Phase-2b) re-runs the AR-dependent arms with the
corrected instrument, reuses the standing supervised units by SHA chain,
adds an instrument pilot with a pre-declared 'leaves zero' threshold before
the full AR arms, and must state in advance how condition (a) is treated.

## D15 -- attempt 7: silent death inside the seventh few-shot unit (18-19 Sep 2026)

Attempt 7 ran six fine few-shot units (39.8 h) and died at 90% of the seventh
(`P3 finetune b64 s1`) with no Python traceback in the log, the D11 signature
of a signal death; cgroup counters were requested and are pending. Attempt 8
restarted per the manual, resumed that unit from its epoch-48/50 checkpoint
(128 steps lost), served the six completed units from cache and completed the
protocol. No code change; the R9 mechanism worked as designed.

## Phase-2b amendment mechanics (R19, 22 Sep 2026; post-verdict)

`gate_g2.sanity_min_budget` (default 0 = every budget, the stamped Phase-2
form) exempts budgets below the floor from condition (a); when it is raised
the runner also computes the all-budget form and reports it as
`gate_g2_reference_all_budgets`. E8 gains `ar_only` on the main line (ported
from the wp8 branch) for the instrument pilot. Configurations
`configs/phase2b_v1.json` and `configs/phase2b_pilot.json` and
`PREREG_PHASE2B.md` are added; nothing in `configs/phase2_v1.json` or
`PREREG_PHASE2.md` changes. The choice of the sanity floor (64, the decision
budget) is a post-hoc change and is disclosed in PREREG_PHASE2B Sec. 4.

**R20 (engineering, 22 Sep, after the Phase-2b stamp).** The results page is
named after its report (`report_phase2b.json` -> `RESULTS_phase2b.md`) so the
amendment, which shares `runs/phase2/` with the deciding run, cannot overwrite
the Phase-2 page; the reference gate is rendered beside G2b. Rendering and
naming only: no value changes, configuration untouched, guard verifies.
