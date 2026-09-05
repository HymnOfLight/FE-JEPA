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
