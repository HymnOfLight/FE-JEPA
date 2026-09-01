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
