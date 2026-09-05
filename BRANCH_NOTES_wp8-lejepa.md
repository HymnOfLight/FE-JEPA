# BRANCH NOTES -- wp8-lejepa

Branched 1 Sep 2026 from `wp7-3d` at `99d3674` (post `prereg-phase2` tag,
R10). Purpose: evaluate, under the project's falsification discipline, what
the LeCun JEPA lineage (LeJEPA, LeWorldModel) and the adjacent PDE-JEPA work
(PI-JEPA, AeroJEPA) can contribute to FE-JEPA's accuracy, training cost and
instrumentation -- WITHOUT touching anything the Phase-2 deciding run depends
on. Everything on this branch is exploratory until an E-series
pre-registration stamps it.

## Governance

- No file that the stamped Phase-2 run reads is modified here
  (`configs/phase2_v1.json`, `PREREG_PHASE2.md`, the training loops as
  executed by `run-config`). New code lives in new modules and scripts; the
  2D/3D test suites stay green and bitwise regression tests against the tag
  are re-run before any merge.
- Every experiment below is a falsifiable question with kill criteria declared
  before execution (E-series, fresh SHA-256 per pre-registration). GO and
  NO-GO are both publishable outcomes.
- Prior art now on record and to be cited in the CMAME manuscript: PI-JEPA
  (arXiv:2604.01349), AeroJEPA (arXiv:2605.05586), LeJEPA (2511.08544),
  LeWorldModel (2603.19312). Our differentiator is the training signal: the
  exactly assembled discrete potential energy (Lemma 1), not a residual
  penalty, a teacher network or a reconstruction label.

## Stage 0 -- instruments and code, CPU only (this week)

1. `train/sigreg.py`: SIGReg (Epps–Pulley on random 1-D projections), knot
   form (linear in N) cross-checked against the closed form; differentiable
   regulariser + gradient-free monitor. Tests: closed-form agreement, shape
   discrimination (Gaussian < uniform < collapsed), descent reduces it.
2. `scripts/intrinsic_dimension.py`: TwoNN and PCA dimensions of the node-token
   cloud from a trained state, the SIGReg baseline, and the output-LayerNorm
   check. FINDING already on record: the FE-JEPA encoder ends with a
   LayerNorm, so any latent shaping must act on a BatchNorm projector head
   (LeWM's own fix), never on the raw encoder output.
3. Literature note and CMAME related-work paragraph (delivered outside the
   repo; see the assessment memo of 1 Sep 2026).
   Stage 0.1/0.2 review record: TwoNN made memory-safe (matrix-product
   distances) and corrected to Facco et al.'s fit (validated on manifolds of
   known dimension); SIGReg evaluated in checkpointed projection chunks
   (identical values/gradients, one chunk of memory) and self-protected to
   fp32 under autocast; default 17 knots within 0.5% of the closed form.
   Stage 0.3 review record (1 Sep, executed checks): (i) the Epps-Pulley null
   distribution is N-stable (median ~0.08-0.11, 95th ~0.28-0.39 at
   N = 500/2000/8000 standardized Gaussian) -- the normalisation carries no
   hidden N factor; (ii) LeWM's claim that a final LayerNorm blocks
   distribution shaping DID NOT REPRODUCE on our encoder at toy scale: 60
   Adam steps through the raw LN output reduced SIGReg by 93% (0.347 ->
   0.023), the BatchNorm projector path by 63% from a lower start.
   Consequence for E1: the projector head stays the DEFAULT (independent
   reason -- it decouples the shaped space from the space the decoder and
   the energy anchor consume), but the raw-LN variant is retained as a
   cheap pre-registered ablation rather than excluded on the literature's
   authority; (iii) the instrument's real (non-smoke) path executed end to
   end against a saved state, a config and a corpus.
4. Stage-0 gate for Stage 1: run instrument 2 on the Phase-2 AR states once
   the box is free. Reading rule (pre-declared): if TwoNN ID < 0.25 x latent
   dim, E1 uses a projector head of width ~2 x ID; otherwise full width.

## Stage 1.0 -- arm code in place (1 Sep 2026, CPU-validated)

- E1 arm: `LossConfig.reg_mode = "sigreg_ep" | "sigreg_ep_head"` (the validated
  Epps-Pulley SIGReg; the legacy 2D modes `sigreg`/`sigreg_pooled`/`vicreg_pooled`
  used by the stamped 2D E3 battery are untouched); `ar_sigreg_config(lambda)`;
  the BatchNorm projector head is attached by `pretrain()` before the optimiser
  and stripped from the delivered state (strict-loadable). Pretrain units accept
  a dict loss spec.
- E2 arm: `models/bottleneck.py` (`kind = "bottleneck"`): deterministic FPS
  seeding, Voronoi assignment, scatter-mean pooling, transformer over M tokens,
  per-node decoding; same pack contract and output tail as FE-JEPA, so the
  decoded u feeds the same exact anchor. `needs_pack` lets `compute_loss` and
  the instruments pass the pack; the legacy call path is unchanged.
- Evidence: suite 193; tag-vs-head bitwise regression (AR, balanced supervised,
  MGN) equal on this head; E1/E2 run through the production units.
- Drafts r1 of PREREG_E1 and PREREG_E2 delivered outside the repo (stamp after
  the Phase-2 verdict).
- Stage 1.1/1.2 review record: the bottleneck runs on a real tetrahedral
  gmsh3d instance (forward, exact anchor, supervised step, evaluation); an
  interrupted E1 run resumes bitwise (head included, RNG restored by R9);
  CPU scaling probe on 3D meshes of 106-1,581 nodes: per-node transformer
  t ~ N^1.29 (rising toward N^2, cf. the box's 3.7k->12.3k measurement),
  bottleneck (M = 128) t ~ N^0.43 -- no hidden quadratic, 9.5x faster at
  1.6k nodes and widening. The preconditions bench gains an opt-in
  --bottleneck-tokens M pair of phases (largest in-band + fine) so E2's
  kill/GO lines are written against measured step time and memory.

- Stage 1.3 review record: the drafts were NOT executable -- e8 and p3
  hardcoded the FE-JEPA-role architecture (`"fejepa"`) and the AR loss
  (`"ar"`). Now config-driven: `model.kind` (routing key, stripped before
  the builders) and `pretrain.loss_spec` (dict overrides on AR_CONFIG),
  threaded runner -> e8 -> units and p3 zero/few-shot; defaults reproduce
  Phase-2 byte for byte (legacy bitwise regression re-run: equal). A test
  runs a bottleneck + SIGReg-head configuration through `run_config` end to
  end (E8, P3, gate).

- Stage 1.4 review record: E1's adjudication statistic had no instrument --
  `scripts/latent_separation.py` now computes S (silhouette over quartile
  bins of the first principal component of the per-instance geometry
  descriptor), the 1-NN bin accuracy and the SIGReg monitors, kind-aware;
  tests on constructed clusters (S > 0.95 tight/far, ~0 unstructured, exact
  quartiles). `intrinsic_dimension.py` builds the configured kind. The runner
  refuses a non-fejepa `model.kind` with e6/wp6/e1 enabled BEFORE any corpus
  is generated (test). Drafts r2 carry the exact configuration keys.

- Stage 1.5 review record: the drafts' "AR pretraining only" mechanics did
  not exist (`budgets: []` crashed in `max()`); `experiments.e8.ar_only`
  now skips the supervised grid, naive rows and label-dependent kills
  (marked unevaluated) and the gate tolerates a run without labels cells;
  verified end to end through `run_config` with the bottleneck, the SIGReg
  head and P3. Both instruments ran their real paths on a 3D gmsh corpus with
  a bottleneck state (3D geometry-descriptor branch, kind-aware loading).
  Drafts r3 name the switch. Legacy bitwise regression against the tag: equal.

- Stage 1.6 (tooling for the waiting window): `scripts/audit_phase2_report.py`
  -- an INDEPENDENT audit of the Phase-2 report: provenance (config sha, tag
  in git describe, guard record, TF32 policy, dataset SHAs), accounting
  (ledger, d9 restart block, AR-state SHA chain against the box's sha256sum),
  per-seed re-aggregation of every cell mean, and an explicit re-derivation of
  G2 (a)(b)(c) and KP1-6 compared item by item with the runner's block
  (agrees 10/10 on the mini reports; hand-built-report tests cover each kill
  and a runner-disagreement case). `scripts/e1_lambda_pilot.py` implements
  PREREG_E1 Sec. 3's selection rule (grid, seed 0, largest lambda within the
  tolerance of the AR pilot). Both smoke-validated.

- Stage 1.7: the drafts' kill/GO rules are executable -- `adjudicate_e1.py`
  (K1 parity per seed, K2 no-effect on S, GO with transfer-ratio guard) and
  `adjudicate_e2.py` (K1 accuracy vs the Phase-2 baseline at the seed median,
  K2 speed from the bench's bottleneck phase, GO), tests on hand-built
  inputs for every rule. The COMPLETE chains were rehearsed through the real
  scripts as subprocesses on small corpora: E1 pilot -> base run -> shaped
  run -> four separation readings -> verdict; E2 baseline run -> bottleneck
  run -> bench with bottleneck phase -> verdict. No interface mismatch.

- Stage 1.8 (software-engineering review, behaviour-preserving): analysis
  logic moved out of `scripts/` into the importable package `fejepa/analysis/`
  (common plumbing, intrinsic_dim, separation, adjudicate, audit); scripts are
  thin CLIs; tests import the package (no importlib loading of scripts);
  `audit()` split into four single-purpose functions with an
  `AuditExpectations` dataclass; a shared `tiny_corpus` fixture appended to
  `tests/conftest.py`; the bottleneck is sized by `features.spatial_dim`
  (padding/slicing hack removed; coordinate/spec mismatch raises). Evidence
  of zero behaviour change: suite 205 unchanged; legacy paths bitwise-equal
  to the tag; the audit CLI ALL OK on the dress-rehearsal report; the E1
  chain re-run through the thin CLIs reproduces the previous verdict numbers
  bitwise. Optimisation candidates assessed and declined (see the review
  memo of 2 Sep): none justifies added complexity; the order-of-magnitude
  lever is E2's architecture, adjudicated by experiment.

- Stage 1.9 (review follow-through): the E2 chain (baseline run, bottleneck
  run, bench bottleneck phase, adjudicate_e2 CLI, separation on a bottleneck
  state) re-run through the thin CLIs on 3D gmsh after the refactor -- OK;
  `e1_lambda_pilot.py` moved onto the package plumbing (all six analysis
  scripts now consistent); `tests/test_wp8_arms.py` consumes the shared
  `tiny_corpus` fixture (the fixture has a real consumer; local helper gone).

- Stage 1.10 (semantic review of the instruments): the drafts measure on the
  VAL set, but `instance_files` defaulted to the training pool prefix -- a
  methodological mismatch (latents shaped on those very instances). The
  instruments now select the run's held-out validation split from the
  config's `split` block by default (`--subset val`; `pool` remains
  available and is recorded in the output); tested against the runner's
  own split and exercised on a run config. PREREG_E1 draft r4 pins it.

- Stage 1.11 (text-to-code correspondence of the drafts): three findings.
  E1's GO rule uses the transfer ratio while its mechanics disabled P3 --
  P3 now supports a zero-shot-only form (`fewshot_budgets: []`,
  `naive_budget: 0`) and the drafts enable it. The runner's label need
  ignored `ar_only` (it would buy pool labels on a fresh corpus) and always
  labelled the fine prefix -- both fixed, so the E-series economy is
  guaranteed by construction (test: on fresh corpora exactly the val and
  fine-eval labels are bought). The head-width rule is now computed by the
  instrument (`suggested_head_width`). Drafts E1 r5 / E2 r4.

- Stage 1.13 (plan gap review, 5 Sep): the E-series had no configuration
  files and no cheap way to validate one. `scripts/make_e_series_configs.py`
  derives `configs/e2_m{512,1024}.json` and `configs/e1_2d_{base,shaped}.json`
  from the stamped Phase-2 / Phase-1 configurations (committed); `run-config
  --dry-run` validates a configuration through the guards, model kind, plan
  and label need and stops before any data work (the prereg guard becomes a
  reported item under dry-run only); unfilled `loss_spec` placeholders are
  refused. `count_steps` respects `ar_only`. The adjudicator reports the
  transfer guard explicitly ("not evaluated" in 2D, where the Phase-1
  configuration has no transfer set -- E1 draft r6 adjudicates the 2D stage
  on K1/K2/S). B5 reading added: leave-one-out ridge R^2 from pooled latents
  to the geometry descriptor (`probe_r2_geometry`).

- Stage 1.14 (5 Sep): main-line R13 merged (D10 record corrected; test
  hygiene -- the branch's bench and anchor files kept their already-merged
  versions). The GENERATED configurations were executed end to end at
  scaled size for the first time (only paths and sizes overridden):
  E2 (bottleneck run + Phase-2-shaped baseline + bench phase + adjudicator)
  and E1 (Phase-1 shape, both arms + separation + adjudicator with the
  transfer guard reported "not evaluated"). Finding: the E1 configuration
  inherits `labelled_policy: asis`, so the run buys no labels and requires
  the val split to be pre-labelled (`fejepa label ...`, as on the box since
  Phase 1); the ledger reads 0 by construction. PREREG_E1 draft r7 records
  the precondition.

- Stage 1.15 (5 Sep): the branch leaves the stamped Phase-2 artefacts
  byte-unchanged (0-line diff against the tag); the committed E-series
  configurations reproduce byte for byte from the generator, now enforced
  by a test (a silent hand edit fails it). PREREG_E1 draft r8 names the
  baseline-state reuse mechanism (Phase-1 AR states placed in the base run's
  `e8_states` and consumed via `--reuse-states`, SHA-chained) and corrects
  the 2D cost estimate (~9 h per arm).

- Stage 1.16 (5 Sep): coverage gap closed -- the R9 resume path was only
  tested for FE-JEPA; E2's bottleneck AR units (FPS-seeded packs rebuilt on
  resume) now have a bitwise resume test. With this, every E-series unit
  type (FE-JEPA AR, AR+SIGReg head, bottleneck AR, supervised) has an
  executed resume proof. Branch checks have converged: the next information
  comes from the box (Phase-2 verdict, intrinsic-dimension reading, E2 bench
  phases).

- Stage 1.17 (5 Sep, property tests): SIGReg is rotation-invariant in
  expectation (isotropic sketch; test). The bottleneck was NOT independent
  of the mesh's node numbering: FPS tie-breaking and seed order followed
  node indices, so equidistant nodes could land in different tokens under a
  renumbering (a permutation test on the shared feature battery first proved
  the battery itself equivariant to 1e-16). FPS now breaks ties by
  coordinates and returns seeds in canonical lexicographic order; the
  permutation test passes (outputs equal up to float summation order). No
  stamped or measured artefact depended on the previous order.

- Stage 1.18 (5 Sep): edge-case and process tests -- SIGReg and the
  BatchNorm head are finite (value and gradient) on a collapsed constant
  embedding; bottleneck units survive spawn pickling under workers=2.
  `RUNBOOK_E_SERIES.md` lists every E-series command in execution order.

- Stage 1.19 (5 Sep): four more executed checks -- the bottleneck is
  load-scale blind (prediction scales exactly with the load battery, the
  invariance FE-JEPA rests on); SIGReg's gradient through the chunked,
  checkpointed path matches finite differences; the configuration hash
  changes with the nested `loss_spec` values (lambda, head width) so E1's
  stamp covers them; `--dry-run` reports `verified` against a stamped file.

- Stage 1.20 (5 Sep): the instruments build the model on the GPU when
  available (`--device auto`; 256 in-band 3D encodes on the box CPU would
  take ~25 min); load-case equivariance of the bottleneck tested; PREREG_E1
  draft r9 pre-declares the pilot outcome when no lambda is admissible
  (NO-GO-AT-PILOT, pilot JSON as the record).

- Stage 1.21 (5 Sep): state reuse and the analysis loader normalise a
  torch.compile `_orig_mod.` key prefix (defensive: Phase-1 states are clean,
  but a prefixed state would otherwise force a silent retrain); the runbook
  names Phase-1's state directory concretely (`runs/e8_states/`, from its
  `out` of `runs/report_rec8_v2.json`).

- Stage 1.22 (5 Sep, statistical power of the rules): E1's GO rule ("S
  improves at every seed") had no effect-size floor, so three same-signed
  noise-level deltas would have passed. `adjudicate_e1` now requires every
  seed's delta to exceed max(0.02, 2 x the AR arm's seed SD of S) and
  reports the floor; the separation reading reports PC1 loadings and
  variance share (which descriptors the bins follow). PREREG_E1 draft r10.

- Stage 1.23 (5 Sep): instance-level uncertainty of S (bootstrap 95% CI
  over validation instances, bins fixed) reported next to the point
  estimate; both verdict CLIs record the SHA-256 of every input file
  (`inputs_sha256`) so a verdict is traceable to its exact inputs; the E2
  verdict reports seed spreads of the energy gap for interpreting a marginal
  K1.

## Stage 1 -- E-series pre-registrations (box free, after the deciding run)

**E1 -- latent shaping and cross-geometry separation.** Question: does adding
lambda * SIGReg to the AR arm improve latent separation ACROSS GEOMETRIES
(the gap left open by Proposition 1, which bounds separation across load
cases on a shared K only) and the P3 transfer ratio, at no cost to the
in-band gate metrics? Arms: AR (current) vs AR+SIGReg (single lambda found by
bisection on a pilot, then frozen), 3 seeds, shared corpus and labels.
Kill: (a) in-band disp or energy-gap parity worsens by more than the
Phase-2 parity band; (b) no measurable change in the cross-geometry
separation statistic (pre-declared: silhouette-type score of instance means
across geometry descriptors) at all seeds. GO: separation improves at every
seed AND P3 ratio does not worsen. Cost estimate: one extra AR arm
(~3 x 22 h at 3D scale) -- or run on the 2D corpus first (hours).

**E2 -- token bottleneck with continuous decoder (speed lever).** Question:
does an AeroJEPA-style architecture (FPS-seeded local aggregation into M
tokens, attention over tokens, conditional per-node decoder; conditions
injected by zero-initialised AdaLN) reach parity with the current per-node
transformer in the energy norm while cutting the fine-scale step time by an
order of magnitude? The energy anchor is untouched (decoded u feeds the same
exact energy). Arms: current architecture vs bottleneck (M in {512, 1024}),
AR training, 3 seeds. Kill: energy-norm parity worse than the Phase-2 band at
b_max; or fine-scale step time not below 2 s (from 12.1 s). GO: parity within
band AND fine step time < 1 s. Deliverable either way: the measured
accuracy/cost frontier.

**Explicitly out of scope on this branch:** any change to the Phase-2 gate,
kills, thresholds or battery; any latent-prediction arm trained without the
energy anchor (AeroJEPA needed a reconstruction term to keep physical
validity -- our anchor is that guard and is never dropped); soft PDE-residual
penalties (PI-JEPA's own ablation: neutral to harmful); EMA/stop-gradient
heuristics (LeJEPA/LeWM: unnecessary with SIGReg).

## Stage 2 -- decision

With the Phase-2 verdict and E1/E2 on record, decide whether the bottleneck
architecture becomes the Phase-3 main line (fire extension) and whether
SIGReg enters the default AR loss. Both decisions are taken on measured
numbers against pre-declared criteria, not on the literature.
