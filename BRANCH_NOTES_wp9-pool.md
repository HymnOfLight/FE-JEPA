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
  do not change with the mesh (the battery's L1 load instead of its largest
  nodal force; per-node load densities; load summaries free of the mesh)
  remove the fine-mesh amplitude deficit of wp8's post-hoc reading 4e? S
  enters the pre-registration only if session 1's rule 1 admits it.

C2 (adaptive allocation of the training budget) and 3D are not in this round.

## Governance (carried from wp8)

- No file a stamped run reads is changed: the stamped configurations, the
  PREREG files, the default training and evaluation paths. Every new switch
  defaults to the old behaviour, with a test that the default path is bitwise
  unchanged; Stage 0b proves it against `wp8-lejepa` with
  `scripts/regress_against_branch.py`.
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
sizes scaled down. Suite 350.

## Stage 0b (open; written while session 1 runs)

- S switch (default off): decode scale = the battery's L1 load; per-node load
  densities; mesh-free load summaries. Tests: the default path bitwise
  unchanged; outputs scale with the loads; S's features and decode scale
  unchanged across h on R-type remeshes.
- 2D multi-holdout evaluation in the runner (F1-F5, evaluation-only, pinned by
  manifest) and c* readings in the evaluation outputs (secondary, label-free).
- C1 and S configurations from session 1's decisions; the regression of the
  default path against `wp8-lejepa`; PREREG_W9.md; the pre-run audit; stamp
  and tag.
