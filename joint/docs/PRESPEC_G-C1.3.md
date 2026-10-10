# Pre-specification of gate G-C1.3

Stamped 10 October 2026, 23:02 BST. At that time no member drawn with this gate's seeds had been exported or solved, and no training of this gate had started.

## Question

Can the project's transformer, trained on the total potential energy alone and never shown a solution, predict for extended end plate joints it has not seen:
- the initial stiffness;
- the contact zone against the column?

Both are judged against the reference finite element model.

**Hypothesis.** Trained on the energy alone, the transformer meets C1 and C2 below on the held-out members.

- **The reference** is the idealised elastic contact model of G-C1a (`fejoint/joint.py`):
  - bolts with heads that bear unilaterally;
  - a rigid, frictionless column flange;
  - a 300 mm beam segment with its end section rigid and loaded;
  - the 10 mm mesh with one layer of cells in the end plate, the beam flanges and the web.
- **Distances of the reference to the other models** (FS1 to FS4, measured moduli; `probes/c13_scale.json`, `probes/c13_segment_effect.json`, `docs/G-C1a2_verdict.md`):
  - it is 1.0% to 2.2% stiffer than the same model on G-C1a's fine mesh with the full beam;
  - the higher-fidelity model of G-C1a2 lies between 0.2% softer and 2.3% stiffer than it.
- **The gate does not test the reference against the experiments.** That gap is stated and unexplained.
  - The same model on G-C1a's fine mesh with the full beam is 2.05 to 2.99 times stiffer than the measured initial stiffness of the eight joint tests (G-C1a). The reference itself is 2.08 to 3.05 times. The higher-fidelity model is 2.11 to 3.04 times (G-C1a2).
  - On the same laboratory's welded T-stubs, the higher-fidelity T-stub model is 1.10 times the tests in the geometric mean, and 0.79 to 1.55 times test by test (G-C1b, a narrow pass).
  - With the loaded end held as theirs holds it, it is 7% to 9% stiffer than the authors' own finite element model on the three specimens both modelled (`docs/source_GiraoCoelho2006.md`). For WT7_M12 this is an estimate: the medium mesh's effect of holding the end, applied to the fine-mesh value.
  - The machinery of the reference model is 8% to 18% stiffer than that T-stub model. It gives 1.24 times the tests in the geometric mean, with 15 of 25 tests inside the band (G-C1b's baseline).
- **Name.** The C plan of 9 October 2026 called this gate G-C1b; that name now belongs to the T-stub gate.

## Authority

**The C plan of 9 October 2026** set out stage C1.3, its arms, its metrics and its gate. It proposed two thresholds for the project owner: a median stiffness error of at most 5% and a contact overlap of at least 0.9.

**On 10 October 2026 the project owner** wrote the following (messages translated from Chinese; the originals are kept in the correspondence record):
- 09:58 BST, "carry on as you recommend": this adopted G-C1a's recommendation, "B, then C ... Then A for C1.3". Option A keeps the idealised model as the truth for stage C1.3, judges the surrogate against it, and states the gap between the model and the tests as unexplained.
- 19:09 BST, "carry on with the recommended options": this adopted the recommendations of that evening's report. They were the idealised model on the 10 mm mesh as the reference (`docs/C1.3_data_design.md`; the preparation note had offered 10 mm or 7.5 mm), and a pre-specification written after 15 October, which the next message lifted.
- 20:37 BST, "no need to set the 15 October milestone; carry on directly": the reply to a report that recommended the 300 mm segment. That report listed, among the owner's decisions, the thresholds and the parameter ranges, to be set when this pre-specification is written.

**Adopted:** the 300 mm segment (`docs/C1.3_data_design.md`).

**Not set by the owner:** the thresholds and the parameter ranges. This document fixes them at the C plan's proposals (5% and 0.9) and at the ranges of `docs/C1.3_data_design.md`. The owner may change them until the first training run starts; a change is recorded with this document.

**Chosen here:** softplus at $T = 10^{-3}$. `docs/C1.3_preparation_note.md` left it and ReLU as equal candidates. Softplus had the smaller energy gap and stiffness error in both seeds of the two-dimensional probe, and it was the pilot's map.

**Departures from the C plan**
- The C plan also named an arm trained on the energy plus labels, and a graph network if the size allowed. Neither is run here.
  - This gate asks whether the energy alone suffices.
  - Another arm of three seeds would add up to 43% to the budget.
  - The graph network is not the project's model, whose encoder is the subject of the gate.
- The C plan's displacement error is reported as the relative error of the physical displacements, beside the energy-norm error.
- The C plan set the GPU budget after a cost probe on the GPU host, before the pre-specification. That probe was not run. The budget is fixed here at 200 epochs, and the bench, after the stamp and before any training, stops the run above 0.25 s a step.

**Changes.** Any change after this stamp is reported as a change after the stamp. No threshold, metric or arm changes once the first training run has started.

## What was seen before the stamp

**Earlier gates.** All the results of G-C1a, G-C1a2, G-C1b with its addendum, and the reading G-C1c. The source note on the authors' 2006 model (`docs/source_GiraoCoelho2006.md`).

**Exploratory checks.** None of them used a member drawn with this gate's seeds.
- The scale of the reference model (`probes/c13_scale.py`).
- The beam segment's effect on FS1 to FS4: +0.2% to +0.9% in $S$ (`probes/c13_segment_effect.json`).
- A two-dimensional network probe of the contact map, two seeds (`probes/c13_net/`).
- A direct optimisation of single fields on a T-stub, without a network (`probes/param_probe.py`).
- Smoke tests on one joint, seed 999 (`probes/c13_cost/smoke_train*.json`).
- The cost probe (`probes/c13_cost/cost_probe.py`):
  - members drawn with seed 20261010, exported and not solved;
  - a small model on the CPU, 2.1 s a step (`probes/c13_cost/c13_cost_cpu_small.json`).

**A pilot over a small family**, run on the CPU (`probes/c13_cost/pilot_family.py`, with results in `pilot_energy.json` and `pilot_supervised.json`):
- seeds 101 (64 training joints) and 102 (16 held-out), the 10 mm mesh with the 300 mm segment;
- a small model: width 64, depth 2, 4 heads;
- Adam with a one-cycle schedule (10% warm-up), no clipping, instances drawn with replacement, 6,000 steps, one seed;
- held-out medians at the end:

| Arm | Energy gap | Median of $\lvert S_{pred}/S_{ref} - 1\rvert$ | Contact overlap | Energy-norm error |
|---|---|---|---|---|
| Label-free | 4.7% | 7.9% | 0.77 | 0.22 |
| Supervised, 64 labels | 110% | 3.5% | 0.24 | 1.06 |

- The signed medians of the stiffness error were +7.9% (label-free) and −2.8% (supervised).
- In the label-free arm, from 4,500 to 6,000 steps the energy gap fell from 6.0% to 4.7% while the stiffness error rose from 5.3% to 7.9%.

**The seeds first planned for this gate.**
- Before they were retired, a check of the command line exported the first training member and the first held-out member drawn with them (20261011 and 20261012). It timed training steps of the full model on the CPU on the training member; that time was not kept and was not used for the budget. Neither member was solved, and the exports were deleted.
- **What was solved.** The test suite, run several times on the evening of 10 October 2026, drew its members with those two seeds. Each time it solved two training and two held-out members, and trained a tiny model on them. Only the test outcomes, and the messages of any failing test, were read; no metric of those members was examined.
- Both seeds were then retired, and this gate's samples were redrawn with 20261013 and 20261014. The tests of `c13/` now draw with seeds 1 and 3, and the family tests with 3, 7 and 8. The reviews' checks drew only with seeds in SEEDS_SEEN (1, 3, 7 and 8).

**This gate's samples.** No member drawn with 20261013 or 20261014 has been exported or solved.

**The tested layouts.** The four tested layouts have been solved in this gate's configuration, by the tests and by the reviews' checks. They are reported, not judged.

**The test suite.** The code's tests, including two-instance training runs with a tiny model.

## Expected outcome, stated before the run

- **C1 is uncertain.** The pilot's stiffness error was 5.3% after 4,500 of its 6,000 steps and 7.9% at the end, while its energy gap kept falling. The run's model, training set and budget are larger, and it trains with AdamW and clipping.
- **C2 is likely to fail.** The pilot's contact overlap stayed below 0.8: it ended at 0.77.
- **The gate is therefore expected to fail**, on C2 at least.
- **The supervised arms** are expected to read the stiffness more closely than the label-free arm. They are expected to carry far larger energy errors and to miss the contact.

## Configuration

**The configuration** is `c13/config.py`.
- Its canonical JSON has SHA-256 `727d9bde73e5129a3ea9d26b3d6f3ac93d1f7f19505cd9605000c601152bb11e`.
- The training and evaluation commands refuse to run unless they are given this hash.

**The repository code** the run imports is pinned in the configuration, file by file, at wp9-pool 92dfc93. The list comes from an import trace of training and the bench. Training, the bench and `write-config --repo` refuse to run against other files.

| File | SHA-256 |
|---|---|
| `src/fejepa/__init__.py` | `6b4a69dde91fbd83c5c16707a00e4c9e6a2e02647aca1d3132b313e4b20f9fdc` |
| `src/fejepa/anchor/__init__.py` | `88d852286c71e468491b7d0b34cec2c467bd4c96724a26ca8ebcc6b8f92306e4` |
| `src/fejepa/anchor/energy.py` | `2f8afe7243ee2cfbeb23ed505475d1f780de9ad020eedf0216f6a36cb272e802` |
| `src/fejepa/models/__init__.py` | `88d852286c71e468491b7d0b34cec2c467bd4c96724a26ca8ebcc6b8f92306e4` |
| `src/fejepa/models/features.py` | `0af43e0a87fda6bc177e9b446a720ac11e3e116bc29e445446edf846162c208d` |
| `src/fejepa/models/fejepa.py` | `3263988641477cfab564c5c622f8cf3406d98e45078134da8c6905d4b410b847` |
| `src/fejepa/train/__init__.py` | `88d852286c71e468491b7d0b34cec2c467bd4c96724a26ca8ebcc6b8f92306e4` |
| `src/fejepa/train/schedule.py` | `d53b55877c65efc37ee0a3877362451d8227ef84f581df64138ac1b469bd57c5` |

**SHA-256 of the gate's own code at stamping** (paths relative to `joint/`):

| File | SHA-256 |
|---|---|
| `c13/__init__.py` | `3bdb83e76e81b0ff504ddb6f388907fb3e5244f72f0e6c5a9dfaa194995df30b` |
| `c13/__main__.py` | `e3300fc899d7dfcca5f0845e3f0447b1d74509a77413f82b4cb10a135d8dbad8` |
| `c13/config.py` | `c394306beaca690eb6d911f6d61aea7d6a9deffcb3c193d44e62f0fdccf9a222` |
| `c13/data.py` | `b8cb567905fbf66ae6a3bf484a66570c4fca4c7f911c0bfd5e106819b03947e4` |
| `c13/model.py` | `bb9e816d9a4b086f834d57b6db841b70ea7dcbbe47e2f9bedde3d88829513496` |
| `c13/train.py` | `3fef3105e461d46870d82205e2c05e6b29958886864e028f5a2d38152ae7175c` |
| `c13/evaluate.py` | `b019ab0ac53bab8b5d83dfe1bccb669dbea6930ff28e9559b49e200d999f5a20` |
| `fejoint/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `fejoint/family.py` | `7ebed5af12d08c243d858074b299a80c48d5423670f90a8a4ce20a7a904e6a27` |
| `fejoint/export.py` | `c766b9f1fec5f86e27e1ae9f8a9be797df680886312e08d7ffa89d39d65217fe` |
| `fejoint/joint.py` | `23b7fb48f9a587384c797b597408e85b8c8e368dad69631af90c16c564a5c044` |
| `fejoint/specimens.py` | `dc7df5acb125e506c849e41af6dd89d4716256c44ee9d7c34247e126df588dfb` |
| `fejoint/hex8i.py` | `27e4964ef755c1055fd9159e860db2f7542952fac9e1237053ef3cdd1babc0f7` |
| `fejoint/contact.py` | `75885802990155b4634a8be58035658ead51510f0758ad1ce2d2778b5750db8a` |
| `fejoint/linsolve.py` | `a829f4f5d6becd7c1081db542f3ade923dc95ed7725527a2ad9f948a6eb6739a` |

A copy of these files is kept in `docs/G-C1.3_stamped_code`. The file `docs/G-C1.3_code.sha256` lists every hash above for `sha256sum -c`.

## Data

**Family** (`fejoint/family.py`)
- Eight parameters, each sampled uniformly within its range, all in mm: tp 8 to 22, eX 25 to 40, p 80 to 100, p23 190 to 220, w 80 to 110, overhang 0 to 15, LX 55 to 85, c_bot 20 to 45.
- Layouts that break the validity rules are rejected and redrawn.
  - Every bearing patch (radius 14.1 mm) clears the beam's flanges, its web and the plate's edges by at least 2 mm.
  - The rows are in order, and the plate is at least as wide as the flanges.
- An overhang below 2 mm is set to zero in the geometry; the features carry the drawn value.
- The beam, the column flange, the bolts (M20, no washers) and the load arm are FS1's. Every part takes $E = 210{,}000$ MPa and $\nu = 0.3$.

**Samples**
- 1,024 training members, seed 20261013.
- 128 held-out members, seed 20261014.
- The four tested layouts, FS1 to FS4, reported and not judged.
  - Each has its own plate and bolt layout from Table 2, with FS1's beam and the nominal moduli.
  - Their overhangs, a fraction of a millimetre below zero, are set to zero.
- The code refuses any seed used before the stamp.

**Export** (`fejoint/export.py`)
- The condensed nodal problem of each member: the bolt heads and the rigid end section are eliminated exactly.
- The non-negative unknowns are $u_x$ on the column face and the separation slots of the bearing patches.
- The pilot's held-out members had 2,172 to 2,406 nodes.
- Exported problems reproduce `fejoint.joint.analyse` within $10^{-9}$, the tests' tolerance. This holds for a tested layout and for a family member, with the full beam and with the 300 mm segment. The differences observed are below $10^{-11}$.

## Model and training

**Network**
- The repository's encoder (`fejepa.models.fejepa.build_encoder`, pinned above): width 256, depth 8, 8 heads, with its activation checkpointing.
- A copy of its field decoder, last layer zeroed at the start.
- Per-node features (23): coordinates, Dirichlet flags, load, node flags and family parameters.
- Output times the export's displacement scale, which is computed without any solve.
- Softplus at $T = 10^{-3}$ on the non-negative unknowns.

**Training**, as the repository's pretraining loop:
- AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, gradient clipping at 1.0, for every arm. The repository's supervised default of $1.5 \times 10^{-3}$ is not used, so that the arms differ only in their loss and their members.
- The repository's cosine schedule with 5% warm-up, over each run's own steps.
- One instance per step, in a fresh random order each epoch.
- 200 epochs for every arm.
- Float32 weights and activations, with float32 matrix products at the precision "highest", which turns TF32 off in cuBLAS. The attention kernel is the one torch selects for float32. The repository's runner turns TF32 on; this run does not.
- A non-finite loss stops the run, and a run whose predictions are not finite records that instead of predictions. Either stop is final and is not retried.
- Every attempt of a run is logged in `attempts.log` in its run folder: its start, any resume and its end.

**Arms**, three seeds each (0, 1, 2):

| Arm | Loss | Training members used | Labels | Steps per seed |
|---|---|---|---|---|
| label_free | energy over its label-free scale | 1,024 | 0 | 204,800 |
| supervised_16 | mean squared error in units of the scale | the first 16 | 16 | 3,200 |
| supervised_64 | the same | the first 64 | 64 | 12,800 |
| supervised_256 | the same | the first 256 | 256 | 51,200 |
| supervised_1024 | the same | 1,024 | 1,024 | 204,800 |

**Budget**
- 1,430,400 steps in all.
- The bench (section 3 of the runbook) times the full model on the GPU host before any training. It times label-free steps on the first 8 training members: 5 warm-up steps, then 40 timed.
- At 0.25 s a step the training would take about 99 hours.

## Criteria (on the label-free arm, held-out members)

**C1.** The median over the three seeds of the per-seed median of $\lvert S_{pred}/S_{ref} - 1\rvert$ is at most 0.05.
- $S$ is the paper's initial stiffness from the deflection at DT1 (Girão Coelho et al. 2004, eqs 2 to 5). It is read from the predicted field and from the exact solution of the same problem.
- With the segment, the deflection at DT1 is that of the rigid end section, carried to DT1 by beam theory.

**C2.** The median over the three seeds of the per-seed median contact overlap is at least 0.9.
- The overlap is the intersection over union of the column-face nodes with $u_x \le 10^{-3} u_{scale}$, in the prediction and in the reference alike.

**Verdict.** G-C1.3 passes if and only if C1 and C2 hold.
- A label-free seed whose run stopped on a non-finite loss, or whose predictions were not finite, is final. It enters the medians over seeds with $\lvert S_{pred}/S_{ref} - 1\rvert$ infinite and an overlap of 0. A FAIL that such a seed decides is reported as a failure of training at these settings, with the seeds named.
- A label-free seed missing for any other reason makes the verdict INCOMPLETE.
- Supervised seeds do not enter the verdict. A missing or non-finite supervised seed is reported as such.

**Reported, not judged**
- For every arm and seed, on the held-out and the tested members:
  - the energy gap and the energy-norm error;
  - the relative errors of the physical displacements and of the von Mises stress;
  - zero-field screening, and the minimum of the non-negative unknowns.
- The field rescaled along its own ray, $\alpha = f^T u / u^T K u$: its stiffness error and energy gap. It needs no reference.
- The four tested layouts.
- The supervised arms against the label-free arm, with no winner set in advance.
- The metrics of every member, in `verdict.json`.

## Interpretation, fixed in advance

**PASS.** Label-free training of the project's transformer reproduced the reference model's initial stiffness within 5% and its contact zone with an overlap of at least 0.9.
- This holds in the median over 128 unseen layouts of this family and over three training seeds.
- It holds on the 10 mm mesh with the 300 mm segment, for the elastic response with frictionless contact.
- The gap of the reference model to the joint tests stands as stated.
- A PASS does not by itself change step 1 of the industrial summary. That step names a check against the measured initial stiffness, which the reference model misses by 2 to 3 times (G-C1a). Whether and how a PASS is reported there is for the owner to decide. Any wording says that the surrogate reproduces the reference model, not the tests, and gives that gap.

**FAIL on C2 only.** At this budget the label-free surrogate reads the stiffness of unseen layouts within 5%, but does not locate the contact zone to an overlap of 0.9.

**FAIL on C1 only.** It locates the contact zone but does not read the stiffness within 5%.

**FAIL on both.** At this budget it does neither.

**In every FAIL**
- The thresholds stay.
- The rescaled stiffness and the supervised arms are described, not substituted for the criteria.
- Step 1 of the industrial summary stays "To be developed".
- Any next stage is pre-specified anew.

**INCOMPLETE.** A label-free seed is missing for a reason other than a non-finite loss.
- It is resumed from its last checkpoint, or run again from the start if it has none, with the stamped code and configuration. Nothing else changes.
- Every attempt is logged in the run's `attempts.log`.
- While the verdict is INCOMPLETE, the evaluation withholds every label-free metric and forms no criterion value.
- If a resumed run fails again with the same error, stop and report. Any change then, as after a bench stop, is a change after the stamp.

**The supervised comparison** is descriptive. A supervised arm that reads $S$ better while its fields carry larger energy errors is reported as exactly that.

## Stop conditions

- **Checks.** A precondition, test, hash or count in the runbook differs from what the runbook states: stop and report before going on.
- **The bench.** Its mean exceeds 0.25 s a step, or any of its 45 losses is not finite: stop before any training, and report to the owner.
- **The labels.** The label check fails: stop before any training, and report. It fails if a member is not labelled exactly once, if a file differs from the manifest or the ledger, or if a worst optimality residual lies outside the limits in `c13/data.py`: gap and multiplier no lower than $-10^{-9}$, complementarity at most $10^{-9}$, the multipliers of separated unknowns and the residuals of unconstrained unknowns at most $10^{-6}$.
- **Errors.** A training run that ends with an error other than a non-finite stop: stop the training loop and report.
- **Time.** Training has not finished within twice the time the bench predicts: report to the owner. The runs go on unless the owner stops them.
  - A stop before the three label-free seeds have ended leaves the verdict INCOMPLETE until they are resumed.
  - A stop after that leaves the supervised seeds not run reported as missing; the verdict stands.
- Nothing else uses the GPU during the bench and the training.

## Procedure

`RUNBOOK_C13.md`, run on the GPU host, in its sections:
0. preconditions, the worktree, the hash check and the tests;
1. the configuration hash and the repository check;
2. generation, without solves;
3. the bench and its stop rule;
4. labels for the held-out, tested and training members, and their check;
5. the fifteen training runs, which resume if interrupted;
6. evaluation;
7. the return package.

## References

- Girão Coelho, A.M., Bijlaard, F.S.K., Simões da Silva, L. (2004). Experimental assessment of the ductility of extended end plate connections. *Engineering Structures* 26(9), 1185-1206. doi:10.1016/j.engstruct.2000.09.001
