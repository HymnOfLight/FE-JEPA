# Gate G-C1.3: verdict

**What was pre-specified**
- `docs/PRESPEC_G-C1.3.md`, configuration SHA-256 `727d9bde73e5129a3ea9d26b3d6f3ac93d1f7f19505cd9605000c601152bb11e`, stamped at 23:02 BST on 10 October 2026.
- The tag `prereg-c13`, made at 23:13 BST the same evening, points to commit `002d19e`. The repository code the run imports is pinned file by file (wp9-pool `92dfc93`).
- The operator worked from a Chinese translation of `RUNBOOK_C13.md`. `RUNBOOK_C13_addendum.md` records its three supplements: the remote address after the repository moved, a second fetch route, and a provenance line.

**Where the numbers come from**
- **The return package**: `c13_return.tar.gz`, SHA-256 `8b88839c9f452bbdd2e2e906d790d34505dc784bf862e5dbe055225326d6cb7b`, 16,107,228 bytes, packed on the host at 23:23 CST on 13 October 2026. The SHA-256 reported with it matches the file.
  - Its files are kept as received in `records/c13/return/`.
  - `records/c13/return_files.sha256` lists their hashes.
- **The verdict and every metric** come from `records/c13/return/verdict.json`, which the stamped `c13/evaluate.py` wrote on the GPU host.
- **The author's recomputation**: `audit/c13_recompute.py` (not under the stamp; written after the run) reproduces them. It regenerates the held-out and tested instances from the stamped code, re-solves seven of them, and recomputes every metric of the label-free arm from the returned predictions and labels. Its outputs are `records/c13/audit/recompute.json`, with the BLAS on 25 threads as on the host, and `recompute_blas2.json`, on 2.
- **The tables** at the end are generated from those records by `audit/c13_write_verdict.py`, also written after the run.

**Independent audit** (13 October 2026), with its own code:
- The return package has the SHA-256 reported, and its 198 files are those kept in `records/c13/return`, byte for byte.
- The run used the stamped commit `002d19e`: its configuration hash, its 15 code hashes and its 8 repository hashes are those of the stamp. The 30 tests passed, all 1,156 members were labelled once and passed the label check, and each of the 15 runs ran once, with no resume, to its configured number of steps.
- Every metric of the three label-free seeds on the 132 held-out and tested members equals `verdict.json` exactly, and so do C1 (0.285%), C2 (0.9440) and the verdict, PASS.
- The label-free predictions cover exactly the 128 held-out and 4 tested members. They are finite and keep every non-negative unknown at or above zero.
- The held-out members are the configured draws and share none with the training members. The 132 returned labels match the ledger and meet the optimality conditions to within 2.2e-12. Eight held-out members re-solved with the host's solver reproduce them bit for bit.
- The records show three departures from the runbook and its addendum, those listed under Deviations below. None of them can change a number.
- The audit found the draft of this verdict wrong in three places: on the cause of the differing instance files, on what rescaling does to the stiffness error, and in leaving out two reports that the pre-specification requires. It raised twelve smaller points of wording, rounding and tables. This text takes up all of them.

## Run record

**Host** (`provenance.log`, 00:48 CST on 13 October 2026; CST is UTC+8)
- `autodl-container-8j9m4jx891-ed8b652f`, the host of wp9's session 2 (`records/wp9/session2/provenance.txt` and `DEVIATIONS_W9.md`, W1, on wp9-pool).
- An RTX 5090 with NVIDIA driver 580.95.05; a Xeon Platinum 8470Q with 25 CPUs; a memory limit of 98,784,247,808 bytes.
- torch 2.12.1+cu130, numpy 2.4.6, scipy 1.18.0. pypardiso did not load, so the labels were solved with scipy's SuperLU.

**Preconditions** (sections 0 and 1 of the runbook)
- The worktree was at the tag and clean, with 34 GB free. pgrep found no other run of the gate's or the repository's code.
- All 23 hashes of `docs/G-C1.3_code.sha256` checked, and the 30 tests passed.
- The configuration hash was the stamped one, the fifteen code hashes those of the stamp, and the eight repository files checked.

**Generation and bench**
- 1,156 instances were generated.
- The bench took 0.0466 s a step on average. All 45 of its losses were finite, and it predicted 18.5 h of training, well inside the stop rule of 0.25 s a step.

**Labels**
- All 1,156 members were labelled once, and the label check passed.
- The worst residuals were 2.4e-12 or smaller in size, against limits of 1e-9 and 1e-6.

**Training**
- From 03:30:52 to 22:46:52 CST on 13 October 2026 (20:30:52 BST on 12 October to 15:46:52 BST on 13 October): 19.3 h, inside twice the prediction.
- Each of the fifteen runs ran once, from start to end, with no resume. The runs followed one another within 2 s, and each finished with its configured number of steps. No loss and no prediction was non-finite.

**Evaluation**
- Every run had its predictions. No run was missing and none was non-finite.

## Deviations

None of them changes a number of the run.

1. **The GPU reading before the run.**
   - In section 0c and before the bench, `nvidia-smi` reported a utilisation of 100% with 0 MiB of memory in use. The runbook's comment expected about 0%, and its rule is to stop and report on any output that differs from a comment.
   - The operator went on.
   - With 0 MiB in use, no process held memory on the GPU (inferred). The bench, at 0.0466 s a step, was no slower than the label-free training that followed the 0% reading: about 0.053 s a step in the median epoch, with epochs of 30 s to 58 s. So the records show no sign of a competing load. The 100% reading was most likely a reporting artefact (inferred).
2. **The provenance line ran once**, after section 0c, and not again before sections 3 and 5 as the addendum asked.
   - No record after 00:48 CST names the host or the driver: the bench and the histories name only the GPU model and torch. So the records cannot exclude a change of host before the bench or before training.
   - The fifteen runs followed one another within 2 s, so the training itself was not interrupted.
3. **A stray time stamp.** `train_times.log` holds a third time stamp, 23:11:25 CST, after the end of training.
   - No run has a second start in its log or its `attempts.log`, and `gpu_checks.log` holds only two readings, so the training block was not run again.
   - `evaluate.log` was written at 23:18 CST, so the stamp may mark the start of the evaluation (inferred).

The records show no other departure. Some outputs that the runbook checks by eye were not logged: the `fresh` line and the fetch of section 0b, the remote address of the addendum's first item, and the summary lines of sections 2 and 3. The files those lines read, the manifest and `bench.json`, hold the expected values.

## Verdict

**PASS. Both criteria hold, judged as pre-specified.**
- **C1**: the median over the three seeds of the per-seed median stiffness error is 0.285%, against a threshold of 5%. The seeds give 0.285%, 0.384% and 0.274%.
- **C2**: the median over the three seeds of the per-seed median contact overlap is 0.944, against a threshold of 0.9. The seeds give 0.945, 0.942 and 0.944.

**The recomputation**
- Every metric of the three label-free runs was recomputed from the returned predictions and labels, by the author and by the audit, each with its own code. All agree with the returned verdict to the last bit, and so do C1, C2 and the verdict.
- The author regenerated the held-out and tested instances from the stamped code, with numpy 2.4.4 and scipy 1.17.1 (the host had 2.4.6 and 1.18.0). The bytes of the instance files depend on the number of BLAS threads.
  - With the BLAS of numpy and scipy on 25 threads, as on the host, all 132 files equal the manifest's SHA-256 (`records/c13/audit/recompute.json`). The audit found the same on 8 and 24 threads. (`bench.json` records 25 threads for torch; the host's environment sets 25 for the BLAS, `DEVIATIONS_W9.md`, W3, on wp9-pool.)
  - On 2 threads only 6 do (`recompute_blas2.json`); the audit found the same on 1 to 5 threads. The thread count changes the last bits of a few small entries of the condensed stiffness: at most 1.5e-11, against a largest entry of about 1.6e7 (audit).
  - The metrics computed from either set agree with the returned ones to the last bit.
- Seven members (five held-out, two tested) re-solved by the author with the host's solver, scipy's SuperLU, on 25 threads reproduce the returned labels bit for bit, in the ledger's numbers of iterations. On 2 threads they agree to within 9e-15 of the largest displacement and 5e-15 in S.

## What this means

**As fixed in advance (PASS)**
- Label-free training of the project's transformer reproduced the reference model's initial stiffness within 5% and its contact zone with an overlap of at least 0.9.
- This holds in the median over 128 unseen layouts of this family and over three training seeds.
- It holds on the 10 mm mesh with the 300 mm segment, for the elastic response with frictionless contact.
- **The gap of the reference model to the joint tests stands.** The same model on G-C1a's fine mesh with the full beam is 2.05 to 2.99 times stiffer than the measured initial stiffness of the eight joint tests, and the reference itself 2.08 to 3.05 times. Why is not known.

**Against the expectation stated before the run**
- The pre-specification expected a FAIL, at least on C2, because the CPU pilot's contact overlap had ended at 0.77. It held C1 uncertain: the pilot's stiffness error had risen from 5.3% after 4,500 of its 6,000 steps to 7.9% at the end while its energy gap kept falling, and the run's model, training set and budget were larger, with AdamW and clipping.
- The run passed both criteria. The median stiffness error was about 17.5 times below its threshold, and the median overlap 0.044 above its threshold. In each seed 21 to 26 of the 128 held-out members had an overlap below 0.9.
- The pilot differed in its model (width 64, depth 2, 4 heads), its training set (64 members), its budget (6,000 steps), its optimiser and schedule (Adam, one-cycle, no clipping, instances drawn with replacement), its held-out set (16 members) and its seeds (one). Which of these accounts for the difference was not tested.
- The supervised arms were expected to read the stiffness more closely than the label-free arm, with far larger energy errors, and to miss the contact. Their energy errors were far larger and their overlaps lower in the median, in every arm. On the held-out members only the arm with 1,024 labels read the stiffness more closely, a little in the median and more in the 90th percentile; on the four tested layouts the arms with 256 and 1,024 labels did (below).

**The supervised arms** (descriptive, as pre-specified). The label-free arm used no solution.
- **On the held-out members**, with 1,024 labels, one exact solution per training member, the median stiffness error was 0.18% to 0.28%.
  - That overlaps the label-free arm's 0.27% to 0.38%, and is a little below it in the median over seeds: 0.26% against 0.29%.
  - The 90th percentile was 0.73% to 0.86%, against 1.11% to 1.33%.
- Its fields were further from the solution in energy and in contact:
  - an energy gap of 2.7% to 3.2%, against 0.09% to 0.21%;
  - an energy-norm error of 0.17 to 0.18, against 0.03 to 0.05;
  - a von Mises error of 9.5% to 10.7%, against 2.0% to 2.9%;
  - a contact overlap of 0.74 to 0.86, against about 0.94.
- On the held-out members, with 16, 64 or 256 labels every reported error was larger, and every overlap smaller, than the label-free arm's.
  - With 16 labels no field beat the zero field in energy, and with 64 labels 96% to 98% did.
  - With 256 labels all did, as all label-free fields did.
- **On the four tested layouts** the arms with 256 and 1,024 labels read the stiffness more closely than the label-free arm: a median abs S error over the twelve layout and seed pairs of 0.29% and 0.18%, against 0.70%.
  - Their energy gaps were 9.1% and 3.5%, against 0.34%, and their overlaps 0.80 and 0.86, against 0.96, in the median.

**Other readings** (reported, not judged)
- The label-free fields were within 0.4% to 0.5% of the solutions in displacement, and within 2% to 3% in von Mises stress, in the median.
- Every label-free field, held-out and tested, beat the zero field in energy.
- No non-negative unknown of any arm was negative. In the label-free fields 1.7% to 3.0% of them are exactly zero, where softplus underflows in float32.
- **Rescaling along the field's own ray**:
  - It left the energy gap unchanged to the reported precision, and lowered the median stiffness error from 0.29%, 0.38% and 0.27% to 0.15%, 0.31% and 0.14%.
  - The median of alpha was within 0.0003 of 1, and the median of abs(alpha - 1) was 0.0013 to 0.0017; its largest value was 0.038.
- **The four tested layouts**: over the three seeds, the label-free stiffness errors were 0.01% to 1.5% in size, and the contact overlaps 0.93 to 0.97. The tables give every arm.

**What it does not show**
- Anything about the joint tests, that is, the experiments.
- Plasticity, preload or friction.
- Other meshes, other beam lengths, or layouts outside this family.
- **How the worst members fare.** The criteria are medians.
  - One held-out member is 7.3% to 8.2% too stiff in every seed: eval_0058, a thin plate with a wide gauge.
  - In each seed, 21 to 26 of the 128 members have an overlap below 0.9. The lowest, 0.74 to 0.77, is that of eval_0012 in every seed.

## Consequences

- **The industrial summary** (as pre-specified). A PASS does not by itself change step 1 of the industrial summary. That step names a check against the measured initial stiffness, which the reference model misses by 2 to 3 times.
  - Whether and how this PASS is reported there is for the project owner to decide.
  - Any wording says that the surrogate reproduces the reference model, not the tests, and gives that gap.
- **The next stage** (plasticity, stage C2 of the C plan of 9 October 2026) will be pre-specified anew. The pre-specification sets this for a FAIL and says nothing of it for a PASS.
- **The gap to the joint tests** remains the open question of the reference model.

## Numbers


<!-- generated by audit/c13_write_verdict.py from records/c13/return; config 727d9bde73e5129a3ea9d26b3d6f3ac93d1f7f19505cd9605000c601152bb11e -->

### Criteria (label-free arm, 128 held-out members)

| Seed | Median of abs(S_pred / S_ref - 1) | Median contact overlap |
|---|---|---|
| 0 | 0.285% | 0.9453 |
| 1 | 0.384% | 0.9423 |
| 2 | 0.274% | 0.9440 |
| **Median over seeds** | **0.285%** (C1: at most 5%) | **0.9440** (C2: at least 0.9) |

C1 = True; C2 = True; verdict = **PASS**. Missing runs: 0; non-finite runs: 0. Recomputed by `audit/c13_recompute.py` from the returned predictions and labels: C1 0.285%, C2 0.9440, PASS.

### Every arm and seed, held-out members (reported)

Medians over the 128 members unless marked; p90 and p10 are the 90th and 10th percentiles.

| Run | abs S error | abs S error, p90 | Contact overlap | Overlap, p10 | Energy gap | Energy-norm error | Displacement error | Von Mises error | Zero field beaten | Non-negative minimum | alpha | abs S error, rescaled | Energy gap, rescaled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| label_free s0 | 0.29% | 1.11% | 0.945 | 0.878 | 0.10% | 0.031 | 0.42% | 2.0% | 100% | 0 | 1.0002 | 0.15% | 0.10% |
| label_free s1 | 0.38% | 1.33% | 0.942 | 0.863 | 0.21% | 0.046 | 0.55% | 2.9% | 100% | 0 | 0.9997 | 0.31% | 0.21% |
| label_free s2 | 0.27% | 1.12% | 0.944 | 0.881 | 0.09% | 0.029 | 0.40% | 2.0% | 100% | 0 | 0.9999 | 0.14% | 0.09% |
| supervised_16 s0 | 2.72% | 7.79% | 0.520 | 0.402 | 147.89% | 1.222 | 4.02% | 87.2% | 0% | 0 | 0.3995 | 968.58% | 59.51% |
| supervised_16 s1 | 3.10% | 7.23% | 0.446 | 0.281 | 208.38% | 1.452 | 4.45% | 108.9% | 0% | 4.3e-41 | 0.3209 | 5319.44% | 66.89% |
| supervised_16 s2 | 2.92% | 7.78% | 0.407 | 0.243 | 199.96% | 1.417 | 4.38% | 106.5% | 0% | 3.2e-41 | 0.3300 | 4000.18% | 65.71% |
| supervised_64 s0 | 1.60% | 4.02% | 0.531 | 0.414 | 34.59% | 0.590 | 1.98% | 36.9% | 96% | 0 | 0.7396 | 61.71% | 25.27% |
| supervised_64 s1 | 1.25% | 4.08% | 0.513 | 0.396 | 35.22% | 0.596 | 1.86% | 34.2% | 98% | 0 | 0.7358 | 62.84% | 25.83% |
| supervised_64 s2 | 1.44% | 3.88% | 0.645 | 0.514 | 44.33% | 0.671 | 2.03% | 38.8% | 98% | 0 | 0.6905 | 83.33% | 30.39% |
| supervised_256 s0 | 0.47% | 1.79% | 0.776 | 0.680 | 10.05% | 0.319 | 0.87% | 19.1% | 100% | 0 | 0.9065 | 16.19% | 9.10% |
| supervised_256 s1 | 0.47% | 1.53% | 0.787 | 0.664 | 9.09% | 0.303 | 0.93% | 18.5% | 100% | 0 | 0.9160 | 14.07% | 8.29% |
| supervised_256 s2 | 0.51% | 1.51% | 0.735 | 0.595 | 8.63% | 0.296 | 0.86% | 17.8% | 100% | 0 | 0.9185 | 14.18% | 7.95% |
| supervised_1024 s0 | 0.18% | 0.73% | 0.833 | 0.723 | 2.71% | 0.166 | 0.40% | 9.5% | 100% | 0 | 0.9733 | 4.16% | 2.65% |
| supervised_1024 s1 | 0.26% | 0.81% | 0.745 | 0.662 | 2.90% | 0.170 | 0.43% | 9.9% | 100% | 0 | 0.9719 | 4.39% | 2.82% |
| supervised_1024 s2 | 0.28% | 0.86% | 0.860 | 0.762 | 3.25% | 0.182 | 0.47% | 10.7% | 100% | 0 | 0.9679 | 5.05% | 3.16% |

The non-negative minimum is the smallest non-negative unknown over the 128 members, in units of u_scale: none is negative.

### The four tested layouts, label-free arm (reported, not judged)

| Layout | S error, seeds 0 / 1 / 2 | Contact overlap, seeds 0 / 1 / 2 | Energy gap, seeds 0 / 1 / 2 |
|---|---|---|---|
| FS1 | -0.01% / +0.65% / +0.81% | 0.967 / 0.951 / 0.967 | 0.49% / 0.57% / 0.49% |
| FS2 | +0.08% / +0.51% / +0.75% | 0.967 / 0.967 / 0.967 | 0.09% / 0.19% / 0.10% |
| FS3 | +0.01% / +0.56% / +0.87% | 0.932 / 0.945 / 0.945 | 0.08% / 0.20% / 0.08% |
| FS4 | +0.96% / +1.33% / +1.48% | 0.949 / 0.949 / 0.966 | 0.55% / 0.67% / 0.57% |

### The four tested layouts, every arm and seed (reported, not judged)

| Run | Layout | S error | Contact overlap | Energy gap | Energy-norm error | Displacement error | Von Mises error | Zero field beaten | Non-negative minimum | alpha | S error, rescaled | Energy gap, rescaled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| label_free s0 | FS1 | -0.01% | 0.967 | 0.49% | 0.070 | 0.44% | 4.3% | yes | 0 | 0.9950 | +0.65% | 0.48% |
| label_free s0 | FS2 | +0.08% | 0.967 | 0.09% | 0.030 | 0.35% | 2.1% | yes | 0 | 0.9996 | +0.14% | 0.09% |
| label_free s0 | FS3 | +0.01% | 0.932 | 0.08% | 0.028 | 0.26% | 1.8% | yes | 0 | 0.9992 | +0.13% | 0.08% |
| label_free s0 | FS4 | +0.96% | 0.949 | 0.55% | 0.075 | 1.01% | 4.5% | yes | 0 | 1.0018 | +0.73% | 0.55% |
| label_free s1 | FS1 | +0.65% | 0.951 | 0.57% | 0.076 | 0.76% | 4.7% | yes | 0 | 0.9991 | +0.77% | 0.57% |
| label_free s1 | FS2 | +0.51% | 0.967 | 0.19% | 0.044 | 0.59% | 3.1% | yes | 0 | 1.0016 | +0.29% | 0.19% |
| label_free s1 | FS3 | +0.56% | 0.945 | 0.20% | 0.045 | 0.57% | 3.0% | yes | 0 | 1.0015 | +0.33% | 0.20% |
| label_free s1 | FS4 | +1.33% | 0.949 | 0.67% | 0.082 | 1.38% | 5.1% | yes | 0 | 1.0034 | +0.89% | 0.67% |
| label_free s2 | FS1 | +0.81% | 0.967 | 0.49% | 0.071 | 0.81% | 4.3% | yes | 0 | 1.0011 | +0.66% | 0.49% |
| label_free s2 | FS2 | +0.75% | 0.967 | 0.10% | 0.031 | 0.72% | 2.2% | yes | 0 | 1.0041 | +0.14% | 0.09% |
| label_free s2 | FS3 | +0.87% | 0.945 | 0.08% | 0.029 | 0.72% | 1.9% | yes | 0 | 1.0047 | +0.13% | 0.08% |
| label_free s2 | FS4 | +1.48% | 0.966 | 0.57% | 0.076 | 1.34% | 4.7% | yes | 0 | 1.0056 | +0.74% | 0.57% |
| supervised_16 s0 | FS1 | +2.48% | 0.586 | 242.76% | 1.566 | 4.49% | 106.1% | no | 0 | 0.2872 | +1843.67% | 70.98% |
| supervised_16 s0 | FS2 | +2.55% | 0.649 | 133.38% | 1.162 | 2.11% | 80.4% | no | 1.9e-29 | 0.4238 | +582.57% | 57.52% |
| supervised_16 s0 | FS3 | +1.45% | 0.550 | 142.82% | 1.201 | 2.34% | 82.4% | no | 3.7e-18 | 0.4073 | +1529.19% | 58.70% |
| supervised_16 s0 | FS4 | +4.57% | 0.507 | 232.48% | 1.532 | 5.57% | 102.3% | no | 0 | 0.2944 | +1381.17% | 70.77% |
| supervised_16 s1 | FS1 | -0.20% | 0.356 | 443.01% | 2.113 | 4.33% | 153.9% | no | 5.3e-34 | 0.1827 | -1362.04% | 80.76% |
| supervised_16 s1 | FS2 | +2.20% | 0.402 | 191.79% | 1.392 | 2.82% | 100.5% | no | 3.9e-14 | 0.3379 | +4061.61% | 65.74% |
| supervised_16 s1 | FS3 | +3.21% | 0.582 | 174.98% | 1.328 | 3.57% | 88.6% | no | 7.4e-15 | 0.3580 | -4762.72% | 63.85% |
| supervised_16 s1 | FS4 | +2.09% | 0.356 | 378.50% | 1.952 | 4.33% | 146.0% | no | 1.0e-35 | 0.2053 | -2608.71% | 78.91% |
| supervised_16 s2 | FS1 | +0.59% | 0.284 | 688.13% | 2.635 | 4.09% | 161.0% | no | 1.0e-38 | 0.1253 | -740.08% | 86.53% |
| supervised_16 s2 | FS2 | +1.95% | 0.434 | 183.55% | 1.359 | 2.53% | 104.2% | no | 1.7e-12 | 0.3480 | +2433.20% | 64.71% |
| supervised_16 s2 | FS3 | +2.81% | 0.494 | 170.21% | 1.305 | 3.19% | 98.5% | no | 4.9e-11 | 0.3647 | -10316.25% | 63.14% |
| supervised_16 s2 | FS4 | +3.10% | 0.258 | 409.91% | 2.032 | 4.06% | 140.6% | no | 3.5e-39 | 0.1917 | -1788.77% | 80.38% |
| supervised_64 s0 | FS1 | -2.75% | 0.581 | 53.77% | 0.737 | 4.19% | 49.9% | yes | 0 | 0.6438 | +82.60% | 33.87% |
| supervised_64 s0 | FS2 | -0.44% | 0.526 | 27.48% | 0.527 | 1.37% | 34.0% | yes | 0 | 0.7808 | +46.48% | 21.39% |
| supervised_64 s0 | FS3 | +0.16% | 0.494 | 34.28% | 0.589 | 1.25% | 33.8% | yes | 0 | 0.7420 | +68.70% | 25.46% |
| supervised_64 s0 | FS4 | -2.14% | 0.573 | 48.17% | 0.697 | 3.37% | 45.6% | yes | 0 | 0.6688 | +71.53% | 31.67% |
| supervised_64 s1 | FS1 | -2.75% | 0.451 | 54.57% | 0.741 | 4.08% | 45.5% | yes | 0 | 0.6405 | +84.08% | 34.20% |
| supervised_64 s1 | FS2 | -1.01% | 0.443 | 30.15% | 0.551 | 1.82% | 33.8% | yes | 0 | 0.7631 | +51.25% | 22.85% |
| supervised_64 s1 | FS3 | -0.04% | 0.473 | 31.67% | 0.566 | 1.22% | 31.8% | yes | 0 | 0.7564 | +62.07% | 23.95% |
| supervised_64 s1 | FS4 | -1.64% | 0.423 | 52.01% | 0.724 | 3.08% | 42.3% | yes | 0 | 0.6529 | +79.04% | 33.53% |
| supervised_64 s2 | FS1 | -2.71% | 0.580 | 63.15% | 0.799 | 3.66% | 53.4% | yes | 0 | 0.6075 | +100.64% | 37.55% |
| supervised_64 s2 | FS2 | -0.99% | 0.644 | 36.45% | 0.607 | 1.81% | 35.8% | yes | 0 | 0.7279 | +64.17% | 26.35% |
| supervised_64 s2 | FS3 | -0.15% | 0.670 | 44.63% | 0.673 | 1.27% | 36.9% | yes | 0 | 0.6877 | +96.23% | 30.67% |
| supervised_64 s2 | FS4 | -1.49% | 0.580 | 60.11% | 0.779 | 2.56% | 51.2% | yes | 0 | 0.6203 | +94.35% | 36.87% |
| supervised_256 s0 | FS1 | -0.15% | 0.824 | 14.99% | 0.389 | 1.45% | 23.5% | yes | 0 | 0.8678 | +20.96% | 12.99% |
| supervised_256 s0 | FS2 | -0.32% | 0.838 | 7.12% | 0.269 | 0.74% | 15.1% | yes | 0 | 0.9309 | +10.89% | 6.61% |
| supervised_256 s0 | FS3 | -0.16% | 0.800 | 7.02% | 0.267 | 0.65% | 15.7% | yes | 0 | 0.9325 | +11.68% | 6.54% |
| supervised_256 s0 | FS4 | +1.28% | 0.806 | 14.61% | 0.384 | 1.08% | 21.6% | yes | 0 | 0.8779 | +20.52% | 12.94% |
| supervised_256 s1 | FS1 | -0.00% | 0.783 | 11.12% | 0.335 | 0.96% | 19.0% | yes | 0 | 0.8989 | +15.41% | 9.99% |
| supervised_256 s1 | FS2 | -0.03% | 0.806 | 6.74% | 0.261 | 0.57% | 15.1% | yes | 0 | 0.9358 | +10.33% | 6.30% |
| supervised_256 s1 | FS3 | +0.04% | 0.818 | 6.36% | 0.254 | 0.55% | 16.2% | yes | 0 | 0.9395 | +10.56% | 5.98% |
| supervised_256 s1 | FS4 | +1.54% | 0.768 | 11.76% | 0.345 | 1.50% | 19.3% | yes | 0 | 0.9022 | +16.42% | 10.72% |
| supervised_256 s2 | FS1 | -0.97% | 0.694 | 12.25% | 0.352 | 1.12% | 21.2% | yes | 0 | 0.8846 | +16.79% | 10.75% |
| supervised_256 s2 | FS2 | -0.38% | 0.758 | 6.47% | 0.256 | 0.56% | 15.5% | yes | 0 | 0.9362 | +9.86% | 6.04% |
| supervised_256 s2 | FS3 | -0.48% | 0.863 | 6.02% | 0.247 | 0.58% | 14.9% | yes | 0 | 0.9397 | +9.93% | 5.64% |
| supervised_256 s2 | FS4 | +0.27% | 0.681 | 12.22% | 0.351 | 0.94% | 20.1% | yes | 0 | 0.8916 | +16.78% | 10.92% |
| supervised_1024 s0 | FS1 | -1.04% | 0.864 | 4.39% | 0.211 | 1.12% | 12.4% | yes | 0 | 0.9506 | +5.85% | 4.14% |
| supervised_1024 s0 | FS2 | +0.40% | 0.922 | 2.19% | 0.149 | 0.42% | 8.2% | yes | 0 | 0.9809 | +3.29% | 2.15% |
| supervised_1024 s0 | FS3 | -0.03% | 0.875 | 2.05% | 0.144 | 0.30% | 8.6% | yes | 0 | 0.9793 | +3.33% | 2.01% |
| supervised_1024 s0 | FS4 | +0.17% | 0.862 | 4.49% | 0.213 | 0.49% | 11.6% | yes | 0 | 0.9578 | +6.01% | 4.31% |
| supervised_1024 s1 | FS1 | -0.96% | 0.735 | 4.99% | 0.224 | 1.00% | 13.9% | yes | 0 | 0.9457 | +6.66% | 4.68% |
| supervised_1024 s1 | FS2 | -0.06% | 0.800 | 2.44% | 0.156 | 0.46% | 9.4% | yes | 0 | 0.9755 | +3.65% | 2.38% |
| supervised_1024 s1 | FS3 | +0.16% | 0.859 | 2.40% | 0.155 | 0.39% | 8.7% | yes | 0 | 0.9772 | +3.89% | 2.34% |
| supervised_1024 s1 | FS4 | -0.13% | 0.773 | 4.69% | 0.218 | 0.53% | 13.8% | yes | 0 | 0.9540 | +6.24% | 4.47% |
| supervised_1024 s2 | FS1 | -0.77% | 0.877 | 4.70% | 0.218 | 1.20% | 12.3% | yes | 0 | 0.9497 | +6.28% | 4.44% |
| supervised_1024 s2 | FS2 | +0.35% | 0.933 | 2.53% | 0.160 | 0.43% | 9.9% | yes | 0 | 0.9773 | +3.81% | 2.48% |
| supervised_1024 s2 | FS3 | +0.15% | 0.946 | 2.42% | 0.157 | 0.37% | 8.9% | yes | 0 | 0.9770 | +3.92% | 2.36% |
| supervised_1024 s2 | FS4 | +0.19% | 0.846 | 4.47% | 0.213 | 0.57% | 12.0% | yes | 0 | 0.9582 | +5.97% | 4.29% |

### Run record

- Bench: 0.0466 s a step on average (median 0.0500, largest 0.0533), 45 losses all finite: True; peak memory 0.21 GiB; predicted training time 18.5 h.
- Labels: 1156 of 1156, check ok: True; worst residuals: min_gap 0.0e+00, min_lambda -4.3e-13, max_complementarity 2.7e-13, max_lambda_inactive 4.3e-13, max_residual_unconstrained 2.4e-12.
- Training: from 13 October 2026 03:30:52 to 13 October 2026 22:46:52 (CST, UTC+8), 19.3 h, one attempt per run:

| Run | Started (CST) | Hours | Steps | Seconds per epoch, median | Peak memory (GiB) |
|---|---|---|---|---|---|
| label_free_s0 | 03:30:53 | 2.96 | 204,800 | 54.24 | 0.45 |
| label_free_s1 | 06:28:43 | 2.87 | 204,800 | 53.66 | 0.45 |
| label_free_s2 | 09:21:00 | 2.92 | 204,800 | 53.94 | 0.45 |
| supervised_16_s0 | 12:16:23 | 0.05 | 3,200 | 0.77 | 0.22 |
| supervised_16_s1 | 12:19:07 | 0.05 | 3,200 | 0.78 | 0.22 |
| supervised_16_s2 | 12:21:53 | 0.04 | 3,200 | 0.74 | 0.22 |
| supervised_64_s0 | 12:24:26 | 0.16 | 12,800 | 2.94 | 0.23 |
| supervised_64_s1 | 12:33:57 | 0.17 | 12,800 | 2.96 | 0.23 |
| supervised_64_s2 | 12:43:53 | 0.17 | 12,800 | 3.11 | 0.23 |
| supervised_256_s0 | 12:54:08 | 0.66 | 51,200 | 12.66 | 0.28 |
| supervised_256_s1 | 13:33:36 | 0.68 | 51,200 | 12.50 | 0.28 |
| supervised_256_s2 | 14:14:14 | 0.68 | 51,200 | 12.51 | 0.28 |
| supervised_1024_s0 | 14:54:46 | 2.58 | 204,800 | 47.99 | 0.47 |
| supervised_1024_s1 | 17:29:52 | 2.65 | 204,800 | 49.45 | 0.47 |
| supervised_1024_s2 | 20:08:55 | 2.63 | 204,800 | 49.33 | 0.47 |

Hours run from one start to the next and include loading the members and the predictions.

### The author's recomputation (`audit/c13_recompute.py`, records in `records/c13/audit/`)

- Run with Python 3.11.15, numpy 2.4.4 and scipy 1.17.1 (the host had numpy 2.4.6 and scipy 1.18.0); re-solves with scipy's SuperLU, as on the host.
- `recompute.json`, with the BLAS of numpy and scipy on 25 threads: 132 held-out and tested instances regenerated with the stamped code; parameters equal: True; node and nonzero counts equal: True; file SHA-256 equal to the manifest's for 132. Label-free metrics recomputed for every member and seed: largest relative difference to the returned values 0.0e+00 over 11 metrics; C1 0.285%, C2 0.9440, PASS.
- `recompute_blas2.json`, with the BLAS of numpy and scipy on 2 threads: 132 held-out and tested instances regenerated with the stamped code; parameters equal: True; node and nonzero counts equal: True; file SHA-256 equal to the manifest's for 6. Label-free metrics recomputed for every member and seed: largest relative difference to the returned values 0.0e+00 over 11 metrics; C1 0.285%, C2 0.9440, PASS.
- Every returned member is the configured draw: True; training and held-out members in common: 0.
- Returned labels match the ledger: True; the ledger's instance hashes match the manifest: True; solver: scipy.
- Re-solved 7 instances (eval_0000, eval_0031, eval_0064, eval_0097, eval_0127, tested_FS1, tested_FS4) in `recompute.json`: bit for bit equal to the returned labels 7 of 7; iterations equal to the ledger's: True; largest difference 0.0e+00 of the largest displacement, 0.0e+00 in S.
- Re-solved 7 instances (eval_0000, eval_0031, eval_0064, eval_0097, eval_0127, tested_FS1, tested_FS4) in `recompute_blas2.json`: bit for bit equal to the returned labels 0 of 7; iterations equal to the ledger's: True; largest difference 8.8e-15 of the largest displacement, 4.4e-15 in S.
- Label-free predictions: finite in every seed: True; smallest non-negative unknown 0, 0, 0; share of the non-negative unknowns that are exactly 0 (softplus underflow in float32) 3.0%, 1.7%, 1.8%.
