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

The tables follow in `docs/G-C1.3_verdict_numbers.md`; they are appended below in the assembled verdict.
