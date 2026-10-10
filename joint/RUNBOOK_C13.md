# RUNBOOK: gate G-C1.3, commands in execution order

**What it runs.** The deciding run of `docs/PRESPEC_G-C1.3.md`: label-free training of the project's transformer on a family of extended end plate joints with contact, against supervised arms, judged on held-out joints.

**Roles.** The **box** is the GPU machine. The **repo** holds the code, the pre-specification and the records.

**When.** Only after wp9's session 2 has finished and its return package has been sent. Nothing else may use the GPU during the bench and the training.
- The run uses its own worktree, `~/autodl-tmp/FE-JEPA-c13`. wp9's checkout is not touched.

**Time and space.**
- Generation: about 10 min on the CPU.
- Bench: a few minutes.
- Labels: about 30 min on the CPU with PARDISO, about 2 h with scipy's solver.
- Training: fifteen runs, 1,430,400 steps in all. At the bench's time per step that is 20 h at 0.05 s and 40 h at 0.1 s. Above 0.25 s a step the run stops before training (section 3).
- Evaluation: about 15 min on the CPU.
- Disk: about 4 GB on the data disk. Host memory: about 15 GB at the peak.

Everything runs in ONE tmux session, named `joint`. Every block below starts with its own `cd` and variables, so it can be run again after a reconnection.

**Any error or traceback, or any output that differs from what a comment states: stop and report.**

## 0. Preconditions (box, once)

0a. The tmux session. On its own line, outside tmux; after a dropped connection, `tmux attach -t joint`:
```bash
tmux new -A -s joint
```

0b. The worktree (once; the first line must print `fresh`, otherwise stop and report):
```bash
test ! -e ~/autodl-tmp/c13 && echo fresh     # must print: fresh
cd ~/autodl-tmp/FE-JEPA
git fetch origin wp10-joint --tags
git worktree add ~/autodl-tmp/FE-JEPA-c13 prereg-c13 2>/dev/null || (cd ~/autodl-tmp/FE-JEPA-c13 && git checkout --detach prereg-c13)
mkdir -p ~/autodl-tmp/c13/records
```

0c. The checks and the suite (about 5 min):
```bash
cd ~/autodl-tmp/FE-JEPA-c13
R=~/autodl-tmp/c13/records
git describe --tags --match prereg-c13 2>&1 | tee -a $R/preconditions.log       # must print exactly: prereg-c13
git status --porcelain --untracked-files=no 2>&1 | tee -a $R/preconditions.log  # must print nothing
git rev-parse HEAD | tee $R/commit.txt
df -h ~/autodl-tmp | tail -n 1 | tee -a $R/preconditions.log                    # at least 10 GB free
pgrep -af "python.*(fejepa|c13/__main__)|spawn_main" | tee -a $R/preconditions.log   # must print nothing
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader | tee -a $R/preconditions.log   # about 0 %, under 1,000 MiB
python -c "import numpy, scipy, torch; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, torch.cuda.is_available())" | tee -a $R/preconditions.log   # torch 2.12.1+cu130 True
cd joint
python -I -c "import sys; sys.path.insert(0, '.'); from fejoint import linsolve; print('label solver:', 'pardiso' if linsolve._PARDISO is not None else 'scipy')" | tee -a $R/preconditions.log
sha256sum -c docs/G-C1.3_code.sha256 2>&1 | tee $R/sha256_check.log | grep -vc ': OK$'   # must print 0
FEJEPA_REPO=.. python -m pytest -q -p no:cacheprovider tests/test_hex8i.py tests/test_contact.py tests/test_family_export.py tests/test_c13.py 2>&1 | tee $R/pytest.log | tail -n 1   # "30 passed" and nothing failed
```
- The hash file lists the gate's code and the eight repository files it imports.
- The four test files are the ones this run depends on. They pass without pypardiso and without pyamg; the solver then falls back to SuperLU, which is fast at these sizes. The other test files belong to earlier gates.
- Warnings in the summary line are expected: an auxiliary reading of the joint model that the tests call and this gate does not use, and a notice from torch's sparse tensors.

## 1. Configuration (box; seconds)

```bash
cd ~/autodl-tmp/FE-JEPA-c13/joint
R=~/autodl-tmp/c13/records
python -I c13/__main__.py write-config --out $R/c13_config.json --repo .. 2>&1 | tee $R/config.log
```
- The first line must be `config sha256 727d9bde73e5129a3ea9d26b3d6f3ac93d1f7f19505cd9605000c601152bb11e`.
- Fifteen file hashes follow; they must equal the table in `docs/PRESPEC_G-C1.3.md`.
- Eight lines `repository ... checked` follow.

## 2. Generation (box; about 10 min; no solve)

```bash
cd ~/autodl-tmp/FE-JEPA-c13/joint
R=~/autodl-tmp/c13/records; D=~/autodl-tmp/c13/data
python -I c13/__main__.py generate --data $D 2>&1 | tee $R/generate.log
python -c "import json; m = json.load(open('$D/manifest.json')); print(m['n'], m['config_sha256'])"   # 1156 727d9bde73e5129a3ea9d26b3d6f3ac93d1f7f19505cd9605000c601152bb11e
```
If interrupted, run the block again: files already written for the same member and configuration are kept, and any other file is refused.

## 3. Bench and its stop rule (box; a few minutes; nothing else on the GPU)

```bash
cd ~/autodl-tmp/FE-JEPA-c13/joint
R=~/autodl-tmp/c13/records; D=~/autodl-tmp/c13/data
pgrep -af "python.*(fejepa|c13/__main__)|spawn_main" | tee -a $R/gpu_checks.log   # must print nothing
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader | tee -a $R/gpu_checks.log   # about 0 %
python -I c13/__main__.py bench --data $D --repo .. --out $R/bench.json 2>&1 | tee $R/bench.log
python -c "import json; b = json.load(open('$R/bench.json')); print('mean_s_per_step', round(b['seconds_per_step_mean'], 4), 'losses_finite', b['losses_finite'], 'peak_GiB', b['peak_memory_GiB'], 'predicted_h', round(1430400 * b['seconds_per_step_mean'] / 3600, 1), 'gpu', b['gpu'])"
```
- If `mean_s_per_step` exceeds 0.25, or `losses_finite` is False, stop and report: no training before the owner decides.
- `predicted_h` is the predicted training time. If the training has not finished within twice that time, report to the owner; the runs go on unless the owner stops them.

## 4. Labels (box; about 30 min on the CPU)

```bash
cd ~/autodl-tmp/FE-JEPA-c13/joint
R=~/autodl-tmp/c13/records; D=~/autodl-tmp/c13/data
for s in eval tested train:1024; do python -I c13/__main__.py label --data $D --split $s 2>&1 | tee -a $R/label.log; done
python -I c13/__main__.py check-labels --data $D 2>&1 | tee $R/check_labels.log
```
- The check prints one line with `"labels": 1156, "instances": 1156` and ending in `"ok": true`.
- If it ends otherwise, or stops with "labels fail the checks", stop and report: no training.
- If the label loop is interrupted, run the block again: labels already written are kept.

## 5. Training (box; the bulk of the time)

```bash
cd ~/autodl-tmp/FE-JEPA-c13/joint
R=~/autodl-tmp/c13/records; D=~/autodl-tmp/c13/data; U=~/autodl-tmp/c13/runs; SHA=727d9bde73e5129a3ea9d26b3d6f3ac93d1f7f19505cd9605000c601152bb11e
pgrep -af "python.*(fejepa|c13/__main__)|spawn_main" | tee -a $R/gpu_checks.log   # must print nothing
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader | tee -a $R/gpu_checks.log   # about 0 %
date | tee -a $R/train_times.log
set -o pipefail
for arm in label_free supervised_16 supervised_64 supervised_256 supervised_1024; do
  for seed in 0 1 2; do
    python -I c13/__main__.py train --data $D --runs $U --arm $arm --seed $seed --repo .. --config-sha $SHA 2>&1 | tee -a $R/train_${arm}_s${seed}.log || break 2
  done
done
set +o pipefail
date | tee -a $R/train_times.log
```
- Each run prints a start line and ends by printing its status and steps, for example `{"status": "finished", "steps": 204800}`.
- Each run checkpoints every 5 epochs. Running the block again resumes interrupted runs from their checkpoints; finished runs return at once.
- A run that stops on a non-finite loss, or whose predictions are not finite, records it in its `history.json`. That stop is final: running the block again leaves the run as it is, and the loop goes on to the next run.
- A run that ends with any other error stops the loop (`break 2`). Stop and report. Once the cause is known, running the block again resumes.

## 6. Evaluation (box; about 15 min on the CPU)

```bash
cd ~/autodl-tmp/FE-JEPA-c13/joint
R=~/autodl-tmp/c13/records; D=~/autodl-tmp/c13/data; U=~/autodl-tmp/c13/runs; SHA=727d9bde73e5129a3ea9d26b3d6f3ac93d1f7f19505cd9605000c601152bb11e
python -I c13/__main__.py evaluate --data $D --runs $U --config-sha $SHA 2>&1 | tee $R/evaluate.log
```
The last line prints the verdict, the two criterion values, and any missing or non-finite runs.

## 7. Return package (box)

```bash
cd ~/autodl-tmp/c13
cp data/manifest.json data/labels/ledger.json runs/verdict.json records/
mkdir -p records/labels records/runs
cp data/labels/eval_*.npz data/labels/tested_*.npz records/labels/
for d in runs/*_s*/; do n=$(basename $d); mkdir -p records/runs/$n; cp $d/history.json $d/attempts.log records/runs/$n/ 2>/dev/null; done
for s in 0 1 2; do cp runs/label_free_s$s/preds.npz records/runs/label_free_s$s/ 2>/dev/null; done
tar czf c13_return.tar.gz records
sha256sum c13_return.tar.gz | tee c13_return.sha256
ls -l c13_return.tar.gz      # about 20 MB
```
- Send `c13_return.tar.gz` and its SHA-256.
- It holds the logs, the configuration, the bench, the manifest, the label ledger, the verdict with the metrics of every member, every run's history and attempts, the label-free predictions, and the labels of the held-out and tested members.
- The instance files can be regenerated from the stamped code. Regenerated on the box they match the manifest's SHA-256; elsewhere, compare the arrays.
- Keep the data, the supervised predictions and the model states on the box until the verdict has been audited.
