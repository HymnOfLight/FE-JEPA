# RUNBOOK -- fine3d (PREREG_FINE3D: R2a and R3), commands in execution order

Written 15 October 2026 (wp9-pool Stage 2, with the draft pre-registration).
PREREG_FINE3D.md is stamped and tagged (`prereg-fine3d`, an annotated tag)
BEFORE the box session; the session then runs R2a (S on the token bottleneck:
the fresh baseline and S), the post-hoc amplitude pass on both, and R3
(label-free fine-tuning of the Phase-2b states at the fine mesh), in one visit
and one tmux session, through `scripts/fine3d_session.sh`. In the sandbox, a
miniature laid out like the box (sizes scaled down, CPU) ran Sec. 0b-0d (0c
without the suite), the session script interrupted twice and restarted, 1',
the return and the adjudication (BRANCH_NOTES Stage 2); the suite on the GPU,
the GPU checks and the real corpora were not run there.

Roles: **box** = the GPU machine; **repo** = code, records, pre-registration,
adjudication.

Governance: the session writes only under `runs/fine3d/`. It reads the Phase-2b
states (`runs/phase2/e8_states/ar_p1024_s*.pt`) and the two corpora
(`runs/data3d_phase2`, `runs/data3d_phase2_fine`), whose manifests and labels it
does not change (the runs buy no labels: R2a's solve ledgers read 0, R3 solves
nothing). It touches no other run's directory. The branch is frozen from the
stamp until the session has returned: Sec. 0c checks that HEAD is exactly the
tagged commit, and the session script checks it again before every step.

Order with R1: when the same visit also runs R1 (branch cmame-paper,
RUNBOOK_CMAME Sec. E), R1 runs first, and its return is packed and sent before
this branch is checked out (the two branches share the checkout and `runs/`).
Do not shut the instance down after R1's return: this session follows at once.
R1's result does not gate this session.

Time: about 29 h -- R2a about 8.5 h per arm (E2's M = 1,024 arm: its three AR
units took 7 h 24 min, then about an hour went to P3's zero-shot pass; S's arm
somewhat longer), the two post-hoc passes about 20 and 12 min (the
first also meshes the 64 remesh instances: wp8's first pass took 542 s for
them), R3 about 3.5 h per seed (800 steps at the transformer's 14.8 s per fine
step, then its evaluations). The instance's balance must cover about 38 h (one
re-run of a step, Sec. 1c, included). Nothing else may use the GPU meanwhile.

## 0. Preconditions (box, once)

0a. The tmux session -- on its own line, outside tmux; after R1 in the same
visit, first leave R1's tmux session (Ctrl-b, then d; R1's session may stay);
after a dropped connection, `tmux attach -t f3`:
```bash
tmux new -A -s f3
```

0b. Fetch the branch and the tag, directly:
```bash
cd ~/autodl-tmp/FE-JEPA
git fetch origin +refs/heads/wp9-pool:refs/remotes/origin/wp9-pool +refs/tags/prereg-fine3d:refs/tags/prereg-fine3d && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short origin/wp9-pool              # the commit the operator instruction names
git rev-parse --short 'prereg-fine3d^{commit}'     # the same commit
```
Continue only if `FETCH-OK` is printed and both lines are the named commit. If
`FETCH-FAILED` is printed, nothing at all is printed for 10 min, or the fetch
has not ended after 20 min (then press Ctrl-C), take the bundle route: upload
the bundle the operator
instruction sends to `~/wp9-pool.bundle` and run, in place of the first block:
```bash
cd ~/autodl-tmp/FE-JEPA
sha256sum ~/wp9-pool.bundle                        # the SHA-256 the operator instruction names
git fetch ~/wp9-pool.bundle +refs/heads/wp9-pool:refs/remotes/origin/wp9-pool +refs/tags/prereg-fine3d:refs/tags/prereg-fine3d && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short origin/wp9-pool              # the commit the operator instruction names
git rev-parse --short 'prereg-fine3d^{commit}'     # the same commit
```
Continue only if the SHA-256 is the named one, `FETCH-OK` is printed and both
lines are the named commit; otherwise stop here and report. The explicit
refspecs matter: the box's clone is single-branch. (Repo side, before the
operator instruction is sent: `git ls-remote origin refs/heads/wp9-pool
'refs/tags/prereg-fine3d^{}'` shows the branch and the annotated tag at the
named commit; the bundle is made from the pushed branch and tag with `git
bundle create wp9-pool.bundle 92dfc93..wp9-pool prereg-fine3d`, the box
holding `92dfc93` since PREREG_W9's sessions, and `git bundle list-heads
wp9-pool.bundle` shows the branch at the named commit and the tag's own
object.)

0c. Checkout, checks and the suite (about 7 min):
```bash
cd ~/autodl-tmp/FE-JEPA
git checkout -B wp9-pool origin/wp9-pool           # "Switched to ..." or "Reset branch ..."
git describe --always --dirty --tags               # must print exactly: prereg-fine3d
git tag --points-at HEAD                           # must print exactly: prereg-fine3d
git rev-parse 'HEAD^{tree}'                        # the tree the operator instruction names
git status --porcelain --untracked-files=no        # must print nothing
df -h ~/autodl-tmp / | tail -n 2                   # >= 15 GB free on the data disk, >= 5 GB on /
pgrep -af "fejepa|spawn_main|holdout"              # must print nothing (R1 finished)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # about 0 %, under 1,000 MiB
grep oom_kill /sys/fs/cgroup/memory.events         # note the number: 1' compares with it
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.12.1+cu130 True
mkdir -p runs/fine3d/session
python -m pytest -q 2>&1 | tee runs/fine3d/session/pytest.log | tail -n 3   # "578 passed, 1 skipped"
```
Any output other than the comments describe: stop and report. The suite's last
line reads `578 passed, 1 skipped` where `pdflatex` is missing (the box: the
paper tables' LaTeX build is not compiled), `579 passed` where it is present.
Anything failed, an error, or another count: stop and report.

0d. The inputs and the dry runs (~3 min):
```bash
cd ~/autodl-tmp/FE-JEPA
python -m fejepa.cli info runs/data3d_phase2 | grep -E 'manifest_sha256|n_instances'
python -m fejepa.cli info runs/data3d_phase2_fine | grep -E 'manifest_sha256|n_instances|n_labelled'
sha256sum runs/phase2/e8_states/ar_p1024_s0.pt runs/phase2/e8_states/ar_p1024_s1.pt runs/phase2/e8_states/ar_p1024_s2.pt
for c in r2a_b_m1024 r2a_s_m1024; do
  python -m fejepa.cli run-config configs/$c.json --dry-run 2>&1 | grep -o '"prereg_status": "[^"]*"'
done
python scripts/r3_energy_finetune.py configs/r3_eft_b16.json --dry-run
```
Expected: `runs/data3d_phase2` with manifest `dc75628290bb...` and 2000
instances; `runs/data3d_phase2_fine` with manifest `0f859ad98a01...`, 320
instances, 320 labelled; the three states `a7b5c71c...`, `e9c0c400...`,
`e0f2210b...`; both dry runs `"prereg_status": "verified"`; R3's dry run one
JSON line with `"states_verified": 3`, 16 training files `instance_00256.npz`
to `instance_00271.npz`, `"n_fine": 256`, `"n_inband": 256` and `"prereg":
{"file": "PREREG_FINE3D.md", "config_sha256": ...}` with the hash of
PREREG_FINE3D's `CONFIG_SHA256[r3_eft_b16]` line. Anything else (a traceback,
`would refuse`, `null`, another hash or count): stop and report.

## 1. The session (box; about 29 h; the same tmux window)
```bash
cd ~/autodl-tmp/FE-JEPA
bash scripts/fine3d_session.sh
```
Steps, in order: `r2a_b_m1024` (about 8.5 h), `r2a_s_m1024` (about 8.5 h),
`amp_r2a_b_m1024` and `amp_r2a_s_m1024` (about 20 and 12 min), `r3_eft_b16`
(about 10.5 h). Normal signs:
- the script first prints the torch version and `cuda True`;
- each R2a log (`runs/fine3d/session/r2a_*.log`) starts with `[prereg] verified
  against PREREG_FINE3D.md`, prints the validation set's labelling lines (it
  is already labelled: no solve), a training line every ~15 min and one line
  per finished seed (about every 2.5 h); then P3: the fine set's labelling
  lines (no solve either) and its zero-shot pass, a line every 32 instances
  (`[p3] zero-shot ...`, a few minutes apart; about an hour in all, longer
  with S); it ends with a
  `gate G2` line that may read `passed=False` (ignore it: it is not a fine3d
  verdict) and the solve ledger reading `'total': 0`;
- each amplitude log prints one JSON line per seed and set, a line of seed
  means, then one line per seed for the remesh pass;
- R3's log prints its plan (JSON) and, per seed: the content check
  (`"passed": true`), a training line every 80 steps (about 20 min), a line
  every 32 instances while it evaluates, and the seed's readings; an
  interrupted seed resumes with `[ckpt] resumed`;
- at the end `status.txt` lists the five steps with `exit=0`.

### 1'. Health check (any time; a second tmux window: Ctrl-b c; back: Ctrl-b 0)
```bash
cd ~/autodl-tmp/FE-JEPA; date; tail -n 3 runs/fine3d/session/status.txt
pgrep -af "run-config|posthoc_amplitude|r3_energy" | cut -c1-120
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
find runs/fine3d -type f -mmin -45 -printf '%TH:%TM %p\n' | sort | tail -n 4
grep -h oom_kill /sys/fs/cgroup/memory.events 2>/dev/null; df -h ~/autodl-tmp / | tail -n 2
```
Healthy: one step process; the GPU in use (memory well above 1,000 MiB); a
file under `runs/fine3d/` written within the last 45 min (a log, a checkpoint,
a state or a record: during training a checkpoint at most every 5 min);
`oom_kill` unchanged since 0c. Otherwise: Ctrl-C in window 0, wait for the
prompt, and send this output plus `tail -n 60` of the running step's log; do
not restart.

### 1c. Interruptions and crashes
- A dropped connection: `tmux attach -t f3`; the session goes on.
- The box restarted or the script was stopped: open tmux again (0a), then
  `pgrep -af "run-config|posthoc_amplitude|r3_energy"` must print nothing;
  then `cd ~/autodl-tmp/FE-JEPA && bash scripts/fine3d_session.sh`. A finished
  step is skipped; an R2a arm with states but no report restarts with
  `--reuse-states` (finished seeds reused, an interrupted one resumed from its
  epoch checkpoint; recorded in its report); R3 skips a finished seed and
  resumes an interrupted one from its checkpoint; every attempt keeps its log
  and `status.txt` time-stamps each attempt. The script refuses to start while
  another copy runs.
- A step ends with `exit` other than 0 (the script goes on to the next step):
  when the script has finished, read `tail -n 30 runs/fine3d/session/<step>.log`.
  If it mentions `out of memory`, the exit is 137, or `oom_kill` rose: do not
  re-run; go to Sec. 2 and report. If R3 stopped with `do not reproduce the
  report's`: do not re-run; go to Sec. 2 and report. Otherwise re-run the
  script once as above; if the step fails again, go to Sec. 2 and report.
- `STOP before <step>: the code is ...` in `status.txt`: the checkout changed
  during the session; go to Sec. 2 and report.
- Never delete anything under `runs/fine3d/`.

## 2. Return (box; minutes)
```bash
cd ~/autodl-tmp/FE-JEPA
OUT=~/fine3d_return && rm -rf $OUT && mkdir -p $OUT/session $OUT/posthoc
cp runs/fine3d/session/status.txt runs/fine3d/session/*.log* $OUT/session/
for a in r2a_b_m1024 r2a_s_m1024; do
  [ -d runs/fine3d/$a ] || continue
  mkdir -p $OUT/$a; cp runs/fine3d/$a/report.json* runs/fine3d/$a/RESULTS*.md $OUT/$a/ 2>/dev/null
done
cp runs/fine3d/posthoc/amp_*.json $OUT/posthoc/ 2>/dev/null
mkdir -p $OUT/r3_eft_b16; cp runs/fine3d/r3_eft_b16/*.json* $OUT/r3_eft_b16/ 2>/dev/null
( echo "HEAD $(git rev-parse HEAD)"; echo "tree $(git rev-parse 'HEAD^{tree}')"
  echo "describe $(git describe --always --dirty --tags)"
  echo "tags $(git tag --points-at HEAD | tr '\n' ' ')"
  echo "--- git status --porcelain ---"; git status --porcelain | grep -v '^?? runs/' | head -40
  python -c "import torch, gmsh; print('torch', torch.__version__, torch.cuda.is_available(), 'gmsh', gmsh.__version__)"
  echo "--- reports ---"; sha256sum runs/fine3d/*/report.json runs/fine3d/posthoc/amp_*.json
  echo "--- states ---"; ls -l --full-time runs/fine3d/*/e8_states/ runs/fine3d/r3_eft_b16/
  sha256sum runs/fine3d/*/e8_states/ar_*.pt runs/fine3d/r3_eft_b16/*.pt runs/phase2/e8_states/ar_p1024_s*.pt
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
  echo "hostname $(hostname)"; echo "nproc $(nproc)"; lscpu | grep 'Model name'
  cat /sys/fs/cgroup/memory.max 2>/dev/null; cat /sys/fs/cgroup/memory.events 2>/dev/null
  df -h ~/autodl-tmp / | tail -n 2
) > $OUT/provenance.txt 2>&1
cd ~ && tar czf fine3d_return_$(date +%Y%m%d).tgz fine3d_return \
    && sha256sum fine3d_return_*.tgz && du -h fine3d_return_*.tgz
```
Send the tarball. The states stay on the box (their SHA-256 are in the
provenance file and in the reports). Then shut the instance down without
releasing it.

## 3. Adjudication (repo)
```bash
python scripts/adjudicate_fine3d.py \
    --r2a-b <return>/r2a_b_m1024/report.json --r2a-s <return>/r2a_s_m1024/report.json \
    --amp-b <return>/posthoc/amp_r2a_b_m1024.json --amp-s <return>/posthoc/amp_r2a_s_m1024.json \
    --r3 <return>/r3_eft_b16/report.json --status <return>/session/status.txt \
    --out records/wp9/fine3d/fine3d_verdict.json
```
It reads the Phase-2b report and E2's M = 1,024 report from `records/wp8/`
(each checked by SHA-256) and the stamped lines from PREREG_FINE3D.md, and
requires every report to have run on `prereg-fine3d`. R2a, the R2b
recommendation, R3-E and R3-V per PREREG_FINE3D Sec. 6, the secondary readings
of Sec. 7, refusals and deviations; the inputs' SHA-256, the session's attempts
and the adjudicating code (its files against the tag's) are recorded in the
verdict file. A part whose report is refused or not in the return is NOT
EVALUATED; the other is still adjudicated.
