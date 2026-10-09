# RUNBOOK -- wp9-pool, commands in execution order

Stage 0a (1 Oct 2026) wrote Sec. 1, session 1 -- readings, timing and data
generation; nothing is trained and nothing is adjudicated. Its outputs that
select what session 2 runs are read by the rules committed before it ran
(`scripts/w9_session1_decisions.py`; BRANCH_NOTES_wp9-pool.md, "Pre-declared
readings"). Stage 0b (1-2 Oct 2026) wrote Sec. 2-3 and 5: session 2 (C1 and,
if rule 1 admits it, S), the adjudication, and the optional torch-stack timing
(Sec. 1e of Stage 0a, moved after session 2). Stage 0d (2 Oct 2026) added
Sec. 4: the paper's cost table, surrogate inference against the direct solve
(nothing in PREREG_W9 reads it); Stage 0e (2 Oct 2026) added to it the CG
stopped at the surrogate's accuracy and the paper's field figures. Stage 0f
(9 Oct 2026, with the stamp): Sec. 4's timing (without the F5 family, which
session 1 makes) and field export ran on 8 October as RUNBOOK_CMAME Sec. A
(branch cmame-paper, records in its `records/cmame/timing/`), so Sec. 4 now
measures only what the paper's cost table still lacks, the threads its CPU
solvers used; Sec. 5 is withdrawn; Sec. 0b fetches directly, in one line,
with a git bundle as the other route; Sec. 2a states the memory reading
expected on the current host.
PREREG_W9.md is stamped and tagged
(`prereg-w9`) BEFORE session 1, covering every arm the rules can select, so
the box runs session 1 and session 2 in one visit: Sec. 0, session 1, its
return, the plan (it refuses to plan unless session 1 passed its gate), the
arms, their return. In the sandbox, a miniature laid out like the box (sizes
scaled down, CPU) ran Sec. 0's git checks, Sec. 1a-1d, 1f, 2a, 2b with a
worker killed mid-arm and the restart, 2d and Sec. 3 from the two returns
(the rules' inputs edited so that every arm runs; 1f and 2d with shorter
provenance blocks and no tarball; the suite, the GPU checks and Stage 0e's
Sec. 4-5 not run there; Sec. 4's scripts ran on small CPU runs in the tests).

Roles: **box** = the GPU machine; **repo** = code, records, pre-registration,
adjudication.

Governance: nothing here touches a stamped configuration of another
pre-registration, a run's `e8_states/` other than wp9's own, or `runs/data2d`;
session 1 writes under `runs/w9/`, session 2 writes under `runs/w9/` and reads
E1's states; Sec. 4 writes under `runs/w9/threads/` and reads E1's and
Phase-2b's states and corpora. The branch is frozen from the stamp until
session 2 has returned: the box checks that HEAD is exactly the tagged commit,
the plan checks it again, and the command script checks it before every arm.

Time and money: session 1 about 1 h; session 2 about 10 h without S,
about 13 h with S, 2-6 h more if N_max runs one seed at a time; Sec. 4 about
25 min. The instance's balance must cover about 24 h. The instance's host
changed after E1 and before RUNBOOK_CMAME's runs of 8-9 October: E1 ran on a
Xeon Platinum 8470Q host with 25 CPUs allotted (`os.cpu_count()` 208, so
E1's worker set-up gave each of three workers 69 torch threads); the
instance now runs on a Xeon Gold 6459C host with 16 CPUs allotted
(`os.cpu_count()` 128: 42 threads per worker) and 92 GiB of memory; the GPU,
its driver and torch are E1's. CM2D's 2D transformer steps ran there at
36-41 ms with three workers.
Everything runs in ONE tmux session; nothing else may use the GPU meanwhile.

## 0. Preconditions (box, once)

0a. The tmux session -- on its own line, outside tmux (skip it if a tmux
status bar is already showing); after a dropped connection, `tmux attach -t w9`:
```bash
tmux new -A -s w9
```

0b. Fetch the branch and the tag, directly:
```bash
cd ~/autodl-tmp/FE-JEPA
git fetch origin +refs/heads/wp9-pool:refs/remotes/origin/wp9-pool +refs/tags/prereg-w9:refs/tags/prereg-w9 && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short origin/wp9-pool        # the commit the operator instruction names
git rev-parse --short 'prereg-w9^{commit}'   # the same commit
```
Continue only if `FETCH-OK` is printed and both lines are the named commit.
If nothing at all is printed for 10 min, or the fetch has not ended after 20
min, press Ctrl-C (then neither line appears) and take the bundle route. The explicit refspecs matter: the box's
clone is single-branch, so a plain `git fetch` or `git pull` does not see a
new branch or tag. (On 8 October AutoDL's network route returned HTTP 503
while the direct fetch worked; later that day the direct fetch timed out and
a git bundle served.) On `FETCH-FAILED`, or after a Ctrl-C: upload the bundle
that the operator instruction sends to `~/wp9-pool.bundle` and run, in place
of the first block:
```bash
cd ~/autodl-tmp/FE-JEPA
sha256sum ~/wp9-pool.bundle                  # the SHA-256 the operator instruction names
git fetch ~/wp9-pool.bundle +refs/heads/wp9-pool:refs/remotes/origin/wp9-pool +refs/tags/prereg-w9:refs/tags/prereg-w9 && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short origin/wp9-pool        # the commit the operator instruction names
git rev-parse --short 'prereg-w9^{commit}'   # the same commit
```
Continue only if the SHA-256 is the named one, `FETCH-OK` is printed and both
lines are the named commit; otherwise stop here and report. (Repo side,
before the operator instruction is sent: `git ls-remote origin
refs/heads/wp9-pool 'refs/tags/prereg-w9^{}'` shows the branch and the
annotated tag at the named commit; the bundle is made from the pushed branch
and tag with `git bundle create wp9-pool.bundle 414a372..wp9-pool prereg-w9`,
Stage 0e being on the box since cmame-paper, whose history holds it, was
fetched there; `git bundle list-heads` shows the branch at the named commit
and the tag's own object, and a clone fetched from the bundle resolves
`prereg-w9^{commit}` to the named commit.)

0c. Checkout, checks and the suite (~5 min):
```bash
cd ~/autodl-tmp/FE-JEPA
git checkout -B wp9-pool origin/wp9-pool     # "Switched to ..." or "Reset branch ...", maybe "set up to track"
git describe --tags --match prereg-w9        # must print exactly: prereg-w9
git rev-parse 'HEAD^{tree}'                  # the tree the operator instruction names
git status --porcelain --untracked-files=no  # must print nothing
df -h ~/autodl-tmp / | tail -n 2             # >= 10 GB free on the data disk, >= 5 GB on / (the suite needs 5)
pgrep -af "fejepa|spawn_main"                # must print nothing
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # about 0 %, under 1,000 MiB
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.12.1+cu130 True
mkdir -p runs/w9/session1
python -m pytest -q 2>&1 | tee runs/w9/session1/pytest.log | tail -n 3   # "445 passed" (BRANCH_NOTES' count)
```
Any output other than the comments describe: stop and report. The suite's last
line reads `445 passed`, then warnings and the time; `444 passed, 1 skipped`
is also correct where `pdflatex` is missing (the paper tables' LaTeX build is
then not compiled). Anything failed, an error, or another count: stop and
report.

## 1. Session 1 (box; ~1 h; nothing else on the GPU)

Inputs: E1's base report (the records copy, `records/wp8/e1/e1_2d_base/report.json`,
byte-identical to the box's), its three states (`runs/e1_2d_base/e8_states/`) and
its corpus (`runs/data2d`). `w9_c0.py` refuses states whose SHA-256 is not the
one the report trained, a corpus whose manifest is not the report's, and an
evaluation family with a file that differs from its manifest. 1b's memory audit
and 1c measure memory and time: nothing else may run on the GPU. If the block
is interrupted, paste the whole block again: earlier logs and status files
are kept with a time suffix, 1a verifies the families it finished instead of
regenerating them, and every other step runs again.
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/w9/session1
[ -s $P/status.txt ] && mv $P/status.txt $P/status.txt.$(date +%Y%m%d-%H%M%S); : > $P/status.txt
run() { local log=$1; shift; [ -e $P/$log ] && mv $P/$log $P/$log.$(date +%Y%m%d-%H%M%S)
        "$@" 2>&1 | tee $P/$log; echo "$log exit=${PIPESTATUS[0]}" | tee -a $P/status.txt; }
R=records/wp8/e1/e1_2d_base/report.json; S=runs/e1_2d_base/e8_states
# 1a. OOD-2D v1 (F1-F5, 256 each), the remesh set R (16 geometries x 5 h) and the
#     in-band holdout IB (2,048 training-family instances): gmsh, labelled by direct
#     solve, manifests pin every file (CPU; ~5-10 min). F5 and R come first; a family
#     that fails is recorded and the others still run. A re-run verifies a family
#     that has a manifest instead of regenerating it.
run ood2d.log python scripts/w9_make_ood2d.py --out runs/w9/ood2d --record $P/ood2d.json
# 1b. C0 readings on E1's base states (~15-30 min)
run c0_val.log python scripts/w9_c0.py val --report $R --states-dir $S --out $P/c0_val.json
run c0_trainval.log python scripts/w9_c0.py trainval --report $R --states-dir $S \
    --out $P/c0_trainval.json
run c0_amp2d.log python scripts/w9_c0.py amp2d --report $R --states-dir $S \
    --family-dir runs/w9/ood2d/F5 --remesh-dir runs/w9/ood2d/R --out $P/c0_amp2d.json
run c0_memory.log python scripts/w9_c0.py memory --report $R --out $P/c0_memory.json
# 1c. 2D step timing with activation checkpointing on and off: E1's worker set-up
#     (three units at once) and in-process (~15-20 min)
run profile_w9.log python scripts/posthoc_profile_2d.py --variants worker3 worker3_no_ckpt \
    threads_w3 threads_w3_no_ckpt --no-sup --no-profile --out $P/profile_2d_w9.json
# 1d. the decisions (seconds)
run decisions.log python scripts/w9_session1_decisions.py --dir $P --out $P/decisions.json
cat $P/status.txt
```
`status.txt` holds seven lines. Note every line that does not end `exit=0`
for the report, but do not re-run a step with other arguments: the plan (2a)
decides whether session 2 can run. The C0 commands print one JSON summary
each; the memory audit prints one line per pool mark (1,024, 4,096, 12,800,
25,600; it stops early, and says so, if the process's residency passes 60% of
the host or the device memory) and then the step measurement.
`decisions.json` lists, per rule, the value and the numbers it read; a rule
whose input is missing or malformed is recorded as undecided (with the error),
the others are decided, and 1d then exits non-zero naming the undecided rules.
Dependencies: `amp2d` needs 1a's F5 and R; 1d needs 1b and 1c; nothing else
depends on another step.

### 1f. Return of session 1 (box; minutes)
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/w9/session1; S=runs/e1_2d_base/e8_states
OUT=~/wp9_session1_return && rm -rf $OUT && mkdir -p $OUT
cp $P/*.json $P/*.log* $P/status.txt* $OUT/
for f in IB F1 F2 F3 F4 F5 R; do mkdir -p $OUT/ood2d/$f; cp runs/w9/ood2d/$f/manifest.json $OUT/ood2d/$f/; done
( echo "HEAD $(git rev-parse HEAD)"; echo "tree $(git rev-parse 'HEAD^{tree}')"
  echo "describe $(git describe --tags --match prereg-w9)"
  echo "--- git status --porcelain ---"; git status --porcelain | grep -v '^?? runs/' | head -40
  echo "--- states ---"; sha256sum $S/ar_p1024_s*.pt
  python -c "import torch, gmsh; print('torch', torch.__version__, 'gmsh', gmsh.__version__)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
  echo "--- compute apps (may be empty inside a container) ---"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  echo "hostname $(hostname)"; echo "nproc $(nproc)"; grep -E 'MemTotal|MemAvailable' /proc/meminfo
  echo "--- cgroup ---"; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max 2>/dev/null
  echo "oom.group $(cat /sys/fs/cgroup/memory.oom.group 2>/dev/null)"
  cat /sys/fs/cgroup/memory.events 2>/dev/null; df -h ~/autodl-tmp / | tail -n 2
) > $OUT/provenance.txt 2>&1
cd ~ && tar czf wp9_session1_return_$(date +%Y%m%d).tgz wp9_session1_return \
    && sha256sum wp9_session1_return_*.tgz && du -h wp9_session1_return_*.tgz
```
Send the tarball now and go on with Sec. 2; do not shut down. The instances
under `runs/w9/ood2d/` stay on the box (the manifests in the return pin them);
session 2 evaluates on them. Do not delete or regenerate them.

## 2. Session 2 -- C1 (and S) training and evaluation (box; 10-19 h)

### 2a. Checks and the plan (minutes)
```bash
cd ~/autodl-tmp/FE-JEPA
cat runs/w9/session1/status.txt
pgrep -af "fejepa|spawn_main"                # must print nothing
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # about 0 %, under 1,000 MiB
grep oom_kill /sys/fs/cgroup/memory.events 2>/dev/null   # note the number: 2b' compares with it
cat /sys/fs/cgroup/memory.max                # the container's memory limit in bytes (see below)
df -h ~/autodl-tmp / | tail -n 2
python scripts/w9_session2_plan.py --session1 runs/w9/session1 --out-dir runs/w9/session2 \
    2>&1 | tee runs/w9/session2_plan.log
```
If a process is listed or the GPU is busy, stop and report. The plan prints
either `STOP` with the reasons, or the arms, any notes, one summary line and
`GO -- bash .../commands.sh`; anything else (a Python traceback) counts as
STOP. The gate: the required session-1 steps (ood2d, c0_val, c0_amp2d,
c0_memory) exited 0; the decision file is the frozen rules' output on the
readings beside it (recomputed here), written on `prereg-w9`, which this
checkout must also be; rules 1-2 decided and a pool found; the seven evaluation
sets complete and the manifests on disk the record's; E1's validation arrays
reproduced; a GPU and E1's torch; 5 GB free on the data disk; every selected
configuration passes its dry run (stamped, holdouts and E1's states
verified). It selects the stamped configurations (the 1,024 arm; N_max =
`w9_c1_n25600` or `w9_c1_n12800` by rule 2; the fresh baseline `w9_b_n1024`;
`w9_s_n1024` only if rule 1 admitted S; the 4,096 arm), sets workers (rule
2's for N_max, 3 otherwise) and activation checkpointing (rule 3; kept on for
an arm that would not fit without it, and when rule 3 is undecided), and
writes `runs/w9/session2/plan.json` and `commands.sh`.

The summary line reads `S admitted: yes|no | N_max ... with k worker(s) at a
time | checkpointing on|off | usable host memory X GB (source) | E1
reproduction d`. X must be the number `cat /sys/fs/cgroup/memory.max` printed
above, divided by 1e9 and given to one decimal, with the source
`cgroup_limit` (on the host of 8-9 October 98,784,247,808 bytes, 92 GiB, so
`98.8 GB (cgroup_limit)`); otherwise treat it as STOP (the memory audit
would have measured against something other than the container's limit). A
limit other than 98,784,247,808 bytes means the instance changed host
again: report it with the plan's output. On STOP: send the STOP lines and
`runs/w9/session2_plan.log`; then Sec. 4 (its own checks decide whether it
can run); then shut the instance down without releasing it.

### 2b. The runs (hours; the same tmux window)
```bash
cd ~/autodl-tmp/FE-JEPA
bash runs/w9/session2/commands.sh
```
Order: the 1,024 arm (E1's states, evaluation only; 10-30 min), N_max, the
fresh baseline, S (if it was admitted), then 4,096. Normal signs:
- the script first prints the torch version, E1's, and `cuda True`;
- each arm's log starts with `[prereg] verified against PREREG_W9.md`; with
  three workers nothing new is printed for about 3 h (one line per finished
  seed); N_max at 25,600 prepares its instances for minutes first; N_max one
  seed at a time takes about 5-9 h and prints training progress;
- each arm ends with a `gate G1'` line that may read `passed=False` (ignore
  it: it is not a wp9 verdict) and the solve ledger reading `'total': 0`;
- at the end `status.txt` lists every arm of `plan.json` with `exit=0`.

### 2b'. Health check (any time; a second tmux window: Ctrl-b c; back: Ctrl-b 0)
```bash
cd ~/autodl-tmp/FE-JEPA; date; tail -n 3 runs/w9/session2/status.txt
pgrep -af "fejepa.cli run-config" | cut -c1-120; echo "workers: $(pgrep -fc spawn_main)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
find runs/w9 -path '*/e8_states/*' -mmin -60 -printf '%TH:%TM %f\n' | sort | tail -n 6
grep -h oom_kill /sys/fs/cgroup/memory.events 2>/dev/null; df -h ~/autodl-tmp / | tail -n 2
```
Healthy: one `run-config` process (the running arm); `workers: 3`
(`workers: 0` while N_max runs one seed at a time: it trains in that process
itself); the GPU in use (memory well above 1,000 MiB, utilization mostly above
0 %); `oom_kill` unchanged since 2a; a state file written within the last hour
(not expected in an arm's first 90 min or during the 1,024 arm). Otherwise:
Ctrl-C in window 0, wait for the prompt, and send this output plus `tail -n 60`
of the running arm's log (`runs/w9/session2/<arm>.log`); do not restart.

### 2c. Interruptions and crashes
- A dropped connection: `tmux attach -t w9`; the runs go on.
- The box restarted or the script was stopped: open tmux again (0a), then
  `pgrep -af fejepa.cli` must print nothing; then
  `cd ~/autodl-tmp/FE-JEPA && bash runs/w9/session2/commands.sh`. A finished
  arm is skipped; an arm with states but no complete report restarts with
  `--reuse-states` (finished seeds reused, an interrupted one resumed from its
  epoch checkpoint; recorded in its report); every attempt keeps its log
  (`<arm>.log.<time>`) and `status.txt` time-stamps each attempt. The script
  refuses to start while another copy runs.
- An arm ends with `exit` other than 0 (the script goes on to the next arm):
  when the script has finished, read `tail -n 30 runs/w9/session2/<arm>.log`.
  If it mentions `out of memory` or `a worker process died`, or the exit is
  137, or `oom_kill` rose: do not re-run; go to 2d and report. Otherwise re-run
  the script once as above; if the arm fails again, go to 2d and report.
- `STOP before <arm>: the code is ...` in `status.txt`: the checkout changed
  during the session; go to 2d and report.
- Never delete an arm's `e8_states/`, report or log.

### 2d. Return of session 2 (box)
```bash
cd ~/autodl-tmp/FE-JEPA
OUT=~/wp9_session2_return && rm -rf $OUT && mkdir -p $OUT
cp runs/w9/session2/plan.json runs/w9/session2/commands.sh runs/w9/session2/status.txt \
    runs/w9/session2/*.log* runs/w9/session2_plan.log runs/w9/session1/decisions.json $OUT/
for a in c1_n1024 c1_n4096 c1_n25600 c1_n12800 b_n1024 s_n1024; do
  [ -d runs/w9/$a ] || continue
  mkdir -p $OUT/$a; cp runs/w9/$a/report.json* runs/w9/$a/RESULTS*.md $OUT/$a/ 2>/dev/null
done
( echo "HEAD $(git rev-parse HEAD)"; echo "tree $(git rev-parse 'HEAD^{tree}')"
  echo "describe $(git describe --tags --match prereg-w9)"
  echo "--- git status --porcelain ---"; git status --porcelain | grep -v '^?? runs/' | head -40
  python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
  echo "--- reports ---"; sha256sum runs/w9/*/report.json
  echo "--- states ---"; ls -l --full-time runs/w9/*/e8_states/
  sha256sum runs/w9/*/e8_states/ar_*.pt runs/e1_2d_base/e8_states/ar_*.pt
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
  echo "hostname $(hostname)"; echo "nproc $(nproc)"
  echo "oom.group $(cat /sys/fs/cgroup/memory.oom.group 2>/dev/null)"
  cat /sys/fs/cgroup/memory.events 2>/dev/null; df -h ~/autodl-tmp / | tail -n 2
) > $OUT/provenance.txt 2>&1
cd ~ && tar czf wp9_session2_return_$(date +%Y%m%d).tgz wp9_session2_return \
    && sha256sum wp9_session2_return_*.tgz && du -h wp9_session2_return_*.tgz
```
Send the tarball. The states stay on the box (their SHA-256 are in the
provenance file and in each report); so do the evaluation sets. Then Sec. 4
(the threads of the paper's solvers); then shut the instance down without
releasing it.

## 3. Adjudication (repo)
```bash
python scripts/adjudicate_w9.py --session1 <session-1 return> --session2 <session-2 return> \
    --out records/wp9/w9_verdict.json
```
It reads the session-1 readings, `decisions.json`, `ood2d.json` and the R
manifest from the first return; `plan.json`, `status.txt`, `provenance.txt`
and one report per arm directory from the second (a report the provenance
lists must be there with its SHA-256, and an arm the status file shows
finished must have its report); E1's report from `records/wp8/e1/`; and
PREREG_W9.md. It recomputes the decisions with the frozen rules and requires
the decisions and every run to have run on `prereg-w9`. H1 and H2 per
PREREG_W9 Sec. 6, the secondary readings of Sec. 7, refused reports and
deviations; the inputs' SHA-256 and the adjudicating code's are recorded in
the verdict file.

## 4. After 2d (also after a STOP in 2a): the threads of the paper's solvers (box; ~25 min)
The CMAME paper's cost table (RUNBOOK_CMAME Sec. A, run on 8 October,
recorded in cmame-paper's `records/cmame/timing/`) timed the direct solve
(SuperLU) and conjugate gradients on the CPU in the box's environment
(`OMP_NUM_THREADS` and `MKL_NUM_THREADS` 16, `OPENBLAS_NUM_THREADS` unset; 16
CPUs allotted); how many threads the solvers used was not recorded
(threadpoolctl is not installed there). This section times the same solvers
on the same first instances twice, in that environment and with one thread
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` 1): in 2D on
E1's first 8 validation instances, in 3D on Phase-2b's first 8 validation and
first 2 fine-mesh instances (the direct solve of a fine instance takes about
two minutes and 7-10 GB of host memory); and it records the BLAS libraries
numpy and scipy load, with their thread pools read at run time
(`scripts/blas_threads.py`). The 3D set is timed once more in the box's
environment after the one-thread run, so that a difference between the two
settings can be told from drift. Nothing in PREREG_W9 reads it. Read only: no
state, corpus or report is written. A failure here costs nothing else; report
it.
First the checks, on their own:
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/w9/threads; mkdir -p $P
( date; pgrep -af "fejepa|spawn_main"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  env | grep -E "^(OMP|MKL|OPENBLAS|GOTO)_" | sort ) 2>&1 | tee $P/precheck.txt
```
`pgrep` must print nothing and the GPU line must read about 0 % and under
1,000 MiB; otherwise stop here and report (the timings would be wrong). The
lines after them include `MKL_NUM_THREADS=16` and `OMP_NUM_THREADS=16`, and
no `OPENBLAS_` or `GOTO_` line: the environment of the cost table; if they
differ, note them for the report and go on. Then:
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/w9/threads
[ -s $P/status.txt ] && mv $P/status.txt $P/status.txt.$(date +%Y%m%d-%H%M%S); : > $P/status.txt
run() { local log=$1; shift; [ -e $P/$log ] && mv $P/$log $P/$log.$(date +%Y%m%d-%H%M%S)
        "$@" 2>&1 | tee $P/$log; echo "$log exit=${PIPESTATUS[0]}" | tee -a $P/status.txt; }
run timing_2d_env.log python scripts/time_inference_vs_solve.py \
    --report records/wp8/e1/e1_2d_base/report.json --states-dir runs/e1_2d_base/e8_states \
    --n-val 8 --solvers direct cg --out $P/timing_2d_env.json
run timing_2d_one.log env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scripts/time_inference_vs_solve.py \
    --report records/wp8/e1/e1_2d_base/report.json --states-dir runs/e1_2d_base/e8_states \
    --n-val 8 --solvers direct cg --out $P/timing_2d_one.json
run timing_3d_env.log python scripts/time_inference_vs_solve.py \
    --report records/wp8/e2/baseline/report_phase2b.json --states-dir runs/phase2/e8_states \
    --n-val 8 --n-fine 2 --solvers direct cg --out $P/timing_3d_env.json
run timing_3d_one.log env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scripts/time_inference_vs_solve.py \
    --report records/wp8/e2/baseline/report_phase2b.json --states-dir runs/phase2/e8_states \
    --n-val 8 --n-fine 2 --solvers direct cg --out $P/timing_3d_one.json
run timing_3d_env2.log python scripts/time_inference_vs_solve.py \
    --report records/wp8/e2/baseline/report_phase2b.json --states-dir runs/phase2/e8_states \
    --n-val 8 --n-fine 2 --solvers direct cg --out $P/timing_3d_env2.json
run blas_env.log python scripts/blas_threads.py
run blas_one.log env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python scripts/blas_threads.py
python -c "import numpy, scipy; numpy.show_config(); scipy.show_config()" > $P/blas.txt 2>&1
( echo "HEAD $(git rev-parse HEAD)"; echo "describe $(git describe --tags --match prereg-w9)"
  env | grep -E "^(OMP|MKL|OPENBLAS|GOTO)_" | sort
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep -E "^Model name|^CPU\(s\)|^Thread|^Socket"; echo "nproc $(nproc)"
  cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null; free -g | head -n 2
) > $P/machine.txt 2>&1
cat $P/status.txt
OUT=~/wp9_threads_return && rm -rf $OUT && mkdir -p $OUT && cp $P/* $OUT/
cd ~ && tar czf wp9_threads_return_$(date +%Y%m%d).tgz wp9_threads_return \
    && sha256sum wp9_threads_return_*.tgz && du -h wp9_threads_return_*.tgz
```
Each timing step prints one line per instance (on the 3D fine set one every
minute or two, longer with one thread: long silences there are normal), then
one line per set with the medians and the largest deviation of each solver's
solution from the stored labels (`direct_label_max_rel_dev` below 1e-12, `cg`
below about 1e-8), and last the check of the timed model against the
report's own arrays (0 or round-off). A per-instance line that reads
`skipped` for a solver means that solver ran out of its 900 s budget: report
it. `blas_threads.py` prints one JSON object; in the box's environment its
`pools` should list the BLAS that numpy loads and the one SciPy's SuperLU
links, each with 16 threads, and with one thread 1. `status.txt` must show
the seven steps with `exit=0`. Send the tarball (well under 1 MB) with the
SHA-256 line.

## 5. Withdrawn: the torch stack against the host
Stage 0b scheduled an optional timing of the v2.1.5 code under torch
2.11.0+cu128 (wp8's post-hoc question why the v2.1.5 2D step ran about three
times slower on this box). It needed AutoDL's network route and a 3-4 GB
download, and neither wp9 nor the paper reads it; it is not run.
