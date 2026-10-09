# RUNBOOK -- cmame-paper, commands in execution order

The box sessions that the CMAME manuscript (`paper/cmame/`) needs. Roles:
**box** = the GPU machine; **repo** = code, records, pre-registration and
the manuscript.

- Sec. A: timing and field export (read only; ~1.5 h; no pre-registration).
- Sec. B: CM2D, the two-dimensional supervised networks retrained with the
  current code beside E1's label-free states, with a stiffness-norm control
  (PREREG_CM2D; ~11 h, up to ~22 h; only after PREREG_CM2D is stamped and
  tagged `prereg-cm2d`, except its readiness check B0).
- Sec. C: CM2D's adjudication (repo).
- Sec. D: the error spectra of CM2D's models (read only; ~30 min; post hoc,
  no pre-registration; after Sec. C).

No section writes another's files: each reads E1's or Phase-2b's states and
corpora (Sec. D also CM2D's kept states and its committed return); Sec. A
and B0 write only under `runs/cmame/`, Sec. B otherwise only under
`runs/cm2d/`, Sec. D only under `runs/cmame/spectra/`; none writes to a
stamped configuration of another pre-registration, another run's
`e8_states/` or `runs/data2d`.

## A. Timing and field export, standalone (box; ~1.5 h)

The same measurements as Sec. 4 of `RUNBOOK_W9.md` (Stages 0d-0e of
`wp9-pool`), run on their own and before PREREG_W9's stamp, with the
diagnostics of cmame-paper Stage 1 in the field export. Nothing in PREREG_W9
reads them, and they read no output of wp9's sessions; the 2D timing therefore
runs on E1's validation instances only (the OOD family F5 is made by wp9's
session 1, which has not run). When this section has returned complete, wp9's
Sec. 4 need not be repeated, except its 2D timing on F5 if wp9 wants it. It
ran on 8 October 2026 on commit `61f018f` and returned complete; the return
is in `records/cmame/timing/` (`records/cmame/README.md`).

What it measures. The manuscript's cost table: per instance, on the same
instances and this machine, the surrogate's inference (GPU, batch of one, every
load case in one pass; its accuracy on those instances recorded too) against
exact solves of the same system: the direct solve that bought every label
(SuperLU on the free block, one process), unpreconditioned conjugate gradients
to a relative residual of 1e-10 from zero, the same CG started from the
surrogate's prediction, and the same CG from zero stopped as soon as it is as
accurate as the surrogate in energy, for the raw prediction and for the
prediction rescaled by the label-free amplitude c*. 2D: E1's states on 32
validation instances (a few minutes). 3D: Phase-2b's states on 32 in-band
validation instances (about 10 min) and 8 of the fine set, whose direct solves
take several minutes and an estimated 7-10 GB of host memory each; each solver
stops starting new solves after 15 min per set (the one in progress
completes), and the surrogate is timed on every instance. Then the
manuscript's field figures, per-load energies and diagnostics: Phase-2b's
label-free, labels-only and graph-network states (seeds 0-2; the label-free
ones checked against the report, the supervised ones against
`records/wp9/phase2_supervised_states.json`, all of them also by content) on
all 256 validation instances, and three instances chosen by rules fixed in the
script (about 15-25 min, estimated). Read only: no state, corpus or report is
written; everything goes under `runs/cmame/timing/`. A failure here costs
nothing else; report it.

### A0. Preconditions (box)

A0a. The tmux session -- on its own line, outside tmux (skip it if a tmux
status bar is already showing); after a dropped connection, `tmux attach -t cm`:
```bash
tmux new -A -s cm
```

A0b. Fetch the branch (directly: on 8 October 2026, by the operator's
report, AutoDL's network route answered HTTP 503 and the direct fetch
worked; the box's clone is single-branch, hence the explicit refspec):
```bash
cd ~/autodl-tmp/FE-JEPA
( for i in 1 2 3; do git fetch origin +refs/heads/cmame-paper:refs/remotes/origin/cmame-paper && exit 0; sleep 10; done
  exit 1 ) && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short origin/cmame-paper     # the commit the operator instruction names
```
Continue only if `FETCH-OK` is printed and the commit is the named one
(an error printed by a failed first attempt before `FETCH-OK` is harmless);
otherwise stop here and report.

A0c. Checkout, checks and the suite (~5 min):
```bash
cd ~/autodl-tmp/FE-JEPA
git checkout -B cmame-paper origin/cmame-paper   # "Switched to ..." or "Reset branch ...", maybe "set up to track"
git rev-parse 'HEAD^{tree}'                  # the tree the operator instruction names
git status --porcelain --untracked-files=no  # must print nothing
df -h ~/autodl-tmp / | tail -n 2             # >= 3 GB free on the data disk and on /
pgrep -af "fejepa|spawn_main"                # must print nothing
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # about 0 %, under 1,000 MiB
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.12.1+cu130 True
mkdir -p runs/cmame/timing
python -m pytest -q --deselect tests/test_cmame_material.py::test_manuscript_compiles \
    --deselect tests/test_w9_session2.py::test_paper_material_compiles \
    2>&1 | tee runs/cmame/timing/pytest.log | tail -n 3
```
The suite's last line reads `535 passed, 2 deselected`, then warnings and the
time (the two LaTeX builds are left out: they need a TeX installation that
this section does not use). Anything failed, an error, or another count: stop
and report. Any other output than the comments describe: stop and report.

### A1. The measurements (box; ~1.5 h; nothing else on the GPU)

First the checks, on their own:
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cmame/timing; mkdir -p $P
( date; pgrep -af "fejepa|spawn_main"
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader ) 2>&1 | tee $P/precheck.txt
```
`pgrep` must print nothing and the GPU line must read about 0 % and under
1,000 MiB; otherwise stop here and report (the timings would be wrong). Then:
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cmame/timing
[ -s $P/status.txt ] && mv $P/status.txt $P/status.txt.$(date +%Y%m%d-%H%M%S); : > $P/status.txt
run() { local log=$1; shift; [ -e $P/$log ] && mv $P/$log $P/$log.$(date +%Y%m%d-%H%M%S)
        "$@" 2>&1 | tee $P/$log; echo "$log exit=${PIPESTATUS[0]}" | tee -a $P/status.txt; }
run timing_2d.log python scripts/time_inference_vs_solve.py \
    --report records/wp8/e1/e1_2d_base/report.json --states-dir runs/e1_2d_base/e8_states \
    --out $P/timing_2d.json
run timing_3d.log python scripts/time_inference_vs_solve.py \
    --report records/wp8/e2/baseline/report_phase2b.json --states-dir runs/phase2/e8_states \
    --out $P/timing_3d.json
[ -d $P/fields ] && mv $P/fields $P/fields.$(date +%Y%m%d-%H%M%S)
run fields_3d.log python scripts/export_fields.py \
    --report records/wp8/e2/baseline/report_phase2b.json --states-dir runs/phase2/e8_states \
    --out $P/fields
( echo "HEAD $(git rev-parse HEAD)"; echo "tree $(git rev-parse 'HEAD^{tree}')"
  git status --porcelain --untracked-files=no
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep -E "^Model name|^CPU\(s\)|^Thread|^Socket"; echo "nproc $(nproc)"
  cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null; free -g | head -n 2
) > $P/machine.txt 2>&1
cat $P/status.txt
```
What to expect. Each timing step prints one line per instance (on the 3D fine
set one every few minutes: long silences there are normal), then one line per
set with the medians and the largest deviation of each solver's solution from
the stored labels (`direct_label_max_rel_dev`: round-off, below 1e-12; `cg`
and `cg_warm` below about 1e-8; the two matching kinds print `null`, they stop
short of a full solution), and last the check of the timed model against the
report's own arrays (`disp_rel_l2_max_rel_dev` and
`energy_gap_rel_max_rel_dev`, 0 or round-off). In the JSON,
`cg_match_unreached`, `cg_match_cstar_unreached` and both `*_iters_mismatch`
are 0. The field export prints a line every 16 validation instances, one line
per figure and last a summary line:
- `content_median_rel_dev` about 0 for the `ar_*` and `labels_*` states and
  about 1e-4 for the `mgn_*` ones (the graph network's CUDA reductions are not
  bitwise reproducible); `left_out` and `content_mismatch` empty;
- the figures' files `instance_00485.npz` (fig5, validation index 107),
  `instance_00130.npz` (fig6, validation index 154) and `instance_00077.npz`
  (fig7, fine-set index 77);
- under `diagnostics`, for each of `ar`, `labels` and `mgn`:
  `prop1_bound_ratio_max` at most 1 (Proposition 1's bound),
  `stress_identity_max_rel_dev` and `gap_identity_max_rel_dev` round-off
  (below about 1e-6), and `u_dirichlet_max` exactly 0.

`status.txt` must show the three steps with `exit=0`. `exit=4` from the field
export: a supervised state failed both its hash and its content check and was
left out, the rest exported; `exit=5`: a state used on its hash did not
reproduce the report's arrays. Report any of these, and any diagnostic outside
the ranges above.

### A2. Return (box; minutes)
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cmame/timing
OUT=~/cmame_timing_return && rm -rf $OUT && mkdir -p $OUT && cp -r $P/* $OUT/
cd ~ && tar czf cmame_timing_return_$(date +%Y%m%d).tgz cmame_timing_return \
    && sha256sum cmame_timing_return_*.tgz && du -h cmame_timing_return_*.tgz
```
The tarball is a few tens of MB with the fields. Send it with the SHA-256
line; then shut the instance down without releasing it. Do not run wp9's
sessions from this branch: PREREG_W9 is not stamped.

### A3. In the repo
The return is checked (HEAD and tree against the pushed commit, the states'
and report's hashes recorded in the JSON files, the exit codes) and committed
under `records/cmame/timing/`; the manuscript's cost table, field figures and
per-load counts are then generated from it.

## B. CM2D: the 2D supervised networks retrained, with a stiffness-norm control (box; ~11-22 h)

PREREG_CM2D.md governs. One run of the stamped configuration
`configs/cm2d_v1.json` on E1's corpus, split and seeds: E1's three label-free
states evaluated (not trained; each checked by SHA-256), and the supervised
grid trained with the current code -- the labels-only transformer and the same
transformer trained on the relative stiffness-norm error at 16, 64, 256 and
1,024 labels, the graph network at 64 and 1,024, and the naive rows.
2,284,800 supervised steps in 30 units on three workers: about 11 h if the
graph network's step is as long as the transformer's, 18-22 h at two to three
times as long, which may well be the case (not measured; PREREG_CM2D Sec. 7).
The instance's balance must cover at least 30 h (more is safer: no training
step has been timed on the instance's current host). Everything is written
under `runs/cm2d/`. The branch is frozen from the stamp until the return: the
checkout must be exactly the tagged commit. Before the operator instruction
is sent, the repo side checks the remote:
`git ls-remote origin refs/heads/cmame-paper 'refs/tags/*'` shows the branch
at the stamped commit and, of all tags, only `refs/tags/prereg-cm2d^{}` at
that commit (PREREG_CM2D Sec. 6: no other tag on it). It ran on 8-9
October 2026 on the stamped commit `094c804`, one attempt, exit 0, and
returned complete; the return and the verdict are in `records/cmame/cm2d/`
(`records/cmame/README.md`).

### B0. Readiness (box; optional, after A2 or on any visit before the stamp; read only; ~10 min)
The checks of B2 without the tag and the stamp, on the commit the operator
instruction names for B0, which must contain `scripts/cm2d_precheck.py`
(Stage 1's commit `58f079f`, on which Sec. A was first planned, does not;
Sec. A and B0 both ran on Stage 3's, `61f018f`).
They also evaluate E1's three states on the validation split and train two
steps that are not kept, so a FAIL here changes the plan before the stamp
instead of after it. Nothing else may run meanwhile (not during A1).
```bash
cd ~/autodl-tmp/FE-JEPA
( for i in 1 2 3; do git fetch origin +refs/heads/cmame-paper:refs/remotes/origin/cmame-paper && exit 0; sleep 10; done
  exit 1 ) && echo FETCH-OK || echo FETCH-FAILED
git checkout -B cmame-paper origin/cmame-paper   # "Switched to ..." or "Reset branch ...", maybe "set up to track"
git rev-parse --short HEAD                   # the commit the operator instruction names for B0
git status --porcelain --untracked-files=no  # must print nothing
P=runs/cmame; mkdir -p $P
python scripts/cm2d_precheck.py --pre-stamp --out $P/cm2d_ready.json 2>&1 \
    | tee $P/cm2d_ready.log | grep -v '^\[dry-run\] {'
```
What you see: a `[dry-run] prereg guard would refuse: ... unstamped ...`
line, a banner (`=== fejepa v2 run: configs/cm2d_v1.json | ...`), a
`[plan] steps by experiment: {'e8': 2284800, 'total': 2284800}` line, a
`[sup:none] step 2/2 (100%) knorm=...` line (the two training steps),
possibly torch `UserWarning` blocks (harmless), then one `[precheck]` line
per check: `info` for `tag` (it may read `fatal: No names found ...`),
`report_git`, `head` and `stamp`, `ok` for every other check, and last
`GO`. Send `runs/cmame/cm2d_ready.log` beside the Sec. A tarball (or on its
own). A traceback, or a last line other than `GO` / `STOP: ...`, counts as
STOP.

### B1. Preconditions (box)

B1a. The tmux session -- on its own line, outside tmux (skip it if a tmux
status bar is already showing); after a dropped connection, `tmux attach -t cm`:
```bash
tmux new -A -s cm
```

B1b. Fetch the branch and the tag (directly, as in A0b; the tag's refspec is
forced, so that a stale local tag of that name is replaced):
```bash
cd ~/autodl-tmp/FE-JEPA
( for i in 1 2 3; do git fetch origin +refs/heads/cmame-paper:refs/remotes/origin/cmame-paper +refs/tags/prereg-cm2d:refs/tags/prereg-cm2d && exit 0; sleep 10; done
  exit 1 ) && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short origin/cmame-paper     # the commit the operator instruction names
git rev-parse --short 'prereg-cm2d^{commit}' # the same commit
```
Continue only if `FETCH-OK` is printed and both lines are the named commit
(an error printed by a failed first attempt before `FETCH-OK` is harmless);
otherwise stop here and report.

B1c. Checkout, checks and the suite (~5 min):
```bash
cd ~/autodl-tmp/FE-JEPA
git checkout -B cmame-paper origin/cmame-paper   # "Switched to ..." or "Reset branch ...", maybe "set up to track"
git describe --tags --match prereg-cm2d      # must print exactly: prereg-cm2d
git rev-parse 'HEAD^{tree}'                  # the tree the operator instruction names
git status --porcelain --untracked-files=no  # must print nothing
df -h ~/autodl-tmp / | tail -n 2             # >= 7 GB free on the data disk, >= 3 GB on /
pgrep -af "fejepa|spawn_main|scripts/[A-Za-z0-9_]+\.py"   # must print nothing
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # about 0 %, under 1,000 MiB
grep oom_kill /sys/fs/cgroup/memory.events 2>/dev/null   # note the number: B2' compares with it
mkdir -p runs/cm2d
python -m pytest -q --deselect tests/test_cmame_material.py::test_manuscript_compiles \
    --deselect tests/test_w9_session2.py::test_paper_material_compiles \
    2>&1 | tee runs/cm2d/pytest.log | tail -n 3
```
The suite's last line reads `535 passed, 2 deselected`, then warnings and the time.
Anything failed, an error, another count, or any other output than the
comments describe: stop and report.

### B2. Checks and the run (box; ~11-22 h; nothing else on the GPU)
The checks (~10 min; they evaluate E1's states and train two steps that are
not kept):
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cm2d
for f in precheck.json precheck.log; do [ -e $P/$f ] && mv $P/$f $P/$f.$(date +%Y%m%d-%H%M%S); done
python scripts/cm2d_precheck.py --out $P/precheck.json 2>&1 \
    | tee $P/precheck.log | grep -v '^\[dry-run\] {'
```
What you see: `[prereg] verified against PREREG_CM2D.md: <12 hex>...`, the
banner and the `[plan]` line, the `[sup:none] step 2/2 (100%) knorm=...`
line, possibly torch `UserWarning` blocks (harmless), then one `[precheck]`
line per check: `info` for `head`, `ok` for every other check, and last `GO`.
On `STOP: ...` (or a traceback, or another last line): stop here, send
`runs/cm2d/precheck.log`, and shut the instance down without releasing it. On
`GO`, in the same window (the run uses up the GO: pasting this block again
prints the STOP line and starts nothing):
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cm2d
run() { local log=$1; shift; [ -e $P/$log ] && mv $P/$log $P/$log.$(date +%Y%m%d-%H%M%S)
        echo "$log start $(date -u +%FT%TZ)" >> $P/status.txt
        "$@" 2>&1 | tee $P/$log; echo "$log exit=${PIPESTATUS[0]}" | tee -a $P/status.txt; }
grep -q '"go": true' $P/precheck.json 2>/dev/null || echo "STOP: no unused GO in $P/precheck.json -- nothing started"
grep -q '"go": true' $P/precheck.json 2>/dev/null && mv $P/precheck.json $P/precheck.json.used && run run.log python -m fejepa.cli run-config configs/cm2d_v1.json
```
Normal signs:
- the log starts with `[prereg] verified against PREREG_CM2D.md` and
  `[plan] steps by experiment: {'e8': 2284800, 'total': 2284800}`; torch
  `UserWarning` blocks (sparse tensors, a tensor converted to a scalar) are
  harmless, as in E1's log;
- the labelling stage only checks the labels (no solves);
  `[E8 (AR pretrain)] starting: 3 units` evaluates E1's states in minutes;
- `[E8 (supervised grid)] starting: 30 units`, then for each finished unit a
  line from its worker (`[sup:none E8 labels b16 s0] step 3200/3200 (100%)
  disp=...`, `... knorm=...` for the stiffness-norm row) and one from the
  run (`[E8 (supervised grid)] k/30 ... | labels b16 s0`); the 1,024-label
  transformer units take about 3 h each and the graph network's possibly
  6-9 h, so hours without a new line are normal;
- at the end, in this order: `runs/cm2d/report.json`, `RESULTS.md`, a figure,
  a `gate G1'` line that reads `passed=False` (it always does here: it is not
  a CM2D verdict), and the solve ledger with `'total': 0`;
- `status.txt` shows `run.log start ...` and `run.log exit=0`.

### B2'. Health check (any time; a second tmux window: Ctrl-b c; back: Ctrl-b 0)
```bash
cd ~/autodl-tmp/FE-JEPA; date; tail -n 2 runs/cm2d/run.log; tail -n 2 runs/cm2d/status.txt
pgrep -af "fejepa.cli run-config" | cut -c1-120; echo "workers: $(pgrep -fc spawn_main)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
find runs/cm2d/e8_states -type f -mmin -30 -printf '%TH:%TM %f\n' | sort | tail -n 6
grep -h oom_kill /sys/fs/cgroup/memory.events 2>/dev/null; df -h ~/autodl-tmp / | tail -n 2
```
Healthy: one `run-config` process; `workers: 3`; the GPU in use (memory well
above 1,000 MiB); `oom_kill` unchanged since B1c (or since B3 after a box
restart); a file under `e8_states/` written within the last 30 min (the
units' epoch checkpoints, every 5-6 min). Also normal -- check again 15 min
later: before `[E8 (supervised grid)] starting: 30 units` and for a few
minutes after it, `find` prints `No such file or directory` or nothing (E1's
states are only evaluated); after `[E8 (supervised grid)] done in`,
`workers: 0` and the GPU nearly idle for a few minutes (naive rows, report).
Once `status.txt` shows `run.log exit=`, the run has ended: go to B3 or B4,
not here. Otherwise: Ctrl-C in window 0, wait for the prompt, send this
output plus `tail -n 60 runs/cm2d/run.log`; then B4 (return) and shut the
instance down without releasing it; do not restart.

### B3. Interruptions and crashes
- A dropped connection: `tmux attach -t cm`; the run goes on.
- If `runs/cm2d/report.json` exists, the run has finished: do not restart,
  whatever happened afterwards; go to B4.
- The box restarted, or the run ended with an exit other than 0, and there
  is no report (after a Ctrl-C of yours under B2': B4, no restart): read
  `tail -n 30 runs/cm2d/run.log; tail -n 3 runs/cm2d/status.txt; grep oom_kill /sys/fs/cgroup/memory.events`.
  If the log mentions `out of memory` or `a worker process died`, or the
  status file shows `exit=137`, or `oom_kill` rose (it starts again from 0
  after a box restart): do not restart; go to B4 and report. Otherwise
  restart once as below. (After a worker's death the run's own message ends
  "restart the run with --reuse-states": this rule takes precedence.)
- Never delete `runs/cm2d/e8_states/`, the report, a log or `status.txt`.

The restart keeps the finished units (from the unit cache) and the
interrupted ones (up to three, from their epoch checkpoints); E1's states are
checked and evaluated again; all of it is recorded in the report. Open tmux
again (B1a) and run `pgrep -af fejepa.cli`: if it prints a line, the run is
still going -- back to B2', no restart. Note the `oom_kill` number again (B2'
compares with it from now on). Then the checks for a restart (the checkout,
the GPU, and that no report exists):
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cm2d
for f in precheck_restart.json precheck_restart.log; do [ -e $P/$f ] && mv $P/$f $P/$f.$(date +%Y%m%d-%H%M%S); done
python scripts/cm2d_precheck.py --restart --out $P/precheck_restart.json 2>&1 \
    | tee $P/precheck_restart.log | grep -v '^\[dry-run\] {'
```
On `GO` (read as in B2), the restart (it uses up the GO as in B2):
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cm2d
run() { local log=$1; shift; [ -e $P/$log ] && mv $P/$log $P/$log.$(date +%Y%m%d-%H%M%S)
        echo "$log start $(date -u +%FT%TZ)" >> $P/status.txt
        "$@" 2>&1 | tee $P/$log; echo "$log exit=${PIPESTATUS[0]}" | tee -a $P/status.txt; }
grep -q '"go": true' $P/precheck_restart.json 2>/dev/null || echo "STOP: no unused GO in $P/precheck_restart.json -- not restarted"
grep -q '"go": true' $P/precheck_restart.json 2>/dev/null && mv $P/precheck_restart.json $P/precheck_restart.json.used && run run.log python -m fejepa.cli run-config configs/cm2d_v1.json --reuse-states
```
If the checks print STOP, or the restarted run fails again: go to B4 and
report.

### B4. Return (box; minutes)
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cm2d
T=$(date +%Y%m%d-%H%M)
OUT=~/cm2d_return && rm -rf $OUT && mkdir -p $OUT
cp $P/report.json $P/RESULTS*.md $P/figure1_energy_gap*.png $P/status.txt $P/precheck* \
    $P/pytest.log $P/run.log* $OUT/ 2>/dev/null
( echo "HEAD $(git rev-parse HEAD)"; echo "tree $(git rev-parse 'HEAD^{tree}')"
  echo "describe $(git describe --tags --match prereg-cm2d)"
  echo "describe-all $(git describe --always --dirty --tags)"
  echo "--- git status --porcelain ---"; git status --porcelain | grep -v '^?? runs/' | head -40
  python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
  echo "--- report ---"; sha256sum $P/report.json
  echo "--- states ---"; ls -l --full-time $P/e8_states/*.pt
  sha256sum $P/e8_states/*.pt runs/e1_2d_base/e8_states/ar_*.pt
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader
  echo "hostname $(hostname)"; echo "nproc $(nproc)"
  cat /sys/fs/cgroup/memory.events 2>/dev/null; df -h ~/autodl-tmp / | tail -n 2
) > $OUT/provenance.txt 2>&1
cd ~ && tar czf cm2d_return_$T.tgz cm2d_return \
    && sha256sum cm2d_return_$T.tgz && du -h cm2d_return_$T.tgz
```
Send the tarball with the SHA-256 line (a few MB). The states stay on the box
(their SHA-256 are in the provenance file); then shut the instance down
without releasing it.

## C. CM2D's adjudication (repo)
In a checkout of the branch (a clone of its own, not a copy of a worktree),
with the tag fetched by a forced refspec, so that a stale local tag of that
name cannot remain:
```bash
git fetch origin +refs/heads/cmame-paper:refs/remotes/origin/cmame-paper +refs/tags/prereg-cm2d:refs/tags/prereg-cm2d
git rev-parse 'prereg-cm2d^{commit}'         # the stamped commit
git tag --points-at 'prereg-cm2d^{commit}'   # exactly one line: prereg-cm2d
git checkout -B cmame-paper origin/cmame-paper   # frozen at the stamped commit until the return
git describe --tags --match prereg-cm2d      # exactly: prereg-cm2d
git status --porcelain --untracked-files=no  # must print nothing
python scripts/adjudicate_cm2d.py --return <the unpacked cm2d_return> \
    --out records/cmame/cm2d/verdict.json
```
It reads the report, `status.txt`, `provenance.txt` and the run logs from the
return, E1's report from `records/wp8/e1/`, July's from `records/phase1/`
(descriptive only) and PREREG_CM2D.md, checks the recorded commit and tree
against the tag `prereg-cm2d`, and writes H1, H2a and H2b per PREREG_CM2D
Sec. 4, the reading H3, the secondary readings of Sec. 5, the label-free row's
reuse checks and the deviations, with every input's SHA-256 and the
adjudicating code's. The return is committed under `records/cmame/cm2d/` with
the verdict. It ran on 9 October 2026 (`records/cmame/README.md`).

## D. The error spectra of CM2D's models, post hoc (box; ~30 min)

What it measures. CM2D's H2b changed only the norm of the supervised loss,
from the relative Euclidean displacement error (L_D, row `labels`) to the
relative stiffness-norm error (L_K, row `labels_knorm`), and the von Mises
error halved. By the manuscript's Corollary "Modewise contraction", the
energy norm weights the error in each eigenmode of the stiffness matrix by
its eigenvalue and the Euclidean norm weights all modes equally. This section
measures how each model's error is spread over those modes: the free
stiffness block of each of the 256 validation instances is diagonalised once
(at most about 3,600 free dofs), and every model's error on every load case
is projected on its eigenvectors, for CM2D's kept states at 1,024 labels
(L_D, L_K and the graph network, seeds 0-2) and E1's three label-free states;
the spectra are binned by each mode's stiffness relative to the solution's
own Rayleigh quotient (and, for completeness, by mode rank). Every model
predicts each instance twice: as in the run, with TF32 products, and once
more with TF32 off, so that the rounding of TF32, which falls in the stiff
modes, is measured beside the errors.
Also recorded per load case: the error's Euclidean and stiffness norms, the
von Mises errors, the energies and checks of the manuscript's identities.
Specified after CM2D's verdict: post hoc, reported only; no verdict reads it
and nothing in PREREG_CM2D does. Read only: every state's SHA-256 is checked
against CM2D's return (`records/cmame/cm2d/return/provenance.txt`) before
any model runs, every model is also checked against the report's
per-instance arrays, and everything goes under `runs/cmame/spectra/`. A
failure here costs nothing else; report it.

### D0. Preconditions (box)

D0a. The tmux session -- on its own line, outside tmux (skip it if a tmux
status bar is already showing); after a dropped connection, `tmux attach -t cm`:
```bash
tmux new -A -s cm
```

D0b. Fetch the branch (directly, as in A0b):
```bash
cd ~/autodl-tmp/FE-JEPA
( for i in 1 2 3; do git fetch origin +refs/heads/cmame-paper:refs/remotes/origin/cmame-paper && exit 0; sleep 10; done
  exit 1 ) && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short origin/cmame-paper     # the commit the operator instruction names
```
Continue only if `FETCH-OK` is printed and the commit is the named one (an
error printed by a failed first attempt before `FETCH-OK` is harmless). If
nothing at all is printed for 10 min, press Ctrl-C (then neither line
appears) and take the bundle route. On
`FETCH-FAILED` (later on 8 October 2026, at B1b, the direct fetch timed out
and CM2D's operator fetched from a git bundle): upload the bundle that the
operator instruction sends to `~/cmame-paper.bundle` and run, in place of
the first block:
```bash
cd ~/autodl-tmp/FE-JEPA
sha256sum ~/cmame-paper.bundle               # the SHA-256 the operator instruction names
git fetch ~/cmame-paper.bundle +refs/heads/cmame-paper:refs/remotes/origin/cmame-paper && echo FETCH-OK || echo FETCH-FAILED
git rev-parse --short origin/cmame-paper     # the commit the operator instruction names
```
Continue only if the SHA-256 is the named one, `FETCH-OK` is printed and the
commit is the named one; otherwise stop here and report. (Repo side: the
bundle is made from the pushed branch with
`git bundle create cmame-paper.bundle 094c804..cmame-paper`, the stamped
commit being on the box since Sec. B; `git bundle list-heads` shows
`refs/heads/cmame-paper` at the named commit.)

D0c. Checkout, checks and the suite (~5 min):
```bash
cd ~/autodl-tmp/FE-JEPA
git checkout -B cmame-paper origin/cmame-paper   # "Switched to ..." or "Reset branch ...", maybe "set up to track" or "up to date"
git rev-parse 'HEAD^{tree}'                  # the tree the operator instruction names
git status --porcelain --untracked-files=no  # must print nothing
ls runs/cm2d/e8_states/*_b1024_s*.pt runs/e1_2d_base/e8_states/ar_p1024_s*.pt | wc -l   # 12
df -h ~/autodl-tmp / | tail -n 2             # >= 5 GB free on the data disk and on / (the suite needs 5)
pgrep -af "fejepa|spawn_main|scripts/[A-Za-z0-9_]+\.py"   # must print nothing
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader   # about 0 %, under 1,000 MiB
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.12.1+cu130 True
mkdir -p runs/cmame/spectra
python -m pytest -q --deselect tests/test_cmame_material.py::test_manuscript_compiles \
    --deselect tests/test_w9_session2.py::test_paper_material_compiles \
    2>&1 | tee runs/cmame/spectra/pytest.log | tail -n 3
```
The suite's last line reads `535 passed, 2 deselected`, then warnings and the
time. Anything failed, an error, another count, or any other output than the
comments describe: stop and report.

### D1. The export (box; ~10 min; nothing else on the GPU)
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cmame/spectra
[ -s $P/status.txt ] && mv $P/status.txt $P/status.txt.$(date +%Y%m%d-%H%M%S); : > $P/status.txt
run() { local log=$1; shift; [ -e $P/$log ] && mv $P/$log $P/$log.$(date +%Y%m%d-%H%M%S)
        "$@" 2>&1 | tee $P/$log; echo "$log exit=${PIPESTATUS[0]}" | tee -a $P/status.txt; }
[ -d $P/export ] && mv $P/export $P/export.$(date +%Y%m%d-%H%M%S)
run spectra.log python scripts/cm2d_spectra.py --report records/cmame/cm2d/return/report.json \
    --provenance records/cmame/cm2d/return/provenance.txt --states-dir runs/cm2d/e8_states \
    --out $P/export --device cuda
( echo "HEAD $(git rev-parse HEAD)"; echo "tree $(git rev-parse 'HEAD^{tree}')"
  git status --porcelain --untracked-files=no
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  lscpu | grep -E "^Model name|^CPU\(s\)|^Thread|^Socket"; echo "nproc $(nproc)"
  cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null; free -g | head -n 2
) > $P/machine.txt 2>&1
cat $P/status.txt
```
What to expect: possibly torch `UserWarning` blocks (harmless), a line every
16 validation instances (`[spectra] val 16/256 | ... s`), one line for the
figure instance, `[spectra] fig2d: val #221 instance_27356.npz (... nodes)`,
and last a summary line with six keys:
- `content_median_rel_dev` (two numbers per state) about 0 for the `ar_*`,
  `labels_*` and `labels_knorm_*` states; for the `mgn_*` ones small but not
  necessarily 0 (the graph network's CUDA reductions are not bitwise
  reproducible: about 1e-4 in 3D in Sec. A; a median above 1e-3 ends with
  exit 5); `content_mismatch` empty;
- `prop1_bound_ratio_max` at most 1 for every row (Proposition "Energy gap
  and stress error");
- `rayleigh_ratio_median` and `pairs`: the readings themselves; no range is
  expected of them;
- `figures`: `{"fig2d": ["instance_27356.npz", 221]}`.

`status.txt` must show `spectra.log exit=0`. `exit=5`: a model did not
reproduce the report's arrays (everything else is written). `exit=1` with a
message naming a file and its SHA-256: a refusal, before anything is
measured or written under `export/`. Any other exit, or `exit=1` with a
traceback or another message: send `spectra.log`. Report any of these, and
any reading outside the ranges above (a `prop1_bound_ratio_max` above 1, a
transformer's deviation that is not about 0); in every case go on to D2:
the return carries the logs and whatever was written.

### D2. Return (box; minutes)
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/cmame/spectra
T=$(date +%Y%m%d-%H%M)
OUT=~/cm2d_spectra_return && rm -rf $OUT && mkdir -p $OUT && cp -r $P/* $OUT/
cd ~ && tar czf cm2d_spectra_return_$T.tgz cm2d_spectra_return \
    && sha256sum cm2d_spectra_return_$T.tgz && du -h cm2d_spectra_return_$T.tgz
```
The tarball is about 35 MB. Send it with the SHA-256 line; then shut the
instance down without releasing it.

### D3. In the repo
The return is checked (HEAD and tree against the pushed commit, the states'
and the report's hashes recorded in `spectra.json`, the exit code, the
identities on every array, and, end to end, the verdict: the pair
`labels_vs_labels_knorm`'s ratios of means of the relative gap and of the
von Mises error agree with `base_mean / new_mean` of H2a and H2b in
`records/cmame/cm2d/verdict.json`, 4.596 and 2.215, to round-off when both
rows' content deviations are 0) and committed under
`records/cmame/spectra/`; the manuscript's reading of the errors' spectral
content is then generated from it.
