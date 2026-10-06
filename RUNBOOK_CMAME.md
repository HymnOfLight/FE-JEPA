# RUNBOOK -- cmame-paper, commands in execution order

The box sessions that the CMAME manuscript (`paper/cmame/`) needs. Roles:
**box** = the GPU machine; **repo** = code, records and the manuscript.

## A. Timing and field export, standalone (box; ~1.5 h)

The same measurements as Sec. 4 of `RUNBOOK_W9.md` (Stages 0d-0e of
`wp9-pool`), run on their own and before PREREG_W9's stamp, with the
diagnostics of cmame-paper Stage 1 in the field export. Nothing in PREREG_W9
reads them, and they read no output of wp9's sessions; the 2D timing therefore
runs on E1's validation instances only (the OOD family F5 is made by wp9's
session 1, which has not run). When this section has returned complete, wp9's
Sec. 4 need not be repeated, except its 2D timing on F5 if wp9 wants it.

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

A0b. Fetch the branch (a subshell carries AutoDL's network route, so nothing
stays set; the box's clone is single-branch, hence the explicit refspec):
```bash
cd ~/autodl-tmp/FE-JEPA
( source /etc/network_turbo
  for i in 1 2 3; do git fetch origin +refs/heads/cmame-paper:refs/remotes/origin/cmame-paper && exit 0; sleep 10; done
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
The suite's last line reads `455 passed, 2 deselected`, then warnings and the
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
