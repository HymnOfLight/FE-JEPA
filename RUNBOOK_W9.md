# RUNBOOK -- wp9-pool, commands in execution order

Stage 0a (1 Oct 2026): Sec. 1, session 1 -- readings, timing and data
generation; nothing is trained and nothing is adjudicated. Its outputs that
feed a decision are read by the rules committed before it runs
(`scripts/w9_session1_decisions.py`; BRANCH_NOTES_wp9-pool.md, "Pre-declared
readings"). Session 2 (C1 and, if rule 1 admits it, S) runs only after
PREREG_W9.md is stamped and tagged (Stage 0b; Sec. 2 is written then). Every
block of Sec. 1 except 1e was executed end to end at small scale in the
sandbox (a miniature E1-like run; sizes scaled down, commands otherwise as
below); 1e needs the box's network and GPU.

Roles: **box** = the GPU machine; **repo** = code, records, pre-registration.

Governance: nothing here touches a stamped configuration, a PREREG file, a
run's `e8_states/`, `runs/data2d` or any other run directory; session 1 writes
under `runs/w9/` only (and, in 1e, a virtual environment on the data disk).

## 0. Preconditions (box, once)
```bash
cd ~/autodl-tmp/FE-JEPA
source /etc/network_turbo               # AutoDL's academic network route (GitHub)
for i in 1 2 3; do git fetch origin +refs/heads/wp9-pool:refs/remotes/origin/wp9-pool --tags && break; sleep 10; done
unset http_proxy https_proxy
git rev-parse --short origin/wp9-pool   # the commit the operator instruction names
git checkout -B wp9-pool origin/wp9-pool
git rev-parse HEAD^{tree}               # the tree the operator instruction names
git status --porcelain --untracked-files=no   # must print nothing
mkdir -p runs/w9/session1
python -m pytest -q 2>&1 | tee runs/w9/session1/pytest.log   # 350 passed (BRANCH_NOTES carries the count)
```
The explicit refspec matters: the box's clone is single-branch, so a plain
`git fetch` or `git pull` does not see a new branch.

## 1. Session 1 (box; ~45 min-1 h, plus ~20-30 min for 1e; GPU otherwise idle)

Inputs: E1's base report (the records copy, `records/wp8/e1/e1_2d_base/report.json`,
byte-identical to the box's), its three states (`runs/e1_2d_base/e8_states/`) and
its corpus (`runs/data2d`). `w9_c0.py` refuses states whose SHA-256 is not the
one the report trained, a corpus whose manifest is not the report's, and an
evaluation family with a file that differs from its manifest. 1b's memory audit
and 1c measure memory and time: nothing else may run on the GPU.
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/w9/session1; : > $P/status.txt
run() { local log=$1; shift; "$@" 2>&1 | tee $P/$log; echo "$log exit=${PIPESTATUS[0]}" | tee -a $P/status.txt; }
R=records/wp8/e1/e1_2d_base/report.json; S=runs/e1_2d_base/e8_states
# 1a. OOD-2D v1 (F1-F5, 256 each) and the remesh set R (16 geometries x 5 h): gmsh,
#     labelled by direct solve, manifests pin every file (CPU; ~3-5 min). F5 and R
#     come first; a family that fails is recorded and the others still run. A re-run
#     verifies a family that has a manifest instead of regenerating it.
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
`status.txt` must hold seven lines, each ending `exit=0`. The C0 commands print
one JSON summary each; the memory audit prints one line per pool mark (1,024,
4,096, 12,800, 25,600; it stops early, and says so, if the process's residency
passes 60% of the host or the device memory) and then the step measurement.
`decisions.json` lists, per rule, the value and the numbers it read; a rule
whose input is missing or malformed is recorded as undecided (with the error),
the others are decided, and 1d then exits non-zero naming the undecided rules.
A failing step is not re-run with other arguments. Dependencies: `amp2d` needs
1a's F5 and R; 1d needs 1b and 1c; nothing else depends on another step.

### 1e. Optional: the torch stack against the host (~20-30 min)
wp8's post-hoc timing could not separate the box's torch stack (2.12.1+cu130)
from its host as the cause of the ~3x slower 2D step of the v2.1.5 code
(WP2's run of 31 July 2026 used torch 2.11.0+cu128 on the same GPU model). This
times the v2.1.5 code in-process under both stacks in one session. The torch
wheels are ~3-4 GB from download.pytorch.org (the network decides the time). A
failure here is reported and costs nothing else.
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/w9/session1
run() { local log=$1; shift; "$@" 2>&1 | tee $P/$log; echo "$log exit=${PIPESTATUS[0]}" | tee -a $P/status.txt; }
[ -d ../FE-JEPA-v215/src ] || git worktree add ../FE-JEPA-v215 v2.1.5
run profile_v215_t212.log python scripts/posthoc_profile_2d.py --src ../FE-JEPA-v215/src \
    --variants default threads_w3 --no-sup --no-profile --out $P/profile_2d_v215_t212.json
source /etc/network_turbo
python -m venv --system-site-packages ~/autodl-tmp/venv-t211
~/autodl-tmp/venv-t211/bin/pip install torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128 2>&1 | tee $P/venv_t211.log
unset http_proxy https_proxy
~/autodl-tmp/venv-t211/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))" \
    2>&1 | tee -a $P/venv_t211.log      # 2.11.0+cu128 NVIDIA GeForce RTX 5090
run profile_v215_t211.log ~/autodl-tmp/venv-t211/bin/python scripts/posthoc_profile_2d.py \
    --src ../FE-JEPA-v215/src --variants default threads_w3 --no-sup --no-profile \
    --out $P/profile_2d_v215_t211.json
cat $P/status.txt
```
`status.txt` then holds nine lines. The virtual environment shares the box's
other packages and shadows only torch and its CUDA libraries (~5 GB on the data
disk; keep it for later timings). Both outputs record the torch version they ran.

### 1f. Return (box)
```bash
cd ~/autodl-tmp/FE-JEPA
P=runs/w9/session1; S=runs/e1_2d_base/e8_states
OUT=~/wp9_session1_return && rm -rf $OUT && mkdir -p $OUT
cp $P/*.json $P/*.log $P/status.txt $OUT/
for f in F1 F2 F3 F4 F5 R; do mkdir -p $OUT/ood2d/$f; cp runs/w9/ood2d/$f/manifest.json $OUT/ood2d/$f/; done
( echo "HEAD $(git rev-parse HEAD)"; echo "tree $(git rev-parse HEAD^{tree})"
  echo "--- git status --porcelain ---"; git status --porcelain | grep -v '^?? runs/' | head -40
  echo "--- states ---"; sha256sum $S/ar_p1024_s*.pt
  echo "--- v215 worktree ---"; git -C ../FE-JEPA-v215 rev-parse HEAD 2>&1
  python -c "import torch, gmsh; print('torch', torch.__version__, 'gmsh', gmsh.__version__)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
  echo "hostname $(hostname)"; echo "nproc $(nproc)"; grep -E 'MemTotal|MemAvailable' /proc/meminfo
  echo "--- cgroup ---"; cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max 2>/dev/null
  cat /sys/fs/cgroup/memory.events 2>/dev/null; df -h ~/autodl-tmp | tail -n 1
) > $OUT/provenance.txt 2>&1
cd ~ && tar czf wp9_session1_return_$(date +%Y%m%d).tgz wp9_session1_return \
    && sha256sum wp9_session1_return_*.tgz && du -h wp9_session1_return_*.tgz
```
The instances under `runs/w9/ood2d/` stay on the box (the manifests in the
return pin them); session 2 evaluates on them. Do not delete or regenerate them.

## 2. Session 2 -- C1 (and S) training and evaluation
Written at Stage 0b, after session 1's return, with PREREG_W9.md.
