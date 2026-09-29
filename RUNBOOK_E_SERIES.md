# RUNBOOK -- E-series (wp8-lejepa), commands in execution order

Stage 1.28-1.30 (29 Sep 2026). Every block below was executed end to end at small
scale in the sandbox with the guards ON (scaled copies of the configurations,
labelled PREREG files stamped through the CLI). The lambda pilot (1b) and the
bench (2a) run BEFORE stamping, by design: they produce the numbers the stamp
records, under rules already committed in PREREG_E1.md / PREREG_E2.md. Every
run (1d, 2c) starts only after its pre-registration is stamped and tagged.
Governance: nothing here touches `configs/phase2_v1.json`, `configs/phase2b_v1.json`,
`PREREG_PHASE2.md` or `PREREG_PHASE2B.md`.

Roles: **box** = the GPU machine (runs and instruments); **repo** = config
generation, stamping, commit and tag (from the pilot's returned JSON).

## 0. Preconditions (box, once)
```bash
cd ~/autodl-tmp/FE-JEPA && git fetch --tags && git checkout wp8-lejepa && git pull
python -m pytest -q                     # 264 passed (BRANCH_NOTES carries the current count)
mkdir -p runs/wp8                       # tee opens its log before any script creates the directory
```

## 1. E1 -- 2D latent shaping

### 1a. Corpus identity (box; must match before anything else)
```bash
python -m fejepa.cli info runs/data2d
# n_instances 30000; manifest_sha256 3553396d183115b843512a35cba1a1e8448b9e83e10c1d1a5b02b235a7c79d26
# (the Phase-1 corpus: val 256 + pool prefix 1024 labelled). Any other hash: STOP and report.
```

### 1b. lambda pilot (box; ~1-2 h; buys no labels; runs under the E1 runs' TF32 policy)
```bash
python scripts/e1_lambda_pilot.py --config configs/phase1_rec8_v2.json --data runs/data2d \
    --n-train 512 --n-val 128 --epochs 20 --head-width auto --out runs/wp8/e1_pilot.json \
    2>&1 | tee runs/wp8/e1_pilot.log
```
Checks in the JSON: `pilot_ledger.total == 0`, `manifest_sha256_before == manifest_sha256_after`,
`pilot_val` names pool[512:640] of the E1 split, `numeric_policy.tf32` is true, `prereg.sha256`
is the SHA-256 of the committed `PREREG_E1.md` draft (the rule the pilot executes), `git` names the head. Return `e1_pilot.json` + log.
`selected_lambda: null` = NO-GO-AT-PILOT: E1 ends here and the JSON is the record.

### 1c. Fill, stamp, tag (repo)
```bash
python scripts/make_e_series_configs.py --e1-lambda <selected_lambda> --e1-head-width <head_width>
python -m pytest -q                                   # the generator test reproduces the filled configs
# PREREG_E1.md (in the repo since Stage 1.30; write LAMBDA and WIDTH from the pilot JSON) carries three labelled lines:
#   CONFIG_SHA256[e1_2d_base]   = <fill before tagging>
#   CONFIG_SHA256[e1_2d_shaped] = <fill before tagging>
#   CONFIG_SHA256[e1_2d_raw_s0] = <fill before tagging>
for arm in base shaped raw_s0; do python -m fejepa.cli run-config configs/e1_2d_$arm.json --dry-run; done
#   -> "would refuse: ... unstamped"
for arm in base shaped raw_s0; do python -m fejepa.cli prereg configs/e1_2d_$arm.json --stamp --prereg-file PREREG_E1.md; done
for arm in base shaped raw_s0; do python -m fejepa.cli run-config configs/e1_2d_$arm.json --dry-run; done
#   -> "prereg_status": "verified" for all three
# commit; record PREREG_E1.md's blob SHA-256 in the file (second commit); tag prereg-e1; push the tag;
# verify: git ls-remote --tags origin prereg-e1  (and HTTP 200 on the tag ref) BEFORE any GPU time
```
The guard refuses every arm until all three lines are stamped: no arm can be
tuned after another arm's result is seen.

### 1d. Runs (box; tmux; ~9 h per three-seed arm, ~3 h for raw_s0)
```bash
git pull --tags && git describe --tags          # must print prereg-e1
for arm in base shaped raw_s0; do
  mkdir -p runs/e1_2d_$arm
  python -m fejepa.cli run-config configs/e1_2d_$arm.json 2>&1 | tee runs/e1_2d_$arm/run.log
done
```
Each log starts with `[prereg] verified against PREREG_E1.md`; each solve ledger reads 0.

### 1e. Separation readings (box; minutes)
```bash
for arm in base shaped; do for s in 0 1 2; do
  python scripts/latent_separation.py --config configs/e1_2d_$arm.json \
      --state runs/e1_2d_$arm/e8_states/ar_p1024_s$s.pt --data runs/data2d \
      --out runs/wp8/sep_${arm}_s$s.json; done; done
python scripts/latent_separation.py --config configs/e1_2d_raw_s0.json \
    --state runs/e1_2d_raw_s0/e8_states/ar_p1024_s0.pt --data runs/data2d \
    --out runs/wp8/sep_raw_s0.json                        # ablation: reported only
```

### 1f. Verdict (anywhere)
```bash
python scripts/adjudicate_e1.py --base-report runs/e1_2d_base/report.json \
    --shaped-report runs/e1_2d_shaped/report.json \
    --base-sep runs/wp8/sep_base_s*.json --shaped-sep runs/wp8/sep_shaped_s*.json \
    --out runs/wp8/e1_verdict.json
```
K1 compares seed means behind a noise guard, max(10%, 2 x SE_rel), and
reports the threshold it applied. The adjudicator pairs S with seeds through
the state paths the separation files record, refuses files from the other arm or not measured on `val`, and
refuses reports that differ beyond the loss specification.

## 2. E2 -- token bottleneck (3D; baseline = the Phase-2b AR cells)

### 2a. Bench (box; ~30 min for both M)
```bash
for M in 512 1024; do
  python scripts/bench_phase2_preconditions.py configs/phase2b_v1.json --bottleneck-tokens $M \
      --out runs/wp8/bench_e2_m$M.json 2>&1 | tee runs/wp8/bench_e2_m$M.log
done
```
Green: `bottleneck<M>_fine.peak_gib` well below the card. `ms_per_step` of the
bottleneck phases is set-up-free (differential timing, `"timing": "differential"`)
-- the number E2's K2/GO lines use.

### 2b. Stamp and tag (repo)
PREREG_E2.md (in the repo since Stage 1.30) records the baseline file's SHA-256
(`report_phase2b.json` = 320b6db5060ecae9f4747327228c7705d30877bece67d09af2f68d6c466f2794)
and carries two labelled lines, `CONFIG_SHA256[e2_m512]` and `CONFIG_SHA256[e2_m1024]`.
Dry-run both (refuse), stamp both with `fejepa prereg configs/e2_m<M>.json --stamp
--prereg-file PREREG_E2.md`, dry-run both (verified), commit, blob SHA, tag
`prereg-e2`, push, verify the remote tag.

### 2c. Runs (box; tmux)
```bash
git pull --tags && git describe --tags          # must print prereg-e2 (or prereg-e2-N-g... after
                                                # engineering-only commits, ledgered)
for M in 512 1024; do
  mkdir -p runs/e2_m$M
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m fejepa.cli run-config configs/e2_m$M.json --label-workers 1 2>&1 | tee runs/e2_m$M/run.log
done
```
The corpora are read-only here: the runs buy no labels (solve ledger 0) and
write only under `runs/e2_m<M>/`.

### 2d. Verdicts (anywhere)
```bash
for M in 512 1024; do
  python scripts/adjudicate_e2.py --base-report runs/phase2/report_phase2b.json \
      --e2-report runs/e2_m$M/report.json --bench runs/wp8/bench_e2_m$M.json --tokens $M \
      --out runs/wp8/e2_verdict_M$M.json
done
```
K1 compares seed means behind a noise guard, max(10%, 2 x SE_rel); the
verdict reports the achieved resolution. The adjudicator refuses any baseline
whose config SHA-256 is not the Phase-2b stamp (the Phase-2 report's AR cells are the D14 defect: against them any
architecture passes parity), a report of the wrong M, reports that differ
beyond the architecture, and a bench without set-up-free timing.

## 3. Interruptions
Same command with `--reuse-states`, in a new tmux session and with a new log
name (units cached, the in-flight unit resumed from its epoch checkpoint).
Never delete a run's `e8_states/`.
