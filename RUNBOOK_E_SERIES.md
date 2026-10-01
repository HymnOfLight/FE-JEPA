# RUNBOOK -- E-series (wp8-lejepa), commands in execution order

Stage 1.28-1.35 (29 Sep - 1 Oct 2026). Stage 1.34: 1b, 1c, 2a and 2b are DONE -- the pilot and
the bench returned (`records/wp8/`), PREREG_E1 and PREREG_E2 are stamped in one commit,
and the tags `prereg-e1` and `prereg-e2` both point at it. Stage 1.35: 1d-1f and 2c-2d are
DONE (1 Oct) -- E1 NO-GO, E2 KILLED at both M; the returned records are in
`records/wp8/e1/` and `records/wp8/e2/`. Every block below was executed end to end at small
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
cd ~/autodl-tmp/FE-JEPA
# Stage 1.33: an explicit refspec -- `git pull` does not advance a single-branch clone
git fetch origin +refs/heads/wp8-lejepa:refs/remotes/origin/wp8-lejepa --tags
git checkout -B wp8-lejepa origin/wp8-lejepa
git rev-parse HEAD^{tree}               # the tree the operator instruction names
git status --porcelain --untracked-files=no   # must print nothing (no local edits carried over)
mkdir -p runs/wp8                       # tee opens its log before any script creates the directory
python -m pytest -q 2>&1 | tee runs/wp8/pytest.log   # 289 passed (BRANCH_NOTES carries the count)
```

## 1. E1 -- 2D latent shaping

### 1a. Corpus identity (box; must match before anything else)
```bash
python -m fejepa.cli info runs/data2d
# n_instances 30000; manifest_sha256 3553396d183115b843512a35cba1a1e8448b9e83e10c1d1a5b02b235a7c79d26
# (the Phase-1 corpus: val 256 + pool prefix 1024 labelled). Any other hash: STOP and report.
```

### 1b. lambda pilot (box; ~0.5 h; buys no labels; runs under the E1 runs' TF32 policy) -- DONE 29 Sep, `records/wp8/e1_pilot.json`
```bash
python scripts/e1_lambda_pilot.py --config configs/phase1_rec8_v2.json --data runs/data2d \
    --n-train 512 --n-val 128 --epochs 20 --head-width auto --out runs/wp8/e1_pilot.json \
    2>&1 | tee runs/wp8/e1_pilot.log
```
Checks in the JSON: `pilot_ledger.total == 0`, `manifest_sha256_before == manifest_sha256_after`,
`pilot_val` names pool[512:640] of the E1 split, `numeric_policy.tf32` is true, `lr` is the
arms' `e8.ar_lr` (1e-3), `prereg.sha256` is the SHA-256 of the committed `PREREG_E1.md` draft
(the rule the pilot executes), `git` names the head. Return `e1_pilot.json` + log.
`selected_lambda: null` = NO-GO-AT-PILOT: E1 ends here and the JSON is the record.

### 1c. Fill, stamp, tag (repo) -- DONE in Stage 1.34 (from `records/wp8/e1_pilot.json`)
```bash
python scripts/make_e_series_configs.py --e1-from-pilot records/wp8/e1_pilot.json --fill-prereg PREREG_E1.md
#   reads lambda and head width FROM the pilot JSON (refuses smoke, null lambda, ledger > 0,
#   a changed manifest, another split/config/TF32 policy) and writes them, with the pilot
#   JSON's SHA-256, into PREREG_E1.md's parameter line and the shaped/raw configs
python -m pytest -q                                   # generator + PREREG/config agreement tests
# PREREG_E1.md carries three labelled lines:
#   CONFIG_SHA256[e1_2d_base]   = <fill before tagging>
#   CONFIG_SHA256[e1_2d_shaped] = <fill before tagging>
#   CONFIG_SHA256[e1_2d_raw_s0] = <fill before tagging>
for arm in base shaped raw_s0; do python -m fejepa.cli run-config configs/e1_2d_$arm.json --dry-run; done
#   -> "would refuse: ... unstamped"
for arm in base shaped raw_s0; do python -m fejepa.cli prereg configs/e1_2d_$arm.json --stamp --prereg-file PREREG_E1.md; done
for arm in base shaped raw_s0; do python -m fejepa.cli run-config configs/e1_2d_$arm.json --dry-run; done
#   -> "prereg_status": "verified" for all three
# footer: PREREG_E1_SHA256 = SHA-256 of the file with that line reading <record after commit>
#   (recorded in the stamp commit itself; tests/test_e_series_records.py checks it)
# commit; tag prereg-e1; push the tag;
# verify: git ls-remote --tags origin prereg-e1  (and HTTP 200 on the tag ref) BEFORE any GPU time
```
The guard refuses every arm until all three lines are stamped: no arm can be
tuned after another arm's result is seen.

### 1d. Runs (box; tmux; ~2-3 h per three-seed arm, < 1 h for raw_s0)
```bash
git fetch origin +refs/heads/wp8-lejepa:refs/remotes/origin/wp8-lejepa --tags
git checkout -B wp8-lejepa origin/wp8-lejepa
git describe --tags --match prereg-e1           # must print prereg-e1 (prereg-e2 is on the same
                                                # commit; plain `git describe` may name either)
for arm in base shaped raw_s0; do
  mkdir -p runs/e1_2d_$arm
  python -m fejepa.cli run-config configs/e1_2d_$arm.json 2>&1 | tee runs/e1_2d_$arm/run.log
done
```
Each log starts with `[prereg] verified against PREREG_E1.md`; each solve ledger reads 0.

### 1e. Separation readings (box; minutes)
```bash
# each file records the measured state's SHA-256 and the configuration's canonical SHA-256
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
refuses reports that differ beyond the loss specification. Stage 1.31: it also
refuses unstamped reports, swapped arms (the base must carry no loss_spec, the
shaped arm must be sigreg_ep_head with a filled lambda and width), non-finite
per-seed metrics, and separation files that are invalid, not on all 256
validation instances, measured with another configuration, or on a state
whose SHA-256 is not the one the report trained. Stage 1.33: and files measured
on another corpus (data manifest SHA-256); a diverged (non-finite) shaped-arm
seed counts as K1 (KILLED, S not evaluated), a non-finite AR-arm value is refused.

## 2. E2 -- token bottleneck (3D; baseline = the Phase-2b AR cells)

### 2a. Bench (box; ~30 min for both M) -- DONE 29 Sep, `records/wp8/bench_e2_m{512,1024}.json` (all valid, no re-bench)
```bash
for M in 512 1024; do
  python scripts/bench_phase2_preconditions.py configs/phase2b_v1.json --bottleneck-tokens $M \
      --out runs/wp8/bench_e2_m$M.json 2>&1 | tee runs/wp8/bench_e2_m$M.log
done
```
Green: `bottleneck<M>_fine.peak_gib` well below the card and `"valid": true` on both
bottleneck phases. `ms_per_step` of the bottleneck phases is set-up-free: the median
of three differential pairs of n1 = 10 / n2 = 110 steps (`estimates_ms`, `pairs`;
`"valid"` = every pair positive) -- the number E2's K2/GO lines use; `prepare_ms` is
one instance preparation (features, seeds, token assignment) for the cost frontier.
Each bottleneck phase records the decoder it timed (`"decode_k": 6`, the continuous
decoder of PREREG_E2 r10 and the E2 configurations); the adjudicator refuses a bench
of another decoder or protocol. Run it on an otherwise idle GPU (no other process in
`nvidia-smi`). If a bottleneck phase is invalid, re-bench that M once (PREREG_E2 Sec. 3:
the re-bench supersedes the first file; keep both):
```bash
M=1024      # the M to re-bench
python scripts/bench_phase2_preconditions.py configs/phase2b_v1.json --bottleneck-tokens ${M} \
    --out runs/wp8/bench_e2_m${M}_rerun.json 2>&1 | tee runs/wp8/bench_e2_m${M}_rerun.log
```

### 2b. Stamp and tag (repo) -- DONE in Stage 1.34 (same commit as E1)
PREREG_E2.md (in the repo since Stage 1.30) records the baseline file's SHA-256
(`report_phase2b.json` = 320b6db5060ecae9f4747327228c7705d30877bece67d09af2f68d6c466f2794)
and carries two labelled lines, `CONFIG_SHA256[e2_m512]` and `CONFIG_SHA256[e2_m1024]`.
Dry-run both (refuse), stamp both with `fejepa prereg configs/e2_m<M>.json --stamp
--prereg-file PREREG_E2.md`, dry-run both (verified), record the bench and the cost
projection (Sec. 3, 6), fill the footer PREREG_E2_SHA256 as for E1, commit, tag
`prereg-e2`, push, verify the remote tag.

### 2c. Runs (box; tmux)
```bash
git fetch origin +refs/heads/wp8-lejepa:refs/remotes/origin/wp8-lejepa --tags
git checkout -B wp8-lejepa origin/wp8-lejepa
git describe --tags --match prereg-e2           # must print prereg-e2 (or prereg-e2-N-g... after
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
for M in 512 1024; do          # the bench PREREG_E2 Sec. 3 records (no re-bench was needed)
  python scripts/adjudicate_e2.py --base-report runs/phase2/report_phase2b.json \
      --e2-report runs/e2_m$M/report.json --bench records/wp8/bench_e2_m$M.json --tokens $M \
      --out runs/wp8/e2_verdict_M$M.json
done
```
K1 compares seed means behind a noise guard, max(10%, 2 x SE_rel); the
verdict reports the achieved resolution. The adjudicator refuses any baseline
whose config SHA-256 is not the Phase-2b stamp (the Phase-2 report's AR cells are the D14 defect: against them any
architecture passes parity), a report of the wrong M, reports that differ
beyond the architecture, and a bench without set-up-free timing. Stage 1.31:
also an unstamped E2 report, non-finite per-seed values, and a bench whose
bottleneck<M>_fine phase is missing (formerly read as KILLED), invalid, not
from CUDA, from a smoke run, or of another M. Stage 1.33: also an E2 report
whose decoder is not decode_k = 6 and a bench of another decoder or protocol;
a diverged (non-finite) bottleneck seed counts as K1 (KILLED), a non-finite
baseline value is refused.

## 3. Interruptions
Same command with `--reuse-states`, in a new tmux session and with a new log
name (units cached, the in-flight unit resumed from its epoch checkpoint).
Never delete a run's `e8_states/`. On CUDA a resumed unit is not bitwise
identical to an uninterrupted one (nor are two fresh runs: atomic reductions
in the attention backward and the bottleneck's scatter-mean); the report's
`d9_restart` block records every resumption -- state it, do not hide it. The
bitwise-resume tests pin the CPU.
