# RUNBOOK -- E-series (wp8-lejepa), commands in execution order

Every line below has been executed at least once (scaled) in the sandbox.
Run on the box after the Phase-2 verdict is on record and the GPU is free.
Governance: nothing here touches `configs/phase2_v1.json` or `PREREG_PHASE2.md`.

## 0. Preconditions (once)
```bash
git checkout wp8-lejepa && git pull && python -m pytest -q        # 229 passed (check BRANCH_NOTES for the current count)
```

## 1. Stage-0 readings on the Phase-2 AR states (~10 min)
```bash
python scripts/intrinsic_dimension.py --config configs/phase2_v1.json --device auto \
    --state runs/phase2/e8_states/ar_p1024_s0.pt --data runs/data3d_phase2 \
    --n-instances 64 --out runs/wp8/intrinsic_dim_s0.json
# repeat for s1, s2; read suggested_head_width (0 = full width) -> PREREG_E1
```

## 2. E1 (2D) -- lambda pilot, then both arms, then adjudication
```bash
# 2a. lambda selection (pre-declared rule; ~1-2 h at 2D scale)
python scripts/e1_lambda_pilot.py --config configs/phase1_rec8_v2.json \
    --data runs/data2d --n-train 512 --n-val 128 --epochs 20 --out runs/wp8/e1_pilot.json
# 2b. fill PREREG_E1.md and configs/e1_2d_shaped.json (lambda_reg, sigreg_head_width),
#     regenerate nothing by hand: edit the two placeholders only, then
python -m fejepa.cli run-config configs/e1_2d_shaped.json --dry-run   # must print [dry-run] ... verified
# 2c. stamp PREREG_E1.md (same four steps as Phase 2: stamp, commit, blob sha, tag e1-stamped)
# 2d. base arm: reuse the Phase-1 AR states if present
# Phase-1's out was runs/report_rec8_v2.json, so its states live in runs/e8_states/
mkdir -p runs/e1_2d_base/e8_states && cp runs/e8_states/ar_p1024_s*.pt runs/e1_2d_base/e8_states/
tmux new -s e1 ; python -m fejepa.cli run-config configs/e1_2d_base.json --reuse-states 2>&1 | tee runs/e1_2d_base/run.log
python -m fejepa.cli run-config configs/e1_2d_shaped.json 2>&1 | tee runs/e1_2d_shaped/run.log
# 2e. separation readings on the val split (both arms, every seed)
for arm in base shaped; do for s in 0 1 2; do
  python scripts/latent_separation.py --config configs/e1_2d_$arm.json \
      --state runs/e1_2d_$arm/e8_states/ar_p1024_s$s.pt --data runs/data2d \
      --out runs/wp8/sep_${arm}_s$s.json; done; done
# 2f. verdict
python scripts/adjudicate_e1.py --base-report runs/e1_2d_base/report.json \
    --shaped-report runs/e1_2d_shaped/report.json \
    --base-sep runs/wp8/sep_base_s*.json --shaped-sep runs/wp8/sep_shaped_s*.json \
    --out runs/wp8/e1_verdict.json
```

## 3. E2 (3D) -- bench gate, two M, adjudication against the Phase-2 report
```bash
# 3a. speed/memory envelope of the bottleneck (the K2/GO numbers; ~30 min)
python scripts/bench_phase2_preconditions.py configs/phase2_v1.json \
    --bottleneck-tokens 512 --out runs/wp8/bench_e2_m512.json
python scripts/bench_phase2_preconditions.py configs/phase2_v1.json \
    --bottleneck-tokens 1024 --out runs/wp8/bench_e2_m1024.json
# 3b. validate and stamp
python -m fejepa.cli run-config configs/e2_m512.json --dry-run
python -m fejepa.cli run-config configs/e2_m1024.json --dry-run
# stamp PREREG_E2.md (four steps; tag e2-stamped)
# 3c. runs (tmux; --reuse-states only for a restart)
python -m fejepa.cli run-config configs/e2_m512.json  2>&1 | tee runs/e2_m512/run.log
python -m fejepa.cli run-config configs/e2_m1024.json 2>&1 | tee runs/e2_m1024/run.log
# 3d. verdicts
python scripts/adjudicate_e2.py --base-report runs/phase2/report_phase2.json \
    --e2-report runs/e2_m512/report.json --bench runs/wp8/bench_e2_m512.json --tokens 512
python scripts/adjudicate_e2.py --base-report runs/phase2/report_phase2.json \
    --e2-report runs/e2_m1024/report.json --bench runs/wp8/bench_e2_m1024.json --tokens 1024
```

## 4. Interruptions
Same command with `--reuse-states` in a new tmux session and a new log name
(units cached, in-flight unit resumed from its epoch checkpoint -- bitwise
tests exist for every unit type). Never delete a run's `e8_states/`.
