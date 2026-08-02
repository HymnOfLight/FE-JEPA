# FE-JEPA v2.0

Physics-faithful, label-free FEA surrogates around an exact assembled-energy anchor.
This package is a ground-up refactor matching the **Experimental Plan v2.0 one-to-one**
-- see `PLAN_MAP.md` for the full traceability matrix (including what was intentionally
removed) and `PREREG.md` for the frozen gate.

## Install (on the GPU box)

    pip install -e ".[torch,gen,dev]"     # torch + scikit-fem/gmsh + pytest
    pytest                                 # full suite (numpy tests run anywhere;
                                           # torch/skfem/gmsh tests auto-skip if absent)

## Quick start

    # 1. end-to-end smoke on any machine with torch (synthetic backend, CPU, minutes)
    fejepa run-config configs/smoke.json

    # 2. cost projection for the deciding run (measured ms/step, plan Sec.9)
    fejepa bench --device cuda --config configs/phase1_rec8_v2.json

    # 3. freeze the pre-registration (now executable): stamp the hash, commit, tag;
    #    the deciding run REFUSES to start if the config changed after tagging
    fejepa prereg configs/phase1_rec8_v2.json --stamp
    git add PREREG.md configs/phase1_rec8_v2.json && git commit -m prereg
    git tag prereg-v2.0
    # 4. THE deciding run (plan WP1; reuses the audited Phase-1 corpus at runs/data2d)
    fejepa run-config configs/phase1_rec8_v2.json

## Performance on the target box (RTX 5090 32GB, 25 vCPU, torch 2.8/CUDA 12.8)

One dim-256 batch-1 training uses a fraction of the 5090, so the big lever is running
independent grid units concurrently: set `"workers": 3` (rec8 default) or override with
`fejepa run-config <cfg> --workers 4`; watch `nvidia-smi` and raise until SM
utilization saturates. Serial and parallel runs are bit-identical (tested). TF32 is on
by default (`"tf32": false` to disable) and never touches the float64 FE solves,
metrics, or anchor tests. Labelling fans direct solves over `label_workers`
(default `min(8, cpu)`) CPU processes with identical ledger accounting.
`fejepa bench --device cuda --config <cfg>` measures ms/step under the config's own
model size and TF32 policy, and projects hours single-stream and with your `workers`
setting. Lint gate: `ruff check src tests --select F,E9,B` is kept clean.

Every run prints stage banners, per-experiment unit progress with ETA (e.g.
`[E8] 12/30 (40%) elapsed 02:10:41 eta 03:16:02 | labels+anchor b64 s1`), and ~10
in-training milestones per training (`log_every`: 0=auto, -1=silent, N=every N steps).
Device: set `"device": "auto"|"cpu"|"cuda"` in the config or override per run with
`fejepa run-config <cfg> --device cuda`.

The runner also auto-writes `RESULTS.md` (the WP1 acceptance artifact) and
`figure1_energy_gap.png` (Deliverable 4) next to the report; re-render any report
with `fejepa results runs/report_rec8_v2.json --figures`.

The report lands at `runs/report_rec8_v2.json` with: every experiment's metrics +
per-seed/per-instance arrays, kill-condition verdicts, Gate G1' (a/b/c with reasons),
the solve ledger, and the mandatory provenance block.

## Data economy (plan WP5)

    fejepa generate runs/pool --n 30000 --backend gmsh --jobs -1     # unlabelled
    fejepa label runs/pool --n-val 256 --pool-prefix 1024            # buy exactly these
    fejepa info runs/pool

## Theory falsification pass (WP6) and the 3D contract (WP7 foundation)

    fejepa theory --data runs/data2d --n-val 16     # GPU-free; also in every run
    fejepa theory --synthetic 8                      # anywhere, seconds

`fe/tet3d.py` proves the node-major contract in 3D (assembly, recovery, and the
unchanged solve/anchor/polish/theory pipeline); Phase 2 proper remains gated on G1'.

## Predict-then-polish inference (WP3 / C3)

    from fejepa.metrics import torch_predictor
    from fejepa.polish import polished
    fast   = polished(torch_predictor(model, "cuda"), k=5)      # Chebyshev trade
    exact  = polished(torch_predictor(model, "cuda"), tol=1e-8) # warm-started solve

Every report also carries `data_economy` -- the WP5 asymmetry table (labelled
instances bought vs. the unlabeled pool AR consumed), derived from the solve ledger.

## Layout

    src/fejepa/{fe,data,anchor,models,train,experiments}/ ...  # plan refs in docstrings
    configs/{smoke.json, phase1_rec8_v2.json}
    tests/                                                     # numpy suite + guarded torch suite
