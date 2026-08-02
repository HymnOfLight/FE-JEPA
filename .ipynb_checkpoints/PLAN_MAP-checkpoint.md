# PLAN_MAP -- Plan v2.0 <-> Code, one-to-one

Every runtime capability in this package traces to a plan v2.0 item, and every plan
item that calls for code traces to a module here. Anything in the old codebase that
plan v2.0 does not call for was **removed** (see "Intentionally absent").

## Claims and core machinery

| Plan item | Code | Test |
|---|---|---|
| Sec.1 Lemma 1: exact anchor, analytic gradient | `anchor/energy.py` (`EnergyAnchor`, `pi_h`, `energy_gap`) | `test_anchor.py` (grad == K u - F to machine precision, torch); `test_stress_energy.py` (gap identity, numpy) |
| Sec.2.1 verified FE core: assembly, node-major dofs, load battery | `fe/elasticity.py` (skfem), `fe/synthetic.py` (structured CST twin) | `test_solve.py`, `test_stress_energy.py` |
| Sec.2.1 archives / manifest ordering (audit V4) | `data/archive.py` | `test_archive.py`, `test_protocol_gate.py` (split determinism) |
| Sec.2.5 asset: exact CG/direct solver; old assembly-level Gate G0 | `fe/solve.py` | **G0 is now a test**: `test_solve.py::test_solver_reproduces_reference` (plan schedules no runtime G0) |
| Sec.2.5 asset: stress recovery == 0.5 u^T K u | `fe/stress.py` | `test_stress_energy.py` |
| C3 polish primitives (iterations-to-tol, k-step CG) | `fe/solve.py::cg_iterations_to_tol / cg_k_steps` | `test_solve.py` |
| C1 Amortized-Ritz training (zero labels) | `train/losses.py::AR_CONFIG`, `train/pretrain.py::amortized_ritz` | `test_losses_train.py` (torch) |
| C4 / WP2 redesign: BFS region masking | `models/fejepa.py::region_target_mask` | `test_models.py` (torch-free mask props in `test_metrics.py::test_region_mask`) |
| C4 / WP2 redesign: cross-attention predictor over the context *set* | `models/fejepa.py::CrossAttentionPredictor`, `FEJEPA.masked_prediction` | `test_models.py` (torch) |
| WP0: `sigreg_pooled` mode | `models/regularizers.py::sigreg_pooled` | `test_losses_train.py` (torch) |
| Sec.2.5 asset: VICReg-on-pooled + ring buffer | `models/regularizers.py::vicreg_pooled`, `PooledBuffer` | `test_losses_train.py` (torch) |
| Sec.2.5 asset: load-summary conditioning | `models/features.py::load_summary` | `test_metrics.py::test_features` |
| WP0: geometry-descriptor channel (from `arch.meta`, broadcast) | `models/features.py::geometry_descriptor`, `FeatureSpec.geometry` | `test_metrics.py::test_features` |
| Sec.5 item 4 lambda policy: gradient-balanced anchor primary; fixed lambda alongside; grid secondary w/ bias named | `train/supervised.py` (`anchor_mode` none/fixed/balanced), `experiments/e1_anchor.py` (`grid_best.selection_bias=true`) | `test_losses_train.py` (torch), `test_experiments_smoke.py` |
| Sec.4 baseline requirement: trained MeshGraphNets | `models/gnn.py::build_mesh_gnn` + E8 `include_mgn` + `runner.make_mgn_factory` | `test_models.py` (torch) |

## Metrics (Sec.4, frozen)

| Plan item | Code | Test |
|---|---|---|
| Relative energy gap (primary) | `metrics.py::energy_gap_rel` (via `anchor/energy.py`) | `test_metrics.py` (identity: gap == 0.5 \|\|e\|\|_K^2) |
| vM rel-L2, peak-vM error, critical recall (top-10% elements) | `metrics.py::vm_suite` | `test_metrics.py` |
| CG iterations-to-tolerance | `fe/solve.py` + `experiments/e7_polish.py` | `test_solve.py` |
| Displacement rel-L2 (secondary; never the anchored headline) | `metrics.py::displacement_errors` | `test_metrics.py` |
| Standardized effective rank + input-feature floor (B4 repair) | `metrics.py::effective_rank(standardized=True)`, `e3_collapse.py::input_floor` | `test_metrics.py::test_effective_rank` |
| Label-efficiency AUC | `metrics.py::label_efficiency_auc` | `test_metrics.py` |
| Per-instance arrays everywhere (B6) | `metrics.py::evaluate_model` | `test_metrics.py`, `test_baselines.py` |

## Battery (Sec.6) and gate (Sec.5)

| Plan item | Code | Kill encoded | Test |
|---|---|---|---|
| E1' | `experiments/e1_anchor.py` | K1 (<25% egap impr everywhere) | `test_experiments_smoke.py` (torch) |
| E2 (post-WP2, one-shot verdict) | `experiments/e2_jepa.py` | K3 (JEPA within 3% of AR everywhere) | `test_experiments_smoke.py` (torch) |
| E3' | `experiments/e3_collapse.py` | K4 (best std-rank ratio <= 1.5) | smoke (torch); probe math in `test_metrics.py` |
| E4' | `experiments/e4_meshviews.py` | K6 (reduction < 10% at max coarsen) | smoke (torch) |
| E5' repaired sanity | `experiments/e5_sanity.py` + `baselines.py` | fails => bug hunt; feeds gate (a) | `test_baselines.py` (numpy) |
| E6 | `experiments/e6_alignment.py` | rho < 0.3 within-geometry | smoke (torch) |
| E7 | `experiments/e7_polish.py` | K5 (<20% iteration savings) | `test_solve.py` (primitives, numpy) |
| E8 regime grid + AUC + MGN column + mandatory naive rows (zero / scale-aware / k-NN, plan Sec.4 "every headline table") | `experiments/e8_regimes.py` (`naive_baseline_cells`) | K2 + C1-advantage (<40% everywhere) | `test_experiments_smoke.py` (torch); `test_baselines.py::test_e8_naive_baseline_cells` (numpy) |
| Gate G1' (a/b/c, fails closed, retired criterion echoed) | `experiments/gate.py` | -- | `test_protocol_gate.py` (numpy, crafted results) |
| P-A / P-B named, never conflated | `experiments/protocol.py::PIPELINE_PA/PB`; E8 `ar_axis` note | -- | -- |
| Pre-registration text | `PREREG.md` (commit + tag before the deciding run) | -- | -- |
| Plan-coverage invariant: every battery ID resolves to an importable `run_*`, wired in the runner, present in both configs; smoke enables all eight; rec8 keeps E2 disabled per prereg | `tests/test_plan_coverage.py` | -- | itself (numpy CI) |

## Process repairs (Sec.2.4 B1-B7 / WP0) and infrastructure

| Plan item | Code | Test |
|---|---|---|
| B1 provenance block; "runs without it are void" | `report.py::provenance / write_report` (writer refuses) | `test_report.py` |
| B2 E5 strawman repair | `baselines.py` (zero / poly / scale-aware / k-NN field) | `test_baselines.py` |
| B3 lambda selection bias named | `e1_anchor.py::grid_best.selection_bias` | `test_experiments_smoke.py` |
| B4 rank instrument repair | `metrics.effective_rank` + E3' probes + floor | `test_metrics.py` |
| B5 gate certifies measured things only | `gate.py` (fails closed on unmeasured) | `test_protocol_gate.py` |
| B6 per-seed / per-instance arrays | `protocol.mean_std`, `metrics.evaluate_model`, E1'/E8 records | `test_metrics.py` |
| B7 / WP5 data economy + solve ledger | `fe/solve.py::SolveLedger`, generator `labelled='none'`, `runner._label_files/_label_need` | `test_archive.py`, `test_cost.py::test_ledger` |
| WP1 deciding run config | `configs/phase1_rec8_v2.json` (E2 disabled per Sec.5 item 6) | JSON parse in CI |
| Sec.9 cost model + bench | `experiments/cost.py` | `test_cost.py` (hand count) |
| Runner (stages, order E1'->E5'->E8->E2->E3'->E4'->E6->E7->gate) | `experiments/runner.py` | `test_experiments_smoke.py::test_runner_smoke` (torch) |
| Sec.9 operability: run progress (stage banners, per-unit ETA in every experiment, ~10 trainer milestones; `log_every` 0=auto/-1=silent/N) | `progress.py`; `Task` wiring in E1'-E8; banners + labelling progress in `runner.py` | `test_progress_and_device.py` |
| Device selection: config `device` auto/cpu/cuda, threaded to every trainer/anchor/predictor; CLI `fejepa run-config --device` override | `runner.run_config(device_override)`, `cli.py`, `SupervisedConfig/PretrainConfig.device`, `AnchorCache(device)` | `test_progress_and_device.py` |
| Sec.5 item 3, honest seed variation: model init varies with the seed, not only data order | `protocol.seeded_factory`, used by every experiment | `test_models.py` (torch) |
| Sec.9 rational use of the box: unit-level parallelism (`workers`) -- E1'/E8 grids as independent payloads over one GPU; serial == parallel (bit-identical, single-threaded) | `experiments/parallel.py` (`map_units`, `supervised_unit`, `pretrain_unit`); E1'/E8 payload paths; `FEJEPA_WORKER_THREADS` | `test_experiments_smoke.py::test_e1_parallel_matches_serial` (torch); `test_progress_and_device.py::test_map_units_inline_preserves_order_and_verbosity` |
| Sec.9: TF32 precision policy (float32 matmul only; float64 solves/metrics/anchor tests untouched); config `tf32`, echoed in the report | `runtime.py::setup_torch`, `runner` (`runtime_policy` in payload) | `test_runner_smoke` asserts the policy block |
| Sec.9: no per-step GPU->CPU syncs (loss parts stay tensors; balanced-anchor scale accumulates on-device) | `train/losses.py`, `train/supervised.py` | `test_losses_train.py` (torch) |
| WP5 at 25 vCPUs: parallel labelling with identical ledger accounting | `runner._label_one/_label_files(workers)`, `label_workers` config | `test_progress_and_device.py::test_parallel_labelling_matches_serial` (runs in numpy CI) |
| Static-analysis gate: `ruff check src tests --select F,E9,B` is clean (no undefined names in the torch-only modules; loop-capture hazards root-caused; all grid `zip`s are `strict=`); empty training sets fail loud instead of raising `UnboundLocalError` or silently evaluating an untrained model | whole tree; `train/pretrain.py`, `train/supervised.py`, `experiments/parallel.py` guards | `test_losses_train.py::test_empty_training_sets_fail_loud` (torch); `test_map_units_empty_payloads`; ruff in CI |
| Sec.9 step-cost reductions, provably equivalent: JEPA step reuses the battery latents (one fewer encoder pass); k-NN baseline caches one Delaunay per neighbour; BFS masking on a deque; contiguous CSR operand for the anchor mat-vec; adjacency built only when masking is on (AR never touches it); `bench` measures under the same TF32 policy as real runs | `models/fejepa.py::masked_prediction(z_full=...)`, `train/losses.py`, `baselines.KNNFieldBaseline`, `anchor/energy.py` | `test_models.py::test_masked_prediction_reuse_is_identical` (torch); `test_baselines.py::test_knn_triangulation_cache_reused` |
| CLI | `cli.py` | smoke via `--help` in CI |
| Synthetic backend (tests/smoke/bench without gmsh/skfem) | `fe/synthetic.py` | entire numpy test suite builds on it |

## Work packages (code demands)

| WP | Code demand | Where | Test |
|---|---|---|---|
| WP0 | provenance, per-seed/per-instance, E5' repair, standardized rank, ledger, `sigreg_pooled`, geometry channel | rows above | rows above |
| WP1 | deciding config; **executable prereg freeze**; acceptance artifact: **filled RESULTS.md** + Figure-1 energy-gap curve (Deliverables 3-4), auto-written by the runner and re-renderable via `fejepa results` | `report.py::stamp_prereg/verify_prereg`, `cli prereg/results`, `results.py::write_results/write_figures`, runner guard + auto-render | `test_prereg_stamp_verify_roundtrip`; `test_results_render.py`; numpy runner smoke asserts RESULTS.md |
| WP2 | region masking, cross-attn predictor, pooled regularizers, E2 producer; **pre-E2 mask-ratio sweep {0.2,0.4,0.6}** (held-out masked-pred MSE + rank sentinel, recommended ratio; label-free, no kill) | rows above; `experiments/wp2_masking.py::run_wp2`, runner `wp2` block | rows above; `test_experiments_smoke.py::test_wp2_smoke` (torch) |
| WP3 | E7 table; **fe/solve wired into evaluation/inference** | `polish.py` (`polish_battery`, `polished`), `fe/solve` warm-start `x0`; E7 consumes it | `test_solve.py::test_polish_battery_and_wrapper`, `::test_solve_warm_start_matches_and_converges` |
| WP4 | MGN in the harness + E8 column; encoder decision documented | rows above; PLAN_MAP absent-list | rows above |
| WP5 | unlabelled pool, labelling stage, ledger; **the data-asymmetry table written into every report** | `runner.data_economy_summary` -> `payload["data_economy"]` | `test_data_economy_summary_math`; numpy runner smoke |
| WP6 | the note itself is LaTeX; its acceptance ("internal falsification pass") is executable: conditioning-lemma inequality + exact mode-contraction rate (C5b), Chebyshev polish bound (C5c), Prop.1 within-geometry premise + numeric counterexample to the naive cross-geometry extension (C5d). C5 kill trips on any violated inequality; the counterexample being found is the expected, scoping outcome | `theory.py`; runner `wp6` block (GPU-free, on in smoke+rec8); `cli theory`; RESULTS.md section | `tests/test_theory.py` (all checks execute in numpy CI) |
| WP7 | full Phase 2 (SimJEB + official splits, 3D features/encoder, linear attention, AMP/compile validation) stays G1'-gated and intentionally absent; the plan's contract clause -- "3D P1 tetrahedra, SAME node-major contract" -- is proven now: 3D assembly + recovery under dof `3i+c`, and archive/solve/anchor/energy-gap/polish/theory-checks run on 3D instances unchanged | `fe/tet3d.py` | `tests/test_tet3d.py` (numpy CI) |
| WP8 | release hygiene: configs/provenance/ledger in reports; checkpoints/runs ignored | `.gitignore` | `test_work_packages_with_code_demands_are_implemented` |

## Intentionally absent (plan-cited removals)

| Removed | Why (plan citation) |
|---|---|
| Network-mode Gate G0 (energy descent on one instance) | Plan v2.0 battery schedules no runtime G0; the assembly-level guarantee lives on as `test_solve.py` (Sec.2.5 note). |
| Uniform node-dropout masking | Superseded by WP2 region masking; keeping both would violate "no superfluous parts". |
| AMP (bf16) / `torch.compile` runtime layer | Plan Sec.2.5 tags it unvalidated; validation is scheduled in WP7 (Phase 2). TF32 (`runtime.py`) is shipped as the numerically-safe subset. |
| Linear/slice-attention encoder | WP4 "encoder decision" is explicitly open; Phase-1 plates run on full attention (Sec.2.5). |
| 3D assembly / SimJEB / OOD splits | WP7, conditional on G1'. |
| Old single-lambda E1, displacement-gated G1 | Superseded by E1' + G1' (Sec.5 items 1-2: criterion retired with evidence). |
