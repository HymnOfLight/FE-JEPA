# FE-JEPA

**Energy-Anchored Joint-Embedding Predictive Pretraining for Finite-Element Surrogates.**

This repository implements the **Phase 0** infrastructure of the FE-JEPA research
proposal (`FE-JEPA_proposal_EN.tex`): a label-free pretraining framework for FEA
surrogates on unstructured meshes that combines masked latent prediction, an
*exact* assembled-energy physics anchor, mesh-refinement views as exact
augmentations, and SIGReg collapse control.

The package (`fejepa`, v0.1) provides the data pipeline, the differentiable
energy anchor, the model backbones, the combined training objective, supervised
baselines, and the **Gate G0** neural-solver sanity check called for in the
proposal's execution plan.

## What is implemented (Phases 0–1)

| Proposal component | Module |
| --- | --- |
| 2D linear-elasticity assembly `K, F` (node-major, SPD after Dirichlet elim.) | `fejepa.fe.elasticity` |
| Parametric generator (plates with holes), multi-resolution views | `fejepa.fe.generator`, `fejepa.fe.meshing` |
| Per-instance archive format `(mesh, K, F-battery, U*)` | `fejepa.data.archive` |
| **Assembled-energy anchor** `Π_h(û)=½ûᵀKû − Fᵀû` (Lemma 1, differentiable) | `fejepa.anchor.energy` |
| Graph-transformer encoder with physics tokens (Transolver-class) | `fejepa.models.encoder` |
| Latent predictor `p_φ`, field decoder `g_ψ` | `fejepa.models.predictor`, `fejepa.models.decoder` |
| SIGReg isotropic-Gaussian collapse control | `fejepa.models.sigreg` |
| Combined objective `L = L_pred + λ_E L_phys + λ_S SIGReg + λ_I L_inv` | `fejepa.losses` |
| MeshGraphNets-style GNN baseline | `fejepa.models.gnn` |
| Supervised baseline / fine-tuning / label-efficiency sweep (RQ2) | `fejepa.train.supervised` |
| Physics-informed fine-tuning (energy anchor as supervised-consistent term) | `fejepa.train.supervised` (`lambda_phys`) |
| **Label-free amortized-Ritz training** (energy anchor only, no labels) | `fejepa.train.amortized_ritz` |
| Cosine LR schedule + warmup (shared by all trainers) | `fejepa.train.schedule` |
| Training-regime comparison (RQ1/RQ4 credit assignment) | `fejepa.experiments.regimes` |
| Naive polynomial surrogate (E5 sanity target) | `fejepa.baselines` |
| Label-free pretraining loop (in-memory + dataset variants) | `fejepa.train.pretrain` |
| **Gate G0** neural-solver sanity check | `fejepa.train.g0` |
| **Phase-1 falsification battery (E1–E5) + Gate G1** | `fejepa.experiments.falsification` |
| Stress / von Mises recovery (CST, consistent with assembled `K`) | `fejepa.fe.stress` |
| Cross-resolution transfer eval (RQ3 / E4) + multi-resolution datasets | `fejepa.experiments.falsification`, `fejepa.fe.generator` |
| Metrics: rel-L2 (disp. & von Mises), max-stress error, critical-region recall, energy gap, effective rank | `fejepa.metrics` |

### The exactness lemma, numerically verified

Lemma 1 of the proposal states that for SPD `K` with `U* = K⁻¹F`,

```
Π_h(û) − Π_h(U*) = ½‖û − U*‖²_K,    ∇Π_h(û) = K(û − U*),
```

so the label-free anchor gradient equals the supervised energy-norm gradient
*exactly*. The test suite checks the gradient identity (to machine precision),
the energy-gap relation, and that minimizing the anchor recovers the FE solution
(`tests/test_anchor.py`). Gate G0 (`tests/test_g0.py`) trains the encoder+decoder
on a single instance using **only** the energy anchor and confirms convergence to
the FE solution in energy norm — validating the full differentiable-assembly path.

## Installation

```bash
pip install -e .            # core (numpy, scipy, scikit-fem, meshio, torch)
pip install -e ".[gen,dev]" # + gmsh (mesh generation) and pytest/matplotlib
```

`gmsh` needs system GL libraries on headless Linux:

```bash
sudo apt-get install -y libglu1-mesa
```

## Quickstart

```bash
# 1. Generate a labelled 2D dataset of plate-with-hole instances.
fejepa generate --out data/train2d -n 200 --seed 0

# 2. Inspect it.
fejepa info --data data/train2d

# 3. Gate G0: the neural solver must reach the FE solution in energy norm.
fejepa gate-g0 --steps 2500

# 4. Label-free FE-JEPA pretraining.
fejepa pretrain --data data/train2d --ckpt runs/fejepa.pt --epochs 5

# 5. Phase-1 pre-registered falsification battery + Gate G1 decision.
fejepa battery --data data/train2d --out runs/report.json \
    --budgets 16,64,256 --experiments E1,E3,E5
```

### Running on GPU / full-scale (config-driven)

All training entry points accept `--device cuda`; the energy anchor uses a torch
sparse mat-vec that runs on CPU or CUDA without host round-trips. For the full
Phase-1 pipeline (dataset generation + regime comparison + battery +
label-efficiency sweep) use the config runner:

```bash
# Edit configs/phase1_2d.json (device, dataset size, epochs) for your hardware.
fejepa run-config configs/phase1_2d.json        # writes JSON reports to runs/

# Quick CPU pipeline check (tiny, just validates the plumbing):
fejepa run-config configs/smoke_cpu.json
```

Numeric outputs land in the configured `out` JSON reports; transcribe them into
[`RESULTS.md`](RESULTS.md), which ships with blank result tables.

### Label-free amortized Ritz, and what the energy anchor buys you

The proposal's central mechanism is that minimizing the *assembled discrete
energy* across an instance distribution amortizes the Ritz minimization — and by
Lemma 1 the per-instance fixed point is the FE solution, with **no labels**:

```bash
# Compare labels / labels+anchor / anchor-only on the same instances.
fejepa regimes --data data/train2d --n-train 40 --epochs 60 --out runs/regimes.json
```

The relative energy gap equals the squared relative energy-norm error
(`‖û−U*‖²_K / ‖U*‖²_K`), i.e. strain/stress accuracy — the quantity engineers act
on. Headline numbers from full-scale (GPU) runs are recorded in
[`RESULTS.md`](RESULTS.md) (currently blank, to be filled by the maintainer).

> Qualitative CPU smoke observation (illustrative only, **not** a headline
> result): on coarse 2D plates, adding the anchor to label training left
> displacement rel-L2 unchanged while cutting the relative energy gap ~11×
> (poor strains → good strains), and label-free anchor-only training reached
> comparable physical consistency with **zero** solved fields. See `RESULTS.md`
> for the authoritative tables.

### Phase 1: falsification battery and Gate G1

The battery (`fejepa.experiments.falsification`) runs the proposal's
pre-registered kill tests and applies the Gate G1 go/no-go criteria:

| ID | Claim under test | Pre-registered kill condition |
| --- | --- | --- |
| **E1** | the assembled-energy anchor is valuable | removing `L_phys` changes fine-tuned error by < 3% at every budget |
| **E2** | the JEPA SSL component is valuable | anchor-only matches full FE-JEPA within 3% |
| **E3** | collapse control works | with `λ_S = 0` latents collapse (low effective rank) |
| **E4** | mesh-refinement views help | `L_inv` on/off shows no cross-resolution gap difference |
| **E5** | basic sanity | fails to beat a naive polynomial surrogate at any budget |

`E1`, `E3`, `E5` are runnable on CPU at modest scale; `E2`/`E4` are wired but
intended for the full-scale (GPU) Phase-1 run. Headline battery verdicts and
Gate G1 decisions are recorded in [`RESULTS.md`](RESULTS.md). The energy anchor's
benefit is strongest once the surrogate is adequately trained; at very short
budgets it can destabilize early optimization, and the harness reports this
honestly rather than hiding it.

Programmatic label-efficiency study (RQ2), comparing from-scratch vs. fine-tuned:

```python
from fejepa.train.supervised import label_efficiency_sweep

scratch  = label_efficiency_sweep("data/train2d", budgets=[16, 64, 256])
finetune = label_efficiency_sweep("data/train2d", budgets=[16, 64, 256],
                                  init_ckpt="runs/fejepa.pt")
```

## Testing

```bash
pytest -q            # full suite (includes the Gate G0 convergence test)
pytest -q --ignore=tests/test_g0.py   # fast subset
```

## Status and scope

This is Phase-0/Phase-1 research scaffolding: correct, tested, CPU-runnable, and
faithful to the proposal's formal setup, including the pre-registered
falsification battery and Gate G1 decision logic. It is **not** yet tuned for the
headline label-efficiency results (those need the larger corpora and GPU training
of a full Phase-1/2 run). The energy anchor and Lemma 1 are exact for linear
elastostatics; hyperelastic/nonlinear anchors (Phase 3) are not yet included.

See `FE-JEPA_proposal_EN.tex` for the full motivation, theory, and execution plan.
