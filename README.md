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

## What is implemented (Phase 0)

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
| Label-free pretraining loop | `fejepa.train.pretrain` |
| **Gate G0** neural-solver sanity check | `fejepa.train.g0` |
| Metrics: rel-L2, energy gap `Π_h(û)−Π_h(U*) = ½‖û−U*‖²_K` | `fejepa.metrics` |

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
```

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

This is Phase-0 research scaffolding: correct, tested, CPU-runnable, and faithful
to the proposal's formal setup. It is **not** yet tuned for the headline
label-efficiency results (those are Phases 1–2, requiring larger corpora and GPU
training). The energy anchor and Lemma 1 are exact for linear elastostatics;
hyperelastic/nonlinear anchors (Phase 3) are not yet included.

See `FE-JEPA_proposal_EN.tex` for the full motivation, theory, and execution plan.
