"""FE-JEPA v2.0 -- ground-up refactor matching the Experimental Plan v2.0 one-to-one.

Layout (plan reference in each module docstring; full traceability in PLAN_MAP.md):

  fe/           finite-element core: assembly, generation, stress recovery, exact solves,
                synthetic backend (tests / smoke / bench)
  data/         instance archives + manifests (SHA-256 provenance)
  anchor/       the Lemma-1 assembled-energy anchor            (plan Sec.1)
  models/       features, encoder, decoder, cross-attention predictor,
                pooled-granularity regularizers, MeshGraphNets baseline
  train/        losses (AR / JEPA), pretraining, supervised (+ gradient-balanced anchor)
  baselines.py  repaired sanity baselines: zero, poly, scale-aware, k-NN field (plan B2/E5')
  metrics.py    frozen metric hierarchy                        (plan Sec.4)
  report.py     provenance block, solve ledger integration     (plan B1/B6/WP0)
  experiments/  E1'..E8, Gate G1', runner, cost model          (plan Sec.5, Sec.6, Sec.9)
"""

__version__ = "2.1.5+wp7.4"
