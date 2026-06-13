"""FE-JEPA: Energy-Anchored Joint-Embedding Predictive Pretraining for FE surrogates.

This package implements the Phase 0 infrastructure described in the FE-JEPA
research proposal:

* ``fejepa.fe``      -- 2D linear-elasticity finite-element assembly (K, F).
* ``fejepa.fe.generator`` -- parametric generator of plates with holes/notches.
* ``fejepa.data``    -- on-disk instance archive format and PyTorch datasets.
* ``fejepa.anchor``  -- the differentiable assembled-energy anchor (Lemma 1).
* ``fejepa.models``  -- encoder / predictor / decoder backbones and SIGReg.
* ``fejepa.losses``  -- the combined FE-JEPA training objective (Eq. loss).
* ``fejepa.train``   -- Gate G0 neural-solver sanity check and pretraining loop.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fejepa")
except PackageNotFoundError:  # pragma: no cover - editable / source checkouts
    __version__ = "0.1.0"

__all__ = ["__version__"]
