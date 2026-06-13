"""Finite-element assembly and parametric data generation for 2D elasticity."""

from fejepa.fe.elasticity import (
    FEProblem,
    LoadCase,
    Material,
    assemble_problem,
    lame_from_E_nu,
)

__all__ = [
    "FEProblem",
    "LoadCase",
    "Material",
    "assemble_problem",
    "lame_from_E_nu",
]
