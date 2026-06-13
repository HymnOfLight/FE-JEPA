"""Finite-element assembly and parametric data generation for 2D elasticity."""

from fejepa.fe.elasticity import (
    FEProblem,
    LoadCase,
    Material,
    assemble_problem,
    lame_from_E_nu,
)
from fejepa.fe.stress import element_strain_stress, strain_energy

__all__ = [
    "FEProblem",
    "LoadCase",
    "Material",
    "assemble_problem",
    "lame_from_E_nu",
    "element_strain_stress",
    "strain_energy",
]
