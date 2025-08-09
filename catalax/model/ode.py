from __future__ import annotations

from pydantic import Field

from .equation import Equation
from .species import Species


class ODE(Equation):
    species: Species = Field(..., exclude=True)
    observable: bool = True
