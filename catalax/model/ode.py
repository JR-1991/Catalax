from __future__ import annotations

from pydantic import Field

from .equation import Equation
from .state import State


class ODE(Equation):
    state: State = Field(..., exclude=True)
    observable: bool = True
