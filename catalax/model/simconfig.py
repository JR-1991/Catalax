from dataclasses import dataclass
from typing import Any

from diffrax import Tsit5
from jax import Array


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""

    t1: int | Array
    nsteps: int | None = None
    t0: int | Array = 0
    dt0: float = 0.1
    solver: Any = Tsit5
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 4096
    throw: bool = True
