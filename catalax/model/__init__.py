from .model import Model
from .inaxes import InAxes
from .simconfig import SimulationConfig

import arviz as az  # noqa: F401

# Set plotting style
az.style.use("arviz-doc")  # type: ignore

__all__ = [
    "SimulationConfig",
    "Model",
    "InAxes",
]
