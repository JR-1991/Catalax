from typing import Dict, List
from pydantic import BaseModel, PrivateAttr
import jax

from sysbiojax.model.model import Model


class Simulation(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    term = 0
    dt0: float

    _simulation_func = PrivateAttr(default=None)


class ParameterEstimator(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    model: Model
    data: jax.Array
    time: jax.Array

    def _init_lmfit_params(self):
        pass

    def fit(self):
        pass
