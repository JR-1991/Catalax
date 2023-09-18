from typing import List, Dict, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from diffrax import (
    AbstractSolver,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)

from pydantic import BaseModel, PrivateAttr

from .symbolicmodule import SymbolicModule
from catalax.model.ode import ODE


class Stack(eqx.Module):
    modules: List[eqx.Module]

    def __init__(
        self,
        parameters: List[str],
        odes: List[ODE],
        **kwargs,
    ):
        positionals = {
            "parameters": parameters,
            "states": [str(ode.species.symbol) for ode in odes],
        }

        self.modules = [
            SymbolicModule(ode.equation, positionals=positionals)  # type: ignore
            for ode in odes
        ]

    def __call__(self, t, y, args):
        parameters = args
        rates = jnp.stack(
            [module(t=t, parameters=parameters, states=y) for module in self.modules],  # type: ignore
            axis=-1,
        )

        return rates


class Simulation(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    odes: List[ODE]
    parameters: List[str]
    stoich_mat: jax.Array
    dt0: float = 0.1
    solver: AbstractSolver = Tsit5
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 64**4

    _simulation_func = PrivateAttr(default=None)

    def _prepare_func(self, in_axes=None):
        """Applies all the necessary transformations to the term and prepares the simulation function"""

        stack = Stack(parameters=self.parameters, odes=self.odes)

        def _simulate_system(y0, parameters, time):
            sol = diffeqsolve(
                terms=ODETerm(stack),  # type: ignore
                solver=self.solver(),  # type: ignore
                t0=0,
                t1=time[-1],
                dt0=self.dt0,  # type: ignore
                y0=y0,
                args=parameters,  # type: ignore
                saveat=SaveAt(ts=time),  # type: ignore
                stepsize_controller=PIDController(rtol=self.rtol, atol=self.atol, step_ts=time),  # type: ignore
                max_steps=self.max_steps,
                # throw=False,
            )

            return sol.ts, sol.ys

        if in_axes is not None:
            return eqx.filter_jit(jax.vmap(_simulate_system, in_axes=in_axes))
        else:
            return eqx.filter_jit(_simulate_system)

    def __call__(self, y0, parameters, time) -> Any:
        if self._simulation_func is None:
            raise ValueError("Simulation function not initialized")

        return self._simulation_func(y0, parameters, time)
