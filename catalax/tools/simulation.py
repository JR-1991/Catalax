from typing import List, Optional, Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    ConstantStepSize,
    PIDController,
    SaveAt,
    diffeqsolve,
)
from pydantic import BaseModel, PrivateAttr, ConfigDict

from catalax.model.inaxes import InAxes
from catalax.model.ode import ODE
from .symbolicmodule import SymbolicModule

NON_ADAPTIVE_SOLVERS = [
    dfx.Euler,
    dfx.Heun,
]


class Stack(eqx.Module):
    modules: List[eqx.Module]

    def __init__(
        self,
        parameters: List[str],
        odes: List[ODE],
        constants: List[str],
        **kwargs,
    ):
        positionals = {
            "parameters": parameters,
            "constants": constants,
            "states": [str(ode.species.symbol) for ode in odes],
        }

        self.modules = [
            SymbolicModule(ode.equation, positionals=positionals)  # type: ignore
            for ode in odes
        ]

    def __call__(self, t, y, args):
        (parameters, constants) = args
        rates = jnp.stack(
            [
                module(
                    t=t,
                    parameters=parameters,
                    states=y,
                    constants=constants,
                )
                for module in self.modules
            ],  # type: ignore
            axis=-1,
        )

        return rates


class Simulation(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    odes: List[ODE]
    parameters: List[str]
    constants: List[str]
    stoich_mat: jax.Array
    dt0: float = 0.1
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 64**4
    sensitivity: Optional[InAxes] = None
    solver: Any = dfx.Tsit5

    _simulation_func = PrivateAttr(default=None)

    def _prepare_func(self, in_axes=None):
        """Applies all the necessary transformations to the term and prepares the simulation function"""

        stack = Stack(
            parameters=self.parameters,
            odes=self.odes,
            constants=self.constants,
        )

        if self.solver in NON_ADAPTIVE_SOLVERS:
            controller = lambda time: ConstantStepSize()
        else:
            controller = lambda time: PIDController(
                rtol=self.rtol, atol=self.atol, step_ts=time
            )

        def _simulate_system(y0, parameters, constants, time):
            sol = diffeqsolve(
                terms=dfx.ODETerm(stack),
                solver=self.solver(),  # type: ignore
                t0=0,
                t1=time[-1],
                dt0=self.dt0,  # type: ignore
                y0=y0,
                args=(parameters, constants),  # type: ignore
                saveat=SaveAt(ts=time),  # type: ignore
                stepsize_controller=controller(time),  # type: ignore
                max_steps=self.max_steps,
            )

            return sol.ys

        if self.sensitivity is not None:
            assert isinstance(self.sensitivity, InAxes), (
                "Expected sensitivity to be an instance of 'InAxes'"
            )

            def sens_fun(y0s, parameters, constants, time):
                return _simulate_system(y0s, parameters, constants, time)

            index = self.sensitivity.value.index(0)

            if in_axes is not None:
                return eqx.filter_jit(
                    jax.vmap(jax.jacobian(sens_fun, argnums=index), in_axes=in_axes)
                )

            return eqx.filter_jit(jax.jacobian(sens_fun, argnums=int(index)))

        if in_axes is not None:
            return eqx.filter_jit(jax.vmap(_simulate_system, in_axes=in_axes))
        else:
            return eqx.filter_jit(_simulate_system)

    def __call__(self, y0, parameters, constants, time) -> Any:
        if self._simulation_func is None:
            raise ValueError("Simulation function not initialized")

        return self._simulation_func(y0, parameters, constants, time)
