from typing import Callable, List, Optional, Any, Tuple

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
            SymbolicModule(ode.equation, positionals=positionals) for ode in odes
        ]

    def __call__(self, t, y, args):
        parameters, constants = args
        rates = jnp.stack(
            [
                module(  # type: ignore
                    t=t,
                    parameters=parameters,
                    states=y,
                    constants=constants,
                )
                for module in self.modules
            ],
            axis=-1,
        )

        return rates


class Simulation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

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

    _simulation_func: Optional[Callable] = PrivateAttr(default=None)

    def _prepare_func(self, in_axes=None) -> Tuple[Callable, Stack]:
        """Applies all the necessary transformations to the term and prepares the simulation function"""
        stack = Stack(
            parameters=self.parameters,
            odes=self.odes,
            constants=self.constants,
        )

        controller = self._create_controller()
        simulate_system = self._create_simulate_system(stack, controller)

        if self.sensitivity is not None:
            return self._prepare_sensitivity_func(simulate_system, in_axes)  # type: ignore

        return self._prepare_standard_func(simulate_system, in_axes)  # type: ignore

    def _create_controller(self):
        """Create the appropriate stepsize controller"""
        if self.solver in NON_ADAPTIVE_SOLVERS:
            return ConstantStepSize()
        else:
            return PIDController(rtol=self.rtol, atol=self.atol)

    def _create_simulate_system(self, stack, controller):
        """Create the core simulation function"""

        def _simulate_system(y0, parameters, constants, time):
            sol = diffeqsolve(
                terms=dfx.ODETerm(stack),
                solver=self.solver(),
                t0=0,
                t1=time[-1],
                dt0=self.dt0,
                y0=y0,
                args=(parameters, constants),
                saveat=SaveAt(ts=time),
                stepsize_controller=controller,
                max_steps=self.max_steps,
            )
            return sol.ys

        return _simulate_system

    def _prepare_sensitivity_func(self, simulate_system, in_axes):
        """Prepare simulation function with sensitivity analysis"""
        assert isinstance(self.sensitivity, InAxes), (
            "Expected sensitivity to be an instance of 'InAxes'"
        )

        def sens_fun(y0s, parameters, constants, time):
            return simulate_system(y0s, parameters, constants, time)

        index = self.sensitivity.value.index(0)

        if in_axes is not None:
            return (
                eqx.filter_jit(
                    jax.vmap(jax.jacobian(sens_fun, argnums=index), in_axes=in_axes)
                ),
                self,
            )

        return (
            eqx.filter_jit(jax.jacobian(sens_fun, argnums=int(index))),
            self,
        )

    def _prepare_standard_func(self, simulate_system, in_axes):
        """Prepare standard simulation function"""
        if in_axes is not None:
            return (
                eqx.filter_jit(jax.vmap(simulate_system, in_axes=in_axes)),
                self,
            )
        else:
            return (
                eqx.filter_jit(simulate_system),
                self,
            )

    def __call__(self, y0, parameters, constants, time) -> Any:
        if self._simulation_func is None:
            raise ValueError("Simulation function not initialized")

        return self._simulation_func(y0, parameters, constants, time)
