from typing import List, Dict, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    AbstractSolver,
    Kvaerno5,
    ODETerm,
    PIDController,
    SaveAt,
    Tsit5,
    diffeqsolve,
)
from jax import Array
from pydantic import BaseModel, Field, PrivateAttr


class Stack(eqx.Module):
    modules: List[eqx.Module]

    def __call__(self, t, y, args):
        species_maps, parameter_maps, parameters = args

        ys = {symbol: y[..., i] for symbol, i in species_maps.items()}
        params = {symbol: parameters[i] for symbol, i in parameter_maps.items()}

        return jnp.stack(
            [module(t=t, **ys, **params) for module in self.modules],  # type: ignore
            axis=-1,
        )


class Simulation(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    term: ODETerm
    dt0: float
    parameter_maps: Dict[str, int]
    species_maps: Dict[str, int]
    solver: AbstractSolver = Tsit5
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 4096

    _simulation_func = PrivateAttr(default=None)

    def _prepare_func(self, in_axes=None):
        """Applies all the necessary transformations to the term and prepares the simulation function"""

        def _simulate_system(y0, parameters, time):
            sol = diffeqsolve(
                terms=self.term,  # type: ignore
                solver=self.solver(),  # type: ignore
                t0=0,
                t1=time[-1],
                dt0=self.dt0,  # type: ignore
                y0=y0,
                args=(
                    self.species_maps,
                    self.parameter_maps,
                    parameters,
                ),  # type: ignore
                saveat=SaveAt(ts=time),  # type: ignore
                stepsize_controller=PIDController(rtol=self.rtol, atol=self.atol, step_ts=time),  # type: ignore
                max_steps=self.max_steps,
            )

            return sol.ts, sol.ys

        if in_axes is not None:
            self._simulation_func = jax.jit(jax.vmap(_simulate_system, in_axes=in_axes))
        else:
            self._simulation_func = jax.jit(_simulate_system)

    def __call__(self, y0, parameters, time) -> Any:
        if self._simulation_func is None:
            raise ValueError("Simulation function not initialized")

        return self._simulation_func(y0, parameters, time)


def simulate(
    term: ODETerm,
    y0: Array,
    t0: int,
    t1: int,
    dt0: float,
    parameters: Array,
    parameter_maps: Dict[str, int],
    species_maps: Dict[str, int],
    saveat: SaveAt,
    max_steps: int,
    stepsize_controller: PIDController = PIDController(rtol=1e-5, atol=1e-5),
    solver=Kvaerno5(),
):
    """Simulates a given model"""

    sol = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=(species_maps, parameter_maps, parameters),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )

    return sol.ts, sol.ys
