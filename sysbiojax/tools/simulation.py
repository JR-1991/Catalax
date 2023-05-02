from typing import Dict, List

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import ODETerm, PIDController, SaveAt, diffeqsolve, Kvaerno5
from jax import Array


class Stack(eqx.Module):
    modules: List[eqx.Module]

    def __call__(self, t, y, args):
        species_maps, parameter_maps, parameters = args

        ys = {symbol: y[..., i] for symbol, i in species_maps.items()}
        params = {symbol: parameters[i] for symbol, i in parameter_maps.items()}

        return jnp.stack(
            [module(**ys, **params) for module in self.modules],  # type: ignore
            axis=-1,
        )


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
    )

    return sol.ts, sol.ys
