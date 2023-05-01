from typing import Dict, List, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import Kvaerno5, ODETerm, PIDController, SaveAt, diffeqsolve
from jax import Array


class Stack(eqx.Module):
    modules: List[eqx.Module]

    def __call__(self, t, y, args):
        maps, parameters, constants = args

        ys = {symbol: y[..., i] for symbol, i in maps.items()}

        return jnp.stack(
            [jnp.array(module(**ys, **parameters, **constants)) for module in self.modules],  # type: ignore
            axis=-1,
        )


def simulate(
    term: ODETerm,
    y0: Array,
    t0: int,
    t1: int,
    dt0: float,
    solver,
    parameters: Dict[str, float],
    maps: Dict[str, int],
    constants: Dict[str, float],
    saveat: SaveAt,
    stepsize_controller: PIDController,
):
    """Simulates a given model"""

    sol = diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=(maps, parameters, constants),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
    )

    return sol.ts, sol.ys
