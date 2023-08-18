from typing import List, Optional

import jax
import jax.numpy as jnp
import diffrax

from catalax import Model

from .mlp import MLP
from .neuralbase import NeuralBase


class NeuralODE(NeuralBase):
    def __init__(
        self,
        data_size: int,
        width_size: int,
        depth: int,
        model: Model,
        observable_indices: List[int],
        solver=diffrax.Tsit5,
        *,
        key,
        **kwargs,
    ):
        super().__init__(
            data_size=data_size,
            width_size=width_size,
            depth=depth,
            observable_indices=observable_indices,
            solver=solver,
            model=model,
            **kwargs,
        )

        # Save solver and MLP
        self.func = MLP(data_size, width_size, depth, key=key)  # type: ignore

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            self.solver(),  # type: ignore
            t0=0.0,  # type: ignore
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys

    def predict(
        self,
        y0s: jax.Array,
        t1: int,
        t0: int = 0,
        nsteps: int = 1000,
        times: Optional[jax.Array] = None,
    ):
        if times is None:
            # Generate time points, if not explicitly given
            times = jnp.linspace(t0, t1, nsteps)

        if y0s.shape[0] > 1 and len(times.shape) == 1:
            # If multiple initial conditions are given, repeat time points
            times = jnp.stack([times] * y0s.shape[0], axis=0)

        # Single simulation case
        if len(times.shape) == 1 and len(y0s.shape) == 1:
            return times, self(times, y0s)

        # Batch simulation case
        assert (
            len(times.shape) == 2 and len(y0s.shape) == 2
        ), f"Incompatible shapes: time shape = {times.shape}, y0 shape = {y0s.shape}. Both must be 2D."

        assert (
            times.shape[0] == y0s.shape[0]
        ), f"Incompatible shapes: time shape = {times.shape}, y0.shape = {y0s.shape}. First dimension must be equal in case of batches."

        return times, jax.vmap(self, in_axes=(0, 0))(times, y0s)
